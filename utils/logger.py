# utils/logger.py
import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from functools import wraps
from typing import Any


# Color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class DiscordBotLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'

        # Set level based on debug mode
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger.setLevel(log_level)

        # Remove existing handlers
        self.logger.handlers = []

        # Create console handler with custom formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create custom formatter
        if self.debug_mode:
            formatter = DebugFormatter()
        else:
            formatter = ProductionFormatter()

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        log_file = os.getenv('LOG_FILE')
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)

    def log_data(self, direction: str, data_type: str, data: Any, user_id: str = None, channel_id: str = None):
        """Log data with direction indicator"""
        if not self.debug_mode:
            return

        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'direction': direction,  # 'IN' or 'OUT'
            'type': data_type,
            'user_id': user_id,
            'channel_id': channel_id,
            'data': data
        }

        if direction == 'IN':
            self.logger.debug(f"{Colors.CYAN}◀── INCOMING {data_type}{Colors.ENDC}", extra={'data': log_entry})
        else:
            self.logger.debug(f"{Colors.GREEN}──▶ OUTGOING {data_type}{Colors.ENDC}", extra={'data': log_entry})


class DebugFormatter(logging.Formatter):
    """Detailed formatter for debug mode"""

    def format(self, record):
        # Base format
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Color based on level
        if record.levelno == logging.DEBUG:
            level_color = Colors.BLUE
        elif record.levelno == logging.INFO:
            level_color = Colors.GREEN
        elif record.levelno == logging.WARNING:
            level_color = Colors.YELLOW
        elif record.levelno == logging.ERROR:
            level_color = Colors.RED
        else:
            level_color = Colors.WHITE

        # Format the message
        formatted = f"{Colors.WHITE}[{timestamp}]{Colors.ENDC} "
        formatted += f"{level_color}{record.levelname:8}{Colors.ENDC} "
        formatted += f"{Colors.CYAN}{record.name:20}{Colors.ENDC} "
        formatted += f"{Colors.WHITE}{record.getMessage()}{Colors.ENDC}"

        # Add data if present
        if hasattr(record, 'data') and record.data:
            data = record.data.get('data', {})
            if isinstance(data, dict):
                # Pretty print dictionaries
                formatted += f"\n{Colors.MAGENTA}{'─' * 50}{Colors.ENDC}"
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=2)
                        # Indent each line
                        value_str = '\n'.join(['  ' + line for line in value_str.split('\n')])
                        formatted += f"\n{Colors.YELLOW}{key}:{Colors.ENDC}\n{value_str}"
                    else:
                        formatted += f"\n{Colors.YELLOW}{key}:{Colors.ENDC} {value}"
                formatted += f"\n{Colors.MAGENTA}{'─' * 50}{Colors.ENDC}"
            else:
                formatted += f"\n  {Colors.YELLOW}Data:{Colors.ENDC} {data}"

        return formatted


class ProductionFormatter(logging.Formatter):
    """Simple formatter for production"""

    def format(self, record):
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] {record.levelname:8} {record.name}: {record.getMessage()}"


class JsonFormatter(logging.Formatter):
    """JSON formatter for file output"""

    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'data'):
            log_obj['data'] = record.data

        if record.exc_info:
            log_obj['exception'] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str) -> DiscordBotLogger:
    """Get a logger instance"""
    return DiscordBotLogger(name)


def log_method(logger_name: str = None):
    """Decorator to log method calls"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)

            # Get useful info from args
            self_obj = args[0] if args else None
            ctx = args[1] if len(args) > 1 and hasattr(args[1], 'author') else None

            if logger.debug_mode:
                # Log method entry
                log_data = {
                    'method': func.__name__,
                    'module': func.__module__,
                }

                if ctx:
                    log_data['user'] = str(ctx.author)
                    log_data['channel'] = str(ctx.channel.name)
                    log_data['guild'] = str(ctx.guild.name) if ctx.guild else 'DM'

                # Add kwargs if present
                if kwargs:
                    log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}

                logger.log_data('IN', 'METHOD_CALL', log_data)

            try:
                # Execute the method
                result = await func(*args, **kwargs)

                if logger.debug_mode:
                    # Log successful completion
                    logger.log_data('OUT', 'METHOD_RESULT', {
                        'method': func.__name__,
                        'status': 'success',
                        'result_type': type(result).__name__ if result else 'None'
                    })

                return result

            except Exception as e:
                # Log error
                logger.logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)

                if logger.debug_mode:
                    logger.log_data('OUT', 'METHOD_ERROR', {
                        'method': func.__name__,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)

            if logger.debug_mode:
                log_data = {
                    'method': func.__name__,
                    'module': func.__module__,
                }

                if kwargs:
                    log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}

                logger.log_data('IN', 'METHOD_CALL', log_data)

            try:
                result = func(*args, **kwargs)

                if logger.debug_mode:
                    logger.log_data('OUT', 'METHOD_RESULT', {
                        'method': func.__name__,
                        'status': 'success'
                    })

                return result

            except Exception as e:
                logger.logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
