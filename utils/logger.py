"""
Advanced Logging Utility for PairProgrammer Discord Bot

This module provides a comprehensive logging system with colored console output,
structured JSON logging, method call tracking, and debug/production modes.
It's designed specifically for Discord bot applications with features like
data flow tracking, method decoration, and context-aware logging.

Key Features:
    - Colored console output with customizable formatters
    - Structured JSON logging for file output
    - Method call tracking and automatic logging
    - Debug vs production logging modes
    - Data flow visualization (IN/OUT tracking)
    - Discord context integration (user, channel, guild)
    - Exception handling and stack trace logging

Logging Modes:
    - Debug Mode: Detailed logs with colors, data tracking, method calls
    - Production Mode: Clean, essential logs for production environments

Environment Variables:
    DEBUG: Enable debug mode ('true'/'false')
    LOG_FILE: Optional file path for JSON log output

Usage:
    # Get a logger instance
    logger = get_logger(__name__)
    logger.logger.info("Application started")
    
    # Log data flow
    logger.log_data('IN', 'USER_MESSAGE', {'content': 'hello'}, 
                   user_id='123', channel_id='456')
    
    # Method decoration for automatic logging
    @log_method()
    async def my_function():
        pass

Author: PairProgrammer Team
"""

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, UTC
from functools import wraps
from typing import Any


class Colors:
    """
    ANSI color codes for terminal output formatting.
    
    Provides color constants for console output in debug mode.
    Colors are used to differentiate log levels, data types,
    and flow directions for better readability.
    """
    BLUE = '\033[94m'      # Debug level logs
    GREEN = '\033[92m'     # Info level logs, outgoing data
    YELLOW = '\033[93m'    # Warning level logs, field names
    RED = '\033[91m'       # Error level logs
    MAGENTA = '\033[95m'   # Data separators
    CYAN = '\033[96m'      # Incoming data, logger names
    WHITE = '\033[97m'     # General text, timestamps
    ENDC = '\033[0m'       # Reset color
    BOLD = '\033[1m'       # Bold text


class DiscordBotLogger:
    """
    Advanced logger class for Discord bot applications.
    
    Provides structured logging with support for debug/production modes,
    colored console output, file logging, and data flow tracking.
    Designed specifically for Discord bot context with user and channel
    information integration.
    
    Attributes:
        logger (logging.Logger): Standard Python logger instance
        debug_mode (bool): Whether debug mode is enabled
        
    Features:
        - Automatic log level setting based on DEBUG environment variable
        - Colored console output in debug mode
        - Optional JSON file logging
        - Data flow tracking with direction indicators
        - Discord context integration (user_id, channel_id)
        
    Example:
        logger = DiscordBotLogger('my_module')
        logger.logger.info("Starting process")
        logger.log_data('IN', 'USER_MESSAGE', {'content': 'hello'})
    """
    
    def __init__(self, name: str):
        """
        Initialize the DiscordBotLogger with the specified name.
        
        Args:
            name (str): Logger name, typically the module name (__name__)
            
        Environment Variables:
            DEBUG: Enable debug mode ('true'/'false', default: 'false')
            LOG_FILE: Optional file path for JSON log output
            
        Logger Configuration:
            - Debug mode: DEBUG level with colored DebugFormatter
            - Production mode: INFO level with ProductionFormatter
            - File output: JSON format with structured data
        """
        self.logger = logging.getLogger(name)
        self.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'

        # Set level based on debug mode
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # Create console handler with custom formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create appropriate formatter based on mode
        if self.debug_mode:
            formatter = DebugFormatter()
        else:
            formatter = ProductionFormatter()

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if LOG_FILE environment variable is specified
        log_file = os.getenv('LOG_FILE')
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)

    def log_data(self, direction: str, data_type: str, data: Any, user_id: str = None, channel_id: str = None):
        """
        Log structured data with directional flow indicators.
        
        This method is used to track data flow through the bot, particularly
        useful for debugging Discord interactions, API calls, and data processing.
        Only logs in debug mode to avoid performance impact in production.
        
        Args:
            direction (str): Data flow direction ('IN' for incoming, 'OUT' for outgoing)
            data_type (str): Type of data being logged (e.g., 'USER_MESSAGE', 'API_RESPONSE')
            data (Any): The actual data to log (will be JSON serialized)
            user_id (str, optional): Discord user ID for context
            channel_id (str, optional): Discord channel ID for context
            
        Visual Indicators:
            - IN: ◀── INCOMING with cyan color
            - OUT: ──▶ OUTGOING with green color
            
        Data Structure:
            {
                'timestamp': ISO timestamp,
                'direction': 'IN' or 'OUT',
                'type': data_type,
                'user_id': user_id,
                'channel_id': channel_id,
                'data': actual_data
            }
            
        Example:
            # Log incoming user message
            logger.log_data('IN', 'USER_MESSAGE', {
                'content': 'Hello bot!',
                'length': 10
            }, user_id='123456789', channel_id='987654321')
            
            # Log outgoing API response
            logger.log_data('OUT', 'AI_RESPONSE', {
                'provider': 'openai',
                'model': 'gpt-4',
                'response_length': 150
            })
            
        Note:
            This method only logs in debug mode. In production mode,
            it returns immediately without logging to maintain performance.
        """
        if not self.debug_mode:
            return

        log_entry = {
            'timestamp': datetime.now(UTC).isoformat(),
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
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

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
        timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] {record.levelname:8} {record.name}: {record.getMessage()}"


class JsonFormatter(logging.Formatter):
    """JSON formatter for file output"""

    def format(self, record):
        log_obj = {
            'timestamp': datetime.now(UTC).isoformat(),
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
            # noinspection PyArgumentList
            log_obj['exception'] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str) -> DiscordBotLogger:
    """
    Factory function to create a DiscordBotLogger instance.
    
    This is the primary way to obtain a logger instance throughout
    the application. It ensures consistent logger configuration
    and proper initialization.
    
    Args:
        name (str): Logger name, typically the module name (__name__)
        
    Returns:
        DiscordBotLogger: Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.logger.info("Module initialized")
    """
    return DiscordBotLogger(name)


def log_method(logger_name: str = None):
    """
    Decorator to automatically log method calls and results.
    
    This decorator provides automatic logging for method entry, exit,
    parameters, results, and exceptions. It handles both synchronous
    and asynchronous methods automatically and includes Discord context
    when available.
    
    Args:
        logger_name (str, optional): Custom logger name. If not provided,
                                   uses the decorated function's module name.
                                   
    Features:
        - Automatic async/sync detection
        - Discord context extraction (user, channel, guild)
        - Parameter logging (truncated for security)
        - Result type logging
        - Exception tracking with stack traces
        - Debug mode only (no performance impact in production)
        
    Logged Information:
        - Method name and module
        - Discord context (if available)
        - Keyword arguments (truncated to 100 chars)
        - Success/failure status
        - Result type
        - Exception details and stack traces
        
    Example:
        @log_method()
        async def process_user_command(self, ctx, user_input: str):
            # Method implementation
            return result
            
        # With custom logger name
        @log_method('custom_logger')
        def sync_method(self, data):
            return processed_data
            
    Discord Context:
        When the second argument is a Discord context object (ctx),
        the decorator automatically extracts and logs:
        - User information (ctx.author)
        - Channel name (ctx.channel.name)
        - Guild name (ctx.guild.name) or 'DM' for direct messages
        
    Performance:
        The decorator checks debug mode before performing any logging
        operations, ensuring minimal performance impact in production.
    """

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
