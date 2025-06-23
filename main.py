"""
PairProgrammer Discord Bot - Main Entry Point

This module initializes and runs the PairProgrammer Discord bot, an AI-powered
assistant designed to help with programming tasks, document management, and
project collaboration.

The bot integrates multiple AI models (OpenAI, Anthropic), provides semantic
search through conversation history, and includes features for file management,
GitHub integration, and persistent memory storage.

Usage:
    python main.py

Environment Variables Required:
    - DISCORD_TOKEN: Discord bot token
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - PINECONE_API_KEY: Pinecone vector database API key
    - AWS_ACCESS_KEY_ID: AWS S3 access key
    - AWS_SECRET_ACCESS_KEY: AWS S3 secret key
    - S3_BUCKET_NAME: S3 bucket name for file storage
    - GITHUB_TOKEN: GitHub API token

Author: PairProgrammer Team
"""

import discord
from discord.ext import commands
from dotenv import load_dotenv

from config import DISCORD_TOKEN, DEBUG
from database.models import init_db
from utils.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Log startup
logger.logger.info(f"Starting Discord Bot (Debug Mode: {DEBUG})")

# Initialize database
logger.logger.info("Initializing database...")
init_db()
logger.logger.info("Database initialized")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    """
    Event handler called when the bot has finished logging in and is ready.
    
    Logs the bot's username, guild count, and guild names for monitoring purposes.
    This event is triggered once the bot has successfully connected to Discord
    and is ready to receive events.
    """
    logger.logger.info(f'Bot logged in as {bot.user}')
    logger.log_data('OUT', 'BOT_READY', {
        'username': str(bot.user),
        'guild_count': len(bot.guilds),
        'guilds': [g.name for g in bot.guilds]
    })


@bot.event
async def on_command(ctx):
    """
    Event handler called whenever a command is invoked.
    
    Logs detailed information about each command execution including:
    - Command name
    - User who invoked it
    - Channel/guild context
    - Full message content
    
    Args:
        ctx (commands.Context): The context object containing command information
    """
    logger.log_data('IN', 'COMMAND', {
        'command': ctx.command.name,
        'user': str(ctx.author),
        'channel': str(ctx.channel.name) if hasattr(ctx.channel, 'name') else 'DM',
        'guild': str(ctx.guild.name) if ctx.guild else 'DM',
        'full_message': ctx.message.content
    }, user_id=str(ctx.author.id), channel_id=str(ctx.channel.id))


@bot.event
async def on_command_error(ctx, error):
    """
    Event handler called whenever a command raises an exception.
    
    Logs detailed error information for debugging and monitoring purposes.
    This includes the command name, error details, and user context.
    
    Args:
        ctx (commands.Context): The context object containing command information
        error (Exception): The exception that was raised during command execution
    """
    logger.logger.error(f"Command error: {error}", exc_info=True)
    logger.log_data('OUT', 'COMMAND_ERROR', {
        'command': ctx.command.name if ctx.command else 'Unknown',
        'error': str(error),
        'user': str(ctx.author)
    })


async def load_extensions():
    """
    Load all bot extensions (cogs) that provide command functionality.
    
    This function loads each cog module and handles any errors that occur
    during the loading process. Each cog provides a specific set of commands:
    
    - ai_chat: AI model interactions and chat commands
    - memory: Persistent memory and recall functionality
    - file_manager: File upload, download, and document analysis
    - github_integration: GitHub repository tracking and management
    - admin: Administrative commands and database management
    - help_system: Dynamic help system for commands
    
    Raises:
        Exception: If any cog fails to load, the error is logged but the
                  bot continues to load other cogs.
    """
    cogs = [
        'cogs.ai_chat',
        'cogs.memory',
        'cogs.file_manager',
        'cogs.github_integration',
        'cogs.admin',
        'cogs.help_system',
        'cogs.arxiv_integration'
    ]

    for cog in cogs:
        try:
            await bot.load_extension(cog)
            logger.logger.info(f"Loaded cog: {cog}")
        except Exception as e:
            logger.logger.error(f"Failed to load cog {cog}: {e}")


async def main():
    """
    Main entry point for the Discord bot.
    
    This function handles the bot startup sequence:
    1. Load all extensions (cogs)
    2. Start the bot with the Discord token
    
    The bot runs within an async context manager to ensure proper
    cleanup when the application shuts down.
    
    Raises:
        discord.LoginFailure: If the Discord token is invalid
        aiohttp.ClientError: If there are network connectivity issues
    """
    async with bot:
        await load_extensions()
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
