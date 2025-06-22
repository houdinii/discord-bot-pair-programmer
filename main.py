import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from utils.logger import get_logger
from database.models import init_db  # Add this import

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Log startup
debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
logger.logger.info(f"Starting Discord Bot (Debug Mode: {debug_mode})")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    logger.logger.info(f'Bot logged in as {bot.user}')

    # Initialize database tables
    try:
        await init_db()
        logger.logger.info("Database initialized successfully")
    except Exception as e:
        logger.logger.error(f"Failed to initialize database: {e}")
        # Continue running even if DB init fails

    logger.log_data('OUT', 'BOT_READY', {
        'username': str(bot.user),
        'guild_count': len(bot.guilds),
        'guilds': [g.name for g in bot.guilds]
    })


@bot.event
async def on_command(ctx):
    logger.log_data('IN', 'COMMAND', {
        'command': ctx.command.name,
        'user': str(ctx.author),
        'channel': str(ctx.channel.name) if hasattr(ctx.channel, 'name') else 'DM',
        'guild': str(ctx.guild.name) if ctx.guild else 'DM',
        'full_message': ctx.message.content
    }, user_id=str(ctx.author.id), channel_id=str(ctx.channel.id))


@bot.event
async def on_command_error(ctx, error):
    logger.logger.error(f"Command error: {error}", exc_info=True)
    logger.log_data('OUT', 'COMMAND_ERROR', {
        'command': ctx.command.name if ctx.command else 'Unknown',
        'error': str(error),
        'user': str(ctx.author)
    })

    # Send user-friendly error message
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore command not found

    await ctx.send(f"❌ An error occurred: {str(error)[:200]}")


# Load cogs
async def load_extensions():
    cogs = [
        'cogs.ai_chat',
        'cogs.memory',
        'cogs.file_manager',
        'cogs.github_integration'
    ]

    for cog in cogs:
        try:
            await bot.load_extension(cog)
            logger.logger.info(f"Loaded cog: {cog}")
        except Exception as e:
            logger.logger.error(f"Failed to load cog {cog}: {e}")


# Run bot
async def main():
    async with bot:
        await load_extensions()
        await bot.start(os.getenv('DISCORD_TOKEN'))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
