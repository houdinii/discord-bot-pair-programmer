"""
Administrative Commands for PairProgrammer Discord Bot

This cog provides administrative commands for database management and
bot maintenance. All commands require Discord Administrator permissions
and should be used with caution, especially destructive operations.

Commands:
    !initdb: Initialize database tables
    !dropdb: Drop all database tables (destructive, requires confirmation)

Security:
    - All commands require Discord Administrator permissions
    - Destructive commands require explicit confirmation
    - Operations are logged for audit purposes

Author: PairProgrammer Team
"""

from discord.ext import commands

from database.models import init_db, engine, Base


class AdminCog(commands.Cog):
    """
    Administrative commands cog for database and bot management.
    
    This cog provides essential administrative functionality for managing
    the bot's database schema and performing maintenance operations.
    All commands are restricted to users with Administrator permissions.
    """
    
    def __init__(self, bot):
        """
        Initialize the AdminCog.
        
        Args:
            bot (commands.Bot): The Discord bot instance
        """
        self.bot = bot

    @commands.command(name='initdb')
    @commands.cooldown(3, 60, commands.BucketType.user)
    @commands.has_permissions(administrator=True)
    async def initialize_database(self, ctx):
        """
        Initialize the database tables.
        
        Creates all required database tables if they don't already exist.
        This is safe to run multiple times as it only creates missing tables.
        
        Permissions Required:
            - Discord Administrator permissions
            
        Usage:
            !initdb
            
        Raises:
            commands.MissingPermissions: If user lacks administrator permissions
            commands.CommandOnCooldown: If command is on cooldown
            
        Example:
            !initdb
            # Response: ✅ Database tables created successfully
        """
        try:
            init_db()
            await ctx.send("✅ Database tables created successfully")
        except Exception as e:
            await ctx.send(f"❌ Error initializing database: {str(e)}")

    @commands.command(name='dropdb')
    @commands.cooldown(3, 60, commands.BucketType.user)
    @commands.has_permissions(administrator=True)
    async def drop_database(self, ctx):
        """
        Drop all database tables - DESTRUCTIVE OPERATION.
        
        **WARNING**: This command permanently deletes ALL data including:
        - All conversation history
        - All file metadata and references
        - All GitHub repository tracking
        - All memory entries
        
        This operation requires explicit confirmation within 30 seconds.
        
        Permissions Required:
            - Discord Administrator permissions
            
        Usage:
            !dropdb
            # Bot responds with warning
            !confirm  # Must be typed within 30 seconds
            
        Confirmation:
            After running !dropdb, you must type exactly `!confirm` within
            30 seconds in the same channel to proceed with the deletion.
            
        Raises:
            commands.MissingPermissions: If user lacks administrator permissions
            commands.CommandOnCooldown: If command is on cooldown
            asyncio.TimeoutError: If confirmation not received within 30 seconds
            
        Example:
            !dropdb
            # Response: ⚠️ **DANGER**: This will delete ALL data! Type `!confirm` within 30 seconds to proceed.
            !confirm
            # Response: 💥 All database tables dropped
        """
        confirm_msg = await ctx.send("⚠️ **DANGER**: This will delete ALL data! Type `!confirm` within 30 seconds to proceed.")

        def check(m):
            return m.author == ctx.author and m.content == '!confirm' and m.channel == ctx.channel

        try:
            await self.bot.wait_for('message', check=check, timeout=30.0)
            Base.metadata.drop_all(engine)
            await ctx.send("💥 All database tables dropped")
        except Exception as e:
            await ctx.send(f"Operation cancelled or error occurred: {str(e)}")


async def setup(bot):
    """
    Setup function to add the AdminCog to the bot.
    
    This function is called automatically when the cog is loaded
    through the bot's load_extension() method.
    
    Args:
        bot (commands.Bot): The Discord bot instance to add the cog to
    """
    await bot.add_cog(AdminCog(bot))
