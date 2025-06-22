# cogs/admin.py
from discord.ext import commands

from database.models import init_db, engine, Base


class AdminCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='initdb')
    @commands.has_permissions(administrator=True)
    async def initialize_database(self, ctx):
        """Initialize the database tables (admin only)"""
        try:
            await init_db()
            await ctx.send("✅ Database tables created successfully")
        except Exception as e:
            await ctx.send(f"❌ Error initializing database: {str(e)}")

    @commands.command(name='dropdb')
    @commands.has_permissions(administrator=True)
    async def drop_database(self, ctx):
        """Drop all database tables (admin only, DANGEROUS)"""
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
    await bot.add_cog(AdminCog(bot))
