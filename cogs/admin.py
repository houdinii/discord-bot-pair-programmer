import discord
from discord.ext import commands
from database.models import init_db, get_db, GitHubRepo, FileMetadata, Conversation
from utils.logger import get_logger

logger = get_logger(__name__)


class AdminCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='dbinit')
    @commands.is_owner()  # Only bot owner can run this
    async def initialize_database(self, ctx):
        """Force database initialization"""
        try:
            await init_db()
            await ctx.send("✅ Database tables initialized successfully")
        except Exception as e:
            await ctx.send(f"❌ Database initialization failed: {str(e)}")

    @commands.command(name='dbstats')
    async def database_stats(self, ctx):
        """Show database statistics"""
        try:
            db = get_db()

            repo_count = db.query(GitHubRepo).count()
            file_count = db.query(FileMetadata).count()
            conv_count = db.query(Conversation).count()

            embed = discord.Embed(
                title="Database Statistics",
                color=0x00ff00
            )
            embed.add_field(name="GitHub Repos", value=repo_count, inline=True)
            embed.add_field(name="Files", value=file_count, inline=True)
            embed.add_field(name="Conversations", value=conv_count, inline=True)

            await ctx.send(embed=embed)

            db.close()
        except Exception as e:
            await ctx.send(f"❌ Error getting stats: {str(e)}")

    @commands.command(name='dbtest')
    @commands.is_owner()
    async def test_database(self, ctx):
        """Test database connection"""
        try:
            db = get_db()
            result = db.execute("SELECT 1").scalar()
            db.close()

            if result == 1:
                await ctx.send("✅ Database connection successful!")
            else:
                await ctx.send("❌ Database returned unexpected result")
        except Exception as e:
            await ctx.send(f"❌ Database connection failed: {str(e)}")


async def setup(bot):
    await bot.add_cog(AdminCog(bot))
