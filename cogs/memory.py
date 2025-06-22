# cogs/memory.py
import discord
from discord.ext import commands

from services.vector_service import VectorService


class MemoryCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.vector_service = VectorService()

    @commands.command(name='remember')
    async def remember_context(self, ctx, tag: str, *, description: str):
        """
        Remember important context with a tag
        Usage: !remember project_auth We're working on JWT authentication for the user service
        """

        # Store in vector database
        vector_id = await self.vector_service.store_memory(
            user_id=str(ctx.author.id),
            channel_id=str(ctx.channel.id),
            tag=tag,
            content=description
        )

        embed = discord.Embed(
            title="Memory Saved",
            description=f"Tagged as: **{tag}**\nContent: {description}",
            color=0x00ff00
        )
        embed.add_field(name="Vector ID", value=vector_id[:8] + "...", inline=True)
        await ctx.send(embed=embed)

    @commands.command(name='recall')
    async def recall_memory(self, ctx, *, query: str):
        """
        Recall memories by searching for similar content
        Usage: !recall authentication
        Usage: !recall React PostgreSQL
        Note: This searches the CONTENT of memories, not the tags. Use !list_memories to see all tags.
        """

        # Search for memories by content
        results = await self.vector_service.search_similar(
            query=query,
            channel_id=str(ctx.channel.id),
            content_type=['memory'],
            top_k=5
        )

        if not results:
            await ctx.send(
                f"❌ No memories found with content related to: {query}\n💡 Tip: !recall searches memory content, not tags. Use !list_memories to see all memory tags.")
            return

        embed = discord.Embed(
            title=f"Recalled Memories: {query}",
            color=0x0099ff
        )

        for result in results:
            metadata = result['metadata']
            score = result['score']

            embed.add_field(
                name=f"Tag: {metadata.get('tag', 'N/A')} (Relevance: {score:.2f})",
                value=f"{metadata.get('content', 'N/A')}\n*Saved: {metadata.get('timestamp', 'N/A')[:10]}*",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='get_memory')
    async def get_memory_by_tag(self, ctx, tag: str):
        """
        Get a specific memory by its tag
        Usage: !get_memory project_stack
        """

        # Get memory by exact tag match
        memory = await self.vector_service.get_memory_by_tag(
            channel_id=str(ctx.channel.id),
            tag=tag
        )

        if not memory:
            await ctx.send(f"❌ No memory found with tag: {tag}")
            return

        embed = discord.Embed(
            title=f"Memory: {tag}",
            description=memory['content'],
            color=0x0099ff
        )
        embed.add_field(name="Saved", value=memory['timestamp'][:10], inline=True)

        await ctx.send(embed=embed)

    @commands.command(name='list_memories')
    async def list_memories(self, ctx):
        """
        List all memory tags in this channel
        Usage: !list_memories
        """

        memories = await self.vector_service.list_memory_tags(
            channel_id=str(ctx.channel.id)
        )

        if not memories:
            await ctx.send("❌ No memories saved in this channel")
            return

        embed = discord.Embed(
            title=f"Saved Memories ({len(memories)} total)",
            color=0x0099ff
        )

        for memory in memories[:25]:  # Discord embed field limit
            embed.add_field(
                name=f"Tag: {memory['tag']}",
                value=f"{memory['content'][:100]}..." if len(memory['content']) > 100 else memory['content'],
                inline=False
            )

        if len(memories) > 25:
            embed.add_field(
                name="Note",
                value=f"Showing first 25 of {len(memories)} memories",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='forget')
    async def forget_memory(self, ctx, tag: str):
        """
        Delete memories by tag
        Usage: !forget project_auth
        """

        # Delete from vector database
        success = await self.vector_service.delete_by_tag(
            channel_id=str(ctx.channel.id),
            tag=tag
        )

        if success:
            await ctx.send(f"✅ Forgot all memories with tag: {tag}")
        else:
            await ctx.send(f"❌ No memories found with tag: {tag}")

    @commands.command(name='stats')
    async def vector_stats(self, ctx):
        """Show vector database statistics"""
        stats = await self.vector_service.get_stats()

        embed = discord.Embed(
            title="Vector Database Stats",
            color=0x0099ff
        )

        if 'error' in stats:
            embed.add_field(name="Error", value=stats['error'], inline=False)
        else:
            embed.add_field(name="Total Vectors", value=stats['total_vectors'], inline=True)
            embed.add_field(name="Dimension", value=stats['dimension'], inline=True)
            embed.add_field(name="Index Fullness", value=f"{stats['index_fullness']:.2%}", inline=True)

        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(MemoryCog(bot))