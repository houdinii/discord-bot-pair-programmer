# cogs/ai_chat.py
import discord
from discord.ext import commands

from services.ai_service import AIService
from services.vector_service import VectorService


class AIChatCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ai_service = AIService()
        self.vector_service = VectorService()

    @commands.command(name='chat')
    async def chat_with_ai(self, ctx, provider: str = 'openai', model: str = 'gpt-4', *, message: str):
        """
        Chat with AI models with context from previous conversations
        Usage: !chat openai gpt-4 How do I implement a binary search?
        Usage: !chat anthropic claude-3-sonnet What's the difference between async and sync?
        """

        # Validate provider
        if provider not in ['openai', 'anthropic']:
            await ctx.send("❌ Provider must be 'openai' or 'anthropic'")
            return

        try:
            # Show typing indicator
            async with ctx.typing():
                # Get relevant context from vector database
                context = await self.vector_service.get_context_for_ai(
                    query=message,
                    channel_id=str(ctx.channel.id),
                    max_context_length=2000
                )

                # Get AI response with context - FIXED PARAMETERS
                response = await self.ai_service.get_ai_response(
                    provider=provider,
                    model=model,
                    user_message=message,
                    context=context
                )

                # Store conversation in vector database
                await self.vector_service.store_conversation(
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    message=message,
                    response=response,
                    ai_model=f"{provider}:{model}"
                )

            # Split long responses
            if len(response) > 2000:
                chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
                for chunk in chunks:
                    await ctx.send(f"```\n{chunk}\n```")
            else:
                await ctx.send(f"**{provider.title()} ({model}):**\n```\n{response}\n```")

        except Exception as e:
            print(f"[ERROR] Exception in !chat handler: {e}")
            await ctx.send(f"❌ Error: {str(e)}")

    @commands.command(name='models')
    async def list_models(self, ctx):
        """List available AI models"""
        embed = discord.Embed(
            title="Available AI Models",
            description="Choose your AI provider and model",
            color=0x00ff00
        )

        embed.add_field(
            name="OpenAI",
            value="• gpt-4\n• gpt-3.5",
            inline=True
        )

        embed.add_field(
            name="Anthropic",
            value="• claude-3-opus\n• claude-3-sonnet\n• claude-3-haiku",
            inline=True
        )

        embed.add_field(
            name="Usage",
            value="!chat [provider] [model] [your message]",
            inline=False
        )

        await ctx.send(embed=embed)

    @commands.command(name='quick')
    async def quick_chat(self, ctx, *, message: str):
        """Quick chat with default model (GPT-4)"""
        await self.chat_with_ai(ctx, 'openai', 'gpt-4', message=message)

    @commands.command(name='search')
    async def search_context(self, ctx, *, query: str):
        """Search through conversation and memory history"""
        async with ctx.typing():
            results = await self.vector_service.search_similar(
                query=query,
                channel_id=str(ctx.channel.id),
                top_k=5
            )

            if not results:
                await ctx.send("❌ No relevant context found")
                return

            embed = discord.Embed(
                title=f"Search Results: {query}",
                color=0x0099ff
            )

            for result in results:
                metadata = result['metadata']
                score = result['score']

                if metadata['type'] == 'conversation':
                    title = f"Conversation (Score: {score:.2f})"
                    value = f"**Q:** {metadata.get('message', 'N/A')[:100]}...\n**A:** {metadata.get('response', 'N/A')[:100]}..."
                elif metadata['type'] == 'memory':
                    title = f"Memory: {metadata.get('tag', 'N/A')} (Score: {score:.2f})"
                    value = metadata.get('content', 'N/A')[:200] + "..."
                elif metadata['type'] == 'document':
                    title = f"Document: {metadata.get('filename', 'N/A')} (Score: {score:.2f})"
                    value = metadata.get('content', 'N/A')[:200] + "..."
                else:
                    title = f"{metadata['type'].title()} (Score: {score:.2f})"
                    value = str(metadata.get('content', ''))[:200] + "..."

                embed.add_field(name=title, value=value, inline=False)

            await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(AIChatCog(bot))
