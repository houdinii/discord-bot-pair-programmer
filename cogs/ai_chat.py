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
        Debug version of !chat that prints/logs each step and surfaces errors.
        """
        try:
            # 1) Acknowledge we got the command
            print(f"[DEBUG] !chat invoked: provider={provider}, model={model}, message={message}")
            await ctx.send("🔍 Debug: Received your request. Fetching context…")

            # 2) Show typing
            async with ctx.typing():
                # 3) Get context
                context = await self.vector_service.get_context_for_ai(
                    query=message,
                    channel_id=str(ctx.channel.id),
                    max_context_length=2000
                )
                print(f"[DEBUG] Context fetched: {context!r}")
                await ctx.send(f"🔍 Debug: Context length = {len(context)}")

                # 4) Get AI response
                response = await self.ai_service.get_ai_response(
                    messages=[{"role": "user", "content": message}],
                    provider=provider,
                    model=model,
                    context=context
                )
                print(f"[DEBUG] AI responded: {response!r}")

            # 5) Send it back
            await ctx.send(f"**{provider.title()} ({model}):**\n```\n{response}\n```")

        except Exception as e:
            # 6) If anything blows up, log it and notify the channel
            print(f"[ERROR] Exception in !chat handler:", exc_info=True)
            await ctx.send(f"❌ Debug: I hit an error: `{e}`")

    # @commands.command(name='chat')
    # async def chat_with_ai(self, ctx, provider: str = 'openai', model: str = 'gpt-4', *, message: str):
    #     """
    #     Chat with AI models with context from previous conversations
    #     Usage: !chat OpenAI GPT-4 How do I implement a binary search?
    #     Usage: !chat anthropic claude-3-sonnet What's the difference between async and sync?
    #     """
    #
    #     # Validate provider and model
    #     if provider not in ['openai', 'anthropic']:
    #         await ctx.send("❌ Provider must be 'openai' or 'anthropic'")
    #         return
    #
    #     # Show typing indicator
    #     async with ctx.typing():
    #         # Get relevant context from vector database
    #         context = await self.vector_service.get_context_for_ai(
    #             query=message,
    #             channel_id=str(ctx.channel.id),
    #             max_context_length=2000
    #         )
    #
    #         # Build messages for AI
    #         messages = [{"role": "user", "content": message}]
    #
    #         # Get AI response with context
    #         response = await self.ai_service.get_ai_response(
    #             messages=messages,
    #             provider=provider,
    #             model=model,
    #             context=context
    #         )
    #
    #         # Store conversation in vector database
    #         await self.vector_service.store_conversation(
    #             user_id=str(ctx.author.id),
    #             channel_id=str(ctx.channel.id),
    #             message=message,
    #             response=response,
    #             ai_model=f"{provider}:{model}"
    #         )
    #
    #         # Split long responses
    #         if len(response) > 2000:
    #             chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
    #             for chunk in chunks:
    #                 await ctx.send(f"```\n{chunk}\n```")
    #         else:
    #             await ctx.send(f"**{provider.title()} ({model}):**\n```\n{response}\n```")

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
                    value = f"**Q:** {metadata['message'][:100]}...\n**A:** {metadata['response'][:100]}..."
                elif metadata['type'] == 'memory':
                    title = f"Memory: {metadata['tag']} (Score: {score:.2f})"
                    value = metadata['content'][:200] + "..."
                elif metadata['type'] == 'document':
                    title = f"Document: {metadata['filename']} (Score: {score:.2f})"
                    value = metadata['content'][:200] + "..."
                else:
                    title = f"{metadata['type'].title()} (Score: {score:.2f})"
                    value = str(metadata.get('content', ''))[:200] + "..."

                embed.add_field(name=title, value=value, inline=False)

            await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(AIChatCog(bot))
