# cogs/ai_chat.py
import discord
from discord.ext import commands

from services.ai_service import AIService
from services.vector_service import VectorService
from utils.logger import get_logger, log_method

logger = get_logger(__name__)


class AIChatCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ai_service = AIService()
        self.vector_service = VectorService()
        logger.logger.info("AIChatCog initialized")

    async def send_long_message(self, ctx, content: str, provider: str = None, model: str = None):
        """Helper to send long messages split into chunks"""
        logger.log_data('IN', 'SEND_MESSAGE', {
            'content_length': len(content),
            'provider': provider,
            'model': model,
            'channel': str(ctx.channel.name),
            'user': str(ctx.author)
        })

        # Add header if provider/model specified
        if provider and model:
            header = f"**{provider.title()} ({model}):**\n"
            content = header + content

        # Discord's limit is 2000 chars
        if len(content) <= 1990:
            await ctx.send(content)
            logger.log_data('OUT', 'MESSAGE_SENT', {'chunks': 1, 'total_length': len(content)})
        else:
            # Split into chunks
            chunks = []
            lines = content.split('\n')
            current_chunk = ""

            for line in lines:
                if len(current_chunk) + len(line) + 1 > 1990:
                    chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk += ("\n" if current_chunk else "") + line

            if current_chunk:
                chunks.append(current_chunk)

            logger.log_data('OUT', 'MESSAGE_CHUNKS', {'chunks': len(chunks), 'total_length': len(content)})

            # Send chunks
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await ctx.send(chunk)
                else:
                    # Add continuation marker for subsequent chunks
                    await ctx.send(f"```\n{chunk}\n```" if not chunk.startswith("```") else chunk)

    @commands.command(name='chat')
    @log_method()
    async def chat_with_ai(self, ctx, provider: str = 'openai', model: str = 'chatgpt-4o-latest', *, message: str):
        """
        Chat with AI models with context from previous conversations
        Usage: !chat openai chatgpt-4o-latest How do I implement a binary search?
        Usage: !chat anthropic claude-sonnet-4-0 What's the difference between async and sync?
        """

        logger.log_data('IN', 'CHAT_COMMAND', {
            'user': str(ctx.author),
            'channel': str(ctx.channel.name),
            'provider': provider,
            'model': model,
            'message': message,
            'message_length': len(message)
        }, user_id=str(ctx.author.id), channel_id=str(ctx.channel.id))

        # Validate provider
        if provider not in ['openai', 'anthropic']:
            error_msg = f"Provider must be 'openai' or 'anthropic', got: {provider}"
            logger.logger.warning(error_msg)
            await ctx.send(f"❌ {error_msg}")
            return

        # Validate model
        if model not in self.ai_service.available_models[provider]:
            error_msg = f"Model '{model}' not available for {provider}"
            logger.logger.warning(error_msg)
            await ctx.send(f"❌ {error_msg}. Use !models to see available models.")
            return

        try:
            # Show typing indicator
            async with ctx.typing():
                logger.logger.debug(f"Fetching context for query: {message[:50]}...")

                # Get relevant context from vector database
                context = await self.vector_service.get_context_for_ai(
                    query=message,
                    channel_id=str(ctx.channel.id),
                    max_context_length=2000
                )

                logger.log_data('IN', 'VECTOR_CONTEXT', {
                    'context_length': len(context),
                    'context_preview': context[:200] + '...' if len(context) > 200 else context
                })

                logger.logger.debug(f"Getting AI response from {provider}/{model}")

                # Get AI response with context
                response = await self.ai_service.get_ai_response(
                    provider=provider,
                    model=model,
                    user_message=message,
                    context=context
                )

                logger.log_data('OUT', 'AI_RESPONSE', {
                    'response_length': len(response),
                    'response_preview': response[:200] + '...' if len(response) > 200 else response,
                    'provider': provider,
                    'model': model
                })

                logger.logger.debug("Storing conversation in vector database")

                # Store conversation in vector database
                vector_id = await self.vector_service.store_conversation(
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    message=message,
                    response=response,
                    ai_model=f"{provider}:{model}"
                )

                logger.log_data('OUT', 'VECTOR_STORED', {
                    'vector_id': vector_id,
                    'type': 'conversation'
                })

            # Send response with proper chunking
            await self.send_long_message(ctx, response, provider, model)

            logger.logger.info(f"Successfully processed chat command for {ctx.author}")

        except Exception as e:
            logger.logger.error(f"Exception in !chat handler: {e}", exc_info=True)
            await ctx.send(f"❌ Error: {str(e)}")

    @commands.command(name='models')
    @log_method()
    async def list_models(self, ctx):
        """List available AI models"""
        logger.log_data('IN', 'LIST_MODELS', {
            'user': str(ctx.author),
            'channel': str(ctx.channel.name)
        })

        embed = discord.Embed(
            title="Available AI Models",
            description="Choose your AI provider and model",
            color=0x00ff00
        )

        # OpenAI models
        openai_models = "\n".join([f"• {m}" for m in self.ai_service.available_models["openai"][:10]])
        embed.add_field(
            name="OpenAI",
            value=openai_models,
            inline=True
        )

        # Anthropic models
        anthropic_models = "\n".join([f"• {m}" for m in self.ai_service.available_models["anthropic"]])
        embed.add_field(
            name="Anthropic",
            value=anthropic_models,
            inline=True
        )

        embed.add_field(
            name="Usage",
            value="!chat [provider] [model] [your message]\n!quick [message] (uses chatgpt-4o-latest)",
            inline=False
        )

        await ctx.send(embed=embed)
        logger.logger.info(f"Sent models list to {ctx.author}")

    @commands.command(name='quick')
    @log_method()
    async def quick_chat(self, ctx, *, message: str):
        """Quick chat with default model (chatgpt-4o-latest) - includes context!"""
        logger.log_data('IN', 'QUICK_CHAT', {
            'user': str(ctx.author),
            'channel': str(ctx.channel.name),
            'message': message,
            'message_length': len(message)
        })

        # Call the main chat method with default parameters
        # This ensures context is included just like regular chat
        await self.chat_with_ai(ctx, 'openai', 'chatgpt-4o-latest', message=message)

    @commands.command(name='search')
    @log_method()
    async def search_context(self, ctx, *, query: str):
        """Search through conversation and memory history"""
        logger.log_data('IN', 'SEARCH_CONTEXT', {
            'user': str(ctx.author),
            'channel': str(ctx.channel.name),
            'query': query
        })

        async with ctx.typing():
            results = await self.vector_service.search_similar(
                query=query,
                channel_id=str(ctx.channel.id),
                top_k=5
            )

            logger.log_data('OUT', 'SEARCH_RESULTS', {
                'query': query,
                'results_count': len(results),
                'result_scores': [r['score'] for r in results]
            })

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
            logger.logger.info(f"Sent {len(results)} search results to {ctx.author}")


async def setup(bot):
    await bot.add_cog(AIChatCog(bot))
