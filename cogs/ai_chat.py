# cogs/ai_chat.py
from datetime import datetime

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
                    # Add continuation marker for further chunks
                    await ctx.send(f"```\n{chunk}\n```" if not chunk.startswith("```") else chunk)

    @commands.command(name='askdoc')
    async def ask_about_document(self, ctx, file_id: int, *, question: str):
        """
        Ask a question about a specific document
        Usage: !askdoc 123 What is the main topic of this paper?
        """
        from database.models import FileMetadata, get_db

        # Verify file exists
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found")
            return

        async with ctx.typing():
            # Search specifically for this document's content
            doc_results = await self.vector_service.search_documents(
                query=question,
                channel_id=str(ctx.channel.id),
                file_id=str(file_id),
                top_k=10
            )

            logger.log_data('IN', 'ASKDOC_SEARCH', {
                'file_id': file_id,
                'question': question,
                'results_found': len(doc_results)
            })

            if not doc_results:
                # Try a broader search without the question filter
                doc_results = await self.vector_service.search_documents(
                    query=file_metadata.filename,  # Search by filename
                    channel_id=str(ctx.channel.id),
                    file_id=str(file_id),
                    top_k=10
                )

                if not doc_results:
                    await ctx.send("❌ No content found for this document. It may not be indexed properly.")
                    return

            # Build context from document chunks
            doc_context = f"Document: {file_metadata.filename}\n"
            doc_context += f"Description: {file_metadata.description}\n"
            doc_context += f"File Type: {file_metadata.file_type or 'Unknown'}\n\n"
            doc_context += "Document Content:\n\n"

            # Sort chunks by chunk_index to maintain order
            sorted_chunks = sorted(doc_results,
                                   key=lambda x: x['metadata'].get('chunk_index', 0))

            for chunk in sorted_chunks[:5]:  # Use top 5 chunks
                chunk_content = chunk['content']
                # Remove the enhanced search prefix if present
                if "Content:\n" in chunk_content:
                    chunk_content = chunk_content.split("Content:\n", 1)[1]

                chunk_idx = chunk['metadata'].get('chunk_index', 0)
                total_chunks = chunk['metadata'].get('total_chunks', 1)
                doc_context += f"[Part {chunk_idx + 1}/{total_chunks}]\n{chunk_content}\n\n---\n\n"

            # Create a system message that emphasizes using the document content
            system_msg = f"""You are analyzing the document '{file_metadata.filename}'. 
            Use ONLY the provided document content to answer the question. 
            If the answer is not in the provided content, say so clearly."""

            # Get AI response with document context
            response = await self.ai_service.get_ai_response(
                provider='openai',
                model='chatgpt-4o-latest',
                user_message=f"{system_msg}\n\nQuestion: {question}",
                context=doc_context
            )

            # Store this Q&A in vector database for future reference
            await self.vector_service.store_conversation(
                user_id=str(ctx.author.id),
                channel_id=str(ctx.channel.id),
                message=f"[Document Q&A - {file_metadata.filename}] {question}",
                response=response,
                ai_model="openai:chatgpt-4o-latest"
            )

            # Send response
            embed = discord.Embed(
                title=f"📄 {file_metadata.filename}",
                description=f"**Question:** {question}",
                color=0x0099ff
            )

            # Split response if needed
            if len(response) > 1024:
                embed.add_field(name="Answer", value=response[:1024] + "...", inline=False)
                await ctx.send(embed=embed)

                # Send rest in code blocks
                remaining = response[1024:]
                while remaining:
                    chunk = remaining[:1900]
                    await ctx.send(f"```{chunk}```")
                    remaining = remaining[1900:]
            else:
                embed.add_field(name="Answer", value=response, inline=False)
                await ctx.send(embed=embed)

    @commands.command(name='compare')
    async def compare_documents(self, ctx, file_id1: int, file_id2: int):
        """
        Compare two documents
        Usage: !compare 123 456
        """
        # Implementation for comparing two documents
        # This would fetch both documents and ask AI to compare them
        pass

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

            if hasattr(self.bot, 'autosave_channels') and str(ctx.channel.id) in self.bot.autosave_channels:
                # Check if the conversation seems important
                important_keywords = ['project', 'using', 'stack', 'decided', 'will use', 'going with',
                                      'authentication', 'database', 'api', 'endpoint', 'requirement']

                message_lower = message.lower()
                if any(keyword in message_lower for keyword in important_keywords):
                    # Auto-save as memory
                    tag = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    summary = f"Q: {message[:100]}... A: {response[:200]}..."

                    await self.vector_service.store_memory(
                        user_id=str(ctx.author.id),
                        channel_id=str(ctx.channel.id),
                        tag=tag,
                        content=summary
                    )

                    logger.logger.info(f"Auto-saved conversation as memory with tag: {tag}")

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

    @commands.command(name='autosave')
    @log_method()
    async def toggle_autosave(self, ctx):
        """Toggle automatic memory saving for important conversations"""
        # Store this preference per channel
        channel_id = str(ctx.channel.id)

        # You could store this in a database or in-memory dict
        if not hasattr(self.bot, 'autosave_channels'):
            self.bot.autosave_channels = set()

        if channel_id in self.bot.autosave_channels:
            self.bot.autosave_channels.remove(channel_id)
            await ctx.send("❌ Automatic memory saving disabled for this channel")
        else:
            self.bot.autosave_channels.add(channel_id)
            await ctx.send("✅ Automatic memory saving enabled for this channel")

    @commands.command(name='context')
    @log_method()
    async def show_context(self, ctx, *, query: str = "recent conversations"):
        """Show what context would be retrieved for a query"""
        async with ctx.typing():
            context = await self.vector_service.get_context_for_ai(
                query=query,
                channel_id=str(ctx.channel.id),
                max_context_length=4000
            )

            if not context:
                await ctx.send("❌ No context found for this query")
                return

            # Split context into chunks for Discord
            chunks = []
            lines = context.split('\n')
            current_chunk = "```\nRetrieved Context:\n"

            for line in lines:
                if len(current_chunk) + len(line) + 4 > 1990:  # 4 for closing ```
                    chunks.append(current_chunk + "```")
                    current_chunk = "```\n"
                current_chunk += line + "\n"

            if current_chunk != "```\n":
                chunks.append(current_chunk + "```")

            for chunk in chunks:
                await ctx.send(chunk)


async def setup(bot):
    await bot.add_cog(AIChatCog(bot))
