"""
AI Chat Cog for PairProgrammer Discord Bot

This cog provides the core AI interaction functionality, including chat commands,
document analysis, context retrieval, and conversation management. It integrates
with multiple AI providers and maintains conversation context through vector
database storage.

Key Features:
    - Multi-provider AI chat (OpenAI, Anthropic)
    - Context-aware conversations with memory
    - Document-specific Q&A capabilities
    - Message chunking for Discord's character limits
    - Automatic conversation storage for context building
    - Model selection and provider switching
    - Search and context preview functionality

Commands:
    - !chat: Full-featured AI chat with provider/model selection
    - !quick: Quick chat with default GPT-4 model
    - !askdoc: Question-answering about uploaded documents
    - !models: List available AI models
    - !search: Search conversation history
    - !context: Preview context retrieval
    - !autosave: Toggle automatic memory saving

Integration:
    - AIService: Multi-provider AI model access
    - VectorService: Context storage and retrieval
    - Discord: Message handling and user interaction

Author: PairProgrammer Team
"""

from datetime import datetime
import discord
from discord.ext import commands
from services.ai_service import AIService
from services.vector_service import VectorService
from utils.logger import get_logger, log_method

logger = get_logger(__name__)


class AIChatCog(commands.Cog):
    """
    Discord cog providing AI chat functionality and conversation management.
    
    This cog serves as the primary interface for AI interactions within Discord,
    handling multiple AI providers, context management, and conversation storage.
    It provides both simple quick-chat functionality and advanced features like
    document Q&A and context-aware conversations.
    
    Attributes:
        bot (commands.Bot): Discord bot instance
        ai_service (AIService): AI provider interface
        vector_service (VectorService): Vector database for context storage
        
    Features:
        - Multi-provider AI support (OpenAI, Anthropic)
        - Conversation context from vector database
        - Document-specific questioning
        - Message chunking for Discord limits
        - Automatic conversation storage
        - Debug and search capabilities
        
    Example Usage:
        !quick How do I implement async/await in Python?
        !chat anthropic claude-sonnet-4-0 Explain decorators
        !askdoc 123 What are the key findings in this paper?
    """
    
    def __init__(self, bot):
        """
        Initialize the AIChatCog with required services.
        
        Args:
            bot (commands.Bot): Discord bot instance
            
        Services Initialized:
            - AIService: For multi-provider AI interactions
            - VectorService: For context storage and retrieval
        """
        self.bot = bot
        self.ai_service = AIService()
        self.vector_service = VectorService()
        logger.logger.info("AIChatCog initialized")

    async def send_long_message(self, ctx, content: str, provider: str = None, model: str = None):
        """
        Send long messages to Discord with automatic chunking.
        
        Handles Discord's 2000-character limit by intelligently splitting long
        messages into multiple chunks while preserving formatting and adding
        provider/model headers when specified.
        
        Args:
            ctx (commands.Context): Discord command context
            content (str): Message content to send
            provider (str, optional): AI provider name for header
            model (str, optional): AI model name for header
            
        Features:
            - Respects Discord's 2000-character limit
            - Intelligent line-break splitting
            - Provider/model attribution headers
            - Code block continuation formatting
            - Logging of message chunks and lengths
            
        Chunking Strategy:
            1. If content fits in 1990 chars, then send it as single message
            2. Otherwise, split on any line breaks to preserve formatting
            3. Add continuation markers for code blocks
            4. Log chunk information for debugging
            
        Example:
            await self.send_long_message(
                ctx, 
                long_ai_response,
                provider='openai',
                model='gpt-4'
            )
            # Results in: **OpenAI (gpt-4):** followed by chunked content
        """
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

    @commands.command(name='askdoc', aliases=['ad', 'askedoc', 'question', 'qd'])
    @commands.cooldown(3, 60, commands.BucketType.user)
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

        # Send initial status message
        status_msg = await ctx.send(f"🔍 Analyzing {file_metadata.filename}...")

        try:
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
                        query=str(file_metadata.filename),  # Search by filename
                        channel_id=str(ctx.channel.id),
                        file_id=str(file_id),
                        top_k=10
                    )

                    if not doc_results:
                        await status_msg.edit(
                            content="❌ No content found for this document. It may not be indexed properly.")
                        return

                # Build context from document chunks
                doc_context = f"Document: {file_metadata.filename}\n"
                doc_context += f"Description: {file_metadata.description}\n"
                doc_context += f"File Type: {file_metadata.file_type or 'Unknown'}\n\n"
                doc_context += "=== DOCUMENT CONTENT ===\n\n"

                # Sort chunks by chunk_index to maintain order
                sorted_chunks = sorted(doc_results,
                                       key=lambda x: x['metadata'].get('chunk_index', 0))

                # Use more chunks for better context
                for chunk in sorted_chunks[:8]:  # Increased from 5 to 8
                    chunk_content = chunk['content']
                    # Remove the enhanced search prefix if present
                    if "Content:\n" in chunk_content:
                        chunk_content = chunk_content.split("Content:\n", 1)[1]

                    chunk_idx = chunk['metadata'].get('chunk_index', 0)
                    total_chunks = chunk['metadata'].get('total_chunks', 1)
                    doc_context += f"[Part {chunk_idx + 1}/{total_chunks}]\n{chunk_content}\n\n"

                # Update status
                await status_msg.edit(content=f"🤔 Thinking about your question...")

                # Create the question in a format that works better with the AI service
                formatted_question = f"""Based on the document '{file_metadata.filename}', please answer the following question:

    Question: {question}

    Please base your answer only on the content provided from the document."""

                # Get AI response with document context
                response = await self.ai_service.get_ai_response(
                    provider='openai',
                    model='chatgpt-4o-latest',
                    user_message=formatted_question,
                    context=doc_context
                )

                # Delete status message
                await status_msg.delete()

                # Store this Q&A in vector database for future reference
                await self.vector_service.store_conversation(
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    message=f"[Document Q&A - {file_metadata.filename}] {question}",
                    response=response,
                    ai_model="openai:chatgpt-4o-latest"
                )

                # Send response - FIXED PART
                embed = discord.Embed(
                    title=f"📄 {file_metadata.filename}",
                    color=0x0099ff
                )

                # Safely add the question field
                if len(question) > 1024:
                    embed.add_field(name="Question", value=question[:1021] + "...", inline=False)
                else:
                    embed.add_field(name="Question", value=question, inline=False)

                # Add file info
                embed.set_footer(text=f"File ID: {file_id} | {len(sorted_chunks)} chunks analyzed")

                # Handle response based on length
                if len(response) <= 1024:
                    # Response fits in embed
                    embed.add_field(name="Answer", value=response, inline=False)
                    await ctx.send(embed=embed)
                elif len(response) <= 4000:
                    # Response fits in embed description + field
                    embed.description = f"**Answer:**\n{response[:2000]}"
                    if len(response) > 2000:
                        embed.add_field(
                            name="Answer (continued)",
                            value=response[2000:3024],  # Max 1024 chars
                            inline=False
                        )
                    await ctx.send(embed=embed)

                    # Send any remaining text
                    if len(response) > 3024:
                        remaining = response[3024:]
                        while remaining:
                            chunk = remaining[:1900]
                            remaining = remaining[1900:]
                            await ctx.send(chunk)
                else:
                    # Very long response - send the embed with preview, then full response
                    embed.add_field(
                        name="Answer Preview",
                        value=response[:1021] + "...",
                        inline=False
                    )
                    await ctx.send(embed=embed)

                    # Send full response in chunks
                    await ctx.send("**Full Answer:**")
                    remaining = response
                    while remaining:
                        chunk = remaining[:1900]
                        remaining = remaining[1900:]
                        if len(chunk) > 1900:  # Safety check
                            chunk = chunk[:1900]
                        await ctx.send(chunk)

        except discord.HTTPException as e:
            logger.logger.error(f"Discord error in askdoc: {str(e)}", exc_info=True)
            try:
                await status_msg.edit(content=f"❌ Discord error: {str(e)}")
            except Exception:
                await ctx.send(f"❌ Error sending response: {str(e)}")
        except Exception as e:
            logger.logger.error(f"Error in askdoc command: {str(e)}", exc_info=True)
            try:
                await status_msg.edit(content=f"❌ Error: {str(e)}")
            except Exception:
                await ctx.send(f"❌ Error processing document question: {str(e)}")

    @commands.command(name='asksimple', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def ask_simple(self, ctx, file_id: int, *, question: str):
        """
        Simpler version of askdoc for debugging
        Usage: !asksimple 123 What is this about?
        """
        from database.models import FileMetadata, get_db

        # Get file info
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found")
            return

        await ctx.send(f"📄 Searching {file_metadata.filename}...")

        # Get document chunks
        doc_results = await self.vector_service.search_documents(
            query=question,
            channel_id=str(ctx.channel.id),
            file_id=str(file_id),
            top_k=5
        )

        if not doc_results:
            await ctx.send("❌ No content found")
            return

        # Build simple context
        context = f"From document '{file_metadata.filename}':\n\n"
        for i, chunk in enumerate(doc_results[:3]):
            # noinspection PyUnresolvedReferences
            content = chunk['content']
            if "Content:\n" in content:
                content = content.split("Content:\n", 1)[1]
            context += f"{content[:500]}...\n\n"

        # Simple message
        message = f"Question about {file_metadata.filename}: {question}"

        # Get response using the standard chat flow
        response = await self.ai_service.get_ai_response(
            provider='openai',
            model='chatgpt-4o-latest',
            user_message=message,
            context=context
        )

        # Send response
        await self.send_long_message(ctx, response, provider='openai', model='chatgpt-4o-latest')

    @commands.command(name='debugask', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def debug_askdoc(self, ctx, file_id: int):
        """
        Debug what askdoc sees
        Usage: !debugask 2
        """
        from database.models import FileMetadata, get_db

        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found")
            return

        # Search for chunks
        doc_results = await self.vector_service.search_documents(
            query="main findings results conclusion",  # Common terms
            channel_id=str(ctx.channel.id),
            file_id=str(file_id),
            top_k=3
        )

        embed = discord.Embed(
            title=f"Debug: {file_metadata.filename}",
            color=0xffff00
        )

        embed.add_field(name="File ID", value=str(file_id), inline=True)
        embed.add_field(name="Chunks Found", value=str(len(doc_results)), inline=True)

        if doc_results:
            # Show first chunk
            content = doc_results[0]['content']
            if "Content:\n" in content:
                content = content.split("Content:\n", 1)[1]

            embed.add_field(
                name="First Chunk Preview",
                value=content[:300] + "...",
                inline=False
            )

            # Test AI call with minimal context
            try:
                test_response = await self.ai_service.get_ai_response(
                    provider='openai',
                    model='chatgpt-4o-latest',
                    user_message="What is this document about?",
                    context=content[:500]
                )

                embed.add_field(
                    name="AI Test Response",
                    value=test_response[:300] + "...",
                    inline=False
                )
            except Exception as e:
                embed.add_field(
                    name="AI Test Error",
                    value=str(e),
                    inline=False
                )

        await ctx.send(embed=embed)

    @commands.command(name='chat', aliases=['c', 'ask', 'ai'])
    @commands.cooldown(3, 60, commands.BucketType.user)
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

            # noinspection PyUnresolvedReferences
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

    @commands.command(name='models', aliases=['model', 'list_models'])
    @commands.cooldown(3, 60, commands.BucketType.user)
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

    @commands.command(name='quick', aliases=['q', 'gpt', 'ask4'])
    @commands.cooldown(3, 60, commands.BucketType.user)
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

    @commands.command(name='search', aliases=['s', 'find', 'lookup'])
    @commands.cooldown(3, 60, commands.BucketType.user)
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

    @commands.command(name='autosave', aliases=['as', 'auto', 'toggle_save'])
    @log_method()
    @commands.cooldown(3, 60, commands.BucketType.user)
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

    @commands.command(name='context', aliases=['ctx', 'preview', 'show_context'])
    @commands.cooldown(3, 60, commands.BucketType.user)
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
