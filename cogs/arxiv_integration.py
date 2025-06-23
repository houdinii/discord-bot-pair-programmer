from datetime import datetime, timezone

import discord
from discord.ext import commands, tasks

from config import ARXIV_SUGGESTION_CHANNEL
from services.ai_service import AIService
from services.arxiv_service import ArxivService
from services.recommendation_service import RecommendationService
from services.vector_service import VectorService
from utils.logger import get_logger

logger = get_logger(__name__)


class ArxivCog(commands.Cog):
    """arXiv integration with paper search, summaries, and AI-powered analysis"""

    def __init__(self, bot):
        self.bot = bot
        self.arxiv_service = ArxivService()
        self.ai_service = AIService()
        self.vector_service = VectorService()
        self.recommendation_service = RecommendationService(
            self.vector_service,
            self.arxiv_service
        )

        # Start suggestion task if channel is configured
        # Check if ARXIV_SUGGESTION_CHANNEL exists and is not None/empty
        if ARXIV_SUGGESTION_CHANNEL:
            self.daily_suggestions.start()

    def cog_unload(self):
        """Clean up tasks when cog is unloaded"""
        if hasattr(self, 'daily_suggestions'):
            self.daily_suggestions.cancel()

    @commands.command(name='arxiv_search', aliases=['search_papers', 'find_papers'])
    @commands.cooldown(5, 60, commands.BucketType.user)
    async def search_arxiv(self, ctx, *, query: str):
        """
        Search arXiv papers with advanced filtering and AI-powered relevance

        Usage:
            !arxiv_search machine learning transformers
            !as "attention is all you need"
            !find_papers quantum computing algorithms

        Features:
            - Searches titles, abstracts, and authors
            - Shows relevance scores and key metadata
            - Provides quick actions for each paper
        """
        if len(query.strip()) < 3:
            await ctx.send("❌ Search query must be at least 3 characters")
            return

        status_msg = await ctx.send(f"🔍 Searching arXiv for: **{query}**...")

        try:
            async with ctx.typing():
                # Search papers
                papers = await self.arxiv_service.search_papers(query, max_results=8)

                if not papers:
                    await status_msg.edit(content=f"❌ No papers found for: {query}")
                    return

                # Delete status message
                await status_msg.delete()

                # Create results embed
                embed = discord.Embed(
                    title=f"📚 arXiv Search Results",
                    description=f"Found {len(papers)} papers for: **{query}**",
                    color=0x0099ff
                )

                for i, paper in enumerate(papers[:5]):  # Show top 5 results
                    # Truncate title and abstract for display
                    title = paper['title'][:80] + "..." if len(paper['title']) > 80 else paper['title']
                    abstract = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']

                    # Format authors
                    authors = ", ".join(paper['authors'][:3])
                    if len(paper['authors']) > 3:
                        authors += f" (+{len(paper['authors']) - 3} more)"

                    embed.add_field(
                        name=f"📄 {title}",
                        value=f"**ID:** `{paper['id']}`\n"
                              f"**Authors:** {authors}\n"
                              f"**Category:** {paper['primary_category']}\n"
                              f"**Abstract:** {abstract}\n"
                              f"[View Paper]({paper['abs_url']})",
                        inline=False
                    )

                # Add quick actions
                embed.add_field(
                    name="🚀 Quick Actions",
                    value="• `!arxiv_load [paper_id]` - Load paper for analysis\n"
                          "• `!arxiv_summary [paper_id]` - Get AI summary\n"
                          "• `!arxiv_questions [paper_id]` - Generate discussion questions\n"
                          "• `!arxiv_code [paper_id]` - Generate code examples",
                    inline=False
                )

                await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error searching arXiv: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error searching arXiv: {str(e)}")

    @commands.command(name='arxiv_load', aliases=['al', 'load_paper'])
    @commands.cooldown(3, 300, commands.BucketType.user)
    async def load_paper(self, ctx, paper_id: str, *, description: str = ""):
        """Load and index an arXiv paper for AI analysis"""
        clean_id = self.arxiv_service.clean_paper_id(paper_id)

        status_msg = await ctx.send(f"📥 Loading paper `{clean_id}`...")

        try:
            async with ctx.typing():
                # Get paper metadata first
                await status_msg.edit(content=f"📥 Getting metadata for `{clean_id}`...")
                metadata = await self.arxiv_service.get_paper_metadata(clean_id)

                if not metadata:
                    await status_msg.edit(content=f"❌ Paper `{clean_id}` not found on arXiv")
                    return

                # Check if already loaded - be more specific in the search
                existing = await self.vector_service.search_similar(
                    query=f"arxiv_id:{clean_id}",  # More specific search
                    channel_id=str(ctx.channel.id),
                    content_type=['arxiv_paper'],
                    top_k=5
                )

                # Double-check by looking at metadata
                already_loaded = False
                for result in existing:
                    if result['metadata'].get('arxiv_id') == clean_id:
                        already_loaded = True
                        break

                if already_loaded:
                    await status_msg.edit(
                        content=f"✅ Paper `{clean_id}` already loaded!\n"
                                f"Use `!q` to ask questions about it or `!askdoc` for document-specific queries."
                    )
                    return

                # Download and parse PDF
                await status_msg.edit(content=f"📄 Downloading and parsing PDF...")
                content, pdf_metadata = await self.arxiv_service.download_and_parse_pdf(clean_id)

                # Chunk the content
                await status_msg.edit(content=f"🔄 Indexing content...")
                documents = self.arxiv_service.chunk_paper_content(
                    content, clean_id, metadata['title']
                )

                # Store in vector database with clear metadata
                vector_ids = []
                for doc in documents:
                    # Enhanced document content for better search
                    enhanced_content = f"arXiv Paper: {metadata['title']}\n"
                    enhanced_content += f"Authors: {', '.join(metadata['authors'])}\n"
                    enhanced_content += f"Abstract: {metadata['abstract'][:200]}...\n"
                    enhanced_content += f"Content:\n{doc.page_content}"

                    vector_id = await self.vector_service.store_document_chunk(
                        filename=f"arxiv_{clean_id}.pdf",
                        content=enhanced_content,
                        user_id=str(ctx.author.id),
                        channel_id=str(ctx.channel.id),
                        metadata={
                            **doc.metadata,
                            'arxiv_id': clean_id,  # Key identifier
                            'user_loaded': True,  # Mark as user-loaded
                            'load_timestamp': datetime.now(timezone.utc).isoformat(),
                            'description': description or f"arXiv paper: {metadata['title']}",
                            'authors': ', '.join(metadata['authors']),
                            'published': metadata['published'].isoformat() if metadata['published'] else None,
                            'categories': ', '.join(metadata['categories']),
                            'pdf_pages': pdf_metadata['total_pages']
                        }
                    )
                    vector_ids.append(vector_id)

                # Delete status message
                await status_msg.delete()

                # Success embed
                embed = discord.Embed(
                    title="📚 Paper Loaded Successfully!",
                    description=f"**{metadata['title']}**",
                    color=0x00ff00,
                    url=metadata['abs_url']
                )

                embed.add_field(
                    name="📄 Paper Info",
                    value=f"**ID:** `{clean_id}`\n"
                          f"**Authors:** {', '.join(metadata['authors'][:3])}\n"
                          f"**Published:** {metadata['published'].strftime('%Y-%m-%d') if metadata['published'] else 'Unknown'}\n"
                          f"**Category:** {metadata['primary_category']}",
                    inline=False
                )

                embed.add_field(
                    name="🔍 Processing Stats",
                    value=f"• **Chunks indexed:** {len(vector_ids)}\n"
                          f"• **Pages processed:** {pdf_metadata['total_pages']}\n"
                          f"• **Content length:** {len(content):,} characters",
                    inline=False
                )

                embed.add_field(
                    name="🚀 What's Next?",
                    value=f"• `!q Tell me about this paper` - General discussion\n"
                          f"• `!arxiv_summary {clean_id}` - Get AI summary\n"
                          f"• `!arxiv_questions {clean_id}` - Discussion questions\n"
                          f"• `!arxiv_code {clean_id}` - Generate code examples",
                    inline=False
                )

                await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error loading paper {clean_id}: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error loading paper: {str(e)}")

    @commands.command(name='arxiv_summary', aliases=['arxiv_sum', 'paper_summary'])
    @commands.cooldown(3, 120, commands.BucketType.user)
    async def summarize_paper(self, ctx, paper_id: str, summary_type: str = "comprehensive"):
        """
        Generate AI-powered paper summaries

        Summary types:
            - comprehensive: Detailed academic summary
            - key_points: Main findings and contributions
            - layman: Simple, accessible explanation
            - methodology: Focus on methods and approach

        Usage:
            !arxiv_summary 2304.03442
            !paper_summary 2304.03442 layman
        """
        clean_id = self.arxiv_service.clean_paper_id(paper_id)

        # Validate summary type
        valid_types = ['comprehensive', 'key_points', 'layman', 'methodology']
        if summary_type not in valid_types:
            await ctx.send(f"❌ Invalid summary type. Choose from: {', '.join(valid_types)}")
            return

        status_msg = await ctx.send(f"📝 Generating {summary_type} summary for `{clean_id}`...")

        try:
            async with ctx.typing():
                # Check if paper is loaded
                results = await self.vector_service.search_similar(
                    query=f"arxiv:{clean_id}",
                    channel_id=str(ctx.channel.id),
                    content_type=['arxiv_paper'],
                    top_k=10
                )

                if not results:
                    await status_msg.edit(
                        content=f"❌ Paper `{clean_id}` not loaded. Use `!arxiv_load {clean_id}` first."
                    )
                    return

                # Get paper metadata
                paper_title = results[0]['metadata'].get('title', 'Unknown Paper')

                # Build context from paper chunks
                context = f"arXiv Paper: {paper_title}\n"
                context += f"Paper ID: {clean_id}\n\n"

                # Add content from chunks
                for result in results:
                    chunk_content = result['content']
                    if "Content:\n" in chunk_content:
                        chunk_content = chunk_content.split("Content:\n", 1)[1]
                    context += chunk_content + "\n\n"

                # Generate summary based on type
                if summary_type == "comprehensive":
                    prompt = f"""Please provide a comprehensive academic summary of this paper. Include:
                    - Main research question and objectives
                    - Key methodology and approach
                    - Primary findings and results
                    - Significance and implications
                    - Limitations and future work

                    Paper content: {context[:4000]}"""

                elif summary_type == "key_points":
                    prompt = f"""Extract the key points from this paper in bullet format:
                    - Main contribution/innovation
                    - Key findings
                    - Important results
                    - Practical implications

                    Paper content: {context[:4000]}"""

                elif summary_type == "layman":
                    prompt = f"""Explain this paper in simple terms that anyone can understand:
                    - What problem does it solve?
                    - What did they do?
                    - What did they find?
                    - Why does it matter?

                    Avoid technical jargon and explain concepts clearly.

                    Paper content: {context[:4000]}"""

                elif summary_type == "methodology":
                    prompt = f"""Focus on the methodology and technical approach of this paper:
                    - Research design and methods
                    - Data and datasets used
                    - Algorithms or techniques
                    - Experimental setup
                    - Evaluation metrics

                    Paper content: {context[:4000]}"""

                # Get AI response
                await status_msg.edit(content=f"🤔 AI is analyzing the paper...")

                response = await self.ai_service.get_ai_response(
                    provider='openai',
                    model='chatgpt-4o-latest',
                    user_message=prompt,
                    context=""  # Context is included in the prompt
                )

                # Store this summary for future reference
                await self.vector_service.store_conversation(
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    message=f"[arXiv Summary - {summary_type}] {clean_id}",
                    response=response,
                    ai_model="openai:chatgpt-4o-latest"
                )

                # Delete status message
                await status_msg.delete()

                # Send summary
                embed = discord.Embed(
                    title=f"📄 {summary_type.replace('_', ' ').title()} Summary",
                    description=f"**{paper_title}**",
                    color=0x0099ff
                )

                embed.add_field(name="Paper ID", value=f"`{clean_id}`", inline=True)
                embed.add_field(name="Summary Type", value=summary_type.replace('_', ' ').title(), inline=True)

                # Handle long responses
                if len(response) <= 1024:
                    embed.add_field(name="Summary", value=response, inline=False)
                    await ctx.send(embed=embed)
                else:
                    # Send embed with preview, then full response
                    embed.add_field(
                        name="Summary Preview",
                        value=response[:1021] + "...",
                        inline=False
                    )
                    await ctx.send(embed=embed)

                    # Send full response in chunks
                    await ctx.send("**Full Summary:**")
                    remaining = response
                    while remaining:
                        chunk = remaining[:1900]
                        remaining = remaining[1900:]
                        await ctx.send(chunk)

        except Exception as e:
            logger.logger.error(f"Error summarizing paper {clean_id}: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error generating summary: {str(e)}")

    @commands.command(name='arxiv_questions', aliases=['aq', 'paper_questions'])
    @commands.cooldown(3, 120, commands.BucketType.user)
    async def generate_questions(self, ctx, paper_id: str):
        """
        Generate stimulating discussion questions about a paper

        Creates thought-provoking questions to facilitate:
        - Deep understanding of the paper
        - Critical analysis and discussion
        - Connections to other research
        - Practical applications

        Usage:
            !arxiv_questions 2304.03442
            !aq 2304.03442
        """
        clean_id = self.arxiv_service.clean_paper_id(paper_id)

        status_msg = await ctx.send(f"🤔 Generating discussion questions for `{clean_id}`...")

        try:
            async with ctx.typing():
                # Check if paper is loaded
                results = await self.vector_service.search_similar(
                    query=f"arxiv:{clean_id}",
                    channel_id=str(ctx.channel.id),
                    content_type=['arxiv_paper'],
                    top_k=5
                )

                if not results:
                    await status_msg.edit(
                        content=f"❌ Paper `{clean_id}` not loaded. Use `!arxiv_load {clean_id}` first."
                    )
                    return

                # Get paper info
                paper_title = results[0]['metadata'].get('title', 'Unknown Paper')
                authors = results[0]['metadata'].get('authors', 'Unknown Authors')

                # Build abstract and key content
                abstract = ""
                key_content = ""

                for result in results:
                    content = result['content']
                    if "Abstract:" in content:
                        abstract_start = content.find("Abstract:") + len("Abstract:")
                        abstract_end = content.find("Content:")
                        if abstract_end > abstract_start:
                            abstract = content[abstract_start:abstract_end].strip()

                    if "Content:\n" in content:
                        chunk_content = content.split("Content:\n", 1)[1]
                        key_content += chunk_content[:500] + "\n"

                # Generate questions
                prompt = f"""Generate 8-10 thought-provoking discussion questions about this research paper. 
                Create questions that:
                - Encourage deep analysis of the methodology and findings
                - Explore implications and applications
                - Connect to broader research areas
                - Challenge assumptions and limitations
                - Stimulate critical thinking

                Paper: {paper_title}
                Authors: {authors}
                Abstract: {abstract}
                Key Content: {key_content[:2000]}

                Format as a numbered list with brief context for each question."""

                response = await self.ai_service.get_ai_response(
                    provider='openai',
                    model='chatgpt-4o-latest',
                    user_message=prompt,
                    context=""
                )

                # Store for future reference
                await self.vector_service.store_conversation(
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    message=f"[arXiv Discussion Questions] {clean_id}",
                    response=response,
                    ai_model="openai:chatgpt-4o-latest"
                )

                # Delete status message
                await status_msg.delete()

                # Send questions
                embed = discord.Embed(
                    title="🤔 Discussion Questions",
                    description=f"**{paper_title}**",
                    color=0x9932cc
                )

                embed.add_field(name="Paper ID", value=f"`{clean_id}`", inline=True)
                embed.add_field(name="Purpose", value="Deep analysis & discussion", inline=True)

                # Handle response length
                if len(response) <= 1024:
                    embed.add_field(name="Questions", value=response, inline=False)
                    await ctx.send(embed=embed)
                else:
                    embed.add_field(
                        name="Questions Preview",
                        value=response[:1021] + "...",
                        inline=False
                    )
                    await ctx.send(embed=embed)

                    # Send full questions
                    await ctx.send("**Complete Discussion Questions:**")
                    remaining = response
                    while remaining:
                        chunk = remaining[:1900]
                        remaining = remaining[1900:]
                        await ctx.send(chunk)

        except Exception as e:
            logger.logger.error(f"Error generating questions for {clean_id}: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error generating questions: {str(e)}")

    @commands.command(name='arxiv_code', aliases=['ac', 'paper_code', 'implement'])
    @commands.cooldown(3, 180, commands.BucketType.user)
    async def generate_code(self, ctx, paper_id: str, *, request: str = ""):
        """
        Generate code implementations based on research papers

        Can generate:
        - Algorithm implementations
        - Model architectures
        - Experimental setups
        - Data processing pipelines
        - Visualization code

        Usage:
            !arxiv_code 2304.03442
            !ac 2304.03442 implement the main algorithm in Python
            !implement 2304.03442 create a PyTorch model
        """
        clean_id = self.arxiv_service.clean_paper_id(paper_id)

        status_msg = await ctx.send(f"💻 Generating code for `{clean_id}`...")

        try:
            async with ctx.typing():
                #  Check if paper is loaded
                results = await self.vector_service.search_similar(
                    query=f"arxiv:{clean_id}",
                    channel_id=str(ctx.channel.id),
                    content_type=['arxiv_paper'],
                    top_k=8
                )

                if not results:
                    await status_msg.edit(
                        content=f"❌ Paper `{clean_id}` not loaded. Use `!arxiv_load {clean_id}` first."
                    )
                    return

                # Get paper info and relevant content
                paper_title = results[0]['metadata'].get('title', 'Unknown Paper')

                # Build context focusing on methodology and algorithms
                context = f"Research Paper: {paper_title}\n"
                context += f"arXiv ID: {clean_id}\n\n"

                # Look for methodology, algorithm, and implementation details
                for result in results:
                    content = result['content']
                    if "Content:\n" in content:
                        chunk_content = content.split("Content:\n", 1)[1]
                        # Prioritize sections likely to contain implementation details
                        if any(keyword in chunk_content.lower() for keyword in
                               ['algorithm', 'method', 'implementation', 'procedure', 'approach', 'model']):
                            context += chunk_content + "\n\n"

                # Generate code based on request or general implementation
                if request:
                    prompt = f"""Based on this research paper, {request}. 

                    Paper: {paper_title}
                    Content: {context[:3000]}

                    Provide:
                    1. Clear, well-commented code
                    2. Explanation of the implementation
                    3. Usage examples
                    4. Any necessary dependencies

                    Use appropriate libraries (PyTorch, TensorFlow, NumPy, etc.) and best practices."""
                else:
                    prompt = f"""Generate a Python implementation of the main algorithm or method from this research paper.

                    Paper: {paper_title}
                    Content: {context[:3000]}

                    Provide:
                    1. Core algorithm implementation
                    2. Helper functions as needed
                    3. Clear documentation and comments
                    4. Usage example
                    5. Required dependencies

                    Focus on the key technical contribution and make it practical to use."""

                await status_msg.edit(content=f"🤖 AI is analyzing and implementing...")

                response = await self.ai_service.get_ai_response(
                    provider='openai',
                    model='chatgpt-4o-latest',
                    user_message=prompt,
                    context=""
                )

                # Store implementation for future reference
                await self.vector_service.store_conversation(
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    message=f"[arXiv Code Implementation] {clean_id}: {request or 'main algorithm'}",
                    response=response,
                    ai_model="openai:chatgpt-4o-latest"
                )

                # Delete status message
                await status_msg.delete()

                # Send implementation
                embed = discord.Embed(
                    title="💻 Code Implementation",
                    description=f"**{paper_title}**",
                    color=0x00ff00
                )

                embed.add_field(name="Paper ID", value=f"`{clean_id}`", inline=True)
                embed.add_field(name="Request", value=request or "Main algorithm", inline=True)

                # For code, we usually want to send the full response
                if len(response) <= 1024:
                    embed.add_field(name="Implementation", value=f"```python\n{response}\n```", inline=False)
                    await ctx.send(embed=embed)
                else:
                    # Send embed header, then code in separate messages
                    await ctx.send(embed=embed)

                    # Send code in chunks, maintaining code block formatting
                    await ctx.send("**Generated Implementation:**")

                    # Split response while trying to maintain code block structure
                    remaining = response
                    in_code_block = False

                    while remaining:
                        chunk_size = 1900
                        chunk = remaining[:chunk_size]
                        remaining = remaining[chunk_size:]

                        # Add code block markers if needed
                        if not in_code_block and '```' not in chunk:
                            chunk = f"```python\n{chunk}"
                            in_code_block = True

                        if in_code_block and not remaining:
                            chunk += "\n```"

                        await ctx.send(chunk)

        except Exception as e:
            logger.logger.error(f"Error generating code for {clean_id}: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error generating code: {str(e)}")

    @commands.command(name='arxiv_citations', aliases=['citations', 'cites'])
    @commands.cooldown(5, 300, commands.BucketType.user)
    async def get_citations(self, ctx, paper_id: str):
        """
        Get citation information for a paper

        Provides:
        - Formatted citations in different styles
        - Citation count (if available)
        - Related papers

        Usage:
            !arxiv_citations 2304.03442
            !cites 2304.03442
        """
        clean_id = self.arxiv_service.clean_paper_id(paper_id)

        status_msg = await ctx.send(f"📚 Getting citations for `{clean_id}`...")

        try:
            # Get paper metadata
            metadata = await self.arxiv_service.get_paper_metadata(clean_id)

            if not metadata:
                await status_msg.edit(content=f"❌ Paper `{clean_id}` not found")
                return

            # Generate different citation formats
            title = metadata['title']
            authors = metadata['authors']
            published = metadata['published']

            # Format authors
            if len(authors) == 1:
                author_str = authors[0]
            elif len(authors) == 2:
                author_str = f"{authors[0]} and {authors[1]}"
            else:
                author_str = f"{authors[0]} et al."

            # Create citation formats
            year = published.year if published else "n.d."

            # APA format
            apa_citation = f"{author_str} ({year}). {title}. arXiv preprint arXiv:{clean_id}."

            # MLA format
            mla_citation = f'{author_str} "{title}." arXiv preprint arXiv:{clean_id} ({year}).'

            # Chicago format
            chicago_citation = f'{author_str} "{title}." arXiv preprint arXiv:{clean_id} ({year}).'

            # BibTeX format
            bibtex_citation = f"""@article{{{clean_id.replace('.', '')},
    title={{{title}}},
    author={{{' and '.join(authors)}}},
    journal={{arXiv preprint arXiv:{clean_id}}},
    year={{{year}}}
}}"""

            await status_msg.delete()

            embed = discord.Embed(
                title="📚 Citation Formats",
                description=f"**{title}**",
                color=0xff6b35,
                url=metadata['abs_url']
            )

            embed.add_field(name="Paper ID", value=f"`{clean_id}`", inline=True)
            embed.add_field(name="Year", value=str(year), inline=True)
            embed.add_field(name="Authors", value=f"{len(authors)} author(s)", inline=True)

            embed.add_field(name="📖 APA Format", value=f"```{apa_citation}```", inline=False)
            embed.add_field(name="📖 MLA Format", value=f"```{mla_citation}```", inline=False)
            embed.add_field(name="📖 BibTeX Format", value=f"```bibtex\n{bibtex_citation}```", inline=False)

            await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error getting citations for {clean_id}: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error getting citations: {str(e)}")

    @commands.command(name='arxiv_recommend', aliases=['ar', 'recommend_papers'])
    @commands.cooldown(5, 180, commands.BucketType.user)
    async def recommend_papers(self, ctx, *, interests: str = ""):
        """
        Get personalized paper recommendations

        Based on:
        - Your specified interests
        - Previously loaded papers
        - Recent trending papers
        - AI/ML categories by default

        Usage:
            !arxiv_recommend
            !ar natural language processing
            !recommend_papers computer vision transformers
        """
        status_msg = await ctx.send("🔍 Finding paper recommendations...")

        try:
            async with ctx.typing():
                # If no interests specified, use recent AI/ML papers
                if not interests:
                    categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
                    papers = await self.arxiv_service.get_paper_suggestions(categories, max_results=6)
                else:
                    # Search based on interests
                    papers = await self.arxiv_service.search_papers(interests, max_results=6)

                if not papers:
                    await status_msg.edit(content="❌ No recommendations found")
                    return

                await status_msg.delete()

                embed = discord.Embed(
                    title="🌟 Paper Recommendations",
                    description=f"{'Based on: ' + interests if interests else 'Recent trending papers in AI/ML'}",
                    color=0xffd700
                )

                for i, paper in enumerate(papers[:4]):  # Top 4 recommendations
                    # Format publication date
                    pub_date = paper['published'].strftime('%Y-%m-%d') if paper['published'] else 'Unknown'

                    embed.add_field(
                        name=f"📄 {paper['title'][:60]}{'...' if len(paper['title']) > 60 else ''}",
                        value=f"**ID:** `{paper['id']}`\n"
                              f"**Authors:** {', '.join(paper['authors'][:2])}\n"
                              f"**Published:** {pub_date}\n"
                              f"**Category:** {paper['primary_category']}\n"
                              f"[View Abstract]({paper['abs_url']})",
                        inline=False
                    )

                embed.add_field(
                    name="🚀 Next Steps",
                    value="• `!arxiv_load [paper_id]` - Load for analysis\n"
                          "• `!arxiv_summary [paper_id]` - Get summary\n"
                          "• `!arxiv_search [topic]` - Search specific topics",
                    inline=False
                )

                await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error getting recommendations: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error getting recommendations: {str(e)}")

    @tasks.loop(hours=24)
    async def daily_suggestions(self):
        """Send personalized daily paper suggestions based on context"""
        try:
            if not ARXIV_SUGGESTION_CHANNEL:
                return

            channel = self.bot.get_channel(int(ARXIV_SUGGESTION_CHANNEL))
            if not channel:
                return

            logger.logger.info(f"Generating daily suggestions for channel {channel.id}")

            # Get personalized recommendations (this does NOT load papers into vector DB)
            recommendations = await self.recommendation_service.get_personalized_recommendations(
                channel_id=str(channel.id),
                max_results=5
            )

            # If no personalized recommendations, fall back to trending papers
            if not recommendations:
                logger.logger.info("No personalized recommendations, using trending papers")
                categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
                recommendations = await self.arxiv_service.get_paper_suggestions(
                    categories,
                    max_results=5
                )
                recommendation_summary = "Today's trending AI/ML papers"
            else:
                # Generate summary of why these were recommended
                recommendation_summary = await self.recommendation_service.generate_recommendation_summary(
                    str(channel.id),
                    recommendations
                )

            # Create embed with recommendations (NO LOADING INTO VECTOR DB)
            embed = discord.Embed(
                title="🌅 Your Daily arXiv Recommendations",
                description=recommendation_summary,
                color=0x00ff88,
                timestamp=datetime.now(timezone.utc)
            )

            # Add paper details
            for i, paper in enumerate(recommendations[:4], 1):
                pub_date = paper['published'].strftime('%Y-%m-%d') if paper.get('published') else 'Unknown'

                # Build field value
                field_value = f"**ID:** `{paper['id']}`\n"
                field_value += f"**Authors:** {', '.join(paper['authors'][:2])}\n"
                field_value += f"**Published:** {pub_date}\n"

                # Add relevance reason if available
                if paper.get('relevance_reason'):
                    field_value += f"**Why:** {paper['relevance_reason']}\n"

                field_value += f"[View Abstract]({paper['abs_url']})"

                embed.add_field(
                    name=f"{i}. {paper['title'][:60]}{'...' if len(paper['title']) > 60 else ''}",
                    value=field_value,
                    inline=False
                )

            # Add interaction analysis
            interests = await self.recommendation_service.analyze_user_interests(str(channel.id))

            if interests['total_interactions'] > 0:
                stats_text = f"📊 Based on {interests['total_interactions']} interactions"
                if interests['top_keywords']:
                    top_keywords = [kw for kw, _ in interests['top_keywords'][:3]]
                    stats_text += f"\n🔑 Top interests: {', '.join(top_keywords)}"

                embed.add_field(
                    name="Your Research Profile",
                    value=stats_text,
                    inline=False
                )

            # Emphasize that these are suggestions, not loaded papers
            embed.set_footer(text="💡 Use !arxiv_load [paper_id] to analyze any paper")

            await channel.send(embed=embed)

            logger.logger.info(f"Sent {len(recommendations)} personalized recommendations (not loaded)")

        except Exception as e:
            logger.logger.error(f"Error in daily suggestions: {e}", exc_info=True)

    @daily_suggestions.before_loop
    async def before_daily_suggestions(self):
        """Wait for bot to be ready before starting suggestions"""
        await self.bot.wait_until_ready()

    # Add a manual command to trigger recommendations
    @commands.command(name='arxiv_suggest', aliases=['suggest', 'daily'])
    @commands.cooldown(1, 3600, commands.BucketType.channel)  # Once per hour per channel
    async def manual_suggestions(self, ctx):
        """
        Get personalized paper suggestions based on your activity

        Analyzes your:
        - Loaded papers and documents
        - Conversations about research
        - Saved memories and notes
        - GitHub repositories

        Usage:
            !arxiv_suggest
            !suggest
            !daily
        """
        status_msg = await ctx.send("🔍 Analyzing your research interests...")

        try:
            async with ctx.typing():
                # Get personalized recommendations
                recommendations = await self.recommendation_service.get_personalized_recommendations(
                    channel_id=str(ctx.channel.id),
                    max_results=6
                )

                # Get interest analysis
                interests = await self.recommendation_service.analyze_user_interests(
                    str(ctx.channel.id)
                )

                await status_msg.delete()

                # Create main embed
                embed = discord.Embed(
                    title="🎯 Personalized Paper Recommendations",
                    description=await self.recommendation_service.generate_recommendation_summary(
                        str(ctx.channel.id),
                        recommendations
                    ),
                    color=0x00ff88
                )

                # Add recommendations
                if recommendations:
                    for i, paper in enumerate(recommendations[:4], 1):
                        pub_date = paper['published'].strftime('%Y-%m-%d') if paper.get('published') else 'Unknown'

                        field_value = f"**ID:** `{paper['id']}`\n"

                        # Show relevance score if available
                        if paper.get('relevance_score'):
                            field_value += f"**Relevance:** {paper['relevance_score']:.1f}/100\n"

                        if paper.get('relevance_reason'):
                            field_value += f"**Match:** {paper['relevance_reason']}\n"

                        field_value += f"**Published:** {pub_date}\n"
                        field_value += f"[View]({paper['abs_url']})"

                        embed.add_field(
                            name=f"{i}. {paper['title'][:50]}{'...' if len(paper['title']) > 50 else ''}",
                            value=field_value,
                            inline=False
                        )
                else:
                    embed.add_field(
                        name="No personalized recommendations yet",
                        value="Load some papers and interact with them to get personalized suggestions!",
                        inline=False
                    )

                # Add interest profile
                if interests['total_interactions'] > 0:
                    profile_parts = []

                    if interests['top_keywords']:
                        top_kw = [kw for kw, count in interests['top_keywords'][:5]]
                        profile_parts.append(f"**Keywords:** {', '.join(top_kw)}")

                    if interests['top_categories']:
                        top_cat = [cat for cat, _ in interests['top_categories'][:3]]
                        profile_parts.append(f"**Categories:** {', '.join(top_cat)}")

                    if interests['total_papers'] > 0:
                        profile_parts.append(f"**Papers analyzed:** {interests['total_papers']}")

                    embed.add_field(
                        name="📊 Your Research Profile",
                        value="\n".join(profile_parts) if profile_parts else "Building your profile...",
                        inline=False
                    )

                embed.set_footer(text="Recommendations improve as you use the bot more!")

                await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error generating suggestions: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error generating suggestions: {str(e)}")

    # Add a command to analyze research interests
    @commands.command(name='arxiv_profile', aliases=['research_profile', 'my_interests'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def show_research_profile(self, ctx):
        """
        Show your research profile based on your activity

        Displays:
        - Top research interests
        - Frequently accessed papers
        - Preferred categories
        - Favorite authors

        Usage:
            !arxiv_profile
            !research_profile
            !my_interests
        """
        status_msg = await ctx.send("📊 Analyzing your research profile...")

        try:
            async with ctx.typing():
                interests = await self.recommendation_service.analyze_user_interests(
                    str(ctx.channel.id),
                    days_back=90  # Look at last 3 months
                )

                await status_msg.delete()

                embed = discord.Embed(
                    title="🔬 Your Research Profile",
                    description=f"Based on {interests['total_interactions']} interactions",
                    color=0x9932cc
                )

                # Top keywords/interests
                if interests['top_keywords']:
                    keywords_text = "\n".join([
                        f"• **{kw}** ({count} mentions)"
                        for kw, count in interests['top_keywords'][:8]
                    ])
                    embed.add_field(
                        name="🔑 Research Interests",
                        value=keywords_text[:1024],  # Discord field limit
                        inline=False
                    )

                # Top categories
                if interests['top_categories']:
                    categories_text = "\n".join([
                        f"• **{cat}** ({count} papers)"
                        for cat, count in interests['top_categories']
                    ])
                    embed.add_field(
                        name="📚 Preferred Categories",
                        value=categories_text,
                        inline=True
                    )

                # Top authors
                if interests['top_authors']:
                    authors_text = "\n".join([
                        f"• {author[:30]}"
                        for author, _ in interests['top_authors']
                    ])
                    embed.add_field(
                        name="👥 Followed Authors",
                        value=authors_text,
                        inline=True
                    )

                # Statistics
                stats_text = f"📄 Papers loaded: **{interests['total_papers']}**\n"
                stats_text += f"💬 Total interactions: **{interests['total_interactions']}**"

                embed.add_field(
                    name="📈 Activity Stats",
                    value=stats_text,
                    inline=False
                )

                # Recommendations
                embed.add_field(
                    name="💡 Improve Your Profile",
                    value="• Load more papers with `!arxiv_load`\n"
                          "• Ask questions with `!q` about papers\n"
                          "• Save insights with `!remember`\n"
                          "• Get suggestions with `!arxiv_suggest`",
                    inline=False
                )

                await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error showing research profile: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error analyzing profile: {str(e)}")

    @commands.Cog.listener()
    async def on_message(self, message):
        """Monitor for research discussions and suggest papers when relevant"""
        # Don't respond to bots or DMs
        if message.author.bot or not message.guild:
            return

        # Check if message mentions research topics
        research_triggers = [
            "looking for papers",
            "any papers about",
            "research on",
            "studies about",
            "literature on",
            "papers related to",
            "recommendations for"
        ]

        message_lower = message.content.lower()

        for trigger in research_triggers:
            if trigger in message_lower:
                # Extract the topic after the trigger
                topic_start = message_lower.find(trigger) + len(trigger)
                topic = message.content[topic_start:].strip()

                # Limit topic length
                if len(topic) > 100:
                    topic = topic[:100]

                # Suggest searching
                embed = discord.Embed(
                    title="📚 Paper Search Helper",
                    description=f"I noticed you're looking for research papers!",
                    color=0x00ff88
                )

                embed.add_field(
                    name="Quick Actions",
                    value=f"• `!arxiv_search {topic}` - Search for papers\n"
                          f"• `!arxiv_suggest` - Get personalized recommendations\n"
                          f"• `!arxiv_recommend {topic}` - Topic-based suggestions",
                    inline=False
                )

                await message.reply(embed=embed, mention_author=False)
                break

    # Add a weekly summary task
    @tasks.loop(hours=168)  # Weekly
    async def weekly_research_summary(self):
        """Send weekly research summary and recommendations"""
        try:
            if not ARXIV_SUGGESTION_CHANNEL:
                return

            channel = self.bot.get_channel(int(ARXIV_SUGGESTION_CHANNEL))
            if not channel:
                return

            # Analyze the week's activity
            interests = await self.recommendation_service.analyze_user_interests(
                str(channel.id),
                days_back=7
            )

            # Get recommendations
            recommendations = await self.recommendation_service.get_personalized_recommendations(
                channel_id=str(channel.id),
                max_results=10
            )

            embed = discord.Embed(
                title="📊 Weekly Research Summary",
                description="Your research activity this week",
                color=0xffd700,
                timestamp=datetime.now(timezone.utc)
            )

            # Activity summary
            if interests['total_interactions'] > 0:
                activity_text = f"• Interactions: **{interests['total_interactions']}**\n"

                if interests['total_papers'] > 0:
                    activity_text += f"• Papers explored: **{interests['total_papers']}**\n"

                if interests['top_keywords']:
                    top_topics = [kw for kw, _ in interests['top_keywords'][:3]]
                    activity_text += f"• Top topics: {', '.join(top_topics)}"

                embed.add_field(
                    name="📈 This Week's Activity",
                    value=activity_text,
                    inline=False
                )

            # Top recommendations for next week
            if recommendations:
                rec_text = ""
                for i, paper in enumerate(recommendations[:5], 1):
                    rec_text += f"{i}. [{paper['title'][:40]}...]({paper['abs_url']})\n"

                embed.add_field(
                    name="🎯 Recommended for Next Week",
                    value=rec_text,
                    inline=False
                )

            embed.set_footer(text="Keep exploring! Your recommendations improve with usage.")

            await channel.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error in weekly summary: {e}", exc_info=True)

    @commands.command(name='arxiv_loaded', aliases=['loaded_papers', 'my_papers'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def show_loaded_papers(self, ctx):
        """
        Show papers you've actually loaded and indexed

        This shows only papers loaded via !arxiv_load, not just recommended papers.
        """
        try:
            # Get papers that were explicitly loaded
            results = await self.vector_service.search_similar(
                query="arxiv paper",
                channel_id=str(ctx.channel.id),
                content_type=['arxiv_paper'],
                top_k=50
            )

            # Group by paper ID to avoid duplicates
            loaded_papers = {}
            for result in results:
                metadata = result['metadata']
                arxiv_id = metadata.get('arxiv_id')
                if arxiv_id and metadata.get('user_loaded'):
                    if arxiv_id not in loaded_papers:
                        loaded_papers[arxiv_id] = {
                            'title': metadata.get('title', 'Unknown'),
                            'authors': metadata.get('authors', ''),
                            'load_timestamp': metadata.get('load_timestamp'),
                            'chunks': 0
                        }
                    loaded_papers[arxiv_id]['chunks'] += 1

            if not loaded_papers:
                await ctx.send("📚 No papers loaded yet. Use `!arxiv_load [paper_id]` to load papers for analysis.")
                return

            embed = discord.Embed(
                title="📚 Your Loaded Papers",
                description=f"You have {len(loaded_papers)} papers loaded and indexed",
                color=0x0099ff
            )

            # Sort by load timestamp (most recent first)
            sorted_papers = sorted(
                loaded_papers.items(),
                key=lambda x: x[1].get('load_timestamp', ''),
                reverse=True
            )

            for paper_id, info in sorted_papers[:10]:  # Show top 10
                title = info['title'][:50] + "..." if len(info['title']) > 50 else info['title']

                field_value = f"**ID:** `{paper_id}`\n"
                field_value += f"**Authors:** {info['authors'][:50]}...\n" if len(info['authors']) > 50 else f"**Authors:** {info['authors']}\n"
                field_value += f"**Chunks:** {info['chunks']}\n"

                if info.get('load_timestamp'):
                    load_date = info['load_timestamp'][:10]  # Just the date part
                    field_value += f"**Loaded:** {load_date}"

                embed.add_field(
                    name=f"📄 {title}",
                    value=field_value,
                    inline=False
                )

            if len(loaded_papers) > 10:
                embed.add_field(
                    name="Note",
                    value=f"Showing 10 of {len(loaded_papers)} loaded papers",
                    inline=False
                )

            embed.set_footer(text="These papers are available for !q, !arxiv_summary, etc.")

            await ctx.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error showing loaded papers: {e}", exc_info=True)
            await ctx.send(f"❌ Error retrieving loaded papers: {str(e)}")

async def setup(bot):
    await bot.add_cog(ArxivCog(bot))
