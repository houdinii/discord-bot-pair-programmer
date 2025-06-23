import discord
from discord.ext import commands, tasks
from datetime import datetime, timezone
from typing import List, Optional
import asyncio

from services.arxiv_service import ArxivService
from services.ai_service import AIService
from services.vector_service import VectorService
from config import ARXIV_SUGGESTION_CHANNEL
from utils.logger import get_logger

logger = get_logger(__name__)


class ArxivCog(commands.Cog):
    """arXiv integration with paper search, summaries, and AI-powered analysis"""

    def __init__(self, bot):
        self.bot = bot
        self.arxiv_service = ArxivService()
        self.ai_service = AIService()
        self.vector_service = VectorService()

        # Start suggestion task if channel is configured
        if hasattr(self, 'ARXIV_SUGGESTION_CHANNEL') and ARXIV_SUGGESTION_CHANNEL:
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
    @commands.cooldown(3, 300, commands.BucketType.user)  # Longer cooldown due to PDF processing
    async def load_paper(self, ctx, paper_id: str, *, description: str = ""):
        """
        Load and index an arXiv paper for AI analysis

        This downloads the PDF, extracts text, and stores it in the vector database
        for context-aware AI conversations and analysis.

        Usage:
            !arxiv_load 2304.03442
            !al 2304.03442 "Interesting paper about transformers"
        """
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

                # Check if already loaded
                existing = await self.vector_service.search_similar(
                    query=f"arxiv:{clean_id}",
                    channel_id=str(ctx.channel.id),
                    content_type=['arxiv_paper'],
                    top_k=1
                )

                if existing:
                    await status_msg.edit(
                        content=f"✅ Paper `{clean_id}` already loaded!\n"
                                f"Use `!askdoc` or `!q` to ask questions about it."
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

                # Store in vector database
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
                            'arxiv_id': clean_id,
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
        """Send daily paper suggestions to configured channel"""
        try:
            if not hasattr(self, 'ARXIV_SUGGESTION_CHANNEL') or not ARXIV_SUGGESTION_CHANNEL:
                return

            channel = self.bot.get_channel(int(ARXIV_SUGGESTION_CHANNEL))
            if not channel:
                return

            # Get recent papers in AI/ML categories
            categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
            papers = await self.arxiv_service.get_paper_suggestions(categories, max_results=3)

            if not papers:
                return

            embed = discord.Embed(
                title="📅 Daily arXiv Suggestions",
                description=f"Fresh papers from {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                color=0x00ff88,
                timestamp=datetime.now(timezone.utc)
            )

            for paper in papers:
                pub_date = paper['published'].strftime('%Y-%m-%d') if paper['published'] else 'Unknown'

                embed.add_field(
                    name=f"📄 {paper['title'][:50]}{'...' if len(paper['title']) > 50 else ''}",
                    value=f"**ID:** `{paper['id']}`\n"
                          f"**Authors:** {', '.join(paper['authors'][:2])}\n"
                          f"**Published:** {pub_date}\n"
                          f"[View Abstract]({paper['abs_url']})",
                    inline=False
                )

            embed.set_footer(text="Use !arxiv_load [paper_id] to analyze any paper")

            await channel.send(embed=embed)

        except Exception as e:
            logger.logger.error(f"Error in daily suggestions: {e}", exc_info=True)

    @daily_suggestions.before_loop
    async def before_daily_suggestions(self):
        """Wait for bot to be ready before starting suggestions"""
        await self.bot.wait_until_ready()


async def setup(bot):
    await bot.add_cog(ArxivCog(bot))
