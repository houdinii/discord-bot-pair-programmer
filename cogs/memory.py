"""
Memory System Cog for PairProgrammer Discord Bot

This cog provides a comprehensive memory management system that allows users
to save, retrieve, and organize important information within Discord channels.
It uses vector database storage for semantic search and persistent memory
across conversations.

Key Features:
    - Tagged memory storage with semantic search
    - Channel-based memory isolation
    - Content-based memory retrieval
    - Memory management (save, recall, list, delete)
    - Vector database statistics and debugging
    - Automatic timestamping and metadata

Memory Types:
    - Project information and decisions
    - Technical specifications and requirements
    - Team agreements and conventions
    - Important links and resources
    - Meeting notes and action items

Commands:
    - !remember: Save tagged memories
    - !recall: Search memories by content
    - !get_memory: Retrieve specific memory by tag
    - !list_memories: List all saved memories
    - !forget: Delete memories by tag
    - !stats: Show vector database statistics
    - !debug_memories: Debug memory storage issues

Integration:
    - VectorService: Vector database operations
    - Channel isolation: Each Discord channel has separate memory space
    - AI Context: Memories automatically included in AI responses

Author: PairProgrammer Team
"""

import discord
from discord.ext import commands

from services.vector_service import VectorService


class MemoryCog(commands.Cog):
    """
    Discord cog for persistent memory and information management.
    
    This cog enables users to create a persistent knowledge base within
    Discord channels using a vector database for storage and semantic search.
    Each channel maintains its own isolated memory space for privacy and
    organization.
    
    Attributes:
        bot (commands.Bot): Discord bot instance
        vector_service (VectorService): Vector database service for storage
        
    Memory Organization:
        - Tagged storage: Each memory has a unique tag for easy retrieval
        - Content-based search: Semantic search across memory content
        - Channel isolation: Memories are scoped to specific Discord channels
        - Timestamping: Automatic timestamp tracking for all memories
        
    Use Cases:
        - Project documentation and decisions
        - Technical specifications and requirements
        - Team conventions and standards
        - Important URLs and resources
        - Meeting notes and action items
        
    Example Usage:
        !remember auth_strategy "Using JWT tokens with 1-hour expiration"
        !recall authentication
        !list_memories
    """
    def __init__(self, bot):
        """
        Initialize the MemoryCog with required services.
        
        Args:
            bot (commands.Bot): Discord bot instance
            
        Services Initialized:
            - VectorService: For memory storage and retrieval operations
        """
        self.bot = bot
        self.vector_service = VectorService()

    @commands.command(name='remember', aliases=['r', 'save', 'mem', 'note'])
    async def remember_context(self, ctx, tag: str, *, description: str):
        """
        Save important information with a searchable tag for future retrieval.
        
        Creates a new memory entry in the vector database with the specified
        tag and content. The memory is stored with channel isolation and
        automatic timestamping for organization and retrieval.
        
        Args:
            ctx (commands.Context): Discord command context
            tag (str): Unique identifier for the memory (no spaces)
            description (str): Content to remember (supports multiple words)
            
        Memory Storage:
            - Vector database storage for semantic search
            - Channel-based isolation (memories are private to this channel)
            - Automatic timestamping for chronological organization
            - Metadata preservation (user ID, channel ID, creation time)
            
        Tag Requirements:
            - Must be a single word (no spaces)
            - Should be descriptive and unique within the channel
            - Case-sensitive for exact retrieval
            - Examples: auth_strategy, database_config, api_endpoints
            
        Content Features:
            - Supports any length of text content
            - Automatically indexed for semantic search
            - Can include technical details, URLs, code snippets
            - Preserves formatting and special characters
            
        Response:
            Shows confirmation with:
            - Memory tag
            - Full content preview
            - Vector database ID (truncated)
            - Success confirmation
            
        Example:
            !remember auth_strategy "Using JWT tokens with 1-hour expiration and refresh tokens"
            !save database_config "PostgreSQL 14 with connection pooling via pgbouncer"
            !note meeting_notes "Decided to use React 18 with TypeScript for frontend"
            
        Aliases: r, save, mem, note
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

    @commands.command(name='recall', aliases=['rc', 'find_memory', 'search_memory'])
    async def recall_memory(self, ctx, *, query: str):
        """
        Search and retrieve memories based on content similarity.
        
        Performs semantic search across all memory content in the current
        channel to find relevant information. Uses AI embeddings to match
        concepts and related terms, not just exact keyword matches.
        
        Args:
            ctx (commands.Context): Discord command context
            query (str): Search terms or concepts to find
            
        Search Features:
            - Semantic search using AI embeddings
            - Searches memory CONTENT, not tags
            - Finds conceptually related information
            - Returns relevance scores for ranking
            
        Search Capabilities:
            - Keyword matching (exact and partial)
            - Concept similarity (related technologies, patterns)
            - Context awareness (understanding relationships)
            - Multi-term queries (searches across multiple concepts)
            
        Results Display:
            - Memory tag and relevance score
            - Memory content (full text)
            - Creation timestamp
            - Sorted by relevance (highest first)
            
        Search Scope:
            - Only searches memories in the current Discord channel
            - Returns maximum 5 most relevant results
            - Shows relevance score (0.0 to 1.0, higher = more relevant)
            
        Example Queries:
            !recall authentication (finds auth-related memories)
            !rc React PostgreSQL (finds memories about both technologies)
            !search_memory JWT tokens (finds token-related content)
            !find_memory database connection (finds DB connection info)
            
        Tips:
            - Use multiple keywords for better results
            - Try different phrasings if no results found
            - Use !list_memories to see all available memories
            
        Aliases: rc, find_memory, search_memory
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
                f"❌ No memories found with content related to: {query}\n💡 Tip: Try !list_memories to see all saved memories.")
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

    @commands.command(name='get_memory', aliases=['gm', 'getmem', 'memory'])
    async def get_memory_by_tag(self, ctx, tag: str):
        """
        Retrieve a specific memory using its exact tag.
        
        Looks up and displays a memory by its exact tag name. This is useful
        when you know the specific tag and want to retrieve the full content
        without searching.
        
        Args:
            ctx (commands.Context): Discord command context
            tag (str): Exact tag name of the memory to retrieve
            
        Tag Matching:
            - Requires exact tag match (case-sensitive)
            - No partial matching or fuzzy search
            - Faster than content search for known tags
            
        Memory Display:
            - Full memory content (no truncation)
            - Memory tag as title
            - Creation timestamp
            - Complete formatting preservation
            
        Comparison with !recall:
            - !get_memory: Exact tag lookup, fast retrieval
            - !recall: Content search, semantic matching
            
        Use Cases:
            - Quick retrieval of known memories
            - Accessing frequently referenced information
            - Verifying exact memory content
            - Sharing specific documented decisions
            
        Error Handling:
            - Clear message if tag doesn't exist
            - Suggestions to use !list_memories to see available tags
            
        Example:
            !get_memory auth_strategy
            !gm database_config
            !memory api_endpoints
            
        Tips:
            - Use !list_memories to see all available tags
            - Tags are case-sensitive
            - Use underscores instead of spaces in tags
            
        Aliases: gm, getmem, memory
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

    @commands.command(name='list_memories', aliases=['lm', 'memories', 'list_mem', 'mems'])
    async def list_memories(self, ctx):
        """
        Display all saved memories in the current Discord channel.
        
        Shows a comprehensive list of all memories stored in this channel,
        including tags, content previews, and organization information.
        Useful for browsing available memories and finding relevant tags.
        
        Display Information:
            - Memory tag names
            - Content preview (first 100 characters)
            - Total memory count
            - Organized in chronological order
            
        Channel Isolation:
            - Only shows memories from the current Discord channel
            - Each channel maintains separate memory storage
            - No cross-channel memory visibility
            
        Content Preview:
            - Shows first 100 characters of each memory
            - Adds "..." if content is longer
            - Preserves important context while keeping display compact
            
        Pagination:
            - Shows maximum 25 memories per command
            - Displays total count if more than 25 exist
            - Use specific commands to access remaining memories
            
        Empty State Handling:
            - Clear guidance if no memories exist
            - Shows vector database statistics for troubleshooting
            - Provides instructions for creating first memory
            
        Organization Tips:
            Memories are useful for:
            - Project decisions and specifications
            - Technical configuration details
            - Team agreements and conventions
            - Important URLs and resources
            - Meeting notes and action items
            
        Example Output:
            Saved Memories (3 total)
            • auth_strategy: Using JWT tokens with 1-hour expiration...
            • database_config: PostgreSQL 14 with connection pooling...
            • api_endpoints: REST API with versioning at /api/v1/...
            
        Example:
            !list_memories
            !lm
            !memories
            
        Aliases: lm, memories, list_mem, mems
        """

        # Try the alternative method that uses multiple queries
        memories = await self.vector_service.get_all_memories(
            channel_id=str(ctx.channel.id)
        )

        if not memories:
            # If still no results, check if there are ANY vectors
            stats = await self.vector_service.get_stats()
            if stats['total_vectors'] == 0:
                await ctx.send("❌ No memories saved yet. Use !remember [tag] [content] to save memories.")
            else:
                await ctx.send(
                    f"❌ No memories found in this channel (Total vectors in database: {stats['total_vectors']})")
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

    @commands.command(name='forget', aliases=['fg', 'delete_memory', 'delmem'])
    async def forget_memory(self, ctx, tag: str):
        """
        Delete a memory by its tag name permanently.
        
        Removes a memory from the vector database using its exact tag.
        This operation cannot be undone, so use with caution for important
        information.
        
        Args:
            ctx (commands.Context): Discord command context
            tag (str): Exact tag name of the memory to delete
            
        Deletion Process:
            - Searches for exact tag match in current channel
            - Permanently removes from vector database
            - Cannot be undone or recovered
            - Immediate confirmation of deletion
            
        Security:
            - Only deletes memories from the current Discord channel
            - Requires exact tag match (case-sensitive)
            - No wildcard or partial deletion
            
        Use Cases:
            - Removing outdated information
            - Cleaning up temporary notes
            - Correcting mistakes (delete and re-create)
            - Managing storage space
            
        Alternatives to Deletion:
            Instead of deleting, consider:
            - Creating updated memory with new tag
            - Using versioned tags (auth_v1, auth_v2)
            - Adding "DEPRECATED" prefix to old memories
            
        Error Handling:
            - Clear message if tag doesn't exist
            - No accidental deletion of wrong memories
            - Confirmation message on successful deletion
            
        Example:
            !forget old_auth_method
            !fg deprecated_config
            !delete_memory temp_notes
            
        Warning:
            This action cannot be undone. Make sure you have the correct
            tag name and that you really want to delete the memory.
            
        Aliases: fg, delete_memory, delmem
        """

        # Delete it from vector database
        success = await self.vector_service.delete_by_tag(
            channel_id=str(ctx.channel.id),
            tag=tag
        )

        if success:
            await ctx.send(f"✅ Forgot all memories with tag: {tag}")
        else:
            await ctx.send(f"❌ No memories found with tag: {tag}")

    @commands.command(name='stats', aliases=['st', 'statistics', 'info'])
    async def vector_stats(self, ctx):
        """
        Display comprehensive vector database statistics and information.
        
        Shows technical details about the vector database status, capacity,
        and usage metrics. Useful for monitoring system health and
        understanding storage utilization.
        
        Statistics Displayed:
            - Total Vectors: Number of stored items across all channels
            - Dimension: Vector embedding dimensions (typically 1536)
            - Index Fullness: Percentage of database capacity used
            
        Vector Types Included:
            - Memories (tagged information)
            - Conversations (chat history)
            - Documents (uploaded files)
            - GitHub repository content
            
        Technical Information:
            - Vector database: Pinecone (cloud-hosted)
            - Embedding model: OpenAI text-embedding-3-small
            - Dimension size: 1536-dimensional vectors
            - Storage: Distributed across multiple namespaces
            
        Monitoring Use Cases:
            - Check system capacity and usage
            - Verify data is being stored correctly
            - Monitor growth trends
            - Troubleshoot storage issues
            
        Index Fullness:
            - 0-50%: Normal usage, plenty of capacity
            - 50-80%: Moderate usage, monitor growth
            - 80-95%: High usage, consider cleanup
            - 95%+: Near capacity, action required
            
        Related Commands:
            - !debug_memories: Memory-specific debugging
            - !list_memories: Channel-specific memory count
            
        Example Output:
            Vector Database Stats
            • Total Vectors: 1,247
            • Dimension: 1536
            • Index Fullness: 23.45%
            
        Example:
            !stats
            !statistics
            
        Aliases: st, statistics, info
        """
        stats = await self.vector_service.get_stats()

        embed = discord.Embed(
            title="Vector Database Stats",
            color=0x0099ff
        )

        embed.add_field(name="Total Vectors", value=stats['total_vectors'], inline=True)
        embed.add_field(name="Dimension", value=stats['dimension'], inline=True)
        embed.add_field(name="Index Fullness", value=f"{stats['index_fullness']:.2%}", inline=True)

        await ctx.send(embed=embed)

    @commands.command(name='debug_memories', aliases=['dm', 'debug_mem', 'checkmem'])
    async def debug_memories(self, ctx):
        """
        Debug memory storage and retrieval for troubleshooting issues.
        
        Provides detailed diagnostic information about memory storage,
        search functionality, and database connectivity. Useful for
        troubleshooting when memories aren't saving or retrieving correctly.
        
        Diagnostic Checks:
            - Vector database connectivity
            - Total vector count across all channels
            - Channel-specific memory search
            - Sample memory retrieval
            - Search functionality testing
            
        Debug Information Displayed:
            - Total vectors in database
            - Current Discord channel ID
            - Number of memories found in current channel
            - Sample search results with scores
            - Memory metadata and types
            
        Search Test:
            Performs a test search using common keywords:
            "project React Node PostgreSQL JWT API"
            This helps verify that:
            - Search functionality is working
            - Memories are properly indexed
            - Relevance scoring is functioning
            
        Troubleshooting Use Cases:
            - Memories not appearing in !list_memories
            - !recall command returning no results
            - Vector database connection issues
            - Channel isolation problems
            
        Common Issues Diagnosed:
            - Empty vector database
            - Channel ID mismatches
            - Indexing failures
            - Search query problems
            - Metadata corruption
            
        Debug Output Includes:
            - Vector database statistics
            - Channel-specific information
            - Search result samples
            - Memory type distribution
            - Relevance scores for validation
            
        Example:
            !debug_memories
            !dm
            !checkmem
            
        Note:
            This command is primarily for troubleshooting and may show
            technical information that's not needed for normal usage.
            
        Aliases: dm, debug_mem, checkmem
        """
        # First check stats
        stats = await self.vector_service.get_stats()

        # Try to search for recent memories
        results = await self.vector_service.search_similar(
            query="project React Node PostgreSQL JWT API",  # Keywords from your test memories
            channel_id=str(ctx.channel.id),
            content_type=['memory'],
            top_k=10
        )

        embed = discord.Embed(
            title="Memory Debug Info",
            color=0xffff00
        )

        embed.add_field(name="Total Vectors", value=stats['total_vectors'], inline=True)
        embed.add_field(name="Channel ID", value=str(ctx.channel.id), inline=True)
        embed.add_field(name="Search Results", value=f"{len(results)} found", inline=True)

        if results:
            for i, result in enumerate(results[:3]):
                metadata = result['metadata']
                embed.add_field(
                    name=f"Result {i + 1}: {metadata.get('tag', 'N/A')}",
                    value=f"Type: {metadata.get('type', 'N/A')}\nScore: {result['score']:.2f}",
                    inline=False
                )

        await ctx.send(embed=embed)


async def setup(bot):
    """
    Set up the MemoryCog for the Discord bot.
    
    This function is called by the Discord.py framework to initialize
    and register the MemoryCog with the bot instance.
    
    Args:
        bot (commands.Bot): The Discord bot instance
    """
    await bot.add_cog(MemoryCog(bot))
