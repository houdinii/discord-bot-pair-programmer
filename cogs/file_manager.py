# cogs/file_manager.py
import discord
from discord.ext import commands
from services.s3_service import S3Service
from services.vector_service import VectorService
from database.models import FileMetadata, get_db
import os
import tempfile
import PyPDF2
from datetime import datetime, UTC
from typing import List, Optional
import mimetypes

from sqlalchemy import func
from utils.logger import get_logger

logger = get_logger(__name__)


class FileManagerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.s3_service = S3Service()
        self.vector_service = VectorService()

        # Supported file types for text extraction
        self.text_extractors = {
            '.pdf': self._extract_pdf_text,
            '.txt': self._extract_text_file,
            '.md': self._extract_text_file,
            '.py': self._extract_text_file,
            '.js': self._extract_text_file,
            '.json': self._extract_text_file,
            '.yml': self._extract_text_file,
            '.yaml': self._extract_text_file,
            '.csv': self._extract_text_file,
            '.log': self._extract_text_file,
        }

    async def _extract_file_content(self, file_path: str, filename: str) -> Optional[str]:
        """Extract text content from various file types"""
        ext = os.path.splitext(filename)[1].lower()

        if ext in self.text_extractors:
            try:
                return await self.text_extractors[ext](file_path)
            except Exception as e:
                logger.logger.error(f"Error extracting content from {filename}: {e}")
                return None
        else:
            logger.logger.info(f"No text extractor for file type: {ext}")
            return None

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.logger.error(f"Error extracting PDF text: {e}")
            raise

    async def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()

    async def _index_file_content(self, file_id: int, filename: str, content: str,
                                  user_id: str, channel_id: str, description: str) -> List[str]:
        """Chunk and index file content in vector database"""
        vector_ids = []

        # Chunk the content
        chunks = self._chunk_text(content, chunk_size=1500)

        logger.logger.info(f"Indexing {len(chunks)} chunks for file {filename} (ID: {file_id})")

        for i, chunk in enumerate(chunks):
            # Create metadata for this chunk
            chunk_metadata = {
                "file_id": str(file_id),  # Ensure it's a string
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "description": description,
                "file_type": os.path.splitext(filename)[1].lower()
            }

            # Store in vector database
            vector_id = await self.vector_service.store_document_chunk(
                filename=filename,
                content=chunk,
                user_id=user_id,
                channel_id=channel_id,
                metadata=chunk_metadata
            )
            vector_ids.append(vector_id)

            logger.logger.debug(f"Stored chunk {i + 1}/{len(chunks)} with ID: {vector_id}")

        return vector_ids

    def _chunk_text(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Split text into chunks while preserving sentence boundaries"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        sentences = text.replace('\n\n', '\n<PARAGRAPH>\n').split('. ')
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.replace('\n<PARAGRAPH>\n', '\n\n')
            if len(current_chunk) + len(sentence) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (". " if current_chunk else "") + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @commands.command(name='upload', aliases=['up', 'add', 'attach'])
    async def upload_file(self, ctx, *, description: str = "No description"):
        """
        Upload a file with optional description
        Usage: !upload "API documentation for user service"
        Then attach a file to your message
        """
        if not ctx.message.attachments:
            await ctx.send("❌ Please attach a file to upload")
            return

        attachment = ctx.message.attachments[0]

        # Send initial status message
        status_msg = await ctx.send(f"📤 Uploading {attachment.filename}...")

        # Download file temporarily
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, attachment.filename)

        try:
            await attachment.save(temp_file_path)

            # Upload to S3
            s3_key = await self.s3_service.upload_file(
                temp_file_path,
                attachment.filename,
                str(ctx.author.id)
            )

            if not s3_key:
                await status_msg.edit(content="❌ Failed to upload file to storage")
                return

            # Save metadata to database
            db = get_db()
            file_metadata = FileMetadata(
                filename=attachment.filename,
                s3_key=s3_key,
                user_id=str(ctx.author.id),
                file_type=attachment.content_type or mimetypes.guess_type(attachment.filename)[0],
                description=description,
                upload_timestamp=datetime.now(UTC)
            )

            db.add(file_metadata)
            db.commit()
            file_id = file_metadata.id

            # Extract and index content
            await status_msg.edit(content=f"✅ File uploaded\n📊 Extracting and indexing content...")

            content = await self._extract_file_content(temp_file_path, attachment.filename)
            vector_ids = []
            if content:
                # Chunk and store in vector database
                vector_ids = await self._index_file_content(
                    file_id=file_id,
                    filename=attachment.filename,
                    content=content,
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    description=description
                )

                logger.logger.info(f"Indexed file {attachment.filename} with {len(vector_ids)} chunks")

            db.close()

            # Delete status message
            await status_msg.delete()

            # Create success embed
            embed = discord.Embed(
                title="File Uploaded Successfully",
                color=0x00ff00
            )
            embed.add_field(name="Filename", value=attachment.filename, inline=True)
            embed.add_field(name="File ID", value=str(file_id), inline=True)
            embed.add_field(name="Size", value=f"{attachment.size / 1024:.1f} KB", inline=True)
            embed.add_field(name="Description", value=description, inline=False)

            if content:
                embed.add_field(
                    name="Content Indexed",
                    value=f"✅ {len(vector_ids)} chunks indexed for search",
                    inline=False
                )
                embed.add_field(
                    name="Content Preview",
                    value=content[:300] + "..." if len(content) > 300 else content,
                    inline=False
                )

            await ctx.send(embed=embed)

        except Exception as e:
            await status_msg.edit(content=f"❌ Error uploading file: {str(e)}")
            logger.logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @commands.command(name='files', aliases=['f', 'list', 'ls', 'docs'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_files(self, ctx, user: discord.Member = None):
        """
        List uploaded files
        Usage: !files
        Usage: !files @user
        """
        target_user = user or ctx.author
        target_user_id = str(target_user.id)

        db = get_db()
        files = db.query(FileMetadata).filter(
            FileMetadata.user_id == target_user_id
        ).order_by(FileMetadata.upload_timestamp.desc()).limit(20).all()
        db.close()

        if not files:
            await ctx.send(f"❌ No files found for {target_user.mention}")
            return

        embed = discord.Embed(
            title=f"Files uploaded by {target_user.display_name} ({len(files)} recent)",
            color=0x0099ff
        )

        for file in files[:10]:  # Show max 10 in embed
            embed.add_field(
                name=f"ID: {file.id} - {file.filename}",
                value=f"{file.description[:50]}...\n*Uploaded: {file.upload_timestamp.strftime('%Y-%m-%d %H:%M')}*",
                inline=False
            )

        if len(files) > 10:
            embed.add_field(
                name="Note",
                value=f"Showing 10 of {len(files)} files. Use !searchfiles for more options.",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='searchfiles', aliases=['sf', 'searchdocs', 'finddoc', 'fd'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def search_files(self, ctx, *, query: str):
        """
        Search files by content or metadata
        Usage: !searchfiles authentication API
        """
        # Search in vector database
        results = await self.vector_service.search_similar(
            query=query,
            channel_id=str(ctx.channel.id),
            content_type=['document'],
            top_k=10
        )

        if not results:
            await ctx.send(f"❌ No files found matching: {query}")
            return

        # Get unique file IDs
        file_ids = set()
        file_results = {}

        for result in results:
            metadata = result['metadata']
            file_id = metadata.get('file_id')
            if file_id and file_id not in file_results:
                file_results[file_id] = {
                    'filename': metadata.get('filename'),
                    'score': result['score'],
                    'preview': result['content'][:200] + "...",
                    'chunk_info': f"Chunk {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                }
                file_ids.add(file_id)

        # Get file metadata from database
        db = get_db()
        files = db.query(FileMetadata).filter(FileMetadata.id.in_(file_ids)).all()
        file_dict = {f.id: f for f in files}
        db.close()

        embed = discord.Embed(
            title=f"File Search Results: {query}",
            description=f"Found {len(file_results)} files",
            color=0x0099ff
        )

        for file_id, result in sorted(file_results.items(), key=lambda x: x[1]['score'], reverse=True)[:5]:
            file_meta = file_dict.get(int(file_id))
            if file_meta:
                embed.add_field(
                    name=f"{result['filename']} (ID: {file_id}, Score: {result['score']:.2f})",
                    value=f"**Description:** {file_meta.description[:50]}...\n"
                          f"**Match:** {result['preview']}\n"
                          f"*{result['chunk_info']}*",
                    inline=False
                )

        await ctx.send(embed=embed)

    @commands.command(name='fileinfo', aliases=['fi', 'info', 'details'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def file_info(self, ctx, file_id: int):
        """
        Get detailed information about a file
        Usage: !fileinfo 123
        """
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found")
            return

        # Get vector database info
        search_results = await self.vector_service.search_similar(
            query=file_metadata.filename,
            channel_id=str(ctx.channel.id),
            content_type=['document'],
            top_k=50
        )

        # Count chunks for this file
        chunk_count = sum(1 for r in search_results
                          if r['metadata'].get('file_id') == str(file_id))

        embed = discord.Embed(
            title=f"File Information: {file_metadata.filename}",
            color=0x0099ff
        )

        embed.add_field(name="File ID", value=str(file_id), inline=True)
        embed.add_field(name="Uploaded By", value=f"<@{file_metadata.user_id}>", inline=True)
        embed.add_field(name="Upload Date", value=file_metadata.upload_timestamp.strftime('%Y-%m-%d %H:%M'),
                        inline=True)
        embed.add_field(name="File Type", value=file_metadata.file_type or "Unknown", inline=True)
        embed.add_field(name="Indexed Chunks", value=str(chunk_count), inline=True)
        embed.add_field(name="Description", value=file_metadata.description, inline=False)

        # Add download button
        download_url = self.s3_service.generate_presigned_url(file_metadata.s3_key)
        if download_url:
            embed.add_field(
                name="Download",
                value=f"[Click here to download]({download_url})\n*Link expires in 1 hour*",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='getfile', aliases=['get', 'download', 'dl'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def get_file(self, ctx, file_id: int):
        """
        Get a download link for a file
        Usage: !getfile 123
        """
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found")
            return

        # Generate presigned URL
        download_url = self.s3_service.generate_presigned_url(file_metadata.s3_key)

        if not download_url:
            await ctx.send("❌ Failed to generate download link")
            return

        embed = discord.Embed(
            title=f"Download: {file_metadata.filename}",
            description=file_metadata.description,
            color=0x00ff00
        )
        embed.add_field(name="Download Link", value=f"[Click here]({download_url})", inline=False)
        embed.add_field(name="Expires", value="1 hour", inline=True)

        await ctx.send(embed=embed)

    @commands.command(name='deletefile', aliases=['delete', 'rm', 'remove'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def delete_file(self, ctx, file_id: int):
        """
        Delete a file (only file owner can delete)
        Usage: !deletefile 123
        """
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id,
            FileMetadata.user_id == str(ctx.author.id)
        ).first()

        if not file_metadata:
            await ctx.send("❌ File not found or you don't have permission to delete it")
            db.close()
            return

        # Delete from S3
        success = await self.s3_service.delete_file(file_metadata.s3_key)

        if success:
            # Delete from vector database
            await self.vector_service.delete_document_chunks(
                file_id=str(file_id),
                channel_id=str(ctx.channel.id)
            )

            # Delete from database
            db.delete(file_metadata)
            db.commit()
            await ctx.send(f"✅ Deleted file: {file_metadata.filename}")
        else:
            await ctx.send("❌ Failed to delete file from storage")

        db.close()

    @commands.command(name='papers', aliases=['p', 'documents', 'pdfs'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_papers(self, ctx, filter_type: str = "all"):
        """
        List papers/documents with filtering
        Usage: !papers all
        Usage: !papers pdf
        Usage: !papers recent
        """
        db = get_db()

        query = db.query(FileMetadata)

        if filter_type == "pdf":
            query = query.filter(FileMetadata.filename.like('%.pdf'))
        elif filter_type == "recent":
            # Last 7 days
            from datetime import timedelta
            week_ago = datetime.now(UTC) - timedelta(days=7)
            query = query.filter(FileMetadata.upload_timestamp >= week_ago)

        files = query.order_by(FileMetadata.upload_timestamp.desc()).limit(20).all()
        db.close()

        if not files:
            await ctx.send(f"❌ No files found for filter: {filter_type}")
            return

        embed = discord.Embed(
            title=f"Papers/Documents ({filter_type})",
            description=f"Found {len(files)} files",
            color=0x0099ff
        )

        for file in files[:10]:
            file_type = "📄" if file.filename.endswith('.pdf') else "📝"
            embed.add_field(
                name=f"{file_type} {file.filename} (ID: {file.id})",
                value=f"{file.description[:60]}...\n*By <@{file.user_id}> on {file.upload_timestamp.strftime('%Y-%m-%d')}*",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='reindex', aliases=['ri', 'refresh', 'rescan'])
    @commands.has_permissions(administrator=True)
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def reindex_file(self, ctx, file_id: int):
        """
        Re-index a file's content (admin only)
        Usage: !reindex 123
        """
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found")
            return

        status_msg = await ctx.send(f"🔄 Re-indexing {file_metadata.filename}...")
        temp_file_path = None

        try:
            # Download file from S3
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, file_metadata.filename)

            success = await self.s3_service.download_file(file_metadata.s3_key, temp_file_path)

            if not success:
                await status_msg.edit(content="❌ Failed to download file from storage")
                return

            # Delete existing chunks
            await self.vector_service.delete_document_chunks(
                file_id=str(file_id),
                channel_id=str(ctx.channel.id)
            )

            # Extract and re-index content
            content = await self._extract_file_content(temp_file_path, file_metadata.filename)

            if content:
                vector_ids = await self._index_file_content(
                    file_id=file_id,
                    filename=file_metadata.filename,
                    content=content,
                    user_id=file_metadata.user_id,
                    channel_id=str(ctx.channel.id),
                    description=file_metadata.description
                )

                await status_msg.edit(
                    content=f"✅ Re-indexed {file_metadata.filename} with {len(vector_ids)} chunks"
                )
            else:
                await status_msg.edit(content="❌ Could not extract content from file")

        except Exception as e:
            await status_msg.edit(content=f"❌ Error re-indexing file: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @commands.command(name='debugfile', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def debug_file(self, ctx, file_id: int):
        """
        Debug command to check file indexing status
        Usage: !debugfile 123
        """
        db = get_db()
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.id == file_id
        ).first()
        db.close()

        if not file_metadata:
            await ctx.send("❌ File not found in database")
            return

        # Search for chunks of this file
        results = await self.vector_service.search_documents(
            query=file_metadata.filename,
            channel_id=str(ctx.channel.id),
            file_id=str(file_id),
            top_k=50
        )

        # Also try a general search
        general_results = await self.vector_service.search_similar(
            query=file_metadata.filename,
            channel_id=str(ctx.channel.id),
            content_type=['document'],
            top_k=10
        )

        embed = discord.Embed(
            title=f"Debug Info: {file_metadata.filename}",
            color=0xffff00
        )

        embed.add_field(name="File ID", value=str(file_id), inline=True)
        embed.add_field(name="Channel ID", value=str(ctx.channel.id), inline=True)
        embed.add_field(name="Chunks Found", value=str(len(results)), inline=True)

        # Check chunk details
        if results:
            chunk_indices = sorted(set(r['metadata'].get('chunk_index', -1) for r in results))
            total_chunks = results[0]['metadata'].get('total_chunks', 'Unknown')
            embed.add_field(
                name="Chunk Details",
                value=f"Indices: {chunk_indices}\nTotal Expected: {total_chunks}",
                inline=False
            )

            # Show sample content
            sample = results[0]['content'][:200] + "..."
            embed.add_field(name="Sample Content", value=f"```{sample}```", inline=False)

        embed.add_field(
            name="General Search Results",
            value=f"{len(general_results)} results found",
            inline=False
        )

        await ctx.send(embed=embed)

    @commands.command(name='filestats', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def file_statistics(self, ctx):
        """Show file upload statistics for the channel"""
        db = get_db()

        # Get stats
        total_files = db.query(FileMetadata).count()
        # TODO: FIX TOTAL SIZE
        # total_size = db.query(func.sum(FileMetadata.file_size)).scalar() or 0
        total_size = 4096
        file_types = db.query(
            FileMetadata.file_type,
            func.count(FileMetadata.id)
        ).group_by(FileMetadata.file_type).all()

        db.close()

        embed = discord.Embed(
            title="📊 File Statistics",
            color=0x0099ff
        )

        embed.add_field(name="Total Files", value=str(total_files), inline=True)
        embed.add_field(name="Total Size", value=f"{total_size / 1024 / 1024:.1f} MB", inline=True)

        # Top file types
        type_str = "\n".join([f"{ft}: {count}" for ft, count in file_types[:5]])
        embed.add_field(name="Top File Types", value=type_str or "None", inline=False)

        await ctx.send(embed=embed)

    @commands.command(name='export')
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def export_conversations(self, ctx, days: int = 7):
        """Export recent conversations to a file"""
        # TODO: Complete export recent conversations
        # Get recent conversations from vector DB
        # Format as markdown
        # Upload to S3 and provide download link

    @commands.command(name='import_memories', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def import_memories(self, ctx):
        """Import memories from a JSON file"""
        # TODO: COMPLETE BULK IMPORT MEMORIES
        # Parse attached JSON file
        # Bulk insert memories with tags

    @commands.command(name='advanced_search', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def advanced_search(self, ctx, content_type: str, *, query: str):
        """Search with filters: !advanced_search docs authentication"""
        # TODO complete the advanced file search command
        # Search only specific content types
        # Support date ranges, authors, etc.

    @commands.command(name='backup', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    @commands.has_permissions(administrator=True)
    async def backup_channel(self, ctx):
        """Backup all channel data to a file"""
        # TODO Complete the backup channel command
        # Export all memories, conversations, file metadata
        # Create downloadable archive

    @commands.command(name='usage', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def show_usage(self, ctx, days: int = 30):
        """Show usage statistics for the channel"""
        # TODO Complete the usage statistics command
        # Track command usage
        # Show most active users
        # Popular search queries
        # AI model preferences

    @commands.command(name='compare', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def compare_files(self, ctx, file_id1: int, file_id2: int):
        """Compare two documents for similarities and differences"""
        # TODO: Complete compare files
        # Extract both documents
        # Use AI to analyze differences
        # Highlight key distinctions

    @commands.command(name='summarize', aliases=[])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def summarize_channel(self, ctx, hours: int = 24):
        """Summarize recent channel activity"""
        # TODO: Complete the summarize entire channel logic
        # Get recent conversations
        # Generate AI summary of key topics
        # Highlight important decisions


async def setup(bot):
    await bot.add_cog(FileManagerCog(bot))
