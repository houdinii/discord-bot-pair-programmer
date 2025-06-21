import discord
from discord.ext import commands
from services.s3_service import S3Service
from database.models import FileMetadata, get_db
import os
import tempfile
import PyPDF2
from datetime import datetime, UTC


class FileManagerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.s3_service = S3Service()

    @commands.command(name='upload')
    async def upload_file(self, ctx, description: str = "No description"):
        """
        Upload a file with optional description
        Usage: !upload "API documentation for user service"
        Then attach a file to your message
        """
        if not ctx.message.attachments:
            await ctx.send("❌ Please attach a file to upload")
            return

        attachment = ctx.message.attachments[0]

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
                await ctx.send("❌ Failed to upload file to storage")
                return

            # Save metadata to database
            db = get_db()
            file_metadata = FileMetadata(
                filename=attachment.filename,
                s3_key=s3_key,
                user_id=str(ctx.author.id),
                file_type=attachment.content_type,
                description=description,
                upload_timestamp=datetime.now(UTC)
            )

            db.add(file_metadata)
            db.commit()
            file_id = file_metadata.id
            db.close()

            # Extract text content if it's a PDF
            content_preview = ""
            if attachment.filename.lower().endswith('.pdf'):
                content_preview = await self._extract_pdf_text(temp_file_path)

            embed = discord.Embed(
                title="File Uploaded Successfully",
                color=0x00ff00
            )
            embed.add_field(name="Filename", value=attachment.filename, inline=True)
            embed.add_field(name="File ID", value=str(file_id), inline=True)
            embed.add_field(name="Size", value=f"{attachment.size / 1024:.1f} KB", inline=True)
            embed.add_field(name="Description", value=description, inline=False)

            if content_preview:
                embed.add_field(
                    name="Content Preview",
                    value=content_preview[:500] + "..." if len(content_preview) > 500 else content_preview,
                    inline=False
                )

            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"❌ Error uploading file: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @commands.command(name='files')
    async def list_files(self, ctx, user_id: str = None):
        """
        List uploaded files
        Usage: !files
        Usage: !files @user (to see someone else's files)
        """
        target_user_id = user_id or str(ctx.author.id)

        db = get_db()
        files = db.query(FileMetadata).filter(
            FileMetadata.user_id == target_user_id
        ).order_by(FileMetadata.upload_timestamp.desc()).limit(20).all()
        db.close()

        if not files:
            await ctx.send("❌ No files found")
            return

        embed = discord.Embed(
            title=f"Uploaded Files ({len(files)})",
            color=0x0099ff
        )

        for file in files:
            embed.add_field(
                name=f"ID: {file.id} - {file.filename}",
                value=f"{file.description}\n*Uploaded: {file.upload_timestamp.strftime('%Y-%m-%d %H:%M')}*",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='getfile')
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

    @commands.command(name='deletefile')
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
            # Delete from database
            db.delete(file_metadata)
            db.commit()
            await ctx.send(f"✅ Deleted file: {file_metadata.filename}")
        else:
            await ctx.send("❌ Failed to delete file from storage")

        db.close()

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages[:3]:  # Only first 3 pages for preview
                    text += page.extract_text()
                return text.strip()
        except Exception as e:
            return f"Could not extract text from PDF: {str(e)}"


async def setup(bot):
    await bot.add_cog(FileManagerCog(bot))
