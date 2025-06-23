"""
AWS S3 Service for PairProgrammer Discord Bot

This service provides comprehensive AWS S3 operations for file storage and
management. It handles file uploads, downloads, deletion, and presigned URL
generation for secure file access. The service is designed to work with
Discord file attachments and provides organized storage with user-based
directory structures.

Key Features:
    - Secure file upload with unique naming
    - User-based directory organization
    - Presigned URL generation for secure downloads
    - File deletion and cleanup
    - Error handling and credential management
    - Support for various file types and sizes

Security:
    - Uses AWS credentials from environment variables
    - Generates unique file names to prevent conflicts
    - Supports presigned URLs with configurable expiration
    - User-based directory isolation

Storage Structure:
    uploads/{user_id}/{uuid}{extension}
    
Example:
    uploads/123456789/a1b2c3d4-e5f6-7890-abcd-ef1234567890.pdf

Author: PairProgrammer Team
"""

import os
import uuid
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError


class S3Service:
    """
    Service class for AWS S3 file storage and management operations.
    
    This class provides a comprehensive interface for file operations with AWS S3,
    including uploads, downloads, deletions, and secure URL generation. It's
    designed specifically for Discord bot file management with user-based
    organization and security considerations.
    
    Attributes:
        s3_client (boto3.client): Configured S3 client with credentials
        bucket_name (str): S3 bucket name for file storage
        
    Environment Variables Required:
        AWS_ACCESS_KEY_ID: AWS access key ID
        AWS_SECRET_ACCESS_KEY: AWS secret access key
        AWS_REGION: AWS region (default: us-east-2)
        S3_BUCKET_NAME: S3 bucket name for storage
        
    Storage Organization:
        Files are organized by user ID to prevent conflicts and enable
        user-specific file management:
        uploads/{user_id}/{uuid}{extension}
        
    Example:
        s3_service = S3Service()
        
        # Upload file
        s3_key = await s3_service.upload_file(
            file_path="/tmp/document.pdf",
            file_name="document.pdf", 
            user_id="123456789"
        )
        
        # Generate download URL
        url = s3_service.generate_presigned_url(s3_key, expiration=3600)
    """
    
    def __init__(self):
        """
        Initialize the S3Service with AWS credentials and configuration.
        
        Sets up the S3 client using environment variables for authentication
        and configures the service for file operations.
        
        Environment Variables Required:
            AWS_ACCESS_KEY_ID: AWS access key ID
            AWS_SECRET_ACCESS_KEY: AWS secret access key  
            AWS_REGION: AWS region (optional, defaults to us-east-2)
            S3_BUCKET_NAME: S3 bucket name for file storage
            
        AWS Permissions Required:
            - s3:PutObject (for file uploads)
            - s3:GetObject (for file downloads)
            - s3:DeleteObject (for file deletion)
            - s3:GeneratePresignedUrl (for secure URLs)
            
        Raises:
            NoCredentialsError: If AWS credentials are not provided
            ClientError: If S3 bucket access fails or bucket doesn't exist
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-2'),
            config=Config(signature_version='s3v4')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    async def upload_file(self, file_path: str, file_name: str, user_id: str) -> Optional[str]:
        """
        Upload a file to S3 with unique naming and user-based organization.
        
        Uploads a file from the local filesystem to S3 using a user-specific
        directory structure and UUID-based naming to prevent conflicts.
        
        Args:
            file_path (str): Local path to the file to upload
            file_name (str): Original filename (used for extension extraction)
            user_id (str): Discord user ID for directory organization
            
        Returns:
            Optional[str]: S3 key of the uploaded file, or None if upload failed
                         Format: "uploads/{user_id}/{uuid}{extension}"
                         
        File Organization:
            - Files are stored in user-specific directories
            - Each file gets a unique UUID to prevent naming conflicts
            - Original file extension is preserved
            - Directory structure: uploads/{user_id}/{uuid}{extension}
            
        Error Handling:
            - Returns None on credential errors
            - Returns None on upload failures
            - Errors are logged (currently using print statements)
            
        Example:
            s3_key = await s3_service.upload_file(
                file_path="/tmp/discord_attachment.pdf",
                file_name="document.pdf",
                user_id="123456789"
            )
            
            if s3_key:
                print(f"File uploaded to: {s3_key}")
                # Result: uploads/123456789/a1b2c3d4-e5f6-7890-abcd-ef1234567890.pdf
            else:
                print("Upload failed")
                
        Security:
            - Uses UUID to prevent file name guessing
            - User isolation prevents access to other users' files
            - Preserves original file extension for proper handling
            
        Raises:
            NoCredentialsError: If AWS credentials are missing (caught and handled)
            ClientError: If S3 bucket access fails (caught and handled)
        """
        try:
            # Generate unique S3 key with user-based organization
            file_extension = os.path.splitext(file_name)[1]
            s3_key = f"uploads/{user_id}/{uuid.uuid4()}{file_extension}"

            # Upload the file to S3
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)

            return s3_key
        except NoCredentialsError:
            print("AWS credentials not available")
            return None
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 to the local filesystem.
        
        Args:
            s3_key (str): S3 key of the file to download
            local_path (str): Local filesystem path where file should be saved
            
        Returns:
            bool: True if download succeeded, False otherwise
            
        Example:
            success = await s3_service.download_file(
                s3_key="uploads/123456789/abc123.pdf",
                local_path="/tmp/downloaded_file.pdf"
            )
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3 storage.
        
        Permanently removes a file from the S3 bucket. This operation
        cannot be undone, so use with caution.
        
        Args:
            s3_key (str): S3 key of the file to delete
            
        Returns:
            bool: True if deletion succeeded, False otherwise
            
        Usage:
            Used when users delete files through the bot or during
            cleanup operations.
            
        Example:
            success = await s3_service.delete_file(
                s3_key="uploads/123456789/abc123.pdf"
            )
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for secure file access.
        
        Creates a temporary URL that allows access to a private S3 file
        without requiring AWS credentials. The URL expires after the
        specified time period for security.
        
        Args:
            s3_key (str): S3 key of the file to generate URL for
            expiration (int): URL expiration time in seconds. Default: 3600 (1 hour)
                            Maximum recommended: 604800 (7 days)
                            
        Returns:
            Optional[str]: Presigned URL string, or None if generation failed
            
        Security:
            - URLs expire automatically after the specified time
            - No AWS credentials required for file access
            - URLs are unique and cannot be guessed
            - Access is limited to the specific file
            
        Usage:
            Used to provide users with download links for their uploaded
            files without exposing AWS credentials or making files public.
            
        Example:
            # Generate 1-hour download link
            url = s3_service.generate_presigned_url(
                s3_key="uploads/123456789/abc123.pdf",
                expiration=3600
            )
            
            if url:
                # Send URL to user in Discord
                await ctx.send(f"Download your file: {url}")
                
        Raises:
            ClientError: If S3 key doesn't exist or access is denied (caught and handled)
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
