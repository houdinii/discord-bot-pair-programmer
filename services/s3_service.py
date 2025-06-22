import os
import uuid
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError


class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            config=Config(signature_version='s3v4')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    async def upload_file(self, file_path: str, file_name: str, user_id: str) -> Optional[str]:
        """Upload the file to S3 and return the S3 key"""
        try:
            # Generate unique S3 key
            file_extension = os.path.splitext(file_name)[1]
            s3_key = f"uploads/{user_id}/{uuid.uuid4()}{file_extension}"

            # Upload the file
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)

            return s3_key
        except NoCredentialsError:
            print("AWS credentials not available")
            return None
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download the file from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    async def delete_file(self, s3_key: str) -> bool:
        """Delete the file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """Generate a presigned URL for file access"""
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
