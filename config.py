"""
Configuration Management for PairProgrammer Discord Bot

This module handles all configuration settings for the PairProgrammer Discord bot,
loading values from environment variables with appropriate defaults. It manages
database connections, API keys, and service configurations.

Environment Variables:
    Core Bot Settings:
        DISCORD_TOKEN (str): Discord bot token (required)
        DEBUG (str): Enable debug mode ('true'/'false', default: 'false')
    
    Database Configuration:
        DB_TYPE (str): Database type ('sqlite' or 'mysql', default: 'sqlite')
        DATABASE_URL (str): SQLite database path (default: 'sqlite:///bot_memory.db')
        
        For MySQL:
            DB_USERNAME (str): MySQL username
            DB_PASSWORD (str): MySQL password  
            DB_HOST (str): MySQL host
            DB_PORT (str): MySQL port (default: '3306')
            DB_NAME (str): MySQL database name
    
    AI Service API Keys:
        OPENAI_API_KEY (str): OpenAI API key for GPT models
        ANTHROPIC_API_KEY (str): Anthropic API key for Claude models
        PINECONE_API_KEY (str): Pinecone vector database API key
        PINECONE_INDEX_NAME (str): Pinecone index name (default: 'discord-pair-programmer')
    
    External Services:
        GITHUB_TOKEN (str): GitHub personal access token
        AWS_ACCESS_KEY_ID (str): AWS access key for S3 storage
        AWS_SECRET_ACCESS_KEY (str): AWS secret key for S3 storage
        S3_BUCKET_NAME (str): S3 bucket name for file storage

Usage:
    from config import DISCORD_TOKEN, DATABASE_URL, OPENAI_API_KEY
    
Author: PairProgrammer Team
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Database Configuration
# =============================================================================

DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # Database type: 'sqlite' or 'mysql'

if DB_TYPE == 'mysql':
    # MySQL database configuration
    DB_USERNAME = os.getenv('DB_USERNAME')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_NAME = os.getenv('DB_NAME')
    DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    # SQLite fallback configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///bot_memory.db')

# =============================================================================
# Discord Bot Configuration
# =============================================================================

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')  # Discord bot token (required)
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'  # Enable debug logging

# =============================================================================
# AI Service API Keys
# =============================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # OpenAI API key for GPT models
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')  # Anthropic API key for Claude models
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Pinecone vector database API key
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'discord-pair-programmer')  # Pinecone index name
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # GitHub personal access token

# =============================================================================
# AWS S3 Configuration for File Storage
# =============================================================================

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')  # AWS access key ID
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')  # AWS secret access key
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')  # S3 bucket name for file storage

# =============================================================================
# Arxiv Configuration
# =============================================================================

ARXIV_SUGGESTION_CHANNEL = os.getenv('ARXIV_SUGGESTION_CHANNEL')
