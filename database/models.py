"""
Database Models for PairProgrammer Discord Bot

This module defines the SQLAlchemy ORM models for the bot's database schema.
It handles conversation history, file metadata, and GitHub repository tracking.

The database supports both SQLite (default) and MySQL backends, configured
through environment variables.

Tables:
    conversations: Stores AI chat history with context and metadata
    file_metadata: Tracks uploaded files and their S3 storage locations
    github_repos: Manages GitHub repositories tracked per Discord channel

Usage:
    from database.models import init_db, get_db, Conversation
    
    # Initialize database
    init_db()
    
    # Get database session
    db = get_db()
    conversations = db.query(Conversation).all()
    db.close()

Author: PairProgrammer Team
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, UTC

from config import DATABASE_URL

# SQLAlchemy declarative base for model definitions
Base = declarative_base()


class Conversation(Base):
    """
    Model for storing AI conversation history.
    
    This table stores all chat interactions between users and AI models,
    including the original user message, AI response, model used, and
    context metadata for future reference and training.
    
    Attributes:
        id (int): Primary key, auto-incrementing conversation ID
        user_id (str): Discord user ID who sent the message
        channel_id (str): Discord channel ID where conversation occurred
        message (str): Original user message/question
        response (str): AI model's response
        ai_model (str): AI model used (e.g., 'openai:gpt-4', 'anthropic:claude-3')
        timestamp (datetime): When the conversation occurred (UTC)
        context_tags (str): JSON string of context tags for categorization
        
    Indexes:
        - Primary key on id
        - Consider adding indexes on user_id, channel_id, timestamp for queries
    """
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    channel_id = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text)
    ai_model = Column(String(50))
    timestamp = Column(DateTime, default=datetime.now(UTC))
    context_tags = Column(Text)  # JSON string of tags


class FileMetadata(Base):
    """
    Model for tracking uploaded file metadata and S3 storage information.
    
    This table maintains a registry of all files uploaded through the bot,
    including their S3 storage locations, file types, and user-provided
    descriptions for document search and retrieval.
    
    Attributes:
        id (int): Primary key, auto-incrementing file ID
        filename (str): Original filename as uploaded by user
        s3_key (str): S3 object key for file storage location
        user_id (str): Discord user ID who uploaded the file
        file_type (str): File extension/type (pdf, txt, py, etc.)
        upload_timestamp (datetime): When the file was uploaded (UTC)
        description (str): User-provided description of the file contents
        
    Supported File Types:
        Documents: pdf, txt, md
        Code: py, js, ts, json, yaml, yml
        Data: csv, log
        
    Indexes:
        - Primary key on id
        - Consider adding indexes on user_id, file_type for filtering
    """
    __tablename__ = 'file_metadata'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    s3_key = Column(String(500), nullable=False)
    user_id = Column(String(50), nullable=False)
    file_type = Column(String(50))
    upload_timestamp = Column(DateTime, default=datetime.now(UTC))
    description = Column(Text)


class GitHubRepo(Base):
    """
    Model for tracking GitHub repositories monitored per Discord channel.
    
    This table manages which GitHub repositories are being tracked in each
    Discord channel, enabling repository-specific commands like issue tracking,
    code search, and pull request monitoring.
    
    Attributes:
        id (int): Primary key, auto-incrementing repository tracking ID
        repo_name (str): Repository name in format 'owner/repo'
        repo_url (str): Full GitHub repository URL
        channel_id (str): Discord channel ID where repo is tracked
        is_active (bool): Whether tracking is currently active
        added_by (str): Discord user ID who added the repository
        added_timestamp (datetime): When repository tracking was added (UTC)
        
    Business Rules:
        - One repository can be tracked in multiple channels
        - Each channel can track multiple repositories
        - Only active repositories appear in channel commands
        - Users can only remove repositories they added (unless admin)
        
    Indexes:
        - Primary key on id
        - Consider compound index on (channel_id, is_active) for queries
        - Consider index on repo_name for cross-channel lookups
    """
    __tablename__ = 'github_repos'

    id = Column(Integer, primary_key=True)
    repo_name = Column(String(255), nullable=False)
    repo_url = Column(String(500), nullable=False)
    channel_id = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    added_by = Column(String(50), nullable=False)
    added_timestamp = Column(DateTime, default=datetime.now(UTC))


# =============================================================================
# Database Engine and Session Configuration
# =============================================================================

# Create SQLAlchemy engine from configured database URL
engine = create_engine(DATABASE_URL)

# Session factory for creating database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Initialize the database by creating all tables.
    
    This function creates all tables defined by the SQLAlchemy models
    if they don't already exist. It's safe to call multiple times as
    it only creates missing tables.
    
    Should be called once at application startup to ensure the database
    schema is properly initialized.
    
    Raises:
        sqlalchemy.exc.SQLAlchemyError: If database connection fails
        sqlalchemy.exc.OperationalError: If table creation fails
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Get a database session for performing database operations.
    
    Returns a new SQLAlchemy session that should be used for database
    queries and modifications. The caller is responsible for closing
    the session when done.
    
    Returns:
        sqlalchemy.orm.Session: Database session for queries
        
    Usage:
        db = get_db()
        try:
            # Perform database operations
            conversations = db.query(Conversation).all()
            # ... other operations
        finally:
            db.close()
    """
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let the caller close
