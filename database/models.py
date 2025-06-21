from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, UTC
import os

Base = declarative_base()


class Conversation(Base):
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
    __tablename__ = 'file_metadata'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    s3_key = Column(String(500), nullable=False)
    user_id = Column(String(50), nullable=False)
    file_type = Column(String(50))
    upload_timestamp = Column(DateTime, default=datetime.now(UTC))
    description = Column(Text)


class GitHubRepo(Base):
    __tablename__ = 'github_repos'

    id = Column(Integer, primary_key=True)
    repo_name = Column(String(255), nullable=False)
    repo_url = Column(String(500), nullable=False)
    channel_id = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    added_by = Column(String(50), nullable=False)
    added_timestamp = Column(DateTime, default=datetime.now(UTC))


# Database initialization
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///bot_memory.db'))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


async def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()
