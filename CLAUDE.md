# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Discord bot called "PairProgrammer" built with Python that serves as an AI-powered programming assistant. The bot integrates multiple AI models (OpenAI and Anthropic), provides file management capabilities, GitHub integration, and maintains conversation memory using vector databases.

## Development Commands

### Running the Bot
```bash
python main.py
```

### Database Management
The bot uses SQLAlchemy with automatic database initialization. Database tables are created automatically on startup through `database/models.py:init_db()`.

### Dependencies
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

**Main Entry Point**: `main.py`
- Initializes Discord bot with `!` command prefix
- Loads all cogs (extensions) dynamically
- Handles global error logging and command tracking

**Configuration**: `config.py`
- Manages environment variables and API keys
- Supports both SQLite (default) and MySQL databases
- Configures OpenAI, Anthropic, Pinecone, GitHub, and AWS S3 credentials

**Database Models**: `database/models.py`
- `Conversation`: Stores chat history with AI models
- `FileMetadata`: Tracks uploaded files and S3 storage
- `GitHubRepo`: Manages connected GitHub repositories per channel

### Cogs (Discord Extensions)

**AI Chat** (`cogs/ai_chat.py`):
- Primary interface for AI conversations
- Supports multiple AI providers (OpenAI, Anthropic)
- Handles message chunking for Discord's 2000 character limit
- Integrates with vector service for context retrieval

**Memory** (`cogs/memory.py`):
- Manages conversation persistence
- Vector database integration for semantic search
- Context retrieval from past conversations

**File Manager** (`cogs/file_manager.py`):
- File upload/download to S3
- File metadata tracking
- Document processing and analysis

**GitHub Integration** (`cogs/github_integration.py`):
- Repository connection and management
- Code analysis and interaction

**Admin** (`cogs/admin.py`):
- Bot administration commands
- Database management utilities

**Help System** (`cogs/help_system.py`):
- Dynamic command help generation
- Bot feature documentation

### Services

**AI Service** (`services/ai_service.py`):
- Abstracts AI model interactions
- Supports OpenAI (GPT-4o, o1 series) and Anthropic (Claude 4.0 series)
- Handles model selection and parameter configuration

**Vector Service** (`services/vector_service.py`):
- Pinecone integration for semantic search
- Document embedding and retrieval
- Context-aware conversation enhancement

**GitHub Service** (`services/github_service.py`):
- GitHub API integration
- Repository analysis and code interaction

**S3 Service** (`services/s3_service.py`):
- AWS S3 file storage
- File upload/download management

### Utilities

**Logger** (`utils/logger.py`):
- Structured logging with data tracking
- Method decoration for automatic logging
- User and channel context preservation

## Environment Variables

Required environment variables (configure in `.env`):

```
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=discord-pair-programmer
GITHUB_TOKEN=your_github_token
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your_s3_bucket_name

# Optional database configuration (defaults to SQLite)
DB_TYPE=sqlite
DATABASE_URL=sqlite:///bot_memory.db

# For MySQL:
# DB_TYPE=mysql
# DB_USERNAME=username
# DB_PASSWORD=password
# DB_HOST=localhost
# DB_PORT=3306
# DB_NAME=discord_bot

DEBUG=false
```

## Database Schema

The bot automatically creates these tables:
- `conversations`: AI chat history with context tags
- `file_metadata`: File upload tracking and S3 references
- `github_repos`: Per-channel GitHub repository connections

## Key Libraries

- `discord.py`: Discord bot framework
- `langchain`: AI model abstraction layer
- `sqlalchemy`: Database ORM
- `pinecone`: Vector database for semantic search
- `boto3`: AWS S3 integration
- `PyGithub`: GitHub API client

## Deployment

The bot is configured for PebbleHost deployment with:
- `pebble-python-config.json`: Specifies Python 3.12 and entry point
- `pebblehost.yml`: Reverse proxy and Git management configuration