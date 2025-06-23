# PairProgrammer Discord Bot

An AI-powered Discord bot designed to assist with programming tasks, document management, and project collaboration. The bot integrates multiple AI models, provides intelligent context retrieval, and maintains conversation memory to create a seamless development assistance experience.

## ✨ Features

- **🤖 Multi-AI Model Support**: Chat with OpenAI GPT-4o/o1 series and Anthropic Claude 4.0 models
- **🧠 Intelligent Context**: Semantic search through conversation history using vector databases
- **📄 Document Management**: Upload, analyze, and query documents (PDF, code files, etc.)
- **🐙 GitHub Integration**: Track repositories, search code, manage issues and pull requests
- **💾 Memory System**: Remember and recall important project context
- **🔍 Smart Search**: Find relevant information across conversations, documents, and memories

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Discord Bot Token
- API Keys for OpenAI and/or Anthropic
- Pinecone API Key (for vector database)
- AWS S3 Bucket (for file storage)
- GitHub Token (for repository integration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PairProgrammer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
# Required
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=discord-pair-programmer
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your_s3_bucket_name
GITHUB_TOKEN=your_github_token

# Optional
DEBUG=false
DB_TYPE=sqlite
DATABASE_URL=sqlite:///bot_memory.db
```

4. Run the bot:
```bash
python main.py
```

## 📖 Command Reference

All commands use the `!` prefix. Most commands have multiple aliases for convenience.

### 🤖 AI & Chat Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `!chat` | `c`, `ask`, `ai` | Chat with AI models with context | `!chat openai chatgpt-4o-latest How do I implement async?` |
| `!quick` | `q`, `gpt`, `ask4` | Quick chat with default GPT-4 model | `!quick How do I use decorators?` |
| `!models` | `model`, `list_models` | List available AI models | `!models` |
| `!search` | `s`, `find`, `lookup` | Search conversation history | `!search authentication` |
| `!context` | `ctx`, `preview` | Show context for a query | `!context recent conversations` |
| `!autosave` | `as`, `auto` | Toggle auto-saving conversations | `!autosave` |

### 📄 Document Management

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `!upload` | `up`, `add`, `attach` | Upload files with description | `!upload "API docs"` + attach file |
| `!files` | `f`, `list`, `ls`, `docs` | List uploaded files | `!files` or `!files @user` |
| `!askdoc` | `ad`, `question`, `qd` | Ask questions about documents | `!askdoc 123 What is the main idea?` |
| `!fileinfo` | `fi`, `details` | Get file information | `!fileinfo 123` |
| `!getfile` | `get`, `download` | Get download link | `!getfile 123` |
| `!papers` | `p`, `documents`, `pdfs` | List documents with filters | `!papers pdf` |
| `!deletefile` | `delete`, `rm` | Delete files (owner only) | `!deletefile 123` |

### 🧠 Memory System

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `!remember` | `r`, `save`, `mem` | Save important context | `!remember auth_setup Using JWT for authentication` |
| `!recall` | `rc`, `find_memory` | Search memories | `!recall authentication` |
| `!get_memory` | `gm`, `memory` | Get specific memory by tag | `!get_memory auth_setup` |
| `!list_memories` | `lm`, `memories` | List all memory tags | `!list_memories` |
| `!forget` | `fg`, `delete_memory` | Delete memories | `!forget old_project` |
| `!stats` | `st`, `statistics` | Show database statistics | `!stats` |

### 🐙 GitHub Integration

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `!address` | `add_repo`, `track` | Track GitHub repository | `!address user/repo` |
| `!repos` | `repositories`, `lr` | List tracked repositories | `!repos` |
| `!repoinfo` | `repo`, `about` | Get repository information | `!repoinfo user/repo` |
| `!issues` | `i`, `bugs` | List repository issues | `!issues user/repo` |
| `!prs` | `pr`, `pulls` | List pull requests | `!prs user/repo` |
| `!codesearch` | `cs`, `code`, `grep` | Search code in repository | `!codesearch user/repo function_name` |
| `!createissue` | `ci`, `newissue` | Create new issue | `!createissue user/repo "Title" Description` |
| `!removerepo` | `rr`, `untrack` | Stop tracking repository | `!removerepo user/repo` |

### 📚 Help & Information

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `!help` | `h`, `?`, `commands` | Show command help | `!help` or `!help ai` |
| `!commands_table` | `ct`, `table` | Compact command table | `!commands_table` |
| `!tips` | `tip`, `hints` | Show usage tips | `!tips` |
| `!aliases` | `a`, `shortcuts` | Show command aliases | `!aliases` |

## 🎯 Usage Examples

### Basic AI Chat
```
!quick How do I implement binary search in Python?
!chat anthropic claude-sonnet-4-0 Explain async/await in JavaScript
```

### Document Analysis
```
!upload "Project requirements document"
# Attach your PDF file
!askdoc 1 What are the main requirements?
!files  # List all uploaded documents
```

### Memory Management
```
!remember project_stack We're using React frontend with Node.js backend
!remember db_choice Decided on PostgreSQL for the database
!recall database  # Find all memories about database choices
```

### GitHub Integration
```
!address microsoft/vscode  # Track the VS Code repository
!issues microsoft/vscode   # See recent issues
!codesearch microsoft/vscode editor.action  # Search for code
```

## 🔧 Configuration

### Database Options

**SQLite (Default)**:
```env
DB_TYPE=sqlite
DATABASE_URL=sqlite:///bot_memory.db
```

**MySQL**:
```env
DB_TYPE=mysql
DB_USERNAME=username
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=discord_bot
```

### Supported File Types

The bot can process these file types:
- **Documents**: PDF, TXT, MD
- **Code**: PY, JS, TS, JSON, YAML, YML
- **Data**: CSV, LOG
- **Max file size**: 25MB (Discord limit)

### AI Models Available

**OpenAI Models**:
- gpt-4o, chatgpt-4o-latest, gpt-4o-mini
- o1, o1-preview, o1-mini, o3-mini
- gpt-4, gpt-4-turbo, gpt-3.5-turbo

**Anthropic Models**:
- claude-opus-4-0, claude-sonnet-4-0
- claude-3-5-sonnet-latest, claude-3-5-haiku-latest
- claude-3-opus-latest

## 🛠️ Architecture

The bot follows a modular architecture with these components:

- **`main.py`**: Bot initialization and entry point
- **`config.py`**: Configuration management
- **`database/`**: SQLAlchemy models and database operations
- **`services/`**: AI, vector database, GitHub, and S3 services
- **`cogs/`**: Discord command handlers organized by feature
- **`utils/`**: Logging and utility functions

## 🔒 Permissions

The bot requires these Discord permissions:
- Send Messages
- Read Message History
- Use Slash Commands
- Attach Files
- Embed Links
- Read Messages

Admin commands require Discord Administrator permissions.

## 🚨 Rate Limits

Most commands have cooldowns (3 uses per 60 seconds per user) to prevent spam and manage API costs. The bot includes automatic retry logic for API rate limits.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🐛 Troubleshooting

### Common Issues

**Bot not responding**:
- Check Discord token validity
- Verify bot has required permissions
- Check console logs for errors

**AI commands failing**:
- Verify API keys in `.env` file
- Check rate limits on AI providers
- Ensure Pinecone index exists

**File uploads not working**:
- Verify AWS S3 credentials and bucket access
- Check file size (Discord 25MB limit)
- Ensure supported file type

**GitHub integration issues**:
- Verify GitHub token permissions
- Check repository URL format
- Ensure repository is public or token has access

### Getting Help

Use the `!help` command for in-bot assistance, or `!tips` for best practices and usage tips.

---

**Note**: This bot is designed for development teams and programming assistance. Always review AI-generated code before using in production.