# Documentation TODO List

This document tracks the documentation needs for the PairProgrammer Discord bot project.

## Priority Order for Documentation

### 1. User Documentation
- [X] **README.md** - Main project documentation for end users with command reference

### 2. Core Application Files
- [X] **main.py** - Bot entry point and initialization
- [X] **config.py** - Configuration management

### 3. Database Layer
- [X] **database/models.py** - Database models and initialization

### 4. Services Layer
- [ ] **services/ai_service.py** - AI model integration service
- [ ] **services/vector_service.py** - Vector database operations
- [ ] **services/github_service.py** - GitHub API integration
- [ ] **services/s3_service.py** - AWS S3 file storage service

### 5. Bot Cogs (Discord Extensions)
- [ ] **cogs/ai_chat.py** - Primary AI chat functionality
- [ ] **cogs/memory.py** - Conversation memory management
- [ ] **cogs/file_manager.py** - File upload/download operations
- [ ] **cogs/github_integration.py** - GitHub repository integration
- [X] **cogs/admin.py** - Administrative commands
- [X] **cogs/help_system.py** - Dynamic help system

### 6. Utilities
- [ ] **utils/logger.py** - Logging utility functions

### 7. Package Initialization Files
- [X] **__init__.py** (root)
- [X] **database/__init__.py**
- [X] **services/__init__.py**
- [X] **cogs/__init__.py**

## Documentation Standards

### For Python Files:
- Add module-level docstrings explaining the file's purpose
- Document all classes with class docstrings
- Document all functions/methods with docstrings including:
  - Purpose and behavior
  - Parameters (with types)
  - Return values (with types)
  - Exceptions raised
  - Usage examples where helpful

### For Discord Commands:
- Use Discord.py's built-in help parameter
- Include usage examples
- Document required permissions
- List command aliases

### For User Documentation:
- Clear setup instructions
- Environment variable configuration
- Command usage examples
- Troubleshooting guide
- Feature overview

## Notes
- Follow Google-style docstrings for consistency
- Include type hints where missing
- Add inline comments for complex logic
- Ensure all public APIs are documented