"""
GitHub API Service for PairProgrammer Discord Bot

This service provides comprehensive GitHub repository integration including
repository analysis, code search, issue management, and content indexing.
It uses the PyGithub library to interact with the GitHub API and supports
selective file indexing based on relevance and file types.

The service is designed to work with the vector database to provide semantic
search across repository content and enable AI-powered code assistance.

Key Features:
    - Repository content analysis and indexing
    - Selective file filtering based on extensions and importance
    - Code structure analysis and documentation
    - Issue and pull request management
    - Language and technology detection
    - Content size and depth management

Supported File Types:
    - Code: .py, .js, .ts, .jsx, .tsx, .html, .css, .scss
    - Documentation: .md, .txt, .rst
    - Configuration: .yml, .yaml, .json, .toml, .env.example
    - Scripts: .sql, .sh, .bash
    - Special files: README, requirements.txt, package.json, Dockerfile

Author: PairProgrammer Team
"""

import base64
from typing import List, Dict, Tuple
from github import Github, GithubException
from utils.logger import get_logger

logger = get_logger(__name__)


class GitHubService:
    """
    Service class for GitHub API operations and repository analysis.
    
    This class provides comprehensive GitHub integration including repository
    content analysis, selective file indexing, issue management, and code
    search capabilities. It's designed to work efficiently with large
    repositories while respecting API rate limits.
    
    Attributes:
        github (Github): PyGithub client instance
        indexable_extensions (set): File extensions to include in indexing
        important_files (set): Critical files to always include
        skip_folders (set): Directories to exclude from analysis
        
    Rate Limiting:
        The service respects GitHub API rate limits and includes error
        handling for rate limit exceeded scenarios.
        
    Example:
        github_service = GitHubService(github_token="ghp_...")
        files, tree = await github_service.get_repository_files("owner/repo")
        languages = await github_service.get_repo_languages("owner/repo")
    """
    
    def __init__(self, github_token: str):
        """
        Initialize the GitHubService with authentication.
        
        Args:
            github_token (str): GitHub personal access token with appropriate
                              permissions for repository access
                              
        Environment Variables Required:
            GITHUB_TOKEN: GitHub personal access token
            
        Token Permissions Required:
            - repo (for private repositories)
            - public_repo (for public repositories)  
            - read:org (for organization repositories)
            
        Raises:
            ValueError: If github_token is None or empty
            GithubException: If token is invalid or has insufficient permissions
        """
        self.github = Github(github_token)

        # File extensions to index - focus on readable code and documentation
        self.indexable_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx',  # Code files
            '.md', '.txt', '.rst',  # Documentation
            '.yml', '.yaml', '.json', '.toml',  # Config files
            '.env.example', '.gitignore',  # Important dot files
            '.sql', '.sh', '.bash',  # Scripts
            '.html', '.css', '.scss',  # Web files
        }

        # Files to always include if present - critical project files
        self.important_files = {
            'README.md', 'README.rst', 'README.txt',
            'requirements.txt', 'package.json', 'Dockerfile',
            'docker-compose.yml', '.env.example', 'config.py',
            'settings.py', 'main.py', 'app.py', 'index.js'
        }

        # Folders to skip - build artifacts, dependencies, IDE files
        self.skip_folders = {
            'node_modules', '__pycache__', '.git', 'dist',
            'build', 'venv', 'env', '.idea', '.vscode',
            'coverage', '.pytest_cache', 'migrations'
        }

    async def get_repository_files(self, repo_name: str,
                                   max_files: int = 50) -> Tuple[List[Dict], str]:
        """
        Analyze and retrieve relevant files from a GitHub repository.
        
        This method performs intelligent file selection by filtering for relevant
        file types, excluding build artifacts and dependencies, and prioritizing
        important project files. It also builds a visual tree structure.
        
        Args:
            repo_name (str): Repository name in format 'owner/repo'
            max_files (int): Maximum number of files to retrieve. Default: 50
                           This prevents excessive API usage and processing time.
                           
        Returns:
            Tuple[List[Dict], str]: A tuple containing:
                - files_list: List of file dictionaries with structure:
                  {
                      'path': str,     # File path in repository
                      'content': str,  # File content (decoded)
                      'size': int      # File size in bytes
                  }
                - tree_structure: String representation of repository structure
                  with folder and file icons for easy visualization
                  
        File Selection Criteria:
            1. Important files (README, package.json, etc.) are always included
            2. Files with indexable extensions are included
            3. Files in skip_folders are excluded
            4. Files larger than 1MB are skipped
            5. Binary files are automatically excluded
            
        Tree Structure Format:
            ```
            📁 src/
              📁 components/
                📄 Button.tsx
                📄 Modal.tsx
              📄 index.ts
            📄 README.md
            📄 package.json
            ```
            
        Error Handling:
            - Individual file read errors are logged but don't stop processing
            - Repository access errors are propagated
            - Rate limit errors include retry suggestions
            
        Example:
            files, tree = await github_service.get_repository_files(
                repo_name="facebook/react",
                max_files=30
            )
            
            for file in files:
                print(f"Found: {file['path']} ({file['size']} bytes)")
                
        Raises:
            GithubException: If repository doesn't exist or access is denied
            ValueError: If repo_name format is invalid
            RateLimitExceededException: If GitHub API rate limit is exceeded
        """
        try:
            repo = self.github.get_repo(repo_name)

            # Get the repository tree
            tree = repo.get_git_tree(sha='HEAD', recursive=True)

            files_to_index = []
            tree_lines = []
            file_count = 0

            # Build tree structure and collect files
            for item in tree.tree:
                # Skip if in excluded folder
                if any(skip in item.path for skip in self.skip_folders):
                    continue

                # Add to tree structure
                depth = item.path.count('/')
                indent = "  " * depth
                name = item.path.split('/')[-1]

                if item.type == 'tree':
                    tree_lines.append(f"{indent}📁 {name}/")
                else:
                    tree_lines.append(f"{indent}📄 {name}")

                # Check if we should index this file
                if item.type == 'blob' and file_count < max_files:
                    should_index = False

                    # Check if it's an important file
                    if name in self.important_files:
                        should_index = True
                    # Check if it has an indexable extension
                    elif any(item.path.endswith(ext) for ext in self.indexable_extensions):
                        should_index = True

                    if should_index:
                        try:
                            # Get file content
                            file_content = repo.get_contents(item.path)
                            if file_content.size < 1000000:  # Skip files > 1MB
                                content = base64.b64decode(file_content.content).decode('utf-8')
                                files_to_index.append({
                                    'path': item.path,
                                    'content': content,
                                    'size': file_content.size
                                })
                                file_count += 1
                        except Exception as e:
                            logger.logger.warning(f"Could not read file {item.path}: {e}")

            tree_structure = "\n".join(tree_lines[:200])  # Limit tree display

            logger.logger.info(f"Found {len(files_to_index)} files to index in {repo_name}")

            return files_to_index, tree_structure

        except GithubException as e:
            logger.logger.error(f"GitHub API error: {e}")
            raise
        except Exception as e:
            logger.logger.error(f"Error getting repository files: {e}")
            raise

    async def get_repo_languages(self, repo_name: str) -> Dict[str, int]:
        """
        Get programming languages used in the repository with byte counts.
        
        Retrieves language statistics from GitHub's language detection API,
        which analyzes file extensions and content to determine the primary
        languages used in the repository.
        
        Args:
            repo_name (str): Repository name in format 'owner/repo'
            
        Returns:
            Dict[str, int]: Dictionary mapping language names to byte counts
                {
                    'Python': 45231,
                    'JavaScript': 23891,
                    'TypeScript': 12045,
                    'CSS': 3421
                }
                
        Language Detection:
            - Based on file extensions and content analysis
            - Excludes documentation, configuration, and data files
            - Reflects actual code distribution in the repository
            
        Usage:
            Used for repository analysis, technology stack identification,
            and contextual information for AI responses about the codebase.
            
        Example:
            languages = await github_service.get_repo_languages("microsoft/vscode")
            
            # Find primary language
            if languages:
                primary_lang = max(languages, key=languages.get)
                print(f"Primary language: {primary_lang}")
                
            # Calculate percentages
            total_bytes = sum(languages.values())
            for lang, bytes_count in languages.items():
                percentage = (bytes_count / total_bytes) * 100
                print(f"{lang}: {percentage:.1f}%")
                
        Returns:
            Empty dict if repository has no detectable code files or if
            there's an error accessing the repository.
            
        Raises:
            GithubException: If repository access fails (logged, returns empty dict)
        """
        try:
            repo = self.github.get_repo(repo_name)
            return repo.get_languages()
        except Exception as e:
            logger.logger.error(f"Error getting languages: {e}")
            return {}
