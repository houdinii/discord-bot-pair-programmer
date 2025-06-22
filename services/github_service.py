# services/github_service.py
import base64
from typing import List, Dict, Tuple
from github import Github, GithubException
from utils.logger import get_logger

logger = get_logger(__name__)


class GitHubService:
    def __init__(self, github_token: str):
        self.github = Github(github_token)

        # File extensions to index
        self.indexable_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx',  # Code files
            '.md', '.txt', '.rst',  # Documentation
            '.yml', '.yaml', '.json', '.toml',  # Config files
            '.env.example', '.gitignore',  # Important dot files
            '.sql', '.sh', '.bash',  # Scripts
            '.html', '.css', '.scss',  # Web files
        }

        # Files to always include if present
        self.important_files = {
            'README.md', 'README.rst', 'README.txt',
            'requirements.txt', 'package.json', 'Dockerfile',
            'docker-compose.yml', '.env.example', 'config.py',
            'settings.py', 'main.py', 'app.py', 'index.js'
        }

        # Folders to skip
        self.skip_folders = {
            'node_modules', '__pycache__', '.git', 'dist',
            'build', 'venv', 'env', '.idea', '.vscode',
            'coverage', '.pytest_cache', 'migrations'
        }

    async def get_repository_files(self, repo_name: str,
                                   max_files: int = 50) -> Tuple[List[Dict], str]:
        """
        Get repository files and structure
        Returns: (files_list, tree_structure)
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
        """Get programming languages used in the repository"""
        try:
            repo = self.github.get_repo(repo_name)
            return repo.get_languages()
        except Exception as e:
            logger.logger.error(f"Error getting languages: {e}")
            return {}
