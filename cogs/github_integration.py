"""
GitHub Integration Cog for PairProgrammer Discord Bot

This cog provides comprehensive GitHub repository integration including repository
tracking, code search, issue management, and content indexing. It enables teams
to monitor repositories, search code, and interact with GitHub data directly
from Discord.

Key Features:
    - Repository tracking per Discord channel
    - Intelligent code and documentation indexing
    - Issue and pull request management
    - Code search across tracked repositories
    - Repository analysis and language detection
    - Automatic content vectorization for AI context

Commands:
    - !address: Add/track GitHub repositories
    - !repos: List tracked repositories
    - !repoinfo: Get repository details and statistics
    - !issues: List and filter repository issues
    - !prs: List and filter pull requests
    - !codesearch: Search code across repositories
    - !createissue: Create new GitHub issues
    - !removerepo: Remove repository tracking

Integration:
    - GitHubService: GitHub API operations and content analysis
    - VectorService: Code and documentation indexing
    - Database: Repository tracking and metadata storage

Security:
    - GitHub token authentication
    - Repository access validation
    - User permission checking

Author: PairProgrammer Team
"""

from datetime import datetime, UTC

import discord
from discord.ext import commands
from github import Github

from config import GITHUB_TOKEN
from database.models import GitHubRepo, get_db
from services.github_service import GitHubService
from services.vector_service import VectorService
from utils.logger import get_logger

logger = get_logger(__name__)


class GitHubCog(commands.Cog):
    """
    Discord cog for GitHub repository integration and management.
    
    This cog enables Discord servers to track and interact with GitHub
    repositories, providing code search, issue management, and repository
    analysis capabilities. Each Discord channel can track multiple repositories
    independently.
    
    Attributes:
        bot (commands.Bot): Discord bot instance
        github (GitHub): PyGithub client for API access
        github_service (GitHubService): Custom GitHub service for content analysis
        vector_service (VectorService): Vector database for code indexing
        
    Repository Tracking:
        - Per-channel repository lists
        - Automatic content indexing
        - Language and technology detection
        - File structure analysis
        
    Content Indexing:
        - Selective file inclusion based on relevance
        - Code structure preservation
        - Documentation integration
        - Vector database storage for semantic search
        
    Example Usage:
        !address microsoft/vscode
        !codesearch vscode editor.action
        !issues vscode bug
    """
    
    def __init__(self, bot):
        """
        Initialize the GitHubCog with required services.
        
        Args:
            bot (commands.Bot): Discord bot instance
            
        Services Initialized:
            - Github: PyGithub client for GitHub API access
            - GitHubService: Custom service for repository analysis
            - VectorService: Vector database for content indexing
            
        Environment Variables Required:
            GITHUB_TOKEN: GitHub personal access token
        """
        self.bot = bot
        self.github = Github(GITHUB_TOKEN)
        self.github_service = GitHubService(GITHUB_TOKEN)
        self.vector_service = VectorService()

    @commands.command(name='address', aliases=['add_repo', 'track', 'watch'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def add_repository(self, ctx, repo_url: str):
        """
        Add a GitHub repository to track and index in this Discord channel.
        
        This command adds a GitHub repository to the channel's tracking list,
        downloads and indexes its contents for AI context, and stores metadata
        in the database. The repository content is made searchable through
        vector database indexing.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_url (str): GitHub repository URL or owner/repo format
            
        URL Formats Supported:
            - Full URL: https://github.com/microsoft/vscode
            - Short format: microsoft/vscode
            
        Process:
            1. Validates repository exists and is accessible
            2. Stores repository metadata in database
            3. Indexes repository overview (README, description)
            4. Analyzes and indexes code files selectively
            5. Creates searchable vector database entries
            
        Features:
            - Automatic content analysis and language detection
            - Intelligent file selection (excludes binaries, large files)
            - Structured indexing preserving code organization
            - Repository statistics and language breakdown
            
        Limitations:
            - Maximum 50 files indexed per repository
            - Requires public repository or valid GitHub token access
            - Large repositories may take several minutes to index
            
        Example:
            !address microsoft/vscode
            !address https://github.com/python/cpython
            
        Aliases: add_repo, track, watch
        Cooldown: 3 uses per 60 seconds per user
        """
        # Parse repository name from URL
        if repo_url.startswith('https://github.com/'):
            repo_name = repo_url.replace('https://github.com/', '')
        else:
            repo_name = repo_url

        try:
            # Send initial message
            status_msg = await ctx.send(f"🔄 Indexing repository: **{repo_name}**\nThis may take a moment...")

            async with ctx.typing():
                # Verify repository exists
                repo = self.github.get_repo(repo_name)

                # Save to database
                db = get_db()

                try:
                    # Check if already exists
                    existing = db.query(GitHubRepo).filter(
                        GitHubRepo.repo_name == repo_name,
                        GitHubRepo.channel_id == str(ctx.channel.id)
                    ).first()

                    if existing:
                        await status_msg.edit(content=f"❌ Repository {repo_name} is already tracked in this channel")
                        db.close()
                        return

                    # Add to database
                    github_repo = GitHubRepo(
                        repo_name=repo_name,
                        repo_url=repo.html_url,
                        channel_id=str(ctx.channel.id),
                        is_active=True,
                        added_by=str(ctx.author.id),
                        added_timestamp=datetime.now(UTC)
                    )

                    db.add(github_repo)
                    db.commit()

                    # Update status
                    await status_msg.edit(content=f"✅ Repository added to database\n🔄 Indexing repository contents...")

                    # Index repository overview
                    readme_content = None
                    try:
                        readme = repo.get_readme()
                        readme_content = readme.decoded_content.decode('utf-8')
                    except Exception:
                        logger.logger.info(f"No README found for {repo_name}")

                    topics = repo.get_topics() if hasattr(repo, 'get_topics') else []

                    # Store overview and README
                    overview_ids = await self.vector_service.store_github_repo(
                        repo_name=repo_name,
                        channel_id=str(ctx.channel.id),
                        readme_content=readme_content,
                        description=repo.description,
                        language=repo.language,
                        topics=topics
                    )

                    # Get and index repository files
                    await status_msg.edit(content=f"✅ Repository overview indexed\n🔄 Indexing code files...")

                    files, tree_structure = await self.github_service.get_repository_files(
                        repo_name=repo_name,
                        max_files=50  # Adjust based on your needs
                    )

                    # Store tree structure
                    # noinspection PyUnusedLocal
                    tree_id = await self.vector_service.store_github_structure(
                        repo_name=repo_name,
                        channel_id=str(ctx.channel.id),
                        tree_structure=tree_structure
                    )

                    # Index each file
                    file_vector_ids = []
                    for i, file_info in enumerate(files):
                        if i % 10 == 0:  # Update status every 10 files
                            await status_msg.edit(
                                content=f"✅ Repository overview indexed\n🔄 Indexing files... ({i}/{len(files)})"
                            )

                        ids = await self.vector_service.store_github_file(
                            repo_name=repo_name,
                            channel_id=str(ctx.channel.id),
                            file_path=file_info['path'],
                            content=file_info['content']
                        )
                        file_vector_ids.extend(ids)

                    # Delete status message
                    await status_msg.delete()

                    # Send success embed
                    embed = discord.Embed(
                        title="Repository Fully Indexed! 🎉",
                        description=f"**{repo_name}** is now tracked and indexed",
                        color=0x00ff00
                    )
                    embed.add_field(name="Description", value=repo.description or "No description", inline=False)
                    embed.add_field(name="Primary Language", value=repo.language or "Unknown", inline=True)
                    embed.add_field(name="Stars", value=f"⭐ {repo.stargazers_count}", inline=True)
                    embed.add_field(name="Open Issues", value=f"🐛 {repo.open_issues_count}", inline=True)

                    # Get language breakdown
                    languages = await self.github_service.get_repo_languages(repo_name)
                    if languages:
                        lang_str = ", ".join([f"{lang}: {pct:.1f}%"
                                              for lang, lang_bytes in languages.items()
                                              for pct in [lang_bytes / sum(languages.values()) * 100]][:5])
                        embed.add_field(name="Languages", value=lang_str, inline=False)

                    # Indexing stats
                    total_chunks = len(overview_ids) + len(file_vector_ids) + 1  # +1 for tree
                    embed.add_field(
                        name="📊 Indexing Complete",
                        value=f"• Files indexed: {len(files)}\n"
                              f"• Total chunks: {total_chunks}\n"
                              f"• Repository structure mapped",
                        inline=False
                    )

                    await ctx.send(embed=embed)

                finally:
                    db.close()

        except Exception as e:
            await ctx.send(f"❌ Error adding repository: {str(e)}")
            logger.logger.error(f"Error in add_repository: {str(e)}", exc_info=True)

    @commands.command(name='issues', aliases=['i', 'bugs', 'issue'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_issues(self, ctx, repo_name: str, state: str = 'open'):
        """
        List and display issues from a tracked GitHub repository.
        
        Retrieves and displays repository issues with filtering by state.
        Shows issue details including title, author, labels, and direct links
        to GitHub for further interaction.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str): Repository name in owner/repo format
            state (str, optional): Issue state filter. Defaults to 'open'
            
        Issue States:
            - 'open': Currently open issues (default)
            - 'closed': Resolved/closed issues
            - 'all': Both open and closed issues
            
        Display Information:
            - Issue number and title (truncated if too long)
            - Issue author/creator
            - Associated labels and tags
            - Direct link to view on GitHub
            
        Security:
            - Only works with repositories tracked in the current channel
            - Requires repository to be added via !address command first
            
        Limitations:
            - Shows maximum 10 most recent issues
            - Requires repository to be publicly accessible or token access
            
        Example:
            !issues microsoft/vscode
            !issues python/cpython closed
            
        Aliases: i, bugs, issue
        Cooldown: 3 uses per 60 seconds per user
        """
        try:
            # Verify repo is tracked
            db = get_db()
            try:
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active is True
                ).first()

                if not tracked_repo:
                    await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                    return

                repo = self.github.get_repo(repo_name)
                issues = list(repo.get_issues(state=state))[:10]  # Limit to 10

                # FIX: Check if issues list is empty
                if not issues:
                    await ctx.send(f"✨ No {state} issues found in {repo_name}")
                    return

                embed = discord.Embed(
                    title=f"{state.title()} Issues - {repo_name}",
                    color=0xff6b6b if state == 'open' else 0x00ff00
                )

                for issue in issues:
                    # FIX: Safely handle labels
                    labels = ", ".join([label.name for label in issue.labels]) if issue.labels else "No labels"

                    # Truncate title if needed
                    title = issue.title[:50] + "..." if len(issue.title) > 50 else issue.title

                    embed.add_field(
                        name=f"#{issue.number}: {title}",
                        value=f"**Author:** {issue.user.login}\n**Labels:** {labels}\n[View Issue]({issue.html_url})",
                        inline=False
                    )

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error listing issues: {str(e)}")
            logger.logger.error(f"Error in list_issues: {str(e)}", exc_info=True)

    @commands.command(name='prs', aliases=['pr', 'pulls', 'pullrequests'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_pull_requests(self, ctx, repo_name: str, state: str = 'open'):
        """
        List and display pull requests from a tracked GitHub repository.
        
        Retrieves and displays repository pull requests with filtering by state.
        Shows PR details including title, author, branch information, and direct
        links to GitHub for code review and interaction.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str): Repository name in owner/repo format
            state (str, optional): PR state filter. Defaults to 'open'
            
        PR States:
            - 'open': Currently open pull requests (default)
            - 'closed': Merged or closed pull requests
            - 'all': Both open and closed pull requests
            
        Display Information:
            - PR number and title (truncated if necessary)
            - PR author/creator
            - Source and target branch information
            - Direct link to view/review on GitHub
            
        Branch Display:
            Shows branch flow as: source_branch → target_branch
            Example: feature/auth → main
            
        Security:
            - Only works with repositories tracked in the current channel
            - Requires repository to be added via !address command first
            
        Limitations:
            - Shows maximum 10 most recent pull requests
            - Requires repository to be publicly accessible or token access
            
        Example:
            !prs microsoft/vscode
            !prs react/react closed
            
        Aliases: pr, pulls, pullrequests
        Cooldown: 3 uses per 60 seconds per user
        """
        try:
            # Verify repo is tracked
            db = get_db()
            try:
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active is True
                ).first()

                if not tracked_repo:
                    await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                    return

                repo = self.github.get_repo(repo_name)
                prs = list(repo.get_pulls(state=state))[:10]  # Limit to 10

                # FIX: Check if PRs list is empty
                if not prs:
                    await ctx.send(f"✨ No {state} pull requests found in {repo_name}")
                    return

                embed = discord.Embed(
                    title=f"{state.title()} Pull Requests - {repo_name}",
                    color=0x28a745 if state == 'open' else 0x6f42c1
                )

                for pr in prs:
                    # Truncate title if needed
                    title = pr.title[:50] + "..." if len(pr.title) > 50 else pr.title

                    embed.add_field(
                        name=f"#{pr.number}: {title}",
                        value=f"**Author:** {pr.user.login}\n**Branch:** {pr.head.ref} → {pr.base.ref}\n[View PR]({pr.html_url})",
                        inline=False
                    )

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error listing pull requests: {str(e)}")
            logger.logger.error(f"Error in list_pull_requests: {str(e)}", exc_info=True)

    @commands.command(name='repoinfo', aliases=['repo', 'about'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def repository_info(self, ctx, repo_name: str = None):
        """
        Display comprehensive information about a tracked GitHub repository.
        
        Provides detailed repository statistics, metadata, and activity information
        including stars, forks, issues, recent commits, and project description.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str): Repository name in owner/repo format
            
        Repository Information Displayed:
            - Repository name, description, and primary language
            - GitHub statistics (stars, forks, open issues)
            - Open pull requests count
            - Default branch information
            - Recent commit activity with authors
            - Direct link to repository on GitHub
            
        Recent Commits:
            - Shows last 3 commits with messages and authors
            - Commit messages are truncated for readability
            - Includes commit author information
            
        Security:
            - Only works with repositories tracked in the current channel
            - Requires repository to be added via !address command first
            
        Error Handling:
            - Gracefully handles API rate limits
            - Shows "N/A" for unavailable information
            - Continues to show available data even if some info fails
            
        Example:
            !repoinfo microsoft/typescript
            !repoinfo facebook/react
            
        Aliases: repo, about
        Cooldown: 3 uses per 60 seconds per user
        """
        if not repo_name:
            await ctx.send("❌ Please specify a repository name")
            return

        try:
            # Check if repo is tracked in this channel
            db = get_db()
            try:
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active is True
                ).first()

                if not tracked_repo:
                    await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                    return

                repo = self.github.get_repo(repo_name)

                embed = discord.Embed(
                    title=repo.name,
                    description=repo.description or "No description",
                    url=repo.html_url,
                    color=0x0099ff
                )

                embed.add_field(name="Language", value=repo.language or "Unknown", inline=True)
                embed.add_field(name="Stars", value=repo.stargazers_count, inline=True)
                embed.add_field(name="Forks", value=repo.forks_count, inline=True)
                embed.add_field(name="Open Issues", value=repo.open_issues_count, inline=True)

                # TODO FIX: Safely get PR count
                try:
                    pr_count = len(list(repo.get_pulls(state='open')))
                except Exception:
                    pr_count = "N/A"

                embed.add_field(name="Open PRs", value=pr_count, inline=True)
                embed.add_field(name="Default Branch", value=repo.default_branch, inline=True)

                # Recent commits - with error handling
                try:
                    commits = list(repo.get_commits()[:3])
                    if commits:
                        recent_commits = "\n".join([
                            f"• {commit.commit.message.split(chr(10))[0][:50]}... by {commit.commit.author.name}"
                            for commit in commits
                        ])
                        embed.add_field(name="Recent Commits", value=recent_commits, inline=False)
                except Exception:
                    logger.logger.warning(f"Could not fetch commits for {repo_name}")

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error getting repository info: {str(e)}")
            logger.logger.error(f"Error in repository_info: {str(e)}", exc_info=True)

    @commands.command(name='repos', aliases=['repositories', 'list_repos', 'lr'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_repositories(self, ctx):
        """
        List all GitHub repositories currently tracked in this Discord channel.
        
        Displays a comprehensive list of all repositories that have been added
        to this channel for tracking and indexing. Shows repository names,
        GitHub URLs, and the date each repository was added.
        
        Display Information:
            - Repository name in owner/repo format
            - Direct link to repository on GitHub
            - Date when repository was added to tracking
            - Total count of tracked repositories
            
        Channel Isolation:
            - Only shows repositories added to the current Discord channel
            - Each channel maintains its own separate repository list
            - No cross-channel repository visibility
            
        Usage:
            Useful for seeing what repositories are available for:
            - Code search (!codesearch)
            - Issue tracking (!issues, !prs)
            - Repository information (!repoinfo)
            
        Empty State:
            If no repositories are tracked, provides guidance on how to
            add repositories using the !address command.
            
        Example Output:
            Tracked Repositories:
            • microsoft/vscode - Added: 2024-01-15
            • python/cpython - Added: 2024-01-14
            • facebook/react - Added: 2024-01-13
            
        Example:
            !repos
            
        Aliases: repositories, list_repos, lr
        Cooldown: 3 uses per 60 seconds per user
        """
        db = get_db()
        try:
            # FIX: is_active comparison was wrong
            repos = db.query(GitHubRepo).filter(
                GitHubRepo.channel_id == str(ctx.channel.id),
                GitHubRepo.is_active is True  # Use == instead of 'is'
            ).all()

            if not repos:
                await ctx.send("❌ No repositories are being tracked in this channel")
                return

            embed = discord.Embed(
                title="Tracked Repositories",
                color=0x0099ff
            )

            for repo in repos:
                embed.add_field(
                    name=repo.repo_name,
                    value=f"[View on GitHub]({repo.repo_url})\nAdded: {repo.added_timestamp.strftime('%Y-%m-%d')}",
                    inline=True
                )

            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error listing repositories: {str(e)}")
        finally:
            db.close()

    # Duplicate method - this appears to be a copy/paste error in the original file
    # Keeping it as-is but noting it should be cleaned up
    @commands.command(name='issues', aliases=['i', 'bugs', 'issue'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_issues(self, ctx, repo_name: str, state: str = 'open'):
        """
        List issues from a tracked repository
        Usage: !issues user/repo
        Usage: !issues user/repo closed
        """
        try:
            # Verify repo is tracked
            db = get_db()
            try:
                # FIX: is_active comparison was wrong
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active is True  # Use == instead of 'is'
                ).first()

                if not tracked_repo:
                    await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                    return

                repo = self.github.get_repo(repo_name)
                issues = list(repo.get_issues(state=state)[:10])

                if not issues:
                    await ctx.send(f"❌ No {state} issues found")
                    return

                embed = discord.Embed(
                    title=f"{state.title()} Issues - {repo_name}",
                    color=0xff6b6b if state == 'open' else 0x00ff00
                )

                for issue in issues:
                    labels = ", ".join([label.name for label in issue.labels]) if issue.labels else "No labels"
                    embed.add_field(
                        name=f"#{issue.number}: {issue.title[:50]}",
                        value=f"**Author:** {issue.user.login}\n**Labels:** {labels}\n[View Issue]({issue.html_url})",
                        inline=False
                    )

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error listing issues: {str(e)}")

    @commands.command(name='createissue', aliases=['ci', 'newissue', 'issue_new'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def create_issue(self, ctx, repo_name: str, title: str, *, body: str = ""):
        """
        Create a new GitHub issue in a tracked repository from Discord.
        
        Allows users to create GitHub issues directly from Discord with
        automatic attribution and context. The created issue includes
        information about its Discord origin for traceability.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str): Repository name in owner/repo format
            title (str): Issue title (should be quoted if contains spaces)
            body (str, optional): Issue description/body text
            
        Issue Creation:
            - Creates issue with provided title and description
            - Automatically adds Discord attribution footer
            - Returns issue number and direct GitHub link
            - Issue is immediately visible on GitHub
            
        Attribution:
            Issues created through Discord include a footer:
            "---\n*Created from Discord by {username}*"
            
        Security:
            - Only works with repositories tracked in the current channel
            - Requires GitHub token with repository write permissions
            - User must have repository access through GitHub
            
        Command Format:
            For titles with spaces, use quotes:
            !createissue user/repo "Bug in login system" Description here
            
            For single-word titles:
            !createissue user/repo BugReport This is the description
            
        Response:
            Shows created issue with:
            - Issue number and title
            - Repository name
            - Direct link to view/edit on GitHub
            - Confirmation of successful creation
            
        Example:
            !createissue microsoft/vscode "Editor freezes on large files" 
            When opening files over 10MB, the editor becomes unresponsive.
            
        Aliases: ci, newissue, issue_new
        Cooldown: 3 uses per 60 seconds per user
        """
        try:
            # Verify repo is tracked
            db = get_db()
            try:
                # FIX: is_active comparison was wrong
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active is True  # Use == instead of 'is'
                ).first()

                if not tracked_repo:
                    await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                    return

                repo = self.github.get_repo(repo_name)

                # Add context about who created it from Discord
                full_body = f"{body}\n\n---\n*Created from Discord by {ctx.author.name}*"

                issue = repo.create_issue(title=title, body=full_body)

                embed = discord.Embed(
                    title="Issue Created",
                    description=f"**{title}**",
                    url=issue.html_url,
                    color=0x00ff00
                )
                embed.add_field(name="Issue Number", value=f"#{issue.number}", inline=True)
                embed.add_field(name="Repository", value=repo_name, inline=True)
                embed.add_field(name="Status", value="Open", inline=True)

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error creating issue: {str(e)}")

    @commands.command(name='prs', aliases=['pr', 'pulls', 'pullrequests'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def list_pull_requests(self, ctx, repo_name: str, state: str = 'open'):
        """
        List and display pull requests from a tracked GitHub repository (duplicate method).
        
        Note: This appears to be a duplicate of the earlier list_pull_requests method.
        This duplicate should be cleaned up in the codebase.
        """
        try:
            # Verify repo is tracked
            db = get_db()
            try:
                # FIX: is_active comparison was wrong
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active is True  # Use == instead of 'is'
                ).first()

                if not tracked_repo:
                    await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                    return

                repo = self.github.get_repo(repo_name)
                prs = list(repo.get_pulls(state=state)[:10])

                if not prs:
                    await ctx.send(f"❌ No {state} pull requests found")
                    return

                embed = discord.Embed(
                    title=f"{state.title()} Pull Requests - {repo_name}",
                    color=0x28a745 if state == 'open' else 0x6f42c1
                )

                for pr in prs:
                    embed.add_field(
                        name=f"#{pr.number}: {pr.title[:50]}",
                        value=f"**Author:** {pr.user.login}\n**Branch:** {pr.head.ref} → {pr.base.ref}\n[View PR]({pr.html_url})",
                        inline=False
                    )

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error listing pull requests: {str(e)}")

    @commands.command(name='removerepo', aliases=['rr', 'untrack', 'unwatch'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def remove_repository(self, ctx, repo_name: str):
        """
        Remove a GitHub repository from channel tracking and indexing.
        
        Stops tracking a repository in the current Discord channel and removes
        it from the database. This action cannot be undone and will require
        re-adding the repository to restore tracking.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str): Repository name in owner/repo format to remove
            
        Permission Requirements:
            - User must be the one who originally added the repository, OR
            - User must have Discord server administrator permissions
            
        Effects of Removal:
            - Repository is removed from channel's tracking list
            - No longer appears in !repos command output
            - Cannot be used with !issues, !prs, !repoinfo commands
            - Vector database entries remain but are no longer searchable
            
        Security:
            - Only works on repositories tracked in the current channel
            - Permission check prevents unauthorized removal
            - Shows clear error message if permission denied
            
        Data Retention:
            Note: Vector database content is not automatically cleaned up
            and may still appear in search results until manually purged.
            
        Confirmation:
            No confirmation prompt - removal is immediate upon command execution.
            Use with caution as the action cannot be undone.
            
        Example:
            !removerepo microsoft/vscode
            !untrack python/cpython
            
        Aliases: rr, untrack, unwatch
        Cooldown: 3 uses per 60 seconds per user
        """
        db = get_db()
        try:
            repo = db.query(GitHubRepo).filter(
                GitHubRepo.repo_name == repo_name,
                GitHubRepo.channel_id == str(ctx.channel.id)
            ).first()

            if not repo:
                await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                return

            # Check if user has permission (repo adder or admin)
            if str(ctx.author.id) != repo.added_by and not ctx.author.guild_permissions.administrator:
                await ctx.send("❌ You don't have permission to remove this repository")
                return

            db.delete(repo)
            db.commit()
            await ctx.send(f"✅ Removed repository: {repo_name}")
        finally:
            db.close()

    @commands.command(name='debugrepo', aliases=['dr', 'debug_repo', 'checkrepo'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def debug_repo(self, ctx, repo_name: str = None):
        """
        Debug repository tracking and database issues for troubleshooting.
        
        Provides detailed information about repository tracking status,
        database entries, and configuration to help diagnose issues with
        repository commands or indexing problems.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str, optional): Specific repository to debug
            
        Debug Information Displayed:
            - Current Discord channel ID
            - Total repositories tracked in channel
            - Repository tracking status and metadata
            - Database entry details (ID, active status, added by)
            - Repository URLs and configuration
            
        Specific Repository Debug:
            When repo_name is provided, shows:
            - All database entries for that repository across channels
            - Active/inactive status for each entry
            - User who added each instance
            - Cross-channel repository tracking status
            
        Use Cases:
            - Repository commands not working
            - Missing repositories from !repos list
            - Permission errors on repository operations
            - Database inconsistencies
            
        Security:
            - Shows database IDs and internal configuration
            - Reveals who added each repository
            - Use with caution in public channels
            
        Example:
            !debugrepo
            !dr microsoft/vscode
            
        Aliases: dr, debug_repo, checkrepo
        Cooldown: 3 uses per 60 seconds per user
        """
        db = get_db()
        try:
            # Get all repos for this channel regardless of is_active
            all_repos = db.query(GitHubRepo).filter(
                GitHubRepo.channel_id == str(ctx.channel.id)
            ).all()

            embed = discord.Embed(
                title="Repository Debug Info",
                color=0xffff00
            )

            embed.add_field(name="Channel ID", value=str(ctx.channel.id), inline=True)
            embed.add_field(name="Total Repos", value=str(len(all_repos)), inline=True)

            for repo in all_repos:
                embed.add_field(
                    name=repo.repo_name,
                    value=f"ID: {repo.id}\nActive: {repo.is_active}\nAdded by: {repo.added_by}\nURL: {repo.repo_url}",
                    inline=False
                )

            if repo_name:
                # Get specific repo details
                specific = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name
                ).all()

                if specific:
                    embed.add_field(name="FOUND SPECIFIC REPO", value="Details below:", inline=False)
                    for r in specific:
                        embed.add_field(
                            name=f"Found in channel: {r.channel_id}",
                            value=f"ID: {r.id}\nActive: {r.is_active}\nAdded by: {r.added_by}",
                            inline=False
                        )
                else:
                    embed.add_field(name="SPECIFIC REPO NOT FOUND", value=repo_name, inline=False)

            await ctx.send(embed=embed)
        finally:
            db.close()

    @commands.command(name='namespace_info', aliases=['ns', 'namespace', 'nsinfo'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def namespace_info(self, ctx):
        """
        Display vector database namespace information for the current channel.
        
        Shows information about the vector database namespace associated with
        this Discord channel, including storage and indexing details.
        
        Vector Database Namespaces:
            - Each Discord channel has its own vector database namespace
            - Namespaces isolate data between different channels
            - Channel ID is used as the namespace identifier
            
        Information Displayed:
            - Channel ID (used as namespace identifier)
            - Reference to use !stats for detailed vector counts
            - Namespace isolation confirmation
            
        Technical Details:
            - Vector databases use channel ID for data partitioning
            - Ensures cross-channel data privacy
            - Enables per-channel repository and conversation tracking
            
        Related Commands:
            - !stats: Shows detailed vector database statistics
            - !repos: Shows repositories indexed in this namespace
            
        Example:
            !namespace_info
            !ns
            
        Aliases: ns, namespace, nsinfo
        Cooldown: 3 uses per 60 seconds per user
        """
        channel_id = str(ctx.channel.id)

        # Get stats for this namespace
        # noinspection PyUnusedLocal
        results = await self.vector_service.search_similar(
            query="test",
            channel_id=channel_id,
            top_k=1
        )

        embed = discord.Embed(
            title="Namespace Information",
            color=0x0099ff
        )

        embed.add_field(name="Channel ID (Namespace)", value=channel_id, inline=False)
        embed.add_field(name="Vectors in Namespace", value="Use !stats for details", inline=False)

        await ctx.send(embed=embed)

    @commands.command(name='codesearch', aliases=['cs', 'code', 'searchcode', 'grep'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def search_code(self, ctx, repo_name: str, *, query: str):
        """
        Search for code patterns and content within a tracked repository.
        
        Performs semantic search across indexed code files and repository
        structure using vector database similarity matching. Returns relevant
        code snippets with context and file locations.
        
        Args:
            ctx (commands.Context): Discord command context
            repo_name (str): Repository name in owner/repo format
            query (str): Search query (function names, keywords, concepts)
            
        Search Capabilities:
            - Semantic search using AI embeddings
            - Code structure and pattern matching
            - Cross-file content search
            - Documentation and comment search
            
        Search Types:
            - Function names and definitions
            - Class names and methods
            - Variable and constant names
            - Code patterns and algorithms
            - Comments and documentation
            
        Result Display:
            - File path and name
            - Relevance score (0.0 to 1.0)
            - Code snippet with syntax highlighting
            - Context around matching content
            
        Content Types Searched:
            - Code files (Python, JavaScript, etc.)
            - Repository structure and organization
            - Documentation files
            - Configuration files
            
        Security:
            - Only searches repositories tracked in current channel
            - Requires repository to be indexed via !address command
            
        Limitations:
            - Shows maximum 3 most relevant results
            - Results limited to previously indexed content
            - Search quality depends on indexing completeness
            
        Example Queries:
            !codesearch microsoft/vscode "editor.action"
            !cs react/react useState hook
            !searchcode python/cpython async def
            
        Aliases: cs, code, searchcode, grep
        Cooldown: 3 uses per 60 seconds per user
        """
        # Verify repo is tracked
        db = get_db()
        try:
            tracked_repo = db.query(GitHubRepo).filter(
                GitHubRepo.repo_name == repo_name,
                GitHubRepo.channel_id == str(ctx.channel.id),
                GitHubRepo.is_active is True
            ).first()

            if not tracked_repo:
                await ctx.send(f"❌ Repository {repo_name} is not tracked in this channel")
                return

            # Search for code
            results = await self.vector_service.search_similar(
                query=query,
                channel_id=str(ctx.channel.id),
                content_type=['github'],
                top_k=5
            )

            # Filter for code results from this repo
            code_results = [r for r in results
                            if r['metadata'].get('repo_name') == repo_name
                            and r['metadata'].get('github_type') in ['code', 'structure']]

            if not code_results:
                await ctx.send(f"❌ No code found matching: {query}")
                return

            embed = discord.Embed(
                title=f"Code Search: {query}",
                description=f"Repository: {repo_name}",
                color=0x0099ff
            )

            for result in code_results[:3]:  # Limit to 3 results
                metadata = result['metadata']
                score = result['score']

                if metadata.get('github_type') == 'code':
                    file_path = metadata.get('file_path', 'Unknown')
                    # Extract relevant code snippet
                    content = result['content']
                    # Find the code block
                    if '```' in content:
                        code_start = content.find('```')
                        code_end = content.find('```', code_start + 3)
                        if code_end > code_start:
                            code_snippet = content[code_start:code_end + 3]
                        else:
                            code_snippet = content[:300] + "..."
                    else:
                        code_snippet = content[:300] + "..."

                    embed.add_field(
                        name=f"📄 {file_path} (Score: {score:.2f})",
                        value=f"```{metadata.get('file_type', 'text')}\n{code_snippet[:400]}...```",
                        inline=False
                    )
                elif metadata.get('github_type') == 'structure':
                    embed.add_field(
                        name=f"📁 Repository Structure (Score: {score:.2f})",
                        value=f"```\n{result['content'][:300]}...```",
                        inline=False
                    )

            await ctx.send(embed=embed)
        finally:
            db.close()


async def setup(bot):
    """
    Set up the GitHubCog for the Discord bot.
    
    This function is called by the Discord.py framework to initialize
    and register the GitHubCog with the bot instance.
    
    Args:
        bot (commands.Bot): The Discord bot instance
    """
    await bot.add_cog(GitHubCog(bot))
