# cogs/github_integration.py
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
    def __init__(self, bot):
        self.bot = bot
        self.github = Github(GITHUB_TOKEN)
        self.github_service = GitHubService(GITHUB_TOKEN)
        self.vector_service = VectorService()

    @commands.command(name='address', aliases=['add_repo', 'track', 'watch'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def add_repository(self, ctx, repo_url: str):
        """
        Add a GitHub repository to track in this channel
        Usage: !address https://github.com/user/repo
        Usage: !address user/repo
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
        List issues from a tracked repository
        Usage: !issues user/repo
        Usage: !issues user/repo closed
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
        List pull requests from a tracked repository
        Usage: !prs user/repo
        Usage: !prs user/repo closed
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
        Get detailed information about a tracked repository
        Usage: !repoinfo user/repo
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
        """List all tracked repositories in this channel"""
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
        Create a new issue in a tracked repository
        Usage: !createissue user/repo "Bug in authentication" This is the issue description
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
        List pull requests from a tracked repository
        Usage: !prs user/repo
        Usage: !prs user/repo closed
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
        Remove a repository from tracking
        Usage: !removerepo user/repo
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

    # Debug command to help diagnose issues
    @commands.command(name='debugrepo', aliases=['dr', 'debug_repo', 'checkrepo'])
    @commands.cooldown(3, 60, commands.BucketType.user)
    async def debug_repo(self, ctx, repo_name: str = None):
        """Debug repository tracking issues"""
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
        """Show vector database namespace information"""
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
        Search for code in a tracked repository
        Usage: !codesearch user/repo function_name
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
    await bot.add_cog(GitHubCog(bot))
