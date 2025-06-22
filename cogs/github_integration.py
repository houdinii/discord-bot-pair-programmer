# cogs/github_integration.py
from datetime import datetime, UTC

import discord
from discord.ext import commands
from github import Github

from database.models import GitHubRepo, get_db
from config import GITHUB_TOKEN


class GitHubCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.github = Github(GITHUB_TOKEN)

    @commands.command(name='address')
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
            # Verify repository exists and we have access
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
                    await ctx.send(f"❌ Repository {repo_name} is already tracked in this channel")
                    db.close()
                    return

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

                embed = discord.Embed(
                    title="Repository Added",
                    description=f"Now tracking: **{repo_name}**",
                    color=0x00ff00
                )
                embed.add_field(name="Description", value=repo.description or "No description", inline=False)
                embed.add_field(name="Language", value=repo.language or "Unknown", inline=True)
                embed.add_field(name="Stars", value=repo.stargazers_count, inline=True)
                embed.add_field(name="Open Issues", value=repo.open_issues_count, inline=True)

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error adding repository: {str(e)}")

    @commands.command(name='repos')
    async def list_repositories(self, ctx):
        """List all tracked repositories in this channel"""
        db = get_db()
        try:
            # FIX: is_active comparison was wrong
            repos = db.query(GitHubRepo).filter(
                GitHubRepo.channel_id == str(ctx.channel.id),
                GitHubRepo.is_active == True  # Use == instead of 'is'
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

    @commands.command(name='repoinfo')
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
                # FIX: is_active comparison was wrong
                tracked_repo = db.query(GitHubRepo).filter(
                    GitHubRepo.repo_name == repo_name,
                    GitHubRepo.channel_id == str(ctx.channel.id),
                    GitHubRepo.is_active == True  # Use == instead of 'is'
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
                embed.add_field(name="Open PRs", value=len(list(repo.get_pulls(state='open'))), inline=True)
                embed.add_field(name="Default Branch", value=repo.default_branch, inline=True)

                # Recent commits
                commits = list(repo.get_commits()[:3])
                if commits:
                    recent_commits = "\n".join([
                        f"• {commit.commit.message.split(chr(10))[0][:50]}... by {commit.commit.author.name}"
                        for commit in commits
                    ])
                    embed.add_field(name="Recent Commits", value=recent_commits, inline=False)

                await ctx.send(embed=embed)
            finally:
                db.close()

        except Exception as e:
            await ctx.send(f"❌ Error getting repository info: {str(e)}")

    @commands.command(name='issues')
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
                    GitHubRepo.is_active == True  # Use == instead of 'is'
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

    @commands.command(name='createissue')
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
                    GitHubRepo.is_active == True  # Use == instead of 'is'
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

    @commands.command(name='prs')
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
                    GitHubRepo.is_active == True  # Use == instead of 'is'
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

    @commands.command(name='removerepo')
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
    @commands.command(name='debugrepo')
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


async def setup(bot):
    await bot.add_cog(GitHubCog(bot))
