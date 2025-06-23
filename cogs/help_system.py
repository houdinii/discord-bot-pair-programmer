"""
Help System for PairProgrammer Discord Bot

This cog provides a comprehensive help system with categorized commands,
usage tips, and command aliases. It replaces Discord.py's default help
command with a more user-friendly and feature-rich help interface.

Commands:
    !help [category]: Show categorized help or general overview
    !commands_table: Display all commands in compact table format
    !tips: Show usage tips and best practices
    !aliases [command]: Show command aliases and shortcuts

Features:
    - Categorized command organization (ai, docs, memory, github, admin)
    - Command aliases and shortcuts display
    - Usage tips and best practices
    - Quick start examples for new users
    - Paginated command reference tables

Author: PairProgrammer Team
"""

import discord
from discord.ext import commands
from typing import Optional


class HelpSystem(commands.Cog):
    """
    Dynamic help system cog providing categorized command help and usage guidance.
    
    This cog replaces the default Discord.py help command with a more comprehensive
    and user-friendly help system that organizes commands by category and provides
    usage tips, aliases, and best practices.
    """
    
    def __init__(self, bot):
        """
        Initialize the HelpSystem cog.
        
        Args:
            bot (commands.Bot): The Discord bot instance
        """
        self.bot = bot

    @commands.command(name='help', aliases=['h', '?', 'commands'])
    async def help_command(self, ctx, category: Optional[str] = None):
        """
        Display categorized help information for bot commands.
        
        Shows either a general overview of all command categories with quick start
        examples, or detailed information about a specific category of commands.
        
        Args:
            category (Optional[str]): Specific category to show help for.
                                    Options: 'ai', 'docs', 'memory', 'github', 'admin'
        
        Categories:
            - ai: AI chat and conversation commands
            - docs: Document upload, search, and analysis commands  
            - memory: Memory storage and recall commands
            - github: GitHub repository integration commands
            - admin: Administrative and debugging commands
            
        Usage:
            !help               # Show general overview and quick start
            !help ai            # Show AI/chat commands
            !help docs          # Show document management commands
            !help memory        # Show memory system commands
            !help github        # Show GitHub integration commands
            !help admin         # Show admin and debug commands
            
        Aliases: h, ?, commands
        
        Example:
            !help
            # Shows: General overview with quick start examples
            
            !help ai
            # Shows: All AI and chat-related commands with descriptions
        """

        # Define command categories with aliases
        categories = {
            'ai': {
                'title': '🤖 AI & Chat Commands',
                'description': 'Interact with AI models and manage conversations',
                'commands': [
                    ('!chat', 'c, ask, ai', 'Chat with AI with context'),
                    ('!quick', 'q, gpt, ask4', 'Quick chat with GPT-4'),
                    ('!models', 'model, list_models, lm', 'List available AI models'),
                    ('!search', 's, find, lookup', 'Search conversation history'),
                    ('!context', 'ctx, preview, show_context', 'Preview context retrieval'),
                    ('!autosave', 'as, auto, toggle_save', 'Toggle auto-memory saving'),
                ]
            },
            'docs': {
                'title': '📄 Document Commands',
                'description': 'Upload, search, and analyze documents',
                'commands': [
                    ('!upload', 'up, add, attach', 'Upload & index a file'),
                    ('!files', 'f, list, ls, docs', 'List uploaded files'),
                    ('!searchfiles', 'sf, searchdocs, finddoc, fd', 'Search file contents'),
                    ('!fileinfo', 'fi, info, details', 'Get file details'),
                    ('!getfile', 'get, download, dl', 'Get download link'),
                    ('!askdoc', 'ad, askdoc, question, qd', 'Ask about a document'),
                    ('!papers', 'p, documents, pdfs', 'List papers (all/pdf/recent)'),
                    ('!deletefile', 'delete, rm, remove', 'Delete a file (owner only)'),
                ]
            },
            'memory': {
                'title': '🧠 Memory Commands',
                'description': 'Save and recall important information',
                'commands': [
                    ('!remember', 'r, save, mem, note', 'Save tagged memory'),
                    ('!recall', 'rc, find_memory, search_memory', 'Search memories'),
                    ('!get_memory', 'gm, getmem, memory', 'Get specific memory'),
                    ('!list_memories', 'lm, memories, list_mem, mems', 'List all memories'),
                    ('!forget', 'fg, delete_memory, delmem', 'Delete memory'),
                    ('!stats', 'st, statistics, info', 'Vector DB statistics'),
                ]
            },
            'github': {
                'title': '🐙 GitHub Commands',
                'description': 'Track and search GitHub repositories',
                'commands': [
                    ('!address', 'add_repo, track, watch', 'Track a repository'),
                    ('!repos', 'repositories, list_repos, lr', 'List tracked repos'),
                    ('!repoinfo', 'ri, repo, about', 'Repository details'),
                    ('!issues', 'i, bugs, issue', 'List issues'),
                    ('!prs', 'pr, pulls, pullrequests', 'List pull requests'),
                    ('!codesearch', 'cs, code, searchcode, grep', 'Search code'),
                    ('!createissue', 'ci, newissue, issue_new', 'Create new issue'),
                    ('!removerepo', 'rr, untrack, unwatch', 'Untrack repository'),
                ]
            },
            'admin': {
                'title': '⚙️ Admin Commands',
                'description': 'Administrative and debugging tools',
                'commands': [
                    ('!reindex', 'ri, refresh, rescan', 'Re-index file (admin)'),
                    ('!initdb', 'init, setup_db', 'Initialize database (admin)'),
                    ('!debugfile', 'df, debug_file, checkfile', 'Debug file indexing'),
                    ('!debugrepo', 'dr, debug_repo, checkrepo', 'Debug repo tracking'),
                    ('!debug_memories', 'dm, debug_mem, checkmem', 'Debug memories'),
                    ('!namespace_info', 'ns, namespace, nsinfo', 'Vector DB info'),
                ]
            }
        }

        if category and category.lower() in categories:
            # Show specific category
            cat_data = categories[category.lower()]
            embed = discord.Embed(
                title=cat_data['title'],
                description=cat_data['description'],
                color=0x0099ff
            )

            for cmd, aliases, desc in cat_data['commands']:
                # Format field with command and aliases
                field_name = f"{cmd}"
                if aliases:
                    field_name += f" ({aliases})"

                embed.add_field(
                    name=field_name,
                    value=desc,
                    inline=False
                )

            embed.set_footer(text="Use !help to see all categories | Aliases shown in parentheses")

        else:
            # Show overview
            embed = discord.Embed(
                title="🤖 QQL Pair Programmer Help",
                description="Your AI-powered coding assistant with memory and document analysis",
                color=0x00ff00
            )

            # Quick start with aliases
            embed.add_field(
                name="🚀 Quick Start",
                value=(
                    "`!q How do I use async in Python?` - Quick AI chat\n"
                    "`!up \"API docs\"` + attach file - Upload document\n"
                    "`!r project_stack Using React + PostgreSQL` - Save memory\n"
                    "`!track owner/repo` - Track GitHub repo\n"
                    "*Tip: Short aliases work for most commands!*"
                ),
                inline=False
            )

            # Categories
            embed.add_field(
                name="📚 Command Categories",
                value=(
                    "**!help ai** - AI chat & conversation commands\n"
                    "**!help docs** - Document management commands\n"
                    "**!help memory** - Memory & context commands\n"
                    "**!help github** - GitHub integration commands\n"
                    "**!help admin** - Administrative commands"
                ),
                inline=False
            )

            # Most used commands with their popular aliases
            embed.add_field(
                name="⭐ Popular Commands & Shortcuts",
                value=(
                    "`!q` → Quick AI chat (alias for !quick)\n"
                    "`!ad` → Ask about document (alias for !askdoc)\n"
                    "`!sf` → Search files (alias for !searchfiles)\n"
                    "`!r` → Remember (alias for !remember)\n"
                    "`!cs` → Code search (alias for !codesearch)\n"
                    "`!h` → This help menu (alias for !help)"
                ),
                inline=False
            )

            embed.set_footer(text="Type !help [category] for detailed command info | Most commands have short aliases!")

        await ctx.send(embed=embed)

    @commands.command(name='commands_table', aliases=['ct', 'table', 'cmdtable'])
    async def list_all_commands(self, ctx):
        """
        Display all available commands in a compact table format.
        
        Shows a comprehensive reference of all bot commands organized by category
        in a compact, easy-to-scan table format across multiple pages. Includes
        command syntax and aliases for quick reference.
        
        Usage:
            !commands_table     # Show all commands in table format
            
        Aliases: ct, table, cmdtable
        
        Output:
            Displays 3 pages of commands:
            - Page 1: AI & Document commands
            - Page 2: Memory & GitHub commands  
            - Page 3: Admin & Debug commands
            
        Each entry shows:
            - Full command syntax
            - Available aliases
            - Brief parameter information
            
        Example:
            !commands_table
            # Shows: Multi-page table with all commands and their aliases
        """

        # Create pages for different command groups
        pages = []

        # Page 1: AI & Documents
        page1 = discord.Embed(
            title="📋 Command Reference (Page 1/3)",
            description="**AI & Document Commands**",
            color=0x0099ff
        )

        ai_cmds = """```
AI COMMANDS                      ALIASES
!chat [provider] [model] [msg]  c, ask, ai
!quick [message]                 q, gpt, ask4
!models                          model, listmodels
!search [query]                  s, find, lookup
!context [query]                 ctx, preview
!autosave                        as, auto

DOCUMENT COMMANDS                ALIASES
!upload [description]            up, add, attach
!files [@user]                   f, list, ls, docs
!searchfiles [query]             sf, searchdocs, fd
!askdoc [id] [question]          ad, question, qd
!fileinfo [id]                   fi, details
!getfile [id]                    get, download, dl
!papers [all/pdf/recent]         p, documents, pdfs
!deletefile [id]                 delete, rm, remove
```"""
        page1.add_field(name="Commands", value=ai_cmds, inline=False)
        pages.append(page1)

        # Page 2: Memory & GitHub
        page2 = discord.Embed(
            title="📋 Command Reference (Page 2/3)",
            description="**Memory & GitHub Commands**",
            color=0x0099ff
        )

        mem_git_cmds = """```
MEMORY COMMANDS                  ALIASES
!remember [tag] [desc]           r, save, mem, note
!recall [query]                  rc, find_memory
!get_memory [tag]                gm, getmem, memory
!list_memories                   lm, memories, mems
!forget [tag]                    fg, delmem
!stats                           st, statistics

GITHUB COMMANDS                  ALIASES
!address [repo_url]              add_repo, track, watch
!repos                           repositories, lr
!repoinfo [repo]                 ri, repo, about
!issues [repo] [state]           i, bugs, issue
!prs [repo] [state]              pr, pulls
!codesearch [repo] [query]       cs, code, grep
!createissue [repo] "title"      ci, newissue
!removerepo [repo]               rr, untrack
```"""
        page2.add_field(name="Commands", value=mem_git_cmds, inline=False)
        pages.append(page2)

        # Page 3: Admin & Debug
        page3 = discord.Embed(
            title="📋 Command Reference (Page 3/3)",
            description="**Admin & Debug Commands**",
            color=0x0099ff
        )

        admin_cmds = """```
ADMIN COMMANDS                   ALIASES
!reindex [id]                    ri, refresh, rescan
!initdb                          init, setup_db
!dropdb                          drop, reset_db
!stats                           st, statistics

DEBUG COMMANDS                   ALIASES
!debugfile [id]                  df, checkfile
!debugrepo [repo]                dr, checkrepo
!debug_memories                  dm, checkmem
!namespace_info                  ns, namespace

HELP COMMANDS                    ALIASES
!help [category]                 h, ?, commands
!tips                            tip, hints, howto
```"""
        page3.add_field(name="Commands", value=admin_cmds, inline=False)
        pages.append(page3)

        # Send all pages
        for i, page in enumerate(pages):
            page.set_footer(text=f"Page {i + 1}/{len(pages)} | Use aliases for faster commands!")
            await ctx.send(embed=page)

    @commands.command(name='tips', aliases=['tip', 'hints', 'howto'])
    async def show_tips(self, ctx):
        """
        Display helpful tips and best practices for using the bot effectively.
        
        Provides practical guidance on how to get the most out of the bot's
        features, including workflow recommendations, command shortcuts,
        and productivity tips.
        
        Usage:
            !tips               # Show all tips and best practices
            
        Aliases: tip, hints, howto
        
        Tips Include:
            - Quick command shortcuts and aliases
            - Effective memory usage strategies
            - Better search techniques
            - Document Q&A workflow
            - GitHub integration tips
            - AI conversation context tips
            - Power user workflows
            - Content refresh techniques
            
        Example:
            !tips
            # Shows: Comprehensive list of usage tips and best practices
        """

        embed = discord.Embed(
            title="💡 Pro Tips & Best Practices",
            description="Get the most out of your AI assistant",
            color=0xffd700
        )

        tips = [
            (
                "⚡ Quick Commands",
                "Use short aliases! `!q` instead of `!quick`, `!r` instead of `!remember`, `!cs` for `!codesearch`"
            ),
            (
                "🎯 Effective Memories",
                "Use descriptive tags: `!r auth_flow JWT tokens expire in 24h`"
            ),
            (
                "🔍 Better Search",
                "Be specific: `!sf authentication JWT` instead of just `auth`"
            ),
            (
                "📄 Document Q&A",
                "Upload first: `!up \"API docs\"` → then ask: `!ad [id] How do I authenticate?`"
            ),
            (
                "🐙 GitHub Context",
                "Track repos: `!track owner/repo` → search code: `!cs repo function_name`"
            ),
            (
                "💬 AI Context",
                "The bot remembers! Just use `!q` to continue conversations naturally."
            ),
            (
                "⚡ Power User Workflow",
                "1. `!track` repo → 2. `!up` docs → 3. `!r` key info → 4. `!q` with full context!"
            ),
            (
                "🔄 Refresh Content",
                "File changed? Use `!ri [id]` to re-index it for updated searches"
            )
        ]

        for title, tip in tips:
            embed.add_field(name=title, value=tip, inline=False)

        embed.set_footer(text="Need help? Type !h or !help for commands")
        await ctx.send(embed=embed)

    @commands.command(name='aliases', aliases=['a', 'shortcuts'])
    async def show_aliases(self, ctx, command: Optional[str] = None):
        """
        Display command aliases and shortcuts for faster bot usage.
        
        Shows either aliases for a specific command or a comprehensive list
        of the most useful command shortcuts organized by category. Helps
        users discover faster ways to interact with the bot.
        
        Args:
            command (Optional[str]): Specific command name to show aliases for.
                                   If not provided, shows most useful shortcuts.
        
        Usage:
            !aliases            # Show most useful shortcuts by category
            !aliases chat       # Show all aliases for the 'chat' command
            !aliases upload     # Show all aliases for the 'upload' command
            
        Aliases: a, shortcuts
        
        Categories (when no command specified):
            - Essential: Core commands like !q, !c, !h, !s
            - Documents: File management shortcuts like !up, !f, !sf, !ad
            - Memory: Memory system shortcuts like !r, !rc, !lm, !gm
            - GitHub: Repository shortcuts like !cs, !i, !pr, !lr
            
        Example:
            !aliases
            # Shows: Categorized list of most useful command shortcuts
            
            !aliases chat
            # Shows: All available aliases for the chat command (!c, !ask, !ai)
        """
        if command:
            # Show aliases for specific command
            cmd = self.bot.get_command(command)
            if not cmd:
                await ctx.send(f"❌ Command `{command}` not found")
                return

            embed = discord.Embed(
                title=f"Aliases for !{cmd.name}",
                color=0x0099ff
            )

            if cmd.aliases:
                alias_list = "\n".join([f"• !{alias}" for alias in cmd.aliases])
                embed.add_field(
                    name="Available Aliases",
                    value=f"```\n{alias_list}\n```",
                    inline=False
                )

                # Show usage examples
                examples = f"!{cmd.name} [args]\n"
                examples += "\n".join([f"!{alias} [args]" for alias in cmd.aliases[:3]])
                embed.add_field(
                    name="Usage Examples",
                    value=f"```\n{examples}\n```",
                    inline=False
                )
            else:
                embed.description = "This command has no aliases."

            await ctx.send(embed=embed)
        else:
            # Show most useful aliases
            embed = discord.Embed(
                title="🚀 Most Useful Command Shortcuts",
                description="Quick aliases to speed up your workflow",
                color=0x00ff00
            )

            shortcuts = {
                "Essential": [
                    ("!q", "!quick", "Quick GPT-4 chat"),
                    ("!c", "!chat", "Full AI chat with options"),
                    ("!h", "!help", "Show help menu"),
                    ("!s", "!search", "Search history"),
                ],
                "Documents": [
                    ("!up", "!upload", "Upload file"),
                    ("!f", "!files", "List files"),
                    ("!sf", "!searchfiles", "Search in files"),
                    ("!ad", "!askdoc", "Ask about document"),
                ],
                "Memory": [
                    ("!r", "!remember", "Save memory"),
                    ("!rc", "!recall", "Search memories"),
                    ("!lm", "!list_memories", "List all memories"),
                    ("!gm", "!get_memory", "Get specific memory"),
                ],
                "GitHub": [
                    ("!cs", "!codesearch", "Search code"),
                    ("!i", "!issues", "List issues"),
                    ("!pr", "!prs", "List pull requests"),
                    ("!lr", "!repos", "List repos"),
                ]
            }

            for category, cmds in shortcuts.items():
                value = "\n".join([f"`{alias}` → {full}" for alias, full, _ in cmds])
                embed.add_field(name=f"**{category}**", value=value, inline=True)

            embed.set_footer(text="Type !aliases [command] to see all aliases for a specific command")
            await ctx.send(embed=embed)


async def setup(bot):
    """
    Setup function to add the HelpSystem cog to the bot.
    
    This function replaces Discord.py's default help command with our
    custom help system and adds the HelpSystem cog to the bot.
    
    Args:
        bot (commands.Bot): The Discord bot instance to add the cog to
        
    Note:
        This removes the default Discord.py help command to avoid conflicts
        with our custom help system.
    """
    # Remove default help command
    bot.remove_command('help')
    await bot.add_cog(HelpSystem(bot))
