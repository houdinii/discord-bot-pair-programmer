import asyncio
import os

import discord
# Import cogs
from cogs.ai_chat import AIChatCog
from cogs.file_manager import FileManagerCog
from cogs.github_integration import GitHubCog
from cogs.memory import MemoryCog
from discord.ext import commands
from dotenv import load_dotenv

# Version 0.1.3
print("Starting Pair Programmer Bot Version 0.1.3...")

load_dotenv()


class PairProgrammerBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True

        super().__init__(
            command_prefix='!',
            intents=intents,
            description='AI Pair Programming Assistant'
        )

    async def setup_hook(self):
        # Load cogs
        await self.add_cog(AIChatCog(self))
        await self.add_cog(MemoryCog(self))
        await self.add_cog(FileManagerCog(self))
        await self.add_cog(GitHubCog(self))

        print(f"Bot is ready! Logged in as {self.user}")

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for programming discussions"
            )
        )


async def main():
    bot = PairProgrammerBot()
    async with bot:
        await bot.start(os.getenv('DISCORD_TOKEN'))


if __name__ == "__main__":
    asyncio.run(main())
