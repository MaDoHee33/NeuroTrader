import discord
import logging
import asyncio
from discord.ext import commands

class DiscordBot:
    def __init__(self, config=None):
        self.logger = logging.getLogger("Discord")
        self.config = config or {}
        secrets = self.config.get('secrets', {})
        
        # Try snake_case first, then space separated
        self.token = secrets.get('discord_token') or secrets.get('discord token')
        self.channel_id = secrets.get('discord_channel_id') or secrets.get('discord channel id')
        
        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)
        self.is_ready = False
        
        # Background task loop handled by client
        self.loop = asyncio.get_event_loop()

    async def start(self):
        """Starts the Discord bot in the background."""
        if not self.token:
            self.logger.warning("No Discord Token provided. Bot disabled.")
            return

        @self.client.event
        async def on_ready():
            self.logger.info(f"Logged in as {self.client.user}")
            self.is_ready = True
            
        # Run client without blocking main loop
        asyncio.create_task(self.client.start(self.token))

    async def send(self, message):
        """Sends a message to the default channel."""
        if not self.token or not self.channel_id:
            self.logger.warning(f"Discord disabled or missing channel ID. Msg: {message}")
            return

        try:
            if not self.is_ready:
                # Wait briefly or just log
                # self.logger.warning("Discord not ready yet.")
                pass
                
            channel = self.client.get_channel(int(self.channel_id))
            if channel:
                await channel.send(message)
            else:
                self.logger.warning("Discord channel not found (Check ID).")
        except Exception as e:
            self.logger.error(f"Failed to send Discord msg: {e}")
