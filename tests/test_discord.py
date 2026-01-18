import discord
import asyncio
import yaml
import os

async def test_discord():
    print(f"🔍 DIAGNOSTIC: Testing Discord Connection...")
    print(f"   CWD: {os.getcwd()}")
    
    # 1. Load Secrets
    secret_path = "config/secrets.yaml"
    abs_path = os.path.abspath(secret_path)
    print(f"   Looking for secrets at: {abs_path}")
    
    if not os.path.exists(secret_path):
        print(f"❌ ERROR: File {secret_path} not found!")
        # Try looking in tools/ just in case
        if os.path.exists("tools/secrets.yaml"):
             print("   ⚠️ Found secrets.yaml in tools/! Please move it to config/.")
        return

    try:
        with open(secret_path, 'r') as f:
            secrets = yaml.safe_load(f)
            print("✅ Secrets file loaded.")
    except Exception as e:
        print(f"❌ ERROR: Could not parse YAML: {e}")
        return

    # 2. Extract Token
    token = secrets.get('discord_token') or secrets.get('discord token')
    channel_id = secrets.get('discord_channel_id') or secrets.get('discord channel id')

    if not token:
        print("❌ ERROR: 'discord_token' NOT found in secrets.yaml")
        print(f"   Found keys: {list(secrets.keys())}")
        return
    else:
        print(f"✅ Token found: {token[:5]}...{token[-5:]}")

    if not channel_id:
        print("❌ ERROR: 'discord_channel_id' NOT found in secrets.yaml")
        return
    else:
        print(f"✅ Channel ID found: {channel_id}")

    # 3. Connect
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"✅ CONNECTED as: {client.user}")
        print(f"   Bot ID: {client.user.id}")
        
        try:
            channel = client.get_channel(int(channel_id))
            if channel:
                print(f"✅ Channel found: {channel.name} (ID: {channel.id})")
                await channel.send("🧪 DIAGNOSTIC TEST: Discord connection is working! 🚀")
                print("✅ Message sent successfully.")
            else:
                print(f"❌ ERROR: Channel ID {channel_id} not reachable by this bot.")
                print("   Check: 1. Bot is in the server? 2. Bot has 'View Channel' & 'Send Messages' permissions?")
        except Exception as e:
            print(f"❌ ERROR sending message: {e}")
        
        await client.close()

    @client.event
    async def on_error(event, *args, **kwargs):
        print(f"❌ ERROR in {event}: {args}")
        await client.close()

    print("⏳ Connecting to Discord gateway...")
    try:
        await client.start(token)
    except discord.errors.LoginFailure:
        print("❌ ERROR: Login Failed. Invalid Token?")
    except Exception as e:
        print(f"❌ ERROR: Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_discord())
