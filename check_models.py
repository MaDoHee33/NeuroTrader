
import asyncio
from ollama_client import OllamaClient

async def main():
    client = OllamaClient()
    try:
        print("Fetching model list...")
        models = await client.list_models()
        print(f"Found {len(models)} models:")
        for m in models:
            print(f"- {m.get('name')}")
            
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    asyncio.run(main())
