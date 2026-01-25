import asyncio
from ollama_client import OllamaClient

async def main():
    client = OllamaClient()
    
    print("Checking server status...")
    is_running = await client.is_server_running()
    print(f"Server running: {is_running}")
    
    if is_running:
        print("\nListing models...")
        models = await client.list_models()
        for m in models:
            print(f"- {m['name']}")
            
        print("\nTesting generation (qwen3-coder:480b-cloud)...")
        try:
            response = await client.generate(
                model="qwen3-coder:480b-cloud",
                prompt="Hi"
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
