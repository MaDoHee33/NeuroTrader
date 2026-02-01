import os
import asyncio
from pathlib import Path
from ollama_client import OllamaClient

# Configuration
SOURCE_DIR = Path("src")
MODEL_NAME = "qwen3-coder:480b-cloud" # The "DeepSeek-level" cloud model provided

async def collect_codebase(root_path: Path):
    """Iterate through src and collect all python code"""
    code_content = ""
    file_count = 0
    
    # Priority folders to ensure we capture core logic
    for root, dirs, files in os.walk(root_path):
        if "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    code_content += f"\n\n--- FILE: {file_path} ---\n\n"
                    code_content += content
                    file_count += 1
                    print(f"Read: {file_path}")
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    
    print(f"Total files collected: {file_count}")
    return code_content

async def main():
    client = OllamaClient()
    
    print(f"Collecting codebase from {SOURCE_DIR}...")
    full_code = await collect_codebase(SOURCE_DIR)
    
    if not full_code:
        print("No code found!")
        return

    print(f"Codebase size: {len(full_code)} characters")
    
    prompt = f"""
You are a Principal Software Architect and AI Trading Expert.
I am providing you with the SOURCE CODE of a Python-based Algotrading system named 'NeuroTrader'.

Your Task:
1. **Architectural Review**: Analyze the structure `src/brain` (logic) vs `src/body` (execution). Is it decoupled correctly?
2. **Logic Auditing**: Look at `RLAgent` and `TradingEnv`. Are there obvious bugs in how state/rewards are handled?
3. **Performance Bottlenecks**: Identify why training might be slow based on the code (e.g. redundant loops, heavy calculations in critical paths).
4. **Critical Recommendations**: Suggest 3 specific refactorings to make this professional-grade.

CODEBASE START:
{full_code}
CODEBASE END

Please provide a detailed Markdown report.
"""

    print(f"Sending to {MODEL_NAME}... (This may take a minute)")
    try:
        response = await client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"num_ctx": 128000} # Request large context
        )
        
        print("Analysis Complete.")
        with open("full_system_analysis.md", "w", encoding="utf-8") as f:
            f.write(response)
        print("Saved to full_system_analysis.md")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
