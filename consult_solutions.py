
import asyncio
from pathlib import Path
from ollama_client import OllamaClient

# Configuration
MODEL_NAME = "qwen3-coder:480b-cloud"
ANALYSIS_FILE = Path("full_system_analysis.md")

async def main():
    client = OllamaClient()
    
    if not ANALYSIS_FILE.exists():
        print("Error: Previous analysis not found!")
        return

    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        previous_analysis = f.read()

    prompt = f"""
You are the Principal Architect for NeuroTrader. 
You previously performed an audit (context provided below) and found CRITICAL issues:
1. "Schizophrenic Logic" (Feature Mismatch between Training vs Inference).
2. "Memory Leak/Latency" (Inefficient full-history recalculation).

The user asks: **"How do we fix this and prevent it from happening again?"**

Please provide a **Strategic Refactoring & Prevention Plan**:

### Part 1: The Fix (Code Patterns)
- Show a Python code sketch of a **"Unified Feature Engine"** that works for BOTH:
  - **Batch Mode** (Training on historical DataFrames).
  - **Streaming Mode** (Inference on single ticks, O(1) complexity).
- Essential: It must use the EXACT same logic for both to guarantee consistency.

### Part 2: The Prevention (Guardrails)
- What specific **Unit Tests** should exist to catch this mismatch automatically?
- Define a **"Golden Rule"** for adding new features (e.g., "If it's not in the FeatureRegistry, it doesn't exist").

### Part 3: Architecture Standard
- How do we structure the "Brain" so `RLAgent` and `TradingEnv` never diverge again?

---
CONTEXT (PREVIOUS ANALYSIS):
{previous_analysis}
"""

    print(f"Consulting {MODEL_NAME} for solutions...")
    try:
        response = await client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"num_ctx": 32000}
        )
        
        print("Consultation Complete.")
        output_file = "deepseek_solutions.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Saved solutions to {output_file}")
        
    except Exception as e:
        print(f"Consultation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
