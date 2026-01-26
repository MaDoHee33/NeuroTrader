
import asyncio
from ollama_client import OllamaClient

# Configuration
MODEL_NAME = "qwen3-coder:480b-cloud"

async def main():
    client = OllamaClient()
    
    prompt = """
You are a High-Frequency Trading (HFT) System Architect.

The user asks: **"Can we separate the training system to run on a dedicated GPU server, distinct from the main system? Is this necessary for speed, or is there a better way?"**

Current Context:
- Method: Reinforcement Learning (PPO) via Stable Baselines3.
- Environment: CPU-based `TradingEnv` (Pandas/Numpy).
- Current bottleneck: Likely CPU (environment stepping) rather than GPU (gradient updates) because financial data is low-dimensional compared to images.

Please analyze and provide a recommendation:
1.  **Is Decoupled Training necessary?** (e.g. Training on Cloud GPU -> Export Model -> Inference on Local).
2.  **The "vectorization" bottleneck:** Explain why moving a Pandas-based Env to a standalone GPU server might NOT help unless we use GPU-accelerated Envs (like Isaac Gym/JAX).
3.  **Recommendation:** Should we:
    - A) Split into Client-Server (Training Server + Inference Client)?
    - B) Switch to GPU-accelerated libraries (JAX/CuPy) on the same machine?
    - C) Stick to current architecture but optimize Env (Multiprocessing)?

Provide a concise "Yes/No" verdict and a short architectural advice.
"""

    print(f"Consulting {MODEL_NAME} regarding GPU Architecture...")
    try:
        response = await client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"num_ctx": 8192}
        )
        
        print("Consultation Complete.")
        with open("gpu_architecture_advice.md", "w", encoding="utf-8") as f:
            f.write(response)
        print("Saved advice to gpu_architecture_advice.md")
        
    except Exception as e:
        print(f"Consultation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
