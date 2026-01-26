import asyncio
from ollama_client import OllamaClient

CONTEXT_RL_AGENT = """
class RLAgent:
    # ... (code for RLAgent as seen in verification)
    def __init__(self, config=None, model_path=None):
        self.history_size = 100 
        self.model = PPO.load(...)
        
    def process_bar(self, bar_dict):
        # ... logic to create 19 features including RSI, MACD, BB, etc.
        # Observation space is 19 floats.
        pass
"""

CONTEXT_TRAIN = """
# Training script using Stable Baselines3 PPO
# ...
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,      # Research optimal for trading
            gamma=0.99,                 # Long-term focus 
            gae_lambda=0.95,           
            clip_range=0.2,            
            ent_coef=0.01,             
            n_steps=2048,              
            batch_size=64,             
            n_epochs=10,               
        )
# ...
"""

PROMPT = f"""
You are an expert AI quant researcher. 
I am working on a trading bot 'NeuroTrader' located in `src/brain`. 
We are facing issues with training a SHORT-TERM trading model (Scalping/Intraday).
The training is too slow, unstable, and consumes too many resources.

Here is the current implementation context:
1. Agent (`RLAgent`): Uses 100-bar history, 19 technical features (RSI, MACD, BB, Stoch, etc.).
2. Model: PPO with MlpPolicy from StableBaselines3.
3. Protocol: 1M timesteps, n_steps=2048, batch_size=64.

{CONTEXT_RL_AGENT}

{CONTEXT_TRAIN}

Please analyze and provide a research report with specific solutions:
1. **Algorithmic Improvements**: How to make PPO converge faster for short-term trading? (e.g., Reward shaping, Feature selection, Normalization).
2. **Architecture**: Should we change MlpPolicy to LstmPolicy? Or use a different algorithm like SAC/TD3?
3. **Efficiency**: How to train faster with less resources? (Vectorized environments, Feature caching, skipping frames?)
4. **Stability**: How to prevent catastrophic forgetting or high variance in crypto markets?

Focus on "Low Resource, High Speed, Stable" solutions.
"""

async def main():
    client = OllamaClient()
    print("Sending research request to qwen3-coder:480b-cloud...")
    try:
        response = await client.generate(
            model="qwen3-coder:480b-cloud",
            prompt=PROMPT,
            stream=False
        )
        print("\n--- Research Report Generated ---")
        # print(response) -> Caused encoding error on Windows console
        
        # Save report
        with open("research_report_qwen.md", "w", encoding='utf-8') as f:
            f.write(response)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
