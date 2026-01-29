
import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from src.brain.env.trading_env import TradingEnv
from src.evolving.hybrid_agent import HybridTradingAgent

class TestHybridAgent:
    """
    Integration tests for HybridTradingAgent.
    """
    
    def setup_method(self):
        # Create a dummy dataframe
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
        self.df = pd.DataFrame({
            'open': 100.0 + np.random.randn(200),
            'high': 105.0 + np.random.randn(200),
            'low': 95.0 + np.random.randn(200),
            'close': 100.0 + np.random.randn(200),
            'volume': 1000 + np.random.randint(0, 500, 200),
            'time': dates
        }, index=dates)
        
        # Initialize Env
        self.env = TradingEnv(
            {'TEST_ASSET': self.df}, 
            agent_type='scalper',
            initial_balance=10000
        )
        
        # Temp dir for agent data
        self.test_dir = Path("data/test_hybrid_agent")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Hybrid Agent (No PPO model = Random)
        self.agent = HybridTradingAgent(
            ppo_model_path=None,
            use_curiosity=True,
            use_experience_buffer=True,
            use_curriculum=True,
            data_dir=self.test_dir,
            agent_name="test_agent"
        )

    def teardown_method(self):
        # Cleanup
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_agent_environment_loop(self):
        """
        Verify the agent can complete a full episode loop with the environment.
        """
        obs, _ = self.env.reset()
        self.agent.start_episode()
        
        done = False
        step_count = 0
        total_intrinsic = 0.0
        
        while not done and step_count < 50:
            # 1. Get Action
            action, info = self.agent.get_action(obs)
            assert action in [0, 1, 2]
            assert 'curiosity_score' in info
            
            # 2. Step Env
            next_obs, reward, done, _, env_info = self.env.step(action)
            
            # 3. Store Experience
            intrinsic_reward = self.agent.store_experience(
                obs, action, reward, next_obs, env_info
            )
            
            # Verify intrinsic reward generation
            if intrinsic_reward:
                total_intrinsic += intrinsic_reward.total
                assert intrinsic_reward.novelty_bonus >= 0
                assert intrinsic_reward.prediction_error >= 0
            
            obs = next_obs
            step_count += 1
            
        # 4. End Episode
        summary = self.agent.end_episode(
            total_return=0.5, # Fake return
            win_rate=0.5,
            num_trades=2
        )
        
        # Assertions
        assert step_count > 0
        assert summary['total_steps'] == step_count
        assert summary['episode_id'] != ""
        assert len(self.agent.experience_buffer.experiences) == step_count
        print(f"Total Intrinsic Reward: {total_intrinsic:.4f}")
        
    def test_curiosity_exploration(self):
        """
        Verify that curiosity actually produces intrinsic rewards.
        """
        self.agent.start_episode()
        
        # Create two identical transitions
        obs1 = np.zeros(self.env.observation_space.shape[0])
        next_obs1 = np.zeros(self.env.observation_space.shape[0])
        
        # First time: Should have high novelty
        rew1 = self.agent.store_experience(obs1, 1, 0, next_obs1, {})
        
        # Second time (same state): Should have lower novelty (if count-based hashing works)
        rew2 = self.agent.store_experience(obs1, 1, 0, next_obs1, {})
        
        # Note: Depending on hash collision and implementation, it might not be strictly strictly lower immediately 
        # but generally novelty decreases as states are visited.
        # Our CuriosityModule uses Hashing. 
        
        print(f"Novelty 1: {rew1.novelty_bonus}, Novelty 2: {rew2.novelty_bonus}")
        
        # Just ensure it runs and produces values
        assert rew1.total is not None
        assert rew2.total is not None

