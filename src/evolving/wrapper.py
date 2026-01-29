
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple

from src.evolving.curiosity import CuriosityModule
from src.evolving.experience_buffer import ExperienceBuffer

class HybridRewardWrapper(gym.Wrapper):
    """
    Gym Wrapper to integrate Hybrid AI components (Curiosity + Memory) 
    directly into the training loop.
    
    Logic:
    1. Intercept `step()`
    2. Calculate Intrinsic Reward (Curiosity)
    3. Mix with Extrinsic Reward (PnL)
    4. Store transition in Experience Buffer
    5. Return mixed reward to PPO
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        curiosity: CuriosityModule, 
        buffer: ExperienceBuffer,
        extrinsic_weight: float = 1.0, 
        intrinsic_weight: float = 0.1
    ):
        super().__init__(env)
        self.curiosity = curiosity
        self.buffer = buffer
        self.extrinsic_weight = extrinsic_weight
        self.intrinsic_weight = intrinsic_weight
        
        self.last_obs = None
        self.episode_id = "init"
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        
        # Generate new episode ID for memory
        import uuid
        self.episode_id = str(uuid.uuid4())[:8]
        
        if self.curiosity:
            self.curiosity.reset_episode()
            
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Execute Action in Real Env
        next_obs, extrinsic_reward, done, truncated, info = self.env.step(action)
        
        # 2. Calculate Intrinsic Reward
        intrinsic_reward_val = 0.0
        if self.curiosity and self.last_obs is not None:
            # We assume action is scalar for discrete
            act = action.item() if hasattr(action, 'item') else action
            
            ir_obj = self.curiosity.compute_intrinsic_reward(
                self.last_obs, act, next_obs, extrinsic_reward
            )
            intrinsic_reward_val = ir_obj.total
            
            # Add to info for logging
            info['reward_intrinsic'] = intrinsic_reward_val
            info['reward_extrinsic'] = extrinsic_reward
            
        # 3. Mix Rewards
        # Total = (PnL * w_ex) + (Curiosity * w_in)
        total_reward = (extrinsic_reward * self.extrinsic_weight) + \
                       (intrinsic_reward_val * self.intrinsic_weight)
                       
        # 4. Store in Experience Buffer
        if self.buffer and self.last_obs is not None:
            act = action.item() if hasattr(action, 'item') else action
            
            # Extract metadata
            pnl = info.get('pnl', 0.0)
            trade_info = info.get('last_trade', {})
            
            lesson_tags = []
            if trade_info:
                 if trade_info.get('action') == 'SELL' and pnl > 0:
                     lesson_tags.append('profitable_exit')
            
            self.buffer.add(
                observation=self.last_obs,
                action=act,
                reward=total_reward, # Learn from mixed reward
                next_observation=next_obs,
                pnl=pnl,
                holding_time=info.get('holding_time', 0),
                episode_id=self.episode_id,
                lesson_tags=lesson_tags
            )
            
        # Update state
        self.last_obs = next_obs
        
        return next_obs, total_reward, done, truncated, info
