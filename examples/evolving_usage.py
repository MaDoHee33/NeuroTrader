"""
Self-Evolving AI Usage Examples
================================
Examples showing how to use the new Self-Evolving AI modules.

These examples use fake data to demonstrate functionality
without loading heavy models or data files.
"""

import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_curiosity_module():
    """
    Example: Using CuriosityModule for intrinsic motivation.
    
    The Curiosity Module provides additional rewards for:
    - Visiting novel states
    - Making unpredictable transitions
    - Discovering profitable patterns
    """
    print("\n" + "="*60)
    print("EXAMPLE: CuriosityModule")
    print("="*60)
    
    from src.evolving import CuriosityModule
    
    # Initialize with default settings
    cm = CuriosityModule(
        novelty_weight=0.3,
        prediction_weight=0.4,
        pattern_weight=0.3
    )
    
    print("\nSimulating 100 trading steps...")
    
    total_intrinsic = 0
    for step in range(100):
        # Fake observation (10 features)
        obs = np.random.randn(10).astype(np.float32)
        obs = obs * 0.5 + step * 0.01  # Add some trend
        
        # Fake next observation
        next_obs = obs + np.random.randn(10).astype(np.float32) * 0.1
        
        # Random action
        action = np.random.randint(0, 3)
        
        # Fake extrinsic reward
        ext_reward = np.random.randn() * 0.1
        
        # Get intrinsic reward
        intrinsic = cm.compute_intrinsic_reward(obs, action, next_obs, ext_reward)
        total_intrinsic += intrinsic.total
        
        if step % 25 == 0:
            print(f"  Step {step}: Intrinsic = {intrinsic.total:.4f}, "
                  f"Novelty = {intrinsic.novelty_bonus:.4f}")
    
    # Get statistics
    stats = cm.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Total Intrinsic Reward: {total_intrinsic:.4f}")
    print(f"  Unique States Seen: {stats['unique_states']}")
    print(f"  Curiosity Score: {stats['curiosity_score']:.4f}")
    print(f"  Patterns Found: {stats['patterns_found']}")


def example_experience_buffer():
    """
    Example: Using ExperienceBuffer for lifelong learning.
    
    The Experience Buffer stores trading experiences as structured "stories"
    that can be replayed for learning.
    """
    print("\n" + "="*60)
    print("EXAMPLE: ExperienceBuffer")
    print("="*60)
    
    from src.evolving import ExperienceBuffer
    import tempfile
    
    # Create temporary directory for persistence demo
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "experiences.json"
        
        eb = ExperienceBuffer(
            max_size=1000,
            save_path=save_path,
            auto_save_interval=50
        )
        
        print("\nAdding 50 experiences across different market regimes...")
        
        regimes = ['bull', 'bear', 'sideways', 'volatile']
        
        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = obs + np.random.randn(10).astype(np.float32) * 0.1
            
            # Simulate varying profitability
            pnl = np.random.randn() * 0.02
            
            eb.add(
                observation=obs,
                action=np.random.randint(0, 3),
                reward=pnl * 10,
                next_observation=next_obs,
                pnl=pnl,
                holding_time=np.random.randint(1, 20),
                market_regime=np.random.choice(regimes),
                lesson_tags=['test'] if pnl > 0 else []
            )
        
        # Sample a batch
        print("\n📦 Sampling experiences...")
        
        batch = eb.sample_batch(batch_size=5)
        print(f"  Random batch of 5: {len(batch)} experiences")
        
        profitable = eb.sample_batch(batch_size=5, only_profitable=True)
        print(f"  Profitable only: {len(profitable)} experiences")
        
        bull_exp = eb.sample_batch(batch_size=10, filter_regime='bull')
        print(f"  Bull market only: {len(bull_exp)} experiences")
        
        # Find similar experiences
        query_obs = np.random.randn(10).astype(np.float32)
        similar = eb.get_similar_experiences(query_obs, top_k=3)
        print(f"  Similar to query: {len(similar)} experiences")
        
        # Statistics
        stats = eb.get_stats()
        print(f"\n📊 Buffer Statistics:")
        print(f"  Total Added: {stats['total_added']}")
        print(f"  Profitable Ratio: {stats['profitable_ratio']:.2%}")
        print(f"  By Regime: {stats['by_regime_count']}")


def example_curriculum_manager():
    """
    Example: Using CurriculumManager for progressive difficulty.
    
    The Curriculum Manager tracks performance and adjusts the
    difficulty level of training.
    """
    print("\n" + "="*60)
    print("EXAMPLE: CurriculumManager")
    print("="*60)
    
    from src.evolving import CurriculumManager, DifficultyLevel
    
    cm = CurriculumManager(
        start_level=DifficultyLevel.EASY,
        allow_regression=True
    )
    
    print(f"\nStarting Level: {cm.get_current_level().name}")
    print(f"Config: {cm.get_current_config().description}")
    
    # Simulate episodes with improving performance
    print("\n🎮 Simulating 30 episodes...")
    
    for ep in range(30):
        # Performance improves over time
        win_rate = 0.3 + (ep / 30) * 0.3  # 0.3 → 0.6
        total_return = 0.02 + (ep / 30) * 0.08  # 0.02 → 0.10
        
        result = cm.record_episode(
            total_return=total_return,
            win_rate=win_rate,
            num_trades=10
        )
        
        if result['advanced']:
            print(f"  Episode {ep+1}: 🎉 ADVANCED to {result['new_level'].name}!")
        elif result['regressed']:
            print(f"  Episode {ep+1}: ⬇️ Regressed to {result['new_level'].name}")
    
    # Final report
    report = cm.get_progress_report()
    print(f"\n📊 Final Progress Report:")
    print(f"  Current Level: {report['current_level']}")
    print(f"  Total Episodes: {report['total_episodes']}")
    print(f"  Level History: {[h['level'] for h in report['level_history']]}")


def example_regime_detector():
    """
    Example: Using MarketRegimeDetector for adaptive strategy.
    
    The Regime Detector classifies market conditions in real-time.
    """
    print("\n" + "="*60)
    print("EXAMPLE: MarketRegimeDetector")
    print("="*60)
    
    from src.evolving import MarketRegimeDetector, MarketRegime
    
    detector = MarketRegimeDetector(
        trend_period=20,
        volatility_period=14
    )
    
    print("\n📈 Simulating different market conditions...")
    
    # Phase 1: Bull market
    print("\n  Phase 1: Uptrend")
    for i in range(30):
        price = 100 + i * 0.5 + np.random.randn() * 0.3
        regime, metrics = detector.update(price)
    print(f"    Detected: {regime.value} (trend: {metrics.trend_strength:.2f})")
    
    # Phase 2: Bear market
    print("\n  Phase 2: Downtrend")
    last_price = 100 + 30 * 0.5
    for i in range(30):
        price = last_price - i * 0.5 + np.random.randn() * 0.3
        regime, metrics = detector.update(price)
    print(f"    Detected: {regime.value} (trend: {metrics.trend_strength:.2f})")
    
    # Phase 3: Sideways
    print("\n  Phase 3: Consolidation")
    center = last_price - 30 * 0.5
    for i in range(30):
        price = center + np.sin(i * 0.3) * 2 + np.random.randn() * 0.2
        regime, metrics = detector.update(price)
    print(f"    Detected: {regime.value} (range_bound: {metrics.range_bound:.2f})")
    
    # Phase 4: Volatile
    print("\n  Phase 4: High volatility")
    for i in range(30):
        price = center + np.random.randn() * 5
        regime, metrics = detector.update(price)
    print(f"    Detected: {regime.value} (volatility: {metrics.volatility:.2f})")
    
    # Statistics
    stats = detector.get_stats()
    print(f"\n📊 Detection Statistics:")
    print(f"  Total Steps: {stats['total_steps']}")
    print(f"  Regime Changes: {stats['regime_changes']}")
    print(f"  Regime Counts: {stats['regime_counts']}")


def example_hybrid_agent_mock():
    """
    Example: HybridTradingAgent structure (without loading real model).
    
    This shows the agent's interface without heavy memory usage.
    """
    print("\n" + "="*60)
    print("EXAMPLE: HybridTradingAgent (Mock)")
    print("="*60)
    
    from src.evolving import HybridTradingAgent
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create agent without PPO model (will use random actions)
        agent = HybridTradingAgent(
            ppo_model_path=None,  # No model = random agent
            use_curiosity=True,
            use_experience_buffer=True,
            use_curriculum=True,
            curiosity_weight=0.1,
            data_dir=Path(tmpdir),
            agent_name="test_agent"
        )
        
        print("\n🤖 Agent initialized (no PPO model, using random actions)")
        
        # Simulate one episode
        agent.start_episode()
        
        print("\n🎮 Simulating episode with 50 steps...")
        
        for step in range(50):
            obs = np.random.randn(10).astype(np.float32)
            
            # Get action
            action, info = agent.get_action(obs)
            
            # Simulate environment response
            next_obs = obs + np.random.randn(10).astype(np.float32) * 0.1
            reward = np.random.randn() * 0.1
            
            # Store experience
            intrinsic = agent.store_experience(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                env_info={'pnl': reward * 0.01, 'holding_time': 5}
            )
            
            if step % 20 == 0:
                print(f"  Step {step}: Action={action}, Source={info['source']}, "
                      f"Intrinsic={intrinsic.total:.4f}")
        
        # End episode
        summary = agent.end_episode(
            total_return=0.05,
            win_rate=0.5,
            num_trades=10
        )
        
        print(f"\n📊 Episode Summary:")
        print(f"  Steps: {summary['total_steps']}")
        print(f"  Actions: {summary['action_counts']}")
        print(f"  Avg Reward: {summary['avg_reward']:.4f}")
        
        # Get agent stats
        stats = agent.get_stats()
        print(f"\n🧠 Agent Statistics:")
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Action Distribution: {stats['action_distribution']}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("SELF-EVOLVING AI MODULES - USAGE EXAMPLES")
    print("="*60)
    
    example_curiosity_module()
    example_experience_buffer()
    example_curriculum_manager()
    example_regime_detector()
    example_hybrid_agent_mock()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
