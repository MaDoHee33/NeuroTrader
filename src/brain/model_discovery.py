"""
Smart Model Discovery for NeuroTrader

Automatically finds and selects the best trained model without manual specification.
"""

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def find_best_model(
    models_dir: str = "models",
    workspace: Optional[Path] = None,
    criteria: str = "training_steps"
) -> Optional[Path]:
    """
    Automatically find the best trained model.
    
    Args:
        models_dir: Directory containing models (default: "models")
        workspace: Workspace path (for Colab)
        criteria: Selection criteria:
            - "training_steps": Choose model with most training steps
            - "latest": Choose most recently modified
            - "auto": Try training_steps, fallback to latest
    
    Returns:
        Path to best model, or None if no models found
    
    Example:
        >>> model_path = find_best_model()
        >>> print(f"Using: {model_path}")
        Using: models/ppo_neurotrader.zip (10.0M steps)
    """
    from stable_baselines3 import PPO
    
    # Determine search directory
    if workspace:
        search_dir = workspace / models_dir
    else:
        search_dir = Path(models_dir)
    
    if not search_dir.exists():
        logger.warning(f"Models directory not found: {search_dir}")
        return None
    
    # Find all .zip files
    model_files = list(search_dir.glob("*.zip"))
    
    # Also check checkpoints subdirectory
    checkpoint_dir = search_dir / "checkpoints"
    if checkpoint_dir.exists():
        model_files.extend(checkpoint_dir.glob("*.zip"))
    
    if not model_files:
        logger.warning(f"No model files found in {search_dir}")
        return None
    
    logger.info(f"🔍 Found {len(model_files)} model file(s)")
    
    # Inspect each model
    candidates = []
    
    for model_path in model_files:
        try:
            # Try to load and get training steps
            model = PPO.load(str(model_path), device='cpu')
            num_steps = model.num_timesteps
            modified = model_path.stat().st_mtime
            size = model_path.stat().st_size
            
            candidates.append({
                'path': model_path,
                'steps': num_steps,
                'modified': modified,
                'size': size,
                'name': model_path.name
            })
            
            logger.debug(f"  ✅ {model_path.name}: {num_steps:,} steps")
            
        except Exception as e:
            logger.debug(f"  ⚠️  {model_path.name}: Could not load ({e})")
            continue
    
    if not candidates:
        logger.warning("No valid models found")
        return None
    
    # Select best based on criteria
    if criteria == "training_steps" or criteria == "auto":
        # Sort by training steps (descending)
        candidates.sort(key=lambda x: x['steps'], reverse=True)
        best = candidates[0]
        
        logger.info(f"📊 Selection Criteria: Training Steps")
        
    elif criteria == "latest":
        # Sort by modification time (most recent first)
        candidates.sort(key=lambda x: x['modified'], reverse=True)
        best = candidates[0]
        
        logger.info(f"📊 Selection Criteria: Latest Modified")
    
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    # Show selection
    logger.info(f"✅ Selected Model:")
    logger.info(f"   Path:   {best['path']}")
    logger.info(f"   Steps:  {best['steps']:,} ({best['steps']/1e6:.1f}M)")
    logger.info(f"   Date:   {datetime.fromtimestamp(best['modified']).strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"   Size:   {best['size']/1024:.1f} KB")
    
    # Show alternatives
    if len(candidates) > 1:
        logger.info(f"\n📋 Other Models Available:")
        for i, candidate in enumerate(candidates[1:4], 1):  # Show top 3 alternatives
            logger.info(
                f"   {i}. {candidate['name']}: "
                f"{candidate['steps']:,} steps, "
                f"{datetime.fromtimestamp(candidate['modified']).strftime('%Y-%m-%d %H:%M')}"
            )
    
    return best['path']


def get_model_info(model_path: Path) -> Dict:
    """Get detailed information about a model."""
    from stable_baselines3 import PPO
    
    model = PPO.load(str(model_path), device='cpu')
    stat = model_path.stat()
    
    return {
        'path': str(model_path),
        'name': model_path.name,
        'training_steps': model.num_timesteps,
        'steps_millions': model.num_timesteps / 1e6,
        'size_kb': stat.st_size / 1024,
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'modified_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }


def list_available_models(models_dir: str = "models", workspace: Optional[Path] = None) -> List[Dict]:
    """
    List all available models with their details.
    
    Returns:
        List of model information dictionaries
    """
    from stable_baselines3 import PPO
    
    # Determine search directory
    if workspace:
        search_dir = workspace / models_dir
    else:
        search_dir = Path(models_dir)
    
    if not search_dir.exists():
        return []
    
    # Find all .zip files
    model_files = list(search_dir.glob("*.zip"))
    
    checkpoint_dir = search_dir / "checkpoints"
    if checkpoint_dir.exists():
        model_files.extend(checkpoint_dir.glob("*.zip"))
    
    models = []
    for model_path in model_files:
        try:
            info = get_model_info(model_path)
            models.append(info)
        except Exception as e:
            logger.debug(f"Could not load {model_path.name}: {e}")
            continue
    
    # Sort by training steps (descending)
    models.sort(key=lambda x: x['training_steps'], reverse=True)
    
    return models


# Convenience function
def auto_load_best_model(models_dir: str = "models", workspace: Optional[Path] = None):
    """
    Automatically find and load the best model.
    
    Returns:
        Loaded PPO model
    
    Example:
        >>> model = auto_load_best_model()
        🔍 Found 3 model file(s)
        ✅ Selected Model: ppo_neurotrader.zip (10.0M steps)
    """
    from stable_baselines3 import PPO
    
    best_path = find_best_model(models_dir, workspace)
    
    if best_path is None:
        raise FileNotFoundError("No valid models found")
    
    print(f"📦 Loading model: {best_path.name}")
    model = PPO.load(str(best_path))
    print(f"✅ Model loaded successfully!")
    
    return model


if __name__ == "__main__":
    # Test the smart discovery
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("🔍 Smart Model Discovery Test")
    print("="*60)
    
    # Find best model
    best = find_best_model(criteria="training_steps")
    
    if best:
        print(f"\n✅ Best Model Found: {best}")
        
        # Show all models
        print(f"\n{'='*60}")
        print("📋 All Available Models:")
        print(f"{'='*60}")
        
        models = list_available_models()
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Steps: {model['training_steps']:,} ({model['steps_millions']:.1f}M)")
            print(f"   Size:  {model['size_kb']:.1f} KB")
            print(f"   Date:  {model['modified_str']}")
            print()
    else:
        print("\n❌ No models found")
