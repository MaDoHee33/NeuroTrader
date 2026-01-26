#!/usr/bin/env python3
"""
Script to inspect PPO model metadata and verify training steps.

Usage:
    python inspect_model.py <model_path.zip>
    
Example:
    python inspect_model.py ppo_neurotrader.zip
    python inspect_model.py ppo_checkpoint_10000000_steps.zip
"""

import sys
import os
from pathlib import Path
import zipfile
import json
from datetime import datetime


def inspect_model(model_path: str):
    """
    Inspect a Stable-Baselines3 model file and show metadata.
    
    Args:
        model_path: Path to .zip model file
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ File not found: {model_path}")
        return
    
    print("="*70)
    print(f"🔍 Inspecting Model: {model_path.name}")
    print("="*70)
    
    # Basic file info
    file_size = model_path.stat().st_size
    file_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    
    print(f"\n📁 File Information:")
    print(f"   Path:         {model_path.absolute()}")
    print(f"   Size:         {file_size / 1024:.1f} KB ({file_size:,} bytes)")
    print(f"   Modified:     {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Inspect ZIP contents
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            print(f"\n📦 ZIP Contents:")
            for info in zf.filelist:
                print(f"   - {info.filename:30s} ({info.file_size:>10,} bytes)")
            
            # Try to read metadata
            if 'data' in zf.namelist():
                print(f"\n📊 Model Metadata:")
                
                # Load the model using Stable-Baselines3
                try:
                    from stable_baselines3 import PPO
                    import torch
                    
                    # Load model
                    model = PPO.load(str(model_path), device='cpu')
                    
                    # Get training steps
                    num_timesteps = model.num_timesteps
                    
                    print(f"   ✅ Successfully loaded model")
                    print(f"   🎯 Training Steps: {num_timesteps:,}")
                    print(f"   🎯 Steps (M):      {num_timesteps / 1_000_000:.1f}M")
                    
                    # Policy info
                    print(f"\n🧠 Policy Information:")
                    print(f"   Type:          {type(model.policy).__name__}")
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in model.policy.parameters())
                    print(f"   Parameters:    {total_params:,}")
                    
                    # Learning rate
                    print(f"   Learning Rate: {model.learning_rate}")
                    
                    # Observation/Action space
                    print(f"\n📐 Spaces:")
                    print(f"   Observation:   {model.observation_space}")
                    print(f"   Action:        {model.action_space}")
                    
                except ImportError:
                    print("   ⚠️  stable-baselines3 not available")
                    print("   Cannot load model metadata")
                except Exception as e:
                    print(f"   ⚠️  Error loading model: {e}")
            
    except zipfile.BadZipFile:
        print(f"\n❌ Invalid ZIP file: {model_path}")
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
    
    print("\n" + "="*70)


def compare_models(*model_paths):
    """Compare multiple models side by side."""
    from stable_baselines3 import PPO
    
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    
    results = []
    
    for path in model_paths:
        path = Path(path)
        if not path.exists():
            print(f"⚠️  Skipping {path.name} (not found)")
            continue
        
        try:
            model = PPO.load(str(path), device='cpu')
            file_size = path.stat().st_size / 1024
            modified = datetime.fromtimestamp(path.stat().st_mtime)
            
            results.append({
                'name': path.name,
                'steps': model.num_timesteps,
                'size_kb': file_size,
                'modified': modified
            })
        except Exception as e:
            print(f"⚠️  Error loading {path.name}: {e}")
    
    if results:
        # Sort by steps
        results.sort(key=lambda x: x['steps'], reverse=True)
        
        print(f"\n{'Model':<40} {'Steps':>12} {'Size (KB)':>10} {'Modified':>20}")
        print("-" * 85)
        
        for r in results:
            steps_m = f"{r['steps']/1e6:.1f}M"
            modified_str = r['modified'].strftime('%Y-%m-%d %H:%M')
            print(f"{r['name']:<40} {steps_m:>12} {r['size_kb']:>10.1f} {modified_str:>20}")
        
        # Highlight the best
        best = results[0]
        print(f"\n✅ Latest/Best Model: {best['name']}")
        print(f"   Training Steps: {best['steps']:,} ({best['steps']/1e6:.1f}M)")
    else:
        print("\n❌ No valid models found")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single model:    python inspect_model.py <model.zip>")
        print("  Compare models:  python inspect_model.py model1.zip model2.zip ...")
        print("\nExample:")
        print("  python inspect_model.py ppo_neurotrader.zip")
        print("  python inspect_model.py ppo_*.zip")
        sys.exit(1)
    
    model_paths = sys.argv[1:]
    
    if len(model_paths) == 1:
        # Single model inspection
        inspect_model(model_paths[0])
    else:
        # Compare multiple models
        for path in model_paths:
            inspect_model(path)
        
        print("\n")
        compare_models(*model_paths)


if __name__ == "__main__":
    main()
