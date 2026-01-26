"""
NeuroTrader Training Orchestrator
==================================
Central controller for automated training pipelines.

Features:
- Config-driven training
- Auto-resume from checkpoints
- Multi-role training
- Integration with all skills (Registry, Evaluator, Notifier)
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.skills.model_registry import ModelRegistry
from src.skills.auto_evaluator import AutoEvaluator
from src.skills.notifier import Notifier, EventType, TrainingEvent


@dataclass
class TrainingJob:
    """A single training job configuration."""
    role: str
    symbol: str
    timeframe: str
    total_steps: int
    hyperparams: Dict[str, Any]
    data_path: str
    checkpoint_freq: int = 100000
    auto_backtest: bool = True
    auto_promote: bool = True
    
    @property
    def job_id(self) -> str:
        return f"{self.role}_{self.symbol}_{self.timeframe}"


class TrainingOrchestrator:
    """
    Central training controller.
    
    Usage:
        orchestrator = TrainingOrchestrator("config/training_config.yaml")
        
        # Train all roles
        orchestrator.train_all()
        
        # Train specific role
        orchestrator.train_role("scalper")
        
        # Resume interrupted training
        orchestrator.resume()
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        models_dir = self.config.get('output', {}).get('models_dir', 'models')
        reports_dir = self.config.get('output', {}).get('reports_dir', 'reports')
        
        self.registry = ModelRegistry(models_dir)
        self.evaluator = AutoEvaluator(models_dir, reports_dir)
        self.notifier = Notifier()
        
        # State tracking
        self.active_jobs: List[TrainingJob] = []
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"[WARNING] Config not found: {self.config_path}")
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'defaults': {
                'total_steps': 1000000,
                'checkpoint_every': 100000,
                'auto_backtest': True,
                'auto_promote': True
            },
            'roles': {
                'scalper': {
                    'timeframes': ['M5'],
                    'symbols': ['XAUUSD'],
                    'hyperparams': {'gamma': 0.85, 'learning_rate': 3e-4}
                }
            },
            'data': {
                'base_dir': 'data/processed',
                'pattern': '{symbol}_{timeframe}_processed.parquet'
            }
        }
    
    def _build_data_path(self, symbol: str, timeframe: str) -> str:
        """Build data file path from config pattern."""
        base_dir = self.config.get('data', {}).get('base_dir', 'data/processed')
        pattern = self.config.get('data', {}).get('pattern', '{symbol}_{timeframe}_processed.parquet')
        
        filename = pattern.format(symbol=symbol, timeframe=timeframe)
        return str(ROOT_DIR / base_dir / filename)
    
    def _create_jobs_for_role(self, role: str) -> List[TrainingJob]:
        """Create training jobs for a role based on config."""
        role_config = self.config.get('roles', {}).get(role, {})
        defaults = self.config.get('defaults', {})
        
        jobs = []
        
        symbols = role_config.get('symbols', ['XAUUSD'])
        timeframes = role_config.get('timeframes', ['M5'])
        hyperparams = role_config.get('hyperparams', {})
        
        for symbol in symbols:
            for tf in timeframes:
                data_path = self._build_data_path(symbol, tf)
                
                if not Path(data_path).exists():
                    print(f"[WARNING] Data not found: {data_path}")
                    continue
                
                job = TrainingJob(
                    role=role,
                    symbol=symbol,
                    timeframe=tf,
                    total_steps=defaults.get('total_steps', 1000000),
                    hyperparams=hyperparams,
                    data_path=data_path,
                    checkpoint_freq=defaults.get('checkpoint_every', 100000),
                    auto_backtest=defaults.get('auto_backtest', True),
                    auto_promote=defaults.get('auto_promote', True)
                )
                jobs.append(job)
        
        return jobs
    
    def train_role(self, role: str, resume: bool = True) -> bool:
        """
        Train all configurations for a specific role.
        
        Args:
            role: Agent role (scalper/swing/trend)
            resume: Resume from checkpoint if available
            
        Returns:
            True if all jobs completed successfully
        """
        print(f"\n{'='*60}")
        print(f"[TARGET] TRAINING ORCHESTRATOR: {role.upper()}")
        print(f"{'='*60}")
        
        jobs = self._create_jobs_for_role(role)
        
        if not jobs:
            print(f"[FAILED] No valid jobs found for {role}")
            return False
        
        print(f"[INFO] Created {len(jobs)} training job(s)")
        for job in jobs:
            print(f"   * {job.job_id}")
        
        all_success = True
        
        for job in jobs:
            success = self._execute_job(job, resume=resume)
            
            if success:
                self.completed_jobs.append(job.job_id)
            else:
                self.failed_jobs.append(job.job_id)
                all_success = False
        
        # Print summary
        self._print_summary()
        
        return all_success
    
    def train_all(self, resume: bool = True) -> bool:
        """Train all roles defined in config."""
        roles = list(self.config.get('roles', {}).keys())
        
        print(f"\n{'='*60}")
        print(f"[START] FULL TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Roles to train: {', '.join(r.upper() for r in roles)}")
        
        all_success = True
        
        for role in roles:
            if not self.train_role(role, resume=resume):
                all_success = False
        
        return all_success
    
    def _execute_job(self, job: TrainingJob, resume: bool = True) -> bool:
        """Execute a single training job."""
        print(f"\n{'-'*40}")
        print(f"[START] Starting: {job.job_id}")
        print(f"{'-'*40}")
        
        # Send notification
        self.notifier.training_started(
            role=job.role,
            symbol=job.symbol,
            timeframe=job.timeframe,
            steps=job.total_steps
        )
        
        try:
            # Import training function
            from scripts.train_trinity import train_trinity
            
            # Run training
            train_trinity(
                role=job.role,
                data_path=job.data_path,
                total_timesteps=job.total_steps,
                resume=resume,
                register=True,
                checkpoint_freq=job.checkpoint_freq
            )
            
            print(f"[SUCCESS] Job completed: {job.job_id}")
            return True
            
        except KeyboardInterrupt:
            print(f"[WARNING] Job interrupted: {job.job_id}")
            return False
            
        except Exception as e:
            print(f"[FAILED] Job failed: {job.job_id}")
            print(f"   Error: {e}")
            
            self.notifier.training_failed(
                role=job.role,
                symbol=job.symbol,
                timeframe=job.timeframe,
                error=str(e)
            )
            return False
    
    def resume(self) -> bool:
        """Resume any interrupted training jobs."""
        print("\n[CHECK] Checking for interrupted jobs...")
        
        checkpoints_dir = ROOT_DIR / self.config.get('output', {}).get('checkpoints_dir', 'models/checkpoints')
        
        if not checkpoints_dir.exists():
            print("   No checkpoints found")
            return True
        
        found_jobs = []
        
        for job_dir in checkpoints_dir.iterdir():
            if not job_dir.is_dir():
                continue
            
            state_file = job_dir / "training_state.json"
            if state_file.exists():
                import json
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                parts = job_dir.name.split('_')
                if len(parts) >= 3:
                    found_jobs.append({
                        'role': parts[0],
                        'symbol': parts[1],
                        'timeframe': parts[2],
                        'steps': state.get('steps_completed', 0)
                    })
        
        if not found_jobs:
            print("   No interrupted jobs found")
            return True
        
        print(f"[INFO] Found {len(found_jobs)} interrupted job(s):")
        for job in found_jobs:
            print(f"   * {job['role']}_{job['symbol']}_{job['timeframe']} ({job['steps']:,} steps)")
        
        # Resume each job
        for job in found_jobs:
            self.train_role(job['role'], resume=True)
        
        return True
    
    def status(self) -> Dict[str, Any]:
        """Get current training status."""
        status = {
            'completed': self.completed_jobs,
            'failed': self.failed_jobs,
            'active': [j.job_id for j in self.active_jobs]
        }
        
        # Check registry for models
        for role in ['scalper', 'swing', 'trend']:
            best = self.registry.get_best(role)
            if best:
                status[f'{role}_best'] = {
                    'version': best.get('version'),
                    'metrics': best.get('metrics', {})
                }
        
        return status
    
    def _print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("[INFO] TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"[SUCCESS] Completed: {len(self.completed_jobs)}")
        for job_id in self.completed_jobs:
            print(f"   * {job_id}")
        
        if self.failed_jobs:
            print(f"[FAILED] Failed: {len(self.failed_jobs)}")
            for job_id in self.failed_jobs:
                print(f"   * {job_id}")
        
        print(f"{'='*60}")


# CLI
def main():
    parser = argparse.ArgumentParser(description="NeuroTrader Training Orchestrator")
    parser.add_argument('command', choices=['train', 'resume', 'status'],
                       help="Command to execute")
    parser.add_argument('--role', type=str, choices=['scalper', 'swing', 'trend', 'all'],
                       default='all', help="Role to train")
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help="Path to config file")
    parser.add_argument('--no-resume', action='store_true',
                       help="Start fresh, don't resume")
    
    args = parser.parse_args()
    
    orchestrator = TrainingOrchestrator(args.config)
    
    if args.command == 'train':
        if args.role == 'all':
            orchestrator.train_all(resume=not args.no_resume)
        else:
            orchestrator.train_role(args.role, resume=not args.no_resume)
            
    elif args.command == 'resume':
        orchestrator.resume()
        
    elif args.command == 'status':
        import json
        status = orchestrator.status()
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
