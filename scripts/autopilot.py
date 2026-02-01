"""
NeuroTrader AutoPilot CLI
=========================
Unified command-line interface for automated training operations.

Commands:
    train       - Train models (single role or all)
    resume      - Resume interrupted training
    evaluate    - Evaluate a trained model
    compare     - Compare model versions
    status      - Show training status
    tune        - Run hyperparameter tuning
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def cmd_train(args):
    """Handle train command."""
    from src.skills.training_orchestrator import TrainingOrchestrator
    
    orchestrator = TrainingOrchestrator(args.config)
    
    if args.role == 'all':
        orchestrator.train_all(resume=not args.fresh)
    else:
        orchestrator.train_role(args.role, resume=not args.fresh)


def cmd_resume(args):
    """Handle resume command."""
    from src.skills.training_orchestrator import TrainingOrchestrator
    
    orchestrator = TrainingOrchestrator(args.config)
    orchestrator.resume()


def cmd_evaluate(args):
    """Handle evaluate command."""
    from src.skills.auto_evaluator import AutoEvaluator
    
    evaluator = AutoEvaluator()
    metrics = evaluator.evaluate_model(
        model_path=args.model,
        data_path=args.data,
        role=args.role,
        use_test_set=not args.all_data
    )
    
    if args.json:
        import json
        print(json.dumps(metrics, indent=2))


def cmd_compare(args):
    """Handle compare command."""
    from src.skills.model_registry import ModelRegistry
    
    registry = ModelRegistry("models")
    
    if args.versions:
        versions = [int(v) for v in args.versions.split(',')]
        print(registry.compare(args.role, versions))
    else:
        print(registry.compare(args.role))


def cmd_status(args):
    """Handle status command."""
    from src.skills.model_registry import ModelRegistry
    
    registry = ModelRegistry("models")
    
    print("\n" + "="*60)
    print("[*] NEUROTRADER STATUS")
    print("="*60)
    
    for role in ['scalper', 'swing', 'trend']:
        best = registry.get_best(role)
        versions = registry.list_versions(role)
        
        print(f"\n[+] {role.upper()}")
        print(f"   Versions: {len(versions)}")
        
        if best:
            metrics = best.get('metrics', {})
            print(f"   Best: v{best.get('version', '?'):03d}")
            print(f"   Return: {metrics.get('total_return', 0):.2f}%")
            print(f"   Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"   Best: None")
    
    print("\n" + "="*60)


def cmd_tune(args):
    """Handle tune command."""
    from scripts.tune_trinity import main as tune_main
    
    # Override sys.argv for the tuning script
    sys.argv = [
        'tune_trinity.py',
        '--role', args.role,
        '--data', args.data,
        '--trials', str(args.trials)
    ]
    
    tune_main()


def cmd_quick_train(args):
    """Quick training shortcut."""
    from scripts.train_trinity import train_trinity
    
    # Default data paths
    data_paths = {
        'scalper': 'data/processed/XAUUSD_M5_processed.parquet',
        'swing': 'data/processed/XAUUSD_H1_processed.parquet',
        'trend': 'data/processed/XAUUSD_D1_processed.parquet'
    }
    
    data_path = args.data if args.data else data_paths.get(args.role, data_paths['scalper'])
    
    train_trinity(
        role=args.role,
        data_path=data_path,
        total_timesteps=args.steps,
        resume=args.resume,
        register=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="NeuroTrader AutoPilot - Automated Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autopilot.py train --role scalper          # Train scalper model
  autopilot.py train --all                   # Train all roles
  autopilot.py resume                        # Resume interrupted training
  autopilot.py evaluate --model X --data Y   # Evaluate a model
  autopilot.py compare --role scalper        # Compare scalper versions
  autopilot.py status                        # Show system status
  autopilot.py quick --role scalper          # Quick train with defaults
        """
    )
    
    parser.add_argument('--config', type=str, 
                       default='config/training_config.yaml',
                       help="Path to config file")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--role', type=str, 
                             choices=['scalper', 'swing', 'trend', 'all'],
                             default='all', help='Role to train')
    train_parser.add_argument('--fresh', action='store_true',
                             help='Start fresh, ignore checkpoints')
    train_parser.set_defaults(func=cmd_train)
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume interrupted training')
    resume_parser.set_defaults(func=cmd_resume)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model', type=str, required=True, 
                            help='Path to model file')
    eval_parser.add_argument('--data', type=str, required=True,
                            help='Path to data file')
    eval_parser.add_argument('--role', type=str, required=True,
                            choices=['scalper', 'swing', 'trend'])
    eval_parser.add_argument('--all-data', action='store_true',
                            help='Use all data instead of test set')
    eval_parser.add_argument('--json', action='store_true',
                            help='Output as JSON')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model versions')
    compare_parser.add_argument('--role', type=str, required=True,
                               choices=['scalper', 'swing', 'trend'])
    compare_parser.add_argument('--versions', type=str,
                               help='Comma-separated version numbers')
    compare_parser.set_defaults(func=cmd_compare)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show training status')
    status_parser.set_defaults(func=cmd_status)
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Run hyperparameter tuning')
    tune_parser.add_argument('--role', type=str, required=True,
                            choices=['scalper', 'swing', 'trend'])
    tune_parser.add_argument('--data', type=str, required=True,
                            help='Path to training data')
    tune_parser.add_argument('--trials', type=int, default=20,
                            help='Number of Optuna trials')
    tune_parser.set_defaults(func=cmd_tune)
    
    # Quick train command
    quick_parser = subparsers.add_parser('quick', help='Quick training with defaults')
    quick_parser.add_argument('--role', type=str, required=True,
                             choices=['scalper', 'swing', 'trend'])
    quick_parser.add_argument('--data', type=str, help='Data path (optional)')
    quick_parser.add_argument('--steps', type=int, default=500000,
                             help='Training steps')
    quick_parser.add_argument('--resume', action='store_true',
                             help='Resume from checkpoint')
    quick_parser.set_defaults(func=cmd_quick_train)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
