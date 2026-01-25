"""
NeuroTrader Auto-Evaluator
==========================
Automatic post-training evaluation and model comparison.

Features:
- Runs backtest on test set automatically
- Calculates behavioral metrics
- Compares with previous best
- Auto-promotes if better
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from src.brain.env.trading_env import TradingEnv
from src.brain.features import add_features
from src.analysis.behavior import calculate_behavioral_metrics, generate_text_report
from src.skills.model_registry import ModelRegistry


# Role-specific evaluation criteria
EVALUATION_CRITERIA = {
    'scalper': {
        'primary_metric': 'avg_holding_time',
        'higher_is_better': False,  # Lower holding time = better for scalper
        'threshold': 20,  # Max acceptable avg holding time (steps)
        'secondary_metrics': ['win_rate', 'total_return']
    },
    'swing': {
        'primary_metric': 'sharpe_ratio',
        'higher_is_better': True,
        'threshold': 1.0,
        'secondary_metrics': ['profit_factor', 'max_drawdown']
    },
    'trend': {
        'primary_metric': 'total_return',
        'higher_is_better': True,
        'threshold': 5.0,  # Minimum 5% return
        'secondary_metrics': ['max_drawdown', 'sharpe_ratio']
    }
}


class AutoEvaluator:
    """
    Automatic model evaluation and comparison.
    
    Workflow:
    1. Load trained model
    2. Run backtest on test set
    3. Calculate metrics
    4. Compare with previous best
    5. Auto-promote if better
    6. Generate report
    """
    
    def __init__(self, models_dir: str = "models", reports_dir: str = "reports"):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.registry = ModelRegistry(str(self.models_dir))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model(
        self,
        model_path: str,
        data_path: str,
        role: str,
        use_test_set: bool = True
    ) -> Dict[str, Any]:
        """
        Run full evaluation on a model.
        
        Args:
            model_path: Path to model file (.zip)
            data_path: Path to data file
            role: Agent role (scalper/swing/trend)
            use_test_set: If True, use last 20% of data
            
        Returns:
            Dictionary containing all metrics
        """
        print(f"\n{'='*60}")
        print(f"🔍 AUTO-EVALUATOR: {role.upper()}")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Data : {data_path}")
        
        # Load data
        df = self._load_data(data_path)
        
        if use_test_set:
            split_idx = int(len(df) * 0.8)
            df = df.iloc[split_idx:]
            print(f"Using TEST set: {len(df):,} rows")
        else:
            print(f"Using ALL data: {len(df):,} rows")
        
        # Load model
        env = TradingEnv(df, agent_type=role)
        try:
            model = RecurrentPPO.load(model_path, env=env)
        except:
            model = PPO.load(model_path, env=env)
        
        # Run simulation
        print("Running simulation...")
        results = self._run_simulation(model, env, df)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, role)
        
        # Generate report
        print("\n" + "="*40)
        print("📊 EVALUATION RESULTS")
        print("="*40)
        self._print_metrics(metrics, role)
        
        return metrics
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data."""
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
            
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', drop=False, inplace=True)
        df.sort_index(inplace=True)
        
        if 'ema_9' not in df.columns:
            df = add_features(df)
            df.dropna(inplace=True)
            
        return df
    
    def _run_simulation(self, model, env, df) -> pd.DataFrame:
        """Run trading simulation and collect results."""
        obs, _ = env.reset()
        done = False
        history = []
        lstm_states = None
        
        step = 0
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            history.append({
                'step': step,
                'price': df.iloc[step]['close'] if step < len(df) else 0,
                'equity': info['equity'],
                'position': env.position,
                'action': int(action),
                'reward': reward
            })
            step += 1
            
            if done or truncated:
                break
                
        return pd.DataFrame(history)
    
    def _calculate_metrics(self, df_res: pd.DataFrame, role: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        # Financial metrics
        initial = df_res['equity'].iloc[0]
        final = df_res['equity'].iloc[-1]
        total_return = (final - initial) / initial * 100
        
        # Drawdown
        peak = df_res['equity'].cummax()
        dd = (df_res['equity'] - peak) / peak * 100
        max_dd = dd.min()
        
        # Returns for Sharpe
        df_res['returns'] = df_res['equity'].pct_change().fillna(0)
        sharpe = 0
        if df_res['returns'].std() > 0:
            sharpe = df_res['returns'].mean() / df_res['returns'].std() * np.sqrt(252)
        
        # Behavioral metrics
        beh = calculate_behavioral_metrics(df_res)
        
        metrics = {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'final_equity': final,
            'avg_holding_time': beh.get('avg_holding_time_steps', 0),
            'max_holding_time': beh.get('max_holding_time_steps', 0),
            'total_trades': beh.get('total_trades_approx', 0),
            'win_rate': beh.get('win_rate_pct', 0),
            'profit_factor': beh.get('profit_factor', 0),
            'market_exposure': beh.get('market_exposure_pct', 0),
            'avg_pnl_per_trade': beh.get('avg_pnl_per_trade', 0)
        }
        
        # Check thresholds
        criteria = EVALUATION_CRITERIA.get(role, EVALUATION_CRITERIA['trend'])
        primary = criteria['primary_metric']
        threshold = criteria['threshold']
        higher_better = criteria['higher_is_better']
        
        value = metrics.get(primary, 0)
        if higher_better:
            metrics['meets_threshold'] = value >= threshold
        else:
            metrics['meets_threshold'] = value <= threshold
            
        return metrics
    
    def _print_metrics(self, metrics: Dict, role: str):
        """Pretty print evaluation metrics."""
        criteria = EVALUATION_CRITERIA.get(role, {})
        primary = criteria.get('primary_metric', 'total_return')
        
        print(f"\n💰 Financial Performance:")
        print(f"   Total Return  : {metrics['total_return']:>10.2f}%")
        print(f"   Max Drawdown  : {metrics['max_drawdown']:>10.2f}%")
        print(f"   Sharpe Ratio  : {metrics['sharpe_ratio']:>10.2f}")
        print(f"   Final Equity  : ${metrics['final_equity']:>10,.2f}")
        
        print(f"\n📈 Trading Behavior:")
        print(f"   Total Trades  : {metrics['total_trades']:>10}")
        print(f"   Win Rate      : {metrics['win_rate']:>10.2f}%")
        print(f"   Profit Factor : {metrics['profit_factor']:>10.2f}")
        print(f"   Avg Holding   : {metrics['avg_holding_time']:>10.1f} steps")
        print(f"   Market Exposure: {metrics['market_exposure']:>9.1f}%")
        
        # Highlight primary metric
        status = "✅ PASS" if metrics.get('meets_threshold', False) else "❌ FAIL"
        print(f"\n🎯 Primary Metric ({primary}): {metrics.get(primary, 0):.2f} {status}")
    
    def compare_and_promote(
        self,
        model_path: str,
        data_path: str,
        role: str,
        version: int
    ) -> bool:
        """
        Evaluate model and auto-promote if better than current best.
        
        Returns:
            True if model was promoted
        """
        # Run evaluation
        metrics = self.evaluate_model(model_path, data_path, role)
        
        # Update registry with metrics
        criteria = EVALUATION_CRITERIA.get(role, EVALUATION_CRITERIA['trend'])
        
        return self.registry.auto_promote_if_better(
            role=role,
            new_version=version,
            primary_metric=criteria['primary_metric'],
            higher_is_better=criteria['higher_is_better']
        )
    
    def generate_comparison_report(self, role: str) -> str:
        """Generate markdown comparison report for all versions."""
        versions = self.registry.list_versions(role)
        
        if not versions:
            return f"No models found for {role}"
        
        report = [
            f"# {role.upper()} Model Comparison Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "| Version | Created | Return | Sharpe | Holding | Win Rate |",
            "|---------|---------|--------|--------|---------|----------|"
        ]
        
        for v in versions:
            m = v.get('metrics', {})
            report.append(
                f"| {v['version']} | {v['created_at'][:10]} | "
                f"{m.get('total_return', 0):.1f}% | "
                f"{m.get('sharpe_ratio', 0):.2f} | "
                f"{m.get('avg_holding_time', 0):.0f} | "
                f"{m.get('win_rate', 0):.1f}% |"
            )
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.reports_dir / f"comparison_{role}_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"📝 Report saved: {report_path}")
        return report_text


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Evaluator CLI")
    parser.add_argument('--model', type=str, required=True, help="Path to model")
    parser.add_argument('--data', type=str, required=True, help="Path to data")
    parser.add_argument('--role', type=str, required=True, 
                       choices=['scalper', 'swing', 'trend'])
    parser.add_argument('--compare', action='store_true', 
                       help="Compare all versions")
    
    args = parser.parse_args()
    
    evaluator = AutoEvaluator()
    
    if args.compare:
        print(evaluator.generate_comparison_report(args.role))
    else:
        evaluator.evaluate_model(args.model, args.data, args.role)
