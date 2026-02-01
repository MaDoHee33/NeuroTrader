"""
Walk-Forward Validation Framework for NeuroTrader

This module implements walk-forward analysis to validate trading strategies
and prevent overfitting. It's the gold standard for evaluating RL trading agents.

Key Concepts:
- Split data into multiple train/test windows
- Train on each window, test on the next period
- Calculate Walk-Forward Efficiency (WFE)
- Aggregate out-of-sample results

WFE = Out-of-Sample Return / In-Sample Return
- WFE > 60%: Excellent (robust strategy)
- WFE > 50%: Good
- WFE < 40%: Likely overfitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import BarType


class WalkForwardValidator:
    """
    Implements walk-forward validation for trading strategies.
    
    Example:
        >>> validator = WalkForwardValidator(
        ...     data_path='data/nautilus_catalog',
        ...     bar_type='XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL'
        ... )
        >>> results = validator.validate(
        ...     train_window_years=4,
        ...     test_window_years=1,
        ...     step_years=1
        ... )
        >>> print(f"WFE: {results['wfe']:.2%}")
    """
    
    def __init__(self, data_path: str, bar_type: str):
        """
        Initialize validator.
        
        Args:
            data_path: Path to Nautilus catalog
            bar_type: Bar type string (e.g., 'XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL')
        """
        self.data_path = Path(data_path)
        self.bar_type_str = bar_type
        self.df = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load data from Nautilus catalog."""
        print(f"📂 Loading data from {self.data_path}")
        print(f"📊 Bar type: {self.bar_type_str}")
        
        catalog = ParquetDataCatalog(str(self.data_path))
        bar_type = BarType.from_str(self.bar_type_str)
        
        bars = list(catalog.bars(bar_types=[bar_type]))
        
        data = []
        for bar in bars:
            data.append({
                'timestamp': pd.Timestamp(bar.ts_init, unit='ns'),
                'open': bar.open.as_double(),
                'high': bar.high.as_double(),
                'low': bar.low.as_double(),
                'close': bar.close.as_double(),
                'volume': bar.volume.as_double()
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df):,} bars")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def split_train_val_test(
        self,
        train_pct: float = 0.7,
        val_pct: float = 0.15,
        test_pct: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Simple train/validation/test split.
        
        Args:
            train_pct: Percentage for training (default: 70%)
            val_pct: Percentage for validation (default: 15%)
            test_pct: Percentage for test (default: 15%)
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, \
            "Percentages must sum to 1.0"
        
        n = len(self.df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        splits = {
            'train': self.df.iloc[:train_end].copy(),
            'val': self.df.iloc[train_end:val_end].copy(),
            'test': self.df.iloc[val_end:].copy()
        }
        
        print("\n📊 Data Split:")
        for name, data in splits.items():
            print(f"   {name:5s}: {len(data):7,} bars "
                  f"({data['timestamp'].min().date()} to {data['timestamp'].max().date()})")
        
        return splits
    
    def create_walk_forward_windows(
        self,
        train_window_years: int = 4,
        test_window_years: int = 1,
        step_years: int = 1,
        mode: str = 'rolling'
    ) -> List[Dict[str, pd.DataFrame]]:
        """
        Create walk-forward windows for validation.
        
        Args:
            train_window_years: Size of training window in years
            test_window_years: Size of testing window in years
            step_years: Step size in years (for rolling window)
            mode: 'rolling' or 'anchored'
                - rolling: Fixed window that moves forward
                - anchored: Expanding window from start
        
        Returns:
            List of windows, each containing:
                {'train': DataFrame, 'test': DataFrame, 'window_id': int,
                 'train_dates': tuple, 'test_dates': tuple}
        """
        df = self.df.copy()
        df['year'] = df['timestamp'].dt.year
        
        min_year = df['year'].min()
        max_year = df['year'].max()
        
        windows = []
        window_id = 0
        
        if mode == 'rolling':
            # Rolling window: fixed size, moves forward
            current_year = min_year
            
            while current_year + train_window_years + test_window_years <= max_year:
                train_start_year = current_year
                train_end_year = current_year + train_window_years
                test_end_year = train_end_year + test_window_years
                
                train_data = df[
                    (df['year'] >= train_start_year) & 
                    (df['year'] < train_end_year)
                ].copy()
                
                test_data = df[
                    (df['year'] >= train_end_year) & 
                    (df['year'] < test_end_year)
                ].copy()
                
                if len(train_data) > 0 and len(test_data) > 0:
                    windows.append({
                        'window_id': window_id,
                        'train': train_data,
                        'test': test_data,
                        'train_dates': (
                            train_data['timestamp'].min().date(),
                            train_data['timestamp'].max().date()
                        ),
                        'test_dates': (
                            test_data['timestamp'].min().date(),
                            test_data['timestamp'].max().date()
                        )
                    })
                    window_id += 1
                
                current_year += step_years
                
        else:  # anchored
            # Anchored window: expands from start
            test_start_year = min_year + train_window_years
            
            while test_start_year + test_window_years <= max_year:
                train_data = df[df['year'] < test_start_year].copy()
                test_data = df[
                    (df['year'] >= test_start_year) & 
                    (df['year'] < test_start_year + test_window_years)
                ].copy()
                
                if len(train_data) > 0 and len(test_data) > 0:
                    windows.append({
                        'window_id': window_id,
                        'train': train_data,
                        'test': test_data,
                        'train_dates': (
                            train_data['timestamp'].min().date(),
                            train_data['timestamp'].max().date()
                        ),
                        'test_dates': (
                            test_data['timestamp'].min().date(),
                            test_data['timestamp'].max().date()
                        )
                    })
                    window_id += 1
                
                test_start_year += step_years
        
        print(f"\n🪟 Created {len(windows)} walk-forward windows ({mode} mode):")
        for window in windows:
            print(f"   Window {window['window_id']}: "
                  f"Train[{window['train_dates'][0]} to {window['train_dates'][1]}] "
                  f"→ Test[{window['test_dates'][0]} to {window['test_dates'][1]}]")
        
        return windows
    
    def calculate_wfe(
        self,
        in_sample_returns: List[float],
        out_of_sample_returns: List[float]
    ) -> Dict[str, float]:
        """
        Calculate Walk-Forward Efficiency.
        
        WFE = (Out-of-Sample Annualized Return) / (In-Sample Annualized Return)
        
        Args:
            in_sample_returns: List of in-sample (training) returns per window
            out_of_sample_returns: List of out-of-sample (test) returns per window
        
        Returns:
            Dictionary with metrics:
                - wfe: Walk-Forward Efficiency
                - in_sample_return: Average in-sample return
                - out_of_sample_return: Average out-of-sample return
                - sharpe_is: In-sample Sharpe ratio
                - sharpe_oos: Out-of-sample Sharpe ratio
        """
        is_returns = np.array(in_sample_returns)
        oos_returns = np.array(out_of_sample_returns)
        
        # Annualized returns (assuming returns are already annualized)
        is_return = np.mean(is_returns)
        oos_return = np.mean(oos_returns)
        
        # Calculate WFE
        wfe = oos_return / is_return if is_return != 0 else 0
        
        # Calculate Sharpe ratios
        sharpe_is = np.mean(is_returns) / (np.std(is_returns) + 1e-6)
        sharpe_oos = np.mean(oos_returns) / (np.std(oos_returns) + 1e-6)
        
        results = {
            'wfe': wfe,
            'in_sample_return': is_return,
            'out_of_sample_return': oos_return,
            'sharpe_is': sharpe_is,
            'sharpe_oos': sharpe_oos,
            'num_windows': len(is_returns)
        }
        
        return results
    
    def validate_model(
        self,
        model_path: str,
        windows: List[Dict[str, pd.DataFrame]],
        save_results: bool = True
    ) -> Dict:
        """
        Validate a trained model using walk-forward windows.
        
        Args:
            model_path: Path to trained PPO model
            windows: Walk-forward windows from create_walk_forward_windows()
            save_results: Whether to save results to JSON
        
        Returns:
            Validation results dictionary
        """
        # This is a placeholder - actual implementation would:
        # 1. Load model
        # 2. For each window:
        #    - Optionally retrain on window.train
        #    - Backtest on window.test
        #    - Calculate returns
        # 3. Calculate WFE from all windows
        
        print(f"\n🧪 Validating model: {model_path}")
        print(f"   Testing on {len(windows)} windows")
        
        # Placeholder results
        results = {
            'model_path': model_path,
            'validation_date': datetime.now().isoformat(),
            'num_windows': len(windows),
            'windows': [],
            'wfe': None,
            'summary': None
        }
        
        print("\n⚠️  Model validation not yet implemented")
        print("   This requires integration with backtesting engine")
        print("   Will be implemented in next iteration")
        
        return results
    
    def save_splits(self, splits: Dict[str, pd.DataFrame], output_dir: str):
        """
        Save train/val/test splits to separate Parquet files.
        
        Args:
            splits: Dictionary with 'train', 'val', 'test' DataFrames
            output_dir: Directory to save splits
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, df in splits.items():
            file_path = output_path / f"{name}_data.parquet"
            df.to_parquet(file_path, index=False)
            print(f"💾 Saved {name} split: {file_path} ({len(df):,} rows)")
        
        # Save metadata
        metadata = {
            'split_date': datetime.now().isoformat(),
            'bar_type': self.bar_type_str,
            'total_bars': len(self.df),
            'splits': {
                name: {
                    'rows': len(df),
                    'start_date': str(df['timestamp'].min()),
                    'end_date': str(df['timestamp'].max())
                }
                for name, df in splits.items()
            }
        }
        
        metadata_path = output_path / 'split_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Saved metadata: {metadata_path}")


def main():
    """Example usage of WalkForwardValidator."""
    
    # Initialize validator
    validator = WalkForwardValidator(
        data_path='data/nautilus_catalog',
        bar_type='XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL'
    )
    
    # Method 1: Simple 70/15/15 split
    print("\n" + "="*60)
    print("METHOD 1: Simple Train/Val/Test Split")
    print("="*60)
    
    splits = validator.split_train_val_test(
        train_pct=0.70,
        val_pct=0.15,
        test_pct=0.15
    )
    
    # Save splits
    validator.save_splits(splits, output_dir='data/splits')
    
    # Method 2: Walk-forward windows
    print("\n" + "="*60)
    print("METHOD 2: Walk-Forward Windows (Rolling)")
    print("="*60)
    
    windows = validator.create_walk_forward_windows(
        train_window_years=4,
        test_window_years=1,
        step_years=1,
        mode='rolling'
    )
    
    # Example WFE calculation (with dummy data)
    print("\n" + "="*60)
    print("EXAMPLE: WFE Calculation")
    print("="*60)
    
    # Simulate returns from walk-forward windows
    # In reality, these would come from backtesting
    in_sample_returns = [0.15, 0.18, 0.20, 0.17]  # 15-20% per window
    out_of_sample_returns = [0.10, 0.12, 0.08, 0.11]  # 8-12% per window
    
    wfe_results = validator.calculate_wfe(in_sample_returns, out_of_sample_returns)
    
    print(f"\n📊 Walk-Forward Efficiency Results:")
    print(f"   In-Sample Return:     {wfe_results['in_sample_return']:.2%}")
    print(f"   Out-of-Sample Return: {wfe_results['out_of_sample_return']:.2%}")
    print(f"   WFE:                  {wfe_results['wfe']:.2%}")
    print(f"   In-Sample Sharpe:     {wfe_results['sharpe_is']:.3f}")
    print(f"   Out-of-Sample Sharpe: {wfe_results['sharpe_oos']:.3f}")
    
    # Interpretation
    wfe = wfe_results['wfe']
    if wfe > 0.60:
        print(f"\n✅ Excellent! WFE > 60% indicates robust strategy")
    elif wfe > 0.50:
        print(f"\n👍 Good! WFE > 50% is acceptable")
    elif wfe > 0.40:
        print(f"\n⚠️  Moderate. WFE 40-50% needs improvement")
    else:
        print(f"\n❌ Poor. WFE < 40% suggests overfitting")


if __name__ == '__main__':
    main()
