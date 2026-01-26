"""
Smart Data Discovery for NeuroTrader

Automatically finds and selects the best data/bar type without manual specification.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def find_available_bar_types(
    catalog_path: str = "data/nautilus_catalog",
    workspace: Optional[Path] = None
) -> List[Dict]:
    """
    Scan catalog and find all available bar types.
    
    Returns:
        List of bar type information
    """
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.model.data import BarType
    
    # Determine catalog path
    if workspace:
        catalog_dir = workspace / catalog_path
    else:
        catalog_dir = Path(catalog_path)
    
    if not catalog_dir.exists():
        logger.warning(f"Catalog not found: {catalog_dir}")
        return []
    
    try:
        catalog = ParquetDataCatalog(str(catalog_dir))
        
        # Get all instruments
        instruments = catalog.instruments()
        
        if not instruments:
            logger.warning("No instruments found in catalog")
            return []
        
        logger.info(f"📊 Found {len(instruments)} instrument(s)")
        
        bar_types_info = []
        
        for instrument in instruments:
            # Get bar types for this instrument
            try:
                # LISTING BAR TYPES
                # Note: Some versions of Nautilus don't have catalog.bar_types() exposed cleanly
                # We try the official way first, then fallback to common patterns
                
                bar_specs = []
                if hasattr(catalog, 'bar_types'):
                    try:
                        bar_specs = catalog.bar_types(instrument_ids=[instrument.id])
                    except:
                        pass
                
                # FALLBACK: If API doesn't list them, we construct standard ones to check
                if not bar_specs:
                    # Common timeframes to check
                    from nautilus_trader.model.data import BarType, BarSpecification
                    from nautilus_trader.model.enums import BarAggregation, PriceType
                    
                    # Create a standard list of potential bar types
                    # e.g. XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL
                    timeframes = [
                        (15, BarAggregation.MINUTE),
                        (5, BarAggregation.MINUTE),
                        (1, BarAggregation.MINUTE),
                        (1, BarAggregation.HOUR),
                        (4, BarAggregation.HOUR),
                        (1, BarAggregation.DAY)
                    ]
                    
                    for count, agg in timeframes:
                        try:
                            spec = BarSpecification(count, agg, PriceType.LAST)
                            bt = BarType(instrument.id, spec)
                            bar_specs.append(bt)
                        except:
                            pass

                for bar_type in bar_specs:
                    # Load bars to get count and date range
                    bars = list(catalog.bars(bar_types=[bar_type]))
                    
                    if not bars:
                        continue
                    
                    # Get date range
                    first_bar = bars[0]
                    last_bar = bars[-1]
                    
                    start_date = datetime.fromtimestamp(first_bar.ts_init / 1e9)
                    end_date = datetime.fromtimestamp(last_bar.ts_init / 1e9)
                    
                    # Calculate metrics
                    num_bars = len(bars)
                    days_range = (end_date - start_date).days
                    bars_per_day = num_bars / max(days_range, 1)
                    
                    bar_types_info.append({
                        'bar_type': bar_type,
                        'bar_type_str': str(bar_type),
                        'instrument': str(instrument.id),
                        'num_bars': num_bars,
                        'start_date': start_date,
                        'end_date': end_date,
                        'days_range': days_range,
                        'bars_per_day': bars_per_day,
                        'timeframe': bar_type.spec.aggregation  # M5, M15, H1, etc.
                    })
                    
                    logger.debug(
                        f"  ✅ {bar_type}: {num_bars:,} bars, "
                        f"{start_date.date()} to {end_date.date()}"
                    )
                    
            except Exception as e:
                logger.debug(f"  ⚠️  Error reading {instrument.id}: {e}")
                continue
        
        return bar_types_info
        
    except Exception as e:
        logger.error(f"Error reading catalog: {e}")
        return []


def find_best_bar_type(
    catalog_path: str = "data/nautilus_catalog",
    workspace: Optional[Path] = None,
    criteria: str = "most_data",
    preferred_instruments: List[str] = None,
    preferred_timeframes: List[str] = None
) -> Optional[str]:
    """
    Automatically find the best bar type for training/backtesting.
    
    Args:
        catalog_path: Path to Nautilus catalog
        workspace: Workspace for Colab
        criteria: Selection criteria:
            - "most_data": Most bars (default)
            - "longest_range": Longest date range
            - "recent": Most recent data
        preferred_instruments: List of preferred instruments (e.g., ['XAUUSD', 'BTCUSD'])
        preferred_timeframes: List of preferred timeframes (e.g., ['15-MINUTE', '5-MINUTE'])
    
    Returns:
        Bar type string (e.g., 'XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL')
    """
    bar_types = find_available_bar_types(catalog_path, workspace)
    
    if not bar_types:
        logger.warning("No bar types found in catalog")
        return None
    
    logger.info(f"🔍 Found {len(bar_types)} bar type(s)")
    
    # Filter by preferences
    candidates = bar_types
    
    if preferred_instruments:
        candidates = [
            bt for bt in candidates
            if any(inst in bt['instrument'] for inst in preferred_instruments)
        ]
        logger.info(f"   Filtered by instruments: {len(candidates)} remaining")
    
    if preferred_timeframes:
        candidates = [
            bt for bt in candidates
            if any(tf in bt['bar_type_str'] for tf in preferred_timeframes)
        ]
        logger.info(f"   Filtered by timeframes: {len(candidates)} remaining")
    
    if not candidates:
        logger.warning("No bar types match preferences, using all")
        candidates = bar_types
    
    # Select based on criteria
    if criteria == "most_data":
        candidates.sort(key=lambda x: x['num_bars'], reverse=True)
        best = candidates[0]
        logger.info(f"📊 Selection Criteria: Most Data")
        
    elif criteria == "longest_range":
        candidates.sort(key=lambda x: x['days_range'], reverse=True)
        best = candidates[0]
        logger.info(f"📊 Selection Criteria: Longest Range")
        
    elif criteria == "recent":
        candidates.sort(key=lambda x: x['end_date'], reverse=True)
        best = candidates[0]
        logger.info(f"📊 Selection Criteria: Most Recent")
        
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    # Show selection
    logger.info(f"✅ Selected Bar Type:")
    logger.info(f"   Type:       {best['bar_type_str']}")
    logger.info(f"   Bars:       {best['num_bars']:,}")
    logger.info(f"   Range:      {best['start_date'].date()} to {best['end_date'].date()}")
    logger.info(f"   Duration:   {best['days_range']} days ({best['days_range']/365:.1f} years)")
    logger.info(f"   Bars/Day:   {best['bars_per_day']:.0f}")
    
    # Show alternatives
    if len(candidates) > 1:
        logger.info(f"\n📋 Other Bar Types Available:")
        for i, candidate in enumerate(candidates[1:4], 1):
            logger.info(
                f"   {i}. {candidate['bar_type_str']}: "
                f"{candidate['num_bars']:,} bars, "
                f"{candidate['days_range']} days"
            )
    
    return best['bar_type_str']



def get_recommended_split_dates(
    bar_type_str: str,
    catalog_path: str = "data/nautilus_catalog",
    workspace: Optional[Path] = None,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15
) -> Dict[str, str]:
    """
    Get recommended train/val/test split dates for a bar type.
    
    Returns:
        Dictionary with split dates:
        {
            'train_start': '2014-01-01',
            'train_end': '2023-05-31',
            'val_start': '2023-06-01',
            'val_end': '2024-09-30',
            'test_start': '2024-10-01',
            'test_end': '2026-01-16'
        }
    """
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.model.data import BarType
    
    # Determine catalog path
    if workspace:
        catalog_dir = workspace / catalog_path
    else:
        catalog_dir = Path(catalog_path)
    
    catalog = ParquetDataCatalog(str(catalog_dir))
    bar_type = BarType.from_str(bar_type_str)
    
    # Load bars
    bars = list(catalog.bars(bar_types=[bar_type]))
    
    if not bars:
        raise ValueError(f"No bars found for {bar_type_str}")
    
    # Calculate split indices
    n = len(bars)
    train_end_idx = int(n * train_pct)
    val_end_idx = int(n * (train_pct + val_pct))
    
    # Get dates
    train_start = datetime.fromtimestamp(bars[0].ts_init / 1e9)
    train_end = datetime.fromtimestamp(bars[train_end_idx - 1].ts_init / 1e9)
    val_start = datetime.fromtimestamp(bars[train_end_idx].ts_init / 1e9)
    val_end = datetime.fromtimestamp(bars[val_end_idx - 1].ts_init / 1e9)
    test_start = datetime.fromtimestamp(bars[val_end_idx].ts_init / 1e9)
    test_end = datetime.fromtimestamp(bars[-1].ts_init / 1e9)
    
    splits = {
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'val_start': val_start.strftime('%Y-%m-%d'),
        'val_end': val_end.strftime('%Y-%m-%d'),
        'test_start': test_start.strftime('%Y-%m-%d'),
        'test_end': test_end.strftime('%Y-%m-%d'),
        'train_bars': train_end_idx,
        'val_bars': val_end_idx - train_end_idx,
        'test_bars': n - val_end_idx
    }
    
    logger.info(f"📅 Recommended Splits (70/15/15):")
    logger.info(f"   Train: {splits['train_start']} to {splits['train_end']} ({splits['train_bars']:,} bars)")
    logger.info(f"   Val:   {splits['val_start']} to {splits['val_end']} ({splits['val_bars']:,} bars)")
    logger.info(f"   Test:  {splits['test_start']} to {splits['test_end']} ({splits['test_bars']:,} bars)")
    
    return splits


def auto_configure_training(
    catalog_path: str = "data/nautilus_catalog",
    workspace: Optional[Path] = None
) -> Dict:
    """
    Automatically configure training with best data and model.
    
    Returns:
        Configuration dictionary with model_path, bar_type, and split dates
    """
    from src.brain.model_discovery import find_best_model
    
    print("="*60)
    print("🎯 Auto-Configuration")
    print("="*60)
    
    # Find best bar type
    print("\n📊 Step 1: Finding best data...")
    bar_type = find_best_bar_type(
        catalog_path=catalog_path,
        workspace=workspace,
        criteria="most_data",
        preferred_instruments=['XAUUSD', 'BTCUSD'],
        preferred_timeframes=['15-MINUTE', '5-MINUTE']
    )
    
    if not bar_type:
        raise ValueError("No suitable bar type found")
    
    # Get split dates
    print("\n📅 Step 2: Calculating split dates...")
    splits = get_recommended_split_dates(bar_type, catalog_path, workspace)
    
    # Find best model (optional, may not exist yet)
    print("\n🤖 Step 3: Looking for trained model...")
    try:
        model_path = find_best_model(workspace=workspace)
        print(f"   ✅ Found: {model_path}")
    except:
        print("   ℹ️  No trained model found (train one first)")
        model_path = None
    
    config = {
        'bar_type': bar_type,
        'model_path': str(model_path) if model_path else None,
        **splits
    }
    
    print("\n" + "="*60)
    print("✅ Auto-Configuration Complete!")
    print("="*60)
    
    return config


if __name__ == "__main__":
    # Test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("🔍 Smart Data Discovery Test")
    print("="*60)
    
    # Find all bar types
    bar_types = find_available_bar_types()
    
    print(f"\n📊 Available Bar Types: {len(bar_types)}")
    for bt in bar_types:
        print(f"   - {bt['bar_type_str']}: {bt['num_bars']:,} bars")
    
    # Find best
    print(f"\n{'='*60}")
    best = find_best_bar_type(
        preferred_instruments=['XAUUSD'],
        preferred_timeframes=['15-MINUTE', '5-MINUTE']
    )
    
    if best:
        print(f"\n✅ Best bar type: {best}")
        
        # Get splits
        print(f"\n{'='*60}")
        splits = get_recommended_split_dates(best)
        
        # Auto-config
        print(f"\n{'='*60}")
        config = auto_configure_training()
        
        print(f"\n📋 Final Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
