import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
AUGMENTED_DATA_DIR = ROOT_DIR / "data" / "augmented"

def analyze_impact(file_path):
    print(f"📊 Analyzing: {file_path.name}")
    
    df = pd.read_parquet(file_path)
    
    # Calculate Volatility (Range in Pips or absolute price)
    # Range = High - Low
    df['volatility'] = df['high'] - df['low']
    
    # Calculate Impact Groups
    # 0 = No News
    # >0 = News
    df['has_news'] = df['news_impact_score'] > 0
    
    # Analysis 1: Average Volatility
    print("\n--- Average Volatility (High - Low) ---")
    stats = df.groupby('has_news')['volatility'].describe()[['count', 'mean', 'std', 'max']]
    print(stats)
    
    no_news_mean = stats.loc[False, 'mean']
    news_mean = stats.loc[True, 'mean']
    impact_factor = news_mean / no_news_mean
    
    print(f"\n🔥 IMPACT FACTOR: {impact_factor:.2f}x")
    if impact_factor > 1.2:
        print("✅ Hypothesis Confirmed: News causes significantly higher volatility.")
    else:
        print("❌ Hypothesis Rejected: News has little impact (or data is noisy).")

    # Analysis 2: Breakdown by Score
    print("\n--- Breakdown by Impact Score (0 to 5) ---")
    breakdown = df.groupby('news_impact_score')['volatility'].mean()
    print(breakdown)

def main():
    parser = argparse.ArgumentParser()
    # Find default augmented file
    default_file = list(AUGMENTED_DATA_DIR.glob("*_News.parquet"))
    default_path = str(default_file[0]) if default_file else None
    
    parser.add_argument("--file", type=str, default=default_path, help="Path to augmented parquet file")
    args = parser.parse_args()
    
    if not args.file:
        print("❌ No augmented file found. Run augment_data.py first.")
        return
        
    analyze_impact(Path(args.file))

if __name__ == "__main__":
    main()
