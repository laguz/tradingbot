#!/usr/bin/env python3
"""
Analyze feature importance across multiple tickers to identify the best features for price prediction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from database import init_db
from services.stock_data_service import get_historical_data
from services.ml_features import prepare_features
from services.ml_service import create_model
from config import get_config

config = get_config()


def analyze_feature_importance(tickers, top_n=10):
    """
    Analyze feature importance across multiple tickers.
    
    Args:
        tickers: List of stock symbols to analyze
        top_n: Number of top features to return
    """
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    init_db()
    
    all_importances = []
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        
        # Get historical data
        df = get_historical_data(ticker, '2y', use_cache=True)
        if df.empty:
            print(f"  ❌ No data for {ticker}")
            continue
        
        # Prepare features
        try:
            X, y, feature_names, scaler = prepare_features(df, for_training=True, prediction_horizon=1)
        except Exception as e:
            print(f"  ❌ Feature prep failed: {e}")
            continue
        
        if len(X) < 100:
            print(f"  ❌ Not enough samples: {len(X)}")
            continue
        
        # Train simple RandomForest to get feature importances
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Extract target (single column)
        y_values = y.iloc[:, 0] if hasattr(y, 'iloc') else y
        model.fit(X, y_values)
        
        # Get importances
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_,
            'ticker': ticker
        })
        
        all_importances.append(importances)
        print(f"  ✓ Analyzed {len(feature_names)} features")
    
    if not all_importances:
        print("\n❌ No successful analyses")
        return None
    
    # Combine all importances
    combined = pd.concat(all_importances, ignore_index=True)
    
    # Calculate average importance per feature
    avg_importance = combined.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
    avg_importance = avg_importance.sort_values('mean', ascending=False)
    
    print("\n" + "=" * 80)
    print(f"TOP {top_n} FEATURES (averaged across {len(tickers)} tickers)")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Feature':<25} {'Avg Importance':<15} {'Std Dev':<12} {'Count'}")
    print("-" * 80)
    
    top_features = []
    for i, row in enumerate(avg_importance.head(top_n).itertuples(), 1):
        print(f"{i:<6} {row.feature:<25} {row.mean:<15.4f} {row.std:<12.4f} {row.count}")
        top_features.append(row.feature)
    
    # Show feature categories
    print("\n" + "=" * 80)
    print("FEATURE CATEGORIES IN TOP 10")
    print("=" * 80)
    
    categories = {
        'Lag': [f for f in top_features if 'Lag' in f],
        'Technical': [f for f in top_features if any(x in f for x in ['RSI', 'MACD', 'BB_', 'ATR'])],
        'Volume': [f for f in top_features if any(x in f for x in ['Volume', 'OBV'])],
        'Volatility': [f for f in top_features if any(x in f for x in ['HV_', 'Parkinson'])],
        'Distance': [f for f in top_features if 'Distance' in f],
        'Other': [f for f in top_features if not any(cat in f for cat in ['Lag', 'RSI', 'MACD', 'BB_', 'ATR', 'Volume', 'OBV', 'HV_', 'Parkinson', 'Distance'])]
    }
    
    for cat, features in categories.items():
        if features:
            print(f"\n{cat}: {len(features)} features")
            for f in features:
                print(f"  - {f}")
    
    # Save to file
    output_file = 'feature_importance_analysis.csv'
    avg_importance.to_csv(output_file, index=False)
    print(f"\n✓ Full analysis saved to {output_file}")
    
    # Return top features
    return top_features


if __name__ == '__main__':
    # Analyze across diverse set of tickers
    tickers = ['AAPL', 'SPY', 'TSLA', 'MSFT', 'QQQ', 'RIOT']
    
    top_features = analyze_feature_importance(tickers, top_n=10)
    
    if top_features:
        print("\n" + "=" * 80)
        print("RECOMMENDED TOP 10 FEATURES:")
        print("=" * 80)
        print(top_features)
