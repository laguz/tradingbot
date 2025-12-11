"""
Quick analysis to understand what's going wrong with predictions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import MLPredictionModel
from services.stock_data_service import get_historical_data
from services.ml_features import prepare_features
from datetime import datetime

def analyze_problem():
    """Analyze the prediction problem."""
    
    print("=" * 80)
    print("ML PREDICTION PROBLEM ANALYSIS")
    print("=" * 80)
    
    # Initialize
    init_db()
    
    # Check predictions
    collection = MLPredictionModel.get_collection()
    predictions = list(collection.find({
        'actual_price': {'$ne': None}
    }).limit(10))
    
    print(f"\nSample predictions with actual prices:\n")
    for p in predictions:
        symbol = p['symbol']
        pred = p['predicted_price']
        actual = p['actual_price']
        error = actual - pred
        error_pct = (error / actual) * 100
        
        print(f"{symbol:6s}: Pred ${pred:8.2f} | Actual ${actual:8.2f} | Error ${error:8.2f} ({error_pct:+6.1f}%)")
    
    # Now check if there's a scale issue - get historical data
    print(f"\n" + "=" * 80)
    print("CHECKING HISTORICAL PRICES")
    print("=" * 80)
    
    for symbol in ['AAPL', 'SPY', 'TSLA', 'RIOT']:
        df = get_historical_data(symbol, '1mo', use_cache=True)
        if not df.empty:
            recent_close = df['Close'].iloc[-1]
            recent_mean = df['Close'].tail(20).mean()
            print(f"\n{symbol}:")
            print(f"  Latest close: ${recent_close:.2f}")
            print(f"  20-day mean: ${recent_mean:.2f}")
            
            # Check what features look like
            try:
                X, y, features, scaler = prepare_features(df, for_training=True, prediction_horizon=1)
                print(f"  Training samples: {len(X)}")
                print(f"  Features: {len(features)}")
                if y is not None and len(y) > 0:
                    print(f"  Target range: ${y.min().values[0]:.2f} - ${y.max().values[0]:.2f}")
                    print(f"  Recent targets: {y.tail(3).values.flatten()}")
            except Exception as e:
                print(f"  Error in features: {e}")

if __name__ == '__main__':
    analyze_problem()
