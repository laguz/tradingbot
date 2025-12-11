"""
Test improved ML predictions with validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from services.ml_service import predict_next_days

def test_predictions():
    """Test predictions for multiple tickers."""
    
    print("=" * 80)
    print("TESTING IMPROVED ML PREDICTIONS")
    print("=" * 80)
    
    # Initialize
    init_db()
    
    # Test symbols
    symbols = ['AAPL', 'SPY', 'TSLA', 'RIOT']
    
    for symbol in symbols:
        print(f"\n{'=' * 80}")
        print(f"Testing {symbol}")
        print(f"{'=' * 80}")
        
        try:
            # Get prediction (will force retrain since we cleared models)
            result = predict_next_days(symbol, days=1, force_retrain=False)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                last_close = result['last_close']
                prediction = result['predictions'][0]
                change_pct = ((prediction / last_close) - 1) * 100
                
                print(f"\n✅ Prediction successful for {symbol}:")
                print(f"  Last Close: ${last_close:.2f}")
                print(f"  Prediction: ${prediction:.2f}")
                print(f"  Change: {change_pct:+.2f}%")
                print(f"  Target Date: {result['target_dates'][0]}")
                print(f"  Model Version: {result['model_version']}")
                print(f"  Features: {result['feature_count']}")
                
                # Check if within bounds
                deviation = abs(change_pct)
                if deviation > 30:
                    print(f"  ⚠️  WARNING: Prediction {deviation:.1f}% from last close!")
                elif deviation < 5:
                    print(f"  ℹ️  Conservative prediction (within 5%)")
                else:
                    print(f"  ✓ Reasonable prediction")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("Test Complete")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    test_predictions()
