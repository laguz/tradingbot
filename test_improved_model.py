"""
Test improved ML predictions with time weighting and feature selection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from services.ml_service import predict_next_days

def test_improved_predictions():
    """Test predictions with new time-weighted & feature-selected model."""
    
    print("=" * 80)
    print("TESTING TIME-WEIGHTED TOP-10 FEATURE MODEL")
    print("=" * 80)
    
    # Initialize
    init_db()
    
    # Test symbols
    symbols = ['AAPL', 'SPY']
    
    for symbol in symbols:
        print(f"\n{'=' * 80}")
        print(f"Testing {symbol}")
        print(f"{'=' * 80}")
        
        try:
            # Get prediction (will train with new settings)
            result = predict_next_days(symbol, days=1, force_retrain=True)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                last_close = result['last_close']
                prediction = result['predictions'][0]
                change_pct = ((prediction / last_close) - 1) * 100
                
                print(f"\n✅ Prediction successful:")
                print(f"  Model Version: {result['model_version']}")
                print(f"  Features: {result['feature_count']}")
                print(f"  Last Close: ${last_close:.2f}")
                print(f"  Prediction: ${prediction:.2f}")
                print(f"  Change: {change_pct:+.2f}%")
                
                # Check bounds
                if abs(change_pct) <= 30:
                    print(f"  ✓ Within 30% bounds")
                else:
                    print(f"  ⚠️  Outside bounds")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_improved_predictions()
