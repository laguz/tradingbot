"""
Test script for ML service v2 with enhanced features and ensemble models.
"""

from services.ml_service import predict_next_days
import pandas as pd

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("Testing ML Service v2 - Enhanced Predictions")
print("="*80)

ticker = 'SPY'
print(f"\nTesting predict_next_days for {ticker}...")

try:
    result = predict_next_days(ticker, 1, force_retrain=True)
    
    if 'error' in result:
        print(f"\n❌ FAILED: {result['error']}")
    else:
        print("\n✅ SUCCESS: Predictions generated")
        print(f"\nTicker: {result['ticker']}")
        print(f"Model Version: {result['model_version']}")
        print(f"Features Used: {result['feature_count']}")
        print(f"Last Close ({result['last_date']}): ${result['last_close']:.2f}")
        
        print(f"\n1-Day Prediction:")
        print("-" * 80)
        for i, (date, pred, low, med, high) in enumerate(zip(
            result['target_dates'],
            result['predictions'],
            result['confidence_intervals']['low'],
            result['confidence_intervals']['median'],
            result['confidence_intervals']['high']
        ), 1):
            change = ((pred - result['last_close']) / result['last_close']) * 100
            interval_width = ((high - low) / result['last_close']) * 100
            print(f"Day {i} ({date}): ${pred:.2f} ({change:+.1f}%)")
            print(f"  └─ Interval: ${low:.2f} - ${med:.2f} - ${high:.2f} (width: {interval_width:.1f}%)")
        
        # Summary
        day1_pred = result['predictions'][0]
        day1_change = ((day1_pred - result['last_close']) / result['last_close']) * 100
        print(f"\n📊 Summary:")
        print(f"  Forecast: ${day1_pred:.2f} ({day1_change:+.1f}%)")
        print(f"  95th percentile (optimistic): ${result['confidence_intervals']['high'][0]:.2f}")
        print(f"  5th percentile (pessimistic): ${result['confidence_intervals']['low'][0]:.2f}")
        
except Exception as e:
    print(f"\n❌ CRASHED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
