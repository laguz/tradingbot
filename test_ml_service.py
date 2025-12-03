from services.ml_service import predict_next_days
import pandas as pd

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Testing predict_next_days for SPY...")
try:
    result = predict_next_days('SPY', 5)
    print("\nResult:")
    print(result)
    
    if 'error' in result:
        print("\nFAILED: Error returned.")
    else:
        print("\nSUCCESS: Predictions generated.")
        print(f"Last Close: {result['last_close']}")
        print(f"Predictions: {result['predictions']}")
        
except Exception as e:
    print(f"\nCRASHED: {e}")
    import traceback
    traceback.print_exc()
