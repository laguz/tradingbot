
from models.mongodb_models import StockDataModel, MLPredictionModel
from services.ml_evaluation import save_prediction_to_db
from datetime import datetime, timedelta
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)

def run_test():
    ticker = "TEST_IMMEDIATE_ACTUAL"
    
    # 1. Setup: Create Dummy Stock Data for "Tomorrow" (simulating we moved fwd in time or predicting past)
    # Actually, let's use a date for which we insert stock data manually
    test_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=30)
    
    StockDataModel.get_collection().delete_many({'symbol': ticker})
    MLPredictionModel.get_collection().delete_many({'symbol': ticker})
    
    # Insert stock data
    StockDataModel.upsert_daily(
        symbol=ticker,
        date=test_date,
        open_price=100, high=110, low=90, close=105.5, volume=1000
    )
    print(f"Inserted stock data for {test_date}: Close=105.5")
    
    # 2. Run Prediction Saving for that date
    # This should pick up the actual price of 105.5 immediately
    save_prediction_to_db(
        ticker=ticker,
        target_dates=[test_date],
        predicted_prices=[104.0],
        model_version="v-check",
        features_used=[],
        confidence_scores=[0.9]
    )
    
    # 3. Verify
    doc = MLPredictionModel.get_collection().find_one({'symbol': ticker, 'target_date': test_date})
    
    print(f"Saved Prediction: Predicted={doc.get('predicted_price')}, Actual={doc.get('actual_price')}")
    
    if doc.get('actual_price') == 105.5:
        print("\nSUCCESS: Actual price was populated immediately!")
    else:
        print("\nFAILURE: Actual price was NOT populated.")

    # Cleanup
    StockDataModel.delete_by_symbol(ticker)
    MLPredictionModel.get_collection().delete_many({'symbol': ticker})

if __name__ == "__main__":
    run_test()
