
from services.ml_service import predict_next_days
from models.mongodb_models import MLPredictionModel
from datetime import datetime, timedelta
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)

def verify_save():
    ticker = "TEST_SAVE_FINAL"
    
    # Clean up previous tests
    MLPredictionModel.get_collection().delete_many({'symbol': ticker})
    
    # Run prediction
    # Force retrain false so we don't need real data if we mock it, 
    # but `predict_next_days` might fail if no real data.
    # Instead, let's just mock the save call or inject a fake result.
    # Actually, let's call save_prediction_to_db directly with typical data 
    # since we already tested the model generation part previously.
    
    from services.ml_evaluation import save_prediction_to_db
    
    target_date = datetime.now() + timedelta(days=1)
    
    print(f"Saving prediction for {ticker} on {target_date}")
    save_prediction_to_db(
        ticker=ticker,
        target_dates=[target_date],
        predicted_prices=[150.0],
        model_version="v2.1-test",
        features_used=["test_feature"],
        confidence_scores=[0.95]
    )
    
    # Check if saved
    coll = MLPredictionModel.get_collection()
    
    # Normalization check: target_date in DB should be midnight
    expected_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    doc = coll.find_one({
        'symbol': ticker, 
        'target_date': expected_date
    })
    
    if doc:
        print("\nSUCCESS: Prediction found in database!")
        print(f"Stored Price: {doc['predicted_price']}")
        print(f"Stored Date: {doc['target_date']}")
    else:
        print("\nFAILURE: Prediction NOT found in database.")
        
    # Cleanup
    MLPredictionModel.get_collection().delete_many({'symbol': ticker})

if __name__ == "__main__":
    verify_save()
