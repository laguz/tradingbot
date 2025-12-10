"""
Create a test prediction for a past date and verify backfill works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import MLPredictionModel
from services.ml_evaluation import backfill_actual_prices
from datetime import datetime, timedelta

def test_backfill_with_old_prediction():
    """Create an old prediction and test backfill."""
    
    print("=" * 80)
    print("Testing Backfill with Past Date Prediction")
    print("=" * 80)
    
    init_db()
    
    # Create a test prediction for last week
    target_date = datetime.now() - timedelta(days=7)
    prediction_date = datetime.now() - timedelta(days=8)
    
    print(f"\n1. Creating test prediction...")
    print(f"   Symbol: SPY")
    print(f"   Prediction date: {prediction_date.strftime('%Y-%m-%d')}")
    print(f"   Target date: {target_date.strftime('%Y-%m-%d')}")
    print(f"   Predicted price: $600.00")
    
    pred_id = MLPredictionModel.insert(
        symbol='SPY',
        prediction_date=prediction_date,
        target_date=target_date,
        predicted_price=600.00,
        model_version='test',
        features_used=['test'],
        confidence=0.85,
        actual_price=None  # Not filled yet
    )
    
    print(f"   ✓ Created prediction")
    
    # Query it back by symbol and target_date
    collection = MLPredictionModel.get_collection()
    pred = collection.find_one({
        'symbol': 'SPY',
        'target_date': target_date
    })
    
    if not pred:
        print("   ❌ Could not find created prediction!")
        return
    
    pred_id = pred['_id']
    print(f"   Prediction ID: {pred_id}")
    
    print(f"\n2. Verifying prediction created...")
    print(f"   Actual price (before backfill): {pred.get('actual_price', 'None')}")
    
    # Run backfill
    print(f"\n3. Running backfill...")
    updated_count = backfill_actual_prices()
    
    print(f"   ✓ Backfill complete: {updated_count} predictions updated")
    
    # Check if it was updated
    pred_after = collection.find_one({'_id': pred_id})
    
    print(f"\n4. Checking result...")
    print(f"   Predicted price: ${pred_after['predicted_price']:.2f}")
    print(f"   Actual price (after backfill): {pred_after.get('actual_price', 'None')}")
    
    if pred_after.get('actual_price') is not None:
        actual = pred_after['actual_price']
        predicted = pred_after['predicted_price']
        error = abs(actual - predicted)
        error_pct = (error / actual * 100) if actual else 0
        
        print(f"\n   ✅ SUCCESS! Backfill worked!")
        print(f"   Actual price filled: ${actual:.2f}")
        print(f"   Error: ${error:.2f} ({error_pct:.2f}%)")
    else:
        print(f"\n   ❌ FAILED! Actual price still None")
        print(f"   This means backfill couldn't find data for {target_date.strftime('%Y-%m-%d')}")
    
    # Clean up test prediction
    print(f"\n5. Cleaning up test prediction...")
    collection.delete_one({'_id': pred_id})
    print(f"   ✓ Test prediction deleted")

if __name__ == '__main__':
    test_backfill_with_old_prediction()
