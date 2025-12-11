"""
Verify the backfill fix worked by checking the database.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import MLPredictionModel
from datetime import datetime

def verify_fix():
    """Verify predictions now have actual_price."""
    
    print("=" * 80)
    print("VERIFICATION - Backfill Fix")
    print("=" * 80)
    
    # Initialize
    init_db()
    collection = MLPredictionModel.get_collection()
    
    # Get all predictions for 2025-12-10
    predictions = list(collection.find({
        'target_date': {
            '$gte': datetime(2025, 12, 10, 0, 0, 0),
            '$lt': datetime(2025, 12, 11, 0, 0, 0)
        }
    }))
    
    print(f"\nPredictions for 2025-12-10: {len(predictions)}")
    
    with_actual = 0
    without_actual = 0
    
    for pred in predictions:
        symbol = pred['symbol']
        predicted = pred['predicted_price']
        actual = pred.get('actual_price')
        
        if actual is not None:
            with_actual += 1
            error = abs(actual - predicted)
            error_pct = (error / actual) * 100
            print(f"  ✅ {symbol}: Predicted ${predicted:.2f}, Actual ${actual:.2f}, Error ${error:.2f} ({error_pct:.1f}%)")
        else:
            without_actual += 1
            print(f"  ❌ {symbol}: Predicted ${predicted:.2f}, Actual: None")
    
    print(f"\nSummary:")
    print(f"  With actual_price: {with_actual}")
    print(f"  Without actual_price: {without_actual}")
    
    if without_actual == 0:
        print(f"\n✅ SUCCESS! All predictions have been backfilled!")
    else:
        print(f"\n⚠️  {without_actual} predictions still need backfill")
    
    # Now check overall stats
    all_preds = list(collection.find({}))
    total_with = sum(1 for p in all_preds if p.get('actual_price') is not None)
    total_without = sum(1 for p in all_preds if p.get('actual_price') is None)
    
    print(f"\nOverall Database Stats:")
    print(f"  Total predictions: {len(all_preds)}")
    print(f"  With actual_price: {total_with}")
    print(f"  Without actual_price: {total_without}")

if __name__ == '__main__':
    verify_fix()
