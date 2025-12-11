"""
Check what predictions exist and their dates.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import MLPredictionModel
from datetime import datetime, timedelta

def check_predictions():
    """Check predictions in the database."""
    
    print("=" * 80)
    print("PREDICTION DATABASE ANALYSIS")
    print("=" * 80)
    
    # Initialize
    init_db()
    collection = MLPredictionModel.get_collection()
    
    # Get all predictions
    all_predictions = list(collection.find({}).sort('target_date', 1))
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    
    # Group by date status
    now = datetime.now()
    past = []
    future = []
    
    for pred in all_predictions:
        target = pred['target_date']
        if target < now:
            past.append(pred)
        else:
            future.append(pred)
    
    print(f"\nPast predictions (target_date < now): {len(past)}")
    print(f"Future predictions (target_date >= now): {len(future)}")
    
    # Check actual_price status
    with_actual = sum(1 for p in all_predictions if p.get('actual_price') is not None)
    without_actual = sum(1 for p in all_predictions if p.get('actual_price') is None)
    
    print(f"\nWith actual_price: {with_actual}")
    print(f"Without actual_price: {without_actual}")
    
    # Show past predictions without actual_price
    past_no_actual = [p for p in past if p.get('actual_price') is None]
    print(f"\n❌ Past predictions WITHOUT actual_price: {len(past_no_actual)}")
    
    if past_no_actual:
        print("\nSample past predictions needing backfill:")
        for p in past_no_actual[:10]:
            print(f"  {p['symbol']} - Target: {p['target_date'].strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Predicted: ${p['predicted_price']:.2f} "
                  f"Actual: {p.get('actual_price', 'None')}")
    
    # Show date range
    if all_predictions:
        earliest = min(p['target_date'] for p in all_predictions)
        latest = max(p['target_date'] for p in all_predictions)
        print(f"\nDate range:")
        print(f"  Earliest target: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Latest target: {latest.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    check_predictions()
