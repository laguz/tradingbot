"""
Test the fixed backfill functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from services.ml_evaluation import backfill_actual_prices
from models.mongodb_models import MLPredictionModel
from datetime import datetime, timedelta
from utils.logger import logger


def test_backfill():
    """Test backfill actual prices."""
    
    print("=" * 80)
    print("Testing ML Backfill with Fixed Date Matching")
    print("=" * 80)
    
    # Initialize database
    init_db()
    
    # Check how many predictions need backfill
    collection = MLPredictionModel.get_collection()
    cutoff = datetime.now() - timedelta(days=90)
    
    pending = list(collection.find({
        'actual_price': None,
        'target_date': {'$lt': datetime.now()},
        'prediction_date': {'$gte': cutoff}
    }).limit(10))  # Show first 10
    
    print(f"\nFound {len(pending)} predictions needing backfill (showing first 10)")
    
    if pending:
        print("\nSample predictions to backfill:")
        print(f"{'Symbol':<10} {'Target Date':<15} {'Predicted Price':<18}")
        print("-" * 80)
        for p in pending[:5]:
            print(f"{p['symbol']:<10} {p['target_date'].strftime('%Y-%m-%d'):<15} ${p['predicted_price']:<17.2f}")
    
    # Run backfill
    print("\n" + "=" * 80)
    print("Running backfill...")
    print("=" * 80)
    
    updated_count = backfill_actual_prices()
    
    print(f"\n✅ Backfill complete!")
    print(f"Updated {updated_count} predictions with actual prices")
    
    # Show some results
    if updated_count > 0:
        filled = list(collection.find({
            'actual_price': {'$ne': None}
        }).sort('target_date', -1).limit(10))
        
        print(f"\nRecently filled predictions (showing last 10):")
        print(f"{'Symbol':<10} {'Target Date':<15} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
        print("-" * 80)
        
        for p in filled:
            predicted = p.get('predicted_price', 0)
            actual = p.get('actual_price', 0)
            error = abs(predicted - actual) if actual else 0
            error_pct = (error / actual * 100) if actual else 0
            
            print(f"{p['symbol']:<10} {p['target_date'].strftime('%Y-%m-%d'):<15} "
                  f"${predicted:<14.2f} ${actual:<14.2f} {error_pct:<14.2f}%")


if __name__ == '__main__':
    test_backfill()
