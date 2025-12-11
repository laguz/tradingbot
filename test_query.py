"""
Test the exact query used in backfill_actual_prices.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import MLPredictionModel
from datetime import datetime, timedelta

def test_query():
    """Test the query logic from backfill."""
    
    print("=" * 80)
    print("TESTING BACKFILL QUERY")
    print("=" * 80)
    
    # Initialize
    init_db()
    collection = MLPredictionModel.get_collection()
    
    # Use the exact same query as in backfill_actual_prices
    cutoff = datetime.now() - timedelta(days=90)
    
    print(f"\nCurrent time: {datetime.now()}")
    print(f"Cutoff date: {cutoff}")
    
    query = {
        'actual_price': None,
        'target_date': {'$lt': datetime.now()},
        'prediction_date': {'$gte': cutoff}
    }
    
    print(f"\nQuery: {query}")
    
    pending = list(collection.find(query))
    
    print(f"\nFound {len(pending)} predictions")
    
    for p in pending:
        print(f"  {p['symbol']} - Target: {p['target_date']} "
              f"Predicted: ${p['predicted_price']:.2f} "
              f"Actual: {p.get('actual_price', 'None')}")
    
    # Now let's test if we can actually call backfill
    print("\n" + "=" * 80)
    print("TESTING BACKFILL FUNCTION")
    print("=" * 80)
    
    from services.ml_evaluation import backfill_actual_prices
    
    updated_count = backfill_actual_prices(auto_sync=False)
    
    print(f"\nBackfill returned: {updated_count} updates")
    
    # Check if any were actually updated
    print("\n" + "=" * 80)
    print("CHECKING RESULTS")
    print("=" * 80)
    
    # Re-query
    still_pending = list(collection.find(query))
    print(f"\nStill pending: {len(still_pending)}")
    
    # Check for any with actual_price
    with_actual = list(collection.find({
        'actual_price': {'$ne': None},
    }))
    
    print(f"Total with actual_price: {len(with_actual)}")
    
    if with_actual:
        print("\nPredictions with actual_price:")
        for p in with_actual[:5]:
            print(f"  {p['symbol']} - Target: {p['target_date'].strftime('%Y-%m-%d')} "
                  f"Predicted: ${p['predicted_price']:.2f} "
                  f"Actual: ${p.get('actual_price', 0):.2f}")

if __name__ == '__main__':
    test_query()
