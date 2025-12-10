"""
Debug backfill in detail to see why actual_price isn't updating.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import MLPredictionModel
from services.stock_data_service import get_historical_data
from datetime import datetime, timedelta
import pandas as pd

def debug_backfill():
    """Debug why backfill isn't working."""
    
    print("=" * 80)
    print("BACKFILL DEBUG - Step by Step")
    print("=" * 80)
    
    # Initialize
    init_db()
    collection = MLPredictionModel.get_collection()
    
    # Step 1: Find predictions needing backfill
    cutoff = datetime.now() - timedelta(days=90)
    yesterday = datetime.now() - timedelta(days=1)
    
    print(f"\n1. Searching for predictions...")
    print(f"   - Target date before: {yesterday.strftime('%Y-%m-%d')}")
    print(f"   - Prediction date after: {cutoff.strftime('%Y-%m-%d')}")
    
    pending = list(collection.find({
        'actual_price': None,
        'target_date': {'$lt': yesterday},
        'prediction_date': {'$gte': cutoff}
    }).limit(5))
    
    print(f"   ✓ Found {len(pending)} predictions needing backfill")
    
    if not pending:
        print("\n❌ No predictions found that need backfill!")
        print("   This could mean:")
        print("   - All predictions are already backfilled")
        print("   - Or all predictions are for today/future")
        
        # Check recent predictions
        recent = list(collection.find({}).sort('target_date', -1).limit(5))
        print(f"\n   Recent predictions:")
        for p in recent:
            print(f"   - {p['symbol']} target:{p['target_date'].strftime('%Y-%m-%d')} "
                  f"actual:{p.get('actual_price', 'None')}")
        return
    
    # Step 2: Try to get data for first prediction
    pred = pending[0]
    symbol = pred['symbol']
    target_date = pred['target_date']
    
    print(f"\n2. Testing with {symbol} (target: {target_date.strftime('%Y-%m-%d')})")
    
    # Get historical data
    print(f"\n3. Fetching historical data for {symbol}...")
    df = get_historical_data(symbol, '3m', use_cache=True)
    
    if df is None or df.empty:
        print(f"   ❌ No data returned for {symbol}")
        return
    
    print(f"   ✓ Got {len(df)} rows of data")
    print(f"   Index type: {type(df.index)}")
    print(f"   First date: {df.index[0]}")
    print(f"   Last date: {df.index[-1]}")
    
    # Step 4: Try to match date
    print(f"\n4. Matching target date {target_date.strftime('%Y-%m-%d')}...")
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print("   Converting to DatetimeIndex...")
        df.index = pd.to_datetime(df.index)
    
    # Normalize target date
    if hasattr(target_date, 'date'):
        target_date_only = target_date.date()
    else:
        target_date_only = target_date
    
    target_ts = pd.Timestamp(target_date_only)
    print(f"   Target as Timestamp: {target_ts}")
    
    # Try exact match
    matching_dates = df.index[df.index.normalize() == target_ts]
    
    if len(matching_dates) > 0:
        actual_price = float(df.loc[matching_dates[0], 'Close'])
        print(f"   ✅ FOUND exact match!")
        print(f"   Date: {matching_dates[0]}")
        print(f"   Close price: ${actual_price:.2f}")
        
        # Step 5: Try update
        print(f"\n5. Attempting database update...")
        print(f"   Prediction ID: {pred['_id']}")
        print(f"   Actual price: ${actual_price:.2f}")
        
        result = MLPredictionModel.update_actual_price(str(pred['_id']), actual_price)
        
        if result:
            print(f"   ✅ UPDATE SUCCESSFUL!")
            
            # Verify update
            updated = collection.find_one({'_id': pred['_id']})
            print(f"\n6. Verifying update...")
            print(f"   Predicted: ${updated['predicted_price']:.2f}")
            print(f"   Actual: ${updated.get('actual_price', 'None')}")
            
            if updated.get('actual_price') == actual_price:
                print(f"   ✅ VERIFIED - actual_price is in database!")
            else:
                print(f"   ❌ FAILED - actual_price not saved properly")
        else:
            print(f"   ❌ UPDATE FAILED - no documents modified")
    else:
        print(f"   ❌ No exact match")
        
        # Try nearest
        print(f"\n   Trying nearest date...")
        nearest_idx = df.index.searchsorted(target_ts)
        if nearest_idx < len(df):
            nearest_date = df.index[nearest_idx]
            days_diff = abs((nearest_date - target_ts).days)
            print(f"   Nearest: {nearest_date.date()} ({days_diff} days away)")
            
            if days_diff <= 3:
                actual_price = float(df.loc[nearest_date, 'Close'])
                print(f"   ✓ Close enough - using ${actual_price:.2f}")
            else:
                print(f"   ❌ Too far ({days_diff} days)")
        else:
            print(f"   ❌ No data after target date")

if __name__ == '__main__':
    debug_backfill()
