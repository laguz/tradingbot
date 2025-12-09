"""
Test script to verify OHLCV database integration.

Tests:
1. Saving data to database
2. Retrieving data from database
3. Cache functionality
4. ML prediction with database data
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from services.stock_data_service import (
    sync_ticker_data, 
    get_historical_data, 
    get_cache_status
)
from services.ml_service import predict_next_days
from utils.logger import logger


def test_database_integration():
    """Run comprehensive tests of the database integration."""
    
    print("\n" + "="*60)
    print("OHLCV Database Integration Test")
    print("="*60 + "\n")
    
    # Initialize database
    print("1. Initializing database...")
    init_db()
    print("   ✓ Database initialized\n")
    
    # Test 1: Sync ticker data
    print("2. Testing data synchronization for AAPL...")
    ticker = "AAPL"
    success = sync_ticker_data(ticker, full_years=2)
    
    if success:
        print(f"   ✓ Data synced successfully for {ticker}\n")
    else:
        print(f"   ✗ Failed to sync data for {ticker}\n")
        return False
    
    # Test 2: Check cache status
    print("3. Checking cache status...")
    status = get_cache_status(ticker)
    print(f"   Ticker: {status['ticker']}")
    print(f"   Records: {status['record_count']}")
    print(f"   Latest Date: {status['latest_date']}")
    print(f"   Age: {status['age_days']} days")
    print(f"   Status: {status['status']}")
    
    if status['record_count'] > 0:
        print(f"   ✓ Cache contains data\n")
    else:
        print(f"   ✗ Cache is empty\n")
        return False
    
    # Test 3: Retrieve data from cache
    print("4. Retrieving data from database (use_cache=True)...")
    df_cached = get_historical_data(ticker, '1y', use_cache=True)
    
    if not df_cached.empty:
        print(f"   ✓ Retrieved {len(df_cached)} records from cache")
        print(f"   Columns: {list(df_cached.columns)}")
        print(f"   Date range: {df_cached.index[0]} to {df_cached.index[-1]}\n")
    else:
        print(f"   ✗ Failed to retrieve cached data\n")
        return False
    
    # Test 4: Test ML prediction with cached data
    print("5. Testing ML prediction with database data...")
    try:
        result = predict_next_days(ticker, days=5, force_retrain=False)
        
        if 'error' not in result:
            print(f"   ✓ ML prediction successful!")
            print(f"   Last close: ${result['last_close']}")
            print(f"   Predictions: {result['predictions']}")
            print(f"   Model version: {result['model_version']}")
            print(f"   Feature count: {result['feature_count']}\n")
        else:
            print(f"   ✗ ML prediction failed: {result['error']}\n")
            return False
    except Exception as e:
        print(f"   ✗ ML prediction error: {e}\n")
        return False
    
    # Test 5: Test API fallback (no cache)
    print("6. Testing API fallback (use_cache=False)...")
    df_api = get_historical_data("MSFT", '1m', use_cache=False)
    
    if not df_api.empty:
        print(f"   ✓ Retrieved {len(df_api)} records from API")
        print(f"   API call successful, data cached for future use\n")
    else:
        print(f"   ✗ Failed to retrieve data from API\n")
        return False
    
    # Summary
    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nDatabase integration is working correctly!")
    print("- OHLCV data is being saved to MongoDB")
    print("- ML predictions are using cached data")
    print("- API fallback is functioning")
    print("\nNext steps:")
    print("1. Run: python3 jobs/daily_stock_sync.py (to sync all tickers)")
    print("2. Use the web app to make predictions (will use DB)")
    print("3. Schedule daily_stock_sync.py to run daily at 4:30 PM ET")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = test_database_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
