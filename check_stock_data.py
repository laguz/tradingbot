"""
Check what stock data we have for today's predictions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from models.mongodb_models import StockDataModel
from services.stock_data_service import get_historical_data
from datetime import datetime

def check_stock_data():
    """Check stock data availability."""
    
    print("=" * 80)
    print("STOCK DATA AVAILABILITY")
    print("=" * 80)
    
    # Initialize
    init_db()
    
    symbols = ['AAPL', 'SPY', 'RIOT', 'TSLA']
    target_date = datetime(2025, 12, 10)
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        print(f"  Target date: {target_date.strftime('%Y-%m-%d')}")
        
        # Try to get from DB
        price_from_db = StockDataModel.get_close_price(symbol, target_date)
        print(f"  Price from DB: {price_from_db}")
        
        # Get historical data
        df = get_historical_data(symbol, '1mo', use_cache=True)
        print(f"  Total records: {len(df)}")
        
        if not df.empty:
            print(f"  Latest date in DB: {df.index[-1]}")
            print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
            
            # Check if we have today's data
            import pandas as pd
            target_ts = pd.Timestamp(target_date.date())
            matching = df.index[df.index.normalize() == target_ts]
            
            if len(matching) > 0:
                print(f"  ✅ Has data for {target_date.strftime('%Y-%m-%d')}: ${df.loc[matching[0], 'Close']:.2f}")
            else:
                print(f"  ❌ No data for {target_date.strftime('%Y-%m-%d')}")
                
                # Check the closest previous date
                past_dates = df.index[df.index < target_ts]
                if len(past_dates) > 0:
                    closest_past = past_dates[-1]
                    days_diff = (target_ts - closest_past).days
                    print(f"  📅 Closest past date: {closest_past.date()} ({days_diff} days before)")
                    print(f"     Close price: ${df.loc[closest_past, 'Close']:.2f}")

if __name__ == '__main__':
    check_stock_data()
