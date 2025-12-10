"""
Stock Data Service

Manages OHLCV (Open, High, Low, Close, Volume) data storage and retrieval.
Provides caching layer between Tradier API and MongoDB database.
"""

import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional
from models.mongodb_models import StockDataModel
from services.tradier_service import get_raw_historical_data
from utils.logger import logger


def save_historical_data(ticker: str, df: pd.DataFrame) -> int:
    """
    Save historical OHLCV data to database.
    
    Args:
        ticker: Stock ticker symbol
        df: DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
        
    Returns:
        Number of records inserted
    """
    if df.empty:
        logger.warning(f"No data to save for {ticker}")
        return 0
    
    # Convert DataFrame to list of dicts for MongoDB
    records = []
    for date_idx, row in df.iterrows():
        records.append({
            'symbol': ticker.upper(),
            'date': date_idx.to_pydatetime() if hasattr(date_idx, 'to_pydatetime') else date_idx,
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume'])
        })
    
    try:
        StockDataModel.insert_many(records)
        logger.info(f"Saved {len(records)} OHLCV records for {ticker}")
        return len(records)
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")
        return 0


def get_historical_data(ticker: str, timeframe: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Get historical OHLCV data from database or API.
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Time period ('1m', '3m', '6m', '1y', '2y', '5y')
        use_cache: If True, try to use cached database data
        
    Returns:
        DataFrame with DatetimeIndex and OHLCV columns
    """
    ticker = ticker.upper()
    
    # Calculate required date range
    end_date = datetime.now()
    time_deltas = {'1m': 30, '3m': 90, '6m': 180, '1y': 365, '2y': 730, '5y': 1825}
    days = time_deltas.get(timeframe.lower(), 365)
    
    # We want data starting from this date
    req_start_date = end_date - timedelta(days=days)
    
    # -- SYNC LOGIC --
    
    # 1. Check existing coverage
    latest_date = StockDataModel.get_latest_date(ticker)
    earliest_date = StockDataModel.get_earliest_date(ticker)
    
    need_sync = False
    sync_mode = 'none' # 'full', 'update', 'backfill' (backfill treated as full for now)
    
    if latest_date is None:
        logger.info(f"No data for {ticker}, performing full sync")
        sync_mode = 'full'
        need_sync = True
    else:
        # Check if we have holes at the end (stale data)
        # Allow 1 day lag (e.g. running on Sat morning, Fri data exists)
        days_lag = (end_date - latest_date).days
        if days_lag > 1:
            logger.info(f"Data for {ticker} is stale ({days_lag} days old), updating")
            sync_mode = 'update'
            # If the lag is huge, we might just want to full sync to be safe/easy?
            # But let's try update first.
            need_sync = True
            
        # Check if we have holes at the start (need more history)
        if earliest_date > req_start_date + timedelta(days=5): # Buffer
            logger.info(f"Need more history for {ticker} (Have: {earliest_date.date()}, Need: {req_start_date.date()})")
            sync_mode = 'full' # Easiest to just re-sync full requested range
            need_sync = True
            
    if need_sync:
        if sync_mode == 'full':
            sync_ticker_data(ticker, full_years=(days // 365) + 1)
        elif sync_mode == 'update':
            update_ticker_data(ticker)
            
    # -- CACHE RETRIEVAL --
    
    try:
        records = StockDataModel.find_by_symbol(
            ticker,
            start_date=req_start_date,
            end_date=end_date
        )
        
        if records:
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df.set_index('Date', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.sort_index(inplace=True)
            
            logger.info(f"Returning {len(df)} records for {ticker} from database")
            return df
        else:
            logger.warning(f"No records found for {ticker} even after sync attempt")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error reading from DB for {ticker}: {e}")
        return pd.DataFrame()


def update_ticker_data(ticker: str, days: int = 5) -> bool:
    """
    Update recent data for a ticker.
    Fetches missing delta based on latest cached date.
    
    Args:
        ticker: Stock ticker symbol
        days: Fallback day count/Ignored in favor of smart delta
        
    Returns:
        True if update successful
    """
    ticker = ticker.upper()
    
    try:
        # Get latest date in cache
        latest_date = StockDataModel.get_latest_date(ticker)
        
        if latest_date:
            # Fetch data strictly AFTER last cached date
            # Start date = latest_date + 1 day
            start_date = latest_date + timedelta(days=1)
            
            # If start_date > today, we are up to date
            if start_date.date() > date.today():
                 logger.info(f"{ticker} is already up to date")
                 return True

            logger.info(f"Updating {ticker} from {start_date.date()} to present")
            # Use specific start date
            df = get_raw_historical_data(ticker, start_date=start_date)
            
        else:
            # No cache, do full sync
            logger.info(f"No existing data for {ticker}, performing full sync")
            return sync_ticker_data(ticker)
        
        if df.empty:
            logger.info(f"No new data found for {ticker} (delta update)")
            return True
        
        # Save NEW data only (guaranteed by date filter)
        saved_count = save_historical_data(ticker, df)
        logger.info(f"Updated {saved_count} new records for {ticker}")
        return saved_count > 0
        
    except Exception as e:
        logger.error(f"Error updating {ticker}: {e}")
        return False


def sync_ticker_data(ticker: str, full_years: int = 2) -> bool:
    """
    Perform full data synchronization for a ticker (Additive).
    Does NOT delete existing data. Ideally mostly used for cold starts.
    
    Args:
        ticker: Stock ticker symbol
        full_years: Number of years to fetch if completely empty
        
    Returns:
        True if sync successful
    """
    ticker = ticker.upper()
    
    try:
        # Determine timeframe
        if full_years >= 5: timeframe = '5y'
        elif full_years >= 2: timeframe = '2y'
        else: timeframe = '1y'
        
        logger.info(f"Starting sync for {ticker} ({timeframe})")
        
        # Fetch from API
        df = get_raw_historical_data(ticker, timeframe)
        
        if df.empty:
            logger.error(f"No data fetched for {ticker}")
            return False
            
        # Optimization: We could filter df against existing DB records here 
        # but save_historical_data (assuming upsert/ignore duplicates) handles it.
        # StockDataModel.insert_many handles duplicates gracefully.
        
        # Save all data (Additive)
        ordered_ids = StockDataModel.insert_many(
            [{'symbol': ticker.upper(),
              'date': idx, 
              'open': float(row['Open']), 
              'high': float(row['High']), 
              'low': float(row['Low']), 
              'close': float(row['Close']), 
              'volume': float(row['Volume'])
             } for idx, row in df.iterrows()]
        )
        
        saved_count = len(ordered_ids)
        logger.info(f"Sync complete for {ticker}: {saved_count} new records inserted")
        
        return True
        
    except Exception as e:
        logger.error(f"Error syncing {ticker}: {e}")
        return False


def get_cache_status(ticker: str) -> dict:
    """
    Get cache status information for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with cache information
    """
    ticker = ticker.upper()
    
    try:
        count = StockDataModel.count_records(ticker)
        latest_date = StockDataModel.get_latest_date(ticker)
        
        if latest_date:
            age_days = (datetime.now() - latest_date).days
            is_stale = age_days > 2
        else:
            age_days = None
            is_stale = True
        
        return {
            'ticker': ticker,
            'record_count': count,
            'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None,
            'age_days': age_days,
            'is_stale': is_stale,
            'status': 'fresh' if not is_stale else 'stale' if count > 0 else 'no_data'
        }
    except Exception as e:
        logger.error(f"Error getting cache status for {ticker}: {e}")
        return {
            'ticker': ticker,
            'error': str(e)
        }
