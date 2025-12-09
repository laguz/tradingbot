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
    start_date = end_date - timedelta(days=days)
    
    # Try to get from cache first
    if use_cache:
        try:
            records = StockDataModel.find_by_symbol(
                ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if records and len(records) > 0:
                # Check if cache is recent enough (within last 2 days)
                latest_date = StockDataModel.get_latest_date(ticker)
                if latest_date and (datetime.now() - latest_date).days <= 2:
                    # Convert to DataFrame
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
                    
                    logger.info(f"Loaded {len(df)} records for {ticker} from database cache")
                    return df
                else:
                    logger.debug(f"Cache for {ticker} is stale (latest: {latest_date}), fetching from API")
            else:
                logger.debug(f"No cached data found for {ticker}, fetching from API")
        except Exception as e:
            logger.warning(f"Error reading from cache for {ticker}: {e}, falling back to API")
    
    # Fallback to API
    logger.info(f"Fetching {ticker} data from Tradier API ({timeframe})")
    df = get_raw_historical_data(ticker, timeframe)
    
    # Save to cache for future use
    if not df.empty:
        save_historical_data(ticker, df)
    
    return df


def update_ticker_data(ticker: str, days: int = 5) -> bool:
    """
    Update recent data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of recent days to update (default: 5)
        
    Returns:
        True if update successful
    """
    ticker = ticker.upper()
    
    try:
        # Get latest date in cache
        latest_date = StockDataModel.get_latest_date(ticker)
        
        if latest_date:
            # Fetch data since last cached date
            logger.info(f"Updating {ticker} from {latest_date} to present")
            df = get_raw_historical_data(ticker, '1m')  # Get 1 month to ensure we have overlap
        else:
            # No cache, do full sync
            logger.info(f"No existing data for {ticker}, performing full sync")
            return sync_ticker_data(ticker)
        
        if df.empty:
            logger.warning(f"No new data fetched for {ticker}")
            return False
        
        # Filter to only new data
        df = df[df.index > latest_date]
        
        if df.empty:
            logger.info(f"No new data to update for {ticker}")
            return True
        
        # Save new data
        saved_count = save_historical_data(ticker, df)
        logger.info(f"Updated {saved_count} new records for {ticker}")
        return saved_count > 0
        
    except Exception as e:
        logger.error(f"Error updating {ticker}: {e}")
        return False


def sync_ticker_data(ticker: str, full_years: int = 2) -> bool:
    """
    Perform full data synchronization for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        full_years: Number of years of historical data to fetch (default: 2)
        
    Returns:
        True if sync successful
    """
    ticker = ticker.upper()
    
    try:
        # Determine timeframe based on years requested
        if full_years >= 5:
            timeframe = '5y'
        elif full_years >= 2:
            timeframe = '2y'
        else:
            timeframe = '1y'
        
        logger.info(f"Starting full sync for {ticker} ({timeframe})")
        
        # Fetch from API
        df = get_raw_historical_data(ticker, timeframe)
        
        if df.empty:
            logger.error(f"No data fetched for {ticker}")
            return False
        
        # Clear existing data for this ticker
        deleted = StockDataModel.delete_by_symbol(ticker)
        if deleted > 0:
            logger.info(f"Deleted {deleted} existing records for {ticker}")
        
        # Save all data
        saved_count = save_historical_data(ticker, df)
        logger.info(f"Full sync complete for {ticker}: {saved_count} records saved")
        
        return saved_count > 0
        
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
