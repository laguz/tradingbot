"""
Daily Stock Data Sync Job

Automatically updates stock OHLCV data for active tickers and market indexes.
Should be run daily after market close (4:30 PM ET).
"""

import os
import sys
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.stock_data_service import update_ticker_data, get_cache_status
from utils.logger import logger
from config import get_config

config = get_config()

# Tickers to always keep updated
MARKET_INDEXES = ['SPY', 'QQQ', 'VIX']
COMMON_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']


def get_active_tickers():
    """
    Get list of tickers that have been recently used for ML predictions.
    
    Returns:
        List of unique ticker symbols
    """
    try:
        from models.mongodb_models import MLPredictionModel
        
        # Get predictions from last 30 days
        recent_predictions = MLPredictionModel.find_recent(days=30, limit=1000)
        
        # Extract unique symbols
        symbols = set()
        for pred in recent_predictions:
            if 'symbol' in pred:
                symbols.add(pred['symbol'].upper())
        
        logger.info(f"Found {len(symbols)} active tickers from recent predictions")
        return list(symbols)
        
    except Exception as e:
        logger.warning(f"Could not fetch active tickers from DB: {e}")
        return []


def sync_all_tickers():
    """
    Update data for all active tickers and market indexes.
    
    Returns:
        Dict with sync results
    """
    logger.info("=" * 60)
    logger.info("Starting daily stock data synchronization")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Combine all tickers to update
    active_tickers = get_active_tickers()
    all_tickers = list(set(MARKET_INDEXES + COMMON_TICKERS + active_tickers))
    
    logger.info(f"Total tickers to update: {len(all_tickers)}")
    logger.info(f"Market indexes: {MARKET_INDEXES}")
    logger.info(f"Common tickers: {COMMON_TICKERS}")
    logger.info(f"Active tickers: {active_tickers[:10]}..." if len(active_tickers) > 10 else f"Active tickers: {active_tickers}")
    
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'total': len(all_tickers)
    }
    
    for ticker in all_tickers:
        try:
            logger.info(f"Updating {ticker}...")
            
            # Check cache status first
            status = get_cache_status(ticker)
            
            if status.get('status') == 'fresh':
                logger.info(f"  {ticker}: Already up to date ({status.get('record_count')} records)")
                results['skipped'].append(ticker)
                continue
            
            # Update the ticker
            success = update_ticker_data(ticker, days=5)
            
            if success:
                # Get updated status
                new_status = get_cache_status(ticker)
                logger.info(f"  {ticker}: Updated successfully ({new_status.get('record_count')} records, latest: {new_status.get('latest_date')})")
                results['success'].append(ticker)
            else:
                logger.warning(f"  {ticker}: Update failed")
                results['failed'].append(ticker)
                
        except Exception as e:
            logger.error(f"  {ticker}: Error during update - {e}")
            results['failed'].append(ticker)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Daily sync complete!")
    logger.info(f"  Success: {len(results['success'])}")
    logger.info(f"  Skipped (already fresh): {len(results['skipped'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    if results['failed']:
        logger.warning(f"  Failed tickers: {results['failed']}")
    logger.info("=" * 60)
    
    return results


if __name__ == '__main__':
    try:
        # Initialize database connection
        from database import init_db
        init_db()
        
        # Run sync
        results = sync_all_tickers()
        
        # Exit with error code if any failures
        if results['failed']:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error in daily sync: {e}")
        sys.exit(1)
