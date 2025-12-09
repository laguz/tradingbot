#!/usr/bin/env python3
"""
Daily ML Model Maintenance Job

Runs automated tasks for ML model maintenance:
- Backfill actual prices for past predictions
- Check model performance and trigger retraining if needed
- Log performance metrics

Usage:
    python jobs/daily_ml_maintenance.py
    
Or set up as cron job:
    0 2 * * * cd /path/to/tradingbot && /path/to/venv/bin/python jobs/daily_ml_maintenance.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ml_evaluation import backfill_actual_prices, get_model_performance
from services.ml_market_context import should_retrain_model
from services.ml_service import predict_next_days
from database import get_session
from models.db_models import MLPrediction
from utils.logger import logger
from datetime import datetime, timedelta
import json


def run_backfill():
    """Backfill actual prices for predictions"""
    logger.info("=" * 80)
    logger.info("Starting daily backfill job")
    logger.info("=" * 80)
    
    try:
        updated_count = backfill_actual_prices()
        logger.info(f"✓ Backfilled {updated_count} predictions with actual prices")
        return updated_count
    except Exception as e:
        logger.error(f"✗ Backfill failed: {e}")
        return 0


def check_model_performance():
    """Check performance and trigger retraining if needed"""
    logger.info("\n" + "=" * 80)
    logger.info("Checking model performance")
    logger.info("=" * 80)
    
    session = get_session()
    
    try:
        # Get list of unique tickers with predictions
        tickers = session.query(MLPrediction.symbol).distinct().all()
        tickers = [t[0] for t in tickers]
        
        logger.info(f"Found {len(tickers)} tickers with predictions")
        
        retrain_list = []
        
        for ticker in tickers:
            # Get performance for last 30 days
            perf = get_model_performance(ticker=ticker, days=30)
            
            if 'error' in perf:
                logger.warning(f"  {ticker}: No performance data available")
                continue
            
            # Check if retraining needed
            should_retrain, reason = should_retrain_model(ticker, perf)
            
            if should_retrain:
                logger.warning(f"  {ticker}: Retraining recommended - {reason}")
                retrain_list.append((ticker, reason))
            else:
                logger.info(f"  {ticker}: Performance OK (MAE: ${perf['mae']:.2f}, Dir Acc: {perf['directional_accuracy']:.1f}%)")
        
        return retrain_list
        
    except Exception as e:
        logger.error(f"Error checking performance: {e}")
        return []
    finally:
        session.close()


def auto_retrain_models(retrain_list, max_retrains=3):
    """Automatically retrain models that need it"""
    if not retrain_list:
        logger.info("\n✓ No models need retraining")
        return
    
    logger.info(f"\n" + "=" * 80)
    logger.info(f"Auto-retraining {min(len(retrain_list), max_retrains)} models")
    logger.info("=" * 80)
    
    for i, (ticker, reason) in enumerate(retrain_list[:max_retrains]):
        logger.info(f"\n  [{i+1}/{min(len(retrain_list), max_retrains)}] Retraining {ticker}...")
        logger.info(f"    Reason: {reason}")
        
        try:
            # Force retrain by making a prediction
            result = predict_next_days(ticker, days=5, force_retrain=True)
            
            if 'error' in result:
                logger.error(f"    ✗ Retraining failed: {result['error']}")
            else:
                logger.info(f"    ✓ Retraining successful (model v{result['model_version']})")
                
        except Exception as e:
            logger.error(f"    ✗ Retraining crashed: {e}")


def log_daily_summary(backfill_count, retrain_list):
    """Log summary of daily maintenance"""
    logger.info("\n" + "=" * 80)
    logger.info("DAILY MAINTENANCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Predictions backfilled: {backfill_count}")
    logger.info(f"Models needing retrain: {len(retrain_list)}")
    
    if retrain_list:
        logger.info("\nRetrain recommendations:")
        for ticker, reason in retrain_list:
            logger.info(f"  - {ticker}: {reason}")
    
    logger.info("=" * 80 + "\n")


def main():
    """Main job execution"""
    logger.info("Starting daily ML maintenance job")
    
    # Step 1: Backfill actual prices
    backfill_count = run_backfill()
    
    # Step 2: Check model performance
    retrain_list = check_model_performance()
    
    # Step 3: Auto-retrain if needed (limit to 3 per day to avoid overload)
    auto_retrain_models(retrain_list, max_retrains=3)
    
    # Step 4: Log summary
    log_daily_summary(backfill_count, retrain_list)
    
    logger.info("✓ Daily ML maintenance job complete")


if __name__ == '__main__':
    main()
