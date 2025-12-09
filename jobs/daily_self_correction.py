"""
Daily ML Self-Correction Job

Automatically:
1. Backfill actual prices for all tracked tickers
2. Analyze prediction errors
3. Apply corrections to future predictions
4. Retrain models with poor performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ml_self_correction import (
    backfill_actual_prices,
    analyze_prediction_errors,
    auto_retrain_if_needed
)
from sqlalchemy import distinct
from database import get_db
from models.db_models import MLPrediction
from utils.logger import logger
from datetime import datetime


# Tickers to monitor (add your actively traded tickers)
TRACKED_TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']


def run_daily_self_correction():
    """
    Run daily self-correction routine for all tickers.
    """
    logger.info("="*80)
    logger.info(f"Starting Daily ML Self-Correction - {datetime.now()}")
    logger.info("="*80)
    
    # Get all tickers with predictions
    db = get_db()
    db_tickers = db.query(distinct(MLPrediction.ticker)).all()
    all_tickers = list(set(TRACKED_TICKERS + [t[0] for t in db_tickers]))
    
    logger.info(f"Processing {len(all_tickers)} tickers: {all_tickers}")
    
    summary = {
        'total_tickers': len(all_tickers),
        'backfilled': 0,
        'retrained': 0,
        'errors_analyzed': 0,
        'failed': 0
    }
    
    for ticker in all_tickers:
        logger.info(f"\n--- Processing {ticker} ---")
        
        try:
            # Step 1: Backfill actual prices
            updated = backfill_actual_prices(ticker, days_back=30)
            if updated > 0:
                summary['backfilled'] += updated
                logger.info(f"✅ Backfilled {updated} predictions for {ticker}")
            
            # Step 2: Analyze errors
            analysis = analyze_prediction_errors(ticker, days_back=14)
            if 'error' not in analysis:
                summary['errors_analyzed'] += 1
                
                logger.info(f"📊 Error Analysis for {ticker}:")
                logger.info(f"   MAE: ${analysis['metrics']['mae']:.2f}")
                logger.info(f"   MAPE: {analysis['metrics']['mape']:.1f}%")
                logger.info(f"   Bias: {analysis['bias']['direction']} by {analysis['bias']['magnitude_pct']:.2f}%")
                logger.info(f"   Trend: {'improving' if analysis['trend']['improving'] else 'degrading'}")
            
            # Step 3: Check if retrain needed
            retrain_result = auto_retrain_if_needed(ticker)
            if retrain_result['retrained']:
                summary['retrained'] += 1
                logger.info(f"🔄 Retrained model for {ticker}: {retrain_result['reason']}")
            else:
                logger.info(f"✓ Model OK for {ticker}")
        
        except Exception as e:
            summary['failed'] += 1
            logger.error(f"❌ Failed to process {ticker}: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("Daily Self-Correction Summary")
    logger.info("="*80)
    logger.info(f"Total tickers processed: {summary['total_tickers']}")
    logger.info(f"Predictions backfilled: {summary['backfilled']}")
    logger.info(f"Errors analyzed: {summary['errors_analyzed']}")
    logger.info(f"Models retrained: {summary['retrained']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info("="*80)
    
    return summary


if __name__ == '__main__':
    run_daily_self_correction()
