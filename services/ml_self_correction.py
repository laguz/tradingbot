"""
ML Self-Correction Module - Online Learning from Mistakes

Implements feedback loop to learn from prediction errors:
1. Compares predictions to actual prices from database
2. Calculates error patterns
3. Applies corrections to future predictions
4. Triggers retraining when errors exceed threshold
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.mongodb_models import MLPredictionModel
from services.stock_data_service import get_historical_data
from services.ml_service_enhanced import predict_enhanced
from utils.logger import logger


def fetch_predictions_with_targets(ticker, days_back=30):
    """
    Fetch predictions from database where target date has passed.
    
    Args:
        ticker: Stock ticker
        days_back: How many days back to look
        
    Returns:
        DataFrame with predictions and actual prices
    """
    db = Session()
    try:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Query predictions where target_date is in the past
        predictions = db.query(MLPrediction).filter(
            and_(
                MLPrediction.ticker == ticker.upper(),
                MLPrediction.target_date < datetime.now(),
                MLPrediction.prediction_date >= cutoff_date,
                MLPrediction.actual_price.isnot(None)  # Only predictions with actual prices
            )
        ).all()
        
        if not predictions:
            logger.warning(f"No predictions with actual prices found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for pred in predictions:
            data.append({
                'prediction_date': pred.prediction_date,
                'target_date': pred.target_date,
                'predicted_price': pred.predicted_price,
                'actual_price': pred.actual_price,
                'model_version': pred.model_version,
                'confidence': pred.confidence
            })
        
        df = pd.DataFrame(data)
        
        # Calculate errors
        df['error'] = df['actual_price'] - df['predicted_price']
        df['error_pct'] = (df['error'] / df['actual_price']) * 100
        df['abs_error'] = df['error'].abs()
        df['abs_error_pct'] = df['error_pct'].abs()
        
        logger.info(f"Fetched {len(df)} predictions with actuals for {ticker}")
        
        return df
    finally:
        db.close()


def backfill_actual_prices(ticker, days_back=30):
    """
    Backfill actual prices for predictions in database.
    
    Args:
        ticker: Stock ticker
        days_back: How many days back to backfill
        
    Returns:
        Number of records updated
    """
    db = Session()
    try:
        # Get predictions without actual prices
        cutoff_date = datetime.now() - timedelta(days=days_back)
        predictions = db.query(MLPrediction).filter(
            and_(
                MLPrediction.ticker == ticker.upper(),
                MLPrediction.target_date < datetime.now(),
                MLPrediction.prediction_date >= cutoff_date,
                MLPrediction.actual_price.is_(None)
            )
        ).all()
        
        if not predictions:
            logger.info(f"No predictions to backfill for {ticker}")
            return 0
        
        logger.info(f"Backfilling {len(predictions)} predictions for {ticker}")
        
        # Fetch historical data
        df = get_historical_data(ticker, '3m', use_cache=True)
        if df.empty:
            logger.error(f"Could not fetch historical data for {ticker}")
            return 0
        
        updated_count = 0
        
        for pred in predictions:
            target_date = pred.target_date.date()
            
            # Find matching date in historical data
            if target_date in df.index.date:
                actual_price = float(df.loc[df.index.date == target_date, 'Close'].iloc[0])
                pred.actual_price = actual_price
                updated_count += 1
        
        db.commit()
        logger.info(f"Updated {updated_count} predictions with actual prices")
        
        return updated_count
    finally:
        db.close()


def analyze_prediction_errors(ticker, days_back=30):
    """
    Analyze prediction error patterns.
    
    Args:
        ticker: Stock ticker
        days_back: Days to analyze
        
    Returns:
        Dict with error analysis
    """
    df = fetch_predictions_with_targets(ticker, days_back)
    
    if df.empty:
        return {'error': 'No data available'}
    
    # Calculate metrics
    mae = df['abs_error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    mape = df['abs_error_pct'].mean()
    
    # Bias detection (consistent over/under prediction)
    mean_error = df['error'].mean()
    bias = 'over-predicting' if mean_error > 0 else 'under-predicting'
    
    # Trend in errors over time
    df_sorted = df.sort_values('prediction_date')
    recent_mae = df_sorted.tail(10)['abs_error'].mean()
    older_mae = df_sorted.head(10)['abs_error'].mean() if len(df_sorted) >= 20 else mae
    
    improving = recent_mae < older_mae
    
    analysis = {
        'ticker': ticker,
        'predictions_analyzed': len(df),
        'date_range': {
            'start': df['prediction_date'].min().strftime('%Y-%m-%d'),
            'end': df['prediction_date'].max().strftime('%Y-%m-%d')
        },
        'metrics': {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2)
        },
        'bias': {
            'direction': bias,
            'magnitude': round(abs(mean_error), 2),
            'magnitude_pct': round((mean_error / df['actual_price'].mean()) * 100, 2)
        },
        'trend': {
            'improving': improving,
            'recent_mae': round(recent_mae, 2),
            'older_mae': round(older_mae, 2),
            'change_pct': round(((recent_mae - older_mae) / older_mae) * 100, 1) if older_mae > 0 else 0
        },
        'worst_predictions': df.nlargest(3, 'abs_error')[
            ['target_date', 'predicted_price', 'actual_price', 'error']
        ].to_dict('records')
    }
    
    logger.info(f"Error analysis complete for {ticker}: MAE={mae:.2f}, Bias={bias}")
    
    return analysis


def apply_bias_correction(prediction, ticker, correction_factor=0.5):
    """
    Apply bias correction to a prediction based on historical errors.
    
    Args:
        prediction: Raw prediction value
        ticker: Stock ticker
        correction_factor: How much to correct (0-1, default 0.5)
        
    Returns:
        Corrected prediction
    """
    # Get recent error analysis
    analysis = analyze_prediction_errors(ticker, days_back=14)
    
    if 'error' in analysis:
        return prediction  # No correction if no data
    
    bias_pct = analysis['bias']['magnitude_pct']
    
    # Apply correction
    if analysis['bias']['direction'] == 'over-predicting':
        correction = prediction * (bias_pct / 100) * correction_factor
        corrected = prediction - correction
        logger.info(f"Applied downward correction: ${prediction:.2f} → ${corrected:.2f}")
    else:
        correction = prediction * (bias_pct / 100) * correction_factor
        corrected = prediction + correction
        logger.info(f"Applied upward correction: ${prediction:.2f} → ${corrected:.2f}")
    
    return corrected


def predict_with_self_correction(ticker, days=5, apply_correction=True):
    """
    Make prediction with automatic bias correction based on past errors.
    
    Args:
        ticker: Stock ticker
        days: Number of days to predict
        apply_correction: Whether to apply bias correction
        
    Returns:
        Dict with corrected predictions
    """
    logger.info(f"Predicting {ticker} with self-correction")
    
    # Step 1: Backfill any missing actual prices
    backfill_actual_prices(ticker, days_back=30)
    
    # Step 2: Make raw prediction
    result = predict_enhanced(ticker, days=days)
    
    if 'error' in result:
        return result
    
    # Step 3: Analyze errors
    error_analysis = analyze_prediction_errors(ticker, days_back=14)
    
    # Step 4: Apply correction if requested
    if apply_correction and 'error' not in error_analysis:
        corrected_predictions = []
        original_predictions = result['predictions'].copy()
        
        for pred in original_predictions:
            corrected = apply_bias_correction(pred, ticker, correction_factor=0.5)
            corrected_predictions.append(corrected)
        
        result['predictions'] = corrected_predictions
        result['original_predictions'] = original_predictions
        result['correction_applied'] = True
        result['error_analysis'] = error_analysis
    else:
        result['correction_applied'] = False
        result['correction_reason'] = 'Disabled or insufficient data'
    
    logger.info(f"Self-correcting prediction complete for {ticker}")
    
    return result


def trigger_adaptive_retraining(ticker, error_threshold_mae=5.0):
    """
    Check if model needs retraining based on recent errors.
    
    Args:
        ticker: Stock ticker
        error_threshold_mae: MAE threshold to trigger retrain (default $5)
        
    Returns:
        Tuple (should_retrain: bool, reason: str)
    """
    analysis = analyze_prediction_errors(ticker, days_back=14)
    
    if 'error' in analysis:
        return False, "Insufficient data for analysis"
    
    mae = analysis['metrics']['mae']
    
    # Trigger if MAE exceeds threshold
    if mae > error_threshold_mae:
        return True, f"MAE ({mae:.2f}) exceeds threshold ({error_threshold_mae})"
    
    # Trigger if performance degrading
    if not analysis['trend']['improving']:
        change_pct = abs(analysis['trend']['change_pct'])
        if change_pct > 20:  # 20% worse
            return True, f"Performance degraded by {change_pct:.1f}%"
    
    return False, "Model performance acceptable"


def auto_retrain_if_needed(ticker):
    """
    Automatically retrain model if errors are high.
    
    Args:
        ticker: Stock ticker
        
    Returns:
        Dict with retrain status
    """
    should_retrain, reason = trigger_adaptive_retraining(ticker)
    
    if should_retrain:
        logger.info(f"Triggering auto-retrain for {ticker}: {reason}")
        
        # Retrain with latest data
        result = predict_enhanced(ticker, days=5, force_retrain=True)
        
        return {
            'retrained': True,
            'reason': reason,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        }
    else:
        logger.info(f"No retrain needed for {ticker}: {reason}")
        return {
            'retrained': False,
            'reason': reason
        }
