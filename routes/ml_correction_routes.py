"""
ML Self-Correction API Routes

Provides endpoints for:
- Backfilling actual prices
- Error analysis
- Self-correcting predictions
- Auto-retraining
"""

from flask import Blueprint, jsonify, request
from services.ml_self_correction import (
    backfill_actual_prices,
    analyze_prediction_errors,
    predict_with_self_correction,
    auto_retrain_if_needed,
    fetch_predictions_with_targets
)
from utils.logger import logger

ml_correction = Blueprint('ml_correction', __name__)


@ml_correction.route('/api/ml/backfill/<ticker>', methods=['POST'])
def backfill_prices(ticker):
    """
    Backfill actual prices for predictions.
    
    POST /api/ml/backfill/SPY
    Body: {"days_back": 30}
    """
    try:
        days_back = request.json.get('days_back', 30) if request.json else 30
        
        updated = backfill_actual_prices(ticker, days_back)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'records_updated': updated,
            'days_back': days_back
        })
    except Exception as e:
        logger.error(f"Backfill error: {e}")
        return jsonify({'error': str(e)}), 500


@ml_correction.route('/api/ml/error-analysis/<ticker>')
def get_error_analysis(ticker):
    """
    Get prediction error analysis.
    
    GET /api/ml/error-analysis/SPY?days_back=30
    """
    try:
        days_back = int(request.args.get('days_back', 30))
        
        analysis = analyze_prediction_errors(ticker, days_back)
        
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


@ml_correction.route('/api/ml/predict-corrected/<ticker>')
def predict_corrected(ticker):
    """
    Get self-corrected predictions.
    
    GET /api/ml/predict-corrected/SPY?days=5&correction=true
    """
    try:
        days = int(request.args.get('days', 5))
        apply_correction = request.args.get('correction', 'true').lower() == 'true'
        
        result = predict_with_self_correction(ticker, days, apply_correction)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Corrected prediction failed: {e}")
        return jsonify({'error': str(e)}), 500


@ml_correction.route('/api/ml/auto-retrain/<ticker>', methods=['POST'])
def retrain_if_needed(ticker):
    """
    Check and retrain if model performance is poor.
    
    POST /api/ml/auto-retrain/SPY
    """
    try:
        result = auto_retrain_if_needed(ticker)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Auto-retrain failed: {e}")
        return jsonify({'error': str(e)}), 500


@ml_correction.route('/api/ml/prediction-history/<ticker>')
def get_prediction_history(ticker):
    """
    Get prediction history with actuals.
    
    GET /api/ml/prediction-history/SPY?days_back=30
    """
    try:
        days_back = int(request.args.get('days_back', 30))
        
        df = fetch_predictions_with_targets(ticker, days_back)
        
        if df.empty:
            return jsonify({
                'ticker': ticker,
                'predictions': [],
                'count': 0
            })
        
        # Convert to JSON
        predictions = df.to_dict('records')
        
        # Format dates
        for pred in predictions:
            pred['prediction_date'] = pred['prediction_date'].isoformat()
            pred['target_date'] = pred['target_date'].isoformat()
        
        return jsonify({
            'ticker': ticker,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        logger.error(f"Prediction history failed: {e}")
        return jsonify({'error': str(e)}), 500
