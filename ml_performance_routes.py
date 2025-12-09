"""
ML Performance Dashboard Routes

Provides web interface for viewing ML model performance metrics,
feature importance, and prediction accuracy over time.
"""

from flask import Blueprint, render_template, request, jsonify
from services.ml_evaluation import get_model_performance, backfill_actual_prices, walk_forward_validation, print_validation_summary
from services.ml_service import load_model, predict_next_days
from services.ml_features import prepare_features
from services.ml_optimization import tune_hyperparameters, predict_strike_probability, get_smart_predictions
from services.stock_data_service import get_historical_data
from models.mongodb_models import MLPredictionModel
from datetime import datetime, timedelta
from utils.logger import logger
import pandas as pd
import json

ml_performance = Blueprint('ml_performance', __name__)


@ml_performance.route('/ml/performance')
def show_performance():
    """
    Main ML performance dashboard page.
    Shows overall statistics and recent predictions.
    """
    # Get query parameters
    ticker = request.args.get('ticker', None)
    days = int(request.args.get('days', 30))
    
    # Get performance metrics
    performance = get_model_performance(ticker=ticker, days=days)
    
    # Get recent predictions from database
    try:
        collection = MLPredictionModel.get_collection()
        query_filter = {}
        
        if ticker:
            query_filter['symbol'] = ticker
        
        recent_predictions = list(collection.find(query_filter).sort('prediction_date', -1).limit(50))
        
        # Format for template
        predictions_data = []
        for pred in recent_predictions:
            error = None
            if pred.get('actual_price'):
                error = abs(pred['actual_price'] - pred['predicted_price'])
            
            predictions_data.append({
                'symbol': pred['symbol'],
                'prediction_date': pred['prediction_date'].strftime('%Y-%m-%d %H:%M'),
                'target_date': pred['target_date'].strftime('%Y-%m-%d'),
                'predicted_price': pred['predicted_price'],
                'actual_price': pred.get('actual_price'),
                'error': error,
                'model_version': pred.get('model_version'),
                'confidence': pred.get('confidence')
            })
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        predictions_data = []
    
    return render_template('ml_performance.html', 
                          performance=performance,
                          predictions=predictions_data,
                          ticker=ticker or 'All',
                          days=days)


@ml_performance.route('/api/ml/backfill', methods=['POST'])
def api_backfill():
    """
    Trigger backfill of actual prices for past predictions.
    """
    try:
        updated_count = backfill_actual_prices()
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'message': f'Successfully updated {updated_count} predictions with actual prices'
        })
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/validate/<ticker>', methods=['POST'])
def api_validate(ticker):
    """
    Run walk-forward validation for a specific ticker.
    Returns detailed validation metrics.
    """
    try:
        # Get parameters
        data = request.get_json() or {}
        train_size = data.get('train_size', 252)
        test_size = data.get('test_size', 21)
        n_splits = data.get('n_splits', 5)
        
        logger.info(f"Starting walk-forward validation for {ticker}")
        
        # Fetch historical data
        df = get_historical_data(ticker, '2y', use_cache=True)
        if df.empty:
            return jsonify({'error': 'Could not fetch historical data'}), 404
        
        # Prepare features
        X, y, feature_names, scaler = prepare_features(df, for_training=True)
        
        # Create a simple model trainer function for validation
        from sklearn.ensemble import RandomForestRegressor
        def train_simple_model(X_train, y_train):
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            # Use only first target (1-day ahead) for validation
            model.fit(X_train, y_train.iloc[:, 0] if hasattr(y_train, 'iloc') else y_train)
            return model
        
        # Prepare data for validation
        validation_df = pd.DataFrame(X)
        validation_df['Target_Day1'] = y.iloc[:, 0]
        validation_df['Close'] = df['Close'].iloc[-len(X):]
        
        # Run validation
        results = walk_forward_validation(
            train_simple_model,
            validation_df,
            feature_names,
            train_size=train_size,
            test_size=test_size,
            n_splits=n_splits
        )
        
        if not results:
            return jsonify({'error': 'Insufficient data for validation'}), 400
        
        # Calculate summary statistics
        avg_mae = sum(r['mae'] for r in results) / len(results)
        avg_rmse = sum(r['rmse'] for r in results) / len(results)
        avg_dir_acc = sum(r['directional_accuracy'] for r in results) / len(results)
        
        # Format results for JSON
        formatted_results = []
        for r in results:
            formatted_results.append({
                'split': r['split'],
                'train_period': f"{r['train_start'].strftime('%Y-%m-%d')} to {r['train_end'].strftime('%Y-%m-%d')}",
                'test_period': f"{r['test_start'].strftime('%Y-%m-%d')} to {r['test_end'].strftime('%Y-%m-%d')}",
                'mae': round(r['mae'], 2),
                'rmse': round(r['rmse'], 2),
                'mape': round(r['mape'], 2),
                'directional_accuracy': round(r['directional_accuracy'], 1),
                'n_predictions': r['n_predictions']
            })
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'summary': {
                'avg_mae': round(avg_mae, 2),
                'avg_rmse': round(avg_rmse, 2),
                'avg_directional_accuracy': round(avg_dir_acc, 1),
                'n_splits': len(results)
            },
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Validation failed for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/feature-importance/<ticker>')
def api_feature_importance(ticker):
    """
    Get feature importance for a ticker's model.
    """
    try:
        model, feature_names, scaler = load_model(ticker)
        
        if model is None:
            return jsonify({'error': f'No model found for {ticker}'}), 404
        
        # Get feature importance from first estimator
        from services.ml_evaluation import get_feature_importance
        
        first_estimator = model.estimators_[0]
        if hasattr(first_estimator, 'estimators_'):  # VotingRegressor
            first_base = first_estimator.estimators_[0]
        else:
            first_base = first_estimator
        
        importance_df = get_feature_importance(first_base, feature_names, top_n=20)
        
        # Format for JSON
        features = []
        for _, row in importance_df.iterrows():
            features.append({
                'feature': row['feature'],
                'importance': round(float(row['importance']), 4)
            })
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'features': features
        })
        
    except Exception as e:
        logger.error(f"Error getting feature importance for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/predictions/<ticker>')
def api_predictions_history(ticker):
    """
    Get prediction history for a ticker with actual vs predicted comparison.
    """
    try:
        days = int(request.args.get('days', 30))
        cutoff_date = datetime.now() - timedelta(days=days)
        
        collection = MLPredictionModel.get_collection()
        predictions = list(collection.find({
            'symbol': ticker,
            'prediction_date': {'$gte': cutoff_date}
        }).sort('target_date', 1))
        
        # Format data for charting
        data = []
        for pred in predictions:
            error = None
            if pred.get('actual_price'):
                error = abs(pred['actual_price'] - pred['predicted_price'])
            
            data.append({
                'target_date': pred['target_date'].strftime('%Y-%m-%d'),
                'predicted': pred['predicted_price'],
                'actual': pred.get('actual_price'),
                'error': error,
                'confidence': pred.get('confidence')
            })
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction history for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/tune/<ticker>', methods=['POST'])
def api_tune_hyperparameters(ticker):
    """
    Run hyperparameter tuning for a ticker.
    """
    try:
        data = request.get_json() or {}
        cv_splits = data.get('cv_splits', 3)
        
        logger.info(f"Starting hyperparameter tuning for {ticker}")
        
        result = tune_hyperparameters(ticker, cv_splits=cv_splits)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Tuning failed for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/strike-probability', methods=['POST'])
def api_strike_probability():
    """
    Predict probability of touching a strike price.
    """
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        strike = float(data.get('strike'))
        days = int(data.get('days', 7))
        
        if not ticker or not strike:
            return jsonify({'error': 'Missing ticker or strike'}), 400
        
        result = predict_strike_probability(ticker, strike, days)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Strike probability prediction failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/smart-predictions/<ticker>')
def api_smart_predictions(ticker):
    """
    Get smart predictions combining ML with support/resistance.
    """
    try:
        # Get support/resistance from query params (comma-separated)
        support_str = request.args.get('support', '')
        resistance_str = request.args.get('resistance', '')
        
        support_levels = [float(x) for x in support_str.split(',') if x]
        resistance_levels = [float(x) for x in resistance_str.split(',') if x]
        
        result = get_smart_predictions(ticker, support_levels, resistance_levels)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Smart predictions failed for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@ml_performance.route('/api/ml/batch-predict', methods=['POST'])
def api_batch_predict():
    """Run predictions for multiple tickers at once."""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        days = data.get('days', 5)
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        results = {}
        for ticker in tickers:
            try:
                result = predict_next_days(ticker, days=days)
                results[ticker] = result
            except Exception as e:
                results[ticker] = {'error': str(e)}
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@ml_performance.route('/api/ml/export/<ticker>')
def api_export_predictions(ticker):
    """Export prediction history as CSV."""
    from flask import make_response
    import io
    
    try:
        collection = MLPredictionModel.get_collection()
        predictions = list(collection.find({
            'symbol': ticker
        }).sort('target_date', 1))
        
        if not predictions:
            return jsonify({'error': 'No predictions found'}), 404
        
        output = io.StringIO()
        output.write('prediction_date,target_date,predicted_price,actual_price,error,model_version,confidence\n')
        
        for pred in predictions:
            error = ''
            if pred.get('actual_price'):
                error = abs(pred['actual_price'] - pred['predicted_price'])
            
            output.write(f"{pred['prediction_date']},{pred['target_date']},{pred['predicted_price']},")
            output.write(f"{pred.get('actual_price') or ''},{error},{pred.get('model_version')},{pred.get('confidence') or ''}\n")
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename={ticker}_predictions.csv'
        return response
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        return jsonify({'error': str(e)}), 500


@ml_performance.route('/api/ml/market-context')
def api_market_context():
    """Get current market context (VIX, SPY, regime)."""
    try:
        from services.ml_market_context import get_current_market_context
        context = get_current_market_context()
        return jsonify({'success': True, **context, 'timestamp': context['timestamp'].isoformat()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
