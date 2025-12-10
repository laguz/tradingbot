"""
Model evaluation and performance tracking module.

Provides functions for:
- Walk-forward validation
- Performance metrics calculation (MAE, RMSE, directional accuracy)
- Feature importance analysis
- Database tracking integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.mongodb_models import MLPredictionModel
from services.stock_data_service import get_historical_data
from utils.logger import logger


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dict with MAE, RMSE, and MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'mape': round(mape, 2)
    }


def calculate_directional_accuracy(y_true, y_pred, current_price=None):
    """
    Calculate directional accuracy (% of correct up/down predictions).
    
    Args:
        y_true: Actual future prices
        y_pred: Predicted future prices
        current_price: Current price to compare against (optional)
        
    Returns:
        Directional accuracy as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if current_price is not None:
        # Determine actual direction (up=1, down=0)
        actual_direction = (y_true > current_price).astype(int)
        
        # Determine predicted direction
        pred_direction = (y_pred > current_price).astype(int)
    else:
        # Use consecutive price changes
        if len(y_true) <= 1:
            return 0.0
            
        actual_direction = (np.diff(y_true) > 0).astype(int)
        pred_direction = (np.diff(y_pred) > 0).astype(int)
    
    # Calculate accuracy
    correct = (actual_direction == pred_direction).sum()
    total = len(actual_direction)
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    return round(accuracy, 2)


def walk_forward_validation(model_trainer_func, df, features, train_size=252, test_size=21, n_splits=5):
    """
    Perform walk-forward validation to simulate real trading conditions.
    
    Args:
        model_trainer_func: Function that trains and returns a model
        df: Full dataset with features and targets
        features: List of feature column names
        train_size: Number of days for training window
        test_size: Number of days for testing window
        n_splits: Number of validation splits
        
    Returns:
        List of validation results for each split
    """
    results = []
    total_samples = len(df)
    
    # Calculate step size to get approximately n_splits
    max_possible_splits = (total_samples - train_size) // test_size
    actual_splits = min(n_splits, max_possible_splits)
    
    if actual_splits < 1:
        logger.warning(f"Not enough data for walk-forward validation. Need at least {train_size + test_size} samples.")
        return []
    
    step = max(1, (total_samples - train_size - test_size) // actual_splits)
    
    logger.info(f"Starting walk-forward validation: {actual_splits} splits, train={train_size}, test={test_size}")
    
    for i in range(0, actual_splits * step, step):
        if i + train_size + test_size > total_samples:
            break
            
        # Split data
        train_end = i + train_size
        test_end = min(train_end + test_size, total_samples)
        
        train_df = df.iloc[i:train_end]
        test_df = df.iloc[train_end:test_end]
        
        if len(test_df) < test_size // 2:  # Skip if test set too small
            continue
        
        # Train model
        model = model_trainer_func(train_df[features], train_df['Target_Day1'])
        
        # Predict on test set
        predictions = model.predict(test_df[features])
        actuals = test_df['Target_Day1'].values
        
        # Calculate metrics
        metrics = calculate_metrics(actuals, predictions)
        
        # Calculate directional accuracy
        last_train_close = train_df['Close'].iloc[-1] if 'Close' in train_df.columns else train_df.index[-1]
        dir_accuracy = calculate_directional_accuracy(actuals, predictions, last_train_close)
        
        results.append({
            'split': len(results) + 1,
            'train_start': df.index[i],
            'train_end': df.index[train_end - 1],
            'test_start': df.index[train_end],
            'test_end': df.index[test_end - 1],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'directional_accuracy': dir_accuracy,
            'n_predictions': len(predictions)
        })
    
    logger.info(f"Walk-forward validation complete: {len(results)} splits evaluated")
    
    return results


def get_feature_importance(model, feature_names, top_n=20):
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature names and importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not support feature importance")
        return pd.DataFrame()
    
    # For ensemble models, try to get feature importances from first estimator
    if hasattr(model, 'estimators_'):
        importances = model.estimators_[0].feature_importances_
    else:
        importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(top_n)


def save_prediction_to_db(ticker, target_dates, predicted_prices, model_version, features_used, confidence_scores=None):
    """
    Save prediction to database for performance tracking.
    
    Args:
        ticker: Stock ticker symbol
        target_dates: List of target dates for predictions
        predicted_prices: List of predicted prices
        model_version: Model version identifier
        features_used: List of feature names
        confidence_scores: Optional list of confidence scores
    """
    try:
        predictions = []
        for i, (target_date, predicted_price) in enumerate(zip(target_dates, predicted_prices)):
            confidence = confidence_scores[i] if confidence_scores else None
            
            # Check if we have an actual price for this date (e.g. running on past data)
            actual_price = None
            try:
                from models.mongodb_models import StockDataModel
                actual_price = StockDataModel.get_close_price(ticker, target_date)
            except Exception:
                pass

            predictions.append({
                'symbol': ticker,
                'prediction_date': datetime.now(),
                'target_date': target_date.replace(hour=0, minute=0, second=0, microsecond=0),
                'predicted_price': predicted_price,
                'actual_price': actual_price, 
                'model_version': model_version,
                'features_used': features_used,
                'confidence': confidence
            })
        
        MLPredictionModel.upsert_many(predictions)
        logger.info(f"Saved (upserted) {len(predicted_prices)} predictions for {ticker} to database")
        
    except Exception as e:
        logger.error(f"Error saving predictions to database: {e}")


def backfill_actual_prices():
    """
    Update MLPrediction collection with actual prices for past predictions.
    Should be run daily as a background job.
    
    Returns:
        Number of predictions updated
    """
    updated_count = 0
    
    try:
        # Get predictions from last 90 days where actual_price is None and target_date has passed
        collection = MLPredictionModel.get_collection()
        cutoff = datetime.now() - timedelta(days=90)
        
        pending = list(collection.find({
            'actual_price': None,
            'target_date': {'$lt': datetime.now()},
            'prediction_date': {'$gte': cutoff}
        }))
        
        logger.info(f"Found {len(pending)} predictions to backfill")
        
        # Group by symbol to batch API calls
        by_symbol = {}
        for pred in pending:
            symbol = pred['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(pred)
        
        # Fetch actual prices and prepare bulk updates
        updates = []
        
        for symbol, predictions in by_symbol.items():
            try:
                # 1. Try to get data from local DB first (faster)
                from models.mongodb_models import StockDataModel
                
                # Check if we need to fetch API data for any missing dates
                missing_data = False
                for pred in predictions:
                    target_date = pred['target_date']
                    price = StockDataModel.get_close_price(symbol, target_date)
                    
                    if price is not None:
                        updates.append({
                            'id': str(pred['_id']),
                            'actual_price': price
                        })
                    else:
                        missing_data = True
                
                # 2. If data missing, fetch from API
                if missing_data:
                    df = get_historical_data(symbol, '3m', use_cache=True) 
                    
                    if not df.empty:
                        for pred in predictions:
                            # Skip if we already found it in DB
                            if any(u['id'] == str(pred['_id']) for u in updates):
                                continue
                                
                            target_date_str = pred['target_date'].strftime('%Y-%m-%d')
                            if target_date_str in df.index:
                                actual_price = float(df.loc[target_date_str, 'Close'])
                                updates.append({
                                    'id': str(pred['_id']),
                                    'actual_price': actual_price
                                })
                            else:
                                logger.debug(f"Still no data for {symbol} on {target_date_str}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        # Bulk update
        if updates:
            updated_count = MLPredictionModel.update_actual_prices_bulk(updates)
        
        logger.info(f"Backfilled {updated_count} predictions with actual prices")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"Error in backfill_actual_prices: {e}")
        return 0


def get_model_performance(ticker=None, days=30):
    """
    Get model performance metrics from database.
    
    Args:
        ticker: Optional ticker to filter by
        days: Number of days to look back
        
    Returns:
        Dict with performance statistics
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        collection = MLPredictionModel.get_collection()
        query_filter = {
            'actual_price': {'$ne': None},
            'prediction_date': {'$gte': cutoff_date}
        }
        
        if ticker:
            query_filter['symbol'] = ticker
        
        predictions = list(collection.find(query_filter))
        
        if not predictions:
            return {'error': 'No completed predictions found'}
        
        # Calculate aggregate metrics
        actuals = [p['actual_price'] for p in predictions]
        predicted = [p['predicted_price'] for p in predictions]
        
        metrics = calculate_metrics(actuals, predicted)
        
        # Calculate directional accuracy (need base prices)
        # For simplicity, using predicted vs actual comparison
        correct_direction = sum(1 for a, p in zip(actuals, predicted) 
                               if (a > p) == (predictions[0]['actual_price'] > predictions[0]['predicted_price']))
        dir_accuracy = (correct_direction / len(predictions)) * 100
        
        # Calculate average confidence
        confidences = [p.get('confidence') for p in predictions if p.get('confidence') is not None]
        avg_confidence = round(np.mean(confidences), 2) if confidences else None
        
        return {
            'ticker': ticker or 'All',
            'period_days': days,
            'num_predictions': len(predictions),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'directional_accuracy': round(dir_accuracy, 2),
            'avg_confidence': avg_confidence
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return {'error': str(e)}


def print_validation_summary(results):
    """
    Print a formatted summary of validation results.
    
    Args:
        results: List of validation results from walk_forward_validation
    """
    if not results:
        print("No validation results to display")
        return
    
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("="*80)
    
    for r in results:
        print(f"\nSplit {r['split']}: {r['test_start'].strftime('%Y-%m-%d')} to {r['test_end'].strftime('%Y-%m-%d')}")
        print(f"  MAE: ${r['mae']:.2f} | RMSE: ${r['rmse']:.2f} | MAPE: {r['mape']:.1f}%")
        print(f"  Directional Accuracy: {r['directional_accuracy']:.1f}% | N: {r['n_predictions']}")
    
    # Overall statistics
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_mape = np.mean([r['mape'] for r in results])
    avg_dir_acc = np.mean([r['directional_accuracy'] for r in results])
    
    print("\n" + "-"*80)
    print("OVERALL PERFORMANCE:")
    print(f"  Average MAE: ${avg_mae:.2f}")
    print(f"  Average RMSE: ${avg_rmse:.2f}")
    print(f"  Average MAPE: {avg_mape:.1f}%")
    print(f"  Average Directional Accuracy: {avg_dir_acc:.1f}%")
    print("="*80 + "\n")
