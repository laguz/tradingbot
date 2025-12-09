"""
Enhanced ML Service V3 - Production

Integrates all improvements:
- Quantile regression for proper confidence intervals
- Regime-adaptive hyperparameters and features
- Feature selection to reduce redundancy
"""

from services.ml_service import get_raw_historical_data, save_model, load_model, get_model_path
from services.ml_features import prepare_features
from services.ml_quantile import train_quantile_models, calculate_quantile_intervals, get_interval_width
from services.ml_regime_adaptive import RegimeAdaptivePredictor, detect_regime
from services.ml_feature_selection import auto_select_features
from services.ml_evaluation import save_prediction_to_db
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime, timedelta
from utils.logger import logger
from config import get_config
import pandas as pd
import pickle
import os

config = get_config()
MODEL_VERSION = "3.0-enhanced"

# Global regime predictor
regime_predictor = RegimeAdaptivePredictor()


def predict_enhanced(ticker, days=5, use_quantiles=True, use_regime_adaptation=True, 
                    use_feature_selection=True, force_retrain=False):
    """
    Enhanced prediction with all improvements.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to predict (max 5)
        use_quantiles: Use quantile regression for intervals
        use_regime_adaptation: Adapt to market regime
        use_feature_selection: Apply feature selection
        force_retrain: Force model retraining
        
    Returns:
        Dict with predictions, intervals, regime info, and metadata
    """
    logger.info(f"Enhanced prediction for {ticker} (v{MODEL_VERSION})")
    logger.info(f"Options: quantiles={use_quantiles}, regime={use_regime_adaptation}, feature_selection={use_feature_selection}")
    
    # Fetch data
    df = get_raw_historical_data(ticker, '2y')
    if df.empty:
        return {'error': 'Could not fetch historical data'}
    
    # Prepare features
    try:
        X, y, feature_names, scaler = prepare_features(df, for_training=True)
    except Exception as e:
        return {'error': f'Feature preparation failed: {str(e)}'}
    
    # Detect regime
    regime = None
    if use_regime_adaptation:
        regime = detect_regime(df)
        regime_predictor.current_regime = regime
    
    # Feature selection
    if use_feature_selection and len(feature_names) > 35:
        selected_features = auto_select_features(X, y, max_features=35)
        X = X[selected_features]
        feature_names = selected_features
        logger.info(f"Using {len(selected_features)} selected features")
    
    # Check for existing model
    model_path = get_model_path(ticker).replace('_v2.pkl', '_v3_enhanced.pkl')
    should_train = force_retrain or not os.path.exists(model_path)
    
    if should_train:
        logger.info("Training new enhanced model...")
        
        if use_quantiles:
            # Train quantile models
            quantile_models = train_quantile_models(X, y)
            
            model_data = {
                'quantile_models': quantile_models,
                'features': feature_names,
                'scaler': scaler,
                'regime': regime,
                'trained_at': datetime.now(),
                'version': MODEL_VERSION
            }
        else:
            # Train regular ensemble
            ensemble = MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42)
            )
            ensemble.fit(X, y)
            
            model_data = {
                'model': ensemble,
                'features': feature_names,
                'scaler': scaler,
                'regime': regime,
                'trained_at': datetime.now(),
                'version': MODEL_VERSION
            }
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    else:
        # Load existing model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"Loaded model from {model_path}")
    
    # Make predictions
    X_latest = X.iloc[[-1]]
    last_close = float(df['Close'].iloc[-1])
    
    if use_quantiles and 'quantile_models' in model_data:
        # Use quantile models
        intervals = calculate_quantile_intervals(model_data['quantile_models'], X_latest)
        predictions = intervals['median']
        interval_width = get_interval_width(intervals)
    else:
        # Use regular model
        predictions = model_data['model'].predict(X_latest)[0]
        predictions = predictions.tolist()
        
        # Create simple intervals
        intervals = {
            'low': [p * 0.95 for p in predictions],
            'median': predictions,
            'high': [p * 1.05 for p in predictions]
        }
        interval_width = 5.0
    
    # Generate target dates
    last_date = df.index[-1]
    target_dates = []
    current_date = last_date
    
    for _ in range(days):
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:  # Skip weekends
            current_date += timedelta(days=1)
        target_dates.append(current_date.strftime('%Y-%m-%d'))
    
    # Log predictions to database
    prediction_date = datetime.now()
    for i in range(days):
        save_prediction_to_db(
            ticker, 
            prediction_date, 
            datetime.strptime(target_dates[i], '%Y-%m-%d'),
            predictions[i],
            None,  # actual_price
            MODEL_VERSION,
            1.0 - (interval_width / 100)  # confidence = 1 - interval_width
        )
    
    # Build result
    result = {
        'ticker': ticker,
        'last_close': last_close,
        'last_date': last_date.strftime('%Y-%m-%d'),
        'predictions': predictions[:days],
        'target_dates': target_dates,
        'confidence_intervals': {
            'low': intervals['low'][:days],
            'median': intervals['median'][:days],
            'high': intervals['high'][:days]
        },
        'interval_width_pct': interval_width,
        'model_version': MODEL_VERSION,
        'trained_at': model_data['trained_at'].isoformat(),
        'features_used': len(feature_names),
        'enhancements': {
            'quantile_regression': use_quantiles,
            'regime_adaptive': use_regime_adaptation,
            'feature_selection': use_feature_selection
        }
    }
    
    if regime:
        result['market_regime'] = {
            'trend': regime['trend'],
            'volatility': regime['volatility'],
            'realized_vol': f"{regime['realized_vol']:.1f}%"
        }
    
    logger.info(f"Enhanced prediction complete for {ticker}")
    
    return result
