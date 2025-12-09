"""
ML Model Optimization Module

Provides advanced model features including:
- Hyperparameter tuning with GridSearchCV
- Options-specific predictions (strike probability)
- Integration with support/resistance levels
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from services.stock_data_service import get_historical_data
from services.ml_features import prepare_features
from utils.logger import logger
from config import get_config

config = get_config()


def tune_hyperparameters(ticker, cv_splits=3):
    """
    Perform hyperparameter tuning using GridSearchCV with TimeSeriesSplit.
    
    Args:
        ticker: Stock ticker symbol
        cv_splits: Number of cross-validation splits
        
    Returns:
        Dict with best parameters and scores
    """
    logger.info(f"Starting hyperparameter tuning for {ticker}")
    
    # Fetch data
    df = get_historical_data(ticker, '2y', use_cache=True)
    if df.empty:
        return {'error': 'Could not fetch historical data'}
    
    # Prepare features
    try:
        X, y, feature_names, scaler = prepare_features(df, for_training=True)
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        return {'error': str(e)}
    
    # Use only 1-day ahead target for tuning
    y_1day = y.iloc[:, 0]
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Grid search
    logger.info(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} parameter combinations")
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y_1day)
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert negative MAE to positive
    
    logger.info(f"Tuning complete. Best MAE: ${best_score:.2f}")
    logger.info(f"Best parameters: {best_params}")
    
    return {
        'ticker': ticker,
        'best_params': best_params,
        'best_mae': round(best_score, 2),
        'cv_splits': cv_splits,
        'param_grid': param_grid
    }


def predict_strike_probability(ticker, strike_price, days_to_expiration, model=None):
    """
    Predict probability that stock price will touch a given strike price
    before expiration.
    
    Uses binary classification to predict if price will reach strike.
    
    Args:
        ticker: Stock ticker symbol
        strike_price: Strike price to predict probability for
        days_to_expiration: Number of days until expiration
        model: Optional pre-trained classifier
        
    Returns:
        Dict with probability and recommendation
    """
    logger.info(f"Predicting strike probability for {ticker} @ ${strike_price} in {days_to_expiration} days")
    
    # Fetch historical data
    df = get_historical_data(ticker, '2y', use_cache=True)
    if df.empty:
        return {'error': 'Could not fetch historical data'}
    
    current_price = float(df['Close'].iloc[-1])
    
    # Prepare features
    try:
        X, y, feature_names, scaler = prepare_features(df, for_training=True)
    except Exception as e:
        return {'error': str(e)}
    
    # Create binary target: did price touch strike in next N days?
    # Look at High prices for the next N days
    df_full = df.copy()
    
    touched_strike = []
    for i in range(len(df_full) - days_to_expiration):
        # Check if High in next N days touches strike
        future_highs = df_full['High'].iloc[i+1:i+1+days_to_expiration]
        future_lows = df_full['Low'].iloc[i+1:i+1+days_to_expiration]
        
        if strike_price > current_price:
            # Calls: check if High touched strike
            touched = (future_highs >= strike_price).any()
        else:
            # Puts: check if Low touched strike
            touched = (future_lows <= strike_price).any()
        
        touched_strike.append(1 if touched else 0)
    
    # Trim features to match
    X_binary = X.iloc[:len(touched_strike)]
    y_binary = np.array(touched_strike)
    
    if len(X_binary) < 100:
        return {'error': 'Insufficient data for classification'}
    
    # Train classifier if not provided
    if model is None:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        # Use most recent 80% for training
        split_idx = int(len(X_binary) * 0.8)
        X_train = X_binary.iloc[:split_idx]
        y_train = y_binary[:split_idx]
        
        clf.fit(X_train, y_train)
    else:
        clf = model
    
    # Predict probability for current state
    X_current = X.iloc[[-1]]
    prob = clf.predict_proba(X_current)[0][1]  # Probability of touching strike
    
    # Determine recommendation
    if prob >= 0.7:
        recommendation = "High probability - Good for selling options"
    elif prob >= 0.5:
        recommendation = "Moderate probability - Neutral"
    elif prob >= 0.3:
        recommendation = "Low probability - Consider buying options"
    else:
        recommendation = "Very low probability - Avoid selling options"
    
    return {
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'strike_price': strike_price,
        'days_to_expiration': days_to_expiration,
        'probability': round(float(prob), 3),
        'probability_percent': round(float(prob) * 100, 1),
        'recommendation': recommendation,
        'option_type': 'call' if strike_price > current_price else 'put'
    }


def get_smart_predictions(ticker, support_levels, resistance_levels):
    """
    Combine ML predictions with support/resistance levels to generate
    smart trading recommendations.
    
    Args:
        ticker: Stock ticker symbol  
        support_levels: List of support price levels
        resistance_levels: List of resistance price levels
        
    Returns:
        Dict with predictions and strike recommendations
    """
    from services.ml_service import predict_next_days
    
    # Get ML predictions
    ml_result = predict_next_days(ticker, days=5)
    
    if 'error' in ml_result:
        return ml_result
    
    current_price = ml_result['last_close']
    predictions = ml_result['predictions']
    
    # Find nearest support/resistance
    support_below = [s for s in support_levels if s < current_price]
    resistance_above = [r for r in resistance_levels if r > current_price]
    
    nearest_support = max(support_below) if support_below else None
    nearest_resistance = min(resistance_above) if resistance_above else None
    
    # Analyze predictions relative to S/R levels
    recommendations = []
    
    # Check 5-day prediction
    day5_pred = predictions[4]
    
    if nearest_resistance and day5_pred < nearest_resistance:
        # Predict won't reach resistance - good for call credit spreads
        prob_result = predict_strike_probability(ticker, nearest_resistance, 5)
        recommendations.append({
            'strategy': 'Call Credit Spread',
            'short_strike': round(nearest_resistance, 2),
            'reasoning': f"Prediction (${day5_pred:.2f}) below resistance (${nearest_resistance:.2f})",
            'probability_safe': prob_result['probability_percent'] if 'probability_percent' in prob_result else None
        })
    
    if nearest_support and day5_pred > nearest_support:
        # Predict won't fall to support - good for put credit spreads
        prob_result = predict_strike_probability(ticker, nearest_support, 5)
        recommendations.append({
            'strategy': 'Put Credit Spread',
            'short_strike': round(nearest_support, 2),
            'reasoning': f"Prediction (${day5_pred:.2f}) above support (${nearest_support:.2f})",
            'probability_safe': prob_result['probability_percent'] if 'probability_percent' in prob_result else None
        })
    
    # If prediction crosses resistance - directional play
    if nearest_resistance and day5_pred > nearest_resistance:
        recommendations.append({
            'strategy': 'Call Debit Spread or Long Call',
            'target_strike': round(nearest_resistance, 2),
            'reasoning': f"Prediction (${day5_pred:.2f}) above resistance - bullish breakout expected"
        })
    
    if nearest_support and day5_pred < nearest_support:
        recommendations.append({
            'strategy': 'Put Debit Spread or Long Put',
            'target_strike': round(nearest_support, 2),
            'reasoning': f"Prediction (${day5_pred:.2f}) below support - bearish breakdown expected"
        })
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'ml_predictions': predictions,
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'recommendations': recommendations,
        'confidence_intervals': ml_result['confidence_intervals']
    }
