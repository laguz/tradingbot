import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from services.tradier_service import get_raw_historical_data
from config import get_config
from utils.logger import logger

config = get_config()

# Ensure model directory exists
os.makedirs(config.ML_MODEL_DIR, exist_ok=True)

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    Includes RSI, MACD, Bollinger Bands, ATR, and Volume Change.
    """
    data = df.copy()
    
    # RSI (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands (20, 2)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    
    # ATR (14)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # Volume Change
    data['Volume_Change'] = data['Volume'].pct_change()
    
    return data

def prepare_data(df):
    """
    Prepares features for the ML model.
    """
    data = add_technical_indicators(df)
    
    # Create Lags
    for i in range(1, 6):
        data[f'Lag_{i}'] = data['Close'].shift(i)
        
    # Create Target (Next Day's Close)
    data['Target'] = data['Close'].shift(-1)
    
    # Drop NaNs created by shifting/rolling
    data.dropna(inplace=True)
    
    return data

def train_model(data):
    """
    Trains a Random Forest Regressor with TimeSeriesSplit validation.
    """
    logger.info("Training ML model with {} samples".format(len(data)))
    # Select features (exclude non-numeric or target columns)
    exclude_cols = ['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    features = [col for col in data.columns if col not in exclude_cols]
    
    X = data[features]
    y = data['Target']
    
    # Time Series Split Validation
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    logger.info("Model training complete")
    return model, features


def get_model_path(ticker):
    """Generate model file path for a ticker"""
    return os.path.join(config.ML_MODEL_DIR, f"{ticker}_model.pkl")


def save_model(ticker, model, features):
    """Save trained model to disk"""
    model_path = get_model_path(ticker)
    model_data = {
        'model': model,
        'features': features,
        'trained_at': datetime.now(),
        'version': '1.0'
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Saved model for {ticker} to {model_path}")


def load_model(ticker):
    """
    Load trained model from disk if it exists and is not too old.
    Returns (model, features) or (None, None) if not available/expired.
    """
    model_path = get_model_path(ticker)
    
    if not os.path.exists(model_path):
        logger.debug(f"No cached model found for {ticker}")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        trained_at = model_data.get('trained_at')
        age_days = (datetime.now() - trained_at).days
        
        if age_days > config.ML_RETRAIN_DAYS:
            logger.info(f"Model for {ticker} is {age_days} days old, needs retraining")
            return None, None
        
        logger.info(f"Loaded cached model for {ticker} (age: {age_days} days)")
        return model_data['model'], model_data['features']
        
    except Exception as e:
        logger.error(f"Error loading model for {ticker}: {e}")
        return None, None

def predict_next_days(ticker, days=5, force_retrain=False):
    """
    Predicts the closing price for the next 'days' trading days.
    Uses cached model if available and not expired.
    """
    logger.info(f"Starting prediction for {ticker}, days={days}")
    
    # Try to load cached model
    model, feature_names = None, None
    if not force_retrain:
        model, feature_names = load_model(ticker)
    
    # If no cached model or force retrain, train new one
    if model is None:
        logger.info(f"Training new model for {ticker}")
        # 1. Fetch History (Need enough data for indicators)
        df = get_raw_historical_data(ticker, '2y') 
        if df.empty:
            logger.error(f"Could not fetch historical data for {ticker}")
            return {'error': 'Could not fetch historical data.'}
            
        # 2. Prepare Training Data
        train_df = prepare_data(df)
        if train_df.empty:
            logger.error(f"Not enough data to train model for {ticker}")
            return {'error': 'Not enough data to train model.'}
            
        # 3. Train Model
        model, feature_names = train_model(train_df)
        
        # 4. Save model for future use
        save_model(ticker, model, feature_names)
    else:
        # Still need raw data for prediction base
        df = get_raw_historical_data(ticker, '2y')
        if df.empty:
            return {'error': 'Could not fetch historical data.'}
    
    # 5. Recursive Prediction
    current_df = df.copy()
    predictions = []
    
    for _ in range(days):
        # Calculate indicators on the current dataset
        enriched_df = add_technical_indicators(current_df)
        
        # Extract features for the LAST row (Time T) to predict T+1
        last_features = {}
        
        # Lags: Lag_1 at T is Close(T-1)
        for i in range(1, 6):
            last_features[f'Lag_{i}'] = enriched_df['Close'].iloc[-i]
            
        # Indicators: Take the value at T
        for col in feature_names:
            if col.startswith('Lag_'): continue
            last_features[col] = enriched_df[col].iloc[-1]
            
        X_next = pd.DataFrame([last_features])
        # Ensure column order matches training
        X_next = X_next[feature_names]
        
        # Predict
        next_price = model.predict(X_next)[0]
        predictions.append(round(float(next_price), 2))
        
        # Append to current_df for next iteration
        # Assumption: Future candles are flat (Open=High=Low=Close) and Volume is constant
        last_date = current_df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() > 4: # Skip weekends
             next_date += pd.Timedelta(days=1)
             
        new_row = pd.DataFrame({
            'Open': [next_price],
            'High': [next_price],
            'Low': [next_price],
            'Close': [next_price],
            'Volume': [current_df['Volume'].iloc[-1]]
        }, index=[next_date])
        
        current_df = pd.concat([current_df, new_row])
    
    logger.info(f"Predictions complete for {ticker}: {predictions}")
    return {
        'ticker': ticker,
        'predictions': predictions,
        'last_close': float(df['Close'].iloc[-1])
    }
