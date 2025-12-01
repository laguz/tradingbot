import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from services.tradier_service import get_historical_data

def prepare_data(df):
    """
    Prepares features for the ML model.
    Features: Lag_1 to Lag_5, SMA_5, SMA_20
    Target: Next Day Close (Shifted -1)
    """
    data = df.copy()
    
    # Create Lags
    for i in range(1, 6):
        data[f'Lag_{i}'] = data['Close'].shift(i)
        
    # Create Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Create Target (Next Day's Close)
    data['Target'] = data['Close'].shift(-1)
    
    # Drop NaNs created by shifting/rolling
    data.dropna(inplace=True)
    
    return data

def train_model(data):
    """
    Trains a Random Forest Regressor.
    """
    features = [f'Lag_{i}' for i in range(1, 6)] + ['SMA_5', 'SMA_20']
    X = data[features]
    y = data['Target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features

def predict_next_days(ticker, days=5):
    """
    Predicts the closing price for the next 'days' trading days.
    """
    # 1. Fetch History (Need enough data for lags/SMA)
    chart_data = get_historical_data(ticker, '1y')
    if not chart_data:
        return {'error': 'Could not fetch historical data.'}
        
    df = pd.DataFrame({
        'Date': chart_data['labels'], # Note: These are just strings 'MM-DD' from tradier_service, might need full dates?
        # tradier_service.get_historical_data returns 'labels' as 'MM-DD'. 
        # But it computes them from 'day_entries'. 
        # We might need to fetch raw data again or modify tradier_service to return raw DF.
        # For simplicity, let's re-fetch or assume we can reconstruct.
        # Actually, tradier_service.get_historical_data returns a dict for Chart.js.
        # It's better to use a raw data fetcher. 
        # Let's use get_historical_data but we need the raw values.
        'Close': chart_data['data']
    })
    
    # We need a continuous series to append predictions
    # Let's assume the data is sorted by date ascending (which it is from API)
    
    # 2. Prepare Training Data
    train_df = prepare_data(df)
    if train_df.empty:
         return {'error': 'Not enough data to train model.'}
         
    # 3. Train Model
    model, feature_names = train_model(train_df)
    
    # 4. Recursive Prediction
    # Start with the last known data points
    current_data = df['Close'].tolist()
    predictions = []
    
    for _ in range(days):
        # Construct features for the "next" prediction based on current_data tail
        # We need to simulate the DataFrame row for the "current" day to predict "tomorrow"
        
        # Lags: [Close(T), Close(T-1), ... Close(T-4)]
        lags = {}
        for i in range(1, 6):
            lags[f'Lag_{i}'] = current_data[-i]
            
        # SMAs
        sma_5 = np.mean(current_data[-5:])
        sma_20 = np.mean(current_data[-20:])
        
        # Create feature vector
        features_dict = {**lags, 'SMA_5': sma_5, 'SMA_20': sma_20}
        # Ensure order matches training
        X_next = pd.DataFrame([features_dict])[feature_names]
        
        # Predict
        next_price = model.predict(X_next)[0]
        predictions.append(round(next_price, 2))
        
        # Append to current_data to be used for next iteration
        current_data.append(next_price)
        
    return {
        'ticker': ticker,
        'predictions': predictions,
        'last_close': df['Close'].iloc[-1]
    }
