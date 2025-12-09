"""
Recursive Forecaster - Baseline for Research Comparison

Implements traditional recursive multi-day forecasting where each day's
prediction is used as input for the next day's prediction.

This serves as a baseline to compare against the direct multi-day approach.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from services.ml_features import add_technical_indicators, add_volatility_features
from services.ml_features import add_volume_features, add_time_features, add_price_features
from sklearn.preprocessing import StandardScaler
from utils.logger import logger


class RecursiveForecaster:
    """
    Recursive multi-day stock price forecaster.
    
    Predicts one day ahead, then uses that prediction to create features
    for the next day's prediction. Errors compound over the forecast horizon.
    """
    
    def __init__(self, n_estimators=200, max_depth=15, random_state=42):
        """Initialize the recursive forecaster."""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
    def _prepare_features(self, df, for_training=True):
        """Prepare features for model fitting."""
        data = df.copy()
        
        # Apply feature engineering (simplified version)
        data = add_technical_indicators(data)
        data = add_volatility_features(data)
        data = add_volume_features(data)
        data = add_time_features(data)
        data = add_price_features(data)
        
        # Add lag features
        for i in range(1, 6):
            data[f'Lag_{i}'] = data['Close'].shift(i)
        
        # Create target (next day's price)
        if for_training:
            data['Target'] = data['Close'].shift(-1)
        
        # Drop NaNs
        data.dropna(inplace=True)
        
        if len(data) == 0:
            raise ValueError("Not enough data after feature engineering")
        
        # Select features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target',
                       'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                       'Volume_MA_20', 'Volume_MA_50', 'OBV_MA']
        
        feature_names = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_names]
        y = data['Target'] if for_training else None
        
        return X, y, feature_names, data
    
    def fit(self, X, y):
        """Fit the model on training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Recursive forecaster fitted on {len(X)} samples")
        
    def predict_recursive(self, df, days=5):
        """
        Recursively predict next N days.
        
        Args:
            df: Historical OHLCV DataFrame
            days: Number of days to predict
            
        Returns:
            List of predictions for each day
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_df = df.copy()
        
        logger.info(f"Starting recursive prediction for {days} days")
        
        for day in range(days):
            # Prepare features from current data
            try:
                X, _, _, enriched_df = self._prepare_features(current_df, for_training=False)
            except ValueError as e:
                logger.error(f"Feature prep failed at day {day+1}: {e}")
                break
            
            # Get last row of features
            if len(X) == 0:
                logger.error(f"No features available at day {day+1}")
                break
                
            X_last = X.iloc[[-1]]
            
            # Ensure features match training
            if self.feature_names:
                X_last = X_last[self.feature_names]
            
            # Normalize
            X_scaled = self.scaler.transform(X_last)
            
            # Predict next day
            next_price = self.model.predict(X_scaled)[0]
            predictions.append(float(next_price))
            
            logger.debug(f"Day {day+1}: Predicted ${next_price:.2f}")
            
            # Create synthetic next day using prediction
            # This is where error compounds - we're using a prediction as truth
            next_date = current_df.index[-1] + pd.Timedelta(days=1)
            
            # Skip weekends
            while next_date.dayofweek >= 5:
                next_date += pd.Timedelta(days=1)
            
            # Add predicted row to dataframe
            new_row = pd.DataFrame({
                'Open': [next_price],
                'High': [next_price * 1.01],  # Assume 1% range
                'Low': [next_price * 0.99],
                'Close': [next_price],
                'Volume': [current_df['Volume'].iloc[-1]]  # Use last known volume
            }, index=[next_date])
            
            current_df = pd.concat([current_df, new_row])
        
        logger.info(f"Recursive prediction complete: {len(predictions)} days forecasted")
        
        return predictions


def train_recursive_forecaster(df):
    """
    Train a recursive forecaster on historical data.
    
    Args:
        df: Historical OHLCV DataFrame
        
    Returns:
        Trained RecursiveForecaster instance
    """
    forecaster = RecursiveForecaster()
    
    # Prepare training data
    X, y, feature_names, _ = forecaster._prepare_features(df, for_training=True)
    
    # Fit model
    forecaster.fit(X, y)
    
    return forecaster
