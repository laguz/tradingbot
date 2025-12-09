"""
Feature engineering module for ML stock prediction.

Provides comprehensive feature engineering functions including:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Volatility features (historical volatility)
- Volume features (relative volume, volume profile)
- Time-based features (day of week, days to Friday)
- Price features (distance from moving averages, momentum)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import get_config

config = get_config()

# Import market context if available
try:
    from services.ml_market_context import add_market_context_features
    MARKET_CONTEXT_AVAILABLE = True
except ImportError:
    MARKET_CONTEXT_AVAILABLE = False


def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    Includes RSI, MACD, Bollinger Bands, and ATR.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicator columns
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
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # Bollinger Bands (20, 2)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (rolling_std * 2)
    data['BB_Lower'] = data['SMA_20'] - (rolling_std * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['SMA_20']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # ATR (14)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    data['ATR_Percent'] = data['ATR'] / data['Close']  # Normalized ATR
    
    return data


def add_volatility_features(df):
    """
    Adds volatility-based features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added volatility features
    """
    data = df.copy()
    
    # Historical Volatility (20 days, annualized)
    returns = data['Close'].pct_change()
    data['HV_20'] = returns.rolling(window=20).std() * np.sqrt(252)
    
    # Realized Volatility (different windows)
    data['HV_10'] = returns.rolling(window=10).std() * np.sqrt(252)
    data['HV_50'] = returns.rolling(window=50).std() * np.sqrt(252)
    
    # Volatility ratio (short-term vs long-term)
    data['Vol_Ratio'] = data['HV_10'] / data['HV_50']
    
    # Parkinson volatility (uses High-Low range)
    hl_ratio = np.log(data['High'] / data['Low'])
    data['Parkinson_Vol'] = (hl_ratio ** 2).rolling(window=20).mean() * np.sqrt(252 / (4 * np.log(2)))
    
    return data


def add_volume_features(df):
    """
    Adds volume-based features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added volume features
    """
    data = df.copy()
    
    # Volume change
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Relative volume (vs moving averages)
    data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_MA_50'] = data['Volume'].rolling(window=50).mean()
    data['Rel_Volume_20'] = data['Volume'] / data['Volume_MA_20']
    data['Rel_Volume_50'] = data['Volume'] / data['Volume_MA_50']
    
    # Volume-Price trend
    data['VP_Trend'] = (data['Close'].pct_change() * data['Volume']).rolling(window=10).mean()
    
    # On-Balance Volume (OBV) - vectorized for better performance
    close_diff = data['Close'].diff()
    volume_direction = np.where(close_diff > 0, data['Volume'], 
                                np.where(close_diff < 0, -data['Volume'], 0))
    data['OBV'] = volume_direction.cumsum()
    data['OBV_MA'] = data['OBV'].rolling(window=20).mean()
    
    return data


def add_time_features(df):
    """
    Adds time-based features.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added time features
    """
    data = df.copy()
    
    # Day of week (0=Monday, 4=Friday)
    data['DayOfWeek'] = data.index.dayofweek
    
    # Month
    data['Month'] = data.index.month
    
    # Quarter
    data['Quarter'] = data.index.quarter
    
    # Days to Friday (for weekly options expiration)
    data['DaysToFriday'] = (4 - data.index.dayofweek) % 7
    
    # Is it Monday (often different behavior)
    data['IsMonday'] = (data.index.dayofweek == 0).astype(int)
    
    # Is it Friday (often different behavior)
    data['IsFriday'] = (data.index.dayofweek == 4).astype(int)
    
    return data


def add_price_features(df):
    """
    Adds price-based features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added price features
    """
    data = df.copy()
    
    # Distance from moving averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Distance_SMA_50'] = (data['Close'] - data['SMA_50']) / data['Close']
    data['Distance_SMA_200'] = (data['Close'] - data['SMA_200']) / data['Close']
    
    # EMA distances
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['Distance_EMA_12'] = (data['Close'] - data['EMA_12']) / data['Close']
    
    # Price momentum
    data['Momentum_5'] = data['Close'].pct_change(periods=5)
    data['Momentum_10'] = data['Close'].pct_change(periods=10)
    data['Momentum_20'] = data['Close'].pct_change(periods=20)
    
    # Rate of change
    data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
    
    # High-Low spread
    data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
    
    return data


def add_lag_features(df, lags=5):
    """
    Adds lagged close price features.
    
    Args:
        df: DataFrame with Close column
        lags: Number of lag periods to create
        
    Returns:
        DataFrame with added lag features
    """
    data = df.copy()
    
    for i in range(1, lags + 1):
        data[f'Lag_{i}'] = data['Close'].shift(i)
    
    return data


def prepare_features(df, for_training=True):
    """
    Main feature preparation pipeline.
    Applies all feature engineering and normalization.
    
    Args:
        df: Raw DataFrame with OHLCV data
        for_training: If True, creates target variable
        
    Returns:
        Tuple of (features_df, target, feature_names, scaler)
        - features_df: DataFrame ready for model training/prediction
        - target: Target variable (None if for_training=False)
        - feature_names: List of feature column names
        - scaler: Fitted StandardScaler (None if normalization disabled)
    """
    data = df.copy()
    
    # Apply all feature engineering
    data = add_technical_indicators(data)
    
    if config.ML_ENABLE_VOLATILITY_FEATURES:
        data = add_volatility_features(data)
    
    data = add_volume_features(data)
    
    if config.ML_ENABLE_TIME_FEATURES:
        data = add_time_features(data)
    
    data = add_price_features(data)
    data = add_lag_features(data, lags=5)
    
    # Add market context features if available and enabled
    if MARKET_CONTEXT_AVAILABLE and hasattr(config, 'ML_ENABLE_MARKET_CONTEXT') and config.ML_ENABLE_MARKET_CONTEXT:
        try:
            # Get ticker from DataFrame attributes or use 'UNKNOWN'
            ticker = getattr(data, 'ticker', 'UNKNOWN')
            data = add_market_context_features(data, ticker)
        except Exception as e:
            # Continue without market context if it fails
            pass
    
    # Create targets for multi-day prediction (1 to 5 days ahead)
    if for_training:
        for day in range(1, 6):
            data[f'Target_Day{day}'] = data['Close'].shift(-day)
    
    # Drop NaNs created by rolling windows and shifts
    data.dropna(inplace=True)
    
    if len(data) == 0:
        raise ValueError("Not enough data after feature engineering. Need more historical data.")
    
    # Select feature columns (exclude OHLCV and targets)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date',
                    'Target_Day1', 'Target_Day2', 'Target_Day3', 'Target_Day4', 'Target_Day5',
                    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',  # Keep only derivatives
                    'Volume_MA_20', 'Volume_MA_50', 'OBV_MA']  # Keep only derivatives
    
    feature_names = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_names]
    
    # Extract targets if training
    y = None
    if for_training:
        target_cols = [f'Target_Day{day}' for day in range(1, 6)]
        y = data[target_cols]
    
    # Normalize features if enabled
    scaler = None
    if config.ML_ENABLE_NORMALIZATION:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    
    return X, y, feature_names, scaler


def get_feature_count():
    """
    Returns the expected number of features based on configuration.
    Useful for validation.
    """
    # This is approximate - actual count depends on data
    base_features = 20  # Technical indicators + lags
    vol_features = 5 if config.ML_ENABLE_VOLATILITY_FEATURES else 0
    time_features = 7 if config.ML_ENABLE_TIME_FEATURES else 0
    volume_features = 8
    price_features = 11
    
    return base_features + vol_features + time_features + volume_features + price_features
