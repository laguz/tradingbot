"""
Market Context Features Module

Provides market-wide context features for ML models:
- VIX (volatility index) integration
- Sector correlation (SPY, QQQ)
- Market regime detection (bull/bear/sideways)
- Relative strength calculations
"""

import pandas as pd
import numpy as np
from services.stock_data_service import get_historical_data
from utils.logger import logger


def fetch_vix_data(timeframe='2y'):
    """
    Fetch VIX (CBOE Volatility Index) data.
    
    Args:
        timeframe: Historical period to fetch
        
    Returns:
        DataFrame with VIX data
    """
    # Fetch VIX using Tradier (symbol: ^VIX or VIX)
    try:
        vix_df = get_historical_data('^VIX', timeframe, use_cache=True)
        if vix_df.empty:
            # Try without caret
            vix_df = get_historical_data('VIX', timeframe, use_cache=True)
        
        if not vix_df.empty:
            logger.info(f"Fetched VIX data: {len(vix_df)} records")
        else:
            logger.warning("VIX data not available, will use fallback")
            
        return vix_df
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
        return pd.DataFrame()


def classify_volatility_regime(vix_value):
    """
    Classify market volatility regime based on VIX level.
    
    Args:
        vix_value: Current VIX level
        
    Returns:
        String: 'low_vol', 'medium_vol', or 'high_vol'
    """
    if vix_value < 15:
        return 'low_vol'
    elif vix_value < 25:
        return 'medium_vol'
    else:
        return 'high_vol'


def detect_market_regime(price_series, window=50):
    """
    Detect market regime (bull/bear/sideways) based on price action.
    
    Args:
        price_series: Series of closing prices
        window: Lookback window for regime detection
        
    Returns:
        Series with regime labels
    """
    regimes = []
    
    for i in range(len(price_series)):
        if i < window:
            regimes.append('sideways')  # Default for insufficient data
            continue
        
        # Get recent price window
        recent_prices = price_series.iloc[i-window:i]
        
        # Calculate trend metrics
        first_price = recent_prices.iloc[0]
        last_price = recent_prices.iloc[-1]
        change_pct = ((last_price - first_price) / first_price) * 100
        
        # Calculate price volatility
        returns = recent_prices.pct_change().dropna()
        volatility = returns.std()
        
        # Classify regime
        if change_pct > 10 and volatility < 0.02:
            regime = 'bull'
        elif change_pct < -10 and volatility < 0.02:
            regime = 'bear'
        else:
            regime = 'sideways'
        
        regimes.append(regime)
    
    return pd.Series(regimes, index=price_series.index)


def add_market_context_features(df, ticker):
    """
    Add market context features to stock data.
    
    Adds:
    - VIX level and regime
    - SPY correlation and relative strength
    - QQQ correlation (for tech stocks)
    - Market regime
    
    Args:
        df: Stock OHLCV DataFrame
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with added market context features
    """
    data = df.copy()
    
    # 1. Fetch and add VIX data
    vix_df = fetch_vix_data('2y')
    if not vix_df.empty:
        # Align VIX data with stock data
        vix_aligned = vix_df['Close'].reindex(data.index, method='ffill')
        data['VIX'] = vix_aligned
        data['VIX_Regime'] = data['VIX'].apply(lambda x: classify_volatility_regime(x) if pd.notna(x) else 'medium_vol')
        
        # VIX change
        data['VIX_Change'] = data['VIX'].pct_change()
        
        # VIX MA for trend
        data['VIX_MA_20'] = data['VIX'].rolling(window=20).mean()
        data['VIX_Above_MA'] = (data['VIX'] > data['VIX_MA_20']).astype(int)
    else:
        # Fallback if VIX not available
        logger.warning("VIX not available, using default values")
        data['VIX'] = 20  # Default medium volatility
        data['VIX_Regime'] = 'medium_vol'
        data['VIX_Change'] = 0
        data['VIX_MA_20'] = 20
        data['VIX_Above_MA'] = 0
    
    # 2. SPY correlation (market benchmark)
    spy_df = get_historical_data('SPY', '2y', use_cache=True)
    if not spy_df.empty:
        spy_aligned = spy_df['Close'].reindex(data.index, method='ffill')
        
        # Relative strength vs SPY
        data['RS_vs_SPY'] = (data['Close'] / spy_aligned) * 100
        data['RS_Change'] = data['RS_vs_SPY'].pct_change()
        
        # Rolling correlation with SPY
        stock_returns = data['Close'].pct_change()
        spy_returns = spy_aligned.pct_change()
        data['SPY_Correlation'] = stock_returns.rolling(window=20).corr(spy_returns)
        
        # Beta (stock volatility vs market)
        data['Beta_SPY'] = (stock_returns.rolling(window=20).std() / 
                           spy_returns.rolling(window=20).std())
    else:
        data['RS_vs_SPY'] = 100
        data['RS_Change'] = 0
        data['SPY_Correlation'] = 0.5
        data['Beta_SPY'] = 1.0
    
    # 3. QQQ correlation (for tech exposure)
    qqq_df = get_historical_data('QQQ', '2y', use_cache=True)
    if not qqq_df.empty:
        qqq_aligned = qqq_df['Close'].reindex(data.index, method='ffill')
        
        stock_returns = data['Close'].pct_change()
        qqq_returns = qqq_aligned.pct_change()
        data['QQQ_Correlation'] = stock_returns.rolling(window=20).corr(qqq_returns)
    else:
        data['QQQ_Correlation'] = 0.5
    
    # 4. Market regime detection
    data['Market_Regime'] = detect_market_regime(data['Close'], window=50)
    
    # Convert categorical to numeric for ML
    regime_map = {'bull': 1, 'sideways': 0, 'bear': -1}
    data['Market_Regime_Numeric'] = data['Market_Regime'].map(regime_map)
    
    vix_regime_map = {'low_vol': 0, 'medium_vol': 1, 'high_vol': 2}
    data['VIX_Regime_Numeric'] = data['VIX_Regime'].map(vix_regime_map)
    
    logger.info(f"Added market context features for {ticker}")
    
    return data


def get_current_market_context():
    """
    Get current market context snapshot.
    
    Returns:
        Dict with current market conditions
    """
    # Fetch SPY for market direction
    spy_df = get_historical_data('SPY', '3m', use_cache=True)
    spy_current = float(spy_df['Close'].iloc[-1]) if not spy_df.empty else None
    
    # Fetch VIX for volatility
    vix_df = fetch_vix_data('3m')
    vix_current = float(vix_df['Close'].iloc[-1]) if not vix_df.empty else None
    
    # Determine regime
    if spy_df is not None and not spy_df.empty:
        spy_regime = detect_market_regime(spy_df['Close'], window=50)
        current_regime = spy_regime.iloc[-1]
    else:
        current_regime = 'unknown'
    
    vix_regime = classify_volatility_regime(vix_current) if vix_current else 'unknown'
    
    return {
        'spy_price': spy_current,
        'vix_level': vix_current,
        'vix_regime': vix_regime,
        'market_regime': current_regime,
        'timestamp': pd.Timestamp.now()
    }


def should_retrain_model(ticker, current_performance, threshold_mae=15.0):
    """
    Determine if model should be retrained based on performance degradation.
    
    Args:
        ticker: Stock ticker
        current_performance: Dict with current MAE, RMSE
        threshold_mae: MAE threshold for retraining
        
    Returns:
        Tuple of (should_retrain: bool, reason: str)
    """
    if not current_performance or 'mae' not in current_performance:
        return False, "No performance data available"
    
    current_mae = current_performance['mae']
    
    # Retrain if MAE exceeds threshold
    if current_mae > threshold_mae:
        return True, f"MAE ({current_mae:.2f}) exceeds threshold ({threshold_mae})"
    
    # Check if directional accuracy is poor
    if 'directional_accuracy' in current_performance:
        dir_acc = current_performance['directional_accuracy']
        if dir_acc < 50:
            return True, f"Directional accuracy ({dir_acc:.1f}%) below 50%"
    
    # Check market regime change
    context = get_current_market_context()
    if context['vix_regime'] == 'high_vol':
        return True, "High volatility regime - model may need retuning"
    
    return False, "Performance acceptable"
