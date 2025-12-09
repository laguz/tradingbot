"""
Regime-Adaptive ML Module

Adapts model parameters and feature selection based on market regime.
Improves predictions during different market conditions (bull/bear/high vol).
"""

import pandas as pd
import numpy as np
from services.ml_market_context import get_current_market_context, classify_volatility_regime
from utils.logger import logger
from config import get_config

config = get_config()


def detect_regime(df):
    """
    Detect current market regime from historical data.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Dict with regime classification
    """
    # Get last 50 days for regime detection
    recent = df.tail(50)
    
    # Calculate metrics
    returns = recent['Close'].pct_change()
    cum_return = ((recent['Close'].iloc[-1] / recent['Close'].iloc[0]) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Classify trend
    if cum_return > 10:
        trend = 'bull'
    elif cum_return < -10:
        trend = 'bear'
    else:
        trend = 'sideways'
    
    # Classify volatility
    if volatility < 15:
        vol_regime = 'low_vol'
    elif volatility < 30:
        vol_regime = 'medium_vol'
    else:
        vol_regime = 'high_vol'
    
    regime = {
        'trend': trend,
        'volatility': vol_regime,
        'cum_return_50d': cum_return,
        'realized_vol': volatility
    }
    
    logger.info(f"Detected regime: {trend}, {vol_regime} ({volatility:.1f}% vol)")
    
    return regime


def get_regime_hyperparameters(regime):
    """
    Get optimal hyperparameters for current market regime.
    
    Args:
        regime: Dict with regime classification
        
    Returns:
        Dict with model hyperparameters
    """
    # Base parameters
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'learning_rate': 0.05
    }
    
    # Adjust for high volatility - use simpler model to avoid overfitting
    if regime['volatility'] == 'high_vol':
        params['max_depth'] = 10
        params['min_samples_split'] = 10
        params['min_samples_leaf'] = 4
        logger.info("High volatility: Using more conservative parameters")
    
    # Adjust for low volatility - can use more complex model
    elif regime['volatility'] == 'low_vol':
        params['max_depth'] = 20
        params['min_samples_split'] = 2
        params['min_samples_leaf'] = 1
        logger.info("Low volatility: Using more aggressive parameters")
    
    # Adjust learning rate for trending markets
    if regime['trend'] in ['bull', 'bear']:
        params['learning_rate'] = 0.07  # Faster learning for trends
        logger.info("Trending market: Increased learning rate")
    
    return params


def select_features_by_regime(feature_names, regime):
    """
    Select most relevant features for current regime.
    
    Args:
        feature_names: List of all feature names
        regime: Dict with regime classification
        
    Returns:
        List of selected feature names
    """
    # Core features always included
    core_features = [
        'RSI', 'MACD', 'MACD_Histogram', 'Signal_Line',
        'BB_Position', 'ATR_Percent',
        'Lag_1', 'Lag_2', 'Lag_3'
    ]
    
    selected = set(core_features)
    
    # High volatility - add volatility features
    if regime['volatility'] == 'high_vol':
        vol_features = ['HV_10', 'HV_20', 'HV_50', 'Vol_Ratio', 'Parkinson_Vol', 'ATR']
        selected.update([f for f in vol_features if f in feature_names])
        logger.info("High vol regime: Added volatility features")
    
    # Low volatility - add momentum features
    elif regime['volatility'] == 'low_vol':
        momentum_features = ['Momentum_5', 'Momentum_10', 'Momentum_20', 'ROC_5', 'ROC_10']
        selected.update([f for f in momentum_features if f in feature_names])
        logger.info("Low vol regime: Added momentum features")
    
    # Trending market - add trend features
    if regime['trend'] in ['bull', 'bear']:
        trend_features = [
            'Distance_SMA_20', 'Distance_SMA_50', 'Distance_SMA_200',
            'Distance_EMA_12', 'Distance_EMA_26'
        ]
        selected.update([f for f in trend_features if f in feature_names])
        logger.info("Trending market: Added trend features")
    
    # Sideways market - add range-bound features
    else:
        range_features = ['BB_Width', 'BB_Position', 'HighLow_Spread']
        selected.update([f for f in range_features if f in feature_names])
        logger.info("Sideways market: Added range features")
    
    # Always include lag features
    lag_features = [f for f in feature_names if f.startswith('Lag_')]
    selected.update(lag_features)
    
    # Filter to only existing features
    final_features = [f for f in feature_names if f in selected]
    
    logger.info(f"Selected {len(final_features)} features for current regime")
    
    return final_features


def should_retrain_for_regime(last_regime, current_regime):
    """
    Determine if model should be retrained due to regime change.
    
    Args:
        last_regime: Previous regime dict
        current_regime: Current regime dict
        
    Returns:
        Tuple (should_retrain: bool, reason: str)
    """
    if not last_regime:
        return False, "No previous regime"
    
    # Trend change
    if last_regime['trend'] != current_regime['trend']:
        return True, f"Trend changed: {last_regime['trend']} → {current_regime['trend']}"
    
    # Volatility regime change
    if last_regime['volatility'] != current_regime['volatility']:
        return True, f"Volatility changed: {last_regime['volatility']} → {current_regime['volatility']}"
    
    # Large volatility spike
    vol_change = abs(current_regime['realized_vol'] - last_regime['realized_vol'])
    if vol_change > 10:  # 10% absolute change
        return True, f"Volatility spike: {vol_change:.1f}% change"
    
    return False, "Regime stable"


def get_adaptive_training_window(regime):
    """
    Get optimal training window size based on regime.
    
    Args:
        regime: Dict with regime classification
        
    Returns:
        Number of days for training window
    """
    # High volatility - use shorter window (more recent data)
    if regime['volatility'] == 'high_vol':
        return 252  # 1 year
    
    # Low volatility - can use longer window
    elif regime['volatility'] == 'low_vol':
        return 504  # 2 years
    
    # Medium volatility - standard window
    else:
        return 378  # 1.5 years


class RegimeAdaptivePredictor:
    """
    Wrapper that maintains regime state and adapts predictions.
    """
    
    def __init__(self):
        """Initialize adaptive predictor."""
        self.current_regime = None
        self.last_regime = None
        self.regime_start_date = None
    
    def update_regime(self, df):
        """
        Update current regime from data.
        
        Args:
            df: DataFrame with historical prices
        """
        self.last_regime = self.current_regime
        self.current_regime = detect_regime(df)
        self.regime_start_date = pd.Timestamp.now()
        
        logger.info(f"Regime updated: {self.current_regime}")
    
    def get_hyperparameters(self):
        """Get hyperparameters for current regime."""
        if not self.current_regime:
            return None
        return get_regime_hyperparameters(self.current_regime)
    
    def get_selected_features(self, all_features):
        """Get selected features for current regime."""
        if not self.current_regime:
            return all_features
        return select_features_by_regime(all_features, self.current_regime)
    
    def should_retrain(self):
        """Check if retraining needed due to regime change."""
        if not self.current_regime:
            return False, "No regime data"
        
        return should_retrain_for_regime(self.last_regime, self.current_regime)


# Global instance
regime_predictor = RegimeAdaptivePredictor()
