"""
Improved ML Service with Quantile Regression for Confidence Intervals

Replaces fixed ±5% intervals with proper quantile prediction using GradientBoosting.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils.logger import logger


def create_quantile_models():
    """
    Create three quantile regression models for 10th, 50th, 90th percentiles.
    
    Returns:
        Dict with 'low', 'median', 'high' models
    """
    models = {}
    
    # 10th percentile (conservative lower bound)
    models['low'] = MultiOutputRegressor(
        GradientBoostingRegressor(
            loss='quantile',
            alpha=0.10,
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42
        )
    )
    
    # 50th percentile (median prediction)
    models['median'] = MultiOutputRegressor(
        GradientBoostingRegressor(
            loss='quantile',
            alpha=0.50,
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42
        )
    )
    
    # 90th percentile (optimistic upper bound)
    models['high'] = MultiOutputRegressor(
        GradientBoostingRegressor(
            loss='quantile',
            alpha=0.90,
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42
        )
    )
    
    return models


def train_quantile_models(X, y):
    """
    Train all three quantile models.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame (5 columns for 5-day predictions)
        
    Returns:
        Dict with trained 'low', 'median', 'high' models
    """
    logger.info(f"Training quantile regression models on {len(X)} samples")
    
    models = create_quantile_models()
    
    for name, model in models.items():
        logger.info(f"Training {name} quantile model...")
        model.fit(X, y)
    
    logger.info("Quantile models training complete")
    return models


def calculate_quantile_intervals(quantile_models, X_latest):
    """
    Calculate proper quantile-based confidence intervals.
    
    Args:
        quantile_models: Dict with 'low', 'median', 'high' models
        X_latest: Feature vector for latest data point
        
    Returns:
        Dict with 'low', 'median', 'high' predictions (arrays of 5 values each)
    """
    intervals = {}
    
    for name, model in quantile_models.items():
        predictions = model.predict(X_latest)
        intervals[name] = predictions[0]  # First (and only) row
    
    # Validate that low < median < high
    for i in range(5):
        assert intervals['low'][i] <= intervals['median'][i] <= intervals['high'][i], \
            f"Day {i+1}: Quantile ordering violated"
    
    logger.info("Quantile intervals calculated successfully")
    
    return {
        'low': intervals['low'].tolist(),
        'median': intervals['median'].tolist(),
        'high': intervals['high'].tolist()
    }


def get_interval_width(intervals):
    """
    Calculate average interval width as percentage.
    
    Args:
        intervals: Dict with 'low', 'median', 'high' predictions
        
    Returns:
        Float: Average interval width as percentage
    """
    widths = []
    for i in range(5):
        width_pct = ((intervals['high'][i] - intervals['low'][i]) / intervals['median'][i]) * 100
        widths.append(width_pct)
    
    avg_width = np.mean(widths)
    logger.info(f"Average prediction interval width: {avg_width:.1f}%")
    
    return avg_width
