"""
Market Analysis Service

Handles technical analysis and strategy-specific calculations
that are independent of the data source.
"""

from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
from datetime import date, datetime
from config import get_config
from utils.logger import logger

# Import enhanced S/R if indicators enabled
try:
    from services.market_analysis_enhanced import (
        find_support_resistance_enhanced,
        find_support_resistance_simple
    )
    ENHANCED_SR_AVAILABLE = True
except ImportError:
    ENHANCED_SR_AVAILABLE = False

config = get_config()

def custom_round(price: float) -> float:
    """
    Round price to nearest logical increment.
    """
    if price < 100:
        return round(price)
    else:
        return round(price / 5) * 5

def find_support_resistance(data: pd.DataFrame, window: int = None, tolerance: float = None) -> Tuple[List[float], List[float]]:
    """
    Identifies support and resistance levels using volume-weighted clustering.
    Automatically uses enhanced algorithm with RSI/MACD if SR_USE_INDICATORS is enabled.
    
    Args:
        data: DataFrame with Low, High, Volume columns
        window: Sliding window size for pivot detection
        tolerance: Percentage tolerance for clustering levels
        
    Returns:
        Tuple of (support_levels, resistance_levels) as price lists
    """
    # Use enhanced algorithm if enabled and available
    if ENHANCED_SR_AVAILABLE and config.SR_USE_INDICATORS:
        return find_support_resistance_simple(data, window, tolerance)
    
    # Otherwise use original implementation
    if window is None:
        window = config.SR_WINDOW
    if tolerance is None:
        tolerance = config.SR_TOLERANCE
        
    if data.empty: return [], []
    
    supports = []
    resistances = []
    
    # 1. Identify Pivot Points
    for i in range(window, len(data) - window):
        # Support Pivot (Local Low)
        if data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window+1].min():
            supports.append({'price': data['Low'].iloc[i], 'volume': data['Volume'].iloc[i]})
            
        # Resistance Pivot (Local High)
        if data['High'].iloc[i] == data['High'].iloc[i-window:i+window+1].max():
            resistances.append({'price': data['High'].iloc[i], 'volume': data['Volume'].iloc[i]})

    def cluster_levels(levels):
        if not levels: return []
        
        levels.sort(key=lambda x: x['price'])
        clusters = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            price = levels[i]['price']
            prev_price = current_cluster[-1]['price']
            
            # Check if within tolerance
            if price <= prev_price * (1 + tolerance):
                current_cluster.append(levels[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [levels[i]]
        clusters.append(current_cluster)
        
        # Score Clusters
        scored_clusters = []
        for cluster in clusters:
            avg_price = sum(l['price'] for l in cluster) / len(cluster)
            total_volume = sum(l['volume'] for l in cluster)
            touch_count = len(cluster)
            
            # Score = Volume * Touches (Simple heuristic)
            score = total_volume * touch_count
            scored_clusters.append({'price': avg_price, 'score': score})
            
        # Sort by score and take top N significant levels
        scored_clusters.sort(key=lambda x: x['score'], reverse=True)
        return sorted([c['price'] for c in scored_clusters[:config.SR_MAX_LEVELS]])

    final_supports = cluster_levels(supports)
    final_resistances = cluster_levels(resistances)
    
    logger.debug(f"Found {len(final_supports)} support levels and {len(final_resistances)} resistance levels")
    return final_supports, final_resistances

def calculate_smart_strikes(
    symbol: str, 
    expiration: str, 
    spread_type: str, 
    option_type: str, 
    width: float,
    current_price: float,
    support_levels: List[float],
    resistance_levels: List[float],
    available_strikes: List[float]
) -> Tuple[float, float, str]:
    """
    Calculates optimal strikes based on Support (Put Credit) or Resistance (Call Credit).
    
    Args:
        symbol: Ticker symbol
        expiration: Expiration date string
        spread_type: 'credit' or 'debit' (currently only 'credit' supported for smart selection)
        option_type: 'call' or 'put'
        width: Spread width
        current_price: Current underlying price
        support_levels: List of identified support prices
        resistance_levels: List of identified resistance prices
        available_strikes: List of available strike prices from option chain
        
    Returns:
        Tuple of (short_strike, long_strike, trigger_level_description)
    """
    if not support_levels and not resistance_levels:
         # Log warning but continue with fallback? Or raise?
         # For now, let's allow it to proceed to fallback logic if lists are empty
         pass

    # 1. Determine Target Price based on Strategy
    target_price = None
    trigger_level = None # The level (support/resistance) used for decision
    
    if spread_type == 'credit' and option_type == 'put':
        # Bullish: Sell Put AT or BELOW Support
        # Find the closest support level below current price
        valid_supports = [s for s in support_levels if s < current_price]
        
        # Safety Buffer: Ensure we are at least 1% OTM
        safety_threshold = current_price * 0.99
        
        if valid_supports:
            closest_support = valid_supports[-1]
            # Use the lower of the two (further OTM) to ensure safety
            target_price = min(closest_support, safety_threshold)
            trigger_level = closest_support
        else:
            target_price = current_price * 0.95 # Fallback: 5% OTM
            trigger_level = "Fallback (5% OTM)"
            
    elif spread_type == 'credit' and option_type == 'call':
        # Bearish: Sell Call AT or ABOVE Resistance
        # Find the closest resistance level above current price
        valid_resistances = [r for r in resistance_levels if r > current_price]
        
        # Safety Buffer: Ensure we are at least 1% OTM
        safety_threshold = current_price * 1.01
        
        if valid_resistances:
            closest_resistance = valid_resistances[0]
            # Use the higher of the two (further OTM) to ensure safety
            target_price = max(closest_resistance, safety_threshold)
            trigger_level = closest_resistance
        else:
            target_price = current_price * 1.05 # Fallback: 5% OTM
            trigger_level = "Fallback (5% OTM)"
    else:
        raise ValueError("Auto-selection currently only supports Credit Spreads (Put/Call).")

    # 2. Filter strikes
    if not available_strikes:
        raise ValueError("No strikes provided.")

    # 3. Select Short Strike (Closest to Target Price)
    # We want to sell the option closest to our target level
    short_strike = min(available_strikes, key=lambda x: abs(x - target_price))
    
    # 4. Calculate Long Strike
    if option_type == 'put':
        # Put Credit Spread: Long Strike is LOWER than Short Strike
        long_strike_target = short_strike - width
    else:
        # Call Credit Spread: Long Strike is HIGHER than Short Strike
        long_strike_target = short_strike + width
        
    # Find closest real strike to the calculated long target
    long_strike = min(available_strikes, key=lambda x: abs(x - long_strike_target))
    
    # Validation: Ensure spread width is maintained roughly (don't collapse the spread)
    if option_type == 'put' and long_strike >= short_strike:
         # Try to find a lower strike
         lower_strikes = [s for s in available_strikes if s < short_strike]
         if lower_strikes: long_strike = lower_strikes[-1] # Highest of the lower strikes
         
    if option_type == 'call' and long_strike <= short_strike:
         # Try to find a higher strike
         higher_strikes = [s for s in available_strikes if s > short_strike]
         if higher_strikes: long_strike = higher_strikes[0] # Lowest of the higher strikes

    return short_strike, long_strike, trigger_level
