"""
Enhanced Market Analysis Service

Enhanced support/resistance detection with RSI and MACD confirmation.
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from datetime import date, datetime
from config import get_config
from utils.logger import logger

config = get_config()


def calculate_rsi_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RSI and MACD indicators if not already present.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with RSI and MACD columns added
    """
    data = df.copy()
    
    # Calculate RSI if not present
    if 'RSI' not in data.columns:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD if not present
    if 'MACD' not in data.columns:
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    return data


def find_support_resistance_enhanced(
    data: pd.DataFrame, 
    window: int = None, 
    tolerance: float = None,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    indicator_weight: float = 0.3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Enhanced support/resistance detection with RSI and MACD confirmation.
    
    Args:
        data: DataFrame with OHLCV data
        window: Sliding window size for pivot detection
        tolerance: Percentage tolerance for clustering levels
        rsi_oversold: RSI oversold threshold (default: 30)
        rsi_overbought: RSI overbought threshold (default: 70)
        indicator_weight: Weight for indicator bonus (0-1, default: 0.3)
        
    Returns:
        Tuple of (support_data, resistance_data)
        Each item: {'price': float, 'score': float, 'strength': str, 'indicator_bonus': float}
    """
    if window is None:
        window = config.SR_WINDOW
    if tolerance is None:
        tolerance = config.SR_TOLERANCE
        
    if data.empty:
        return [], []
    
    # Ensure RSI and MACD are calculated
    data = calculate_rsi_macd(data)
    
    supports = []
    resistances = []
    
    # 1. Identify Pivot Points with Indicator Context
    for i in range(window, len(data) - window):
        # Support Pivot (Local Low)
        if data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window+1].min():
            # Calculate indicator bonus for support
            bonus = 0.0
            rsi_val = data['RSI'].iloc[i]
            macd_hist = data['MACD_Histogram'].iloc[i]
            
            # RSI oversold bonus
            if pd.notna(rsi_val) and rsi_val <= rsi_oversold:
                bonus += 0.5
            
            # MACD bullish (histogram positive after bounce)
            if pd.notna(macd_hist) and macd_hist > 0:
                bonus += 0.2
            
            # Check for MACD bullish crossover within next 3 bars
            for j in range(i, min(i+4, len(data))):
                if j > 0 and pd.notna(data['MACD'].iloc[j]) and pd.notna(data['Signal_Line'].iloc[j]):
                    if data['MACD'].iloc[j] > data['Signal_Line'].iloc[j] and \
                       data['MACD'].iloc[j-1] <= data['Signal_Line'].iloc[j-1]:
                        bonus += 0.3
                        break
            
            supports.append({
                'price': data['Low'].iloc[i],
                'volume': data['Volume'].iloc[i],
                'indicator_bonus': bonus
            })
            
        # Resistance Pivot (Local High)
        if data['High'].iloc[i] == data['High'].iloc[i-window:i+window+1].max():
            # Calculate indicator bonus for resistance
            bonus = 0.0
            rsi_val = data['RSI'].iloc[i]
            macd_hist = data['MACD_Histogram'].iloc[i]
            
            # RSI overbought bonus
            if pd.notna(rsi_val) and rsi_val >= rsi_overbought:
                bonus += 0.5
            
            # MACD bearish (histogram negative after rejection)
            if pd.notna(macd_hist) and macd_hist < 0:
                bonus += 0.2
            
            # Check for MACD bearish crossover within next 3 bars
            for j in range(i, min(i+4, len(data))):
                if j > 0 and pd.notna(data['MACD'].iloc[j]) and pd.notna(data['Signal_Line'].iloc[j]):
                    if data['MACD'].iloc[j] < data['Signal_Line'].iloc[j] and \
                       data['MACD'].iloc[j-1] >= data['Signal_Line'].iloc[j-1]:
                        bonus += 0.3
                        break
            
            resistances.append({
                'price': data['High'].iloc[i],
                'volume': data['Volume'].iloc[i],
                'indicator_bonus': bonus
            })

    def cluster_and_score_levels(levels):
        """Cluster levels and calculate final scores with strength classification."""
        if not levels:
            return []
        
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
        
        # Score Clusters with Indicator Enhancement
        scored_clusters = []
        for cluster in clusters:
            avg_price = sum(l['price'] for l in cluster) / len(cluster)
            total_volume = sum(l['volume'] for l in cluster)
            touch_count = len(cluster)
            avg_indicator_bonus = sum(l['indicator_bonus'] for l in cluster) / len(cluster)
            
            # Base score = Volume * Touches
            base_score = total_volume * touch_count
            
            # Enhanced score = base_score * (1 + weighted_bonus)
            indicator_score = avg_indicator_bonus * indicator_weight
            final_score = base_score * (1 + indicator_score)
            
            # Classify strength
            if avg_indicator_bonus > 0.7:
                strength = "Strong"
            elif avg_indicator_bonus > 0.3:
                strength = "Medium"
            else:
                strength = "Weak"
            
            scored_clusters.append({
                'price': avg_price,
                'score': final_score,
                'indicator_bonus': avg_indicator_bonus,
                'strength': strength
            })
            
        # Sort by score and take top N
        scored_clusters.sort(key=lambda x: x['score'], reverse=True)
        return scored_clusters[:config.SR_MAX_LEVELS]

    final_supports = cluster_and_score_levels(supports)
    final_resistances = cluster_and_score_levels(resistances)
    
    # Sort by price for easier use
    final_supports = sorted(final_supports, key=lambda x: x['price'])
    final_resistances = sorted(final_resistances, key=lambda x: x['price'])
    
    logger.debug(
        f"Enhanced S/R: {len(final_supports)} supports, {len(final_resistances)} resistances "
        f"(Strong supports: {sum(1 for s in final_supports if s['strength'] == 'Strong')})"
    )
    
    return final_supports, final_resistances


def find_support_resistance_simple(
    data: pd.DataFrame, 
    window: int = None, 
    tolerance: float = None
) -> Tuple[List[float], List[float]]:
    """
    Original simple support/resistance detection (for backward compatibility).
    Returns just price levels without scores.
    """
    enhanced_supports, enhanced_resistances = find_support_resistance_enhanced(
        data, window, tolerance, indicator_weight=0  # No indicator bonus
    )
    
    support_prices = [s['price'] for s in enhanced_supports]
    resistance_prices = [r['price'] for r in enhanced_resistances]
    
    return support_prices, resistance_prices
