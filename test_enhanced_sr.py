"""
Test the enhanced support/resistance algorithm.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.stock_data_service import get_historical_data
from services.market_analysis_enhanced import find_support_resistance_enhanced
from config import get_config

config = get_config()


def test_enhanced_sr():
    """Test enhanced S/R with a real symbol."""
    
    print("=" * 80)
    print("Testing Enhanced Support/Resistance Algorithm with RSI/MACD")
    print("=" * 80)
    
    symbol = 'SPY'
    print(f"\nFetching data for {symbol}...")
    
    # Get historical data
    hist = get_historical_data(symbol, '6m', use_cache=True)
    
    if hist is None or hist.empty:
        print(f"❌ Could not fetch data for {symbol}")
        return
    
    print(f"✓ Retrieved {len(hist)} bars of data")
    
    # Test enhanced algorithm
    print(f"\nRunning enhanced S/R algorithm...")
    print(f"  SR_USE_INDICATORS: {config.SR_USE_INDICATORS}")
    print(f"  SR_RSI_OVERSOLD: {config.SR_RSI_OVERSOLD}")
    print(f"  SR_RSI_OVERBOUGHT: {config.SR_RSI_OVERBOUGHT}")
    print(f"  SR_INDICATOR_WEIGHT: {config.SR_INDICATOR_WEIGHT}")
    
    supports, resistances = find_support_resistance_enhanced(
        hist,
        rsi_oversold=config.SR_RSI_OVERSOLD,
        rsi_overbought=config.SR_RSI_OVERBOUGHT,
        indicator_weight=config.SR_INDICATOR_WEIGHT
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"SUPPORT LEVELS ({len(supports)} found)")
    print(f"{'='*80}")
    
    if supports:
        print(f"{'Price':<12} {'Strength':<12} {'Indicator Bonus':<20} {'Final Score':<15}")
        print("-" * 80)
        for s in supports:
            print(f"${s['price']:<11.2f} {s['strength']:<12} {s['indicator_bonus']:<20.2f} {s['score']:<15.2f}")
    else:
        print("No support levels found")
    
    print(f"\n{'='*80}")
    print(f"RESISTANCE LEVELS ({len(resistances)} found)")
    print(f"{'='*80}")
    
    if resistances:
        print(f"{'Price':<12} {'Strength':<12} {'Indicator Bonus':<20} {'Final Score':<15}")
        print("-" * 80)
        for r in resistances:
            print(f"${r['price']:<11.2f} {r['strength']:<12} {r['indicator_bonus']:<20.2f} {r['score']:<15.2f}")
    else:
        print("No resistance levels found")
    
    # Count strong levels
    strong_supports = sum(1 for s in supports if s['strength'] == 'Strong')
    strong_resistances = sum(1 for r in resistances if r['strength'] == 'Strong')
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Strong Support Levels: {strong_supports}/{len(supports)}")
    print(f"Strong Resistance Levels: {strong_resistances}/{len(resistances)}")
    print(f"\n✅ Enhanced algorithm test complete!")


if __name__ == '__main__':
    test_enhanced_sr()
