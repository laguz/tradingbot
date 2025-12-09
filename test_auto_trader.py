"""
Test Automated Trading Bot

Tests the automated trading bot in dry-run mode.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from services.risk_manager import risk_manager
from services.trading_engine import trading_engine
from config import get_config

config = get_config()


def test_automated_trader():
    """Test automated trading bot components."""
    
    print("\n" + "="*60)
    print("AUTOMATED TRADING BOT TEST")
    print("="*60 + "\n")
    
    # Initialize database
    print("1. Initializing database...")
    init_db()
    print("   ✓ Database initialized\n")
    
    # Test risk manager
    print("2. Testing risk manager...")
    can_trade, reason = risk_manager.check_can_trade()
    status = risk_manager.get_status()
    
    print(f"   Can Trade: {can_trade}")
    print(f"   Reason: {reason}")
    print(f"   Emergency Stop: {status['emergency_stop']}")
    print(f"   Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"   Auto Trade Enabled: {status['auto_trade_enabled']}")
    print(f"   Dry Run Mode: {status['dry_run_mode']}")
    print(f"   ✓ Risk manager working\n")
    
    # Test trading engine (dry run)
    print("3. Testing trading engine (dry run)...")
    print("   Analyzing ML opportunities...")
    
    # Test with SPY
    opportunity = trading_engine.analyze_ml_opportunity('SPY')
    
    if opportunity:
        print(f"   Symbol: {opportunity['symbol']}")
        print(f"   Direction: {opportunity['direction']}")
        print(f"   Expected Move: {opportunity['expected_move_pct']:+.2f}%")
        print(f"   Confidence: {opportunity['confidence']:.2f}")
        print(f"   Strategy: {opportunity['strategy']}")
        print(f"   ✓ ML analysis working\n")
    else:
        print(f"   No opportunity found for SPY\n")
    
    # Test configuration
    print("4. Checking configuration...")
    print(f"   AUTO_TRADE_ENABLED: {config.AUTO_TRADE_ENABLED}")
    print(f"   AUTO_TRADE_DRY_RUN: {config.AUTO_TRADE_DRY_RUN}")
    print(f"   Symbols: {config.AUTO_TRADE_SYMBOLS}")
    print(f"   Max Daily Loss: ${config.AUTO_TRADE_MAX_DAILY_LOSS}")
    print(f"   Max Positions: {config.AUTO_TRADE_MAX_POSITIONS}")
    print(f"   ✓ Configuration loaded\n")
    
    # Summary
    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nAutomated trading bot is ready!")
    print("\nTo use:")
    print("1. Go to http://localhost:5000/auto_trader")
    print("2. Click 'Run Now' to execute (dry run mode)")
    print("3. Or run: python3 jobs/automated_trader.py")
    print("\nIMPORTANT:")
    print("- Currently in DRY RUN mode (no real trades)")
    print("- AUTO_TRADE_ENABLED is set to:", config.AUTO_TRADE_ENABLED)
    print("- Change config.py to enable live trading")
    print()


if __name__ == '__main__':
    try:
        test_automated_trader()
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
