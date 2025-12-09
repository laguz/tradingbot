"""
Automated Trader Bot

Runs automated trading strategies based on ML predictions.
Execute this script on a schedule or manually to run trading algorithms.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from services.risk_manager import risk_manager
from services.trading_engine import trading_engine
from services.tradier_service import check_and_close_positions, check_and_roll_positions
from utils.logger import logger
from config import get_config

config = get_config()


def run_automated_trader():
    """Main automated trading loop."""
    
    logger.info("=" * 60)
    logger.info("AUTOMATED TRADER STARTED")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'DRY RUN' if config.AUTO_TRADE_DRY_RUN else 'LIVE TRADING'}")
    logger.info("=" * 60)
    
    # Check if trading is allowed
    can_trade, reason = risk_manager.check_can_trade()
    
    if not can_trade:
        logger.warning(f"Trading not allowed: {reason}")
        print_status()
        return
    
    logger.info("✓ Risk checks passed - proceeding with trading")
    
    # Step 1: Analyze opportunities
    logger.info("\n--- Step 1: Analyzing Opportunities ---")
    opportunities = analyze_opportunities()
    
    if not opportunities:
        logger.info("No trading opportunities found")
    
    # Step 2: Execute new trades
    logger.info("\n--- Step 2: Executing New Trades ---")
    new_trades = execute_new_trades(opportunities)
    
    # Step 3: Manage existing positions
    logger.info("\n--- Step 3: Managing Existing Positions ---")
    manage_existing_positions()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("AUTOMATED TRADER COMPLETE")
    logger.info(f"New trades executed: {len(new_trades)}")
    print_status()
    logger.info("=" * 60)


def analyze_opportunities():
    """Analyze trading opportunities across configured symbols."""
    opportunities = []
    
    for symbol in config.AUTO_TRADE_SYMBOLS:
        # Check symbol is allowed
        allowed, reason = risk_manager.check_symbol_allowed(symbol)
        if not allowed:
            continue
        
        # Analyze ML opportunity
        opp = trading_engine.analyze_ml_opportunity(symbol)
        
        if opp and opp.get('strategy'):
            opportunities.append(opp)
            logger.info(f"  ✓ {symbol}: {opp['strategy']} (confidence: {opp['confidence']:.2f})")
        else:
            logger.debug(f"  - {symbol}: No opportunity")
    
    return opportunities


def execute_new_trades(opportunities):
    """Execute new trades based on opportunities."""
    executed = []
    
    if not opportunities:
        logger.info("No opportunities to execute")
        return executed
    
    for opp in opportunities:
        # Check if we can still trade
        can_trade, reason = risk_manager.check_can_trade()
        if not can_trade:
            logger.warning(f"Stopping new trades: {reason}")
            break
        
        # Execute based on strategy
        result = None
        
        if opp['strategy'] == 'sell_put':
            result = trading_engine.execute_wheel_strategy(opp['symbol'], opp)
        elif opp['strategy'] in ['sell_call', 'put_spread', 'call_spread']:
            result = trading_engine.execute_credit_spread_strategy(opp['symbol'], opp)
        
        if result:
            executed.append(result)
            logger.info(f"  ✓ Executed: {opp['symbol']} {opp['strategy']}")
        else:
            logger.info(f"  - Skipped: {opp['symbol']} {opp['strategy']}")
    
    return executed


def manage_existing_positions():
    """Manage existing positions (auto-close, auto-roll)."""
    try:
        # Auto-close profitable positions
        logger.info("Checking for positions to close...")
        close_results = check_and_close_positions()
        
        if close_results:
            closed_count = len(close_results.get('closed', []))
            logger.info(f"  Closed {closed_count} positions")
            
            if close_results.get('closed'):
                for pos in close_results['closed']:
                    logger.info(f"    ✓ Closed {pos['symbol']}: {pos['reason']}")
        
        # Auto-roll positions
        logger.info("Checking for positions to roll...")
        roll_results = check_and_roll_positions()
        
        if roll_results:
            rolled_count = len(roll_results.get('rolled', []))
            logger.info(f"  Rolled {rolled_count} positions")
            
            if roll_results.get('rolled'):
                for pos in roll_results['rolled']:
                    logger.info(f"    ✓ Rolled {pos['symbol']}: {pos['reason']}")
        
    except Exception as e:
        logger.error(f"Error managing positions: {e}")


def print_status():
    """Print current risk status."""
    status = risk_manager.get_status()
    
    logger.info("\n--- Risk Status ---")
    logger.info(f"Can Trade: {status['can_trade']}")
    logger.info(f"Reason: {status['reason']}")
    logger.info(f"Daily P&L: ${status['daily_pnl']:.2f}")
    logger.info(f"Daily Loss Limit: ${status['daily_loss_limit']:.2f}")
    logger.info(f"Open Positions: {status['position_count']}/{status['max_positions']}")
    logger.info(f"Emergency Stop: {status['emergency_stop']}")
    logger.info(f"Dry Run Mode: {status['dry_run_mode']}")


if __name__ == '__main__':
    try:
        # Initialize database
        init_db()
        
        # Run trader
        run_automated_trader()
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\nTrader stopped by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in automated trader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
