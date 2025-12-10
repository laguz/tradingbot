"""
Risk Manager Service

Manages trading risk limits and circuit breakers for automated trading.
"""

from datetime import datetime, date
from typing import Dict, List, Optional
from models.mongodb_models import MLPredictionModel
from services.tradier_service import get_open_positions, get_account_summary
from utils.logger import logger
from config import get_config

config = get_config()


class RiskManager:
    """Manages risk limits for automated trading."""
    
    def __init__(self):
        """Initialize risk manager."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.today = date.today().isoformat()
        self.emergency_stop = False
    
    def check_can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.
        
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check emergency stop
        if self.emergency_stop:
            return False, "Emergency stop activated"
        
        # Check if auto-trading is enabled (or if execution is skipped via dry run)
        if not config.AUTO_TRADE_ENABLED and not config.AUTO_TRADE_DRY_RUN:
            return False, "Auto-trading is disabled in config"
        
        # Update daily P&L
        self._update_daily_pnl()
        
        # Check daily loss limit
        if self.daily_pnl < -config.AUTO_TRADE_MAX_DAILY_LOSS:
            logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False, f"Daily loss limit exceeded (${self.daily_pnl:.2f})"
        
        # Check position count
        positions = get_open_positions()
        if positions and len(positions) >= config.AUTO_TRADE_MAX_POSITIONS:
            return False, f"Max positions reached ({len(positions)}/{config.AUTO_TRADE_MAX_POSITIONS})"
        
        # Check market hours (simplified - you may want to add proper market calendar)
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday or Sunday
            return False, "Market closed (weekend)"
        
        if now.hour < 9 or now.hour >= 16:  # Before 9 AM or after 4 PM ET
            return False, "Outside trading hours (9 AM - 4 PM ET)"
        
        return True, "OK"
    
    def _update_daily_pnl(self):
        """Update daily profit/loss from account."""
        try:
            # Reset if new day
            today_str = date.today().isoformat()
            if today_str != self.today:
                self.today = today_str
                self.daily_pnl = 0.0
                self.daily_trades = 0
                logger.info("New trading day - reset P&L tracking")
            
            # Get current day P&L from account
            summary = get_account_summary()
            if summary and 'day_pl' in summary:
                self.daily_pnl = float(summary['day_pl'] or 0)
                
        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")
    
    def check_position_size(self, quantity: int) -> tuple[bool, str]:
        """
        Check if position size is within limits.
        
        Args:
            quantity: Number of contracts
            
        Returns:
            Tuple of (allowed, reason)
        """
        if quantity > config.AUTO_TRADE_MAX_POSITION_SIZE:
            return False, f"Position size {quantity} exceeds limit {config.AUTO_TRADE_MAX_POSITION_SIZE}"
        
        if quantity <= 0:
            return False, "Position size must be positive"
        
        return True, "OK"
    
    def check_symbol_allowed(self, symbol: str) -> tuple[bool, str]:
        """
        Check if symbol is in allowed list.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Tuple of (allowed, reason)
        """
        symbol = symbol.upper()
        
        if symbol not in config.AUTO_TRADE_SYMBOLS:
            return False, f"Symbol {symbol} not in allowed list: {config.AUTO_TRADE_SYMBOLS}"
        
        return True, "OK"
    
    def activate_emergency_stop(self, reason: str = "Manual"):
        """
        Activate emergency stop - halts all trading.
        
        Args:
            reason: Why emergency stop was activated
        """
        self.emergency_stop = True
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def deactivate_emergency_stop(self):
        """Deactivate emergency stop."""
        self.emergency_stop = False
        logger.info("Emergency stop deactivated")
    
    def get_status(self) -> Dict:
        """
        Get current risk status.
        
        Returns:
            Dict with status information
        """
        self._update_daily_pnl()
        
        positions = get_open_positions()
        position_count = len(positions) if positions else 0
        
        can_trade, reason = self.check_can_trade()
        
        return {
            'can_trade': can_trade,
            'reason': reason,
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': config.AUTO_TRADE_MAX_DAILY_LOSS,
            'position_count': position_count,
            'max_positions': config.AUTO_TRADE_MAX_POSITIONS,
            'emergency_stop': self.emergency_stop,
            'auto_trade_enabled': config.AUTO_TRADE_ENABLED,
            'dry_run_mode': config.AUTO_TRADE_DRY_RUN
        }
    
    def log_trade(self, trade_info: Dict):
        """
        Log trade for monitoring.
        
        Args:
            trade_info: Trade details
        """
        self.daily_trades += 1
        logger.info(f"Trade #{self.daily_trades} logged: {trade_info}")


# Global risk manager instance
risk_manager = RiskManager()
