"""
Trading Engine

Executes automated trading strategies with ML-based decision making.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from services.ml_service import predict_next_days
from services.tradier_service import (
    get_option_expirations, calculate_smart_strikes,
    place_single_option_order, place_vertical_spread_order,
    get_current_price
)
from services.risk_manager import risk_manager
from models.mongodb_models import AutoTradeModel
from utils.logger import logger
from config import get_config

config = get_config()


class TradingEngine:
    """Executes automated trading strategies."""
    
    def __init__(self):
        """Initialize trading engine."""
        self.executed_trades = []
        
    def analyze_ml_opportunity(self, symbol: str) -> Optional[Dict]:
        """
        Analyze trading opportunity using ML predictions.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with opportunity details or None
        """
        try:
            # Get ML prediction
            logger.info(f"Analyzing ML opportunity for {symbol}")
            prediction = predict_next_days(symbol, days=5)
            
            if 'error' in prediction:
                logger.warning(f"ML prediction error for {symbol}: {prediction['error']}")
                return None
            
            # Extract prediction data
            last_close = prediction['last_close']
            predictions = prediction['predictions']
            
            # Calculate average predicted price
            avg_predicted = sum(predictions) / len(predictions)
            
            # Calculate expected move percentage
            expected_move_pct = ((avg_predicted - last_close) / last_close) * 100
            
            # Determine confidence (simplified - you may want to use actual confidence intervals)
            confidence = min(1.0, abs(expected_move_pct) / 10.0)  # Higher moves = higher confidence
            
            # Determine direction
            if expected_move_pct > 2.0:  # Bullish (>2% predicted gain)
                direction = 'bullish'
                strategy = 'sell_put'  # Wheel strategy
            elif expected_move_pct < -2.0:  # Bearish (>2% predicted loss)
                direction = 'bearish'
                strategy = 'sell_call'  # not implemented yet
            else:
                direction = 'neutral'
                strategy = None
            
            opportunity = {
                'symbol': symbol,
                'current_price': last_close,
                'predicted_price': avg_predicted,
                'expected_move_pct': expected_move_pct,
                'direction': direction,
                'strategy': strategy,
                'confidence': confidence,
                'predictions': predictions
            }
            
            logger.info(f"{symbol}: {direction} ({expected_move_pct:+.2f}%), confidence: {confidence:.2f}")
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def execute_wheel_strategy(self, symbol: str, opportunity: Dict) -> Optional[Dict]:
        """
        Execute wheel strategy (sell put on bullish stock).
        
        Args:
            symbol: Stock ticker
            opportunity: Opportunity analysis from ML
            
        Returns:
            Trade result dict or None
        """
        try:
            # Check if confidence meets threshold
            if opportunity['confidence'] < config.AUTO_TRADE_MIN_CONFIDENCE:
                logger.info(f"Skipping {symbol}: confidence {opportunity['confidence']:.2f} < {config.AUTO_TRADE_MIN_CONFIDENCE}")
                return None
            
            # Find suitable expiration
            expirations = get_option_expirations(symbol)
            if 'error' in expirations:
                logger.error(f"Could not get expirations for {symbol}")
                return None
            
            target_exp = self._find_expiration(
                expirations,
                min_days=config.AUTO_TRADE_WHEEL_DTE_MIN,
                max_days=config.AUTO_TRADE_WHEEL_DTE_MAX
            )
            
            if not target_exp:
                logger.warning(f"No suitable expiration found for {symbol}")
                return None
            
            # Calculate smart strike (at support level)
            short_strike, _, trigger_level = calculate_smart_strikes(
                symbol, target_exp, 'credit', 'put', width=5.0
            )
            
            # Determine quantity based on confidence (1-5 contracts)
            quantity = min(
                config.AUTO_TRADE_MAX_POSITION_SIZE,
                max(1, int(opportunity['confidence'] * 5))
            )
            
            # Check risk limits
            allowed, reason = risk_manager.check_position_size(quantity)
            if not allowed:
                logger.warning(f"Position size check failed: {reason}")
                return None
            
            # Prepare order
            order_data = {
                'symbol': symbol,
                'expiration': target_exp,
                'option_type': 'put',
                'strike': str(short_strike),
                'side': 'sell_to_open',
                'quantity': quantity,
                'order_type': 'limit',
                'price': '0.20'  # Minimum premium
            }
            
            logger.info(f"Executing wheel trade: {symbol} {target_exp} ${short_strike}P x{quantity}")
            
            # Execute or simulate
            if config.AUTO_TRADE_DRY_RUN:
                result = {
                    'success': True,
                    'order_id': 'DRY_RUN_' + datetime.now().strftime('%Y%m%d%H%M%S'),
                    'message': 'Dry run - no real order placed'
                }
                logger.info(f"DRY RUN: Would place order: {order_data}")
            else:
                result = place_single_option_order(order_data)
            
            if result and 'error' not in result:
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy': 'wheel',
                    'symbol': symbol,
                    'action': 'open',
                    'details': order_data,
                    'ml_confidence': opportunity['confidence'],
                    'result': result,
                    'dry_run': config.AUTO_TRADE_DRY_RUN
                }
                
                risk_manager.log_trade(trade_record)
                self.executed_trades.append(trade_record)
                
                # Save to database
                AutoTradeModel.insert(trade_record)
                
                return trade_record
            else:
                logger.error(f"Order failed: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing wheel strategy for {symbol}: {e}")
            return None
    
    def execute_credit_spread_strategy(self, symbol: str, opportunity: Dict) -> Optional[Dict]:
        """
        Execute credit spread strategy.
        
        Args:
            symbol: Stock ticker
            opportunity: Opportunity analysis
            
        Returns:
            Trade result or None
        """
        try:
            # Determine spread type (put or call credit spread)
            if opportunity['direction'] == 'bullish':
                option_type = 'put'
            elif opportunity['direction'] == 'bearish':
                option_type = 'call'
            else:
                return None  # Skip neutral
            
            # Find expiration
            expirations = get_option_expirations(symbol)
            if 'error' in expirations:
                return None
            
            target_exp = self._find_expiration(
                expirations,
                min_days=config.AUTO_TRADE_SPREAD_DTE_MIN,
                max_days=config.AUTO_TRADE_SPREAD_DTE_MAX
            )
            
            if not target_exp:
                return None
            
            # Calculate strikes
            short_strike, long_strike, trigger_level = calculate_smart_strikes(
                symbol, target_exp, 'credit', option_type,
                width=config.AUTO_TRADE_SPREAD_WIDTH
            )
            
            # Prepare order
            order_data = {
                'symbol': symbol,
                'expiration': target_exp,
                'spread_type': 'credit',
                'option_type': option_type,
                'short_strike': str(short_strike),
                'long_strike': str(long_strike),
                'quantity': 1,
                'price': str(config.AUTO_TRADE_MIN_CREDIT),
                'spread_width': config.AUTO_TRADE_SPREAD_WIDTH
            }
            
            logger.info(f"Executing credit spread: {symbol} {short_strike}/{long_strike} {option_type}")
            
            if config.AUTO_TRADE_DRY_RUN:
                result = {'success': True, 'order_id': 'DRY_RUN', 'message': 'Dry run'}
                logger.info(f"DRY RUN: Would place spread: {order_data}")
            else:
                result = place_vertical_spread_order(order_data)
            
            if result and 'error' not in result:
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy': 'credit_spread',
                    'symbol': symbol,
                    'action': 'open',
                    'details': order_data,
                    'ml_confidence': opportunity['confidence'],
                    'result': result,
                    'dry_run': config.AUTO_TRADE_DRY_RUN
                }
                
                risk_manager.log_trade(trade_record)
                self.executed_trades.append(trade_record)
                
                # Save to database
                AutoTradeModel.insert(trade_record)
                
                return trade_record
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing credit spread for {symbol}: {e}")
            return None
    
    def _find_expiration(self, expirations: List[str], min_days: int, max_days: int) -> Optional[str]:
        """
        Find expiration within target DTE range.
        
        Args:
            expirations: List of expiration dates
            min_days: Minimum DTE
            max_days: Maximum DTE
            
        Returns:
            Expiration date or None
        """
        today = date.today()
        
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            
            if min_days <= dte <= max_days:
                return exp_str
        
        return None


# Global trading engine instance
trading_engine = TradingEngine()
