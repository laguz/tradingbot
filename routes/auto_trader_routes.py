"""
Auto Trader Routes

Web interface and API for controlling the automated trading bot.
"""

from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required
from services.risk_manager import risk_manager
from services.trading_engine import trading_engine
from models.mongodb_models import AutoTradeModel
from jobs.automated_trader import run_automated_trader
from utils.logger import logger
from config import get_config
from datetime import datetime
import threading

config = get_config()
auto_trader_routes = Blueprint('auto_trader', __name__)

# Bot status tracking
bot_status = {
    'running': False,
    'last_run': None,
    'thread': None
}


@auto_trader_routes.route('/auto_trader')
@login_required
def show_auto_trader():
    """Show auto trader control panel."""
    return render_template('auto_trader.html')


@auto_trader_routes.route('/api/autotrader/status')
@login_required
def get_status():
    """Get current auto trader status."""
    try:
        risk_status = risk_manager.get_status()
        recent_trades = AutoTradeModel.find_recent(days=7, limit=20)
        stats = AutoTradeModel.get_stats(days=1)
        
        # Convert MongoDB ObjectId to string for JSON serialization
        for trade in recent_trades:
            if '_id' in trade:
                trade['_id'] = str(trade['_id'])
        
        return jsonify({
            'success': True,
            'bot_running': bot_status['running'],
            'last_run': bot_status['last_run'],
            'risk_status': risk_status,
            'recent_trades': recent_trades,
            'stats': stats,
            'config': {
                'enabled': config.AUTO_TRADE_ENABLED,
                'dry_run': config.AUTO_TRADE_DRY_RUN,
                'symbols': config.AUTO_TRADE_SYMBOLS,
                'max_positions': config.AUTO_TRADE_MAX_POSITIONS,
                'max_daily_loss': config.AUTO_TRADE_MAX_DAILY_LOSS
            }
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


@auto_trader_routes.route('/api/autotrader/run', methods=['POST'])
@login_required
def run_bot():
    """Manually trigger the trading bot (runs in background thread)."""
    try:
        if bot_status['running']:
            return jsonify({'error': 'Bot is already running'}), 400
        
        # Run in background thread
        def run_in_thread():
            try:
                bot_status['running'] = True
                from datetime import datetime
                run_automated_trader()
                bot_status['last_run'] = datetime.now().isoformat()
            finally:
                bot_status['running'] = False
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        bot_status['thread'] = thread
        
        return jsonify({
            'success': True,
            'message': 'Bot started in background'
        })
        
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        bot_status['running'] = False
        return jsonify({'error': str(e)}), 500


@auto_trader_routes.route('/api/autotrader/emergency_stop', methods=['POST'])
@login_required
def emergency_stop():
    """Activate emergency stop."""
    try:
        # Handle both JSON and form requests
        if request.is_json:
            reason = request.json.get('reason', 'Manual emergency stop')
        else:
            reason = 'Manual emergency stop via web UI'
        
        risk_manager.activate_emergency_stop(reason)
        
        logger.critical(f"Emergency stop activated by user: {reason}")
        
        return jsonify({
            'success': True,
            'message': 'Emergency stop activated'
        })
    except Exception as e:
        logger.error(f"Error activating emergency stop: {e}")
        return jsonify({'error': str(e)}), 500


@auto_trader_routes.route('/api/autotrader/resume', methods=['POST'])
@login_required
def resume_trading():
    """Deactivate emergency stop."""
    try:
        risk_manager.deactivate_emergency_stop()
        
        return jsonify({
            'success': True,
            'message': 'Emergency stop deactivated - trading resumed'
        })
    except Exception as e:
        logger.error(f"Error deactivating emergency stop: {e}")
        return jsonify({'error': str(e)}), 500
