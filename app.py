from flask import Flask, render_template, jsonify, request
from flask_login import login_required, current_user
# UPDATED: Added get_option_expirations to the import list
from services.tradier_service import get_account_summary, get_open_positions, get_yearly_pl, get_historical_data, get_option_expirations, get_current_price, check_and_close_positions
from spread_routes import spreads
from prediction_routes import predictions
from ml_performance_routes import ml_performance
from ml_correction_routes import ml_correction
from auth_routes import auth_routes, init_login_manager
from auto_trader_routes import auto_trader_routes
from config import get_config
from utils.logger import logger
from database import init_db, mongo_db, mongo_client
import os

config = get_config()

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['DEBUG'] = config.DEBUG

logger.info(f"Starting Trading Bot in {os.getenv('FLASK_ENV', 'development')} mode")

# Initialize database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")

# Initialize Flask-Login
init_login_manager(app)
logger.info("Flask-Login initialized with Nostr authentication")

# Register blueprints
app.register_blueprint(auth_routes)
app.register_blueprint(spreads)
app.register_blueprint(predictions)
app.register_blueprint(ml_performance)
app.register_blueprint(ml_correction)
app.register_blueprint(auto_trader_routes)
logger.info("All blueprints registered including ML self-correction and auto trader")

# --- Main Page Routes ---

@app.route('/')
@login_required
def index():
    """
    Renders the main dashboard page with live account data.
    """
    from services.position_display import group_spread_positions
    
    summary_data = get_account_summary()
    positions_data = get_open_positions()
    yearly_pl_data = get_yearly_pl()
    
    # Group spread positions for better display
    if positions_data and isinstance(positions_data, list):
        positions_data = group_spread_positions(positions_data)
    
    return render_template('dashboard.html', 
                           summary=summary_data or {},
                           positions=positions_data or [],
                           yearly_pl=yearly_pl_data or {})

@app.route('/charts')
def charts():
    """
    Renders the chart page and populates initial analysis.
    """
    ticker = request.args.get('ticker', 'TSLA').upper()
    timeframe = request.args.get('timeframe', '6m')
    
    chart_data = get_historical_data(ticker, timeframe)
    levels = chart_data.get('levels') if chart_data else None

    return render_template('charts.html',
                           ticker=ticker,
                           timeframe=timeframe,
                           levels=levels)

# --- API Endpoints ---

@app.route('/api/history/<string:ticker>/<string:timeframe>')
def api_history(ticker, timeframe):
    """
    API endpoint to fetch historical stock data for charts.
    """
    data = get_historical_data(ticker, timeframe)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Could not retrieve data for the given ticker.'}), 404

@app.route('/api/expirations/<string:symbol>')
def api_expirations(symbol):
    """
    API endpoint to fetch option expiration dates for a given symbol.
    """
    expirations = get_option_expirations(symbol)
    if 'error' in expirations:
        return jsonify(expirations), 404
    return jsonify(expirations)

@app.route('/api/price/<string:symbol>')
def api_price(symbol):
    """
    API endpoint to fetch current price for a symbol.
    """
    price = get_current_price(symbol)
    if isinstance(price, dict) and 'error' in price:
        return jsonify(price), 400
    return jsonify({'symbol': symbol.upper(), 'price': price})


# --- Main execution block ---

if __name__ == '__main__':
    app.run(debug=config.DEBUG, port=5000)