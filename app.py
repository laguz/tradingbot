from flask import Flask, render_template, jsonify, request
from services.tradier_service import get_account_summary, get_open_positions, get_yearly_pl, get_historical_data
# Import the new blueprint
from spread_routes import spreads
import os # Import os for the secret key

# Initialize the Flask application
app = Flask(__name__)
# A secret key is required for CSRF protection with Flask-WTF
app.config['SECRET_KEY'] = os.urandom(24)

# Register the blueprint
app.register_blueprint(spreads)

# --- Main Page Routes ---

@app.route('/')
def dashboard():
    """
    Renders the main dashboard page with live account data.
    """
    summary_data = get_account_summary()
    positions_data = get_open_positions()
    yearly_pl_data = get_yearly_pl()
    
    return render_template('dashboard.html', 
                           summary=summary_data or {},
                           positions=positions_data or [],
                           yearly_pl=yearly_pl_data or {})

@app.route('/charts')
def charts():
    """
    Renders the empty chart page. The data will be loaded via a separate API call.
    """
    ticker = request.args.get('ticker', 'GOOGL').upper()
    timeframe = request.args.get('timeframe', '3m')
    
    chart_data = get_historical_data(ticker, timeframe)
    levels = chart_data.get('levels') if chart_data else None

    return render_template('charts.html',
                           ticker=ticker,
                           timeframe=timeframe,
                           levels=levels)

# --- API Endpoint for Chart Data ---

@app.route('/api/history/<string:ticker>/<string:timeframe>')
def api_history(ticker, timeframe):
    """
    API endpoint to fetch historical stock data.
    Called by the JavaScript on the charts page.
    """
    data = get_historical_data(ticker, timeframe)
    if data:
        return jsonify(data)
    else:
        # Return an error response if no data is found
        return jsonify({'error': 'Could not retrieve data for the given ticker.'}), 404

# --- Main execution block ---

if __name__ == '__main__':
    app.run(debug=True, port=5000)