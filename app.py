from flask import Flask, render_template, jsonify
# Import the new function and jsonify
from services.tradier_service import get_account_summary, get_open_positions, get_yearly_pl, get_historical_data

# Initialize the Flask application
app = Flask(__name__)

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
    return render_template('charts.html')

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