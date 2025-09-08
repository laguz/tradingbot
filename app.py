# laguz/tradingbot/tradingbot-bc1f680a95b47c592f111a193bf8d1c99a0bd96d/app.py
from flask import Flask, render_template, jsonify, request
from services.tradier_service import get_account_summary, get_open_positions, get_yearly_pl, get_historical_data

app = Flask(__name__)

# --- Main Page Routes ---

@app.route('/')
def dashboard():
    summary_data = get_account_summary()
    positions_data = get_open_positions()
    yearly_pl_data = get_yearly_pl()
    
    return render_template('dashboard.html', 
                           summary=summary_data or {},
                           positions=positions_data or [],
                           yearly_pl=yearly_pl_data or {})

@app.route('/charts', methods=['GET'])
def charts():
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
    data = get_historical_data(ticker, timeframe)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Could not retrieve data for the given ticker.'}), 404

# --- Main execution block ---

if __name__ == '__main__':
    app.run(debug=True, port=5000)