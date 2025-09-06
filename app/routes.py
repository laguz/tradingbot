import requests
from flask import render_template, request, jsonify, current_app
from app import app
from datetime import datetime, timedelta

def process_stock_data(data, timeframe_days):
    """Processes raw API data into a chart-friendly format."""
    # The key for time series data in the Alpha Vantage response
    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        return None  # Or handle the error appropriately

    time_series = data[time_series_key]
    
    # Get the date 'timeframe_days' ago to filter the data
    cutoff_date = datetime.now() - timedelta(days=timeframe_days)
    
    dates = []
    prices = []
    
    # Sort dates to ensure they are in chronological order for the chart
    sorted_dates = sorted(time_series.keys())

    for date_str in sorted_dates:
        current_date = datetime.strptime(date_str, '%Y-%m-%d')
        if current_date >= cutoff_date:
            dates.append(date_str)
            # We want the closing price, which is key '4. close'
            prices.append(float(time_series[date_str]['4. close']))
            
    # Also get the company symbol from the metadata
    company_symbol = data.get('Meta Data', {}).get('2. Symbol', '')
            
    return {'labels': dates, 'data': prices, 'company': company_symbol}

@app.route('/')
def index():
    """Renders the main page (the View)."""
    return render_template('index.html')

@app.route('/get_stock_data')
def get_stock_data():
    """API endpoint to fetch and process stock data from Alpha Vantage."""
    ticker = request.args.get('ticker')
    # Map front-end timeframe labels to the number of days
    timeframe_map = {'1M': 30, '3M': 90, '6M': 182, '1Y': 365}
    timeframe_str = request.args.get('timeframe', '1M') # Default to 1M if not provided
    timeframe_days = timeframe_map.get(timeframe_str, 30)

    if not ticker:
        return jsonify({'error': 'Stock ticker is required'}), 400

    try:
        api_key = current_app.config['ALPHA_VANTAGE_API_KEY']
        # Using TIME_SERIES_DAILY and outputsize=full to get enough data for up to a year
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}'
        
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (like 404 or 500)
        
        raw_data = response.json()

        # Alpha Vantage returns an error message in the JSON itself for bad tickers
        if "Error Message" in raw_data:
            return jsonify({'error': 'Invalid stock ticker or API call.'}), 400

        chart_data = process_stock_data(raw_data, timeframe_days)
        
        if chart_data is None:
            return jsonify({'error': 'Could not process stock data from API response.'}), 500

        return jsonify(chart_data)

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: Failed to retrieve data from Alpha Vantage.'}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500