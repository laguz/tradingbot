import os
import requests
from dotenv import load_dotenv
from datetime import date, timedelta

# --- API Configuration ---
load_dotenv() 

TRADIER_API_KEY = os.getenv('TRADIER_API_KEY')
TRADIER_ACCOUNT_ID = os.getenv('TRADIER_ACCOUNT_ID')
BASE_URL = 'https://sandbox.tradier.com/v1/'

# --- Standard Headers for API Requests ---
HEADERS = {
    'Authorization': f'Bearer {TRADIER_API_KEY}',
    'Accept': 'application/json'
}

# --- Service Functions ---

def get_account_summary():
    """
    Fetches key account metrics like balance, P/L, and buying power.
    Returns a dictionary with the summary data, or None if an error occurs.
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set in .env file'}

    try:
        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/balances"
        response = requests.get(BASE_URL + endpoint, headers=HEADERS)
        response.raise_for_status() 
        
        balances = response.json().get('balances', {})
        
        summary = {
            'account_balance': balances.get('total_equity'),
            'option_buying_power': balances.get('option_buying_power'),
            'day_pl': balances.get('day_profit_loss'),
        }
        return summary
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch account summary. {e}")
        return None

def get_yearly_pl():
    """
    Fetches the total profit/loss for the current year. (Placeholder)
    """
    return {'yearly_pl': 15890.12}

def get_open_positions():
    """
    Fetches all open positions and enriches them with current market prices.
    Returns a list of positions, or None if an error occurs.
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set in .env file'}

    try:
        positions_endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/positions"
        response = requests.get(BASE_URL + positions_endpoint, headers=HEADERS)
        response.raise_for_status()
        
        positions_data = response.json().get('positions')
        if positions_data is None or 'position' not in positions_data:
            return [] 
            
        positions = positions_data['position']
        if not isinstance(positions, list):
            positions = [positions]

        symbols = [pos['symbol'] for pos in positions]
        if not symbols:
            return []

        quotes_endpoint = "markets/quotes"
        params = {'symbols': ','.join(symbols)}
        quotes_response = requests.get(BASE_URL + quotes_endpoint, headers=HEADERS, params=params)
        quotes_response.raise_for_status()
        quotes = quotes_response.json()['quotes']['quote']
        
        price_map = {quote['symbol']: quote['last'] for quote in quotes}

        enriched_positions = []
        for pos in positions:
            current_price = price_map.get(pos['symbol'], pos['cost_basis'])
            cost_basis_per_share = pos['cost_basis'] / pos['quantity']
            pl_dollars = (current_price - cost_basis_per_share) * pos['quantity']
            pl_percent = (pl_dollars / pos['cost_basis']) * 100 if pos['cost_basis'] != 0 else 0
            
            enriched_positions.append({
                'symbol': pos['symbol'], 'quantity': pos['quantity'], 'entry_price': cost_basis_per_share,
                'current_price': current_price, 'pl_dollars': pl_dollars, 'pl_percent': pl_percent
            })
            
        return enriched_positions
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch open positions. {e}")
        return None

# --- NEW FUNCTION FOR CHART.JS ---
def get_historical_data(ticker, timeframe):
    """
    Fetches historical daily closing prices for a given ticker and timeframe.
    Formats the data for Chart.js.
    """
    if not TRADIER_API_KEY:
        print("ERROR: Tradier API key not set in .env file")
        return None

    # Calculate start date based on timeframe
    end_date = date.today()
    time_deltas = {'1m': 30, '3m': 90, '6m': 180, '1y': 365}
    start_date = end_date - timedelta(days=time_deltas.get(timeframe.lower(), 90))

    endpoint = "markets/history"
    params = {
        'symbol': ticker,
        'interval': 'daily',
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(BASE_URL + endpoint, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        # If the 'history' key is missing, null, or contains no day data, return None
        history_data = data.get('history')
        if not history_data or history_data == 'null' or 'day' not in history_data:
            return None

        # The API returns a single object if one day is found, and a list for multiple days.
        # We ensure it's always a list to handle it consistently.
        day_entries = history_data['day']
        if not isinstance(day_entries, list):
            day_entries = [day_entries]

        if not day_entries:
            return None

        # Format data for Chart.js (labels array and data array)
        chart_data = {
            'labels': [day['date'] for day in day_entries],
            'data': [day['close'] for day in day_entries]
        }
        return chart_data

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch historical data for {ticker}. {e}")
        return None