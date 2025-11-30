import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import date, timedelta, datetime

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

# --- Helper Function ---
def custom_round(price):
    if price < 100:
        return round(price)
    else:
        return round(price / 5) * 5

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

def find_support_resistance(data, window=20):
    all_support, all_resistance = [], []
    if data.empty: return [], []
    for i in range(window, len(data) - window):
        window_slice = data['Close'].iloc[i-window:i+window+1]
        current_price = data['Close'].iloc[i]
        price = current_price.item() if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price
        if np.isclose(price, window_slice.min()): all_support.append(price)
        if np.isclose(price, window_slice.max()): all_resistance.append(price)
    unique_supports = sorted(list(set(all_support)))
    unique_resistances = sorted(list(set(all_resistance)))
    plotted_support = []
    if unique_supports:
        last_support = unique_supports[0]
        plotted_support.append(last_support)
        for level in unique_supports:
            required_diff = 1 if last_support < 100 else 2
            if abs(level - last_support) >= required_diff:
                plotted_support.append(level)
                last_support = level
    plotted_resistance = []
    if unique_resistances:
        last_resistance = unique_resistances[0]
        plotted_resistance.append(last_resistance)
        for level in unique_resistances:
            required_diff = 1 if last_resistance < 100 else 2
            if abs(level - last_resistance) >= required_diff:
                plotted_resistance.append(level)
                last_resistance = level
    return plotted_support, plotted_resistance

def get_historical_data(ticker, timeframe):
    """
    Fetches historical daily closing prices for a given ticker and timeframe.
    Formats the data for Chart.js.
    """
    if not TRADIER_API_KEY:
        print("ERROR: Tradier API key not set in .env file")
        return None

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

        history_data = data.get('history')
        if not history_data or history_data == 'null' or 'day' not in history_data:
            return None

        day_entries = history_data['day']
        if not isinstance(day_entries, list):
            day_entries = [day_entries]

        if not day_entries:
            return None

        df = pd.DataFrame(day_entries)
        df.rename(columns={'close': 'Close', 'date': 'Date'}, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'])

        support_levels, resistance_levels = find_support_resistance(df)

        chart_data = {
            'labels': [d['date'].split('-')[-1] for d in day_entries],
            'data': [d['close'] for d in day_entries],
            'support': support_levels,
            'resistance': resistance_levels,
            'levels': {
                'support': list(dict.fromkeys([custom_round(s) for s in support_levels])),
                'resistance': list(dict.fromkeys([custom_round(r) for r in resistance_levels]))
            }
        }
        return chart_data

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch historical data for {ticker}. {e}")
        return None

def get_option_expirations(symbol):
    """
    Fetches option expiration dates for a given stock symbol.
    """
    if not TRADIER_API_KEY:
        return {'error': 'API Key not set.'}

    params = {'symbol': symbol.upper()}
    endpoint = "markets/options/expirations"
    
    try:
        response = requests.get(BASE_URL + endpoint, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        expirations = data.get('expirations', {}).get('date', [])
        return expirations

    except requests.exceptions.RequestException as e:
        error_details = e.response.json() if e.response else str(e)
        print(f"ERROR: Could not fetch expirations for {symbol}. {error_details}")
        return {'error': f"Failed to fetch expirations: {error_details}"}
    except Exception as e:
        print(f"An unexpected error occurred while fetching expirations: {e}")
        return {'error': f"An unexpected error occurred: {str(e)}"}

def _generate_occ_symbol(symbol, expiration, option_type, strike):
    """
    Generates an OCC option symbol.
    Example: GOOGL250919C00180000
    """
    exp_date = datetime.strptime(expiration, '%Y-%m-%d').strftime('%y%m%d')
    opt_type = 'C' if option_type.lower() == 'call' else 'P'
    strike_price = str(int(float(strike) * 1000)).zfill(8)
    
    # --- CORRECTED LOGIC ---
    # Removed the space padding, which was likely invalidating the symbol.
    formatted_symbol = symbol.upper()
    # --- END OF CORRECTION ---
    
    return f"{formatted_symbol}{exp_date}{opt_type}{strike_price}"

def place_stock_order(form_data):
    """
    Places a market or limit order to buy or sell a stock.
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set.'}

    try:
        order_payload = {
            'class': 'equity',
            'symbol': form_data['symbol'].upper(),
            'side': form_data['side'],
            'quantity': str(form_data['quantity']),
            'type': form_data['order_type'],
            'duration': 'day',
        }
        if form_data['order_type'] == 'limit':
            order_payload['price'] = form_data['price']

        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if e.response is not None:
            try:
                error_details = e.response.json()
                if 'errors' in error_details and 'error' in error_details['errors']:
                    error_message = ". ".join(error_details['errors']['error'])
                else:
                    error_message = str(error_details)
            except ValueError:
                error_message = e.response.text
        print(f"ERROR: Could not place stock order. {error_message}")
        return {'error': f"Failed to place order: {error_message}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {'error': f"An unexpected error occurred: {str(e)}"}


def place_single_option_order(form_data):
    """
    Constructs and places a single-leg option order (market or limit).
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set.'}

    try:
        occ_symbol = _generate_occ_symbol(
            form_data['symbol'],
            form_data['expiration'],
            form_data['option_type'],
            form_data['strike']
        )

        order_payload = {
            'class': 'option',
            'symbol': form_data['symbol'].upper(),
            'option_symbol': occ_symbol,
            'side': form_data['side'],
            'quantity': form_data['quantity'],
            'type': form_data['order_type'],
            'duration': 'day',
        }
        if form_data['order_type'] == 'limit':
            order_payload['price'] = form_data['price']

        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if e.response is not None:
            try:
                error_details = e.response.json()
                if 'errors' in error_details and 'error' in error_details['errors']:
                    error_message = ". ".join(error_details['errors']['error'])
                else:
                    error_message = str(error_details)
            except ValueError:
                error_message = e.response.text
        print(f"ERROR: Could not place single option order. {error_message}")
        return {'error': f"Failed to place order: {error_message}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {'error': f"An unexpected error occurred: {str(e)}"}


def place_vertical_spread_order(form_data):
    """
    Constructs and places a multi-leg vertical spread order.
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set.'}

    try:
        long_occ = _generate_occ_symbol(
            form_data['symbol'], form_data['expiration'],
            form_data['option_type'], form_data['long_strike']
        )
        short_occ = _generate_occ_symbol(
            form_data['symbol'], form_data['expiration'],
            form_data['option_type'], form_data['short_strike']
        )
            
        order_payload = {
            'class': 'multileg',
            'symbol': form_data['symbol'].upper(),
            'type': form_data['spread_type'],
            'price': form_data['price'],
            'duration': 'day',
            'option_symbol[0]': long_occ,
            'side[0]': 'buy_to_open',
            'quantity[0]': form_data['quantity'],
            'option_symbol[1]': short_occ,
            'side[1]': 'sell_to_open',
            'quantity[1]': form_data['quantity'],
        }

        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=HEADERS)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if e.response is not None:
            try:
                error_details = e.response.json()
                if 'errors' in error_details and 'error' in error_details['errors']:
                    error_message = ". ".join(error_details['errors']['error'])
                else:
                    error_message = str(error_details)
            except ValueError:
                error_message = e.response.text

        print(f"ERROR: Could not place order. {error_message}")
        return {'error': f"Failed to place order: {error_message}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {'error': f"An unexpected error occurred: {str(e)}"}


def place_iron_condor_order(form_data):
    """
    Constructs and places a 4-leg iron condor order.
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set.'}

    try:
        spread_type = form_data['spread_type']

        if spread_type == 'credit':
            put_buy_side, put_sell_side = 'buy_to_open', 'sell_to_open'
            call_sell_side, call_buy_side = 'sell_to_open', 'buy_to_open'
        else:
            put_buy_side, put_sell_side = 'sell_to_open', 'buy_to_open'
            call_sell_side, call_buy_side = 'buy_to_open', 'sell_to_open'

        long_put_occ = _generate_occ_symbol(form_data['symbol'], form_data['expiration'], 'put', form_data['long_put_strike'])
        short_put_occ = _generate_occ_symbol(form_data['symbol'], form_data['expiration'], 'put', form_data['short_put_strike'])
        short_call_occ = _generate_occ_symbol(form_data['symbol'], form_data['expiration'], 'call', form_data['short_call_strike'])
        long_call_occ = _generate_occ_symbol(form_data['symbol'], form_data['expiration'], 'call', form_data['long_call_strike'])

        order_payload = {
            'class': 'multileg',
            'symbol': form_data['symbol'].upper(),
            'type': form_data['spread_type'],
            'price': form_data['price'],
            'duration': 'day',
            'option_symbol[0]': long_put_occ, 'side[0]': put_buy_side, 'quantity[0]': form_data['quantity'],
            'option_symbol[1]': short_put_occ, 'side[1]': put_sell_side, 'quantity[1]': form_data['quantity'],
            'option_symbol[2]': short_call_occ, 'side[2]': call_sell_side, 'quantity[2]': form_data['quantity'],
            'option_symbol[3]': long_call_occ, 'side[3]': call_buy_side, 'quantity[3]': form_data['quantity'],
        }

        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if e.response is not None:
            try:
                error_details = e.response.json()
                if 'errors' in error_details and 'error' in error_details['errors']:
                    error_message = ". ".join(error_details['errors']['error'])
                else:
                    error_message = str(error_details)
            except ValueError:
                error_message = e.response.text
        print(f"ERROR: Could not place iron condor order. {error_message}")
        return {'error': f"Failed to place order: {error_message}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {'error': f"An unexpected error occurred: {str(e)}"}

def get_option_chain(symbol, expiration):
    """
    Fetches the option chain for a specific expiration.
    """
    if not TRADIER_API_KEY:
        return []

    params = {'symbol': symbol.upper(), 'expiration': expiration, 'greeks': 'false'}
    endpoint = "markets/options/chains"
    
    try:
        response = requests.get(BASE_URL + endpoint, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        options = data.get('options', {}).get('option', [])
        if not isinstance(options, list):
            options = [options]
            
        return options

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch option chain for {symbol}. {e}")
        return []

def calculate_smart_strikes(symbol, expiration, spread_type, option_type, width):
    """
    Calculates optimal strikes based on Support (Put Credit) or Resistance (Call Credit).
    """
    # 1. Get Historical Data for Support/Resistance
    chart_data = get_historical_data(symbol, '6m') # Use 6 months for reliable levels
    if not chart_data:
        raise ValueError("Could not fetch historical data for analysis.")
        
    support_levels = chart_data.get('support', [])
    resistance_levels = chart_data.get('resistance', [])
    current_price = chart_data['data'][-1]
    
    if not support_levels and not resistance_levels:
         raise ValueError("No support or resistance levels found.")

    # 2. Determine Target Price based on Strategy
    target_price = None
    trigger_level = None # The level (support/resistance) used for decision
    
    if spread_type == 'credit' and option_type == 'put':
        # Bullish: Sell Put AT or BELOW Support
        # Find the closest support level below current price
        valid_supports = [s for s in support_levels if s < current_price]
        
        # Safety Buffer: Ensure we are at least 1% OTM
        safety_threshold = current_price * 0.99
        
        if valid_supports:
            closest_support = valid_supports[-1]
            # Use the lower of the two (further OTM) to ensure safety
            target_price = min(closest_support, safety_threshold)
            trigger_level = closest_support
        else:
            target_price = current_price * 0.95 # Fallback: 5% OTM
            trigger_level = "Fallback (5% OTM)"
            
    elif spread_type == 'credit' and option_type == 'call':
        # Bearish: Sell Call AT or ABOVE Resistance
        # Find the closest resistance level above current price
        valid_resistances = [r for r in resistance_levels if r > current_price]
        
        # Safety Buffer: Ensure we are at least 1% OTM
        safety_threshold = current_price * 1.01
        
        if valid_resistances:
            closest_resistance = valid_resistances[0]
            # Use the higher of the two (further OTM) to ensure safety
            target_price = max(closest_resistance, safety_threshold)
            trigger_level = closest_resistance
        else:
            target_price = current_price * 1.05 # Fallback: 5% OTM
            trigger_level = "Fallback (5% OTM)"
    else:
        raise ValueError("Auto-selection currently only supports Credit Spreads (Put/Call).")

    # 3. Get Option Chain to find real strikes
    chain = get_option_chain(symbol, expiration)
    if not chain:
        raise ValueError("Could not fetch option chain.")
        
    # Filter chain for correct type
    chain = [opt for opt in chain if opt['option_type'] == option_type]
    strikes = sorted(list(set([opt['strike'] for opt in chain])))
    
    if not strikes:
        raise ValueError("No strikes found for this expiration.")

    # 4. Select Short Strike (Closest to Target Price)
    # We want to sell the option closest to our target level
    short_strike = min(strikes, key=lambda x: abs(x - target_price))
    
    # 5. Calculate Long Strike
    if option_type == 'put':
        # Put Credit Spread: Long Strike is LOWER than Short Strike
        long_strike_target = short_strike - width
    else:
        # Call Credit Spread: Long Strike is HIGHER than Short Strike
        long_strike_target = short_strike + width
        
    # Find closest real strike to the calculated long target
    long_strike = min(strikes, key=lambda x: abs(x - long_strike_target))
    
    # Validation: Ensure spread width is maintained roughly (don't collapse the spread)
    if option_type == 'put' and long_strike >= short_strike:
         # Try to find a lower strike
         lower_strikes = [s for s in strikes if s < short_strike]
         if lower_strikes: long_strike = lower_strikes[-1] # Highest of the lower strikes
         
    if option_type == 'call' and long_strike <= short_strike:
         # Try to find a higher strike
         higher_strikes = [s for s in strikes if s > short_strike]
         if higher_strikes: long_strike = higher_strikes[0] # Lowest of the higher strikes

    return short_strike, long_strike, trigger_level