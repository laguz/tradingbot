import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import date, timedelta, datetime
from services.vault_service import vault_service

# --- API Configuration ---
load_dotenv() 

BASE_URL = 'https://sandbox.tradier.com/v1/'

# --- Helper Function for Dynamic Headers ---
def get_headers():
    secrets = vault_service.get_tradier_secrets()
    if not secrets:
        # Fallback to env vars for backward compatibility or if Vault fails
        api_key = os.getenv('TRADIER_API_KEY')
        if not api_key:
            return None
    else:
        api_key = secrets.get('api_key')

    return {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

def get_account_id():
    secrets = vault_service.get_tradier_secrets()
    if secrets:
        return secrets.get('account_id')
    return os.getenv('TRADIER_ACCOUNT_ID')

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
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

    try:
        endpoint = f"accounts/{get_account_id()}/balances"
        response = requests.get(BASE_URL + endpoint, headers=get_headers())
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
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

    try:
        positions_endpoint = f"accounts/{get_account_id()}/positions"
        response = requests.get(BASE_URL + positions_endpoint, headers=get_headers())
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
        quotes_response = requests.get(BASE_URL + quotes_endpoint, headers=get_headers(), params=params)
        quotes_response.raise_for_status()
        quotes = quotes_response.json()['quotes']['quote']
        
        # Map symbol to full quote to get last and ask
        quote_map = {quote['symbol']: quote for quote in quotes}

        enriched_positions = []
        today = date.today()
        
        for pos in positions:
            quote = quote_map.get(pos['symbol'], {})
            cost_basis_per_share = pos['cost_basis'] / pos['quantity'] if pos['quantity'] != 0 else 0
            
            current_price = quote.get('last')
            if current_price is None:
                current_price = cost_basis_per_share
                
            ask_price = quote.get('ask')
            if ask_price is None:
                ask_price = current_price

            pl_dollars = (current_price - cost_basis_per_share) * pos['quantity']
            pl_percent = (pl_dollars / pos['cost_basis']) * 100 if pos['cost_basis'] != 0 else 0
            
            # Calculate DTE from OCC Symbol
            # Format: SYMBOLYYMMDD[C/P]STRIKE
            # e.g. TSLA251219P00410000 -> 251219 -> 2025-12-19
            dte = None
            expiration_str = None
            strike_price = None
            option_type = None
            underlying = None
            
            try:
                import re
                # Parse OCC: Ticker (chars), Date (6 digits), Type (C/P), Strike (8 digits)
                # Example: TSLA251219P00410000
                match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', pos['symbol'])
                if match:
                    underlying = match.group(1)
                    exp_part = match.group(2)
                    opt_type = match.group(3)
                    strike_part = match.group(4)
                    
                    exp_date = datetime.strptime(exp_part, '%y%m%d').date()
                    dte = (exp_date - today).days
                    expiration_str = exp_date.strftime('%Y-%m-%d')
                    
                    strike_price = float(strike_part) / 1000.0
                    option_type = 'call' if opt_type == 'C' else 'put'
            except Exception as e:
                print(f"Error parsing details for {pos['symbol']}: {e}")

            enriched_positions.append({
                'symbol': pos['symbol'], 
                'quantity': pos['quantity'], 
                'entry_price': cost_basis_per_share,
                'current_price': current_price, 
                'ask_price': ask_price,
                'pl_dollars': pl_dollars, 
                'pl_percent': pl_percent,
                'dte': dte,
                'expiration': expiration_str,
                'strike': strike_price,
                'option_type': option_type,
                'underlying': underlying,
                'id': pos['id']
            })
            
        return enriched_positions
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch open positions. {e}")
        return None

def get_current_price(symbol):
    """
    Fetches the current market price for a symbol.
    """
    if not get_headers():
        return {'error': 'API Key not set.'}
        
    try:
        endpoint = "markets/quotes"
        params = {'symbols': symbol.upper()}
        response = requests.get(BASE_URL + endpoint, headers=get_headers(), params=params)
        response.raise_for_status()
        
        quotes = response.json().get('quotes', {}).get('quote')
        if not quotes:
            return {'error': 'No quote found'}
            
        # Handle single quote vs list (though API usually returns list or dict depending on count, for single symbol it might be dict)
        if isinstance(quotes, list):
            quote = quotes[0]
        else:
            quote = quotes
            
        return quote.get('last')
        
    except Exception as e:
        print(f"ERROR: Could not fetch price for {symbol}. {e}")
        return {'error': str(e)}

def find_support_resistance(data, window=5, tolerance=0.015):
    """
    Identifies support and resistance levels using volume-weighted clustering.
    """
    if data.empty: return [], []
    
    supports = []
    resistances = []
    
    # 1. Identify Pivot Points
    for i in range(window, len(data) - window):
        # Support Pivot (Local Low)
        if data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window+1].min():
            supports.append({'price': data['Low'].iloc[i], 'volume': data['Volume'].iloc[i]})
            
        # Resistance Pivot (Local High)
        if data['High'].iloc[i] == data['High'].iloc[i-window:i+window+1].max():
            resistances.append({'price': data['High'].iloc[i], 'volume': data['Volume'].iloc[i]})

    def cluster_levels(levels):
        if not levels: return []
        
        levels.sort(key=lambda x: x['price'])
        clusters = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            price = levels[i]['price']
            prev_price = current_cluster[-1]['price']
            
            # Check if within tolerance
            if price <= prev_price * (1 + tolerance):
                current_cluster.append(levels[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [levels[i]]
        clusters.append(current_cluster)
        
        # Score Clusters
        scored_clusters = []
        for cluster in clusters:
            avg_price = sum(l['price'] for l in cluster) / len(cluster)
            total_volume = sum(l['volume'] for l in cluster)
            touch_count = len(cluster)
            
            # Score = Volume * Touches (Simple heuristic)
            score = total_volume * touch_count
            scored_clusters.append({'price': avg_price, 'score': score})
            
        # Sort by score and take top 5 significant levels
        scored_clusters.sort(key=lambda x: x['score'], reverse=True)
        return sorted([c['price'] for c in scored_clusters[:5]])

    final_supports = cluster_levels(supports)
    final_resistances = cluster_levels(resistances)
    
    return final_supports, final_resistances

def get_historical_data(ticker, timeframe):
    """
    Fetches historical daily closing prices for a given ticker and timeframe.
    Formats the data for Chart.js.
    """
    if not get_headers():
        print("ERROR: Tradier API key not set in Vault or .env")
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
        response = requests.get(BASE_URL + endpoint, headers=get_headers(), params=params)
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
        df.rename(columns={'close': 'Close', 'date': 'Date', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Volume'] = pd.to_numeric(df['Volume'])

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

def get_raw_historical_data(ticker, timeframe):
    """
    Fetches raw historical daily data (OHLCV) for a given ticker.
    Returns a Pandas DataFrame with datetime index and numeric columns.
    """
    if not get_headers():
        print("ERROR: Tradier API key not set in Vault or .env")
        return None

    end_date = date.today()
    time_deltas = {'1m': 30, '3m': 90, '6m': 180, '1y': 365, '2y': 730, '5y': 1825}
    start_date = end_date - timedelta(days=time_deltas.get(timeframe.lower(), 365))

    endpoint = "markets/history"
    params = {
        'symbol': ticker,
        'interval': 'daily',
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(BASE_URL + endpoint, headers=get_headers(), params=params)
        response.raise_for_status()
        data = response.json()

        history_data = data.get('history')
        if not history_data or history_data == 'null' or 'day' not in history_data:
            return pd.DataFrame() # Return empty DF instead of None for consistency

        day_entries = history_data['day']
        if not isinstance(day_entries, list):
            day_entries = [day_entries]

        if not day_entries:
            return pd.DataFrame()

        df = pd.DataFrame(day_entries)
        df.rename(columns={'close': 'Close', 'date': 'Date', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
        
        # Convert types
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
            
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
            
        return df

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch raw historical data for {ticker}. {e}")
        return pd.DataFrame()

def get_option_expirations(symbol):
    """
    Fetches option expiration dates for a given stock symbol.
    """
    if not get_headers():
        return {'error': 'API Key not set.'}

    params = {'symbol': symbol.upper()}
    endpoint = "markets/options/expirations"
    
    try:
        response = requests.get(BASE_URL + endpoint, params=params, headers=get_headers())
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
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

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

        endpoint = f"accounts/{get_account_id()}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=get_headers())
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
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

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

        endpoint = f"accounts/{get_account_id()}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=get_headers())
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
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

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
            'type': form_data['spread_type'],
            'duration': 'day',
            'option_symbol[0]': long_occ,
            'side[0]': 'buy_to_open',
            'quantity[0]': form_data['quantity'],
            'option_symbol[1]': short_occ,
            'side[1]': 'sell_to_open',
            'quantity[1]': form_data['quantity'],
        }
        
        if form_data.get('price'):
            order_payload['price'] = form_data['price']

        endpoint = f"accounts/{get_account_id()}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=get_headers())
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
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

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

        endpoint = f"accounts/{get_account_id()}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=get_headers())
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
    if not get_headers():
        return []

    params = {'symbol': symbol.upper(), 'expiration': expiration, 'greeks': 'false'}
    endpoint = "markets/options/chains"
    
    try:
        response = requests.get(BASE_URL + endpoint, params=params, headers=get_headers())
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

def close_position(position_id, symbol, quantity, limit_price=None):
    """
    Closes a specific position (single leg).
    Optionally uses a limit price.
    """
    if not get_account_id() or not get_headers():
        return {'error': 'API Key or Account ID not set in Vault or .env'}

    try:
        # Determine side: if quantity > 0 (long), we sell to close. If < 0 (short), we buy to close.
        side = 'sell_to_close' if quantity > 0 else 'buy_to_close'
        abs_quantity = abs(quantity)
        
        # Extract underlying ticker from OCC symbol
        import re
        match = re.match(r'([A-Z]+)\d', symbol)
        underlying = match.group(1) if match else symbol # Fallback
        
        order_payload = {
            'class': 'option',
            'symbol': underlying,
            'option_symbol': symbol,
            'side': side,
            'quantity': str(abs_quantity),
            'type': 'market', 
            'duration': 'day',
        }
        
        if limit_price:
            order_payload['type'] = 'limit'
            order_payload['price'] = str(limit_price)

        endpoint = f"accounts/{get_account_id()}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=get_headers())
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(f"ERROR: Could not close position {symbol}. {e}")
        return {'error': str(e)}

def manage_position_state(positions, underlying_prices):
    """
    Updates the persistence state for ITM positions.
    """
    import json
    STATE_FILE = 'position_state.json'
    
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
        
    today_str = date.today().strftime('%Y-%m-%d')
    updated_state = {}
    
    for pos in positions:
        symbol = pos['symbol']
        underlying = pos.get('underlying')
        strike = pos.get('strike')
        option_type = pos.get('option_type')
        
        if not underlying or not strike or not option_type:
            continue
            
        u_price = underlying_prices.get(underlying)
        if u_price is None: continue
        
        is_itm = False
        if option_type == 'put' and u_price < strike:
            is_itm = True
        elif option_type == 'call' and u_price > strike:
            is_itm = True
            
        if is_itm:
            # Get existing state for this position
            pos_state = state.get(symbol, {'itm_consecutive_days': 0, 'last_check_date': ''})
            
            # If checked yesterday (or before), increment. If checked today, keep same.
            # Actually, we want to increment only if it's a NEW day.
            last_check = pos_state.get('last_check_date')
            days = pos_state.get('itm_consecutive_days', 0)
            
            if last_check != today_str:
                days += 1
                
            updated_state[symbol] = {
                'itm_consecutive_days': days,
                'last_check_date': today_str
            }
        else:
            # Reset if not ITM? Or keep history? Usually reset if OTM.
            # If OTM today, we don't track it or reset to 0.
            pass
            
    # Save state
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(updated_state, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {e}")
        
    return updated_state

def check_and_close_positions():
    """
    Checks all open positions and closes them if they meet the criteria:
    - DTE >= 14 and Profit >= 60%
    - DTE < 14 and Profit >= 80%
    - ITM and DTE < 9: Close at Ask + 0.2
    - ITM for 2 consecutive days and Time >= 12:00 PM EST: Close at Ask + 0.2
    """
    positions = get_open_positions()
    if positions is None:
        return {'error': 'Could not fetch positions'}
        
    # 1. Fetch Underlying Prices for ITM check
    unique_underlyings = list(set([p['underlying'] for p in positions if p.get('underlying')]))
    underlying_prices = {}
    if unique_underlyings:
        try:
            endpoint = "markets/quotes"
            params = {'symbols': ','.join(unique_underlyings)}
            response = requests.get(BASE_URL + endpoint, headers=HEADERS, params=params)
            if response.status_code == 200:
                quotes = response.json().get('quotes', {}).get('quote', [])
                if isinstance(quotes, dict): quotes = [quotes]
                for q in quotes:
                    underlying_prices[q['symbol']] = q['last']
        except Exception as e:
            print(f"Error fetching underlying prices: {e}")

    # 2. Update State
    state = manage_position_state(positions, underlying_prices)
    
    # 3. Check Time (EST)
    from datetime import timezone
    now_utc = datetime.now(timezone.utc)
    # Approximate 12PM EST as 17:00 UTC for safety (or check system time if configured)
    is_past_noon_est = now_utc.hour >= 17 

    results = {
        'closed': [],
        'errors': [],
        'checked': 0
    }
    
    # Identify Spreads: Create a set of keys for all LONG positions
    # Key: (underlying, expiration, option_type)
    long_legs = set()
    for p in positions:
        if p['quantity'] > 0 and p.get('underlying') and p.get('expiration') and p.get('option_type'):
            long_legs.add((p['underlying'], p['expiration'], p['option_type']))

    for pos in positions:
        dte = pos.get('dte')
        pl_percent = pos.get('pl_percent')
        symbol = pos['symbol']
        ask_price = pos.get('ask_price')
        quantity = pos['quantity']
        
        if dte is None: continue 
        
        # FILTER: Only process SHORT positions that are part of a SPREAD
        # 1. Must be Short (quantity < 0)
        # 2. Must have a matching Long Leg (same underlying, exp, type)
        if quantity >= 0:
            continue # Skip Long positions
            
        # Check for matching Long Leg
        pos_key = (pos.get('underlying'), pos.get('expiration'), pos.get('option_type'))
        if pos_key not in long_legs:
            continue # Skip Naked/Wheel positions
            
        results['checked'] += 1
        
        should_close = False
        reason = ""
        limit_price = None
        
        # Check ITM State
        pos_state = state.get(symbol, {})
        itm_days = pos_state.get('itm_consecutive_days', 0)
        is_itm = itm_days > 0
        
        # Rule 1: ITM and DTE < 9
        if is_itm and dte < 9:
            should_close = True
            reason = f"ITM and DTE {dte} < 9"
            limit_price = round(ask_price + 0.02, 2) if ask_price else None
            
        # Rule 2: ITM for 2 consecutive days and Time >= 12:00 PM EST (next day implies days >= 2)
        elif is_itm and itm_days >= 2 and is_past_noon_est:
            should_close = True
            reason = f"ITM for {itm_days} days and Time >= 12:00 PM EST"
            limit_price = round(ask_price + 0.02, 2) if ask_price else None

        # Rule 3: Profit Taking (Existing)
        elif dte >= 14 and pl_percent >= 60:
            should_close = True
            reason = f"DTE {dte} >= 14 and Profit {pl_percent:.1f}% >= 60%"
            
        # Rule 4: Profit Taking (Existing)
        elif dte < 14 and pl_percent >= 80:
            should_close = True
            reason = f"DTE {dte} < 14 and Profit {pl_percent:.1f}% >= 80%"
            
        if should_close:
            print(f"Auto-Closing {pos['symbol']}: {reason} (Limit: {limit_price})")
            res = close_position(pos['id'], pos['symbol'], pos['quantity'], limit_price=limit_price)
            
            if 'error' in res:
                results['errors'].append({'symbol': pos['symbol'], 'error': res['error']})
            else:
                results['closed'].append({
                    'symbol': pos['symbol'], 
                    'reason': reason, 
                    'order_id': res.get('order', {}).get('id')
                })
                
    return results

def place_multileg_order(legs, symbol, type, duration='day', price=None):
    """
    Generic function to place a multileg order.
    legs: list of dicts with 'option_symbol', 'side', 'quantity'
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set.'}

    try:
        order_payload = {
            'class': 'multileg',
            'symbol': symbol.upper(),
            'type': type, # 'market', 'debit', 'credit', 'even'
            'duration': duration,
        }
        
        for i, leg in enumerate(legs):
            order_payload[f'option_symbol[{i}]'] = leg['option_symbol']
            order_payload[f'side[{i}]'] = leg['side']
            order_payload[f'quantity[{i}]'] = leg['quantity']
            
        if price:
            order_payload['price'] = price

        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=HEADERS)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        print(f"ERROR: Could not place multileg order. {e}")
        return {'error': str(e)}

def check_and_roll_positions():
    """
    Checks for Wheel positions (single-leg) that need rolling.
    Criteria:
    - Single Leg
    - DTE < 9
    - ITM OR Option Price < 1.01
    
    Action:
    - BTC current
    - STO new (Exp > 42 days, Strike +/- 1)
    """
    positions = get_open_positions()
    if positions is None:
        return {'error': 'Could not fetch positions'}
        
    # Identify Spreads to EXCLUDE them
    long_legs = set()
    for p in positions:
        if p['quantity'] > 0 and p.get('underlying') and p.get('expiration') and p.get('option_type'):
            long_legs.add((p['underlying'], p['expiration'], p['option_type']))
            
    results = {
        'rolled': [],
        'errors': [],
        'checked': 0
    }
    
    for pos in positions:
        # Filter: Must be Short (Wheel is selling options)
        if pos['quantity'] >= 0: continue
        
        # Filter: Must NOT be part of a spread (no matching long leg)
        pos_key = (pos.get('underlying'), pos.get('expiration'), pos.get('option_type'))
        if pos_key in long_legs: continue
        
        results['checked'] += 1
        
        dte = pos.get('dte')
        symbol = pos['symbol']
        current_price = pos.get('current_price', 0)
        strike = pos.get('strike')
        option_type = pos.get('option_type')
        underlying = pos.get('underlying')
        
        if dte is None or strike is None: continue
        
        # Check Roll Criteria
        # 1. DTE < 9
        if dte >= 9: continue
        
        # 2. ITM or Price < 1.01
        is_itm = False
        # Fetch underlying price if not in pos (it should be from get_open_positions)
        # We need the underlying price for ITM check. 
        # get_open_positions enriches with 'current_price' of the OPTION, but we need UNDERLYING price for ITM.
        # Wait, get_open_positions does NOT return underlying price in the dict, only 'underlying' symbol.
        # We need to fetch it.
        
        # Optimization: We could fetch all underlying prices at once like in check_and_close.
        # For now, let's just fetch for the candidate to be safe/simple, or reuse the logic.
        # Actually, let's fetch it.
        try:
            q_res = requests.get(BASE_URL + "markets/quotes", headers=HEADERS, params={'symbols': underlying})
            if q_res.status_code == 200:
                u_price = q_res.json()['quotes']['quote']['last']
                if option_type == 'put' and u_price < strike: is_itm = True
                elif option_type == 'call' and u_price > strike: is_itm = True
            else:
                print(f"Could not fetch quote for {underlying}")
                continue
        except:
            continue
            
        should_roll = False
        reason = ""
        
        if is_itm:
            should_roll = True
            reason = "ITM"
        elif current_price < 1.01:
            should_roll = True
            reason = "Price < $1.01"
            
        if should_roll:
            print(f"Rolling {symbol}: {reason}")
            
            # Find New Option
            try:
                expirations = get_option_expirations(underlying)
                if 'error' in expirations: raise Exception("Could not fetch expirations")
                
                # Find first exp > 42 days
                target_exp = None
                today = date.today()
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    if (exp_date - today).days > 42:
                        target_exp = exp
                        break
                        
                if not target_exp:
                    results['errors'].append({'symbol': symbol, 'error': 'No expiration > 42 days found'})
                    continue
                    
                # Calculate New Strike
                new_strike = strike - 1 if option_type == 'put' else strike + 1
                
                # Generate New OCC Symbol
                new_occ = _generate_occ_symbol(underlying, target_exp, option_type, new_strike)
                
                # Construct Order
                # Leg 1: BTC Current
                leg1 = {
                    'option_symbol': symbol,
                    'side': 'buy_to_close',
                    'quantity': abs(pos['quantity'])
                }
                # Leg 2: STO New
                leg2 = {
                    'option_symbol': new_occ,
                    'side': 'sell_to_open',
                    'quantity': abs(pos['quantity'])
                }
                
                # Submit Market Order (Net Credit/Debit handled by market)
                res = place_multileg_order([leg1, leg2], underlying, 'market')
                
                if 'error' in res:
                    results['errors'].append({'symbol': symbol, 'error': res['error']})
                else:
                    results['rolled'].append({
                        'symbol': symbol,
                        'new_symbol': new_occ,
                        'reason': reason,
                        'order_id': res.get('order', {}).get('id')
                    })
                    
            except Exception as e:
                results['errors'].append({'symbol': symbol, 'error': str(e)})
                
    return results

def place_multileg_order(legs, symbol, type, duration='day', price=None):
    """
    Generic function to place a multileg order.
    legs: list of dicts with 'option_symbol', 'side', 'quantity'
    """
    if not TRADIER_ACCOUNT_ID or not TRADIER_API_KEY:
        return {'error': 'API Key or Account ID not set.'}

    try:
        order_payload = {
            'class': 'multileg',
            'symbol': symbol.upper(),
            'type': type, # 'market', 'debit', 'credit', 'even'
            'duration': duration,
        }
        
        for i, leg in enumerate(legs):
            order_payload[f'option_symbol[{i}]'] = leg['option_symbol']
            order_payload[f'side[{i}]'] = leg['side']
            order_payload[f'quantity[{i}]'] = leg['quantity']
            
        if price:
            order_payload['price'] = price

        endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
        response = requests.post(BASE_URL + endpoint, data=order_payload, headers=HEADERS)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        print(f"ERROR: Could not place multileg order. {e}")
        return {'error': str(e)}

def check_and_roll_positions():
    """
    Checks for Wheel positions (single-leg) that need rolling.
    Criteria:
    - Single Leg
    - DTE < 9
    - ITM OR Option Price < 1.01
    
    Action:
    - BTC current
    - STO new (Exp > 42 days, Strike +/- 1)
    """
    positions = get_open_positions()
    if positions is None:
        return {'error': 'Could not fetch positions'}
        
    # Identify Spreads to EXCLUDE them
    long_legs = set()
    for p in positions:
        if p['quantity'] > 0 and p.get('underlying') and p.get('expiration') and p.get('option_type'):
            long_legs.add((p['underlying'], p['expiration'], p['option_type']))
            
    results = {
        'rolled': [],
        'errors': [],
        'checked': 0
    }
    
    for pos in positions:
        # Filter: Must be Short (Wheel is selling options)
        if pos['quantity'] >= 0: continue
        
        # Filter: Must NOT be part of a spread (no matching long leg)
        pos_key = (pos.get('underlying'), pos.get('expiration'), pos.get('option_type'))
        if pos_key in long_legs: continue
        
        results['checked'] += 1
        
        dte = pos.get('dte')
        symbol = pos['symbol']
        current_price = pos.get('current_price', 0)
        strike = pos.get('strike')
        option_type = pos.get('option_type')
        underlying = pos.get('underlying')
        
        if dte is None or strike is None: continue
        
        # Check Roll Criteria
        # 1. DTE < 9
        if dte >= 9: continue
        
        # 2. ITM or Price < 1.01
        is_itm = False
        # Fetch underlying price if not in pos (it should be from get_open_positions)
        # We need the underlying price for ITM check. 
        # get_open_positions enriches with 'current_price' of the OPTION, but we need UNDERLYING price for ITM.
        # Wait, get_open_positions does NOT return underlying price in the dict, only 'underlying' symbol.
        # We need to fetch it.
        
        # Optimization: We could fetch all underlying prices at once like in check_and_close.
        # For now, let's just fetch for the candidate to be safe/simple, or reuse the logic.
        # Actually, let's fetch it.
        try:
            q_res = requests.get(BASE_URL + "markets/quotes", headers=HEADERS, params={'symbols': underlying})
            if q_res.status_code == 200:
                u_price = q_res.json()['quotes']['quote']['last']
                if option_type == 'put' and u_price < strike: is_itm = True
                elif option_type == 'call' and u_price > strike: is_itm = True
            else:
                print(f"Could not fetch quote for {underlying}")
                continue
        except:
            continue
            
        should_roll = False
        reason = ""
        
        if is_itm:
            should_roll = True
            reason = "ITM"
            
        if should_roll:
            print(f"Rolling {symbol}: {reason}")
            
            # Find New Option
            try:
                expirations = get_option_expirations(underlying)
                if 'error' in expirations: raise Exception("Could not fetch expirations")
                
                # Find first exp > 42 days
                target_exp = None
                today = date.today()
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    if (exp_date - today).days > 42:
                        target_exp = exp
                        break
                        
                if not target_exp:
                    results['errors'].append({'symbol': symbol, 'error': 'No expiration > 42 days found'})
                    continue
                    
                # Calculate New Strike
                new_strike = strike - 1 if option_type == 'put' else strike + 1
                
                # Generate New OCC Symbol
                new_occ = _generate_occ_symbol(underlying, target_exp, option_type, new_strike)
                
                # Construct Order
                # Leg 1: BTC Current
                leg1 = {
                    'option_symbol': symbol,
                    'side': 'buy_to_close',
                    'quantity': abs(pos['quantity'])
                }
                # Leg 2: STO New
                leg2 = {
                    'option_symbol': new_occ,
                    'side': 'sell_to_open',
                    'quantity': abs(pos['quantity'])
                }
                
                # Submit Debit Limit Order
                res = place_multileg_order([leg1, leg2], underlying, 'debit', price=0.90)
                
                if 'error' in res:
                    results['errors'].append({'symbol': symbol, 'error': res['error']})
                else:
                    results['rolled'].append({
                        'symbol': symbol,
                        'new_symbol': new_occ,
                        'reason': reason,
                        'order_id': res.get('order', {}).get('id')
                    })
                    
            except Exception as e:
                results['errors'].append({'symbol': symbol, 'error': str(e)})
                
    return results