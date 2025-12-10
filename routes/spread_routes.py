from flask import Blueprint, render_template, flash, redirect, url_for
from forms import VerticalSpreadForm, StockOrderForm, SingleOptionForm, IronCondorForm
from services.tradier_service import (
    place_vertical_spread_order, get_option_expirations, place_stock_order,
    place_single_option_order, place_iron_condor_order
)
from services.market_analysis import calculate_smart_strikes

spreads = Blueprint('spreads', __name__)

@spreads.route('/spreads/vertical', methods=['GET', 'POST'])
def submit_vertical_spread():
    spread_form = VerticalSpreadForm()
    stock_form = StockOrderForm()
    single_option_form = SingleOptionForm()
    iron_condor_form = IronCondorForm()

    # --- Server-side expiration population for non-JS scenarios ---
    if spread_form.is_submitted() and spread_form.submit.data:
        symbol = spread_form.symbol.data.upper()
        if symbol:
            spread_form.expiration.choices = [(d, d) for d in get_option_expirations(symbol)]
    if single_option_form.is_submitted() and single_option_form.submit_single.data:
        symbol = single_option_form.symbol.data.upper()
        if symbol:
            single_option_form.expiration.choices = [(d, d) for d in get_option_expirations(symbol)]
    if iron_condor_form.is_submitted() and iron_condor_form.submit_condor.data:
        symbol = iron_condor_form.symbol.data.upper()
        if symbol:
            iron_condor_form.expiration.choices = [(d, d) for d in get_option_expirations(symbol)]

    # --- Form Processing ---
    if spread_form.validate_on_submit() and spread_form.submit.data:
        form_data = {**spread_form.data}
        
        # --- Auto-Submission Logic ---
        if form_data.get('auto_strikes'):
            try:
                short_strike, long_strike, trigger_level = calculate_smart_strikes(
                    form_data['symbol'],
                    form_data['expiration'],
                    form_data['spread_type'],
                    form_data['option_type'],
                    float(form_data['spread_width'])
                )
                form_data['short_strike'] = str(short_strike)
                form_data['long_strike'] = str(long_strike)
                flash(f"Auto-Selected Strikes: Short {short_strike}, Long {long_strike} (Based on Level: {trigger_level})", 'info')
            except Exception as e:
                flash(f"Auto-selection failed: {str(e)}", 'danger')
                return redirect(url_for('spreads.submit_vertical_spread'))

        result = place_vertical_spread_order(form_data)
        if result and 'error' not in result:
            flash(f"Successfully submitted vertical spread for {form_data['symbol'].upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            error_message = result.get('error', 'An unknown error.') if result else "Order service returned an empty response."
            flash(f"Vertical spread submission failed: {error_message}", 'danger')

    if stock_form.validate_on_submit() and stock_form.submit_stock.data:
        form_data = {**stock_form.data}
        result = place_stock_order(form_data)
        if result and 'error' not in result:
            flash(f"Successfully submitted market order to {form_data['side']} {form_data['quantity']} share(s) of {form_data['symbol'].upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            error_message = result.get('error', 'An unknown error.') if result else "Order service returned an empty response."
            flash(f"Stock order submission failed: {error_message}", 'danger')

    if single_option_form.validate_on_submit() and single_option_form.submit_single.data:
        form_data = {**single_option_form.data}
        result = place_single_option_order(form_data)
        if result and 'error' not in result:
            flash(f"Successfully submitted single option order for {form_data['symbol'].upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            error_message = result.get('error', 'An unknown error.') if result else "Order service returned an empty response."
            flash(f"Single option order submission failed: {error_message}", 'danger')

    if iron_condor_form.validate_on_submit() and iron_condor_form.submit_condor.data:
        form_data = {**iron_condor_form.data}
        result = place_iron_condor_order(form_data)
        if result and 'error' not in result:
            flash(f"Successfully submitted iron condor order for {form_data['symbol'].upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            error_message = result.get('error', 'An unknown error.') if result else "Order service returned an empty response."
            flash(f"Iron condor order submission failed: {error_message}", 'danger')

    return render_template('vertical_spread.html',
                           spread_form=spread_form,
                           stock_form=stock_form,
                           single_option_form=single_option_form,
                           iron_condor_form=iron_condor_form)

# --- Auto Order Routes ---

@spreads.route('/auto_order')
def auto_order():
    return render_template('auto_order.html')

@spreads.route('/api/calculate_auto_order')
def calculate_auto_order():
    from flask import request, jsonify
    symbol = request.args.get('symbol')
    order_type = request.args.get('type') # 'put' or 'call'
    
    if not symbol or not order_type:
        return jsonify({'error': 'Missing symbol or type'}), 400
        
    try:
        # Determine parameters based on type
        # Put Credit Spread: Sell Put (Bullish) -> Support
        # Call Credit Spread: Sell Call (Bearish) -> Resistance
        option_type = 'put' if order_type == 'put' else 'call'
        spread_type = 'credit'
        
        # Get expirations to find the nearest valid one
        expirations = get_option_expirations(symbol)
        if 'error' in expirations:
            return jsonify(expirations), 400
            
        from datetime import date, timedelta, datetime
        today = date.today()
        
        # Filter for expirations between 18 and 25 days (inclusive)
        valid_expirations = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            if 18 <= days_to_exp <= 25:
                valid_expirations.append(exp_str)
        
        if not valid_expirations:
            return jsonify({'error': 'No expirations found between 18 and 25 days.'}), 400
            
        # Pick the first one (soonest)
        best_expiration = valid_expirations[0]
        
        short_strike, long_strike, trigger_level = calculate_smart_strikes(
            symbol,
            best_expiration,
            spread_type,
            option_type,
            width=5.0 # Fixed 5-wide spread
        )
        
        return jsonify({
            'symbol': symbol,
            'expiration': best_expiration,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'trigger_level': trigger_level,
            'spread_type': spread_type,
            'option_type': option_type,
            'width': 5.0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@spreads.route('/auto_order/submit', methods=['POST'])
def submit_auto_order():
    from flask import request, jsonify
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    try:
        # Construct form_data expected by place_vertical_spread_order
        form_data = {
            'symbol': data['symbol'],
            'expiration': data['expiration'],
            'spread_type': 'credit', # Must be 'credit' for limit credit order, 'limit' is invalid for multileg
            'option_type': data['option_type'],
            'short_strike': data['short_strike'],
            'long_strike': data['long_strike'],
            'quantity': data.get('quantity', 1),
            'price': '0.75', # Fixed limit price
            'spread_width': 5.0
        }
        
        result = place_vertical_spread_order(form_data)
        
        if result and 'error' not in result:
            return jsonify({'status': 'success', 'order_id': result.get('order', {}).get('id')})
        else:
            error_message = result.get('error', 'Unknown error') if result else 'Empty response'
            return jsonify({'error': error_message}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@spreads.route('/api/auto_close', methods=['POST'])
def api_auto_close():
    from flask import jsonify
    from services.tradier_service import check_and_close_positions
    try:
        results = check_and_close_positions()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Wheel Option Routes ---

@spreads.route('/wheel_option')
def wheel_option():
    return render_template('wheel_option.html')

@spreads.route('/api/calculate_wheel')
def calculate_wheel():
    from flask import request, jsonify
    symbol = request.args.get('symbol')
    order_type = request.args.get('type') # 'put' or 'call'
    
    if not symbol or not order_type:
        return jsonify({'error': 'Missing symbol or type'}), 400
        
    try:
        # Determine parameters
        # Sell Put (Bullish) -> Support
        # Sell Call (Bearish) -> Resistance
        option_type = 'put' if order_type == 'put' else 'call'
        spread_type = 'credit' # Still credit strategy logic for S/R
        
        # Get expirations
        expirations = get_option_expirations(symbol)
        if 'error' in expirations:
            return jsonify(expirations), 400
            
        from datetime import date, timedelta, datetime
        today = date.today()
        
        # Filter for expirations between 40 and 50 days (exclusive)
        valid_expirations = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            if 40 < days_to_exp < 50:
                valid_expirations.append(exp_str)
        
        if not valid_expirations:
            return jsonify({'error': 'No expirations found between 40 and 50 days.'}), 400
            
        best_expiration = valid_expirations[0]
        
        # Reuse calculate_smart_strikes to get the short strike
        # We ignore long_strike since this is a single leg
        short_strike, _, trigger_level = calculate_smart_strikes(
            symbol,
            best_expiration,
            spread_type,
            option_type,
            width=5.0 # Width doesn't matter for short strike calculation
        )
        
        return jsonify({
            'symbol': symbol,
            'expiration': best_expiration,
            'strike': short_strike,
            'trigger_level': trigger_level,
            'option_type': option_type,
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@spreads.route('/wheel/submit', methods=['POST'])
def submit_wheel_order():
    from flask import request, jsonify
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    try:
        # Construct form_data for place_single_option_order
        form_data = {
            'symbol': data['symbol'],
            'expiration': data['expiration'],
            'option_type': data['option_type'],
            'strike': str(data['strike']),
            'side': 'sell_to_open', # Wheel strategy: Sell to Open
            'quantity': data.get('quantity', 1),
            'order_type': 'limit',
            'price': '0.20' # Fixed limit price
        }
        
        result = place_single_option_order(form_data)
        
        if result and 'error' not in result:
            return jsonify({'status': 'success', 'order_id': result.get('order', {}).get('id')})
        else:
            error_message = result.get('error', 'Unknown error') if result else 'Empty response'
            return jsonify({'error': error_message}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@spreads.route('/api/auto_roll', methods=['POST'])
def api_auto_roll():
    from flask import jsonify
    from services.tradier_service import check_and_roll_positions
    try:
        results = check_and_roll_positions()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500