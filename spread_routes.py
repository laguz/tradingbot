from flask import Blueprint, render_template, flash, redirect, url_for
from forms import VerticalSpreadForm, StockOrderForm
from services.tradier_service import place_vertical_spread_order, get_option_expirations, place_stock_order

spreads = Blueprint('spreads', __name__)

@spreads.route('/spreads/vertical', methods=['GET', 'POST'])
def submit_vertical_spread():
    spread_form = VerticalSpreadForm()
    stock_form = StockOrderForm()

    if spread_form.is_submitted() and spread_form.submit.data:
        symbol = spread_form.symbol.data.upper()
        if symbol:
            expirations = get_option_expirations(symbol)
            spread_form.expiration.choices = [(d, d) for d in expirations]

    if spread_form.validate_on_submit() and spread_form.submit.data:
        form_data = {
            'symbol': spread_form.symbol.data,
            'expiration': spread_form.expiration.data,
            'spread_type': spread_form.spread_type.data,
            'option_type': spread_form.option_type.data,
            'long_strike': spread_form.long_strike.data,
            'short_strike': spread_form.short_strike.data,
            'quantity': spread_form.quantity.data
        }

        result = place_vertical_spread_order(form_data)
        
        if result and 'error' not in result:
            flash(f"Successfully submitted market order for {spread_form.symbol.data.upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            if result:
                error_message = result.get('error', 'An unknown error occurred.')
            else:
                error_message = "Order service returned an empty response. Please check the logs."
            
            flash(f"Order submission failed: {error_message}", 'danger')

    if stock_form.validate_on_submit() and stock_form.submit_stock.data:
        result = place_stock_order(
            stock_form.symbol.data, 
            stock_form.quantity.data,
            stock_form.order_type.data
        )
        if result and 'error' not in result:
            flash(f"Successfully submitted market order to {stock_form.order_type.data} {stock_form.quantity.data} share(s) of {stock_form.symbol.data.upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            if result:
                error_message = result.get('error', 'An unknown error occurred.')
            else:
                error_message = "Order service returned an empty response. Please check the logs."
            flash(f"Order submission failed: {error_message}", 'danger')

    return render_template('vertical_spread.html', spread_form=spread_form, stock_form=stock_form)