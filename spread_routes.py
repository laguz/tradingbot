from flask import Blueprint, render_template, flash, redirect, url_for
from forms import VerticalSpreadForm
from services.tradier_service import place_vertical_spread_order

spreads = Blueprint('spreads', __name__)

@spreads.route('/spreads/vertical', methods=['GET', 'POST'])
def submit_vertical_spread():
    form = VerticalSpreadForm()
    if form.validate_on_submit():
        form_data = {
            'symbol': form.symbol.data,
            'expiration': form.expiration.data,
            'spread_type': form.spread_type.data,
            'option_type': form.option_type.data,
            'long_strike': form.long_strike.data,
            'short_strike': form.short_strike.data,
            'quantity': form.quantity.data
        }

        result = place_vertical_spread_order(form_data)
        
        if result and 'error' not in result:
            flash(f"Successfully submitted market order for {form.symbol.data.upper()}. Order ID: {result.get('order', {}).get('id')}", 'success')
            return redirect(url_for('spreads.submit_vertical_spread'))
        else:
            error_message = result.get('error', 'An unknown error occurred.')
            flash(f"Order submission failed: {error_message}", 'danger')

    return render_template('vertical_spread.html', form=form)