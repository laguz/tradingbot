from flask import Blueprint, render_template, flash, redirect, url_for
from forms import VerticalSpreadForm, StockOrderForm, SingleOptionForm, IronCondorForm
from services.tradier_service import (
    place_vertical_spread_order, get_option_expirations, place_stock_order,
    place_single_option_order, place_iron_condor_order
)

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