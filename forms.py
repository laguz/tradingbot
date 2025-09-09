from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, InputRequired, Optional

# The custom date validator is no longer needed here as the dates will be pre-validated from the API
# and the user can only select from a list.

class VerticalSpreadForm(FlaskForm):
    symbol = StringField('Stock Ticker', validators=[DataRequired()], default='TSLA')
    expiration = SelectField('Expiration', validators=[DataRequired()], choices=[])
    spread_type = SelectField('Spread Type', choices=[('debit', 'Debit'), ('credit', 'Credit')], validators=[DataRequired()])
    option_type = SelectField('Option Type', choices=[('call', 'Call'), ('put', 'Put')], validators=[DataRequired()])
    long_strike = StringField('Buy Strike', validators=[DataRequired()])
    short_strike = StringField('Sell Strike', validators=[DataRequired()])
    quantity = IntegerField('Quantity', validators=[InputRequired()], default=1)
    price = StringField('Limit Price (Net Debit/Credit)', validators=[DataRequired()])
    submit = SubmitField('Place Vertical Spread Order')

class StockOrderForm(FlaskForm):
    symbol = StringField('Stock Ticker', validators=[DataRequired()], default='RIOT')
    side = SelectField('Action', choices=[('buy', 'Buy'), ('sell', 'Sell')], validators=[DataRequired()])
    order_type = SelectField('Order Type', choices=[('market', 'Market'), ('limit', 'Limit')], validators=[DataRequired()])
    quantity = IntegerField('Quantity', validators=[InputRequired()], default=1)
    price = StringField('Limit Price', validators=[Optional()])
    submit_stock = SubmitField('Place Stock Order')

class SingleOptionForm(FlaskForm):
    symbol = StringField('Stock Ticker', validators=[DataRequired()], default='TSLA')
    expiration = SelectField('Expiration', validators=[DataRequired()], choices=[])
    strike = StringField('Strike Price', validators=[DataRequired()])
    option_type = SelectField('Option Type', choices=[('call', 'Call'), ('put', 'Put')], validators=[DataRequired()])
    side = SelectField('Action', choices=[('buy_to_open', 'Buy to Open'), ('sell_to_open', 'Sell to Open')], validators=[DataRequired()])
    order_type = SelectField('Order Type', choices=[('market', 'Market'), ('limit', 'Limit')], validators=[DataRequired()])
    quantity = IntegerField('Quantity', validators=[InputRequired()], default=1)
    price = StringField('Limit Price', validators=[Optional()])
    submit_single = SubmitField('Place Single Option Order')

class IronCondorForm(FlaskForm):
    symbol = StringField('Stock Ticker', validators=[DataRequired()], default='TSLA')
    spread_type = SelectField('Spread Type', choices=[('credit', 'Credit (Short)'), ('debit', 'Debit (Long)')], validators=[DataRequired()])
    expiration = SelectField('Expiration', validators=[DataRequired()], choices=[])
    short_call_strike = StringField('Short Call Strike', validators=[DataRequired()])
    long_call_strike = StringField('Long Call Strike', validators=[DataRequired()])
    short_put_strike = StringField('Short Put Strike', validators=[DataRequired()])
    long_put_strike = StringField('Long Put Strike', validators=[DataRequired()])
    quantity = IntegerField('Quantity', validators=[InputRequired()], default=1)
    price = StringField('Limit Price (Net Credit/Debit)', validators=[DataRequired()])
    submit_condor = SubmitField('Place Iron Condor Order')