from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, InputRequired

# The custom date validator is no longer needed here as the dates will be pre-validated from the API
# and the user can only select from a list.

class VerticalSpreadForm(FlaskForm):
    symbol = StringField('Stock Ticker', validators=[DataRequired()], default='TSLA')
    
    # CHANGED: This is now a SelectField. Choices will be added via JavaScript.
    expiration = SelectField('Expiration', validators=[DataRequired()], choices=[])
    
    spread_type = SelectField('Spread Type', choices=[('debit', 'Debit'), ('credit', 'Credit')], validators=[DataRequired()])
    option_type = SelectField('Option Type', choices=[('call', 'Call'), ('put', 'Put')], validators=[DataRequired()])
    
    long_strike = StringField('Buy Strike', validators=[DataRequired()])
    short_strike = StringField('Sell Strike', validators=[DataRequired()])
    
    quantity = IntegerField('Quantity', validators=[InputRequired()], default=1)
    
    submit = SubmitField('Preview Spread Order')