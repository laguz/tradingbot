from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, InputRequired, ValidationError
from datetime import datetime

def is_future_date(form, field):
    """Validator to ensure the expiration date is in the future."""
    try:
        exp_date = datetime.strptime(field.data, '%Y-%m-%d').date()
        if exp_date <= datetime.now().date():
            raise ValidationError('Expiration date must be in the future.')
    except ValueError:
        raise ValidationError('Invalid date format. Please use YYYY-MM-DD.')

class VerticalSpreadForm(FlaskForm):
    symbol = StringField('Stock Ticker', validators=[DataRequired()])
    expiration = StringField('Expiration (YYYY-MM-DD)', validators=[DataRequired(), is_future_date])
    
    spread_type = SelectField('Spread Type', choices=[('debit', 'Debit'), ('credit', 'Credit')], validators=[DataRequired()])
    option_type = SelectField('Option Type', choices=[('call', 'Call'), ('put', 'Put')], validators=[DataRequired()])
    
    long_strike = StringField('Buy Strike', validators=[DataRequired()])
    short_strike = StringField('Sell Strike', validators=[DataRequired()])
    
    quantity = IntegerField('Quantity', validators=[InputRequired()], default=1)
    
    submit = SubmitField('Preview Spread Order')