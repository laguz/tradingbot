from flask import Blueprint, render_template, request, flash
from services.ml_service import predict_next_days

predictions = Blueprint('predictions', __name__)

@predictions.route('/prediction', methods=['GET', 'POST'])
def show_prediction():
    prediction_data = None
    ticker = 'TSLA' # Default
    
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'TSLA').upper()
        try:
            result = predict_next_days(ticker, days=5)
            if 'error' in result:
                flash(f"Error: {result['error']}", 'danger')
            else:
                prediction_data = result
        except Exception as e:
            flash(f"Prediction failed: {str(e)}", 'danger')
            
    return render_template('prediction.html', ticker=ticker, prediction=prediction_data)
