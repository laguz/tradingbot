from flask import Blueprint, render_template, request, flash
from services.ml_service import predict_next_days

predictions = Blueprint('predictions', __name__)

@predictions.route('/prediction', methods=['GET', 'POST'])
def show_prediction():
    prediction_data = None
    ticker = 'TSLA'  # Default
    
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'TSLA').upper()
        force_retrain = request.form.get('force_retrain') == 'true'
        
        try:
            result = predict_next_days(ticker, days=5, force_retrain=force_retrain)
            if 'error' in result:
                flash(f"Error: {result['error']}", 'danger')
            else:
                prediction_data = result
                # Add summary message
                last_close = result['last_close']
                median_day5 = result['confidence_intervals']['median'][4]
                change_pct = ((median_day5 - last_close) / last_close) * 100
                direction = "up" if change_pct > 0 else "down"
                flash(f"Prediction complete! Model v{result['model_version']} predicts {abs(change_pct):.1f}% {direction} in 5 days (using {result['feature_count']} features)", 'success')
        except Exception as e:
            flash(f"Prediction failed: {str(e)}", 'danger')
            
    return render_template('prediction.html', ticker=ticker, prediction=prediction_data)

