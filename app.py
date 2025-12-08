from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
# UPDATED: Added get_option_expirations to the import list
from services.tradier_service import get_account_summary, get_open_positions, get_yearly_pl, get_historical_data, get_option_expirations, get_current_price, check_and_close_positions
from services.pubky_auth import generate_challenge, verify_signature
from spread_routes import spreads
from prediction_routes import predictions
from models import User
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

app.register_blueprint(spreads)
app.register_blueprint(predictions)

# --- Auth Routes ---

@app.route('/login', methods=['GET'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/auth/challenge')
def auth_challenge():
    challenge = generate_challenge()
    session['auth_challenge'] = challenge
    return jsonify({'challenge': challenge})

@app.route('/api/auth/verify', methods=['POST'])
def auth_verify():
    data = request.json
    public_key = data.get('public_key')
    signature = data.get('signature')
    
    challenge = session.get('auth_challenge')
    if not challenge:
        return jsonify({'error': 'No challenge active'}), 400
        
    if verify_signature(public_key, signature, challenge):
        user = User(public_key)
        login_user(user)
        session.pop('auth_challenge', None)
        return jsonify({'success': True})
    
    return jsonify({'error': 'Invalid signature'}), 401

# --- Main Page Routes ---

@app.route('/network')
@login_required
def network():
    return render_template('network.html')

@app.route('/')
@login_required
def dashboard():
    """
    Renders the main dashboard page with the live account data.
    """
    summary_data = get_account_summary()
    positions_data = get_open_positions()
    yearly_pl_data = get_yearly_pl()
    
    return render_template('dashboard.html', 
                           summary=summary_data or {},
                           positions=positions_data or [],
                           yearly_pl=yearly_pl_data or {})

@app.route('/charts')
@login_required
def charts():
    """
    Renders the chart page and populates initial analysis.
    """
    ticker = request.args.get('ticker', 'TSLA').upper()
    timeframe = request.args.get('timeframe', '6m')
    
    chart_data = get_historical_data(ticker, timeframe)
    levels = chart_data.get('levels') if chart_data else None

    return render_template('charts.html',
                           ticker=ticker,
                           timeframe=timeframe,
                           levels=levels)

# --- API Endpoints ---

@app.route('/api/history/<string:ticker>/<string:timeframe>')
def api_history(ticker, timeframe):
    """
    API endpoint to fetch historical stock data for charts.
    """
    data = get_historical_data(ticker, timeframe)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Could not retrieve data for the given ticker.'}), 404

@app.route('/api/expirations/<string:symbol>')
def api_expirations(symbol):
    """
    API endpoint to fetch option expiration dates for a given symbol.
    """
    expirations = get_option_expirations(symbol)
    if 'error' in expirations:
        return jsonify(expirations), 404
    return jsonify(expirations)

@app.route('/api/price/<string:symbol>')
def api_price(symbol):
    """
    API endpoint to fetch current price for a symbol.
    """
    price = get_current_price(symbol)
    if isinstance(price, dict) and 'error' in price:
        return jsonify(price), 400
    return jsonify({'symbol': symbol.upper(), 'price': price})


# --- Main execution block ---

if __name__ == '__main__':
    app.run(debug=True, port=5000)