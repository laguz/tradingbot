"""
Nostr Authentication Routes

Handles login, logout, and authentication endpoints.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from services.nostr_auth import NostrAuthService, NostrUser, nostr_auth
from utils.logger import logger

auth_routes = Blueprint('auth', __name__)

# Will be initialized from app.py
login_manager = None


def init_login_manager(app):
    """Initialize Flask-Login."""
    global login_manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in with Nostr to access this page.'
    
    @login_manager.user_loader
    def load_user(pubkey):
        """Load user from session."""
        if pubkey in session.get('nostr_users', {}):
            user_data = session['nostr_users'][pubkey]
            return NostrUser(pubkey, user_data.get('metadata'))
        return None


@auth_routes.route('/login')
def login():
    """Show Nostr login page."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    return render_template('login.html')


@auth_routes.route('/api/auth/challenge', methods=['POST'])
def get_challenge():
    """
    Generate authentication challenge for a pubkey.
    
    Request JSON:
        {
            "pubkey": "hex_public_key"
        }
    
    Response:
        {
            "challenge": "message_to_sign"
        }
    """
    data = request.get_json()
    pubkey = data.get('pubkey')
    
    if not pubkey:
        return jsonify({'error': 'Missing pubkey'}), 400
    
    if len(pubkey) != 64:  # Nostr pubkeys are 32 bytes = 64 hex chars
        return jsonify({'error': 'Invalid pubkey format'}), 400
    
    challenge = nostr_auth.generate_challenge(pubkey)
    
    return jsonify({
        'challenge': challenge,
        'pubkey': pubkey
    })


@auth_routes.route('/api/auth/verify', methods=['POST'])
def verify_signature():
    """
    Verify signature and log in user.
    
    Request JSON:
        {
            "pubkey": "hex_public_key",
            "signature": "hex_signature",
            "metadata": {
                "name": "User Name",
                "picture": "url",
                ...
            }
        }
    
    Response:
        {
            "success": true,
            "user": {
                "pubkey": "...",
                "display_name": "..."
            }
        }
    """
    data = request.get_json()
    pubkey = data.get('pubkey')
    signature = data.get('signature')
    metadata = data.get('metadata', {})
    
    if not pubkey or not signature:
        return jsonify({'error': 'Missing pubkey or signature'}), 400
    
    # Verify signature
    if not nostr_auth.verify_signature(pubkey, signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Create user and log in
    user = NostrUser(pubkey, metadata)
    login_user(user, remember=True)
    
    # Store in session
    if 'nostr_users' not in session:
        session['nostr_users'] = {}
    session['nostr_users'][pubkey] = {
        'metadata': metadata,
        'authenticated_at': user.authenticated_at.isoformat()
    }
    session.modified = True
    
    logger.info(f"User logged in: {user.get_display_name()} ({pubkey[:8]}...)")
    
    return jsonify({
        'success': True,
        'user': {
            'pubkey': pubkey,
            'display_name': user.get_display_name(),
            'metadata': metadata
        }
    })


@auth_routes.route('/logout')
@login_required
def logout():
    """Log out current user."""
    pubkey = current_user.pubkey
    logger.info(f"User logged out: {current_user.get_display_name()}")
    
    # Remove from session
    if 'nostr_users' in session and pubkey in session['nostr_users']:
        del session['nostr_users'][pubkey]
        session.modified = True
    
    logout_user()
    return redirect(url_for('auth.login'))


@auth_routes.route('/api/auth/status')
def auth_status():
    """Get current authentication status."""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'pubkey': current_user.pubkey,
                'display_name': current_user.get_display_name(),
                'metadata': current_user.metadata
            }
        })
    else:
        return jsonify({'authenticated': False})
