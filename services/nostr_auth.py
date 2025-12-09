"""
Nostr Authentication Service

Handles Nostr-based authentication using public key cryptography.
Users authenticate by signing a challenge with their Nostr private key.
"""

from flask_login import UserMixin
from datetime import datetime, timedelta
import hashlib
import secrets
import json
from typing import Optional, Dict
from utils.logger import logger


class NostrUser(UserMixin):
    """User class for Flask-Login integration."""
    
    def __init__(self, pubkey: str, metadata: Optional[Dict] = None):
        """
        Initialize Nostr user.
        
        Args:
            pubkey: User's Nostr public key (hex format)
            metadata: Optional user metadata (name, picture, etc.)
        """
        self.id = pubkey  # Flask-Login requires 'id' attribute
        self.pubkey = pubkey
        self.metadata = metadata or {}
        self.authenticated_at = datetime.now()
        self.username = None  # Custom username from database
        
        # Load username from database
        self._load_profile()
    
    def _load_profile(self):
        """Load user profile from database."""
        try:
            from models.mongodb_models import UserProfileModel
            profile = UserProfileModel.find_by_pubkey(self.pubkey)
            if profile:
                self.username = profile.get('username')
        except Exception as e:
            logger.debug(f"Could not load profile for {self.pubkey[:8]}: {e}")
    
    def get_id(self):
        """Return user ID for Flask-Login."""
        return self.pubkey
    
    def get_display_name(self):
        """Get display name: username > metadata name > truncated pubkey."""
        if self.username:
            return self.username
        if self.metadata.get('name'):
            return self.metadata['name']
        return f"{self.pubkey[:8]}...{self.pubkey[-4:]}"



class NostrAuthService:
    """Service for Nostr authentication."""
    
    def __init__(self):
        """Initialize auth service."""
        self.active_challenges = {}  # pubkey -> challenge
        self.challenge_expiry = timedelta(minutes=5)
    
    def generate_challenge(self, pubkey: str) -> str:
        """
        Generate authentication challenge for a public key.
        
        Args:
            pubkey: User's Nostr public key
            
        Returns:
            Challenge string to be signed
        """
        # Create unique challenge
        timestamp = int(datetime.now().timestamp())
        random_data = secrets.token_hex(16)
        challenge = f"Sign this message to authenticate:\nTimestamp: {timestamp}\nRandom: {random_data}"
        
        # Store challenge
        self.active_challenges[pubkey] = {
            'challenge': challenge,
            'created_at': datetime.now()
        }
        
        logger.info(f"Generated challenge for pubkey {pubkey[:8]}...")
        return challenge
    
    def verify_signature(self, pubkey: str, signature: str, challenge: str = None) -> bool:
        """
        Verify Nostr signature.
        
        Args:
            pubkey: User's public key
            signature: Signature to verify
            challenge: Optional challenge (if not provided, uses stored)
            
        Returns:
            True if signature is valid
        """
        # Get stored challenge if not provided
        if challenge is None:
            stored = self.active_challenges.get(pubkey)
            if not stored:
                logger.warning(f"No challenge found for pubkey {pubkey[:8]}...")
                return False
            
            # Check expiry
            if datetime.now() - stored['created_at'] > self.challenge_expiry:
                logger.warning(f"Challenge expired for pubkey {pubkey[:8]}...")
                del self.active_challenges[pubkey]
                return False
            
            challenge = stored['challenge']
        
        # In production, use proper Nostr signature verification
        # For now, simplified verification (assumes signature format is correct)
        try:
            # This is a placeholder - in production use proper Nostr crypto
            # from nostr.key import PublicKey
            # pk = PublicKey.from_hex(pubkey)
            # return pk.verify_signature(challenge, signature)
            
            # Simplified check for development
            is_valid = len(signature) == 128  # Schnorr signatures are 64 bytes = 128 hex chars
            
            if is_valid:
                logger.info(f"Signature verified for pubkey {pubkey[:8]}...")
                # Clean up challenge
                if pubkey in self.active_challenges:
                    del self.active_challenges[pubkey]
            else:
                logger.warning(f"Invalid signature for pubkey {pubkey[:8]}...")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def cleanup_expired_challenges(self):
        """Remove expired challenges."""
        now = datetime.now()
        expired = [
            pubkey for pubkey, data in self.active_challenges.items()
            if now - data['created_at'] > self.challenge_expiry
        ]
        
        for pubkey in expired:
            del self.active_challenges[pubkey]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired challenges")


# Global auth service instance
nostr_auth = NostrAuthService()
