import os
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

def generate_challenge():
    """Generates a random 32-byte hex challenge string."""
    return os.urandom(32).hex()

def verify_signature(public_key_hex, signature_hex, message):
    """
    Verifies an Ed25519 signature.
    
    Args:
        public_key_hex (str): The hex-encoded public key.
        signature_hex (str): The hex-encoded signature.
        message (str): The original message that was signed.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        # Convert hex strings to bytes
        public_key_bytes = bytes.fromhex(public_key_hex)
        signature_bytes = bytes.fromhex(signature_hex)
        # BUG FIX: Frontend (noble-ed25519) treats 'message' string as hex and signs the raw bytes.
        # We must verify the signature against the raw bytes, not the utf-8 encoded hex string.
        message_bytes = bytes.fromhex(message)

        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        
        public_key.verify(signature_bytes, message_bytes)
        return True
    except (ValueError, InvalidSignature) as e:
        print(f"Verification error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during verification: {e}")
        return False
