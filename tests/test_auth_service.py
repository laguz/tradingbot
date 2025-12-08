import unittest
from services.pubky_auth import generate_challenge, verify_signature
from cryptography.hazmat.primitives.asymmetric import ed25519

class TestAuthService(unittest.TestCase):
    def test_verify_signature(self):
        # Generate keys
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        public_hex = public_key.public_bytes_raw().hex()
        
        # Message must be a hex string now
        # Simulating what happens in app: generate_challenge() returns random hex
        import os
        message = os.urandom(32).hex()
        
        # Sign the RAW bytes, not the utf-8 string
        signature = private_key.sign(bytes.fromhex(message))
        signature_hex = signature.hex()
        
        # Verify Positive
        self.assertTrue(verify_signature(public_hex, signature_hex, message))
        
        # Verify Negative (Wrong Message)
        wrong_message = os.urandom(32).hex()
        self.assertFalse(verify_signature(public_hex, signature_hex, wrong_message))
        
        # Verify Negative (Wrong Signature)
        wrong_sig = b'\x00' * 64
        self.assertFalse(verify_signature(public_hex, wrong_sig.hex(), message))

if __name__ == '__main__':
    unittest.main()
