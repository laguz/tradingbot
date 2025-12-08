import unittest
import requests
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from time import sleep

# Assuming the app is running on localhost:5000
BASE_URL = "http://localhost:5000"

class TestPubkyAuth(unittest.TestCase):
    def test_auth_flow(self):
        # 1. Generate local keys (Client side simulation)
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Get hex representations
        private_bytes = private_key.private_bytes(
            encoding=os.environ.get('ENCODING', 'Raw'), # Not used directly here, just raw bytes needed
            format=os.environ.get('FORMAT', 'Raw'), 
            encryption_algorithm=os.environ.get('ALGO', 'NoEncryption')
        )
        # Using raw bytes for ed25519 is simpler
        private_bytes = private_key.private_bytes_raw()
        public_bytes = public_key.public_bytes_raw()
        
        public_hex = public_bytes.hex()
        
        print(f"Testing with Public Key: {public_hex}")
        
        # 2. Get Challenge
        session = requests.Session()
        resp = session.get(f"{BASE_URL}/api/auth/challenge")
        self.assertEqual(resp.status_code, 200)
        challenge_hex = resp.json()['challenge']
        print(f"Received Challenge: {challenge_hex}")
        
        # 3. Sign Challenge
        # Ed25519 signature
        signature = private_key.sign(challenge_hex.encode('utf-8'))
        signature_hex = signature.hex()
        
        # 4. Verify & Login
        payload = {
            'public_key': public_hex,
            'signature': signature_hex
        }
        
        resp = session.post(f"{BASE_URL}/api/auth/verify", json=payload)
        print(f"Verify Response: {resp.text}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()['success'])
        
        # 5. Check if logged in (access protected route)
        resp = session.get(f"{BASE_URL}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Logout", resp.text)
        print("Login successful and verified via dashboard access.")

if __name__ == '__main__':
    # We can't run this easily against the live server if it's not running in background.
    # So we will rely on unit testing the service functions directly first.
    pass
