import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import from services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.vault_service import VaultService

class TestVaultService(unittest.TestCase):
    def setUp(self):
        # Mock hvac.Client
        self.patcher = patch('services.vault_service.hvac.Client')
        self.MockClient = self.patcher.start()
        
        # Setup environment variables
        os.environ['VAULT_ADDR'] = 'http://localhost:8200'
        os.environ['VAULT_TOKEN'] = 'test-token'
        
        self.vault_service = VaultService()
        self.vault_service.client = self.MockClient()

    def tearDown(self):
        self.patcher.stop()

    def test_is_configured_true(self):
        # Setup mock to return data
        self.vault_service.client.is_authenticated.return_value = True
        self.vault_service.client.secrets.kv.v2.read_secret_version.return_value = {
            'data': {'data': {'api_key': 'foo', 'account_id': 'bar'}}
        }
        
        self.assertTrue(self.vault_service.is_configured())

    def test_is_configured_false(self):
        # Setup mock to raise InvalidPath (secret doesn't exist)
        from hvac.exceptions import InvalidPath
        self.vault_service.client.is_authenticated.return_value = True
        self.vault_service.client.secrets.kv.v2.read_secret_version.side_effect = InvalidPath
        
        self.assertFalse(self.vault_service.is_configured())

    def test_set_secrets(self):
        self.vault_service.set_tradier_secrets('new_key', 'new_id')
        
        self.vault_service.client.secrets.kv.v2.create_or_update_secret.assert_called_with(
            path='tradier',
            secret={'api_key': 'new_key', 'account_id': 'new_id'}
        )

if __name__ == '__main__':
    unittest.main()
