import hvac
import os

class VaultService:
    def __init__(self):
        # Default to localhost if not set
        self.vault_addr = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
        self.vault_token = os.getenv('VAULT_TOKEN')
        self.client = None
        self._connect()

    def _connect(self):
        if self.vault_token:
            try:
                self.client = hvac.Client(
                    url=self.vault_addr,
                    token=self.vault_token
                )
            except Exception as e:
                print(f"Failed to connect to Vault: {e}")
                self.client = None

    def set_token(self, token):
        """Manually sets the Vault token and reconnects."""
        self.vault_token = token
        self._connect()

    def is_configured(self):
        """Checks if Tradier credentials exist in Vault."""
        if not self.client or not self.client.is_authenticated():
            print("Vault client not authenticated.")
            return False
            
        try:
            # Check for secrets at 'secret/data/tradier' (KV v2 path)
            read_response = self.client.secrets.kv.v2.read_secret_version(path='tradier')
            data = read_response['data']['data']
            return 'api_key' in data and 'account_id' in data
        except hvac.exceptions.InvalidPath:
            return False
        except Exception as e:
            print(f"Error checking Vault configuration: {e}")
            return False

    def get_tradier_secrets(self):
        """Retrieves Tradier credentials from Vault."""
        if not self.client:
            return None
            
        try:
            read_response = self.client.secrets.kv.v2.read_secret_version(path='tradier')
            return read_response['data']['data']
        except Exception as e:
            print(f"Error reading from Vault: {e}")
            return None

    def set_tradier_secrets(self, api_key, account_id):
        """Writes Tradier credentials to Vault."""
        if not self.vault_token:
            return False, "Vault token not set. Please provide it."
            
        if not self.client:
             self._connect()
             if not self.client:
                 return False, "Failed to initialize Vault client. Check connection options."

        try:
            if not self.client.is_authenticated():
                 return False, "Vault client is not authenticated. Invalid token?"

            self.client.secrets.kv.v2.create_or_update_secret(
                path='tradier',
                secret={
                    'api_key': api_key,
                    'account_id': account_id
                }
            )
            return True, None
        except Exception as e:
            msg = f"Error writing to Vault: {str(e)}"
            print(msg)
            return False, msg

# Global instance
vault_service = VaultService()
