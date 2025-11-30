import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from services.tradier_service import find_support_resistance, calculate_smart_strikes

class TestTradingLogic(unittest.TestCase):

    def test_find_support_resistance_window(self):
        # Create data with 30 points
        # Local min at index 15 (value 100), others 110
        data = [110] * 30
        data[15] = 100
        df = pd.DataFrame({'Close': data})
        
        # With window=20, it needs 20 points on each side. 
        # Index 15 needs 15-20 to 15+20. 
        # Range is range(20, 10). Empty.
        # So it should return empty lists.
        support, resistance = find_support_resistance(df, window=20)
        self.assertEqual(support, [])
        
        # If we have enough data, say 50 points, and min at 25.
        data = [110] * 50
        data[25] = 100
        df = pd.DataFrame({'Close': data})
        support, resistance = find_support_resistance(df, window=20)
        self.assertEqual(support, [100])
        print("test_find_support_resistance_window passed")

    @patch('services.tradier_service.get_historical_data')
    @patch('services.tradier_service.get_option_chain')
    def test_calculate_smart_strikes_safety_buffer(self, mock_chain, mock_history):
        # Setup
        current_price = 100.0
        support_level = 99.5 # Very close to price
        
        # Mock History
        mock_history.return_value = {
            'support': [support_level],
            'resistance': [],
            'data': [current_price] # Last price
        }
        
        # Mock Chain
        # Strikes: 90, 95, 99, 100
        mock_chain.return_value = [
            {'strike': 90.0, 'option_type': 'put'},
            {'strike': 95.0, 'option_type': 'put'},
            {'strike': 99.0, 'option_type': 'put'},
            {'strike': 99.5, 'option_type': 'put'},
            {'strike': 100.0, 'option_type': 'put'}
        ]
        
        # Test Put Credit Spread
        # Target should be min(99.5, 100 * 0.99 = 99.0) = 99.0
        # So it should pick strike 99.0, not 99.5
        
        short, long_strike, trigger = calculate_smart_strikes('TEST', '2023-01-01', 'credit', 'put', 5)
        
        print(f"Short: {short}, Long: {long_strike}, Trigger: {trigger}")
        
        self.assertEqual(short, 99.0)
        self.assertEqual(trigger, 99.5) # Trigger was the support level
        print("test_calculate_smart_strikes_safety_buffer passed")

if __name__ == '__main__':
    unittest.main()
