import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from services.ml_service import predict_next_days

class TestMLService(unittest.TestCase):

    @patch('services.ml_service.get_historical_data')
    def test_predict_next_days(self, mock_history):
        # Create synthetic data: Linear trend from 100 to 200 over 100 days
        data_len = 100
        prices = np.linspace(100, 200, data_len).tolist()
        dates = [f"2023-01-{i:02d}" for i in range(1, data_len+1)] # Fake dates
        
        # Mock return value (Chart.js format as expected by service)
        mock_history.return_value = {
            'labels': dates,
            'data': prices
        }
        
        # Call prediction
        result = predict_next_days('TEST', days=5)
        
        print(f"Predictions: {result['predictions']}")
        
        # Assertions
        self.assertEqual(len(result['predictions']), 5)
        self.assertEqual(result['ticker'], 'TEST')
        
        # Check if predictions are reasonable (should be > 200 given the trend)
        # Random Forest might not extrapolate perfectly, but should be close to recent values
        first_pred = result['predictions'][0]
        self.assertTrue(first_pred > 190, f"Prediction {first_pred} should be near last close 200")
        
        # Check if we have no NaNs
        self.assertFalse(np.isnan(result['predictions']).any())

if __name__ == '__main__':
    unittest.main()
