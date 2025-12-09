"""
Unit tests for ML features module.
Tests each feature engineering function independently.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.ml_features import (
    add_technical_indicators,
    add_volatility_features,
    add_volume_features,
    add_time_features,
    add_price_features,
    add_lag_features,
    prepare_features
)


class TestMLFeatures(unittest.TestCase):
    
    def setUp(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2021-01-01', periods=700, freq='D')
        dates = dates[dates.dayofweek < 5]  # Remove weekends
        
        # Generate synthetic price data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        
        self.sample_df = pd.DataFrame({
            'Open': close_prices + np.random.randn(len(dates)) * 0.5,
            'High': close_prices + np.abs(np.random.randn(len(dates)) * 1.5),
            'Low': close_prices - np.abs(np.random.randn(len(dates)) * 1.5),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, size=len(dates))
        }, index=dates)
    
    def test_technical_indicators_no_nan_at_end(self):
        """Test that technical indicators are calculated without NaN in final rows"""
        result = add_technical_indicators(self.sample_df)
        
        # Check that we have new columns
        expected_cols = ['RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'ATR']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Last 50 rows should not have NaN (after warmup period)
        self.assertFalse(result['RSI'].iloc[-50:].isna().any())
        self.assertFalse(result['MACD'].iloc[-50:].isna().any())
    
    def test_volatility_features(self):
        """Test volatility feature calculations"""
        result = add_volatility_features(self.sample_df)
        
        expected_cols = ['HV_20', 'HV_10', 'HV_50', 'Vol_Ratio']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Volatility should be positive
        self.assertTrue((result['HV_20'].dropna() > 0).all())
        
        # Vol ratio should make sense
        vol_ratio = result['Vol_Ratio'].dropna()
        self.assertTrue((vol_ratio > 0).all())
    
    def test_volume_features(self):
        """Test volume feature calculations"""
        result = add_volume_features(self.sample_df)
        
        expected_cols = ['Volume_Change', 'Rel_Volume_20', 'OBV']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # OBV should be cumulative
        self.assertTrue(len(result['OBV'].dropna()) > 0)
    
    def test_time_features(self):
        """Test time-based features"""
        result = add_time_features(self.sample_df)
        
        expected_cols = ['DayOfWeek', 'Month', 'DaysToFriday', 'IsMonday', 'IsFriday']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Day of week should be 0-4 (Mon-Fri)
        self.assertTrue((result['DayOfWeek'] >= 0).all())
        self.assertTrue((result['DayOfWeek'] <= 4).all())
        
        # Month should be 1-12
        self.assertTrue((result['Month'] >= 1).all())
        self.assertTrue((result['Month'] <= 12).all())
    
    def test_price_features(self):
        """Test price-based features"""
        result = add_price_features(self.sample_df)
        
        expected_cols = ['Distance_SMA_50', 'Momentum_5', 'ROC_10']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Distance from SMA should be small percentage
        dist_sma = result['Distance_SMA_50'].dropna()
        self.assertTrue((dist_sma.abs() < 0.5).all())  # Within 50%
    
    def test_lag_features(self):
        """Test lag feature creation"""
        result = add_lag_features(self.sample_df, lags=5)
        
        for i in range(1, 6):
            self.assertIn(f'Lag_{i}', result.columns)
        
        # Lag_1 should match previous Close
        self.assertEqual(result['Lag_1'].iloc[-1], result['Close'].iloc[-2])
    
    def test_prepare_features_integration(self):
        """Test the full feature preparation pipeline"""
        X, y, feature_names, scaler = prepare_features(self.sample_df, for_training=True)
        
        # Should return data
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(feature_names)
        
        # Should have 5 target columns
        self.assertEqual(y.shape[1], 5)
        
        # Features should have no NaN
        self.assertFalse(X.isna().any().any())
        
        # Targets should have no NaN
        self.assertFalse(y.isna().any().any())
        
        # Should have substantial features (>30)
        self.assertGreater(len(feature_names), 30)
        
        print(f"✓ Generated {len(feature_names)} features from {len(self.sample_df)} samples")
        print(f"✓ Training data shape: X={X.shape}, y={y.shape}")
    
    def test_prepare_features_prediction_mode(self):
        """Test feature preparation for prediction (no targets)"""
        X, y, feature_names, scaler = prepare_features(self.sample_df, for_training=False)
        
        # Should not create targets
        self.assertIsNone(y)
        
        # Should still have features
        self.assertIsNotNone(X)
        self.assertGreater(len(X), 0)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        small_df = self.sample_df.head(50)  # Too small for all features
        
        with self.assertRaises(ValueError):
            prepare_features(small_df, for_training=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
