"""
Strike Probability Experiment Framework

Research experiment to compare different methods for predicting option strike probabilities:
1. ML-based (Random Forest classifier)
2. Black-Scholes theoretical
3. Historical frequency baseline
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import json
from scipy import stats

from services.ml_optimization import predict_strike_probability
from research.black_scholes_model import black_scholes_probability, calculate_historical_volatility
from research.probability_metrics import calculate_brier_score, calculate_calibration_data, compare_probability_models
from services.tradier_service import get_raw_historical_data
from utils.logger import logger


class StrikeProbabilityExperiment:
    """Orchestrates strike probability comparison experiments."""
    
    def __init__(self, output_dir='research/results'):
        """Initialize experiment runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def calculate_historical_probability(self, df, strike_price, days_ahead, current_idx):
        """
        Calculate probability based on historical frequency.
        
        Args:
            df: Historical OHLCV DataFrame
            strike_price: Strike price
            days_ahead: Forecast horizon
            current_idx: Current position in DataFrame
            
        Returns:
            Probability based on historical frequency
        """
        # Look at all historical periods with similar conditions
        # Check how often price touched strike in next N days
        
        touches = []
        lookback = min(252, current_idx)  # Use up to 1 year of history
        
        for i in range(current_idx - lookback, current_idx):
            if i + days_ahead >= current_idx:
                continue
                
            future_highs = df['High'].iloc[i+1:i+1+days_ahead]
            future_lows = df['Low'].iloc[i+1:i+1+days_ahead]
            
            current_price = df['Close'].iloc[i]
            
            if strike_price > current_price:
                # Check upward touch
                touched = (future_highs >= strike_price).any()
            else:
                # Check downward touch
                touched = (future_lows <= strike_price).any()
            
            touches.append(1 if touched else 0)
        
        if len(touches) == 0:
            return 0.5  # Neutral if no data
        
        return np.mean(touches)
    
    def run_single_comparison(self, ticker: str, test_date: str, strike_offset: float = 0.05, 
                             days_to_expiration: int = 7):
        """
        Compare probability methods for a single stock/date/strike combination.
        
        Args:
            ticker: Stock symbol
            test_date: Date to test (YYYY-MM-DD)
            strike_offset: Strike as % away from current price (e.g., 0.05 = 5%)
            days_to_expiration: Days until expiration
            
        Returns:
            Dict with comparison results
        """
        logger.info(f"Running comparison for {ticker} on {test_date}")
        
        # Fetch historical data
        df = get_raw_historical_data(ticker, '2y')
        if df.empty:
            return {'error': f'No data for {ticker}'}
        
        # Find test date in data
        try:
            test_idx = df.index.get_loc(test_date)
        except KeyError:
            # Find nearest date
            test_idx = df.index.searchsorted(pd.to_datetime(test_date))
            if test_idx >= len(df):
                return {'error': f'Test date {test_date} beyond available data'}
        
        # Check if we have enough future data
        if test_idx + days_to_expiration >= len(df):
            return {'error': 'Insufficient future data for outcome'}
        
        current_price = float(df['Close'].iloc[test_idx])
        strike_price = current_price * (1 + strike_offset)
        
        # Get actual outcome
        future_highs = df['High'].iloc[test_idx+1:test_idx+1+days_to_expiration]
        future_lows = df['Low'].iloc[test_idx+1:test_idx+1+days_to_expiration]
        
        if strike_price > current_price:
            actual_touched = bool((future_highs >= strike_price).any())
        else:
            actual_touched = bool((future_lows <= strike_price).any())
        
        results = {
            'ticker': ticker,
            'test_date': test_date,
            'current_price': round(current_price, 2),
            'strike_price': round(strike_price, 2),
            'strike_offset': strike_offset,
            'days_to_expiration': days_to_expiration,
            'actual_outcome': 1 if actual_touched else 0
        }
        
        # Method 1: ML-Based
        try:
            # Use data up to test_idx
            df_train = df.iloc[:test_idx+1]
            ml_result = predict_strike_probability(ticker, strike_price, days_to_expiration)
            
            if 'probability' in ml_result:
                results['ml_probability'] = round(ml_result['probability'], 4)
            else:
                results['ml_probability'] = None
                results['ml_error'] = ml_result.get('error', 'Unknown error')
        except Exception as e:
            results['ml_probability'] = None
            results['ml_error'] = str(e)
        
        # Method 2: Black-Scholes
        try:
            historical_prices = df['Close'].iloc[:test_idx+1]
            volatility = calculate_historical_volatility(historical_prices)
            
            bs_prob = black_scholes_probability(
                current_price,
                strike_price,
                days_to_expiration,
                volatility
            )
            results['bs_probability'] = round(bs_prob, 4)
            results['bs_volatility'] = round(volatility, 4)
        except Exception as e:
            results['bs_probability'] = None
            results['bs_error'] = str(e)
        
        # Method 3: Historical Frequency
        try:
            hist_prob = self.calculate_historical_probability(
                df, strike_price, days_to_expiration, test_idx
            )
            results['historical_probability'] = round(hist_prob, 4)
        except Exception as e:
            results['historical_probability'] = None
            results['historical_error'] = str(e)
        
        logger.info(f"Comparison complete for {ticker}: Actual={actual_touched}, "
                   f"ML={results.get('ml_probability')}, BS={results.get('bs_probability')}, "
                   f"Hist={results.get('historical_probability')}")
        
        return results
    
    def run_multi_stock_experiment(self, tickers: List[str], n_trials: int = 5, 
                                   strike_offsets: List[float] = [0.03, 0.05, 0.07]):
        """
        Run experiments across multiple stocks, time periods, and strike prices.
        
        Args:
            tickers: List of stock symbols
            n_trials: Number of different time periods to test per stock
            strike_offsets: List of strike offsets to test (e.g., [0.03, 0.05, 0.07] = 3%, 5%, 7% OTM)
            
        Returns:
            Aggregated results
        """
        all_results = []
        
        for ticker in tickers:
            logger.info(f"Processing {ticker} ({tickers.index(ticker)+1}/{len(tickers)})")
            
            # Get data to determine valid test dates
            df = get_raw_historical_data(ticker, '2y')
            if df.empty:
                continue
            
            # Generate test dates (spaced evenly, leaving room for future data)
            dates = df.index[:-30]  # Leave 30 days for testing outcomes
            if len(dates) < n_trials:
                continue
                
            test_dates = dates[::len(dates)//(n_trials+1)][:n_trials]
            
            for test_date in test_dates:
                for strike_offset in strike_offsets:
                    result = self.run_single_comparison(
                        ticker,
                        test_date.strftime('%Y-%m-%d'),
                        strike_offset=strike_offset,
                        days_to_expiration=7
                    )
                    
                    if 'error' not in result:
                        all_results.append(result)
        
        self.results = all_results
        self._save_results()
        
        return self._aggregate_results()
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all experiments."""
        if not self.results:
            return {}
        
        # Extract probabilities and outcomes
        y_true = []
        ml_probs = []
        bs_probs = []
        hist_probs = []
        
        for result in self.results:
            if result.get('actual_outcome') is None:
                continue
                
            y_true.append(result['actual_outcome'])
            
            if result.get('ml_probability') is not None:
                ml_probs.append(result['ml_probability'])
            else:
                ml_probs.append(None)
            
            if result.get('bs_probability') is not None:
                bs_probs.append(result['bs_probability'])
            else:
                bs_probs.append(None)
            
            if result.get('historical_probability') is not None:
                hist_probs.append(result['historical_probability'])
            else:
                hist_probs.append(None)
        
        # Remove None values for each method
        def filter_valid(probs, outcomes):
            valid_pairs = [(p, o) for p, o in zip(probs, outcomes) if p is not None and not np.isnan(p)]
            if not valid_pairs:
                return [], []
            return zip(*valid_pairs)
        
        ml_probs_valid, ml_outcomes = filter_valid(ml_probs, y_true)
        bs_probs_valid, bs_outcomes = filter_valid(bs_probs, y_true)
        hist_probs_valid, hist_outcomes = filter_valid(hist_probs, y_true)
        
        ml_probs_valid = list(ml_probs_valid)
        ml_outcomes = list(ml_outcomes)
        bs_probs_valid = list(bs_probs_valid)
        bs_outcomes = list(bs_outcomes)
        hist_probs_valid = list(hist_probs_valid)
        hist_outcomes = list(hist_outcomes)
        
        aggregated = {
            'n_experiments': len(self.results),
            'n_valid_outcomes': len(y_true)
        }
        
        # Calculate Brier scores
        if len(ml_probs_valid) > 0:
            aggregated['ml'] = {
                'n_valid': len(ml_probs_valid),
                'brier_score': calculate_brier_score(ml_outcomes, ml_probs_valid),
                'mean_probability': round(float(np.mean(ml_probs_valid)), 4)
            }
        
        if len(bs_probs_valid) > 0:
            aggregated['black_scholes'] = {
                'n_valid': len(bs_probs_valid),
                'brier_score': calculate_brier_score(bs_outcomes, bs_probs_valid),
                'mean_probability': round(float(np.mean(bs_probs_valid)), 4)
            }
        
        if len(hist_probs_valid) > 0:
            aggregated['historical'] = {
                'n_valid': len(hist_probs_valid),
                'brier_score': calculate_brier_score(hist_outcomes, hist_probs_valid),
                'mean_probability': round(float(np.mean(hist_probs_valid)), 4)
            }
        
        # Statistical comparison
        if len(ml_probs_valid) > 5 and len(bs_probs_valid) > 5:
            # Paired comparison if same test cases
            ml_brier = aggregated['ml']['brier_score']
            bs_brier = aggregated['black_scholes']['brier_score']
            
            aggregated['comparison'] = {
                'ml_vs_bs_improvement': round((bs_brier - ml_brier) / bs_brier * 100, 2) if bs_brier > 0 else None
            }
        
        logger.info("="*80)
        logger.info("AGGREGATED RESULTS")
        logger.info("="*80)
        if 'ml' in aggregated:
            logger.info(f"ML Brier Score: {aggregated['ml']['brier_score']:.4f}")
        if 'black_scholes' in aggregated:
            logger.info(f"Black-Scholes Brier Score: {aggregated['black_scholes']['brier_score']:.4f}")
        if 'historical' in aggregated:
            logger.info(f"Historical Brier Score: {aggregated['historical']['brier_score']:.4f}")
        logger.info("="*80)
        
        return aggregated
    
    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'strike_prob_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for publication.
        
        Returns:
            LaTeX table string
        """
        agg = self._aggregate_results()
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of Strike Probability Prediction Methods}
\label{tab:strike_probability_comparison}
\begin{tabular}{lcc}
\hline
Method & Brier Score & N Valid \\
\hline
"""
        
        if 'ml' in agg:
            latex += f"ML-Based & ${agg['ml']['brier_score']:.4f}$ & {agg['ml']['n_valid']} \\\\\n"
        
        if 'black_scholes' in agg:
            latex += f"Black-Scholes & ${agg['black_scholes']['brier_score']:.4f}$ & {agg['black_scholes']['n_valid']} \\\\\n"
        
        if 'historical' in agg:
            latex += f"Historical Freq. & ${agg['historical']['brier_score']:.4f}$ & {agg['historical']['n_valid']} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        
        # Save to file
        output_file = self.output_dir / 'strike_probability_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex)
        
        logger.info(f"LaTeX table saved to {output_file}")
        
        return latex


# Convenience functions for quick experiments

SP100_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'VZ',
    'ADBE', 'NFLX', 'CMCSA', 'KO', 'NKE', 'MRK', 'PFE', 'T', 'INTC',
    'CSCO', 'ABT', 'WMT', 'CVX', 'XOM', 'BA', 'MCD', 'GS', 'IBM',
    'QCOM', 'CAT', 'MMM', 'HON', 'AMGN', 'AXP', 'TXN', 'LOW', 'UNP',
    'RTX', 'LMT', 'SPGI', 'GE', 'DHR'
]


def run_quick_strike_experiment():
    """Run a quick experiment on 5 stocks for testing."""
    runner = StrikeProbabilityExperiment()
    tickers = SP100_TICKERS[:5]  # First 5 stocks
    
    logger.info("Running quick strike probability experiment on 5 stocks")
    results = runner.run_multi_stock_experiment(
        tickers, 
        n_trials=3,
        strike_offsets=[0.05]  # Just 5% OTM
    )
    
    print("\n" + "="*80)
    print("QUICK STRIKE PROBABILITY EXPERIMENT RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    print("="*80)
    
    return results


def run_full_strike_experiment():
    """Run full experiment for publication."""
    runner = StrikeProbabilityExperiment()
    
    logger.info("Running FULL strike probability experiment on SP100")
    results = runner.run_multi_stock_experiment(
        SP100_TICKERS[:25],  # Use 25 stocks for reasonable runtime
        n_trials=5,
        strike_offsets=[0.03, 0.05, 0.07]  # Multiple strike distances
    )
    
    # Generate publication materials
    latex_table = runner.generate_latex_table()
    
    print("\n" + "="*80)
    print("FULL STRIKE PROBABILITY EXPERIMENT COMPLETE")
    print("="*80)
    print("Results saved to research/results/")
    print("\nLaTeX Table:")
    print(latex_table)
    print("="*80)
    
    return results
