"""
Research Experiment Framework

Provides utilities for running comparative experiments between forecasting methods.
Includes statistical testing, results aggregation, and visualization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple
import json
from pathlib import Path

from services.ml_service import predict_next_days
from research.recursive_forecaster import RecursiveForecaster, train_recursive_forecaster
from services.tradier_service import get_raw_historical_data
from services.ml_evaluation import calculate_metrics, calculate_directional_accuracy
from utils.logger import logger


class ExperimentRunner:
    """Orchestrates comparison experiments between forecasting methods."""
    
    def __init__(self, output_dir='research/results'):
        """Initialize experiment runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def run_stock_comparison(self, ticker: str, train_end_date: str, test_days: int = 5):
        """
        Compare direct vs recursive forecasting for a single stock.
        
        Args:
            ticker: Stock symbol
            train_end_date: Date to split train/test (YYYY-MM-DD)
            test_days: Number of days to forecast
            
        Returns:
            Dict with comparison results
        """
        logger.info(f"Running comparison for {ticker}")
        
        # Fetch historical data
        df = get_raw_historical_data(ticker, '2y')
        if df.empty:
            return {'error': f'No data for {ticker}'}
        
        # Split train/test
        train_df = df[df.index <= train_end_date]
        test_df = df[df.index > train_end_date]
        
        if len(test_df) < test_days:
            return {'error': f'Insufficient test data for {ticker}'}
        
        # Get actual prices for first test_days
        actual_prices = test_df['Close'].iloc[:test_days].values
        
        results = {
            'ticker': ticker,
            'train_end': train_end_date,
            'test_days': test_days,
            'actual_prices': actual_prices.tolist()
        }
        
        # Method 1: Direct (existing)
        try:
            direct_result = predict_next_days(ticker, days=test_days)
            if 'error' not in direct_result:
                direct_preds = direct_result['predictions']
                results['direct'] = {
                    'predictions': direct_preds,
                    'mae': calculate_metrics(actual_prices, direct_preds)['mae'],
                    'rmse': calculate_metrics(actual_prices, direct_preds)['rmse'],
                    'directional_accuracy': calculate_directional_accuracy(actual_prices, direct_preds)
                }
            else:
                results['direct'] = {'error': direct_result['error']}
        except Exception as e:
            results['direct'] = {'error': str(e)}
        
        # Method 2: Recursive
        try:
            recursive_model = train_recursive_forecaster(train_df)
            recursive_preds = recursive_model.predict_recursive(train_df, days=test_days)
            
            if len(recursive_preds) == test_days:
                results['recursive'] = {
                    'predictions': recursive_preds,
                    'mae': calculate_metrics(actual_prices, recursive_preds)['mae'],
                    'rmse': calculate_metrics(actual_prices, recursive_preds)['rmse'],
                    'directional_accuracy': calculate_directional_accuracy(actual_prices, recursive_preds)
                }
            else:
                results['recursive'] = {'error': f'Only {len(recursive_preds)} predictions generated'}
        except Exception as e:
            results['recursive'] = {'error': str(e)}
        
        logger.info(f"Comparison complete for {ticker}")
        return results
    
    def run_multi_stock_experiment(self, tickers: List[str], n_trials: int = 10):
        """
        Run experiments across multiple stocks with multiple time periods.
        
        Args:
            tickers: List of stock symbols
            n_trials: Number of different time periods to test
            
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
            
            # Generate n_trials different test dates (spaced evenly)
            dates = df.index[:-30]  # Leave room for test period
            test_dates = dates[::len(dates)//n_trials][:n_trials]
            
            for test_date in test_dates:
                result = self.run_stock_comparison(
                    ticker,
                    test_date.strftime('%Y-%m-%d'),
                    test_days=5
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
        
        direct_maes = []
        recursive_maes = []
        direct_dir_acc = []
        recursive_dir_acc = []
        
        for result in self.results:
            if 'error' not in result.get('direct', {}) and 'error' not in result.get('recursive', {}):
                direct_maes.append(result['direct']['mae'])
                recursive_maes.append(result['recursive']['mae'])
                direct_dir_acc.append(result['direct']['directional_accuracy'])
                recursive_dir_acc.append(result['recursive']['directional_accuracy'])
        
        # Statistical tests
        if len(direct_maes) > 1:
            mae_ttest = stats.ttest_rel(direct_maes, recursive_maes)
            dir_ttest = stats.ttest_rel(direct_dir_acc, recursive_dir_acc)
        else:
            mae_ttest = (None, None)
            dir_ttest = (None, None)
        
        aggregated = {
            'n_experiments': len(self.results),
            'n_valid': len(direct_maes),
            'direct': {
                'mae_mean': np.mean(direct_maes),
                'mae_std': np.std(direct_maes),
                'dir_acc_mean': np.mean(direct_dir_acc),
                'dir_acc_std': np.std(direct_dir_acc)
            },
            'recursive': {
                'mae_mean': np.mean(recursive_maes),
                'mae_std': np.std(recursive_maes),
                'dir_acc_mean': np.mean(recursive_dir_acc),
                'dir_acc_std': np.std(recursive_dir_acc)
            },
            'statistical_tests': {
                'mae_ttest_statistic': float(mae_ttest[0]) if mae_ttest[0] is not None else None,
                'mae_ttest_pvalue': float(mae_ttest[1]) if mae_ttest[1] is not None else None,
                'dir_ttest_statistic': float(dir_ttest[0]) if dir_ttest[0] is not None else None,
                'dir_ttest_pvalue': float(dir_ttest[1]) if dir_ttest[1] is not None else None
            }
        }
        
        logger.info("="*80)
        logger.info("AGGREGATED RESULTS")
        logger.info("="*80)
        logger.info(f"Direct MAE: {aggregated['direct']['mae_mean']:.2f} ± {aggregated['direct']['mae_std']:.2f}")
        logger.info(f"Recursive MAE: {aggregated['recursive']['mae_mean']:.2f} ± {aggregated['recursive']['mae_std']:.2f}")
        logger.info(f"MAE t-test p-value: {aggregated['statistical_tests']['mae_ttest_pvalue']:.4f}")
        logger.info("="*80)
        
        return aggregated
    
    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'experiment_results_{timestamp}.json'
        
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
\caption{Comparison of Direct vs Recursive Forecasting Methods}
\label{tab:forecast_comparison}
\begin{tabular}{lcc}
\hline
Metric & Direct & Recursive \\
\hline
MAE (5-day) & $%.2f \pm %.2f$ & $%.2f \pm %.2f$ \\
Directional Acc. (\%%) & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ \\
\hline
\multicolumn{3}{l}{MAE paired t-test: $t=%.2f$, $p=%.4f$} \\
\hline
\end{tabular}
\end{table}
""" % (
            agg['direct']['mae_mean'], agg['direct']['mae_std'],
            agg['recursive']['mae_mean'], agg['recursive']['mae_std'],
            agg['direct']['dir_acc_mean'], agg['direct']['dir_acc_std'],
            agg['recursive']['dir_acc_mean'], agg['recursive']['dir_acc_std'],
            agg['statistical_tests']['mae_ttest_statistic'] or 0,
            agg['statistical_tests']['mae_ttest_pvalue'] or 1.0
        )
        
        # Save to file
        output_file = self.output_dir / 'results_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex)
        
        logger.info(f"LaTeX table saved to {output_file}")
        
        return latex


# SP100 stock list (subset for research)
SP100_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'VZ',
    'ADBE', 'NFLX', 'CMCSA', 'KO', 'NKE', 'MRK', 'PFE', 'T', 'INTC',
    'CSCO', 'ABT', 'WMT', 'CVX', 'XOM', 'BA', 'MCD', 'GS', 'IBM',
    'QCOM', 'CAT', 'MMM', 'HON', 'AMGN', 'AXP', 'TXN', 'LOW', 'UNP',
    'RTX', 'LMT', 'SPGI', 'GE', 'DHR'
]


def run_quick_experiment():
    """Run a quick experiment on 10 stocks for testing."""
    runner = ExperimentRunner()
    tickers = SP100_TICKERS[:10]  # First 10 stocks
    
    logger.info("Running quick experiment on 10 stocks")
    results = runner.run_multi_stock_experiment(tickers, n_trials=3)
    
    print("\n" + "="*80)
    print("QUICK EXPERIMENT RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    print("="*80)
    
    return results


def run_full_experiment():
    """Run full experiment for publication."""
    runner = ExperimentRunner()
    
    logger.info("Running FULL experiment on SP100")
    results = runner.run_multi_stock_experiment(SP100_TICKERS, n_trials=10)
    
    # Generate publication materials
    latex_table = runner.generate_latex_table()
    
    print("\n" + "="*80)
    print("FULL EXPERIMENT COMPLETE")
    print("="*80)
    print("Results saved to research/results/")
    print("\nLaTeX Table:")
    print(latex_table)
    print("="*80)
    
    return results
