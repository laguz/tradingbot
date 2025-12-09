"""
Research Entry Point

Run this script to execute research experiments.

Usage:
    # Quick test (10 stocks, 3 trials each)
    python research/run_experiments.py --quick
    
    # Full experiment (50 stocks, 10 trials each)
    python research/run_experiments.py --full
    
    # Custom experiment
    python research/run_experiments.py --tickers AAPL,MSFT,GOOGL --trials 5
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.experiment_framework import ExperimentRunner, run_quick_experiment, run_full_experiment, SP100_TICKERS
from research.strike_probability_experiment import run_quick_strike_experiment, run_full_strike_experiment
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description='Run forecasting research experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick test (10 stocks, 3 trials) - Phase 1')
    parser.add_argument('--full', action='store_true', help='Run full experiment (50 stocks, 10 trials) - Phase 1')
    parser.add_argument('--phase2-quick', action='store_true', help='Run quick strike probability test - Phase 2')
    parser.add_argument('--phase2-full', action='store_true', help='Run full strike probability experiment - Phase 2')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per stock')
    parser.add_argument('--output-dir', type=str, default='research/results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.quick:
        logger.info("Starting QUICK experiment - Phase 1: Direct vs Recursive")
        results = run_quick_experiment()
        
    elif args.full:
        logger.info("Starting FULL experiment - Phase 1: Direct vs Recursive (this may take several hours)")
        results = run_full_experiment()
    
    elif args.phase2_quick:
        logger.info("Starting QUICK experiment - Phase 2: Strike Probability")
        results = run_quick_strike_experiment()
    
    elif args.phase2_full:
        logger.info("Starting FULL experiment - Phase 2: Strike Probability (this may take 1-2 hours)")
        results = run_full_strike_experiment()
        
    elif args.tickers:
        logger.info(f"Starting CUSTOM experiment")
        tickers = args.tickers.split(',')
        runner = ExperimentRunner(output_dir=args.output_dir)
        results = runner.run_multi_stock_experiment(tickers, n_trials=args.trials)
        runner.generate_latex_table()
        
    else:
        print("Please specify --quick, --full, --phase2-quick, --phase2-full, or --tickers")
        parser.print_help()
        return
    
    logger.info("Experiment complete!")


if __name__ == '__main__':
    main()
