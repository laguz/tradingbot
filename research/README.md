# Research Experiments

This directory contains code for academic research experiments comparing forecasting methods.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test (10 stocks, 3 trials) - ~10 minutes
python research/run_experiments.py --quick

# Run full experiment (50 stocks, 10 trials) - ~2-3 hours
python research/run_experiments.py --full
```

## Structure

```
research/
├── recursive_forecaster.py    # Recursive baseline implementation
├── experiment_framework.py    # Experiment orchestration & analysis
├── run_experiments.py         # CLI entry point
└── results/                   # Output directory (auto-created)
    ├── experiment_results_*.json  # Raw results
    └── results_table.tex         # LaTeX table for publication
```

## Experiments

### Experiment 1: Direct vs Recursive Forecasting

**Hypothesis**: Direct multi-day forecasting reduces cumulative error vs recursive approach.

**Method**:
1. Train both models on same historical data
2. Forecast 5 days ahead
3. Compare MAE, RMSE, directional accuracy
4. Statistical significance via paired t-test

**Expected Results**:
- Direct MAE (Day 5): ~$2.50
- Recursive MAE (Day 5): ~$3.80
- p-value < 0.001 (significant)

### Experiment 2: Error Propagation Analysis

Coming soon: Analyze how prediction error grows with forecast horizon.

## Usage Examples

### Quick Test
```python
from research.experiment_framework import run_quick_experiment

results = run_quick_experiment()
# Tests 10 stocks with 3 time periods each
```

### Custom Experiment
```bash
python research/run_experiments.py --tickers AAPL,MSFT,GOOGL,TSLA --trials 10
```

### Full Publication Experiment
```bash
# This will take 2-3 hours
python research/run_experiments.py --full
```

## Output

### JSON Results
```json
{
  "ticker": "AAPL",
  "train_end": "2023-06-01",
  "actual_prices": [175.43, 177.82, ...],
  "direct": {
    "predictions": [176.20, 178.10, ...],
    "mae": 2.45,
    "directional_accuracy": 68.5
  },
  "recursive": {
    "predictions": [176.15, 179.30, ...],
    "mae": 3.82,
    "directional_accuracy": 62.3
  }
}
```

### LaTeX Table
Automatically generated publication-ready table:

```latex
\begin{table}[h]
\centering
\caption{Comparison of Direct vs Recursive Forecasting Methods}
...
\end{table}
```

## For Publication

1. **Run full experiment**: `python research/run_experiments.py --full`
2. **Check results**: `research/results/experiment_results_*.json`
3. **Get LaTeX table**: `research/results/results_table.tex`
4. **Include in paper**: Copy table to your LaTeX document

## Notes

- All experiments use walk-forward validation (no lookahead bias)
- Random seed set to 42 for reproducibility
- Skips weekends automatically
- Logs all steps for debugging

## Citation

If you use this code in research, please cite:

```bibtex
@software{ml_trading_research,
  title={Direct Multi-Day Stock Forecasting Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tradingbot}
}
```
