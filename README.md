# SPY Signal Model (learning project)

Machine learning pipeline that estimates the probability of a meaningful upward move in SPY over the next 3 trading days, filters signals by calibrated confidence, and backtests them against buy-and-hold.

> **Attribution.** This repository is a personal learning project adapted from
> **trading-model** (link to original repository) by **Lera Andronova** (MIT License).
> The original copyright notice is kept in `LICENSE`. See *My adaptations* below for what differs from the original.

## Pipeline

1. Load SPY daily OHLCV data (`get_data.py`, via `yfinance`)
2. Feature engineering: multi-period returns, moving-average gaps, rolling volatility, ATR, RSI (Wilder smoothing), volume z-score, long-term regime, 20-day range position
3. Target: 3-day forward return above a 0.4% hurdle
4. Chronological split (70% train / 15% validation / 15% test), no shuffling
5. `HistGradientBoostingClassifier` (scikit-learn)
6. Probability calibration with isotonic regression, fitted on the validation set
7. Signal filtering at a confidence threshold (default 0.60)
8. Backtest with a transaction-cost assumption (0.05% per trade), compared with buy-and-hold
9. Export of signals, backtest results, per-bucket precision and a JSON summary

## Metrics reported

ROC AUC, Brier score, classification report, confusion matrix, precision by confidence bucket, strategy return, win rate, average trade return, maximum drawdown.

## Usage

```bash
pip install -r requirements.txt
cd src
python get_data.py      # creates spy_daily.csv
python train_model.py   # writes results to outputs/
```

Outputs: `paper_trading_signals.csv`, `backtest_results.csv`, `confidence_bucket_evaluation.csv`, `model_summary.json`, and three charts (equity curve, precision by bucket, probability distribution).

## My adaptations

<!-- List here only what you actually changed, for example:
- added get_data.py to download the data
- tested other hurdles / thresholds
- added features or another model -->

## Limitations

Single asset, simple backtest (no slippage or position sizing), no live trading. Educational use only; not financial advice.

## License

MIT, see `LICENSE`.
