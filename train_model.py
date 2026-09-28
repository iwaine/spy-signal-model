"""
COMPLETE SPY SIGNAL MODEL

Pipeline:
  1. Load SPY daily data
  2. Engineer market features
  3. Define 3-day forward-return target
  4. Chronological train / validation / test split
  5. Train HistGradientBoostingClassifier
  6. Calibrate probabilities with isotonic regression
  7. Filter signals by confidence threshold
  8. Backtest model signals
  9. Save prediction outputs
  10. Prepare paper-trading signal file

"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_PATH = "spy_daily.csv"
OUTPUT_DIR = "outputs"

HURDLE = 0.004          # 0.4% forward-return hurdle
FWD = 3                 # 3 trading days forward
CONFIDENCE_THRESHOLD = 0.60
INITIAL_CAPITAL = 10_000
TRADE_SIZE = 1.0        # 1.0 = fully invested on signal days
TRANSACTION_COST = 0.0005  # 0.05% per trade assumption

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

required_cols = {"date", "open", "high", "low", "close", "volume"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Loaded {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def wilder_avg(series: pd.Series, n: int) -> pd.Series:
    out = pd.Series(np.nan, index=series.index)
    out.iloc[n - 1] = series.iloc[:n].mean()
    alpha = 1.0 / n

    for i in range(n, len(series)):
        out.iloc[i] = out.iloc[i - 1] + alpha * (series.iloc[i] - out.iloc[i - 1])

    return out


def compute_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()

    gain = wilder_avg(delta.clip(lower=0).fillna(0), n)
    loss = wilder_avg((-delta.clip(upper=0)).fillna(0), n)

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi[(gain == 0) & (loss == 0)] = 50.0
    rsi[(gain > 0) & (loss == 0)] = 100.0

    return rsi


def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return wilder_avg(tr, n)


close = df["close"]

# Returns
for n in [1, 2, 3, 5, 10, 20]:
    df[f"ret{n}"] = close.pct_change(n)

# Moving-average gaps
for n in [10, 20, 50, 200]:
    ma = close.rolling(n).mean()
    df[f"sma{n}_gap"] = (close / ma) - 1

df["sma10_sma20_gap"] = (close.rolling(10).mean() / close.rolling(20).mean()) - 1
df["sma20_sma50_gap"] = (close.rolling(20).mean() / close.rolling(50).mean()) - 1
df["sma50_sma200_gap"] = (close.rolling(50).mean() / close.rolling(200).mean()) - 1

# Volatility
for n in [5, 10, 20]:
    df[f"vol{n}"] = df["ret1"].rolling(n).std()

df["vol_ratio"] = df["vol5"] / df["vol20"]

# ATR
df["atr14_ratio"] = compute_atr(df, 14) / close

# RSI
df["rsi14"] = compute_rsi(close, 14)

# Volume z-score
df["volume_z20"] = (
    (df["volume"] - df["volume"].rolling(20).mean())
    / df["volume"].rolling(20).std()
)

# Regime
df["regime_long_ok"] = (close > close.rolling(200).mean()).astype(int)

# Recent range positioning
df["hl_position_20"] = (
    (close - df["low"].rolling(20).min())
    / (df["high"].rolling(20).max() - df["low"].rolling(20).min() + 1e-9)
)

print("Features computed.")


# ─────────────────────────────────────────────
# 3. TARGET DEFINITION
# ─────────────────────────────────────────────

df["fwd_ret"] = close.shift(-FWD) / close - 1

df["signal"] = 0
df.loc[df["fwd_ret"] > HURDLE, "signal"] = 1
df.loc[df["fwd_ret"] < -HURDLE, "signal"] = -1

# Long model: 1 = meaningful upside setup, 0 = not a long setup
df["target_long"] = (df["signal"] == 1).astype(int)

signal_counts = df["signal"].value_counts().sort_index()

print(f"\nTarget distribution | hurdle={HURDLE * 100:.1f}% | fwd={FWD} days")
print(f"Strong UP   (+1): {signal_counts.get(1, 0):>5}")
print(f"Noise        (0): {signal_counts.get(0, 0):>5}")
print(f"Strong DOWN (-1): {signal_counts.get(-1, 0):>5}")


# ─────────────────────────────────────────────
# 4. FEATURE LIST AND CLEAN DATA
# ─────────────────────────────────────────────

FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
    "sma10_gap", "sma20_gap", "sma50_gap", "sma200_gap",
    "sma10_sma20_gap", "sma20_sma50_gap", "sma50_sma200_gap",
    "vol5", "vol10", "vol20", "vol_ratio",
    "atr14_ratio",
    "rsi14",
    "volume_z20",
    "regime_long_ok",
    "hl_position_20",
]

df_clean = df.dropna(subset=FEATURES + ["target_long", "fwd_ret"]).copy()
df_clean = df_clean.iloc[:-FWD]

print(f"\nClean rows: {len(df_clean)}")


# ─────────────────────────────────────────────
# 5. CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────

n = len(df_clean)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df_clean.iloc[:train_end].copy()
val = df_clean.iloc[train_end:val_end].copy()
test = df_clean.iloc[val_end:].copy()

X_train = train[FEATURES].values
y_train = train["target_long"].values

X_val = val[FEATURES].values
y_val = val["target_long"].values

X_test = test[FEATURES].values
y_test = test["target_long"].values

print(f"\nSplit sizes → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Train period: {train['date'].min().date()} → {train['date'].max().date()}")
print(f"Val period:   {val['date'].min().date()} → {val['date'].max().date()}")
print(f"Test period:  {test['date'].min().date()} → {test['date'].max().date()}")


# ─────────────────────────────────────────────
# 6. TRAIN MODEL
# ─────────────────────────────────────────────

model = HistGradientBoostingClassifier(
    max_iter=400,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=40,
    l2_regularization=0.1,
    random_state=42,
)

model.fit(X_train, y_train)

raw_val_probs = model.predict_proba(X_val)[:, 1]
raw_test_probs = model.predict_proba(X_test)[:, 1]

raw_val_auc = roc_auc_score(y_val, raw_val_probs)
raw_test_auc = roc_auc_score(y_test, raw_test_probs)

print(f"\nRaw model validation AUC: {raw_val_auc:.4f}")
print(f"Raw model test AUC:       {raw_test_auc:.4f}")


# ─────────────────────────────────────────────
# 7. PROBABILITY CALIBRATION
# ─────────────────────────────────────────────

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_val_probs, y_val)

cal_val_probs = calibrator.transform(raw_val_probs)
cal_test_probs = calibrator.transform(raw_test_probs)

val_brier = brier_score_loss(y_val, cal_val_probs)
test_brier = brier_score_loss(y_test, cal_test_probs)

print(f"\nCalibrated validation Brier score: {val_brier:.4f}")
print(f"Calibrated test Brier score:       {test_brier:.4f}")


# ─────────────────────────────────────────────
# 8. SIGNAL FILTERING
# ─────────────────────────────────────────────

test_results = test.copy()
test_results["raw_prob_long"] = raw_test_probs
test_results["calibrated_prob_long"] = cal_test_probs

test_results["model_signal"] = 0
test_results.loc[
    test_results["calibrated_prob_long"] >= CONFIDENCE_THRESHOLD,
    "model_signal"
] = 1

signal_days = test_results[test_results["model_signal"] == 1].copy()

print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
print(f"Signal days in test set: {len(signal_days)} / {len(test_results)}")

if len(signal_days) > 0:
    signal_precision = signal_days["target_long"].mean()
    avg_signal_fwd_ret = signal_days["fwd_ret"].mean()

    print(f"Signal precision: {signal_precision:.3f}")
    print(f"Average forward return on signal days: {avg_signal_fwd_ret * 100:.3f}%")
else:
    print("No signals fired at this confidence threshold.")


# ─────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────

test_pred_default = (cal_test_probs >= 0.50).astype(int)
test_pred_filtered = test_results["model_signal"].values

print("\nDefault 0.50 Threshold Classification Report:")
print(classification_report(y_test, test_pred_default))

print("\nFiltered Signal Classification Report:")
print(classification_report(y_test, test_pred_filtered))

print("\nConfusion Matrix at 0.50 Threshold:")
print(confusion_matrix(y_test, test_pred_default))


# Precision by confidence bucket
test_results["confidence_bucket"] = pd.cut(
    test_results["calibrated_prob_long"],
    bins=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    labels=["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"],
    include_lowest=True,
)

bucket_eval = (
    test_results
    .groupby("confidence_bucket")
    .agg(
        count=("target_long", "size"),
        precision=("target_long", "mean"),
        avg_fwd_ret=("fwd_ret", "mean"),
        avg_probability=("calibrated_prob_long", "mean"),
    )
    .reset_index()
)

print("\nPrecision by confidence bucket:")
print(bucket_eval)


# ─────────────────────────────────────────────
# 10. BACKTEST
# ─────────────────────────────────────────────

backtest = test_results.copy()

# Strategy return only applies on signal days.
# If model_signal = 1, take the 3-day forward return.
# Subtract estimated transaction cost.
backtest["strategy_ret"] = 0.0
backtest.loc[backtest["model_signal"] == 1, "strategy_ret"] = (
    backtest.loc[backtest["model_signal"] == 1, "fwd_ret"] * TRADE_SIZE
    - TRANSACTION_COST
)

backtest["buy_hold_ret"] = backtest["close"].pct_change().fillna(0)

backtest["strategy_equity"] = INITIAL_CAPITAL * (1 + backtest["strategy_ret"]).cumprod()
backtest["buy_hold_equity"] = INITIAL_CAPITAL * (1 + backtest["buy_hold_ret"]).cumprod()

total_strategy_return = backtest["strategy_equity"].iloc[-1] / INITIAL_CAPITAL - 1
total_buy_hold_return = backtest["buy_hold_equity"].iloc[-1] / INITIAL_CAPITAL - 1

num_trades = int(backtest["model_signal"].sum())

if num_trades > 0:
    win_rate = (backtest.loc[backtest["model_signal"] == 1, "strategy_ret"] > 0).mean()
    avg_trade_return = backtest.loc[backtest["model_signal"] == 1, "strategy_ret"].mean()
else:
    win_rate = np.nan
    avg_trade_return = np.nan

running_max = backtest["strategy_equity"].cummax()
drawdown = backtest["strategy_equity"] / running_max - 1
max_drawdown = drawdown.min()

print("\nBacktest Summary:")
print(f"Initial capital:          ${INITIAL_CAPITAL:,.2f}")
print(f"Number of trades:         {num_trades}")
print(f"Strategy return:          {total_strategy_return * 100:.2f}%")
print(f"Buy and hold return:      {total_buy_hold_return * 100:.2f}%")
print(f"Win rate:                 {win_rate * 100:.2f}%" if not np.isnan(win_rate) else "Win rate:                 N/A")
print(f"Average trade return:     {avg_trade_return * 100:.3f}%" if not np.isnan(avg_trade_return) else "Average trade return:     N/A")
print(f"Max drawdown:             {max_drawdown * 100:.2f}%")


# ─────────────────────────────────────────────
# 11. PAPER-TRADING SIGNAL OUTPUT
# ─────────────────────────────────────────────

paper_signals = test_results[
    [
        "date",
        "close",
        "fwd_ret",
        "target_long",
        "raw_prob_long",
        "calibrated_prob_long",
        "model_signal",
    ]
].copy()

paper_signals["action"] = np.where(
    paper_signals["model_signal"] == 1,
    "WATCH_LONG",
    "NO_TRADE",
)

paper_signals["confidence_threshold"] = CONFIDENCE_THRESHOLD

paper_signals.to_csv(os.path.join(OUTPUT_DIR, "paper_trading_signals.csv"), index=False)
backtest.to_csv(os.path.join(OUTPUT_DIR, "backtest_results.csv"), index=False)
bucket_eval.to_csv(os.path.join(OUTPUT_DIR, "confidence_bucket_evaluation.csv"), index=False)

summary = {
    "rows_loaded": int(len(df)),
    "clean_rows": int(len(df_clean)),
    "train_rows": int(len(train)),
    "validation_rows": int(len(val)),
    "test_rows": int(len(test)),
    "hurdle": HURDLE,
    "forward_days": FWD,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "raw_validation_auc": float(raw_val_auc),
    "raw_test_auc": float(raw_test_auc),
    "validation_brier_score": float(val_brier),
    "test_brier_score": float(test_brier),
    "num_trades": int(num_trades),
    "strategy_return": float(total_strategy_return),
    "buy_hold_return": float(total_buy_hold_return),
    "win_rate": None if np.isnan(win_rate) else float(win_rate),
    "average_trade_return": None if np.isnan(avg_trade_return) else float(avg_trade_return),
    "max_drawdown": float(max_drawdown),
}

with open(os.path.join(OUTPUT_DIR, "model_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print(f"\nSaved outputs to: {OUTPUT_DIR}/")
print("Saved files:")
print("- paper_trading_signals.csv")
print("- backtest_results.csv")
print("- confidence_bucket_evaluation.csv")
print("- model_summary.json")


# ─────────────────────────────────────────────
# 12. CHARTS
# ─────────────────────────────────────────────

plt.figure(figsize=(10, 5))
plt.plot(backtest["date"], backtest["strategy_equity"], label="Model Strategy")
plt.plot(backtest["date"], backtest["buy_hold_equity"], label="Buy & Hold")
plt.title("Strategy Equity Curve vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"))
plt.show()


plt.figure(figsize=(8, 5))
plt.bar(bucket_eval["confidence_bucket"].astype(str), bucket_eval["precision"])
plt.title("Precision by Confidence Bucket")
plt.xlabel("Confidence Bucket")
plt.ylabel("Precision")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_by_confidence_bucket.png"))
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(test_results["calibrated_prob_long"], bins=20)
plt.title("Distribution of Calibrated Long Probabilities")
plt.xlabel("Calibrated Probability")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "probability_distribution.png"))
plt.show()


print("\nModel pipeline complete.")
