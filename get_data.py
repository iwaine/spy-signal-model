"""Download SPY daily OHLCV data and save it as spy_daily.csv."""
import yfinance as yf

df = yf.download("SPY", start="2000-01-01", auto_adjust=False, progress=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
df.to_csv("spy_daily.csv", index=False)
print(f"Saved {len(df)} rows to spy_daily.csv")
