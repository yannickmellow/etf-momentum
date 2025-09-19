# main.py
import pandas as pd
import yfinance as yf
import numpy as np

# ---------------------------
# 1. Load ETF universe
# ---------------------------
ETF_FILE = "etf_list.csv"

df = pd.read_csv(ETF_FILE)
df.columns = [c.lower() for c in df.columns]

if "ticker" in df.columns:
    tickers = df["ticker"].dropna().unique().tolist()
else:
    tickers = df.iloc[:, 0].dropna().unique().tolist()

print(f"📥 Loaded {len(tickers)} tickers from {ETF_FILE}")

# ---------------------------
# 2. Parameters
# ---------------------------
lookbacks = {
    "12m": 252,
    "6m": 126,
    "1m": 21,
}
top_n = 10
min_price = 5
min_volume = 100_000

# ---------------------------
# 3. Fetch data
# ---------------------------
def get_price_history(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception as e:
        print(f"⚠️ Failed to fetch {ticker}: {e}")
        return None

price_data = {}
for t in tickers:
    df_price = get_price_history(t, period="1y")
    if df_price is not None:
        price_data[t] = df_price

print(f"✅ Got price history for {len(price_data)} ETFs")

# ---------------------------
# 4. Momentum scoring
# ---------------------------
def compute_score(df):
    if "Close" not in df.columns:
        return None, {}
    close = df["Close"]

    scores = {}
    for label, lb in lookbacks.items():
        if len(close) < lb:
            scores[label] = np.nan
        else:
            ret = close.iloc[-1] / close.iloc[-lb] - 1
            if label == "1m":  # invert short-term return
                ret *= -1
            scores[label] = ret

    composite = (
        0.5 * scores.get("12m", 0)
        + 0.3 * scores.get("6m", 0)
        + 0.2 * scores.get("1m", 0)
    )
    return composite, scores

results = []
for ticker, df_price in price_data.items():
    composite, scores = compute_score(df_price)
    if composite is None:
        continue

    last_price = df_price["Close"].iloc[-1]
    avg_vol = df_price["Volume"].tail(21).mean()

    if last_price < min_price or avg_vol < min_volume:
        continue

    results.append({
        "ticker": ticker,
        "score": composite,
        "12m": scores.get("12m", np.nan),
        "6m": scores.get("6m", np.nan),
        "1m": scores.get("1m", np.nan),
        "last_price": last_price,
        "avg_volume": avg_vol,
    })

# ---------------------------
# 5. Rank and output
# ---------------------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("score", ascending=False).reset_index(drop=True)

print("\n🏆 Top ETFs by momentum:")
print(df_results.head(top_n))

df_results.to_csv("screener_results.csv", index=False)
print("\n📂 Saved full results to screener_results.csv")
