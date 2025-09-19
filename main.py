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
max_52w_discount = 0.2  # ETF must be within 20% of 52-week high
max_corr = 0.7  # max correlation allowed between selected ETFs

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
# 4. Momentum scoring with 52-week high & volatility
# ---------------------------
def compute_score(df):
    if "Close" not in df.columns:
        return None, {}

    close = df["Close"]
    # Ensure it's a single Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    scores = {}
    for label, lb in lookbacks.items():
        if len(close) < lb:
            scores[label] = np.nan
        else:
            ret = close.iloc[-1] / close.iloc[-lb] - 1
            if label == "1m":
                ret *= -1
            scores[label] = ret

    composite = (
        0.5 * scores.get("12m", 0)
        + 0.3 * scores.get("6m", 0)
        + 0.2 * scores.get("1m", 0)
    )

    # 52-week high filter
    high_52w = float(close[-252:].max()) if len(close) >= 252 else float(close.max())
    last_close = float(close.iloc[-1])
    pct_from_high = (high_52w - last_close) / high_52w
    if pct_from_high > max_52w_discount:
        return None, {}

    # Volatility adjustment
    vol = close[-21:].std() / close[-21:].mean() if len(close) >= 21 else 0
    composite_adj = composite / (1 + vol)

    return composite_adj, scores

# ---------------------------
# 5. Apply filters & compute scores
# ---------------------------
candidates = {}
for ticker, df_price in price_data.items():
    score_adj, scores = compute_score(df_price)
    if score_adj is None:
        continue

    last_price = float(df_price["Close"].iloc[-1])
    avg_vol = float(df_price["Volume"].tail(21).mean())

    if pd.isna(last_price) or pd.isna(avg_vol):
        continue
    if last_price < min_price or avg_vol < min_volume:
        continue

    candidates[ticker] = {
        "score": score_adj,
        "12m": scores.get("12m", np.nan),
        "6m": scores.get("6m", np.nan),
        "1m": scores.get("1m", np.nan),
        "last_price": last_price,
        "avg_volume": avg_vol,
        "returns": df_price["Close"].pct_change().fillna(0),  # daily returns
    }

print(f"✅ {len(candidates)} ETFs passed all filters before correlation check")

# ---------------------------
# 6. Select top ETFs with correlation filter
# ---------------------------
selected = []
selected_returns = pd.DataFrame()
sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)

for ticker, info in sorted_candidates:
    if len(selected) >= top_n:
        break

    if not selected:
        selected.append((ticker, info))
        selected_returns[ticker] = info["returns"]
        continue

    # Correlation check
    new_ret = info["returns"]
    corr_with_selected = selected_returns.corrwith(new_ret).max()
    if corr_with_selected > max_corr:
        continue

    selected.append((ticker, info))
    selected_returns[ticker] = new_ret

# ---------------------------
# 7. Output final results
# ---------------------------
df_results = pd.DataFrame([{
    "ticker": t,
    "score": i["score"],
    "12m": i["12m"],
    "6m": i["6m"],
    "1m": i["1m"],
    "last_price": i["last_price"],
    "avg_volume": i["avg_volume"]
} for t, i in selected])

print("\n🏆 Top ETFs after correlation filter:")
print(df_results)

df_results.to_csv("screener_results.csv", index=False)
print("\n📂 Saved final results to screener_results.csv")
