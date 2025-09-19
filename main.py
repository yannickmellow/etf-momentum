import pandas as pd
import yfinance as yf
import warnings
import os

# --- Suppress yfinance FutureWarning ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Paths ---
ETF_CSV = "etf_list.csv"
OUTPUT_HTML = "docs/index.html"

# --- Ensure docs folder exists ---
os.makedirs("docs", exist_ok=True)

# --- Load ETF list ---
etf_df = pd.read_csv(ETF_CSV)
print(f"Loaded {len(etf_df)} ETFs from {ETF_CSV}")

# --- Function to fetch adjusted close safely ---
def fetch_prices(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            print(f"⚠️ No data returned for {ticker}")
            return None
        return data["Close"]  # using Close because auto_adjust=True
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {e}")
        return None

# --- Compute 1-year momentum ---
etf_momentum = []
for idx, row in etf_df.iterrows():
    ticker = row['Ticker']
    name = row['Name']
    prices = fetch_prices(ticker)
    if prices is None or len(prices) < 2:
        continue
    momentum = (prices[-1] / prices[0]) - 1  # simple 1-year return
    etf_momentum.append({"Ticker": ticker, "Name": name, "Momentum": momentum})
    print(f"{name} ({ticker}) -> Momentum: {momentum:.2%} | {len(prices)} data points")

# --- Select top 10 ---
top_etfs = sorted(etf_momentum, key=lambda x: x["Momentum"], reverse=True)[:10]
print("\nTop 10 ETFs by 1-year momentum:")
for etf in top_etfs:
    print(f"{etf['Name']} ({etf['Ticker']}): {etf['Momentum']:.2%}")

# --- Generate HTML ---
html_content = "<html><head><title>ETF Momentum Top 10</title></head><body>"
html_content += "<h1>Top 10 ETFs by 1-Year Momentum</h1>"
html_content += "<table border='1' cellpadding='5' cellspacing='0'>"
html_content += "<tr><th>Rank</th><th>ETF Name</th><th>Ticker</th><th>1-Year Momentum</th></tr>"

for rank, etf in enumerate(top_etfs, start=1):
    html_content += f"<tr><td>{rank}</td><td>{etf['Name']}</td><td>{etf['Ticker']}</td><td>{etf['Momentum']:.2%}</td></tr>"

html_content += "</table></body></html>"

with open(OUTPUT_HTML, "w") as f:
    f.write(html_content)

print(f"\n✅ HTML dashboard written to {OUTPUT_HTML}")
