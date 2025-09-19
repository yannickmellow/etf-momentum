import pandas as pd
import yfinance as yf
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ETF_CSV = "etf_list.csv"  # CSV with 'Ticker','Name'
SHORT_TERM_DAYS = 21      # ~1 month
MEDIUM_TERM_DAYS = 63     # ~3 months
TOP_N = 10
HTML_OUTPUT = "index.html"

# -----------------------------
# READ ETF LIST
# -----------------------------
etf_df = pd.read_csv(ETF_CSV)
etf_list = etf_df.to_dict("records")  # [{'Ticker':..., 'Name':...}, ...]

results = []

# -----------------------------
# CALCULATE MOMENTUM
# -----------------------------
for etf in etf_list:
    ticker = etf['Ticker']
    name = etf['Name']
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)['Adj Close']
        if len(data) < MEDIUM_TERM_DAYS + 1:
            print(f"⚠️ Not enough data for {ticker} ({name})")
            continue

        momentum_1m = (data.iloc[-1] / data.iloc[-SHORT_TERM_DAYS] - 1) * 100
        momentum_3m = (data.iloc[-1] / data.iloc[-MEDIUM_TERM_DAYS] - 1) * 100

        results.append({
            "Ticker": ticker,
            "Name": name,
            "Momentum 1M (%)": round(momentum_1m, 2),
            "Momentum 3M (%)": round(momentum_3m, 2)
        })

    except Exception as e:
        print(f"⚠️ Error fetching {ticker} ({name}): {e}")

# -----------------------------
# SELECT TOP N BY 3-MONTH MOMENTUM
# -----------------------------
top_etfs = sorted(results, key=lambda x: x["Momentum 3M (%)"], reverse=True)[:TOP_N]

# -----------------------------
# GENERATE HTML DASHBOARD
# -----------------------------
html = """
<html>
<head><title>Top ETF Momentum</title></head>
<body>
<h1>Top {n} ETFs by 3-Month Momentum</h1>
<table border="1" cellpadding="5">
<tr>
<th>Rank</th><th>Ticker</th><th>Name</th><th>1-Month Momentum (%)</th><th>3-Month Momentum (%)</th>
</tr>
""".format(n=TOP_N)

for i, etf in enumerate(top_etfs, 1):
    html += f"<tr><td>{i}</td><td>{etf['Ticker']}</td><td>{etf['Name']}</td>"
    html += f"<td>{etf['Momentum 1M (%)']}</td><td>{etf['Momentum 3M (%)']}</td></tr>"

html += "</table></body></html>"

# -----------------------------
# SAVE HTML
# -----------------------------
Path(HTML_OUTPUT).write_text(html)
print(f"✅ Dashboard written to {HTML_OUTPUT}")
