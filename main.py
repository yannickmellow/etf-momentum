import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# File paths
ETF_LIST_FILE = "etf_list.csv"
OUTPUT_HTML = "index.html"

# Momentum parameters
SHORT_TERM_DAYS = 21   # ~1 month
MEDIUM_TERM_DAYS = 63  # ~3 months
TOP_N = 10

# Read ETF list
try:
    etf_df = pd.read_csv(ETF_LIST_FILE)
except Exception as e:
    print(f"⚠️ Error reading {ETF_LIST_FILE}: {e}")
    exit(1)

# Container for results
results = []

# Fetch data and compute momentum
for idx, row in etf_df.iterrows():
    ticker = row['Ticker']
    name = row['Name']
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
        if data.empty:
            raise ValueError("No data returned")
        close = data["Close"]

        # Compute simple momentum
        if len(close) >= MEDIUM_TERM_DAYS:
            momentum_1m = (close[-1] / close[-SHORT_TERM_DAYS] - 1) * 100
            momentum_3m = (close[-1] / close[-MEDIUM_TERM_DAYS] - 1) * 100
            results.append({
                "Ticker": ticker,
                "Name": name,
                "1M (%)": round(momentum_1m, 2),
                "3M (%)": round(momentum_3m, 2),
                "Rank": 0  # placeholder
            })
        else:
            print(f"⚠️ Not enough data for {ticker} ({len(close)} rows)")
    except Exception as e:
        print(f"⚠️ Error fetching {ticker} ({name}): {e}")

# Convert to DataFrame
if not results:
    print("⚠️ No ETF data fetched successfully. Exiting.")
    exit(1)

momentum_df = pd.DataFrame(results)

# Rank by 1M momentum (you can change metric)
momentum_df.sort_values("1M (%)", ascending=False, inplace=True)
momentum_df = momentum_df.head(TOP_N)
momentum_df['Rank'] = range(1, len(momentum_df) + 1)

# Generate HTML
html_content = f"""
<html>
<head>
<title>Top {TOP_N} ETFs by Momentum</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 80%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background-color: #4CAF50; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
</style>
</head>
<body>
<h2>Top {TOP_N} ETFs by Momentum (as of {datetime.today().strftime('%Y-%m-%d')})</h2>
<table>
<tr>
<th>Rank</th>
<th>Ticker</th>
<th>Name</th>
<th>1M (%)</th>
<th>3M (%)</th>
</tr>
"""

for _, row in momentum_df.iterrows():
    html_content += f"""
<tr>
<td>{row['Rank']}</td>
<td>{row['Ticker']}</td>
<td>{row['Name']}</td>
<td>{row['1M (%)']}</td>
<td>{row['3M (%)']}</td>
</tr>
"""

html_content += """
</table>
</body>
</html>
"""

# Write to HTML file
Path(OUTPUT_HTML).write_text(html_content)
print(f"✅ Top {TOP_N} ETFs dashboard written to {OUTPUT_HTML}")
