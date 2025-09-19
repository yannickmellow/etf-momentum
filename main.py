import pandas as pd
import yfinance as yf
from datetime import datetime
import os

# -----------------------------
# User parameters
# -----------------------------
etf_csv_file = "etf_list.csv"  # CSV with Ticker, Name
lookback_periods = {"3m": 63, "6m": 126, "12m": 252}  # trading days ~ 21 per month
top_n = 10
output_html = "docs/index.html"

# -----------------------------
# Helper functions
# -----------------------------
def calculate_momentum(data, lookbacks):
    momentum = {}
    for period_name, days in lookbacks.items():
        if len(data) >= days:
            momentum[period_name] = ((data[-1] / data[-days]) - 1) * 100
        else:
            momentum[period_name] = None
    return momentum

def evaluate_criteria(momentum, ma50, ma200):
    criteria = {}
    criteria['3m+'] = momentum.get('3m') is not None and momentum['3m'] > 0
    criteria['6m+'] = momentum.get('6m') is not None and momentum['6m'] > 0
    criteria['12m+'] = momentum.get('12m') is not None and momentum['12m'] > 0
    criteria['Above MA200?'] = ma200 is not None and momentum.get('current_price', 0) > ma200
    # Build reason string
    reasons = []
    if criteria['3m+']: reasons.append("3m+")
    if criteria['6m+']: reasons.append("6m+")
    if criteria['12m+']: reasons.append("12m+")
    if criteria['Above MA200?']: reasons.append("Above MA200")
    criteria['Reason for Selection'] = ", ".join(reasons) if reasons else "-"
    return criteria

# -----------------------------
# Load ETF list
# -----------------------------
try:
    etf_df = pd.read_csv(etf_csv_file)
    etf_list = etf_df.to_dict(orient="records")  # [{'Ticker': 'SPY', 'Name': 'SPDR S&P 500 ETF'}, ...]
except Exception as e:
    print(f"⚠️ Failed to read {etf_csv_file}: {e}")
    etf_list = []

# -----------------------------
# Fetch data and analyze
# -----------------------------
results = []

for etf_entry in etf_list:
    etf = etf_entry['Ticker']
    etf_name = etf_entry['Name']
    try:
        data = yf.download(etf, period="1y", interval="1d")["Adj Close"]
        if data.empty:
            print(f"⚠️ No data for {etf}, skipping.")
            continue

        momentum = calculate_momentum(data, lookback_periods)
        momentum['current_price'] = data[-1]
        ma50 = data[-50:].mean() if len(data) >= 50 else None
        ma200 = data[-200:].mean() if len(data) >= 200 else None
        criteria = evaluate_criteria(momentum, ma50, ma200)

        results.append({
            "ETF": etf,
            "Name": etf_name,
            "Current Price": round(data[-1], 2),
            "3m Momentum %": round(momentum['3m'], 2) if momentum['3m'] is not None else None,
            "6m Momentum %": round(momentum['6m'], 2) if momentum['6m'] is not None else None,
            "12m Momentum %": round(momentum['12m'], 2) if momentum['12m'] is not None else None,
            "MA50": round(ma50, 2) if ma50 else None,
            "MA200": round(ma200, 2) if ma200 else None,
            "3m+": criteria['3m+'],
            "6m+": criteria['6m+'],
            "12m+": criteria['12m+'],
            "Above MA200?": criteria['Above MA200?'],
            "Reason for Selection": criteria['Reason for Selection']
        })

    except Exception as e:
        print(f"⚠️ Error fetching {etf}: {e}")

# -----------------------------
# Select top N based on sum of momentum %s
# -----------------------------
for r in results:
    r['Momentum Score'] = sum([v for k,v in r.items() if k.endswith('Momentum %') and v is not None])

selected = sorted(results, key=lambda x: x['Momentum Score'], reverse=True)[:top_n]

# -----------------------------
# Generate HTML dashboard
# -----------------------------
html_rows = ""
for r in results:
    highlight = "style='background-color:#d4edda;'" if r in selected else ""
    html_rows += f"""
    <tr {highlight}>
        <td>{r['ETF']}</td>
        <td>{r['Name']}</td>
        <td>{r['Current Price']}</td>
        <td>{r['3m Momentum %']}</td>
        <td>{r['6m Momentum %']}</td>
        <td>{r['12m Momentum %']}</td>
        <td>{r['MA50']}</td>
        <td>{r['MA200']}</td>
        <td>{r['3m+']}</td>
        <td>{r['6m+']}</td>
        <td>{r['12m+']}</td>
        <td>{r['Above MA200?']}</td>
        <td>{r['Reason for Selection']}</td>
    </tr>
    """

html_content = f"""
<html>
<head>
    <title>ETF Momentum Dashboard</title>
    <style>
        table {{border-collapse: collapse; width: 100%;}}
        th, td {{border: 1px solid #ccc; padding: 6px; text-align: center;}}
        th {{background-color: #f2f2f2;}}
    </style>
</head>
<body>
    <h2>ETF Momentum Dashboard - {datetime.now().strftime('%Y-%m-%d')}</h2>
    <p>Top {top_n} ETFs highlighted in green.</p>
    <table>
        <tr>
            <th>ETF</th>
            <th>Name</th>
            <th>Current Price</th>
            <th>3m Momentum %</th>
            <th>6m Momentum %</th>
            <th>12m Momentum %</th>
            <th>MA50</th>
            <th>MA200</th>
            <th>3m+?</th>
            <th>6m+?</th>
            <th>12m+?</th>
            <th>Above MA200?</th>
            <th>Reason for Selection</th>
        </tr>
        {html_rows}
    </table>
</body>
</html>
"""

os.makedirs(os.path.dirname(output_html), exist_ok=True)
with open(output_html, "w") as f:
    f.write(html_content)

print(f"✅ Dashboard written to {output_html}")
