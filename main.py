import yfinance as yf
import pandas as pd
from datetime import datetime

# -----------------------------
# User inputs
# -----------------------------
etf_csv_file = "etf_list.csv"  # CSV file in GitHub repo root
lookback_periods = {
    "3m": 63,   # ~63 trading days
    "6m": 126,
    "12m": 252
}
top_n = 10  # Number of ETFs to select

# -----------------------------
# Helper functions
# -----------------------------
def calculate_momentum(prices, periods):
    momentum_scores = {}
    for period_name, period_days in periods.items():
        if len(prices) >= period_days:
            momentum_scores[period_name] = (prices[-1] / prices[-period_days] - 1) * 100
        else:
            momentum_scores[period_name] = None
    return momentum_scores

def evaluate_criteria(momentum, ma50, ma200):
    criteria = {}
    for period in ["3m", "6m", "12m"]:
        criteria[f"{period}_momentum_positive"] = momentum[period] is not None and momentum[period] > 0
    criteria["above_ma200"] = ma200 is not None and ma50 > ma200
    return criteria

# -----------------------------
# Load ETF list
# -----------------------------
try:
    etf_df = pd.read_csv(etf_csv_file)
    etf_list = etf_df.iloc[:,0].tolist()  # assume first column contains tickers
except Exception as e:
    print(f"⚠️ Failed to read {etf_csv_file}: {e}")
    etf_list = []

# -----------------------------
# Fetch data and analyze
# -----------------------------
results = []

for etf in etf_list:
    try:
        data = yf.download(etf, period="1y", interval="1d")["Adj Close"]
        if data.empty:
            print(f"⚠️ No data for {etf}, skipping.")
            continue

        momentum = calculate_momentum(data, lookback_periods)
        ma50 = data[-50:].mean() if len(data) >= 50 else None
        ma200 = data[-200:].mean() if len(data) >= 200 else None
        criteria = evaluate_criteria(momentum, ma50, ma200)

        results.append({
            "ETF": etf,
            "Current Price": round(data[-1], 2),
            **momentum,
            "MA50": round(ma50, 2) if ma50 else None,
            "MA200": round(ma200, 2) if ma200 else None,
            **criteria
        })

    except Exception as e:
        print(f"⚠️ Error fetching {etf}: {e}")

# -----------------------------
# Convert to DataFrame and rank
# -----------------------------
df = pd.DataFrame(results)
df["12m_rank"] = df["12m"].rank(ascending=False)  # highest 12m momentum is rank 1
df_sorted = df.sort_values("12m_rank").head(top_n)

# -----------------------------
# Generate HTML
# -----------------------------
html_content = f"""
<html>
<head>
    <title>ETF Momentum Scanner</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .pass {{ background-color: #c6f6d5; }}
        .fail {{ background-color: #fed7d7; }}
    </style>
</head>
<body>
    <h1>Top {top_n} ETFs by 12-Month Momentum</h1>
    <table>
        <tr>
            <th>ETF</th>
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
"""

for idx, row in df_sorted.iterrows():
    reason = []
    reason.append("3m momentum positive" if row["3m_momentum_positive"] else "3m momentum negative")
    reason.append("6m momentum positive" if row["6m_momentum_positive"] else "6m momentum negative")
    reason.append("12m momentum positive" if row["12m_momentum_positive"] else "12m momentum negative")
    reason.append("Above MA200" if row["above_ma200"] else "Below MA200")
    reason_text = "; ".join(reason)

    html_content += f"""
    <tr>
        <td>{row['ETF']}</td>
        <td>{row['Current Price']}</td>
        <td>{row['3m']:.2f}</td>
        <td>{row['6m']:.2f}</td>
        <td>{row['12m']:.2f}</td>
        <td>{row['MA50']}</td>
        <td>{row['MA200']}</td>
        <td class="{'pass' if row['3m_momentum_positive'] else 'fail'}">{row['3m_momentum_positive']}</td>
        <td class="{'pass' if row['6m_momentum_positive'] else 'fail'}">{row['6m_momentum_positive']}</td>
        <td class="{'pass' if row['12m_momentum_positive'] else 'fail'}">{row['12m_momentum_positive']}</td>
        <td class="{'pass' if row['above_ma200'] else 'fail'}">{row['above_ma200']}</td>
        <td>{reason_text}</td>
    </tr>
    """

html_content += """
    </table>
    <p>Generated on {}</p>
</body>
</html>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -----------------------------
# Write HTML file
# -----------------------------
with open("index.html", "w") as f:
    f.write(html_content)

print(f"✅ HTML report generated with top {top_n} ETFs.")
