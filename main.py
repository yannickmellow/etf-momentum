import yfinance as yf
import pandas as pd
from datetime import datetime

# -----------------------------
# Load ETF list
# -----------------------------
etf_list = pd.read_csv("etf_list.csv")  # columns: Ticker, Name

# -----------------------------
# Download data and calculate returns
# -----------------------------
etf_returns = {}

for idx, row in etf_list.iterrows():
    etf = row['Ticker']
    name = row['Name']
    try:
        data = yf.download(etf, period="1y", interval="1d", auto_adjust=True)["Adj Close"]
        if len(data) < 2:
            print(f"⚠️ Not enough data for {etf} ({name}), skipping.")
            continue
        ret = (data[-1] / data[0] - 1) * 100
        etf_returns[etf] = {"Name": name, "Return": ret}
    except Exception as e:
        print(f"⚠️ Error fetching {etf} ({name}): {e}")
        continue

# -----------------------------
# Convert to DataFrame
# -----------------------------
df_returns = pd.DataFrame.from_dict(etf_returns, orient='index')
df_returns.index.name = "Ticker"

# -----------------------------
# Select Top 10 ETFs
# -----------------------------
top10 = df_returns.sort_values("Return", ascending=False).head(10)

# -----------------------------
# Generate HTML dashboard (Top 10 only)
# -----------------------------
html_content = f"""
<html>
<head>
    <title>ETF Momentum Dashboard - Top 10</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 50%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .top10 {{ background-color: #d4edda; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>ETF Momentum Dashboard - Top 10</h1>
    <p>Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <table>
        <tr>
            <th>Ticker</th>
            <th>Name</th>
            <th>1-Year Return (%)</th>
        </tr>
"""

for ticker, row in top10.iterrows():
    html_content += f"""
        <tr class="top10">
            <td>{ticker}</td>
            <td>{row['Name']}</td>
            <td>{row['Return']:.2f}</td>
        </tr>
    """

html_content += """
    </table>
</body>
</html>
"""

# -----------------------------
# Write HTML to file
# -----------------------------
with open("index.html", "w") as f:
    f.write(html_content)

print("\n✅ Dashboard generated: index.html")
print("\nTop 10 ETFs by 1-year return:")
print(top10)
