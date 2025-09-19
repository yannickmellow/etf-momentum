import pandas as pd
from yahooquery import Screener

def fetch_etf_universe(count=250):
    """
    Fetches a list of ETFs from Yahoo Finance Screener API and saves to universe.csv
    """
    s = Screener()
    results = s.get_screeners('etf', count=count)

    # Extract tickers & metadata
    etfs = results.get('etf', {}).get('quotes', [])
    rows = []
    for e in etfs:
        rows.append({
            "ticker": e.get("symbol"),
            "name": e.get("shortName"),
            "region": e.get("region") or "Unknown",
            "asset_class": e.get("quoteType") or "ETF",
            "category": e.get("longBusinessSummary")[:50] if e.get("longBusinessSummary") else "Unknown"
        })

    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv("universe.csv", index=False)
    print(f"✅ Saved {len(df)} ETFs to universe.csv")

if __name__ == "__main__":
    fetch_etf_universe(250)
