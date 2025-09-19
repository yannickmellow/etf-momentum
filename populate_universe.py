import pandas as pd
from yahooquery import Screener

def fetch_etf_universe(count=2000, min_volume=100000, min_marketcap=1e8):
    """
    Fetches ETFs from Yahoo Finance Screener, applies liquidity filters,
    and saves to universe.csv
    """
    s = Screener()
    results = s.get_screeners('etf', count=count)

    etfs = results.get('etf', {}).get('quotes', [])
    rows = []
    for e in etfs:
        avg_vol = e.get("averageDailyVolume3Month", 0)
        market_cap = e.get("marketCap", 0)

        # Apply liquidity & size filter
        if avg_vol and avg_vol >= min_volume and market_cap and market_cap >= min_marketcap:
            rows.append({
                "ticker": e.get("symbol"),
                "name": e.get("shortName"),
                "region": e.get("region") or "Unknown",
                "asset_class": e.get("quoteType") or "ETF",
                "avg_volume": avg_vol,
                "market_cap": market_cap
            })

    df = pd.DataFrame(rows)

    # Sort by market cap descending
    df = df.sort_values("market_cap", ascending=False)

    # Save to CSV
    df.to_csv("universe.csv", index=False)
    print(f"✅ Saved {len(df)} liquid ETFs to universe.csv")

if __name__ == "__main__":
    fetch_etf_universe()
