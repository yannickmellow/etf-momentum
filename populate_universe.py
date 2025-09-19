import pandas as pd
from yahooquery import Ticker

def populate_universe(
    input_file="etf_list.csv",
    output_file="universe.csv",
    min_volume=100000,
    min_marketcap=1e8
):
    """
    Reads a list of ETF tickers, fetches metadata from Yahoo Finance,
    applies liquidity filters, and saves final universe.csv
    """
    # Load tickers
    tickers = pd.read_csv(input_file)["ticker"].dropna().unique().tolist()
    print(f"📥 Loaded {len(tickers)} tickers from {input_file}")

    # Query Yahoo Finance
    t = Ticker(tickers, asynchronous=True)
    summary = t.summary_detail
    quotes = t.quote_type

    rows = []
    for ticker in tickers:
        s = summary.get(ticker, {})
        q = quotes.get(ticker, {})

        avg_vol = s.get("averageVolume", 0) or s.get("averageDailyVolume3Month", 0)
        market_cap = s.get("marketCap", 0)

        # Liquidity filter
        if avg_vol and avg_vol >= min_volume and market_cap and market_cap >= min_marketcap:
            rows.append({
                "ticker": ticker,
                "name": q.get("longName") or q.get("shortName") or ticker,
                "region": q.get("region") or "Unknown",
                "asset_class": q.get("quoteType") or "ETF",
                "avg_volume": avg_vol,
                "market_cap": market_cap
            })

    # Save results
    df = pd.DataFrame(rows)
    df = df.sort_values("market_cap", ascending=False)
    df.to_csv(output_file, index=False)

    print(f"✅ Final universe: {len(df)} ETFs saved to {output_file}")

if __name__ == "__main__":
    populate_universe()
