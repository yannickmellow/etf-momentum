"""
ETF Screener + Monthly Rebalancer (scaffold)

Features implemented in this single-file scaffold:
- Load ETF etf_list from CSV (ticker, name, region, sector, tag)
- Fetch historical daily prices using yahooquery (batched)
- Compute multi-timeframe momentum: 12m, 6m, 1m (1m inverted)
- Blend into composite performance score with configurable weights
- 52-week-high proximity filter (exclude ETFs too far below 52-week high)
- Volatility adjustment (score scaled by inverse volatility)
- Correlation-aware selection (limit concentration by tag/sector/region)
- Monthly rebalance simulation with equal-weight allocation
- Output: holdings per rebalance, portfolio NAV time series, CSV exports

NOTES:
- This is a scaffold to iterate from; many production features can be added: caching,
  error handling, logging, parallel fetching, dividend-adjusted total return, commissions/slippage.

CSV etf_list format (example):
# ticker,name,region,sector,tag
SPY,SPDR S&P 500 Trust,US,Equity,LargeCap
EFA,iShares MSCI EAFE ETF,Global,Equity,Developed
VWO,Vanguard FTSE Emerging Markets ETF,Global,Equity,EM
GLD,SPDR Gold Shares,Global,Commodity,Gold
AGG,iShares Core US Aggregate Bond ETF,US,Bond,FixedIncome

Configurable parameters are near the top of the file.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from yahooquery import Ticker
import os
import math

# -------------------- USER PARAMETERS --------------------#

etf_list_CSV = 'etf_list.csv'    # must contain a column 'ticker' and optional 'tag' (sector/country grouping)
START_DATE = '2015-01-01'
END_DATE = None  # None means today
LOOKBACK_12M = 252  # trading days (approx)
LOOKBACK_6M = 126
LOOKBACK_1M = 21
TOP_N = 5            # number of ETFs to hold each rebalance
REBALANCE_DAY = 'month_end'  # 'month_end' or integer day of month
MIN_PRICE_POINTS = 252      # minimum history points required
MAX_DIST_FROM_52W_HIGH_PCT = 0.20  # exclude ETFs more than 20% below 52-week high
VOL_ADJUSTMENT_POWER = 1.0  # higher -> more penalisation of vol
CORRELATION_WINDOW = 252
MAX_EXPOSURE_PER_TAG = 0.5  # no more than 50% of portfolio in same tag (sector/country)
WEIGHTS = { '12m': 0.5, '6m': 0.3, '1m': 0.2 }  # 1m will be inverted in calculation
INITIAL_CAPITAL = 100000.0
CASH_RESERVE_PCT = 0.0  # keep a cash % if desired

# -------------------- UTILITIES --------------------#

# Load static ETF etf_list
etf_file = "etf_list.csv"  # adjust path if needed
tickers = pd.read_csv(etf_file)["ticker"].dropna().unique().tolist()

print(f"📥 Loaded {len(tickers)} tickers from {etf_file}")



def fetch_price_history(tickers, start_date, end_date=None, interval='1d'):
    """Fetch adjusted close prices for tickers using yahooquery.Ticker in batches.
    Returns a DataFrame of adjclose prices indexed by date, columns = tickers.
    """
    t = Ticker(tickers)
    # yahooquery returns 'adjclose' in historical as 'adjclose' or 'close' depending on symbol
    hist = t.history(start=start_date, end=end_date, interval=interval)
    if isinstance(hist, pd.DataFrame) and 'symbol' in hist.columns:
        hist = hist.reset_index().pivot_table(index='date', columns='symbol', values='adjclose')
    else:
        # fallback: return empty
        hist = pd.DataFrame()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist.sort_index()
    return hist

# -------------------- CALCULATIONS --------------------#

def total_return(prices, lookback_days):
    """Compute total return over lookback_days using price series (pandas Series).
    Uses first valid price and last valid price within the lookback window.
    """
    if prices.dropna().shape[0] < 2:
        return np.nan
    # take last available date and date - lookback_days (approx)
    end = prices.last_valid_index()
    start_idx = prices.index.get_indexer([end - pd.Timedelta(days=lookback_days*1.1)], method='nearest')[0]
    start = prices.index[start_idx]
    p0 = prices.loc[start]
    p1 = prices.loc[end]
    if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
        return np.nan
    return (p1 / p0) - 1.0


def pct_from_52w_high(prices):
    last = prices.last_valid_index()
    if last is None:
        return np.nan
    window_start = prices.index[-252] if len(prices) >= 252 else prices.index[0]
    window = prices.loc[window_start:]
    high = window.max(skipna=True)
    if pd.isna(high) or high == 0:
        return np.nan
    last_price = prices.loc[last]
    return (high - last_price) / high


def annualized_vol(prices, window_days=252):
    # compute daily returns and annualized std
    returns = prices.pct_change().dropna()
    if returns.empty:
        return np.nan
    vol = returns.rolling(window=window_days, min_periods=21).std() * math.sqrt(252)
    return vol.iloc[-1]

# -------------------- SCORING --------------------#

def compute_scores(price_df, weights=None):
    """Compute composite scores for all tickers in price_df (DataFrame of adj closes).
    Returns a DataFrame with columns: score, r12, r6, r1, vol, dist52w
    """
    if weights is None:
        weights = WEIGHTS
    results = []

    for col in price_df.columns:
        s = price_df[col].dropna()
        if s.shape[0] < 21:
            results.append((col, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        r12 = s.pct_change(periods=LOOKBACK_12M).iloc[-1] if len(s) > LOOKBACK_12M else (s.iloc[-1] / s.iloc[0] - 1)
        r6 = s.pct_change(periods=LOOKBACK_6M).iloc[-1] if len(s) > LOOKBACK_6M else (s.iloc[-1] / s.iloc[max(0,0)] - 1)
        r1 = s.pct_change(periods=LOOKBACK_1M).iloc[-1] if len(s) > LOOKBACK_1M else (s.iloc[-1] / s.iloc[max(0,0)] - 1)
        # invert 1-month (mean reversion)
        r1_inverted = -r1
        dist52 = pct_from_52w_high(s)
        vol = annualized_vol(s, window_days=CORRELATION_WINDOW)
        # raw blended score
        raw = (weights['12m'] * r12) + (weights['6m'] * r6) + (weights['1m'] * r1_inverted)
        # volatility adjustment: divide by vol^power (higher vol penalises)
        if vol is None or pd.isna(vol) or vol == 0:
            score = np.nan
        else:
            score = raw / (vol ** VOL_ADJUSTMENT_POWER)
        results.append((col, score, r12, r6, r1, vol, dist52))

    out = pd.DataFrame(results, columns=['ticker', 'score', 'r12', 'r6', 'r1', 'vol', 'dist52'])
    out = out.set_index('ticker')
    return out

# -------------------- CORRELATION / CONCENTRATION FILTER --------------------#

def enforce_correlation_and_tags(selected, price_df, etf_list_df, max_per_tag=MAX_EXPOSURE_PER_TAG, top_n=TOP_N):
    """Given an initial selection (list ordered by score desc), enforce tag-based concentration limits.
    This function will iterate down the ranked list and select funds while capping exposure per tag.
    Output is list of chosen tickers (length <= top_n).
    """
    tag_caps = {}
    chosen = []
    weights = {}
    # Equal weight allocation for chosen funds
    for t in selected:
        tag = etf_list_df.loc[etf_list_df['ticker'] == t, 'tag'].values
        tag = tag[0] if len(tag) > 0 else 'Other'
        current_count_for_tag = sum(1 for c in chosen if etf_list_df.loc[etf_list_df['ticker'] == c, 'tag'].values[0] == tag)
        # if giving equal weights, cap is floor(max_per_tag * top_n)
        max_count = math.floor(max_per_tag * top_n)
        # ensure at least 1 slot available if max_count == 0
        if max_count < 1:
            max_count = 1
        if current_count_for_tag < max_count:
            chosen.append(t)
        if len(chosen) >= top_n:
            break
    return chosen

# -------------------- REBALANCE / BACKTEST ENGINE --------------------#

def get_monthly_rebalance_dates(price_df):
    idx = price_df.index
    # choose last trading day of each month
    month_ends = idx.groupby([idx.year, idx.month]).max()
    return pd.to_datetime(month_ends.values)


def run_monthly_backtest(price_df, etf_list_df, start_date=None, end_date=None, top_n=TOP_N):
    # prepare
    if start_date is None:
        start_date = price_df.index[0]
    if end_date is None:
        end_date = price_df.index[-1]

    rebalance_dates = [d for d in get_monthly_rebalance_dates(price_df) if (d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date))]
    holdings_history = []
    nav = pd.Series(index=price_df.loc[rebalance_dates[0]:].index, dtype=float)
    cash = INITIAL_CAPITAL * CASH_RESERVE_PCT
    capital = INITIAL_CAPITAL - cash
    portfolio = {}

    for i, rebalance_date in enumerate(rebalance_dates):
        # compute scores using prices up to rebalance_date
        price_up_to_date = price_df.loc[:rebalance_date]
        scores = compute_scores(price_up_to_date)
        # filter by 52-week high proximity
        scores = scores[scores['dist52'] <= MAX_DIST_FROM_52W_HIGH_PCT]
        scores = scores.dropna(subset=['score'])
        # rank
        ranked = scores.sort_values('score', ascending=False)
        candidates = list(ranked.index)
        chosen = enforce_correlation_and_tags(candidates, price_df, etf_list_df, max_per_tag=MAX_EXPOSURE_PER_TAG, top_n=top_n)
        # assign equal weights
        if len(chosen) == 0:
            portfolio = {}
        else:
            w = 1.0 / len(chosen)
            portfolio = {t: w for t in chosen}
        holdings_history.append({'date': rebalance_date, 'holdings': portfolio.copy(), 'scores_snapshot': ranked.head(20)})

    # Build NAV time series by forward-filling holdings between rebalance dates
    nav_dates = price_df.loc[rebalance_dates[0]:].index
    nav_series = pd.Series(index=nav_dates, dtype=float)
    current_portfolio = {}
    cash = INITIAL_CAPITAL * CASH_RESERVE_PCT
    capital = INITIAL_CAPITAL - cash

    for i in range(len(rebalance_dates)):
        start = rebalance_dates[i]
        end = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else price_df.index[-1]
        holdings = holdings_history[i]['holdings']
        if len(holdings) == 0:
            # stay in cash
            nav_series.loc[start:end] = capital + cash
            continue
        # compute portfolio value each day in period
        tickers = list(holdings.keys())
        weights = np.array([holdings[t] for t in tickers])
        sub = price_df.loc[start:end, tickers].fillna(method='ffill')
        # normalize price to start of period
        start_prices = sub.iloc[0]
        shares = (capital * weights) / start_prices
        # daily value
        daily_vals = sub.multiply(shares, axis=1).sum(axis=1) + cash
        nav_series.loc[start:end] = daily_vals

    nav_series = nav_series.dropna()
    return {
        'nav': nav_series,
        'holdings': holdings_history,
    }

# -------------------- MAIN EXECUTION --------------------#

if __name__ == '__main__':
    # Load etf_list
    uni = load_etf_list(etf_list_CSV)
    tickers = uni['ticker'].tolist()

    # Fetch prices
    print(f'Fetching price history for {len(tickers)} tickers...')
    prices = fetch_price_history(tickers, start_date=START_DATE, end_date=END_DATE)
    if prices.empty:
        raise RuntimeError('Price download failed or returned no data')

    # Truncate to symbols we have
    available = [c for c in prices.columns if c in tickers]
    prices = prices[available]

    # Run a quick ranking for the most recent date
    scores = compute_scores(prices)
    scores = scores[scores['dist52'] <= MAX_DIST_FROM_52W_HIGH_PCT]
    scores = scores.sort_values('score', ascending=False)
    print('Top scoring ETFs (snapshot):')
    print(scores.head(10))

    # Run backtest
    print('\nRunning monthly backtest...')
    res = run_monthly_backtest(prices, uni, top_n=TOP_N)
    nav = res['nav']
    nav.to_csv('nav_timeseries.csv')
    # Save holdings per rebalance
    holdings_df = []
    for h in res['holdings']:
        row = {'date': h['date']}
        for idx, t in enumerate(h['holdings']):
            row[f'holding_{idx+1}'] = t
            row[f'weight_{idx+1}'] = h['holdings'][t]
        holdings_df.append(row)
    pd.DataFrame(holdings_df).to_csv('holdings_history.csv', index=False)
    print('Backtest complete. NAV saved to nav_timeseries.csv, holdings saved to holdings_history.csv')


# -------------------- HTML REPORT --------------------

def generate_html_report(top_holdings, output_file="output.html"):
    """
    top_holdings: list of dicts, each dict = {'ticker': str, 'weight': float, 'score': float, 'r12': float, 'r6': float, 'r1': float}
    """
    html = """
    <html>
    <head>
        <title>ETF Screener Monthly Top Holdings</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; }
            table { border-collapse: collapse; width: 80%; margin: auto; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h2 style="text-align:center;">Top 10 ETFs - Monthly Screener</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Ticker</th>
                <th>Weight</th>
                <th>Score</th>
                <th>12m Return</th>
                <th>6m Return</th>
                <th>1m Return (inverted)</th>
            </tr>
    """

    for i, h in enumerate(top_holdings, start=1):
        html += f"""
            <tr>
                <td>{i}</td>
                <td>{h['ticker']}</td>
                <td>{h['weight']:.2%}</td>
                <td>{h['score']:.4f}</td>
                <td>{h['r12']:.2%}</td>
                <td>{h['r6']:.2%}</td>
                <td>{h['r1']:.2%}</td>
            </tr>
        """

    html += """
        </table>
        <p style="text-align:center;">Generated by ETF Screener</p>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html)
    print(f"✅ HTML report saved to {output_file}")


# -------------------- Prepare Top 10 for HTML --------------------#
# After backtest completes, pick the latest rebalance holdings
latest_holdings = res['holdings'][-1]  # last rebalance
scores_snapshot = latest_holdings['scores_snapshot']  # top scores dataframe

# Map tickers to holdings weights
portfolio = latest_holdings['holdings']
top10 = []
for t in list(portfolio.keys())[:10]:
    row = scores_snapshot.loc[t] if t in scores_snapshot.index else None
    top10.append({
        'ticker': t,
        'weight': portfolio[t],
        'score': row['score'] if row is not None else 0,
        'r12': row['r12'] if row is not None else 0,
        'r6': row['r6'] if row is not None else 0,
        'r1': row['r1'] if row is not None else 0,
    })

generate_html_report(top10, output_file="docs/index.html")  # GitHub Pages default folder



# End of scaffold
