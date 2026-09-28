"""ArithMatrix AVIP Task 3: reproducible stock price visualisation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def validate_dates(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo:
        raise ValueError("Use dates without timezones, in YYYY-MM-DD format.")
    if start > end:
        raise ValueError("The start date must be on or before the end date.")
    return start.normalize(), end.normalize()


def fetch_prices(ticker, start, end):
    """Both user-facing dates are inclusive; yfinance's end is exclusive."""
    import yfinance as yf
    start, end = validate_dates(start, end)
    ticker = ticker.strip().upper()
    if not ticker or any(c.isspace() for c in ticker):
        raise ValueError("Choose one ticker, such as AAPL.")
    raw = yf.download(
        ticker, start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=False, actions=False,
        progress=False, threads=False, multi_level_index=False,
        keepna=True, timeout=30,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo returned no data. Check the ticker, dates and internet connection. "
            "If Yahoo is rate limiting, wait and rerun the download cell. "
            "Do not substitute made-up prices."
        )
    # Explicitly handle the column format if the installed version returns levels.
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if ticker in raw.columns.get_level_values(level):
                raw = raw.xs(ticker, level=level, axis=1)
                break
    required = ["Close", "Adj Close"]
    if not all(col in raw.columns for col in required):
        raise ValueError(f"Expected Close and Adj Close; received {list(raw.columns)}")
    raw = raw[required].copy()
    raw.index.name = "Date"
    metadata = {
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance",
        "source_url": f"https://finance.yahoo.com/quote/{ticker}/history/",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start_inclusive": start.strftime("%Y-%m-%d"),
        "requested_end_inclusive": end.strftime("%Y-%m-%d"),
        "auto_adjust": False,
        "price_basis": "Yahoo Close for price history; Adj Close for moving averages and returns",
        "software_versions": {
            name: importlib.metadata.version(name)
            for name in ["yfinance", "pandas", "numpy", "matplotlib"]
        },
    }
    return raw, metadata


def prepare_prices(raw, start, end):
    """Keep a cleaning audit and do not invent prices on non-trading days."""
    start, end = validate_dates(start, end)
    df = raw[["Close", "Adj Close"]].copy()
    audit = {"downloaded_rows": len(df)}
    df.index = pd.to_datetime(df.index, errors="coerce")
    audit["invalid_dates_removed"] = int(df.index.isna().sum())
    df = df.loc[~df.index.isna()]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    audit["duplicate_dates_removed"] = int(df.index.duplicated(keep="last").sum())
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    in_range = (df.index >= start) & (df.index <= end)
    audit["out_of_range_rows_removed"] = int((~in_range).sum())
    df = df.loc[in_range]
    df = df.apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(df).all(axis=1) & (df > 0).all(axis=1)
    audit["invalid_price_rows_removed"] = int((~valid).sum())
    df = df.loc[valid].copy()
    if len(df) < 3:
        raise ValueError("Select a range containing at least three valid trading observations.")
    df.index.name = "Date"
    # Windows count observations (trading sessions), not calendar days.
    df["SMA_20"] = df["Adj Close"].rolling(20, min_periods=20).mean()
    df["SMA_50"] = df["Adj Close"].rolling(50, min_periods=50).mean()
    df["Daily_Return"] = df["Adj Close"].pct_change(fill_method=None)
    audit["retained_rows"] = len(df)
    audit["actual_first_date"] = df.index[0].strftime("%Y-%m-%d")
    audit["actual_last_date"] = df.index[-1].strftime("%Y-%m-%d")
    return df, audit


def summarise(df):
    returns = df["Daily_Return"].dropna()
    return {
        "observations": len(df),
        "return_observations": len(returns),
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "adjusted_price_change": float(df["Adj Close"].iloc[-1] / df["Adj Close"].iloc[0] - 1),
        "mean_daily_return": float(returns.mean()),
        "daily_volatility": float(returns.std(ddof=1)),
        "annualised_volatility_estimate": float(returns.std(ddof=1) * np.sqrt(252)),
        "best_daily_return": float(returns.max()),
        "worst_daily_return": float(returns.min()),
    }


def interpretation(ticker, stats):
    note = (
        f"Between {stats['first_date']} and {stats['last_date']}, {ticker}'s adjusted "
        f"closing price changed by {stats['adjusted_price_change']:+.2%} across "
        f"{stats['observations']} available trading observations. The arithmetic mean "
        f"daily return was {stats['mean_daily_return']:+.3%}. Daily volatility, measured "
        f"as the sample standard deviation of returns, was {stats['daily_volatility']:.2%}; "
        f"the annualised estimate was {stats['annualised_volatility_estimate']:.2%}, "
        f"using 252 trading sessions per year. The highest observed daily return was "
        f"{stats['best_daily_return']:+.2%}, and the smallest daily return was "
        f"{stats['worst_daily_return']:+.2%}.\n\n"
        "The 20-session moving average responds to recent changes faster than the "
        "50-session average; both lag the underlying price. The return histogram "
        "shows the frequency and spread of observed daily changes, while volatility "
        "describes their variation rather than predicting a direction. Adjusted "
        "prices account for the provider's corporate-action adjustments. Returns "
        "are calculated between consecutive available observations; data gaps can "
        "span more than one trading session. These historical results cover one "
        "stock and one selected period. They do not establish causation or forecast "
        "future performance; the annualisation is an approximation."
    )
    if len(note.split()) > 200:
        raise ValueError("The interpretation exceeds the task's 200-word limit.")
    return note


def export_charts(df, ticker, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    caption = f"{ticker} | {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y} | Yahoo Finance"
    paths = []

    def save(fig, ax, filename):
        ax.grid(alpha=0.18)
        fig.text(0.08, 0.02, caption, fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        path = output / filename
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df.index, df["Close"], color="#136F63", linewidth=1.8)
    ax.set(title=f"{ticker}: closing price history", xlabel="Date",
           ylabel="Yahoo Close (quote currency)")
    save(fig, ax, "01_closing_price.png")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df.index, df["Adj Close"], color="#566573", alpha=0.65, label="Adjusted close")
    ax.plot(df.index, df["SMA_20"], color="#007C83", linewidth=1.8, label="20-session SMA")
    ax.plot(df.index, df["SMA_50"], color="#B7662A", linewidth=1.8, label="50-session SMA")
    ax.set(title=f"{ticker}: adjusted close and moving averages", xlabel="Date",
           ylabel="Adjusted price (quote currency)")
    if len(df) < 50:
        ax.text(0.02, 0.05, "A full 50-session average needs at least 50 observations.",
                transform=ax.transAxes, fontsize=9)
    ax.legend(frameon=False)
    save(fig, ax, "02_moving_averages.png")

    returns = df["Daily_Return"].dropna()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(returns, bins=min(35, max(5, int(np.sqrt(len(returns))))),
            color="#007C83", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1, label="Zero return")
    ax.axvline(returns.mean(), color="#B7662A", linewidth=1.6, label="Mean return")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set(title=f"{ticker}: daily returns distribution", xlabel="Daily adjusted-price return",
           ylabel="Number of observations")
    ax.legend(frameon=False)
    save(fig, ax, "03_daily_returns.png")
    return paths


def export_report(raw, df, meta, audit, stats, note, output):
    output = Path(output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "data/source_prices.csv", index_label="Date")
    df.to_csv(output / "data/analysed_prices.csv", index_label="Date")
    (output / "data/source_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output / "data/cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (output / "interpretation.md").write_text(note + "\n", encoding="utf-8")
    versions = meta.get("software_versions", {})
    requirements = "\n".join(f"{name}=={versions[name]}" if name in versions else name
                              for name in ["yfinance", "pandas", "numpy", "matplotlib"]) + "\n"
    (output / "requirements.txt").write_text(requirements, encoding="utf-8")
    readme = f"""# Stock Price Data Visualisation

ArithMatrix AVIP 2026 | Data Science | Task 3

## Question and scope
How did **{meta['ticker']}** prices, moving averages and daily returns behave during the selected period?
Requested dates: **{meta['requested_start_inclusive']} to {meta['requested_end_inclusive']}**, both inclusive.
Actual observations: **{stats['first_date']} to {stats['last_date']}** ({stats['observations']} rows).

## Data source
[Yahoo Finance historical prices]({meta['source_url']}) fetched with yfinance.
Extraction timestamp (UTC): **{meta['retrieved_at_utc']}**.
`data/source_metadata.json` records the download settings and library versions.
The default stock, AAPL, is quoted in USD; other tickers can use different quote currencies or units.
The saved source snapshot supports reproduction if the provider later revises historical values.

## Run from a terminal
Requires Python 3.10 or newer. From the repository folder:

```bash
python -m pip install -r requirements.txt
python stock_analysis.py --ticker {meta['ticker']} --start {meta['requested_start_inclusive']} --end {meta['requested_end_inclusive']} --output results
```

For another period, change `--start` and `--end`; choose a range with at least 50 trading observations to display both moving averages. Both dates are inclusive. Weekends and market holidays have no rows. The downloader adds one calendar day to the end because yfinance excludes its end date.

To reproduce this saved snapshot without downloading market data again:

```bash
python stock_analysis.py --csv data/source_prices.csv --metadata data/source_metadata.json --output reproduced
```

In the companion Colab notebook, edit `TICKER`, `START_DATE` and `END_DATE`, then rerun from the download cell onwards. No date-selector interface is needed because the task permits documented date-range instructions.

## Method and architecture
`stock_analysis.py` separates downloading, cleaning, calculations, chart generation and reporting into functions.

1. Keep Yahoo's `Close` and `Adj Close` as separate columns (`auto_adjust=False`). Price history uses `Close`; moving averages and returns use `Adj Close` consistently.
2. Parse dates, retain the last duplicate date, sort chronologically, filter the date range and remove invalid, non-finite or non-positive prices. Counts are in `data/cleaning_audit.json`.
3. Do not forward-fill prices or create weekend/holiday observations. Returns are between consecutive available observations, so a missing observation can create a multi-session interval.
4. Calculate trailing 20- and 50-observation simple moving averages. The first 19/49 values are intentionally missing; no future prices are used.
5. Calculate simple returns as `adjusted_close[t] / adjusted_close[t-1] - 1`. The first return is undefined and excluded from statistics.
6. Calculate sample daily volatility (`ddof=1`) and an annualised estimate using `daily_volatility * sqrt(252)`. Annualisation assumes a conventional trading-year length and is approximate.

## Required visualisations
![Closing price](charts/01_closing_price.png)
![Moving averages](charts/02_moving_averages.png)
![Daily returns](charts/03_daily_returns.png)

## Interpretation ({len(note.split())} words)
{note}

## Deliverables
- `stock_analysis.py`: runnable fetch, preparation and analysis script.
- `charts/`: three exported PNG charts.
- `interpretation.md`: interpretation of no more than 200 words.
- `data/`: source snapshot, analysed prices, provenance and cleaning audit.
- `summary.json`: calculated return and volatility metrics (returns stored as decimal fractions).
- `requirements.txt`: dependency versions used for the data run.

## Limitations
One stock and one selected period do not represent the whole market. Adjusted historical data can be revised. The analysis describes past prices, does not predict future returns, and does not model transaction costs or taxes.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", default="results")
    parser.add_argument("--csv", help="Saved source CSV for offline reproduction")
    parser.add_argument("--metadata", help="Matching source_metadata.json for offline reproduction")
    args = parser.parse_args()
    if bool(args.csv) != bool(args.metadata):
        parser.error("Use --csv and --metadata together.")
    if args.csv:
        raw = pd.read_csv(args.csv, index_col="Date", parse_dates=True)
        meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    else:
        raw, meta = fetch_prices(args.ticker, args.start, args.end)
    df, audit = prepare_prices(raw, meta["requested_start_inclusive"], meta["requested_end_inclusive"])
    stats = summarise(df)
    note = interpretation(meta["ticker"], stats)
    export_charts(df, meta["ticker"], Path(args.output) / "charts")
    export_report(raw, df, meta, audit, stats, note, args.output)
    script_target = Path(args.output) / "stock_analysis.py"
    if Path(__file__).resolve() != script_target.resolve():
        script_target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(note)
    print(f"\nSaved analysis to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
