import yfinance as yf


def download_prices(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    prices = data["Close"]

    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()

    returns = prices.pct_change().dropna()

    return prices, returns