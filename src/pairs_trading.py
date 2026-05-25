import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_hedge_ratio_ols(prices, stock_a, stock_b):
    y = prices[stock_a]
    x = prices[stock_b]

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    beta = model.params[stock_b]

    return beta, model


def compute_spread(prices, stock_a, stock_b, beta):
    spread = prices[stock_a] - beta * prices[stock_b]

    return spread


def compute_zscore(spread):
    zscore = (spread - spread.mean()) / spread.std()

    return zscore


def generate_trading_signals(
    prices,
    stock_a,
    stock_b,
    entry_threshold=2.0,
    exit_threshold=0.5
):
    beta, model = compute_hedge_ratio_ols(
        prices,
        stock_a,
        stock_b
    )

    spread = compute_spread(
        prices,
        stock_a,
        stock_b,
        beta
    )

    zscore = compute_zscore(spread)

    signals = pd.DataFrame(index=prices.index)

    signals["Price_A"] = prices[stock_a]
    signals["Price_B"] = prices[stock_b]
    signals["Spread"] = spread
    signals["Zscore"] = zscore

    signals["Position_A"] = 0
    signals["Position_B"] = 0
    signals["Signal"] = "HOLD"

    signals.loc[
        signals["Zscore"] > entry_threshold,
        "Position_A"
    ] = -1

    signals.loc[
        signals["Zscore"] > entry_threshold,
        "Position_B"
    ] = 1

    signals.loc[
        signals["Zscore"] > entry_threshold,
        "Signal"
    ] = f"SHORT_{stock_a}_LONG_{stock_b}"

    signals.loc[
        signals["Zscore"] < -entry_threshold,
        "Position_A"
    ] = 1

    signals.loc[
        signals["Zscore"] < -entry_threshold,
        "Position_B"
    ] = -1

    signals.loc[
        signals["Zscore"] < -entry_threshold,
        "Signal"
    ] = f"LONG_{stock_a}_SHORT_{stock_b}"

    signals.loc[
        signals["Zscore"].abs() < exit_threshold,
        "Signal"
    ] = "EXIT"

    return signals, beta, model


def run_simple_backtest(signals):
    signals = signals.copy()

    signals["Return_A"] = signals["Price_A"].pct_change()
    signals["Return_B"] = signals["Price_B"].pct_change()

    signals["Strategy_Return"] = (
        signals["Position_A"].shift(1) * signals["Return_A"]
        + signals["Position_B"].shift(1) * signals["Return_B"]
    )

    signals["Strategy_Return"] = signals["Strategy_Return"].fillna(0)

    signals["Cumulative_Return"] = (
        1 + signals["Strategy_Return"]
    ).cumprod()

    return signals


def compute_performance_metrics(signals):
    mean_daily_return = signals["Strategy_Return"].mean()
    vol_daily = signals["Strategy_Return"].std()

    if vol_daily == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (
            mean_daily_return / vol_daily
        ) * np.sqrt(252)

    total_return = signals["Cumulative_Return"].iloc[-1] - 1

    running_max = signals["Cumulative_Return"].cummax()
    drawdown = signals["Cumulative_Return"] / running_max - 1
    max_drawdown = drawdown.min()

    metrics = {
        "Total_Return": total_return,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown
    }

    return metrics

def backtest_pair_strategy(
    prices,
    stock_a,
    stock_b,
    entry_threshold=2.0,
    exit_threshold=0.5
):
    beta = np.polyfit(prices[stock_b], prices[stock_a], 1)[0]

    spread = prices[stock_a] - beta * prices[stock_b]
    zscore = (spread - spread.mean()) / spread.std()

    signals = pd.DataFrame(index=prices.index)

    signals["Price_A"] = prices[stock_a]
    signals["Price_B"] = prices[stock_b]
    signals["Spread"] = spread
    signals["Zscore"] = zscore

    signals["Position_A"] = 0
    signals["Position_B"] = 0

    position = 0

    for i in range(len(signals)):
        z = signals["Zscore"].iloc[i]

        if position == 0:
            if z > entry_threshold:
                position = -1
            elif z < -entry_threshold:
                position = 1

        elif position != 0:
            if abs(z) < exit_threshold:
                position = 0

        if position == 1:
            signals.iloc[i, signals.columns.get_loc("Position_A")] = 1
            signals.iloc[i, signals.columns.get_loc("Position_B")] = -1

        elif position == -1:
            signals.iloc[i, signals.columns.get_loc("Position_A")] = -1
            signals.iloc[i, signals.columns.get_loc("Position_B")] = 1

    signals["Return_A"] = prices[stock_a].pct_change()
    signals["Return_B"] = prices[stock_b].pct_change()

    signals["Strategy_Return"] = (
        signals["Position_A"].shift(1) * signals["Return_A"]
        + signals["Position_B"].shift(1) * signals["Return_B"]
    ).fillna(0)

    signals["Cumulative_Return"] = (
        1 + signals["Strategy_Return"]
    ).cumprod()

    total_return = signals["Cumulative_Return"].iloc[-1] - 1

    volatility = signals["Strategy_Return"].std()

    if volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (
            signals["Strategy_Return"].mean() / volatility
        ) * np.sqrt(252)

    running_max = signals["Cumulative_Return"].cummax()
    drawdown = signals["Cumulative_Return"] / running_max - 1
    max_drawdown = drawdown.min()

    number_of_trades = (
        signals["Position_A"].diff().abs().sum() / 2
    )

    return {
        "Entry_Threshold": entry_threshold,
        "Exit_Threshold": exit_threshold,
        "Beta": beta,
        "Total_Return": total_return,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "Number_of_Trades": number_of_trades,
        "Signals": signals
    }


def optimize_thresholds(
    prices,
    stock_a,
    stock_b,
    entry_thresholds=None,
    exit_thresholds=None
):
    if entry_thresholds is None:
        entry_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]

    if exit_thresholds is None:
        exit_thresholds = [0.25, 0.5, 0.75, 1.0]

    optimization_results = []

    for entry in entry_thresholds:
        for exit_ in exit_thresholds:

            if exit_ >= entry:
                continue

            result = backtest_pair_strategy(
                prices=prices,
                stock_a=stock_a,
                stock_b=stock_b,
                entry_threshold=entry,
                exit_threshold=exit_
            )

            optimization_results.append({
                "Entry_Threshold": result["Entry_Threshold"],
                "Exit_Threshold": result["Exit_Threshold"],
                "Beta": result["Beta"],
                "Total_Return": result["Total_Return"],
                "Sharpe_Ratio": result["Sharpe_Ratio"],
                "Max_Drawdown": result["Max_Drawdown"],
                "Number_of_Trades": result["Number_of_Trades"]
            })

    optimization_df = pd.DataFrame(optimization_results)

    optimization_df = optimization_df.sort_values(
        "Sharpe_Ratio",
        ascending=False
    )

    best_params = optimization_df.iloc[0]

    best_result = backtest_pair_strategy(
        prices=prices,
        stock_a=stock_a,
        stock_b=stock_b,
        entry_threshold=best_params["Entry_Threshold"],
        exit_threshold=best_params["Exit_Threshold"]
    )

    return optimization_df, best_result