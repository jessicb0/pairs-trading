import numpy as np
import pandas as pd


def build_benchmark_df(prices, best_signals, stock_a, stock_b):
    benchmark_df = pd.DataFrame(index=prices.index)

    benchmark_df["Pair_Trading"] = best_signals["Cumulative_Return"]

    benchmark_df[f"Buy_Hold_{stock_a}"] = (
        prices[stock_a] / prices[stock_a].iloc[0]
    )

    benchmark_df[f"Buy_Hold_{stock_b}"] = (
        prices[stock_b] / prices[stock_b].iloc[0]
    )

    return benchmark_df


def compute_strategy_performance(benchmark_df):
    performance = pd.DataFrame(columns=[
        "Total Return %",
        "Volatility %",
        "Sharpe Ratio",
        "Max Drawdown %"
    ])

    for strategy in benchmark_df.columns:
        returns = benchmark_df[strategy].pct_change().dropna()

        total_return = (
            benchmark_df[strategy].iloc[-1] - 1
        ) * 100

        volatility = returns.std() * np.sqrt(252) * 100

        if returns.std() == 0:
            sharpe = 0
        else:
            sharpe = (
                returns.mean() / returns.std()
            ) * np.sqrt(252)

        cumulative = benchmark_df[strategy]

        running_max = cumulative.cummax()

        drawdown = cumulative / running_max - 1

        max_drawdown = drawdown.min() * 100

        performance.loc[strategy] = [
            round(total_return, 2),
            round(volatility, 2),
            round(sharpe, 2),
            round(max_drawdown, 2)
        ]

    return performance


def apply_ml_filter(
    signals,
    classifier_result,
    stock_a,
    stock_b,
    entry_threshold,
    proba_threshold=0.55
):
    ml_filtered_signals = signals.copy()

    model = classifier_result["Model"]
    ml_df = classifier_result["ML_DF"]
    features = classifier_result["Features"]

    ml_filtered_signals["ML_Convergence_Prob"] = np.nan

    ml_filtered_signals.loc[
        ml_df.index,
        "ML_Convergence_Prob"
    ] = model.predict_proba(
        ml_df[features]
    )[:, 1]

    ml_filtered_signals["ML_Position_A"] = 0
    ml_filtered_signals["ML_Position_B"] = 0
    ml_filtered_signals["ML_Signal"] = "HOLD"

    condition_short = (
        (ml_filtered_signals["Zscore"] > entry_threshold)
        & (
            ml_filtered_signals["ML_Convergence_Prob"]
            > proba_threshold
        )
    )

    condition_long = (
        (ml_filtered_signals["Zscore"] < -entry_threshold)
        & (
            ml_filtered_signals["ML_Convergence_Prob"]
            > proba_threshold
        )
    )

    ml_filtered_signals.loc[
        condition_short,
        "ML_Position_A"
    ] = -1

    ml_filtered_signals.loc[
        condition_short,
        "ML_Position_B"
    ] = 1

    ml_filtered_signals.loc[
        condition_short,
        "ML_Signal"
    ] = f"SHORT_{stock_a}_LONG_{stock_b}"

    ml_filtered_signals.loc[
        condition_long,
        "ML_Position_A"
    ] = 1

    ml_filtered_signals.loc[
        condition_long,
        "ML_Position_B"
    ] = -1

    ml_filtered_signals.loc[
        condition_long,
        "ML_Signal"
    ] = f"LONG_{stock_a}_SHORT_{stock_b}"

    ml_filtered_signals["ML_Strategy_Return"] = (
        ml_filtered_signals["ML_Position_A"].shift(1)
        * ml_filtered_signals["Return_A"]
        + ml_filtered_signals["ML_Position_B"].shift(1)
        * ml_filtered_signals["Return_B"]
    )

    ml_filtered_signals["ML_Strategy_Return"] = (
        ml_filtered_signals["ML_Strategy_Return"].fillna(0)
    )

    ml_filtered_signals["ML_Cumulative_Return"] = (
        1 + ml_filtered_signals["ML_Strategy_Return"]
    ).cumprod()

    return ml_filtered_signals


def compute_ml_filtered_performance(ml_filtered_signals):
    ml_total_return = (
        ml_filtered_signals["ML_Cumulative_Return"].iloc[-1] - 1
    )

    ml_vol = ml_filtered_signals["ML_Strategy_Return"].std()

    if ml_vol == 0:
        ml_sharpe = 0
    else:
        ml_sharpe = (
            ml_filtered_signals["ML_Strategy_Return"].mean()
            / ml_vol
        ) * np.sqrt(252)

    ml_running_max = (
        ml_filtered_signals["ML_Cumulative_Return"].cummax()
    )

    ml_drawdown = (
        ml_filtered_signals["ML_Cumulative_Return"]
        / ml_running_max
        - 1
    )

    ml_max_drawdown = ml_drawdown.min()

    return {
        "ML_Filtered_Total_Return": ml_total_return,
        "ML_Filtered_Sharpe_Ratio": ml_sharpe,
        "ML_Filtered_Max_Drawdown": ml_max_drawdown
    }