import os
import pandas as pd

def save_outputs(final_table, cointegration_df, performance_result):
    os.makedirs("outputs", exist_ok=True)

    final_table.to_csv(
        "outputs/recommendations.csv",
        index=False
    )

    cointegration_df.to_csv(
        "outputs/cointegration_results.csv",
        index=False
    )

    performance_table = final_table.copy()

    performance_table["Total_Return_%"] = round(
        performance_result["Total_Return"] * 100,
        2
    )

    performance_table["Sharpe_Ratio"] = round(
        performance_result["Sharpe_Ratio"],
        2
    )

    performance_table["Max_Drawdown_%"] = round(
        performance_result["Max_Drawdown"] * 100,
        2
    )

    performance_table["Number_of_Trades"] = int(
        performance_result["Number_of_Trades"]
    )

    performance_table.to_csv(
        "outputs/summary_results.csv",
        index=False
    )


def build_summary_results(
    best_result,
    ml_metrics,
    ml_filtered_signals
):
    summary_results = pd.DataFrame({
        "Strategy": [
            "Pair Trading Optimisé",
            "Pair Trading + ML Filter"
        ],

        "Total Return %": [
            round(
                best_result["Total_Return"] * 100,
                2
            ),

            round(
                ml_metrics["ML_Filtered_Total_Return"] * 100,
                2
            )
        ],

        "Sharpe Ratio": [
            round(
                best_result["Sharpe_Ratio"],
                2
            ),

            round(
                ml_metrics["ML_Filtered_Sharpe_Ratio"],
                2
            )
        ],

        "Max Drawdown %": [
            round(
                best_result["Max_Drawdown"] * 100,
                2
            ),

            round(
                ml_metrics["ML_Filtered_Max_Drawdown"] * 100,
                2
            )
        ],

        "Number of Trades": [
            int(
                best_result["Number_of_Trades"]
            ),

            int(
                (
                    ml_filtered_signals["ML_Position_A"]
                    .diff()
                    .abs()
                    .sum()
                ) / 2
            )
        ]
    })

    return summary_results