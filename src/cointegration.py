import pandas as pd

from statsmodels.tsa.stattools import coint


def test_cointegration(pairs_df, prices, pvalue_threshold):
    cointegration_results = []

    if pairs_df.empty:
        return pd.DataFrame()

    for _, row in pairs_df.iterrows():

        stock_a = row["Stock_A"]
        stock_b = row["Stock_B"]

        series_a = prices[stock_a].dropna()
        series_b = prices[stock_b].dropna()

        common_index = series_a.index.intersection(series_b.index)

        series_a = series_a.loc[common_index]
        series_b = series_b.loc[common_index]

        score, pvalue, _ = coint(series_a, series_b)

        cointegration_results.append({
            "Cluster": row["Cluster"],
            "Stock_A": stock_a,
            "Stock_B": stock_b,
            "Correlation": row["Correlation"],
            "PValue": pvalue,
            "Cointegrated": pvalue < pvalue_threshold
        })

    cointegration_df = pd.DataFrame(cointegration_results)

    if not cointegration_df.empty:
        cointegration_df = cointegration_df.sort_values("PValue")

    return cointegration_df


def select_best_pairs(cointegration_df):
    if cointegration_df.empty:
        return pd.DataFrame()

    best_pairs = cointegration_df[
        cointegration_df["Cointegrated"] == True
    ].copy()

    return best_pairs