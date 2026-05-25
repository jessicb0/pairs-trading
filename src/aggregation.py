import pandas as pd


def build_final_recommendation(
    signals,
    classifier_result_ml,
    sentiment_by_stock,
    stock_a,
    stock_b,
    entry_threshold=2.0
):
    latest_zscore = signals["Zscore"].iloc[-1]

    model = classifier_result_ml["Model"]
    ml_df = classifier_result_ml["ML_DF"]
    features = classifier_result_ml["Features"]

    latest_ml_prediction = model.predict(
        ml_df[features].iloc[[-1]]
    )[0]

    sentiment_a = sentiment_by_stock[
        sentiment_by_stock["Ticker"] == stock_a
    ]["Sentiment_Score"].values[0]

    sentiment_b = sentiment_by_stock[
        sentiment_by_stock["Ticker"] == stock_b
    ]["Sentiment_Score"].values[0]

    sentiment_diff = sentiment_a - sentiment_b

    if latest_zscore > entry_threshold and latest_ml_prediction == 1:
        recommendation = f"SHORT {stock_a} / LONG {stock_b}"

    elif latest_zscore < -entry_threshold and latest_ml_prediction == 1:
        recommendation = f"LONG {stock_a} / SHORT {stock_b}"

    else:
        recommendation = "HOLD"

    final_table = pd.DataFrame({
        "Stock_A": [stock_a],
        "Stock_B": [stock_b],
        "Latest_Zscore": [latest_zscore],
        "ML_Prediction_Convergence": [latest_ml_prediction],
        "Sentiment_A": [sentiment_a],
        "Sentiment_B": [sentiment_b],
        "Sentiment_Difference": [sentiment_diff],
        "Final_Recommendation": [recommendation]
    })

    return final_table