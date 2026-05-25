import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_meta_model(
    signals,
    classifier_result,
    regressor_result,
    sentiment_by_stock,
    stock_a,
    stock_b,
    best_entry,
    proba_threshold=0.55
):
    meta_df = pd.DataFrame(index=signals.index)

    meta_df["Pair_Score"] = np.where(
        signals["Zscore"] > best_entry,
        -1,
        np.where(
            signals["Zscore"] < -best_entry,
            1,
            0
        )
    )

    classifier_model = classifier_result["Model"]
    ml_df = classifier_result["ML_DF"]

    if "Features" in classifier_result:
        features = classifier_result["Features"]
    else:
        features = [
            col for col in ml_df.columns
            if col not in [
                "Target",
                "Target_Convergence",
                "Target_Return",
                "ML_Pred",
                "ML_Convergence_Prob"
            ]
        ]

    meta_df["ML_Convergence_Prob"] = np.nan

    if "ML_Convergence_Prob" in ml_df.columns:
        meta_df.loc[
            ml_df.index,
            "ML_Convergence_Prob"
        ] = ml_df["ML_Convergence_Prob"]
    else:
        meta_df.loc[
            ml_df.index,
            "ML_Convergence_Prob"
        ] = classifier_model.predict_proba(
            ml_df[features]
        )[:, 1]

    reg_model = regressor_result["Model"]

    if "Prediction_DF" in regressor_result:
        prediction_df = regressor_result["Prediction_DF"]

        if "Predicted_Return" in prediction_df.columns:
            meta_df.loc[
                prediction_df.index,
                "Reg_Prediction"
            ] = prediction_df["Predicted_Return"]

        elif "Predicted_Return_J1" in prediction_df.columns:
            meta_df.loc[
                prediction_df.index,
                "Reg_Prediction"
            ] = prediction_df["Predicted_Return_J1"]

        else:
            meta_df["Reg_Prediction"] = np.nan

    else:
        meta_df["Reg_Prediction"] = np.nan

    if meta_df["Reg_Prediction"].isna().all():
        try:
            meta_df.loc[
                ml_df.index,
                "Reg_Prediction"
            ] = reg_model.predict(
                ml_df[features]
            )
        except Exception:
            meta_df["Reg_Prediction"] = 0.0

    sentiment_a = sentiment_by_stock[
        sentiment_by_stock["Ticker"] == stock_a
    ]["Sentiment_Score"].values[0]

    sentiment_b = sentiment_by_stock[
        sentiment_by_stock["Ticker"] == stock_b
    ]["Sentiment_Score"].values[0]

    sentiment_diff = sentiment_a - sentiment_b

    meta_df["Sentiment_Score"] = sentiment_diff

    meta_df["Target"] = (
        signals["Strategy_Return"].shift(-1) > 0
    ).astype(int)

    meta_df = meta_df.dropna()

    x_meta = meta_df[
        [
            "Pair_Score",
            "ML_Convergence_Prob",
            "Reg_Prediction",
            "Sentiment_Score"
        ]
    ]

    y_meta = meta_df["Target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_meta,
        y_meta,
        test_size=0.25,
        shuffle=False
    )

    meta_model = LogisticRegression(
        max_iter=1000
    )

    meta_model.fit(
        x_train,
        y_train
    )

    y_pred = meta_model.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred
    )

    latest_meta_features = x_meta.iloc[[-1]]

    proba_positive_return = meta_model.predict_proba(
        latest_meta_features
    )[0, 1]

    latest_zscore = signals["Zscore"].iloc[-1]

    if proba_positive_return > proba_threshold:

        if latest_zscore < -best_entry:
            final_recommendation = (
                f"LONG {stock_a} / SHORT {stock_b}"
            )

        elif latest_zscore > best_entry:
            final_recommendation = (
                f"SHORT {stock_a} / LONG {stock_b}"
            )

        else:
            final_recommendation = "HOLD"

    else:
        final_recommendation = "HOLD"

    return {
        "Meta_Model": meta_model,
        "Meta_DF": meta_df,
        "X_Meta": x_meta,
        "Y_Meta": y_meta,
        "X_Test": x_test,
        "Y_Test": y_test,
        "Y_Pred": y_pred,
        "Accuracy": accuracy,
        "Classification_Report": report,
        "Probability_Positive_Return": proba_positive_return,
        "Final_Recommendation": final_recommendation,
        "Decision_Threshold": proba_threshold
    }


def build_final_decision_table(
    signals,
    classifier_result,
    regressor_result,
    sentiment_by_stock,
    meta_result,
    stock_a,
    stock_b,
    best_entry,
    decision_threshold=0.55
):
    latest_zscore = signals["Zscore"].iloc[-1]

    classifier_model = classifier_result["Model"]
    ml_df = classifier_result["ML_DF"]

    if "Features" in classifier_result:
        features = classifier_result["Features"]
    else:
        features = [
            col for col in ml_df.columns
            if col not in [
                "Target",
                "Target_Convergence",
                "Target_Return",
                "ML_Pred",
                "ML_Convergence_Prob"
            ]
        ]

    if "ML_Convergence_Prob" in ml_df.columns:
        latest_ml_proba = ml_df["ML_Convergence_Prob"].iloc[-1]
    else:
        latest_ml_proba = classifier_model.predict_proba(
            ml_df[features].iloc[[-1]]
        )[0, 1]

    if "Prediction_DF" in regressor_result:
        prediction_df = regressor_result["Prediction_DF"]

        if "Predicted_Return" in prediction_df.columns:
            latest_reg_prediction = prediction_df[
                "Predicted_Return"
            ].iloc[-1]

        elif "Predicted_Return_J1" in prediction_df.columns:
            latest_reg_prediction = prediction_df[
                "Predicted_Return_J1"
            ].iloc[-1]

        else:
            latest_reg_prediction = np.nan

    else:
        latest_reg_prediction = np.nan

    sentiment_a = sentiment_by_stock[
        sentiment_by_stock["Ticker"] == stock_a
    ]["Sentiment_Score"].values[0]

    sentiment_b = sentiment_by_stock[
        sentiment_by_stock["Ticker"] == stock_b
    ]["Sentiment_Score"].values[0]

    sentiment_diff = sentiment_a - sentiment_b

    pair_score = np.where(
        latest_zscore > best_entry,
        -1,
        np.where(
            latest_zscore < -best_entry,
            1,
            0
        )
    )

    sentiment_score = np.where(
        sentiment_diff > 0,
        1,
        np.where(
            sentiment_diff < 0,
            -1,
            0
        )
    )

    meta_model = meta_result["Meta_Model"]
    x_meta = meta_result["X_Meta"]

    meta_coefficients = pd.DataFrame({
        "Feature": x_meta.columns,
        "Coefficient": meta_model.coef_[0]
    })

    final_decision_meta = pd.DataFrame({
        "Metric": [
            "Latest Zscore",
            "Best Entry Threshold",
            "Pair Score",
            "ML Convergence Probability",
            "Predicted Strategy Return J+1",
            f"Sentiment {stock_a}",
            f"Sentiment {stock_b}",
            "Sentiment Difference",
            "Sentiment Score",
            "Meta Model Accuracy",
            "Meta Model Probability Positive Return",
            "Decision Threshold",
            "Final Recommendation"
        ],
        "Value": [
            latest_zscore,
            best_entry,
            pair_score,
            latest_ml_proba,
            latest_reg_prediction,
            sentiment_a,
            sentiment_b,
            sentiment_diff,
            sentiment_score,
            meta_result["Accuracy"],
            meta_result["Probability_Positive_Return"],
            decision_threshold,
            meta_result["Final_Recommendation"]
        ]
    })

    return final_decision_meta, meta_coefficients