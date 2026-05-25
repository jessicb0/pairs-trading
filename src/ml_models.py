import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier


FEATURES = [
    "Zscore",
    "Zscore_Lag1",
    "Spread",
    "Spread_Lag1",
    "Spread_Change",
    "Return_A_Lag1",
    "Return_B_Lag1"
]


def prepare_ml_dataset(signals):
    ml_df = signals.copy()

    ml_df["Zscore_Lag1"] = ml_df["Zscore"].shift(1)

    ml_df["Spread_Lag1"] = ml_df["Spread"].shift(1)

    ml_df["Spread_Change"] = ml_df["Spread"].diff()

    ml_df["Return_A_Lag1"] = ml_df["Return_A"].shift(1)

    ml_df["Return_B_Lag1"] = ml_df["Return_B"].shift(1)

    ml_df["Zscore_Next"] = ml_df["Zscore"].shift(-1)

    ml_df["Target"] = (
        abs(ml_df["Zscore_Next"])
        < abs(ml_df["Zscore"])
    ).astype(int)

    ml_df = ml_df.dropna()

    return ml_df


def train_classifier(signals):
    ml_df = prepare_ml_dataset(signals)

    X = ml_df[FEATURES]
    y = ml_df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=4
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred
    )

    feature_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    return {
        "Model": model,
        "ML_DF": ml_df,
        "Features": FEATURES,
        "Accuracy": accuracy,
        "Classification_Report": report,
        "Feature_Importance": feature_importance,
        "X_Test": X_test,
        "Y_Test": y_test,
        "Y_Pred": y_pred
    }
    
def train_regressor(signals):
    ml_df = prepare_ml_dataset(signals)

    ml_df["Target_Strategy_Return_Next"] = (
        signals["Strategy_Return"].shift(-1)
    )

    ml_df = ml_df.dropna()

    X_reg = ml_df[FEATURES]

    y_reg = ml_df[
        "Target_Strategy_Return_Next"
    ]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.25,
        shuffle=False
    )

    reg_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42
    )

    reg_model.fit(
        X_reg_train,
        y_reg_train
    )

    y_reg_pred = reg_model.predict(
        X_reg_test
    )

    mae = mean_absolute_error(
        y_reg_test,
        y_reg_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_reg_test,
            y_reg_pred
        )
    )

    r2 = r2_score(
        y_reg_test,
        y_reg_pred
    )

    prediction_df = pd.DataFrame({
        "Real_Return_J1": y_reg_test.values,
        "Predicted_Return_J1": y_reg_pred
    }, index=y_reg_test.index)

    return {
        "Model": reg_model,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Prediction_DF": prediction_df,
        "ML_DF": ml_df,
        "X_Test": X_reg_test,
        "Y_Test": y_reg_test,
        "Y_Pred": y_reg_pred
    }
    
def train_xgboost_classifier(signals):
    ml_df = prepare_ml_dataset(signals)

    X = ml_df[FEATURES]

    y = ml_df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    y_pred = xgb_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred
    )

    feature_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": xgb_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    return {
        "Model": xgb_model,
        "Accuracy": accuracy,
        "Classification_Report": report,
        "Feature_Importance": feature_importance,
        "ML_DF": ml_df,
        "Features": FEATURES
    }