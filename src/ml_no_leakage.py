"""
Fix Data Leakage — Split Temporel Strict pour Modèles ML
---------------------------------------------------------
Remplace le train/test split aléatoire (sklearn défaut) par un split
chronologique strict, obligatoire pour toute donnée financière.

Problème original :
  Si train_test_split(shuffle=True) est utilisé, le modèle voit des
  données du "futur" lors de l'entraînement → métriques artificiellement
  bonnes (data leakage).

Solution :
  - Split temporel pur (train = passé, test = futur)
  - TimeSeriesSplit pour la validation croisée
  - Vérification automatique de l'absence de leakage
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Feature engineering (à adapter selon src/ml_models.py existant)
# ---------------------------------------------------------------------------

def build_ml_features(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features ML en évitant tout look-ahead bias :
    seules des informations disponibles au moment t sont utilisées.

    Features :
      - Z-score courant et ses lags (J-1, J-2, J-3)
      - Variation du spread (J-1, J-2)
      - Rendements passés des deux actions (rolling mean 5j)
      - Volatilité du Z-score sur 10 jours
    """
    df = signals.copy()

    required = ["Zscore", "Spread", "Return_A", "Return_B"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans signals : '{col}'")

    # Lags du Z-score (toujours disponibles à t)
    df["Zscore_Lag1"] = df["Zscore"].shift(1)
    df["Zscore_Lag2"] = df["Zscore"].shift(2)
    df["Zscore_Lag3"] = df["Zscore"].shift(3)

    # Variation du spread
    df["Spread_Diff1"] = df["Spread"].diff(1)
    df["Spread_Diff2"] = df["Spread"].diff(2)

    # Rendements passés (rolling mean 5 jours, shift(1) pour éviter le leakage)
    df["Return_A_MA5"] = df["Return_A"].shift(1).rolling(5).mean()
    df["Return_B_MA5"] = df["Return_B"].shift(1).rolling(5).mean()

    # Volatilité récente du Z-score
    df["Zscore_Std10"] = df["Zscore"].shift(1).rolling(10).std()

    # Cible classification : le spread converge-t-il demain ?
    # (Zscore diminue en valeur absolue)
    df["Target_Convergence"] = (
        df["Zscore"].abs().shift(-1) < df["Zscore"].abs()
    ).astype(int)

    # Cible régression : rendement de la stratégie à J+1
    if "Strategy_Return" in df.columns:
        df["Target_Return"] = df["Strategy_Return"].shift(-1)

    return df.dropna()


FEATURE_COLS = [
    "Zscore", "Zscore_Lag1", "Zscore_Lag2", "Zscore_Lag3",
    "Spread_Diff1", "Spread_Diff2",
    "Return_A_MA5", "Return_B_MA5",
    "Zscore_Std10",
]


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologique strict : les données de test sont TOUJOURS
    postérieures aux données d'entraînement.

    Parameters
    ----------
    df        : DataFrame avec index DatetimeIndex trié
    test_size : fraction réservée au test (défaut 20%)

    Returns
    -------
    (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    _verify_no_leakage(train_df, test_df)

    print(f"Split temporel :")
    print(f"  Train : {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} obs.)")
    print(f"  Test  : {test_df.index[0].date()}  → {test_df.index[-1].date()}  ({len(test_df)} obs.)")

    return train_df, test_df


def _verify_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Vérifie qu'aucune date du train n'est postérieure au test."""
    if train.index.max() >= test.index.min():
        raise ValueError(
            f"DATA LEAKAGE DÉTECTÉ ❌\n"
            f"  Max date train : {train.index.max()}\n"
            f"  Min date test  : {test.index.min()}\n"
            "  Le train contient des données postérieures au test."
        )
    print("  Vérification anti-leakage : OK ✅")


# ---------------------------------------------------------------------------
# Classifieur sans leakage
# ---------------------------------------------------------------------------

def train_classifier_no_leakage(
    signals: pd.DataFrame,
    test_size: float = 0.2,
    n_splits_cv: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict:
    """
    Entraîne un Random Forest avec split temporel strict.
    Inclut une validation croisée temporelle (TimeSeriesSplit).

    Returns
    -------
    dict avec le modèle, les métriques OOS, le rapport de classification,
    l'importance des features, et le DataFrame d'évaluation.
    """
    print("\n" + "=" * 60)
    print("CLASSIFIEUR — SPLIT TEMPOREL STRICT")
    print("=" * 60)

    df = build_ml_features(signals)
    train_df, test_df = temporal_train_test_split(df, test_size)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["Target_Convergence"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["Target_Convergence"]

    # --- Validation croisée temporelle sur le train ---
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)
    clf  = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        clf.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        val_pred  = clf.predict(X_train.iloc[val_idx])
        val_score = accuracy_score(y_train.iloc[val_idx], val_pred)
        cv_scores.append(val_score)
        print(f"  CV fold {fold+1}/{n_splits_cv} accuracy : {val_score:.4f}")

    print(f"  CV accuracy moyenne : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # --- Entraînement final sur tout le train ---
    clf.fit(X_train, y_train)

    # --- Évaluation OOS (out-of-sample) sur le test ---
    y_pred      = clf.predict(X_test)
    y_proba     = clf.predict_proba(X_test)[:, 1]
    oos_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy OOS (hors-échantillon) : {oos_accuracy:.4f}")
    print(f"Accuracy IN-sample              : {accuracy_score(y_train, clf.predict(X_train)):.4f}")
    print("  → Un écart important indique du surapprentissage.")
    print("\nRapport de classification OOS :")
    print(classification_report(y_test, y_pred))

    # DataFrame résultat
    ml_df = test_df.copy()
    ml_df["ML_Pred"]            = y_pred
    ml_df["ML_Convergence_Prob"] = y_proba

    feature_importance = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "Model":                 clf,
        "Accuracy":              oos_accuracy,
        "CV_Scores":             cv_scores,
        "Classification_Report": classification_report(y_test, y_pred),
        "Feature_Importance":    feature_importance,
        "ML_DF":                 ml_df,
        "Train_Period":          (train_df.index[0], train_df.index[-1]),
        "Test_Period":           (test_df.index[0],  test_df.index[-1]),
    }


# ---------------------------------------------------------------------------
# Régresseur sans leakage
# ---------------------------------------------------------------------------

def train_regressor_no_leakage(
    signals: pd.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict:
    """
    Entraîne un Random Forest Regressor avec split temporel strict
    pour prédire le rendement à J+1.
    """
    print("\n" + "=" * 60)
    print("RÉGRESSEUR — SPLIT TEMPOREL STRICT")
    print("=" * 60)

    if "Strategy_Return" not in signals.columns:
        raise ValueError("'Strategy_Return' manquant dans signals. Lancez d'abord run_simple_backtest().")

    df = build_ml_features(signals)
    train_df, test_df = temporal_train_test_split(df, test_size)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["Target_Return"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["Target_Return"]

    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"MAE  (OOS) : {mae:.6f}")
    print(f"RMSE (OOS) : {rmse:.6f}")
    print(f"R²   (OOS) : {r2:.4f}")
    r2_train = r2_score(y_train, reg.predict(X_train))
    print(f"R²   (IN)  : {r2_train:.4f}  (si >> R² OOS → surapprentissage)")

    prediction_df = test_df.copy()
    prediction_df["Predicted_Return"] = y_pred
    prediction_df["Actual_Return"]    = y_test.values

    feature_importance = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": reg.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "Model":              reg,
        "MAE":                mae,
        "RMSE":               rmse,
        "R2":                 r2,
        "Feature_Importance": feature_importance,
        "Prediction_DF":      prediction_df,
    }
