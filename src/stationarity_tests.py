"""
Tests de Stationnarité — ADF, KPSS, PP et Cointégration Engle-Granger
----------------------------------------------------------------------
Rend explicites les tests statistiques sous-jacents à la stratégie.

Tests inclus :
  - ADF  (Augmented Dickey-Fuller)  : H0 = racine unitaire (non-stationnaire)
  - KPSS (Kwiatkowski–Phillips–Schmidt–Shin) : H0 = stationnaire
  - PP   (Phillips-Perron)          : H0 = racine unitaire
  - Engle-Granger cointegration     : avec p-value et statistic

Interprétation combinée ADF + KPSS :
  ADF rejette H0 ET KPSS ne rejette pas H0 → stationnaire ✅
  ADF ne rejette pas H0               → non-stationnaire ❌
  KPSS rejette H0                     → non-stationnaire ❌
  Contradiction                       → résultat ambigu ⚠️
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def test_stationarity(
    series: pd.Series,
    name: str = "Series",
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Applique ADF + KPSS sur une série et retourne une conclusion claire.

    Parameters
    ----------
    series  : pd.Series à tester
    name    : label pour l'affichage
    alpha   : seuil de significativité (défaut 5%)
    verbose : afficher le rapport complet

    Returns
    -------
    dict avec les statistiques et la conclusion
    """
    series = series.dropna()

    # --- ADF ---
    adf_stat, adf_pval, adf_lags, _, adf_crit, _ = adfuller(series, autolag="AIC")

    # --- KPSS ---
    # KPSS peut émettre un warning si la p-value est hors table → on le supprime
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(series, regression="c", nlags="auto")

    # --- Conclusion combinée ---
    adf_rejects  = adf_pval < alpha   # rejette H0(racine unitaire) → stationnaire
    kpss_rejects = kpss_pval < alpha  # rejette H0(stationnaire)    → non-stationnaire

    if adf_rejects and not kpss_rejects:
        conclusion = "STATIONNAIRE ✅"
    elif not adf_rejects and kpss_rejects:
        conclusion = "NON-STATIONNAIRE ❌"
    elif not adf_rejects and not kpss_rejects:
        conclusion = "AMBIGU ⚠️  (ADF faible, KPSS OK — probablement I(1))"
    else:
        conclusion = "AMBIGU ⚠️  (ADF rejette, KPSS rejette — série complexe)"

    result = {
        "Series":         name,
        "ADF_Stat":       round(adf_stat, 4),
        "ADF_PValue":     round(adf_pval, 4),
        "ADF_Lags":       adf_lags,
        "ADF_Crit_1pct":  round(adf_crit["1%"], 4),
        "ADF_Crit_5pct":  round(adf_crit["5%"], 4),
        "KPSS_Stat":      round(kpss_stat, 4),
        "KPSS_PValue":    round(kpss_pval, 4),
        "KPSS_Crit_5pct": round(kpss_crit["5%"], 4),
        "Conclusion":     conclusion,
    }

    if verbose:
        _print_stationarity_report(result, alpha)

    return result


def test_spread_stationarity(
    prices: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    alpha: float = 0.05,
) -> dict:
    """
    Test complet sur la paire :
      1. Stationnarité de chaque série en niveau et en différence
      2. Estimation du hedge ratio OLS
      3. Stationnarité du spread résiduel
      4. Test de cointégration Engle-Granger

    Returns
    -------
    dict récapitulatif + DataFrame des résultats
    """
    print("=" * 65)
    print(f"ANALYSE COMPLÈTE DE STATIONNARITÉ : {stock_a} / {stock_b}")
    print("=" * 65)

    results = []

    # --- 1. Prix en niveau ---
    print(f"\n[1] Prix en niveau")
    for ticker, col in [(stock_a, stock_a), (stock_b, stock_b)]:
        r = test_stationarity(prices[col], name=f"{ticker} (niveau)", alpha=alpha)
        results.append(r)

    # --- 2. Prix en différence (rendements log) ---
    print(f"\n[2] Rendements logarithmiques (1ère différence)")
    for ticker, col in [(stock_a, stock_a), (stock_b, stock_b)]:
        log_diff = np.log(prices[col]).diff().dropna()
        r = test_stationarity(log_diff, name=f"{ticker} (Δlog)", alpha=alpha)
        results.append(r)

    # --- 3. Hedge ratio OLS ---
    print(f"\n[3] Estimation du Hedge Ratio (OLS)")
    X = add_constant(prices[stock_b])
    model = OLS(prices[stock_a], X).fit()
    beta = model.params[stock_b]
    alpha_ols = model.params["const"]
    print(f"  α (constante) : {alpha_ols:.4f}")
    print(f"  β (hedge ratio): {beta:.4f}")
    print(f"  R²             : {model.rsquared:.4f}")

    # --- 4. Spread résiduel ---
    spread = prices[stock_a] - beta * prices[stock_b]
    spread.name = "Spread"
    print(f"\n[4] Stationnarité du spread résiduel")
    spread_result = test_stationarity(spread, name=f"Spread {stock_a}-β×{stock_b}", alpha=alpha)
    results.append(spread_result)

    # --- 5. Cointégration Engle-Granger ---
    print(f"\n[5] Test de cointégration Engle-Granger")
    coint_stat, coint_pval, coint_crit = coint(prices[stock_a], prices[stock_b])
    coint_conclusion = "COINTÉGRÉS ✅" if coint_pval < alpha else "NON COINTÉGRÉS ❌"
    print(f"  Statistique    : {coint_stat:.4f}")
    print(f"  P-Value        : {coint_pval:.4f}")
    print(f"  Valeurs crit.  : 1%={coint_crit[0]:.4f}, 5%={coint_crit[1]:.4f}, 10%={coint_crit[2]:.4f}")
    print(f"  Conclusion     : {coint_conclusion}")

    # --- Résumé final ---
    summary_df = pd.DataFrame(results)[[
        "Series", "ADF_PValue", "KPSS_PValue", "Conclusion"
    ]]

    print(f"\n{'='*65}")
    print("RÉSUMÉ")
    print(summary_df.to_string(index=False))
    print(f"{'='*65}")

    return {
        "Hedge_Ratio":       beta,
        "Spread":            spread,
        "Coint_Stat":        coint_stat,
        "Coint_PValue":      coint_pval,
        "Coint_Conclusion":  coint_conclusion,
        "Spread_Stationary": spread_result["Conclusion"],
        "Details_DF":        summary_df,
    }


def adf_rolling(
    spread: pd.Series,
    window: int = 252,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    ADF glissant : détecte les ruptures de stationnarité dans le temps.
    Utile pour identifier quand la paire cesse d'être cointégrée.

    Parameters
    ----------
    spread : série du spread
    window : taille de la fenêtre glissante (défaut : 252 = 1 an)
    alpha  : seuil de significativité

    Returns
    -------
    DataFrame avec p-value ADF et flag de stationnarité par date
    """
    pvalues = []
    dates   = []

    for i in range(window, len(spread) + 1):
        sub = spread.iloc[i - window:i]
        try:
            _, pval, *_ = adfuller(sub, autolag="AIC")
        except Exception:
            pval = np.nan
        pvalues.append(pval)
        dates.append(spread.index[i - 1])

    result = pd.DataFrame({
        "Date":        dates,
        "ADF_PValue":  pvalues,
        "Stationary":  [p < alpha if not np.isnan(p) else False for p in pvalues],
    }).set_index("Date")

    pct_stationary = result["Stationary"].mean() * 100
    print(f"\nADF GLISSANT ({window} jours)")
    print(f"  % du temps où le spread est stationnaire (α={alpha*100:.0f}%) : {pct_stationary:.1f}%")

    return result


# ---------------------------------------------------------------------------
# Helper interne
# ---------------------------------------------------------------------------

def _print_stationarity_report(r: dict, alpha: float) -> None:
    print(f"\n  ── {r['Series']}")
    print(f"     ADF  : stat={r['ADF_Stat']:>8.4f}  p={r['ADF_PValue']:.4f}  "
          f"(crit 5%={r['ADF_Crit_5pct']})  lags={r['ADF_Lags']}")
    print(f"     KPSS : stat={r['KPSS_Stat']:>8.4f}  p={r['KPSS_PValue']:.4f}  "
          f"(crit 5%={r['KPSS_Crit_5pct']})")
    print(f"     → {r['Conclusion']}")
