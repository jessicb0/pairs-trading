import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransactionCostModel:
    
    commission_pct: float = 0.001   # coût par leg, en % de la valeur notionnelle
    bid_ask_spread: float = 0.0005  # demi-spread : coût de crossing
    slippage_pct:   float = 0.0005  # impact marché estimé


    @property
    def total_one_way(self) -> float:

        return self.commission_pct + self.bid_ask_spread + self.slippage_pct

    @property
    def roundtrip(self) -> float:
        return self.total_one_way * 4


def apply_costs_to_backtest(
    signals: pd.DataFrame,
    cost_model: Optional[TransactionCostModel] = None,
    position_col: str = "Position",
    return_col:   str = "Strategy_Return",
) -> pd.DataFrame:

    if cost_model is None:
        cost_model = TransactionCostModel()

    df = signals.copy()

    # Détection des changements de position (entrée + sortie)
    position_change = df[position_col].diff().abs().fillna(0) > 0

    # Coût = roundtrip / 2 à chaque changement (entrée OU sortie)
    # car le roundtrip couvre l'entrée ET la sortie
    cost_per_event = cost_model.total_one_way * 2   # 2 legs simultanés

    df["Transaction_Cost"]  = position_change.astype(float) * cost_per_event
    df["Net_Return"]        = df[return_col] - df["Transaction_Cost"]
    df["Gross_Cumulative"]  = (1 + df[return_col]).cumprod() - 1
    df["Net_Cumulative"]    = (1 + df["Net_Return"]).cumprod() - 1

    _print_cost_impact(df, cost_model)

    return df


def compute_breakeven_cost(
    signals: pd.DataFrame,
    return_col: str = "Strategy_Return",
    position_col: str = "Position",
) -> float:

    gross_return   = signals[return_col].sum()
    n_trades       = (signals[position_col].diff().abs().fillna(0) > 0).sum()

    if n_trades == 0:
        return 0.0

    breakeven = gross_return / n_trades
    print(f"\nRendement brut total       : {gross_return*100:.2f}%")
    print(f"Nombre de changements pos. : {int(n_trades)}")
    print(f"Coût de breakeven / trade  : {breakeven*100:.4f}%")
    print(f"  → La stratégie reste profitable si le coût par trade < {breakeven*100:.4f}%")

    return breakeven


def cost_sensitivity_analysis(
    signals: pd.DataFrame,
    return_col:   str = "Strategy_Return",
    position_col: str = "Position",
    cost_range: list = None,
) -> pd.DataFrame:

    if cost_range is None:
        cost_range = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]

    position_change = signals[position_col].diff().abs().fillna(0) > 0
    gross_returns   = signals[return_col]

    rows = []
    for cost in cost_range:
        cost_series = position_change.astype(float) * cost * 2
        net_returns = gross_returns - cost_series

        sharpe = (
            (net_returns.mean() / net_returns.std() * np.sqrt(252))
            if net_returns.std() > 0 else 0.0
        )
        total_return = net_returns.sum()
        cum = (1 + net_returns).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

        rows.append({
            "Cost_Per_Trade_pct": round(cost * 100, 3),
            "Total_Return_pct":   round(total_return * 100, 2),
            "Sharpe_Ratio":       round(sharpe, 3),
            "Max_Drawdown_pct":   round(max_dd * 100, 2),
        })

    result = pd.DataFrame(rows)
    print("\nANALYSE DE SENSIBILITÉ AUX COÛTS DE TRANSACTION")
    print(result.to_string(index=False))
    return result



def _print_cost_impact(df: pd.DataFrame, model: TransactionCostModel) -> None:
    total_cost   = df["Transaction_Cost"].sum()
    gross_return = df["Strategy_Return"].sum() if "Strategy_Return" in df else 0
    net_return   = df["Net_Return"].sum()
    n_trades     = (df["Transaction_Cost"] > 0).sum()

    print("IMPACT DES COÛTS DE TRANSACTION")
    print(f"Modèle utilisé :")
    print(f"  Commission       : {model.commission_pct*100:.3f}%")
    print(f"  Bid-Ask spread   : {model.bid_ask_spread*100:.3f}%")
    print(f"  Slippage         : {model.slippage_pct*100:.3f}%")
    print(f"  Coût aller-retour total (4 legs) : {model.roundtrip*100:.3f}%")
    print(f"Nombre de trades          : {int(n_trades)}")
    print(f"Coût total prélevé        : {total_cost*100:.2f}%")
    print(f"Rendement brut            : {gross_return*100:.2f}%")
    print(f"Rendement net             : {net_return*100:.2f}%")
