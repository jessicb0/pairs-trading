import os
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_dendrogram(linkage_matrix, labels, save_path=None):
    plt.figure(figsize=(16, 8))

    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90
    )

    plt.title("Dow Jones - Hierarchical Clustering")
    plt.xlabel("Stocks")
    plt.ylabel("Distance")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_best_pair(prices, stock_a, stock_b, save_path=None):
    normalized_a = prices[stock_a] / prices[stock_a].iloc[0]
    normalized_b = prices[stock_b] / prices[stock_b].iloc[0]

    plt.figure(figsize=(12, 6))

    plt.plot(normalized_a, label=stock_a)
    plt.plot(normalized_b, label=stock_b)

    plt.title(f"Best Pair: {stock_a} / {stock_b}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")

    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    

def plot_clusters_pca(returns, cluster_df, save_path=None):
    X = returns.T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_plot_df = cluster_df.copy()
    cluster_plot_df["PCA_1"] = X_pca[:, 0]
    cluster_plot_df["PCA_2"] = X_pca[:, 1]

    plt.figure(figsize=(12, 8))

    plt.scatter(
        cluster_plot_df["PCA_1"],
        cluster_plot_df["PCA_2"],
        c=cluster_plot_df["Cluster"],
        s=180
    )

    for _, row in cluster_plot_df.iterrows():
        plt.text(
            row["PCA_1"] + 0.03,
            row["PCA_2"] + 0.03,
            row["Ticker"],
            fontsize=10
        )

    plt.title("Dow Jones - Visualisation des clusters en 2D")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    return cluster_plot_df

def plot_spread(spread, title="Spread", save_path=None):
    plt.figure(figsize=(14, 6))

    plt.plot(spread)

    plt.axhline(
        spread.mean(),
        linestyle="--", 
        color="black"
    )

    plt.title(title)
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_zscore(
    zscore,
    entry_threshold=2.0,
    exit_threshold=0.5,
    title="Z-Score",
    save_path=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(zscore)

    plt.axhline(entry_threshold, linestyle="--", color="red")
    plt.axhline(-entry_threshold, linestyle="--", color="red")
    plt.axhline(0)

    plt.title(title)
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_cumulative_return(
    signals,
    title="Backtest Pair Trading",
    save_path=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(signals["Cumulative_Return"])

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    
def plot_optimized_backtest(
    best_signals,
    stock_a,
    stock_b,
    best_entry,
    best_exit,
    save_path=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(best_signals["Cumulative_Return"])

    plt.title(
        f"Backtest optimisé : {stock_a} / {stock_b} "
        f"(entry={best_entry}, exit={best_exit})"
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_optimized_zscore(
    best_signals,
    stock_a,
    stock_b,
    best_entry,
    best_exit,
    save_path=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(best_signals["Zscore"])

    plt.axhline(best_entry, linestyle="--", label="Entry Short A / Long B", color="green")
    plt.axhline(-best_entry, linestyle="--", label="Entry Long A / Short B", color="green")
    plt.axhline(best_exit, linestyle=":", color="red")
    plt.axhline(-best_exit, linestyle=":", color="red")
    plt.axhline(0)

    plt.title(f"Z-score optimisé : {stock_a} / {stock_b}")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    
def plot_benchmark_comparison(
    benchmark_df,
    stock_a,
    stock_b,
    save_path=None
):
    plt.figure(figsize=(14, 7))

    plt.plot(
        benchmark_df["Pair_Trading"],
        label=f"Pair Trading {stock_a}/{stock_b}",
        linewidth=3
    )

    plt.plot(
        benchmark_df[f"Buy_Hold_{stock_a}"],
        label=f"Buy & Hold {stock_a}",
        linestyle="--"
    )

    plt.plot(
        benchmark_df[f"Buy_Hold_{stock_b}"],
        label=f"Buy & Hold {stock_b}",
        linestyle=":"
    )

    plt.title("Pair Trading vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_ml_filtered_strategy(
    best_signals,
    ml_filtered_signals,
    save_path=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(
        best_signals["Cumulative_Return"],
        label="Pair trading optimisé"
    )

    plt.plot(
        ml_filtered_signals["ML_Cumulative_Return"],
        label="Pair trading filtré par ML"
    )

    plt.title("Comparaison : stratégie z-score vs stratégie z-score + ML")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    
    
def plot_regression_predictions(
    prediction_df,
    save_path=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(
        prediction_df.index,
        prediction_df["Real_Return_J1"],
        label="Rendement réel J+1"
    )

    plt.plot(
        prediction_df.index,
        prediction_df["Predicted_Return_J1"],
        label="Rendement prédit J+1"
    )

    plt.title(
        "Prédiction du rendement de la stratégie à J+1"
    )

    plt.xlabel("Date")
    plt.ylabel("Return")

    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

    plt.show()
    
def plot_lstm_training_history(lstm_result, save_path=None):
    history = lstm_result["History"]

    plt.figure(figsize=(12, 5))

    plt.plot(
        history["loss"],
        label="Train loss"
    )

    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_lstm_predictions(lstm_result, save_path=None):
    prediction_df = lstm_result["Prediction_DF"]

    plt.figure(figsize=(14, 6))

    plt.plot(
        prediction_df.index,
        prediction_df["Convergence_Probability"],
        label="LSTM convergence probability"
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Decision threshold"
    )

    plt.title("LSTM Predicted Probability of Spread Convergence")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    
def plot_lstm_regression_predictions(lstm_reg_result, save_path=None):
    prediction_df = lstm_reg_result["Prediction_DF"]

    plt.figure(figsize=(14, 6))

    plt.plot(
        prediction_df.index,
        prediction_df["Actual_Return_J1"],
        label="Rendement réel J+1"
    )

    plt.plot(
        prediction_df.index,
        prediction_df["Predicted_Return_J1_LSTM"],
        label="Rendement prédit J+1 - LSTM"
    )

    plt.title("LSTM Regression - Prédiction du rendement J+1")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()