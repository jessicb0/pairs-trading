import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def compute_correlation_matrix(returns):
    return returns.corr()


def compute_distance_matrix(corr_matrix):
    return 1 - corr_matrix


def compute_hierarchical_clustering(distance_matrix):
    condensed_distance = squareform(distance_matrix)

    linkage_matrix = linkage(
        condensed_distance,
        method="average"
    )

    return linkage_matrix


def compute_clusters(returns, distance_threshold):
    corr_matrix = compute_correlation_matrix(returns)
    distance_matrix = compute_distance_matrix(corr_matrix)
    linkage_matrix = compute_hierarchical_clustering(distance_matrix)

    cluster_labels = fcluster(
        linkage_matrix,
        t=distance_threshold,
        criterion="distance"
    )

    cluster_df = pd.DataFrame({
        "Ticker": returns.columns,
        "Cluster": cluster_labels
    })

    cluster_df = cluster_df.sort_values("Cluster")

    return cluster_df, corr_matrix, linkage_matrix


def find_correlated_pairs(cluster_df, returns, corr_threshold):
    pairs = []

    for cluster in sorted(cluster_df["Cluster"].unique()):

        stocks = cluster_df[
            cluster_df["Cluster"] == cluster
        ]["Ticker"].tolist()

        if len(stocks) < 2:
            continue

        sub_corr = returns[stocks].corr()

        for i in range(len(stocks)):
            for j in range(i + 1, len(stocks)):

                stock_a = stocks[i]
                stock_b = stocks[j]

                corr_value = sub_corr.loc[stock_a, stock_b]

                if corr_value >= corr_threshold:
                    pairs.append({
                        "Cluster": cluster,
                        "Stock_A": stock_a,
                        "Stock_B": stock_b,
                        "Correlation": corr_value
                    })

    pairs_df = pd.DataFrame(pairs)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            "Correlation",
            ascending=False
        )

    return pairs_df