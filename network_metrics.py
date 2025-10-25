#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import networkx as nx
import community  # python-louvain
import bct  # brain connectivity toolbox (pip install bctpy)

# ==== PATHS ====
base_dir = "/media/projects/earlyexp/data/mri/postproc/smooths/adjacency_matrices"
output_csv = os.path.join(base_dir, "network_metrics_summary.csv")

# ==== HELPERS ====

def compute_graph_metrics(adj_df):
    """Compute modularity, integration (global efficiency),
       and segregation (mean clustering coefficient) for one subject."""
    # convert Fisher-Z to correlation (optional: tanh)
    corr = np.tanh(adj_df.values)

    # zero diagonal
    np.fill_diagonal(corr, 0)

    # keep only positive edges
    corr[corr < 0] = 0

    # build undirected weighted graph
    G = nx.from_numpy_array(corr)

    # --- Whole-brain metrics ---
    # Louvain modularity
    partition = community.best_partition(G)
    modularity = community.modularity(partition, G)

    # Global efficiency
    eff_global = bct.efficiency_wei(corr)

    # Mean clustering coefficient
    clustering = bct.clustering_coef_wu(corr)
    seg_global = np.mean(clustering)

    return modularity, eff_global, seg_global, partition


def compute_network_specific_metrics(adj_df, partition):
    """Compute within-network metrics given atlas labels."""
    labels = adj_df.columns.tolist()

    # identify network names (e.g., '7Networks_LH_Vis')
    networks = [l.split('_')[2] if len(l.split('_')) > 2 else l for l in labels]
    unique_networks = sorted(set(networks))

    corr = np.tanh(adj_df.values)
    np.fill_diagonal(corr, 0)
    corr[corr < 0] = 0

    net_results = []

    for net in unique_networks:
        idx = [i for i, n in enumerate(networks) if n == net]
        if len(idx) < 2:
            continue
        submatrix = corr[np.ix_(idx, idx)]

        within_eff = bct.efficiency_wei(submatrix)
        within_clust = np.mean(bct.clustering_coef_wu(submatrix))
        net_results.append({
            "network": net,
            "integration": within_eff,
            "segregation": within_clust
        })
    return net_results


# ==== MAIN LOOP ====

metrics_list = []

for file in sorted(os.listdir(base_dir)):
    if not file.endswith("_adjacency_FisherZ.csv"):
        continue
    sub = file.split("_")[0]  # e.g., sub-001

    path = os.path.join(base_dir, file)
    df = pd.read_csv(path, index_col=0)

    modularity, eff, seg, partition = compute_graph_metrics(df)
    net_stats = compute_network_specific_metrics(df, partition)

    metrics_list.append({
        "subject": sub,
        "modularity": modularity,
        "integration_global": eff,
        "segregation_global": seg
    })

    for net in net_stats:
        metrics_list.append({
            "subject": sub,
            "network": net["network"],
            "integration_network": net["integration"],
            "segregation_network": net["segregation"]
        })

    print(f"✅ {sub} done")

# ==== SAVE ====
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(output_csv, index=False)
print(f"\n🎯 All metrics saved to {output_csv}")
