import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import graph_engine as ge


def compute_all_metrics(csv_path):
    """Load curriculum and compute all 6 structural risk metrics"""
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    G = nx.DiGraph()

    for _, row in df.iterrows():
        # "1,2" -> use the first quarter
        quarter_str = str(row.get("quarter", "1"))
        quarter = int(quarter_str.split(",")[0]) if "," in quarter_str else int(quarter_str)

        G.add_node(
            row["course_code"],
            credits=float(row.get("credits", 5)),
            year=int(row.get("year", 1)),
            quarter=quarter
        )

    for _, row in df.iterrows():
        target = row["course_code"]
        prereqs = str(row.get("prerequisites_formal", ""))

        if prereqs:
            for p in prereqs.split(","):
                p_clean = p.strip()
                if p_clean in G.nodes:
                    G.add_edge(p_clean, target)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Curriculum contains cycles.")

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # compute all 7 metrics, including logical and temporal depth
    print("Computing all 7 metrics...")
    metrics_raw = {
        "blocking_factor": ge.compute_blocking_factor(G),
        "betweenness": nx.betweenness_centrality(G, normalized=True),
        "pagerank": nx.pagerank(G),
        "logical_depth": ge.compute_longest_path_depth(G),
        "temporal_criticality": ge.compute_temporal_criticality(G),
        "articulation_impact": ge.compute_articulation_reach(G)
    }

    metrics = {
        name: ge.normalize(values)
        for name, values in metrics_raw.items()
    }

    results = []
    for node in G.nodes():
        results.append({
            "course_code": node,
            "blocking_factor": metrics["blocking_factor"][node],
            "betweenness": metrics["betweenness"][node],
            "pagerank": metrics["pagerank"][node],
            "logical_depth": metrics["logical_depth"][node],
            "temporal_criticality": metrics["temporal_criticality"][node],
            "articulation_impact": metrics["articulation_impact"][node],
            "year": G.nodes[node]["year"],
            "quarter": G.nodes[node]["quarter"]
        })

    return pd.DataFrame(results)


def analyze_correlations(df):
    """Analyze correlation matrix to identify redundant metrics"""

    metrics = ["blocking_factor", "betweenness", "pagerank",
               "logical_depth", "temporal_criticality", "articulation_impact"]

    corr_matrix = df[metrics].corr()

    print("\nCorrelation matrix (all 6 metrics):")
    print(corr_matrix.round(4))

    print("\nHigh correlations (|r| > 0.5):")

    redundant_pairs = []
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i < j:  # Only upper triangle
                corr_val = corr_matrix.loc[metric1, metric2]
                if abs(corr_val) > 0.5:
                    redundant_pairs.append((metric1, metric2, corr_val))
                    print(f"{metric1:25s} ↔ {metric2:25s}: {corr_val:7.4f}")

    if not redundant_pairs:
        print("No high correlations found (all |r| < 0.5)")

    return corr_matrix


def plot_correlation_heatmap(corr_matrix, title="All 6 Metrics"):
    """Create detailed correlation heatmap"""

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=2, cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1, ax=ax)

    plt.title(f'Metric Correlation Matrix: {title}',
              fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


def compare_3_vs_6_metrics(df):
    """Compare correlation structure of 3-metric composite vs all 6 metrics"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    all_metrics = ["blocking_factor", "betweenness", "pagerank",
                   "logical_depth", "temporal_criticality", "articulation_impact"]
    corr_6 = df[all_metrics].corr()

    sns.heatmap(corr_6, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax1, vmin=-1, vmax=1, annot_kws={'fontsize': 9})
    ax1.set_title('All 6 Metrics', fontsize=12, pad=10)

    selected_metrics = ["blocking_factor", "betweenness", "pagerank"]
    corr_3 = df[selected_metrics].corr()

    sns.heatmap(corr_3, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax2, vmin=-1, vmax=1)
    ax2.set_title('3 Selected Metrics\n(Composite Risk Score)', fontsize=12, pad=10)

    plt.suptitle('Metric Selection: 6 Candidates vs 3 Composite', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    print("\nCorrelation comparison:")
    print(f"\n6-metric approach (all candidates):")
    print(f"  Average |correlation|: {corr_6.abs().mean().mean():.4f}")
    print(f"  Max off-diagonal |correlation|: {corr_6.abs().where(~np.eye(6, dtype=bool)).max().max():.4f}")

    print(f"\n3-metric composite (blocking, betweenness, pagerank):")
    print(f"  Average |correlation|: {corr_3.abs().mean().mean():.4f}")
    print(f"  Max off-diagonal |correlation|: {corr_3.abs().where(~np.eye(3, dtype=bool)).max().max():.4f}")
    print("\nLower correlations mean less redundancy, so a better metric selection.")


def run_full_analysis():
    """Run complete redundancy analysis"""

    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "all_metrics_analysis.csv"

    print("Structural risk metric redundancy analysis")
    print("Computes all 6 metrics to show the multidimensional risk approach:")
    print("combining logical complexity and temporal fragility.\n")

    df = compute_all_metrics(DATA_PATH)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Full results saved to: {OUTPUT_PATH}\n")

    corr_matrix = analyze_correlations(df)

    print("Generating visualizations...\n")

    print("1. Correlation Heatmap (All 6 Metrics)")
    plot_correlation_heatmap(corr_matrix, "All 6 Metrics")

    print("2. 3-Metric vs 6-Metric Comparison")
    compare_3_vs_6_metrics(df)

    print("\nAnalysis complete. Composite structural risk uses 3 metrics:")
    print("  - Blocking factor: downstream credit impact")
    print("  - Betweenness: logical bottleneck position")
    print("  - PageRank: accumulated upstream dependency prestige")


if __name__ == "__main__":
    run_full_analysis()
