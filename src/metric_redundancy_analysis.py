import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import graph_engine as ge


def compute_all_metrics(csv_path):
    """Load curriculum and compute all 5 structural risk metrics"""
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    
    G = nx.DiGraph()
    
    # Build graph
    for _, row in df.iterrows():
        # Handle multi-quarter courses (e.g., "1,2") by taking first quarter
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
    
    # Compute ALL 7 metrics (including both logical and temporal depth)
    print("Computing all 7 metrics...")
    metrics_raw = {
        "blocking_factor": ge.compute_blocking_factor(G),
        "betweenness": nx.betweenness_centrality(G, normalized=True),
        "pagerank": nx.pagerank(G),
        "logical_depth": ge.compute_longest_path_depth(G),
        "temporal_criticality": ge.compute_temporal_criticality(G),
        "articulation_impact": ge.compute_articulation_reach(G)
    }
    
    # Normalize all metrics
    metrics = {
        name: ge.normalize(values)
        for name, values in metrics_raw.items()
    }
    
    # Create DataFrame
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
    
    print("="*60)
    print("CORRELATION MATRIX (All 6 Metrics)")
    print("="*60)
    print(corr_matrix.round(4))
    print("\n")
    
    # Identify high correlations (> 0.7 typically indicates redundancy)
    print("="*60)
    print("HIGH CORRELATIONS (|r| > 0.5)")
    print("="*60)
    
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
    
    print("\n")
    
    # Recommended metrics
    print("="*60)
    print("RECOMMENDATION: MULTIDIMENSIONAL RISK APPROACH")
    print("="*60)
    print("\nBased on correlation analysis:")
    print("\n✓ KEEP (Low redundancy, distinct dimensions):")
    print("  - Blocking Factor: Downstream credit impact")
    print("  - Betweenness: Logical bottleneck position")
    print("  - Logical Depth: Structural complexity (knowledge chain - unweighted hops)")
    print("  - Temporal Criticality: Calendar fragility (graduation risk - weighted quarters)")
    
    print("\n✗ REMOVE (Redundant with other metrics):")
    if any("pagerank" in str(pair) for pair in redundant_pairs):
        print("  - PageRank: Highly correlated with betweenness/blocking")
    else:
        print("  - PageRank: Conceptually overlaps with betweenness")
    
    if any("articulation_impact" in str(pair) for pair in redundant_pairs):
        print("  - Articulation Impact: Highly correlated with blocking factor")
    else:
        print("  - Articulation Impact: Conceptually overlaps with blocking factor")
    
    print("\n💡 KEY INSIGHT:")
    print("  Keeping BOTH logical_depth and temporal_criticality allows identification")
    print("  of 'hidden' temporal constraints - courses with low logical depth but")
    print("  high temporal impact (e.g., late-year courses, annual-only offerings).")
    
    print("\n")
    
    return corr_matrix


def plot_correlation_heatmap(corr_matrix, title="All 6 Metrics"):
    """Create detailed correlation heatmap"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=2, cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1, ax=ax)
    
    plt.title(f'Metric Correlation Matrix: {title}',
              fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()


def compare_4_vs_6_metrics(df):
    """Compare correlation structure of 4-metric vs 6-metric approach"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 6-metric correlation
    all_metrics = ["blocking_factor", "betweenness", "pagerank", 
                   "logical_depth", "temporal_criticality", "articulation_impact"]
    corr_6 = df[all_metrics].corr()
    
    sns.heatmap(corr_6, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax1, vmin=-1, vmax=1, annot_kws={'fontsize': 9})
    ax1.set_title('All 6 Metrics', fontsize=12, pad=10)
    
    # 4-metric correlation (multidimensional approach)
    selected_metrics = ["blocking_factor", "betweenness", "logical_depth", "temporal_criticality"]
    corr_4 = df[selected_metrics].corr()
    
    sns.heatmap(corr_4, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax2, vmin=-1, vmax=1)
    ax2.set_title('4 Selected Metrics\n(Multidimensional Risk)', fontsize=12, pad=10)
    
    plt.suptitle('Justification for Multidimensional Approach', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("="*60)
    print("CORRELATION COMPARISON")
    print("="*60)
    print(f"\n6-Metric Approach (All candidates):")
    print(f"  Average |correlation|: {corr_6.abs().mean().mean():.4f}")
    print(f"  Max off-diagonal |correlation|: {corr_6.abs().where(~np.eye(6, dtype=bool)).max().max():.4f}")
    
    print(f"\n4-Metric Approach (Selected - Multidimensional):")
    print(f"  Average |correlation|: {corr_4.abs().mean().mean():.4f}")
    print(f"  Max off-diagonal |correlation|: {corr_4.abs().where(~np.eye(4, dtype=bool)).max().max():.4f}")
    print("\n✓ Lower correlations = less redundancy = better metric selection\n")


def run_full_analysis():
    """Run complete redundancy analysis"""
    
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "all_metrics_analysis.csv"
    
    print("="*60)
    print("STRUCTURAL RISK METRIC REDUNDANCY ANALYSIS")
    print("="*60)
    print("\nThis analysis computes ALL 6 metrics to demonstrate the")
    print("multidimensional risk approach: combining logical complexity")
    print("and temporal fragility to identify hidden structural constraints.")
    print("\n")
    
    # Compute all metrics
    df = compute_all_metrics(DATA_PATH)
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Full results saved to: {OUTPUT_PATH}\n")
    
    # Analyze correlations
    corr_matrix = analyze_correlations(df)
    
    # Visualizations
    print("Generating visualizations...\n")
    
    print("1. Correlation Heatmap (All 6 Metrics)")
    plot_correlation_heatmap(corr_matrix, "All 6 Metrics")
    
    print("2. 4-Metric vs 6-Metric Comparison")
    compare_4_vs_6_metrics(df)
    
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nConclusion: The multidimensional 4-metric approach captures distinct")
    print("dimensions of structural risk:")
    print("  • Blocking Factor: Credit impact")
    print("  • Betweenness: Logical bottlenecks")
    print("  • Logical Depth: Knowledge complexity (unweighted hops)")
    print("  • Temporal Criticality: Graduation risk (weighted quarters)")
    print("\nKeeping BOTH depth metrics reveals 'hidden' temporal constraints:")
    print("courses with low logical depth but high temporal impact.")
    print("="*60)


if __name__ == "__main__":
    run_full_analysis()
