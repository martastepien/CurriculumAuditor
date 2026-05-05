import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import numpy as np
import pathlib
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graph_engine as ge
from comparative_study import load_and_build_dag


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RANDOM_SEED = 42

ALL_METRICS = ['blocking_factor', 'betweenness', 'pagerank', 'logical_depth',
               'temporal_criticality', 'articulation_impact']

METRIC_LABELS = {
    'blocking_factor':      'Blocking\nFactor',
    'betweenness':          'Betweenness',
    'pagerank':             'PageRank',
    'logical_depth':        'Logical\nDepth',
    'temporal_criticality': 'Temporal\nCriticality',
    'articulation_impact':  'Articulation\nImpact',
}


# --- loaders ---

def load_structural_results():
    path = BASE_DIR / "data" / "processed" / "structural_risk_baseline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run comparative_study.py first. Expected: {path}")
    return pd.read_csv(path)


def load_personal_curriculum():
    personal_path = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"
    main_path     = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"

    if not personal_path.exists():
        return None

    personal_df = pd.read_csv(personal_path)
    main_df     = pd.read_csv(main_path)

    # Map elective placeholder slots (keyed by year/quarter) so we can display real course codes
    slots = defaultdict(list)
    for _, row in main_df.iterrows():
        code = str(row['course_code'])
        if 'Elective' in code:
            q = int(str(row['quarter']).split(',')[0])
            slots[(int(row['year']), q)].append(code)

    label_map  = {}
    slot_usage = defaultdict(int)

    for _, row in personal_df.iterrows():
        code = row['course_code']
        if pd.isna(code):
            continue
        code = str(code)
        is_elective = bool(row.get('is_elective', False))
        if is_elective and not pd.isna(row.get('year')) and not pd.isna(row.get('quarter')):
            q    = int(str(row['quarter']).split(',')[0])
            year = int(row['year'])
            key  = (year, q)
            available = slots.get(key, [])
            idx = slot_usage[key]
            if idx < len(available):
                label_map[available[idx]] = code
                slot_usage[key] += 1
        else:
            label_map[code] = code

    return label_map


def _best_metric_subset(corr_matrix, metrics, n):
    best_subset, best_score = None, float('inf')
    for subset in combinations(metrics, n):
        sub   = corr_matrix.loc[list(subset), list(subset)]
        k     = len(subset)
        score = (sub.abs().sum().sum() - k) / (k * (k - 1))
        if score < best_score:
            best_score, best_subset = score, subset
    return list(best_subset), best_score


# --- structural plots ---

def plot_top_structural_risk(top_n=10):
    df  = load_structural_results()
    top = df.sort_values("structural_risk", ascending=False).head(top_n)

    plt.figure()
    plt.bar(top["course_code"], top["structural_risk"])
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} structural risk courses")
    plt.tight_layout()
    plt.show()


def plot_metric_correlations(n_select=3):
    df = load_structural_results()
    personal_courses = load_personal_curriculum()

    if personal_courses is not None:
        df = df[df['course_code'].isin(personal_courses.keys())]

    available   = [m for m in ALL_METRICS if m in df.columns and df[m].std() > 0]
    corr_matrix = df[available].corr()

    selected, avg_corr = _best_metric_subset(corr_matrix, available, n_select)
    sub_corr = corr_matrix.loc[selected, selected]

    x_labels_all = [METRIC_LABELS[m] for m in available]
    x_labels_sel = [METRIC_LABELS[m] for m in selected]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    title_suffix = " (My Curriculum)" if personal_courses is not None else ""
    sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax1,
                xticklabels=x_labels_all, yticklabels=x_labels_all)
    ax1.set_title(f'All {len(available)} metrics: correlation matrix{title_suffix}')

    sns.heatmap(sub_corr, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax2,
                xticklabels=x_labels_sel, yticklabels=x_labels_sel)
    ax2.set_title(f'Best {n_select} (least redundant)\nAvg |corr| = {avg_corr:.4f}')

    plt.suptitle('Metric redundancy analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\nAuto-selected {n_select} least-correlated metrics: {selected}")
    print(f"Average absolute pairwise correlation: {avg_corr:.3f}")
    print("\nFull correlation matrix:")
    print(corr_matrix.round(4))


def plot_personal_curriculum_risk():
    df = load_structural_results()
    personal_courses = load_personal_curriculum()

    if personal_courses is None:
        print("Personal curriculum file not found. Skipping...")
        return

    personal_df = df[df['course_code'].isin(personal_courses.keys())].copy()
    personal_df['display_code'] = personal_df['course_code'].map(personal_courses)
    personal_df = personal_df.sort_values('structural_risk', ascending=False)

    if len(personal_df) == 0:
        print("No matching courses found.")
        return

    top_n = 15
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Panel 1: overall top 15, highlight my courses
    top_all = df.nlargest(top_n, 'structural_risk').copy()
    top_all['display_code'] = top_all['course_code'].map(lambda c: personal_courses.get(c, c))
    colors = ['red' if c in personal_courses else 'steelblue' for c in top_all['course_code']]
    ax1.barh(range(len(top_all)), top_all['structural_risk'], color=colors)
    ax1.set_yticks(range(len(top_all)))
    ax1.set_yticklabels(top_all['display_code'])
    ax1.set_xlabel('Structural risk')
    ax1.set_title(f'Top {top_n} highest risk courses, overall (red = my courses)')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend(handles=[
        mpatches.Patch(color='red',       label='In my curriculum'),
        mpatches.Patch(color='steelblue', label='Not in my curriculum'),
    ], fontsize=9)

    # Panel 2: ALL personal courses ranked — includes electives at structural_risk=0
    ax2.barh(range(len(personal_df)), personal_df['structural_risk'], color='coral')
    ax2.set_yticks(range(len(personal_df)))
    ax2.set_yticklabels(personal_df['display_code'])
    ax2.set_xlabel('Structural risk')
    ax2.set_title('All my courses ranked by structural risk (electives included)')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    top_course = personal_df.iloc[0]
    print(f"\nMy curriculum statistics:")
    print(f"Total matched courses: {len(personal_df)}")
    print(f"Average risk: {personal_df['structural_risk'].mean():.3f}")
    print(f"Highest risk: {top_course['display_code']} ({top_course['structural_risk']:.3f})")


# --- semantic plots (called from semantic_analysis.py after the pipeline runs) ---

def plot_similarity_heatmap(sim_matrix, course_codes, df):
    # Sort courses chronologically so you can see how similarity clusters by year
    sort_key = {
        row['course_code']: int(row['year']) * 10 + int(row['quarter_parsed'])
        for _, row in df.iterrows()
    }
    sorted_codes = sorted(course_codes, key=lambda c: sort_key[c])
    idx = [course_codes.index(c) for c in sorted_codes]
    matrix = sim_matrix[np.ix_(idx, idx)]

    df_heat = pd.DataFrame(matrix, index=sorted_codes, columns=sorted_codes)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df_heat, cmap='YlOrRd', ax=ax, square=True,
                linewidths=0.2, cbar_kws={'label': 'Cosine Similarity'},
                xticklabels=True, yticklabels=True)
    ax.set_title("Course content similarity matrix (sorted by year/quarter)", fontsize=13)
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_divergence_scatter(divergence_df):
    # Four quadrants split at medians — data-driven boundaries rather than arbitrary cutoffs
    df    = divergence_df.copy()
    x     = df['structural_risk']
    y     = df['semantic_centrality']
    x_mid = x.median()
    y_mid = y.median()

    colors = []
    for xi, yi in zip(x, y):
        if   xi < x_mid and yi >= y_mid:  colors.append('crimson')      # hidden constraint
        elif xi >= x_mid and yi >= y_mid:  colors.append('darkorange')   # doubly visible
        elif xi < x_mid and yi < y_mid:    colors.append('steelblue')    # background
        else:                              colors.append('forestgreen')  # structural-only

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(x, y, c=colors, s=90, alpha=0.85, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(row['course_code'],
                    (row['structural_risk'], row['semantic_centrality']),
                    textcoords='offset points', xytext=(5, 3), fontsize=7, alpha=0.85)

    ax.axvline(x_mid, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y_mid, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.text(x_mid * 0.05, y_mid * 1.6, "Hidden constraints\n(semantic-only)",
            fontsize=8, color='crimson', alpha=0.7)
    ax.text(x_mid * 1.3,  y_mid * 1.6, "Doubly visible\n(both high)",
            fontsize=8, color='darkorange', alpha=0.7)
    ax.text(x_mid * 0.05, y_mid * 0.2, "Background\n(both low)",
            fontsize=8, color='steelblue', alpha=0.7)
    ax.text(x_mid * 1.3,  y_mid * 0.2, "Structural-only\nrisk",
            fontsize=8, color='forestgreen', alpha=0.7)

    ax.legend(handles=[
        mpatches.Patch(color='crimson',     label='Hidden constraint (low struct, high sem)'),
        mpatches.Patch(color='darkorange',  label='Doubly visible (both high)'),
        mpatches.Patch(color='steelblue',   label='Background (both low)'),
        mpatches.Patch(color='forestgreen', label='Structural-only risk'),
    ], fontsize=8, loc='upper right')

    ax.set_xlabel("Structural risk", fontsize=11)
    ax.set_ylabel("Semantic out-centrality", fontsize=11)
    ax.set_title("Structural-semantic divergence analysis", fontsize=13)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_semantic_graph(hidden_deps, divergence_df, threshold):
    if hidden_deps is None or len(hidden_deps) == 0:
        print("No hidden dependencies to plot.")
        return

    G = nx.DiGraph()
    for _, row in hidden_deps.iterrows():
        G.add_edge(row['source_course'], row['target_course'], weight=row['normalized_weight'])

    div_lookup  = dict(zip(divergence_df['course_code'], divergence_df['divergence_score']))
    node_sizes  = [200 + div_lookup.get(n, 0) * 800 for n in G.nodes()]
    node_colors = ['crimson' if div_lookup.get(n, 0) > 0 else 'steelblue' for n in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(13, 10))
    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=2.0)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.55, edge_color='gray',
                           arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)

    ax.legend(handles=[
        mpatches.Patch(color='crimson',   label='Positive divergence (hidden constraint)'),
        mpatches.Patch(color='steelblue', label='Negative divergence'),
    ], fontsize=9)
    ax.set_title(
        f"Hidden semantic dependency graph\n(threshold = {threshold:.3f}, {len(hidden_deps)} edges)",
        fontsize=12
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_concept_clusters(embeddings, cluster_labels, course_codes, divergence_df, n_clusters=5):
    # PCA to 2D for visual inspection of thematic groupings.
    # Node size scales with structural risk so high-risk courses stand out.
    pca   = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(embeddings)
    var_explained = sum(pca.explained_variance_ratio_) * 100

    risk_lookup  = dict(zip(divergence_df['course_code'], divergence_df['structural_risk']))
    cluster_arr  = np.array([cluster_labels[c] for c in course_codes])
    sizes        = np.array([50 + risk_lookup.get(c, 0) * 600 for c in course_codes])

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=cluster_arr, cmap='tab10', s=sizes, alpha=0.8, zorder=3)

    for i, code in enumerate(course_codes):
        ax.annotate(code, (coords[i, 0], coords[i, 1]),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)

    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_title(
        f"Concept clusters: SBERT embeddings (PCA 2D)\n"
        f"node size = structural risk, variance explained: {var_explained:.1f}%",
        fontsize=12
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# --- combined structural + semantic comparison plots (CSV-based) ---

def plot_full_risk_comparison(top_n=15):
    sem_path = BASE_DIR / "data" / "processed" / "semantic_analysis.csv"
    if not sem_path.exists():
        print("Semantic analysis not found. Run semantic_analysis.py first.")
        return

    sem_df    = pd.read_csv(sem_path).drop(columns=['structural_risk'], errors='ignore')
    struct_df = load_structural_results()
    merged    = sem_df.merge(struct_df[['course_code', 'structural_risk']], on='course_code', how='left')

    top_struct_codes = set(merged.nlargest(top_n, 'structural_risk')['course_code'])
    top_sem_codes    = set(merged.nlargest(top_n, 'semantic_centrality')['course_code'])
    both             = top_struct_codes & top_sem_codes

    top_struct = merged.nlargest(top_n, 'structural_risk')
    top_sem    = merged.nlargest(top_n, 'semantic_centrality')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    colors1 = ['crimson' if c in both else 'steelblue' for c in top_struct['course_code']]
    ax1.barh(range(len(top_struct)), top_struct['structural_risk'], color=colors1)
    ax1.set_yticks(range(len(top_struct)))
    ax1.set_yticklabels(top_struct['course_code'])
    ax1.set_xlabel('Structural Risk')
    ax1.set_title(f'Top {top_n} by structural risk')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    colors2 = ['crimson' if c in both else 'steelblue' for c in top_sem['course_code']]
    ax2.barh(range(len(top_sem)), top_sem['semantic_centrality'], color=colors2)
    ax2.set_yticks(range(len(top_sem)))
    ax2.set_yticklabels(top_sem['course_code'])
    ax2.set_xlabel('Semantic out-centrality')
    ax2.set_title(f'Top {top_n} by semantic centrality')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    fig.legend(handles=[
        mpatches.Patch(color='crimson',   label='Appears in both rankings'),
        mpatches.Patch(color='steelblue', label='Single-dimension only'),
    ], loc='lower center', ncol=2, fontsize=10)
    plt.suptitle("Structural vs semantic risk: dual ranking comparison", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_augmented_vs_original(top_n=15):
    aug_path = BASE_DIR / "data" / "processed" / "augmented_graph_comparison.csv"
    if not aug_path.exists():
        print("Augmented graph comparison not found. Run semantic_analysis.py first.")
        return

    aug_df = pd.read_csv(aug_path).sort_values('rank_shift', ascending=False)
    top    = aug_df[aug_df['rank_shift'] != 0].head(top_n)
    colors = ['crimson' if s > 0 else 'steelblue' for s in top['rank_shift']]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), top['rank_shift'], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['course_code'])
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Rank shift (positive = moved up after adding semantic edges)')
    ax.set_title(
        f'Hidden bottlenecks: rank change after augmenting formal graph\nwith semantic edges (top {top_n} by shift)',
        fontsize=12
    )
    ax.grid(axis='x', alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(color='crimson',   label='Hidden bottleneck (gained importance)'),
        mpatches.Patch(color='steelblue', label='Reduced importance'),
    ], fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_personal_empirical_risk():
    personal_path = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"
    if not personal_path.exists():
        print("Personal curriculum file not found. Skipping...")
        return

    personal_df = pd.read_csv(personal_path)
    personal_df = personal_df[personal_df['course_code'].notna()].copy()

    # Build personal DAG — picks up 2IC80's formal prereq (2IRR40) automatically
    G_personal = load_and_build_dag(personal_path)
    risk_scores, _ = ge.compute_structural_risk_score(G_personal)
    risk_df = pd.DataFrame([
        {'course_code': c, 'personal_structural_risk': round(v, 4)}
        for c, v in risk_scores.items()
    ])

    merged = personal_df[['course_code', 'pass_rate_2024', 'is_elective']].merge(
        risk_df, on='course_code', how='left'
    )
    merged['personal_structural_risk'] = merged['personal_structural_risk'].fillna(0.0)
    merged = merged.dropna(subset=['pass_rate_2024'])
    merged['empirical_difficulty'] = 1 - merged['pass_rate_2024']

    mandatory = merged[merged['is_elective'] == False]
    electives  = merged[merged['is_elective'] == True]

    x_mid = mandatory['personal_structural_risk'].median()
    y_mid = mandatory['empirical_difficulty'].median()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: scatter
    ax1.scatter(mandatory['personal_structural_risk'], mandatory['empirical_difficulty'],
                color='steelblue', s=80, alpha=0.85, zorder=3, label='Mandatory')
    ax1.scatter(electives['personal_structural_risk'], electives['empirical_difficulty'],
                color='coral', s=80, alpha=0.85, zorder=3, label='Elective')

    for _, row in merged.iterrows():
        ax1.annotate(row['course_code'],
                     (row['personal_structural_risk'], row['empirical_difficulty']),
                     textcoords='offset points', xytext=(5, 3), fontsize=7, alpha=0.85)

    ax1.axvline(x_mid, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y_mid, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    pad_x, pad_y = x_mid * 0.05, 0.01
    ax1.text(x_mid + pad_x, y_mid + pad_y, "Doubly risky",        fontsize=8, color='crimson',     alpha=0.7)
    ax1.text(pad_x,          y_mid + pad_y, "Hard but isolated",   fontsize=8, color='darkorange',  alpha=0.7)
    ax1.text(x_mid + pad_x, pad_y,          "Structurally risky\nbut passable", fontsize=8, color='forestgreen', alpha=0.7)
    ax1.text(pad_x,          pad_y,          "Low risk",            fontsize=8, color='steelblue',  alpha=0.7)

    ax1.legend(fontsize=9)
    ax1.set_xlabel("Personal structural risk", fontsize=11)
    ax1.set_ylabel("Empirical difficulty (1 - pass rate)", fontsize=11)
    ax1.set_title("Structural vs empirical risk: personal curriculum", fontsize=12)
    ax1.grid(alpha=0.2)

    # Panel 2: elective pass rates vs mandatory mean
    mandatory_mean = mandatory['pass_rate_2024'].mean()
    electives_sorted = electives.sort_values('pass_rate_2024')
    colors = ['crimson' if v < mandatory_mean else 'steelblue' for v in electives_sorted['pass_rate_2024']]

    ax2.barh(range(len(electives_sorted)), electives_sorted['pass_rate_2024'], color=colors)
    ax2.set_yticks(range(len(electives_sorted)))
    ax2.set_yticklabels(electives_sorted['course_code'])
    ax2.axvline(mandatory_mean, color='black', linestyle='--', linewidth=1.2,
                label=f'Mandatory mean ({mandatory_mean:.2f})')
    ax2.set_xlabel("Pass rate 2024", fontsize=11)
    ax2.set_title("Elective pass rates vs mandatory average", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle("Personal curriculum risk profile", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("CurriculumAuditor visualizations\n")

    print("1. Top structural risk courses")
    plot_top_structural_risk(top_n=10)

    print("\n2. Metric correlation analysis")
    plot_metric_correlations(n_select=3)

    print("\n3. My curriculum risk analysis")
    plot_personal_curriculum_risk()

    sem_path = BASE_DIR / "data" / "processed" / "semantic_analysis.csv"
    if sem_path.exists():
        print("\n4. Structural vs semantic dual ranking")
        plot_full_risk_comparison()

        print("\n5. Augmented graph hidden bottlenecks")
        plot_augmented_vs_original()

    print("\n6. Personal curriculum risk profile")
    plot_personal_empirical_risk()

    print("\nAll visualizations complete!")
