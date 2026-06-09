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

    # Map elective placeholder slots by year/quarter so we can display real course codes
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


# --- DAG plot ---

COL_GAP     = 6.0
ROW_GAP     = 14.0
NODE_SPREAD = 3.0


def _dag_layout(G):
    slot_counts = defaultdict(int)
    slot_index  = defaultdict(int)
    for n in G.nodes():
        slot_counts[(G.nodes[n]['year'], G.nodes[n]['quarter'])] += 1

    pos = {}
    for n in G.nodes():
        year    = G.nodes[n]['year']
        quarter = G.nodes[n]['quarter']
        key     = (year, quarter)
        count   = slot_counts[key]
        idx     = slot_index[key]
        slot_index[key] += 1
        x = float(quarter) * COL_GAP
        y = -year * ROW_GAP + (idx - (count - 1) / 2) * NODE_SPREAD
        pos[n] = (x, y)
    return pos


def _draw_dag_on_ax(G, ax, title):
    pos = _dag_layout(G)

    # Draw edges first so ellipses render on top and hide the endpoints cleanly
    edge_styles = []
    for u, v in G.edges():
        if abs(pos[u][0] - pos[v][0]) < 1.0:
            edge_styles.append('arc3,rad=0.25')
        else:
            edge_styles.append('arc3,rad=0.08')

    for (u, v), cstyle in zip(G.edges(), edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrows=True,
                               arrowstyle='-|>', arrowsize=18,
                               edge_color='#555555', alpha=0.6, width=1.4,
                               connectionstyle=cstyle,
                               min_source_margin=0, min_target_margin=4,
                               ax=ax)

    display_labels = {n: 'Elective' if 'Elective' in n else n for n in G.nodes()}
    for node, label in display_labels.items():
        x, y = pos[node]
        ax.text(x, y, label, fontsize=13, fontweight='bold',
                color='#1a2e45', ha='center', va='center', zorder=3,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#d6e8f7',
                          edgecolor='#2c5f8a', linewidth=1.5, alpha=0.95))

    years = sorted({G.nodes[n]['year'] for n in G.nodes()})
    for i in range(len(years) - 1):
        sep_y = -(years[i] + years[i + 1]) / 2 * ROW_GAP
        ax.axhline(sep_y, xmin=0.02, xmax=0.98, color='#cccccc',
                   linestyle='--', linewidth=1.0, zorder=0)
    for year in years:
        ax.text(COL_GAP * 0.35, -year * ROW_GAP, f'Year {year}', fontsize=12,
                fontweight='bold', va='center', color='#666666')
    bottom_y = -max(years) * ROW_GAP - ROW_GAP * 0.7
    for q in range(1, 5):
        ax.text(q * COL_GAP, bottom_y, f'Q{q}', fontsize=11,
                fontweight='bold', ha='center', color='#777777')

    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    ax.set_xlim(min(x_vals) - 2.5, max(x_vals) + 2.5)
    ax.set_ylim(min(y_vals) - 2.5, max(y_vals) + 2.5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')


def plot_dag():
    DATA_PATH     = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"
    PERSONAL_PATH = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"

    G_main = load_and_build_dag(DATA_PATH)

    fig1, ax1 = plt.subplots(figsize=(20, 22))
    _draw_dag_on_ax(G_main, ax1, "Official Curriculum DAG - prerequisite relationships")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
    plt.show()

    if PERSONAL_PATH.exists():
        G_personal = load_and_build_dag(PERSONAL_PATH)
        fig2, ax2 = plt.subplots(figsize=(20, 22))
        _draw_dag_on_ax(G_personal, ax2, "Personal Curriculum DAG - prerequisite relationships")
        plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
        plt.show()


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

    # Panel 2: all personal courses ranked, including electives which score 0
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
    # Four quadrants split at medians, data-driven boundaries rather than fixed cutoffs
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


NODE_COLOR_DEFAULT = '#4C72B0'
NODE_COLOR_ELECTIVE = '#2ca02c'


def _draw_hidden_dep_graph(hidden_deps, ax, title, threshold, elective_codes=None):
    G = nx.DiGraph()
    for _, row in hidden_deps.iterrows():
        G.add_edge(row['source_course'], row['target_course'], weight=row['normalized_weight'])

    node_colors = [
        NODE_COLOR_ELECTIVE if (elective_codes and n in elective_codes) else NODE_COLOR_DEFAULT
        for n in G.nodes()
    ]
    edge_widths = [G[u][v]['weight'] * 8 for u, v in G.edges()]

    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=3.5)
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=22, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.55, edge_color='gray',
                           arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

    ax.set_title(f"{title}\n(threshold = {threshold:.3f}, {len(hidden_deps)} edges)", fontsize=20)
    ax.axis('off')


def plot_semantic_graph(hidden_deps, threshold):
    if hidden_deps is None or len(hidden_deps) == 0:
        print("No hidden dependencies to plot.")
        return

    fig, ax = plt.subplots(figsize=(18, 14))
    _draw_hidden_dep_graph(hidden_deps, ax,
                           "Latent semantic links, official TU/e CSE curriculum", threshold)
    plt.tight_layout()
    plt.show()


def plot_personal_semantic_graph(
    official_hidden_deps, official_threshold,
    personal_hidden_deps, personal_threshold,
    elective_codes=None,
):
    if official_hidden_deps is None or personal_hidden_deps is None:
        print("Missing hidden dependency data. Skipping personal semantic graph.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))

    _draw_hidden_dep_graph(official_hidden_deps, ax1,
                           "Latent semantic links, official TU/e CSE curriculum",
                           official_threshold)
    _draw_hidden_dep_graph(personal_hidden_deps, ax2,
                           "Latent semantic links, student-augmented curriculum",
                           personal_threshold, elective_codes=elective_codes)

    ax2.legend(handles=[
        mpatches.Patch(color=NODE_COLOR_DEFAULT,  label='Required course'),
        mpatches.Patch(color=NODE_COLOR_ELECTIVE, label='Elective'),
    ], loc='lower right', fontsize=18)

    plt.tight_layout()
    plt.show()


def plot_concept_clusters(embeddings, cluster_labels, course_codes, divergence_df, n_clusters=5):
    # PCA to 2D for visual inspection of thematic groupings.
    # Node size scales with structural risk so high-risk courses are easy to spot.
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


# --- semantic analysis plots (CSV-based) ---

def plot_top_semantic_similarity(top_n=20):
    dep_path = BASE_DIR / "data" / "processed" / "hidden_dependencies.csv"
    if not dep_path.exists():
        print("Hidden dependencies not found. Run semantic_analysis.py first.")
        return

    df = pd.read_csv(dep_path).sort_values('similarity_score', ascending=False).head(top_n)
    n_shown = len(df)

    labels = [
        f"{row['source_course']} -> {row['target_course']}\n"
        f"({row['source_year_quarter']} to {row['target_year_quarter']})"
        for _, row in df.iterrows()
    ]

    _, ax = plt.subplots(figsize=(10, n_shown * 0.45 + 1))
    ax.barh(range(n_shown), df['similarity_score'], color='steelblue', alpha=0.85)
    ax.set_yticks(range(n_shown))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Cosine similarity score')
    ax.set_title(f'All detected hidden semantic dependencies, ranked by similarity\n({n_shown} course pairs with no formal prerequisite edge)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_personal_top_hidden_deps(top_n=15):
    official_path = BASE_DIR / "data" / "processed" / "hidden_dependencies.csv"
    personal_path = BASE_DIR / "data" / "processed" / "hidden_dependencies_personal.csv"
    personal_curriculum_path = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"

    if not official_path.exists():
        print("Official hidden dependencies not found. Run semantic_analysis.py first.")
        return
    if not personal_path.exists():
        print("Personal hidden dependencies not found. Run semantic_analysis.py first.")
        return
    if not personal_curriculum_path.exists():
        print("Personal curriculum file not found. Skipping...")
        return

    personal_curriculum_df = pd.read_csv(personal_curriculum_path)

    def _is_truthy(val):
        if pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        return str(val).strip().lower() in ('1', 'true', 'yes', 'y', 't')

    elective_lookup = {
        str(code): _is_truthy(flag)
        for code, flag in zip(
            personal_curriculum_df['course_code'],
            personal_curriculum_df.get('is_elective', [])
        )
    }

    def make_labels(df):
        return [
            f"{row['source_course']} → {row['target_course']}\n"
            f"({row['source_year_quarter']} to {row['target_year_quarter']})"
            for _, row in df.iterrows()
        ]

    official_df = pd.read_csv(official_path).sort_values('similarity_score', ascending=False).head(top_n)
    personal_df = pd.read_csv(personal_path).sort_values('similarity_score', ascending=False).head(top_n)

    required_color = '#4C72B0'  # blue
    elective_color = '#2ca02c'  # green

    # Layout and font sizing for report-quality figures
    title_fs = 26
    label_fs = 20
    tick_fs = 18
    ytick_fs = 20
    legend_fs = 18

    # Use a wider figure so titles/labels have room
    fig_width = 20
    fig_height = max(len(official_df), len(personal_df)) * 1.2 + 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Bar height controls thickness; keep less than 1 to avoid overlap
    bar_height = 0.3
    n_off = len(official_df)
    spacing = bar_height + 0.06  # small gap between bars
    y_off = np.arange(n_off) * spacing
    ax1.barh(y_off, official_df['similarity_score'], height=bar_height,
             color=required_color, alpha=0.95)
    ax1.set_yticks(y_off)
    ax1.set_yticklabels(make_labels(official_df), fontsize=ytick_fs)
    ax1.invert_yaxis()
    ax1.set_xlabel('Cosine similarity', fontsize=label_fs)
    ax1.set_title(f'Official TU/e CSE (top {len(official_df)})', fontsize=title_fs, pad=18)
    ax1.tick_params(axis='x', labelsize=tick_fs)
    ax1.grid(axis='x', alpha=0.25)

    personal_colors = []
    personal_label_colors = []
    for _, row in personal_df.iterrows():
        src = str(row['source_course'])
        tgt = str(row['target_course'])
        has_elective = elective_lookup.get(src, False) or elective_lookup.get(tgt, False)
        personal_colors.append(elective_color if has_elective else required_color)
        personal_label_colors.append(elective_color if has_elective else 'black')

    n_per = len(personal_df)
    y_per = np.arange(n_per) * spacing
    ax2.barh(y_per, personal_df['similarity_score'], height=bar_height,
             color=personal_colors, alpha=0.95)
    ax2.set_yticks(y_per)
    ax2.set_yticklabels(make_labels(personal_df), fontsize=ytick_fs)
    for label, color in zip(ax2.get_yticklabels(), personal_label_colors):
        label.set_color(color)
    ax2.invert_yaxis()
    ax2.set_xlabel('Cosine similarity', fontsize=label_fs)
    ax2.set_title(f'Student-augmented (top {len(personal_df)})', fontsize=title_fs, pad=18)
    ax2.tick_params(axis='x', labelsize=tick_fs)
    ax2.grid(axis='x', alpha=0.25)

    ax2.legend(handles=[
        mpatches.Patch(color=required_color, label='Mandatory dependency pair'),
        mpatches.Patch(color=elective_color, label='Elective involved (source or target)'),
    ], fontsize=legend_fs, loc='lower right')

    plt.suptitle('Latent semantic links: official TU/e CSE vs student-augmented curriculum', fontsize=28, fontweight='bold', y=0.995)
    plt.tight_layout()
    # Restore wider panel spacing used previously
    plt.subplots_adjust(left=0.06, right=0.98, wspace=0.42)
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


def plot_personal_top_structural_risk(top_n=20, include_all_electives=False, ensure_elective=True):
    personal_path = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"
    if not personal_path.exists():
        print("Personal curriculum file not found. Skipping...")
        return

    personal_df = pd.read_csv(personal_path)

    def _is_truthy(val):
        if pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        s = str(val).strip().lower()
        return s in ('1', 'true', 'yes', 'y', 't')

    elective_lookup = {
        code: _is_truthy(val)
        for code, val in zip(personal_df['course_code'], personal_df.get('is_elective', []))
    }

    # Personal
    G_personal = load_and_build_dag(personal_path)
    risk_scores, _ = ge.compute_structural_risk_score(G_personal)
    full_personal = pd.DataFrame([
        {'course_code': c, 'structural_risk': round(v, 4)}
        for c, v in risk_scores.items()
    ]).sort_values('structural_risk', ascending=False).reset_index(drop=True)

    effective_top_n = top_n
    if ensure_elective and not include_all_electives:
        for idx, code in enumerate(full_personal['course_code']):
            if elective_lookup.get(code, False):
                if idx + 1 > effective_top_n:
                    effective_top_n = idx + 1
                break

    # Official
    struct_df = load_structural_results()
    top_official = struct_df.sort_values('structural_risk', ascending=False).head(effective_top_n)
    top_official = top_official.reset_index(drop=True)

    top_personal = full_personal.head(effective_top_n).copy()

    # Use visually distinct colors and add a thin edge for clarity
    required_color = '#4C72B0'   # blue (mandatory)
    elective_color = '#2ca02c'   # green (elective)

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(top_official['course_code'], top_official['structural_risk'],
            color=required_color, edgecolor='black', linewidth=0.6)
    ax1.set_xticks(range(len(top_official)))
    ax1.set_xticklabels(top_official['course_code'], rotation=45, ha='right')
    ax1.tick_params(axis='x', labelsize=9)
    ax1.set_ylabel('Structural risk')
    ax1.set_title(f'Top {effective_top_n}: official curriculum')

    # Build the personal panel dataset. By default keep exactly top_n; if
    # include_all_electives is True, append elective courses and re-sort.
    if include_all_electives:
        elective_codes = [c for c, v in elective_lookup.items() if v]
        elective_rows = [
            {'course_code': c, 'structural_risk': round(risk_scores.get(c, 0), 4)}
            for c in elective_codes if c in risk_scores
        ]
        if elective_rows:
            elect_df = pd.DataFrame(elective_rows)
            combined = pd.concat([top_personal, elect_df], ignore_index=True)
            combined = combined.drop_duplicates(subset='course_code', keep='first')
        else:
            combined = top_personal.copy()
        combined = combined.sort_values('structural_risk', ascending=False).reset_index(drop=True)
    else:
        combined = top_personal.copy()

    # Enforce strict top_n when not including electives
    if not include_all_electives:
        combined = combined.head(top_n)

    colors2 = [elective_color if elective_lookup.get(course_code, False) else required_color
               for course_code in combined['course_code']]

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.bar(combined['course_code'], combined['structural_risk'], color=colors2,
            edgecolor='black', linewidth=0.6)
    ax2.set_xticks(range(len(combined)))
    ax2.set_xticklabels(combined['course_code'], rotation=45, ha='right')
    ax2.tick_params(axis='x', labelsize=9)
    for label, course_code in zip(ax2.get_xticklabels(), combined['course_code']):
        if elective_lookup.get(course_code, False):
            label.set_color(elective_color)
    ax2.set_ylabel('Structural risk')
    title_suffix = f' (showing {len(combined)})' if include_all_electives else ''
    ax2.set_title(f'Top {effective_top_n}: personal curriculum{title_suffix}')
    ax2.legend(handles=[
        mpatches.Patch(color=required_color, label='Required course'),
        mpatches.Patch(color=elective_color, label='Elective'),
    ], fontsize=9)

    # Sync y-axis limits for fair visual comparison
    max_risk = max(top_official['structural_risk'].max(), combined['structural_risk'].max())
    ax1.set_ylim(0, max_risk * 1.05)
    ax2.set_ylim(0, max_risk * 1.05)

    for ax in (ax1, ax2):
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


def plot_personal_empirical_risk():
    official_path = BASE_DIR / "data" / "processed" / "semantic_analysis.csv"
    personal_path = BASE_DIR / "data" / "processed" / "semantic_analysis_personal.csv"

    if not official_path.exists() or not personal_path.exists():
        print("Semantic analysis files not found. Run semantic_analysis.py first.")
        return

    official_df = pd.read_csv(official_path)
    personal_df = pd.read_csv(personal_path)

    official_codes = set(official_df['course_code'])
    personal_codes = set(personal_df['course_code'])

    all_df = pd.concat([official_df, personal_df]).drop_duplicates(subset='course_code')
    all_df['group'] = all_df['course_code'].apply(
        lambda c: 'both' if c in official_codes and c in personal_codes
        else ('official_only' if c in official_codes else 'personal_only')
    )

    x_mid = official_df['structural_risk'].median()
    y_mid = official_df['semantic_centrality'].median()

    group_styles = {
        'official_only': ('lightgray',  'Official only'),
        'both':          ('steelblue',  'In both curricula'),
        'personal_only': ('coral',      'Personal only'),
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for group, (color, label) in group_styles.items():
        subset = all_df[all_df['group'] == group]
        ax.scatter(subset['structural_risk'], subset['semantic_centrality'],
                   color=color, s=75, alpha=0.85, zorder=3, label=label)

    for _, row in all_df.iterrows():
        ax.annotate(row['course_code'],
                    (row['structural_risk'], row['semantic_centrality']),
                    textcoords='offset points', xytext=(5, 4), fontsize=6.5, color='#333333')

    ax.axvline(x_mid, color='gray', linestyle='--', alpha=0.4, linewidth=0.9)
    ax.axhline(y_mid, color='gray', linestyle='--', alpha=0.4, linewidth=0.9)

    ax.legend(fontsize=9)
    ax.set_xlabel("Structural risk", fontsize=10)
    ax.set_ylabel("Semantic centrality", fontsize=10)
    ax.set_title("Structural risk vs semantic centrality: official vs personal curriculum", fontsize=12)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_personal_empirical_risk()
