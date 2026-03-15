import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
from itertools import combinations
from collections import defaultdict


ALL_METRICS = ['blocking_factor', 'betweenness', 'pagerank', 'logical_depth',
               'temporal_criticality', 'articulation_impact']

METRIC_LABELS = {
    'blocking_factor':    'Blocking\nFactor',
    'betweenness':        'Betweenness',
    'pagerank':           'PageRank',
    'logical_depth':      'Logical\nDepth',
    'temporal_criticality': 'Temporal\nCriticality',
    'articulation_impact': 'Articulation\nImpact',
}


def load_structural_results():
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "structural_risk_baseline.csv"

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed file not found at {DATA_PATH}. "
            "Run comparative_study.py first."
        )
    return pd.read_csv(DATA_PATH)


def load_personal_curriculum():
    """
    Load personal curriculum and map elective course codes to the placeholder
    names used in the main database (matched by year/quarter).
    """
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    PERSONAL_PATH = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"
    MAIN_PATH = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"

    if not PERSONAL_PATH.exists():
        return None

    personal_df = pd.read_csv(PERSONAL_PATH)
    main_df = pd.read_csv(MAIN_PATH)

    # Collect elective placeholder slots from main database, keyed by (year, quarter)
    slots = defaultdict(list)
    for _, row in main_df.iterrows():
        code = str(row['course_code'])
        if 'Elective' in code:
            q_str = str(row['quarter'])
            q = int(q_str.split(',')[0]) if ',' in q_str else int(q_str)
            slots[(int(row['year']), q)].append(code)

    # Maps db_code (placeholder or real) -> display label (real course code)
    label_map = {}
    slot_usage = defaultdict(int)

    for _, row in personal_df.iterrows():
        code = row['course_code']
        if pd.isna(code):
            continue
        code = str(code)
        is_elective = bool(row.get('is_elective', False))
        if is_elective and not pd.isna(row.get('year')) and not pd.isna(row.get('quarter')):
            q_str = str(row['quarter'])
            q = int(q_str.split(',')[0]) if ',' in q_str else int(q_str)
            year = int(row['year'])
            key = (year, q)
            available = slots.get(key, [])
            idx = slot_usage[key]
            if idx < len(available):
                placeholder = available[idx]
                label_map[placeholder] = code  # display real code, match on placeholder
                slot_usage[key] += 1
        else:
            label_map[code] = code

    return label_map


def _best_metric_subset(corr_matrix, metrics, n):
    """Return the n metrics with minimum average absolute pairwise correlation."""
    best_subset, best_score = None, float('inf')
    for subset in combinations(metrics, n):
        sub = corr_matrix.loc[list(subset), list(subset)]
        k = len(subset)
        score = (sub.abs().sum().sum() - k) / (k * (k - 1))
        if score < best_score:
            best_score, best_subset = score, subset
    return list(best_subset), best_score


# Plots top N courses by structural risk
def plot_top_structural_risk(top_n=10):
    df = load_structural_results()
    top = df.sort_values("structural_risk", ascending=False).head(top_n)

    plt.figure()
    plt.bar(top["course_code"], top["structural_risk"])
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} Structural Risk Courses")
    plt.tight_layout()
    plt.show()


def plot_metric_correlations(n_select=3):
    """
    Shows a 6×6 correlation matrix for all candidate metrics (filtered to my
    curriculum), then auto-selects the n_select least-correlated subset and
    shows it side by side.
    """
    df = load_structural_results()
    personal_courses = load_personal_curriculum()

    if personal_courses is not None:
        df = df[df['course_code'].isin(personal_courses.keys())]

    # Drop any metric columns that are all-zero or missing (e.g. if CSV is old)
    available = [m for m in ALL_METRICS if m in df.columns and df[m].std() > 0]
    corr_matrix = df[available].corr()

    # Auto-select best subset
    selected, avg_corr = _best_metric_subset(corr_matrix, available, n_select)
    sub_corr = corr_matrix.loc[selected, selected]

    x_labels_all = [METRIC_LABELS[m] for m in available]
    x_labels_sel = [METRIC_LABELS[m] for m in selected]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax1,
                xticklabels=x_labels_all, yticklabels=x_labels_all)
    title_suffix = " (My Curriculum)" if personal_courses is not None else ""

    ax1.set_title(f'All {len(available)} Metrics: Correlation Matrix{title_suffix}')

    sns.heatmap(sub_corr, annot=True, fmt='.4f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax2,
                xticklabels=x_labels_sel, yticklabels=x_labels_sel)
    ax2.set_title(f'Best {n_select} (Least Redundant)\nAvg |corr| = {avg_corr:.4f}')

    plt.suptitle('Metric Redundancy Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\nAuto-selected {n_select} least-correlated metrics: {selected}")
    print(f"Average absolute pairwise correlation: {avg_corr:.3f}")
    print("\nFull correlation matrix:")
    print(corr_matrix.round(4))


def plot_personal_curriculum_risk():
    """Shows my curriculum courses ranked by structural risk."""
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top 15 overall, colour-coded: mine in red, others in steelblue
    top_n = 15
    top_all = df.nlargest(top_n, 'structural_risk').copy()
    top_all['display_code'] = top_all['course_code'].map(
        lambda c: personal_courses.get(c, c)
    )
    colors = ['red' if c in personal_courses else 'steelblue'
              for c in top_all['course_code']]
    ax1.barh(range(len(top_all)), top_all['structural_risk'], color=colors)
    ax1.set_yticks(range(len(top_all)))
    ax1.set_yticklabels(top_all['display_code'])
    ax1.set_xlabel('Structural Risk')
    ax1.set_title(f'Top {top_n} Highest Risk Courses, Overall (red = my courses)')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Top 15 of my courses only
    top_personal = personal_df.head(top_n)
    ax2.barh(range(len(top_personal)), top_personal['structural_risk'], color='coral')
    ax2.set_yticks(range(len(top_personal)))
    ax2.set_yticklabels(top_personal['display_code'])
    ax2.set_xlabel('Structural Risk')
    ax2.set_title(f'My Top {top_n} Highest Risk Courses')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    top_course = personal_df.iloc[0]
    print(f"\nMy curriculum statistics:")
    print(f"Total matched courses: {len(personal_df)}")
    print(f"Average risk: {personal_df['structural_risk'].mean():.3f}")
    print(f"Highest risk: {top_course['display_code']} ({top_course['structural_risk']:.3f})")


if __name__ == "__main__":
    print("CurriculumAuditor visualizations\n")

    print("1. Top structural risk courses (all curriculum)")
    plot_top_structural_risk(top_n=10)

    print("\n2. Metric correlation analysis — all 6 metrics + auto-selected best 3")
    plot_metric_correlations(n_select=3)

    print("\n3. My curriculum risk analysis")
    plot_personal_curriculum_risk()

    print("\nAll visualizations complete!")
