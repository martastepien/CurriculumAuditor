"""
Recompute composite structural risk before and after adding semantic edges
as formal prerequisites, then plot a side-by-side top-10 ranking comparison.
"""

import pathlib
import sys

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graph_engine as ge
from comparative_study import load_and_build_dag

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
CURRICULUM_CSV  = BASE_DIR / "data" / "raw"  / "CSE_curriculum_data.csv"
HIDDEN_DEPS_CSV = BASE_DIR / "data" / "processed" / "hidden_dependencies.csv"
OUTPUT_CSV      = BASE_DIR / "data" / "processed" / "augmented_risk_recomputed.csv"


def build_augmented_dag(base_dag: nx.DiGraph, hidden_deps: pd.DataFrame) -> nx.DiGraph:
    """
    Clone the base DAG and add every semantic edge from hidden_deps that
    (a) connects two existing nodes and (b) does not introduce a cycle.
    Direction: source_course → target_course (same convention as formal prereqs).
    """
    G = base_dag.copy()
    added = skipped_missing = skipped_cycle = 0

    for _, row in hidden_deps.iterrows():
        src, tgt = row["source_course"], row["target_course"]
        if src not in G.nodes or tgt not in G.nodes:
            skipped_missing += 1
            continue
        if G.has_edge(src, tgt):
            continue  # already a formal edge
        G.add_edge(src, tgt)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(src, tgt)
            skipped_cycle += 1
        else:
            added += 1

    print(f"Semantic edges added: {added}  |  skipped (missing node): {skipped_missing}  |  skipped (cycle): {skipped_cycle}")
    return G


def compute_risk_df(G: nx.DiGraph) -> pd.DataFrame:
    risk_scores, _ = ge.compute_structural_risk_score(G)
    return pd.DataFrame(
        [{"course_code": c, "structural_risk": round(v, 4)} for c, v in risk_scores.items()]
    ).sort_values("structural_risk", ascending=False).reset_index(drop=True)


def plot_top10_comparison(before_df: pd.DataFrame, after_df: pd.DataFrame, top_n: int = 10):
    """
    Side-by-side grouped bar chart.
    Shows the union of top-N courses from both rankings with their
    before/after scores as adjacent bars for each course.
    """
    top_before = set(before_df.head(top_n)["course_code"])
    top_after  = set(after_df.head(top_n)["course_code"])
    union_codes = list(top_before | top_after)

    before_map = dict(zip(before_df["course_code"], before_df["structural_risk"]))
    after_map  = dict(zip(after_df["course_code"],  after_df["structural_risk"]))

    rank_before = {c: i for i, c in enumerate(before_df["course_code"])}
    # Sort by augmented rank so the biggest post-augmentation risks read left to right
    rank_after_sort = {c: i for i, c in enumerate(after_df["course_code"])}
    union_codes.sort(key=lambda c: rank_after_sort.get(c, 9999))

    before_vals = [before_map.get(c, 0.0) for c in union_codes]
    after_vals  = [after_map.get(c,  0.0) for c in union_codes]

    n      = len(union_codes)
    x      = np.arange(n)
    width  = 0.35

    color_before = "#4C72B0"   # TU/e blue
    color_after  = "#DD8452"   # warm orange

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_b = ax.bar(x - width / 2, before_vals, width, label="Before augmentation",
                    color=color_before, edgecolor="white", linewidth=0.6, zorder=3)
    bars_a = ax.bar(x + width / 2, after_vals,  width, label="After augmentation",
                    color=color_after,  edgecolor="white", linewidth=0.6, zorder=3)

    # Annotate bars whose rank changed noticeably
    rank_after = {c: i for i, c in enumerate(after_df["course_code"])}
    for i, code in enumerate(union_codes):
        rb = rank_before.get(code, None)
        ra = rank_after.get(code, None)
        if rb is not None and ra is not None:
            shift = rb - ra  # positive = moved up
            if abs(shift) >= 2:
                label_txt = f"▲{shift}" if shift > 0 else f"▼{abs(shift)}"
                label_col = "darkgreen" if shift > 0 else "crimson"
                ax.text(i, max(before_vals[i], after_vals[i]) + 0.008,
                        label_txt, ha="center", va="bottom",
                        fontsize=8, color=label_col, fontweight="bold")

    # Shade courses that are NEW to the top-10 after augmentation
    new_entries = top_after - top_before
    for i, code in enumerate(union_codes):
        if code in new_entries:
            ax.axvspan(i - 0.5, i + 0.5, color="lightyellow", alpha=0.55, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(union_codes, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Composite structural risk score", fontsize=11)
    ax.set_title(
        f"Structural risk: top-{top_n} ranking before vs after adding semantic edges as formal prerequisites\n"
        f"(yellow shading = new entrant to top-{top_n} after augmentation; ▲▼ = rank shift ≥ 2)",
        fontsize=11, pad=12
    )
    ax.set_ylim(0, max(max(before_vals), max(after_vals)) * 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = BASE_DIR / "augmented_risk_top10.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out_path}")
    plt.show()


def main():
    print("Building base DAG …")
    G_base = load_and_build_dag(CURRICULUM_CSV)
    before_df = compute_risk_df(G_base)
    print(f"Base graph: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges")

    print("\nLoading hidden semantic dependencies …")
    hidden_deps = pd.read_csv(HIDDEN_DEPS_CSV)
    print(f"  {len(hidden_deps)} semantic edges found")

    print("\nBuilding augmented DAG …")
    G_aug = build_augmented_dag(G_base, hidden_deps)
    after_df = compute_risk_df(G_aug)
    print(f"Augmented graph: {G_aug.number_of_nodes()} nodes, {G_aug.number_of_edges()} edges")

    merged = before_df.rename(columns={"structural_risk": "risk_before"}).merge(
        after_df.rename(columns={"structural_risk": "risk_after"}),
        on="course_code", how="outer"
    ).fillna(0)
    merged["rank_before"] = merged["risk_before"].rank(ascending=False, method="min").astype(int)
    merged["rank_after"]  = merged["risk_after"].rank(ascending=False, method="min").astype(int)
    merged["rank_shift"]  = merged["rank_before"] - merged["rank_after"]
    merged = merged.sort_values("risk_before", ascending=False)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull comparison saved → {OUTPUT_CSV}")

    print("\n=== Top 10 BEFORE augmentation ===")
    print(before_df.head(10).to_string(index=False))
    print("\n=== Top 10 AFTER augmentation ===")
    print(after_df.head(10).to_string(index=False))

    print("\n=== Biggest rank changes ===")
    changed = merged[merged["rank_shift"].abs() >= 2].sort_values("rank_shift", ascending=False)
    print(changed[["course_code", "risk_before", "risk_after", "rank_before", "rank_after", "rank_shift"]].to_string(index=False))

    print("\nGenerating comparison chart …")
    plot_top10_comparison(before_df, after_df, top_n=10)


if __name__ == "__main__":
    main()
