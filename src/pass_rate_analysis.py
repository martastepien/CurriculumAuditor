"""
Pass-rate analysis: do hidden-dependency target courses have lower pass rates
than non-target courses in the same year group?
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def analyse_pass_rates(save: bool = True) -> pd.DataFrame:
    curriculum = pd.read_csv(BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv")
    hidden_deps = pd.read_csv(BASE_DIR / "data" / "processed" / "hidden_dependencies.csv")

    # Drop courses without pass rate data (all Year 3 + elective placeholders in Year 2)
    df = curriculum.dropna(subset=["pass_rate_2024"]).copy()

    excluded = curriculum[curriculum["pass_rate_2024"].isna()]["course_code"].tolist()

    # Mark target courses (appear as B in any A → B hidden dependency)
    target_codes = set(hidden_deps["target_course"].unique())
    df["is_target"] = df["course_code"].isin(target_codes)

    year_stats = (
        df[~df["is_target"]]
        .groupby("year")["pass_rate_2024"]
        .agg(year_mean_non_targets="mean", year_std_non_targets="std")
        .reset_index()
    )

    targets = (
        df[df["is_target"]][["course_code", "year", "pass_rate_2024", "serious_pass_rate_2024"]]
        .copy()
        .rename(columns={"pass_rate_2024": "pass_rate", "serious_pass_rate_2024": "serious_pass_rate"})
    )
    targets = targets.merge(year_stats, on="year", how="left")
    targets["difference"] = targets["pass_rate"] - targets["year_mean_non_targets"]
    targets = targets.sort_values("difference", ascending=True).reset_index(drop=True)

    print("=" * 70)
    print("Pass-Rate Analysis: Hidden Dependency Target Courses vs Year Peers")
    print("=" * 70)

    pd.set_option("display.float_format", "{:.4f}".format)
    print(targets.to_string(index=False))

    if excluded:
        print(
            f"\nWARNING: {len(excluded)} courses excluded (null pass_rate_2024): "
            + ", ".join(excluded)
        )

    print("\n" + "-" * 70)
    print("Top 5 hidden dependencies by cosine similarity, target pass-rate check")
    print("-" * 70)

    top5 = hidden_deps.nlargest(5, "similarity_score").reset_index(drop=True)

    # Build lookup: course_code -> (pass_rate, year_mean_non_targets)
    rate_lookup = targets.set_index("course_code")[["pass_rate", "year_mean_non_targets"]]

    seen_targets = {}
    for _, row in top5.iterrows():
        src, tgt, sim = row["source_course"], row["target_course"], row["similarity_score"]

        if tgt in seen_targets:
            print(f"  {src} -> {tgt}  (sim={sim:.4f}): same target as above ({seen_targets[tgt]})")
            continue

        if tgt in rate_lookup.index:
            pr = rate_lookup.at[tgt, "pass_rate"]
            ym = rate_lookup.at[tgt, "year_mean_non_targets"]
            below = pr < ym
            label = "YES - BELOW AVERAGE" if below else "NO  - above average"
            seen_targets[tgt] = label
            print(
                f"  {src} -> {tgt}  (sim={sim:.4f}): "
                f"pass_rate={pr:.4f}  year_mean={ym:.4f}  → {label}"
            )
        else:
            seen_targets[tgt] = "N/A"
            print(f"  {src} -> {tgt}  (sim={sim:.4f}): N/A (no pass-rate data for {tgt})")

    if save:
        out_path = BASE_DIR / "data" / "processed" / "pass_rate_analysis.csv"
        targets.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    return targets


if __name__ == "__main__":
    analyse_pass_rates(save=True)
