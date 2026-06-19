"""Pass-rate check figure for Section 5.1.2 (defense slide)."""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "passrate_check.png"

ABOVE_COLOR = "steelblue"
BELOW_COLOR = "darkorange"
REF_COLOR = "#999999"
OUTLIER_COLOR = "#c0392b"
BAND_COLOR = "#888888"

# the only Year 2 non-targets with pass-rate data
YEAR2_REFERENCE_CODES = ["2IX20", "4CBLW00-BCS"]

XLIM = (0.50, 1.06)


def _load_data():
    curriculum = pd.read_csv(BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv")
    hidden_deps = pd.read_csv(BASE_DIR / "data" / "processed" / "hidden_dependencies.csv")
    target_codes = set(hidden_deps["target_course"].unique())

    has_rate = curriculum.dropna(subset=["pass_rate_2024"])
    is_target = curriculum["course_code"].isin(target_codes)

    excluded = curriculum[is_target & curriculum["pass_rate_2024"].isna()]

    year_stats = {}
    year_targets = {}
    for year in (1, 2):
        # mean/std over all courses with data, targets included
        year_df = has_rate[has_rate["year"] == year]
        year_stats[year] = (year_df["pass_rate_2024"].mean(), year_df["pass_rate_2024"].std())
        targets = year_df[year_df["course_code"].isin(target_codes)]
        year_targets[year] = list(zip(targets["course_code"], targets["pass_rate_2024"]))

    nontargets = [
        (code, curriculum.loc[curriculum["course_code"] == code, "pass_rate_2024"].iloc[0])
        for code in YEAR2_REFERENCE_CODES
    ]

    return year_stats, year_targets, nontargets, excluded


def _zscore(value, mean, std):
    return (value - mean) / std


def _draw_panel(ax, mean, std, targets, nontargets, title, note):
    items = []
    for code, val in targets:
        kind = "above" if val >= mean else "below"
        items.append((code, val, kind))
    for code, val in nontargets:
        items.append((code, val, "ref"))

    items.sort(key=lambda t: t[1])
    n = len(items)
    ys = list(range(n))

    ax.axvspan(mean - std, mean + std, color=BAND_COLOR, alpha=0.12, zorder=0)
    ax.axvline(mean, color=BAND_COLOR, linestyle="--", linewidth=1.3, zorder=1)

    color_map = {"above": ABOVE_COLOR, "below": BELOW_COLOR, "ref": REF_COLOR}
    marker_map = {"above": "o", "below": "o", "ref": "D"}
    marker_size = 220 if n <= 5 else 160

    for y, (code, val, kind) in zip(ys, items):
        z = _zscore(val, mean, std)
        outlier = abs(z) > 1
        color = OUTLIER_COLOR if (outlier and kind != "ref") else color_map[kind]

        ax.hlines(y, mean, val, color=color, alpha=0.5, linewidth=2.0, zorder=2)
        ax.scatter(val, y, s=marker_size, color=color, edgecolor="white",
                   linewidth=1.3, zorder=3, marker=marker_map[kind])

        ha = "left" if val >= mean else "right"
        sign = 1 if val >= mean else -1
        label = f"{val:.3f}"
        ax.annotate(label, xy=(val, y), xytext=(sign * 14, 0), textcoords="offset points",
                    va="center", ha=ha, fontsize=9.5 if n <= 5 else 9,
                    fontweight="bold" if outlier and kind != "ref" else "normal",
                    color=color if (outlier and kind != "ref") else "#222222")
        if outlier and kind != "ref":
            ax.annotate(f"{z:+.1f}σ", xy=(val, y), xytext=(sign * 14, -12), textcoords="offset points",
                        va="center", ha=ha, fontsize=8, fontweight="bold", color=color)

    ax.set_yticks(ys)
    ax.set_yticklabels([code for code, _, _ in items], fontsize=10.5 if n <= 5 else 9.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.set_xlim(*XLIM)

    ax.text(mean, -0.55, f"mean = {mean:.3f}", ha="center", va="bottom",
            fontsize=9.5, color=BAND_COLOR, style="italic")

    ax.set_xlabel("Pass rate")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_facecolor("none")
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="y", length=0)

    if note:
        ax.text(0.5, -0.24, note, transform=ax.transAxes, ha="center",
                fontsize=9, color="#666666", style="italic")


def plot_passrate_check():
    year_stats, year_targets, nontargets, excluded = _load_data()

    year1_mean, year1_std = year_stats[1]
    year2_mean, year2_std = year_stats[2]

    n_below_y2 = sum(1 for _, v in year_targets[2] if v < year2_mean)
    n_total_y2 = len(year_targets[2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.333, 7.5))

    _draw_panel(
        ax1, year1_mean, year1_std, year_targets[1], [],
        f"Year 1  (σ = {year1_std:.3f})",
        "High year-to-year spread makes the mean\na weak benchmark here",
    )
    _draw_panel(
        ax2, year2_mean, year2_std, year_targets[2], nontargets,
        f"Year 2  (σ = {year2_std:.3f}), {n_below_y2} of {n_total_y2} targets below mean",
        "All 7 targets shown, only 2IRR90 falls outside the ±1σ band,\n"
        "the rest are within normal year-to-year spread",
    )

    fig.suptitle("Pass-rate check: latent semantic link target courses vs. year mean",
                 fontsize=16, fontweight="bold", y=0.99)

    excl_text = "; ".join(
        f"{row.course_code} (Year {row.year}, Q{row.quarter})"
        for row in excluded.itertuples()
    )
    fig.text(0.5, 0.925, f"Excluded: {excl_text}, no pass-rate data available",
             ha="center", fontsize=9.5, color="#666666", style="italic")

    legend_handles = [
        mpatches.Patch(color=ABOVE_COLOR, label="Above mean"),
        mpatches.Patch(color=BELOW_COLOR, label="Below mean, normal spread"),
        mpatches.Patch(color=OUTLIER_COLOR, label="Outside ±1σ (outlier)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=REF_COLOR,
               markersize=9, label="Non-target reference"),
        Line2D([0], [0], color=BAND_COLOR, linestyle="--", linewidth=1.3, label="Year mean"),
        mpatches.Patch(color=BAND_COLOR, alpha=0.12, label="±1σ band"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.20, 1, 0.90])
    fig.savefig(OUT_PATH, dpi=300, transparent=True)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    plot_passrate_check()
