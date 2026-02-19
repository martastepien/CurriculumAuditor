import pandas as pd
import matplotlib.pyplot as plt
import pathlib


def load_structural_results():
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "structural_risk_baseline.csv"

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed file not found at {DATA_PATH}. "
            "Run comparative_study.py first."
        )

    df = pd.read_csv(DATA_PATH)
    return df

#Plots top N courses by structural risk
def plot_top_structural_risk(top_n=10):
    df = load_structural_results()

    top = df.sort_values(
        "structural_risk",
        ascending=False
    ).head(top_n)

    plt.figure()
    plt.bar(top["course_code"], top["structural_risk"])
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} Structural Risk Courses")
    plt.tight_layout()
    plt.show()

#Plots distribution of any metric column. Example: 'betweenness', 'blocking_factor', etc.
def plot_metric_distribution(metric_name):
    df = load_structural_results()

    if metric_name not in df.columns:
        raise ValueError(
            f"{metric_name} not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    plt.figure()
    plt.hist(df[metric_name], bins=15)
    plt.title(f"Distribution of {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_top_structural_risk(top_n=10)
