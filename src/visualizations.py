import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pathlib
import numpy as np


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


def load_personal_curriculum():
    """Load the personal curriculum to identify which courses the student is taking"""
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"
    
    if not DATA_PATH.exists():
        return None
    
    df = pd.read_csv(DATA_PATH)
    return set(df['course_code'].values)

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


def plot_metric_correlations():
    """Shows correlation between the 3 metrics to validate non-redundancy"""
    df = load_structural_results()
    
    metrics = ['blocking_factor', 'betweenness', 'delay_depth']
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation'})
    plt.title('Metric Correlation Matrix\n(Low correlation = non-redundant metrics)')
    plt.tight_layout()
    plt.show()
    
    print("\nCorrelation Analysis:")
    print(corr_matrix)
    print("\nInterpretation: Values close to 0 indicate metrics capture different aspects of risk")


def plot_personal_curriculum_risk():
    """Highlights courses in your personal curriculum and their risk levels"""
    df = load_structural_results()
    personal_courses = load_personal_curriculum()
    
    if personal_courses is None:
        print("Personal curriculum file not found. Skipping...")
        return
    df['in_personal'] = df['course_code'].isin(personal_courses)
    # Separate personal and non-personal
    personal_df = df[df['in_personal']].sort_values('structural_risk', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    #Top risk courses with personal ones being highlighted
    top_n = 15
    top_df = df.nlargest(top_n, 'structural_risk')
    colors = ['red' if x else 'steelblue' for x in top_df['in_personal']]
    
    ax1.barh(range(len(top_df)), top_df['structural_risk'], color=colors)
    ax1.set_yticks(range(len(top_df)))
    ax1.set_yticklabels(top_df['course_code'])
    ax1.set_xlabel('Structural Risk')
    ax1.set_title(f'Top {top_n} Highest Risk Courses (Red = Your Courses)')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # My personal courses- ranked by risk
    if len(personal_df) > 0:
        top_personal = personal_df.head(20)
        ax2.barh(range(len(top_personal)), top_personal['structural_risk'], color='coral')
        ax2.set_yticks(range(len(top_personal)))
        ax2.set_yticklabels(top_personal['course_code'])
        ax2.set_xlabel('Structural Risk')
        ax2.set_title('Your Highest Risk Courses')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        print(f"\nPersonal curriculum statistics:")
        print(f"Total courses in your curriculum: {len(personal_df)}")
        print(f"Average risk in your courses: {personal_df['structural_risk'].mean():.3f}")
        print(f"Highest risk course: {personal_df.iloc[0]['course_code']} ({personal_df.iloc[0]['structural_risk']:.3f})")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("CurriculumAuditor visualizations\n")
    
    print("1. Top structural risk courses")
    plot_top_structural_risk(top_n=10)
    
    print("\n2. Metric correlation analysis (validates 3-metric approach)")
    plot_metric_correlations()
    
    print("\n3. Personal curriculum risk analysis")
    plot_personal_curriculum_risk()
    
    print("\nAll visualizations complete!")

    
