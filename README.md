# CurriculumAuditor
A structural risk analysis tool for curriculum design that identifies critical courses based on graph-theoretic metrics and dependency analysis. Meant to help answer the reaserch question of "How do graph-
theoretic metrics and neural models differ in their identification of curricular
risk, and what do these differences reveal about ’hidden’ structural constraints
in university degree programs?"

## Overview

CurriculumAuditor analyzes course prerequisite structures to identify "bottleneck" courses that could delay student progression if failed. It builds a directed acyclic graph (DAG) from course dependencies and computes multiple risk metrics to highlight structurally critical courses.

## Features

- **Graph-based curriculum modeling**: Converts course prerequisites into a DAG
- **Three-metric risk analysis** (minimizes redundancy while maintaining interpretability):
  - **Blocking factor**: Measures proportion of downstream credits affected by failure
  - **Betweenness centrality**: Identifies courses on critical prerequisite paths (structural bottlenecks)
  - **Delay depth**: Calculates longest path depth to quantify maximum graduation delays
- **Elective group support**: Handles flexible course options (e.g., capstone choices)
- **Visualization tools**: Charts for top risk courses and metric distributions

## Project structure

```
BEP/
├── data/
│   ├── raw/
│   │   └── CSE_curriculum_data.csv    # Input curriculum data
│   └── processed/
│       └── structural_risk_baseline.csv # Output risk scores
├── src/
│   ├── graph_engine.py                # Core risk metric algorithms
│   ├── comparative_study.py           # Pipeline for analysis
│   ├── visualizations.py              # Plotting utilities
│   ├── risk_model_gnn.py             # (Future: GNN models)
│   └── semantic_analysis.py          # (Future: NLP analysis)
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install pandas networkx numpy matplotlib
  ```

## Usage

### 1. Prepare your data

Create a CSV file with the following required columns:

| Column | Description | Example |
|--------|-------------|---------|
| `course_code` | Unique course identifier | `2IT60` |
| `title` | Course name | `Logic and Set Theory` |
| `credits` | Credit hours | `5` |
| `year` | Study year | `1` |
| `prerequisites_formal` | Comma-separated prerequisite codes | `2IT60, 2IT80, 2IRR90` |

**Important**: Multiple prerequisites should be comma-separated (spaces optional). For example `2IT60, 2IT80, 2IRR90` and `2IT60,2IT80,2IRR90` are acceptable formats.

Optional columns: `quarter`, `learning_line`, `prerequisites_informal`, `category`, `pass_rate`, etc.

### 2. Run Structural Analysis

```bash
cd src
python comparative_study.py
```

This will:
- Load your curriculum data from `data/raw/CSE_curriculum_data.csv`
- Build the prerequisite graph
- Compute all risk metrics
- Save results to `data/processed/structural_risk_baseline.csv`
- Display top 10 highest-risk courses

### 3. Visualize Results

```python
from visualizations import plot_top_structural_risk, plot_metric_distribution

# Show top 15 highest risk courses
plot_top_structural_risk(top_n=15)

# View distribution of blocking factors
plot_metric_distribution("blocking_factor")

# View other metrics
plot_metric_distribution("betweenness")
plot_metric_distribution("delay_depth")
plot_metric_distribution("structural_risk")
```

## Configuring Elective Groups

Some curricula have course groups where students choose one option (e.g., capstone projects). Configure these in [graph_engine.py](src/graph_engine.py):

```python
ELECTIVE_GROUPS = {
    "capstone_group": {"2IRR60", "2IRR70", "2IRR80"},
    "specialization_group": {"2IC51", "2IC52", "2IC53"}
}
```

Elective siblings won't count toward each other's blocking factors.

## Output format
The processed CSV contains:
- `course_code`: Course identifier
- `structural_risk`: Aggregated risk score (0-1, normalized)
- `blocking_factor`: Proportion of downstream credits blocked
- `betweenness`: Network centrality score (critical path position)
- `delay_depth`: Longest path from this course (maximum delay potential)
- `year`: Academic year

## Risk score calculation

The risk score uses **three non-redundant metrics** based on research by Saqr & López-Pernas, which demonstrated that many graph metrics are highly correlated. This selection minimizes redundancy while maintaining distinct conceptual dimensions:

Default weights (customizable in `compute_structural_risk_score()`):
- **Blocking Factor: 40%** - Direct downstream credit impact
- **Betweenness: 30%** - Critical path position (structural bottlenecks)
- **Delay Depth: 30%** - Maximum delay potential for graduation

Each metric captures a unique aspect of curricular risk:
- **Blocking factor** → Volume of affected content (how much is blocked?)
- **Betweenness** → Structural position (how critical is the path?)
- **Delay depth** → Temporal impact (how long is the delay?)
