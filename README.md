# CurriculumAuditor

A tool for auditing university curricula by combining graph-theoretic risk metrics with semantic content analysis. Built as part of a bachelor's thesis investigating how these two approaches differ in identifying high-risk courses and what those differences reveal about hidden structural constraints in degree programs.

## What it does

Takes a course prerequisite graph, computes structural risk scores per course, then runs a semantic analysis to find latent content dependencies that the formal prerequisite structure misses. The two perspectives are compared to surface courses that are riskier than they look (or less risky than they appear).

## Project structure

```
BEP/
├── data/
│   ├── raw/                          # Input CSVs (curriculum data)
│   └── processed/                    # Computed risk scores and analysis output
├── src/
│   ├── graph_engine.py               # Risk metric computation
│   ├── comparative_study.py          # Main analysis pipeline
│   ├── semantic_analysis.py          # Semantic similarity + latent semantic links detection
│   └── visualizations.py            # All plots
```

## Risk metrics

Three metrics contribute equally (1/3 each) to the composite `structural_risk` score:

- Blocking factor: Share of downstream credits blocked if this course is failed
- Betweenness centrality: How often this course sits on critical prerequisite paths
- PageRank: Importance weighted by the importance of predecessors

All scores are normalized to 0–1 before combining.

## Input format

CSV with columns: `course_code`, `title`, `credits`, `year`, `prerequisites_formal` (comma-separated, e.g. `2IT60, 2IT80`).

## Elective groups

Courses where students pick one option can be grouped so siblings don't inflate each other's blocking factors. Configure in `graph_engine.py`:

```python
ELECTIVE_GROUPS = {
    "capstone_group": {"2IRR60", "2IRR70", "2IRR80"},
}
```
