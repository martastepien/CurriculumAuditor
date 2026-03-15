import pandas as pd
import networkx as nx
import os
import graph_engine as ge
import pathlib


#loading and building DAG from CSV
def load_and_build_dag(csv_path):
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    G = nx.DiGraph()

    for _, row in df.iterrows():
        # Handle multi-quarter courses (e.g., "1,2") by taking first quarter
        quarter_str = str(row.get("quarter", "1"))
        quarter = int(quarter_str.split(",")[0]) if "," in quarter_str else int(quarter_str)
        
        G.add_node(
            row["course_code"],
            credits=float(row.get("credits", 5)),
            year=int(row.get("year", 1)),
            quarter=quarter
        )

    for _, row in df.iterrows():
        target = row["course_code"]
        prereqs = str(row.get("prerequisites_formal", ""))

        if prereqs:
            for p in prereqs.split(","):
                p_clean = p.strip()
                if p_clean in G.nodes:
                    G.add_edge(p_clean, target)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Curriculum contains cycles.")

    return G


#pipeline to run structural risk analysis and save results
def run_structural_pipeline():

    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "structural_risk_baseline.csv"

    G = load_and_build_dag(DATA_PATH)

    risk_scores, metrics = ge.compute_structural_risk_score(G)

    results = []

    for node in G.nodes():
        results.append({
            "course_code": node,
            "structural_risk": round(risk_scores[node], 4),
            "blocking_factor": round(metrics["block"][node], 4),
            "betweenness": round(metrics["bet"][node], 4),
            "pagerank": round(metrics["pagerank"][node], 4),
            "logical_depth": round(metrics["logical"][node], 4),
            "temporal_criticality": round(metrics["temporal"][node], 4),
            "articulation_impact": round(metrics["articulation"][node], 4),
            "year": G.nodes[node]["year"],
            "quarter": G.nodes[node]["quarter"]
        })

    df = pd.DataFrame(results).sort_values(
        "structural_risk",
        ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nStructural risk analysis complete.")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_structural_pipeline()


