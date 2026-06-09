import pandas as pd
import numpy as np
import json
import pathlib
import networkx as nx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graph_engine as ge
from comparative_study import load_and_build_dag


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RANDOM_SEED = 42


class SemanticCurriculumAnalyzer:
    """
    Detects hidden curricular dependencies using Sentence-BERT embeddings. Courses with high semantic similarity but no formal prerequisite edge
    represent conceptual dependencies the graph doesn't document.
    """
    def __init__(self, csv_path, structural_csv_path, model_name='all-mpnet-base-v2', model=None):
        self.csv_path = pathlib.Path(csv_path)
        self.structural_csv_path = pathlib.Path(structural_csv_path)
        self.model = model if model is not None else SentenceTransformer(model_name)
        self.df = None
        self.embeddings = None
        self.course_codes = None
        self.sim_matrix = None
        self.threshold = None
        self.hidden_deps = None
        self.semantic_centrality = None
        self.divergence_df = None
        self.correlation = None
        self.cluster_labels = None

    def load_and_encode(self):
        df = pd.read_csv(self.csv_path).fillna("")

        # skip elective placeholders and empty rows
        df = df[df['course_code'].notna() & (df['course_code'].str.strip() != "")].copy()
        df = df[~df['course_code'].str.contains("Elective", na=False)].copy()
        df = df.reset_index(drop=True)

        # Multi-quarter entries like "1,2" → use the first quarter for ordering
        df['quarter_parsed'] = df['quarter'].apply(
            lambda q: int(str(q).split(',')[0])
        )

        corpus = [
            (str(row['title']) + ". " + str(row['content']) + " " + str(row['learning_outcomes_text'])).strip()
            for _, row in df.iterrows()
        ]

        self.df = df
        self.course_codes = df['course_code'].tolist()

        print(f"Encoding {len(corpus)} courses with SBERT...")
        self.embeddings = self.model.encode(corpus, show_progress_bar=True)
        print(f"Embeddings shape: {self.embeddings.shape}")

    def compute_similarity_matrix(self):
        self.sim_matrix = cosine_similarity(self.embeddings)
        np.fill_diagonal(self.sim_matrix, 0.0)

        upper = self.sim_matrix[np.triu_indices_from(self.sim_matrix, k=1)]
        print(f"Similarity  min: {upper.min():.3f}, max: {upper.max():.3f}, mean: {upper.mean():.3f}")

    def detect_hidden_dependencies(self):
        # Top 5% of pairwise similarities, with a floor at 0.6 to avoid weak edges.
        # Below 0.6 is considered weak by SBERT literature standards.
        upper = self.sim_matrix[np.triu_indices_from(self.sim_matrix, k=1)]
        self.threshold = float(max(0.6, np.percentile(upper, 95)))
        print(f"Semantic edge threshold (95th percentile, min 0.6): {self.threshold:.4f}")

        formal_edges = set()
        for _, row in self.df.iterrows():
            target = row['course_code']
            prereqs_str = str(row.get('prerequisites_formal', ''))
            if prereqs_str.strip():
                for p in prereqs_str.split(','):
                    p_clean = p.strip()
                    if p_clean in self.course_codes:
                        formal_edges.add((p_clean, target))

        time_key = {
            row['course_code']: (int(row['year']), int(row['quarter_parsed']))
            for _, row in self.df.iterrows()
        }

        records = []
        n = len(self.course_codes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim = self.sim_matrix[i][j]
                if sim < self.threshold:
                    continue
                ca, cb = self.course_codes[i], self.course_codes[j]
                if time_key[ca] >= time_key[cb]:  # ca must come strictly before cb
                    continue
                if (ca, cb) in formal_edges:
                    continue
                # Normalize weight relative to threshold: 0 = just above threshold, 1 = strongest
                norm_w = (sim - self.threshold) / (1.0 - self.threshold)
                ya, qa = time_key[ca]
                yb, qb = time_key[cb]
                records.append({
                    'source_course': ca,
                    'target_course': cb,
                    'similarity_score': round(float(sim), 4),
                    'normalized_weight': round(float(norm_w), 4),
                    'source_year_quarter': f"Y{ya}Q{qa}",
                    'target_year_quarter': f"Y{yb}Q{qb}",
                })

        self.hidden_deps = pd.DataFrame(records).sort_values(
            'similarity_score', ascending=False
        ).reset_index(drop=True)

        print(f"Hidden semantic dependencies found: {len(self.hidden_deps)}")
        return self.hidden_deps

    def compute_semantic_centrality(self):
        # For each course, take the mean similarity to its top-3 later courses.
        # Using top-3 mean rather than a full sum avoids one outlier dominating the score.
        time_key = {
            row['course_code']: (int(row['year']), int(row['quarter_parsed']))
            for _, row in self.df.iterrows()
        }

        raw_centrality = {}
        for i, code in enumerate(self.course_codes):
            later_sims = sorted(
                [self.sim_matrix[i][j] for j, other in enumerate(self.course_codes)
                 if time_key[other] > time_key[code]],
                reverse=True
            )
            k = min(3, len(later_sims))
            raw_centrality[code] = float(np.mean(later_sims[:k])) if k > 0 else 0.0

        self.semantic_centrality = ge.normalize(raw_centrality)
        return self.semantic_centrality

    def compute_divergence(self, structural_df=None):
        if structural_df is None:
            structural_df = pd.read_csv(self.structural_csv_path)

        rows = [
            {'course_code': code, 'semantic_centrality': round(self.semantic_centrality[code], 4)}
            for code in self.course_codes
        ]
        sem_df = pd.DataFrame(rows)

        # Count how many hidden dependencies each course is the source of
        if self.hidden_deps is not None and len(self.hidden_deps) > 0:
            dep_counts = self.hidden_deps.groupby('source_course').size().reset_index(name='hidden_dep_count')
        else:
            dep_counts = pd.DataFrame(columns=['source_course', 'hidden_dep_count'])

        sem_df = sem_df.merge(
            dep_counts, left_on='course_code', right_on='source_course', how='left'
        ).drop(columns='source_course', errors='ignore')
        sem_df['hidden_dep_count'] = sem_df['hidden_dep_count'].fillna(0).astype(int)

        # Top-3 most content-similar courses per course (temporal order irrelevant here)
        top_similar = {}
        for i, code in enumerate(self.course_codes):
            sims = [(self.course_codes[j], float(self.sim_matrix[i][j]))
                    for j in range(len(self.course_codes)) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            top_similar[code] = json.dumps([c for c, _ in sims[:3]])
        sem_df['top_similar_courses'] = sem_df['course_code'].map(top_similar)

        merged = sem_df.merge(
            structural_df[['course_code', 'structural_risk']], on='course_code', how='left'
        )
        merged['structural_risk'] = merged['structural_risk'].fillna(0.0).round(4)
        # Positive divergence = semantically central but structurally invisible (hidden constraint)
        merged['divergence_score'] = (merged['semantic_centrality'] - merged['structural_risk']).round(4)

        self.divergence_df = merged[[
            'course_code', 'semantic_centrality', 'structural_risk',
            'divergence_score', 'hidden_dep_count', 'top_similar_courses'
        ]].sort_values('divergence_score', ascending=False).reset_index(drop=True)

        return self.divergence_df

    def compute_structural_semantic_correlation(self):
        # Pearson for magnitude, Spearman for rank agreement.
        # Structural risk is not normally distributed, so Spearman is the more robust measure.
        x = self.divergence_df['structural_risk']
        y = self.divergence_df['semantic_centrality']

        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)

        self.correlation = {
            'pearson':  (round(pr, 4), round(pp, 4)),
            'spearman': (round(sr, 4), round(sp, 4)),
        }

        print(f"\nCorrelation (structural risk vs semantic centrality):")
        print(f"  Pearson  r = {pr:.4f}  (p = {pp:.4f})")
        print(f"  Spearman r = {sr:.4f}  (p = {sp:.4f})")

        if abs(sr) < 0.4:
            print("  → Low: semantic layer reveals structure beyond the formal graph.")
        elif abs(sr) < 0.7:
            print("  → Moderate: some overlap, but semantic layer adds new information.")
        else:
            print("  → High: formal prerequisites already reflect conceptual structure.")

        return self.correlation

    def run_augmented_graph_experiment(self):
        # Add semantic edges to the formal DAG and recompute structural metrics.
        DATA_PATH = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"
        G_orig = load_and_build_dag(DATA_PATH)
        G_aug = G_orig.copy()

        added = 0
        for _, row in self.hidden_deps.iterrows():
            src, tgt, w = row['source_course'], row['target_course'], row['normalized_weight']
            if src in G_aug.nodes and tgt in G_aug.nodes:
                if not nx.has_path(G_aug, tgt, src):  # skip if it would create a cycle
                    G_aug.add_edge(src, tgt, weight=w)
                    added += 1

        print(f"\nAugmented graph: added {added} semantic edges (DAG preserved)")

        aug_risk_scores, _ = ge.compute_structural_risk_score(G_aug)
        orig_risk_scores, _ = ge.compute_structural_risk_score(G_orig)

        orig_rank = {c: r + 1 for r, c in enumerate(sorted(orig_risk_scores, key=orig_risk_scores.get, reverse=True))}
        aug_rank  = {c: r + 1 for r, c in enumerate(sorted(aug_risk_scores,  key=aug_risk_scores.get,  reverse=True))}

        records = [{
            'course_code':    code,
            'original_risk':  round(orig_risk_scores[code], 4),
            'augmented_risk': round(aug_risk_scores[code],  4),
            'original_rank':  orig_rank[code],
            'augmented_rank': aug_rank[code],
            'rank_shift':     orig_rank[code] - aug_rank[code],
        } for code in orig_risk_scores]

        aug_df = pd.DataFrame(records).sort_values('rank_shift', ascending=False)

        print("\nTop 5 hidden bottlenecks (greatest upward rank shift):")
        print(aug_df.head(5)[['course_code', 'original_rank', 'augmented_rank', 'rank_shift']].to_string(index=False))

        return aug_df

    def run_concept_cluster_analysis(self, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        self.cluster_labels = dict(zip(self.course_codes, labels.tolist()))
        return self.cluster_labels

    def run_pipeline(self):
        OUTPUT_DIR = BASE_DIR / "data" / "processed"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("SEMANTIC CURRICULUM ANALYSIS PIPELINE")
        print("=" * 60)

        print("\n[1/6] Loading and encoding courses...")
        self.load_and_encode()

        print("\n[2/6] Computing similarity matrix...")
        self.compute_similarity_matrix()

        print("\n[3/6] Detecting hidden dependencies...")
        self.detect_hidden_dependencies()

        print("\n[4/6] Computing semantic centrality...")
        self.compute_semantic_centrality()

        print("\n[5/6] Computing divergence scores...")
        self.compute_divergence()
        self.compute_structural_semantic_correlation()

        print("\n[6/6] Running augmented graph experiment...")
        aug_df = self.run_augmented_graph_experiment()

        self.divergence_df.to_csv(OUTPUT_DIR / "semantic_analysis.csv", index=False)
        self.hidden_deps.to_csv(OUTPUT_DIR / "hidden_dependencies.csv", index=False)
        aug_df.to_csv(OUTPUT_DIR / "augmented_graph_comparison.csv", index=False)

        print(f"\nSaved outputs to {OUTPUT_DIR}/")

        print("\n--- Top 5 hidden constraints (high semantic centrality, low structural risk) ---")
        print(self.divergence_df.head(5)[
            ['course_code', 'semantic_centrality', 'structural_risk', 'divergence_score']
        ].to_string(index=False))


def run_personal_semantic_pipeline(personal_csv_path, reuse_model, output_dir):
    """
    Runs the semantic hidden-dependency pipeline on the personal curriculum,
    reusing an already-loaded SBERT model to avoid a second heavy model load.
    Uses the personal DAG's own structural risk scores for divergence computation.
    Returns the SemanticCurriculumAnalyzer instance.
    """
    personal_csv_path = pathlib.Path(personal_csv_path)
    output_dir = pathlib.Path(output_dir)

    print("\n" + "=" * 60)
    print("PERSONAL CURRICULUM SEMANTIC PIPELINE")
    print("=" * 60)

    # Build personal DAG and compute structural risk on that subset
    G_personal = load_and_build_dag(personal_csv_path)
    personal_risk_scores, _ = ge.compute_structural_risk_score(G_personal)
    personal_struct_df = pd.DataFrame([
        {'course_code': c, 'structural_risk': round(v, 4)}
        for c, v in personal_risk_scores.items()
    ])

    # reuse model to avoid loading it twice
    analyzer = SemanticCurriculumAnalyzer(
        personal_csv_path,
        structural_csv_path=personal_csv_path,  # not read from disk; overridden below
        model=reuse_model
    )

    print("\n[1/4] Loading and encoding personal courses...")
    analyzer.load_and_encode()

    print("\n[2/4] Computing similarity matrix...")
    analyzer.compute_similarity_matrix()

    print("\n[3/4] Detecting hidden dependencies...")
    analyzer.detect_hidden_dependencies()

    print("\n[4/4] Computing semantic centrality and divergence...")
    analyzer.compute_semantic_centrality()
    analyzer.compute_divergence(structural_df=personal_struct_df)

    analyzer.hidden_deps.to_csv(output_dir / "hidden_dependencies_personal.csv", index=False)
    analyzer.divergence_df.to_csv(output_dir / "semantic_analysis_personal.csv", index=False)
    print(f"\nSaved personal semantic outputs to {output_dir}/")

    return analyzer


if __name__ == "__main__":
    CSV_PATH        = BASE_DIR / "data" / "raw" / "CSE_curriculum_data.csv"
    STRUCTURAL_PATH = BASE_DIR / "data" / "processed" / "structural_risk_baseline.csv"
    PERSONAL_PATH   = BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv"
    OUTPUT_DIR      = BASE_DIR / "data" / "processed"

    analyzer = SemanticCurriculumAnalyzer(CSV_PATH, STRUCTURAL_PATH)
    analyzer.run_pipeline()
    analyzer.run_concept_cluster_analysis(n_clusters=5)

    personal_analyzer = None
    if PERSONAL_PATH.exists():
        personal_analyzer = run_personal_semantic_pipeline(PERSONAL_PATH, analyzer.model, OUTPUT_DIR)

    print("\nGenerating visualizations...")
    from visualizations import (
        plot_similarity_heatmap,
        plot_divergence_scatter,
        plot_semantic_graph,
        plot_concept_clusters,
        plot_personal_semantic_graph,
    )
    plot_similarity_heatmap(analyzer.sim_matrix, analyzer.course_codes, analyzer.df)
    plot_divergence_scatter(analyzer.divergence_df)
    plot_semantic_graph(analyzer.hidden_deps, analyzer.threshold)
    plot_concept_clusters(analyzer.embeddings, analyzer.cluster_labels, analyzer.course_codes, analyzer.divergence_df)

    if personal_analyzer is not None:
        personal_df = pd.read_csv(BASE_DIR / "data" / "raw" / "personal_CSE_curriculum.csv")
        elective_codes = set(
            personal_df.loc[personal_df['is_elective'] == True, 'course_code'].dropna()
        )
        plot_personal_semantic_graph(
            analyzer.hidden_deps, analyzer.threshold,
            personal_analyzer.hidden_deps, personal_analyzer.threshold,
            elective_codes=elective_codes,
        )

    print("\nSemantic analysis complete.")
