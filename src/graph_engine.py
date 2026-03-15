import networkx as nx
import numpy as np


# Defining elective groups- like capstone courses
ELECTIVE_GROUPS = {
    "capstone_group": {"2IRR60", "2IRR70", "2IRR80"}
}
def is_in_elective_group(node):
    for group in ELECTIVE_GROUPS.values():
        if node in group:
            return True
    return False

def get_group_members(node):
    for group in ELECTIVE_GROUPS.values():
        if node in group:
            return group
    return None

#longest path
def compute_longest_path_depth(G):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG.")

    topo = list(nx.topological_sort(G))
    longest = {node: 0 for node in topo}

    for node in topo:
        for successor in G.successors(node):
            longest[successor] = max(
                longest[successor],
                longest[node] + 1
            )

    return longest

#blocking factor
def compute_blocking_factor(G):
    credit_attrs = nx.get_node_attributes(G, "credits")
    total_credits = sum(credit_attrs.values()) if credit_attrs else 1

    blocking = {}

    for node in G.nodes():

        descendants = nx.descendants(G, node)

        # Remove elective siblings from descendant set
        if is_in_elective_group(node):
            group = get_group_members(node)
            descendants = {
                d for d in descendants
                if d not in group
            }

        downstream_credits = sum(
            G.nodes[d].get("credits", 0)
            for d in descendants
        )

        blocking[node] = downstream_credits / total_credits
    return blocking


#articulation reach impact
def compute_articulation_reach(G):

    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    def reachable_count(graph):
        reachable = set()
        for r in roots:
            if r in graph:
                reachable.update(nx.descendants(graph, r))
                reachable.add(r)
        return len(reachable)

    baseline = reachable_count(G)
    impact = {}

    for node in G.nodes():
        # If elective, simulate removal but check if sibling remains
        if is_in_elective_group(node):
            group = get_group_members(node)

            # If removing node still leaves another elective alive,
            # no structural collapse occurs.
            remaining = group - {node}
            if any(r in G.nodes for r in remaining):
                impact[node] = 0
                continue
        G_temp = G.copy()
        G_temp.remove_node(node)
        new_count = reachable_count(G_temp)

        impact[node] = (
            (baseline - new_count) / baseline
            if baseline > 0 else 0
        )
    return impact


# Temporal criticality - weighted longest path in QUARTERS
def compute_temporal_criticality(G):
    """
    Weighted longest path: Measures how far into the curriculum (in quarters)
    a student must progress to reach each course. Captures the actual time-cost of failure.
    Unlike logical depth (which counts hops), this accounts for:
    - Course scheduling (year/quarter positions)
    - Time gaps between prerequisites and dependents
    - Maximum graduation delay potential if a course is failed or delayed.
    Returns: Dict mapping each node to its maximum "quarter reach" from degree start.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Temporal criticality requires a DAG.")
    topo = list(nx.topological_sort(G))
    
    # Calculate absolute time position for each node (quarters from degree start)
    node_time = {
        n: (G.nodes[n]['year'] - 1) * 4 + G.nodes[n]['quarter'] 
        for n in G.nodes
    }
    # max_time_reach[v] = furthest calendar quarter needed to reach node v
    # Initialize with the node's own scheduled time
    max_time_reach = {node: node_time[node] for node in G.nodes}
    
    for u in topo:
        for v in G.successors(u):
            # v depends on u, so v can only be completed AFTER u is done
            # The +1 represents that you must complete at least one more time period
            max_time_reach[v] = max(
                max_time_reach[v], 
                max_time_reach[u] + 1
            )
    return max_time_reach


# Min max normalization.
def normalize(metric_dict):
    if not metric_dict:
        return {}

    values = np.array(list(metric_dict.values()))
    min_val = values.min()
    max_val = values.max()

    if max_val - min_val == 0:
        return {k: 0 for k in metric_dict}

    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in metric_dict.items()
    }


# Multidimensional structural risk: computes all 6 candidate metrics
def compute_structural_risk_score(G, weights=None, selected_metrics=None):
    """
    Computes all 6 structural metrics:
    - block:       Blocking Factor — downstream credit impact
    - bet:         Betweenness — logical bottleneck position
    - pagerank:    PageRank — accumulated upstream dependency prestige
    - logical:     Logical Depth — knowledge chain depth (unweighted hops)
    - temporal:    Temporal Criticality — graduation delay risk (quarters)
    - articulation: Articulation Impact — reachability drop on node removal

    selected_metrics controls which subset is used for the composite risk score.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Structural risk requires a DAG.")

    if selected_metrics is None:
        selected_metrics = ["block", "bet", "logical", "temporal"]

    if weights is None:
        w = 1.0 / len(selected_metrics)
        weights = {m: w for m in selected_metrics}

    metrics_raw = {
        "block":        compute_blocking_factor(G),
        "bet":          nx.betweenness_centrality(G, normalized=True),
        "pagerank":     nx.pagerank(G, alpha=0.85),
        "logical":      compute_longest_path_depth(G),
        "temporal":     compute_temporal_criticality(G),
        "articulation": compute_articulation_reach(G),
    }

    metrics = {name: normalize(values) for name, values in metrics_raw.items()}

    risk_scores = {}
    for node in G.nodes():
        risk_scores[node] = sum(
            metrics[m][node] * weights.get(m, 0)
            for m in selected_metrics
        )
    return risk_scores, metrics


