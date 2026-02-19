import networkx as nx
import numpy as np

#longest path
def compute_longest_path_depth(G):
    """Measures systemic delay risk by identifying longest prerequisite chains. Assumes G is a DAG."""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG for the longest path computation.")

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
    """Downstream credit impact normalized by total curriculum credits."""
    credit_attrs = nx.get_node_attributes(G, "credits")
    total_credits = sum(credit_attrs.values()) if credit_attrs else 1

    blocking = {}

    for node in G.nodes():
        descendants = nx.descendants(G, node)
        downstream_credits = sum(
            G.nodes[d].get("credits", 0) for d in descendants
        )
        blocking[node] = downstream_credits / total_credits

    return blocking


#articulation reach impact
def compute_articulation_reach(G):
    """Measures structural vulnerability as reachability loss rom entry nodes when a node is removed"""
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
        G_temp = G.copy()
        G_temp.remove_node(node)
        new_count = reachable_count(G_temp)
        impact[node] = (baseline - new_count) / baseline if baseline > 0 else 0

    return impact
