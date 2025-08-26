# src/graph_analysis.py
from __future__ import annotations
import argparse
import os
import json
from typing import Dict, List, Tuple, Any
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms import community as nx_community

# ---------- Helpers ----------
def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if a.max() == a.min():
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min())

# ---------- Graph builders ----------
def build_synthetic_network(kind="barabasi", n=200, m=2, p=0.02, seed=42) -> nx.Graph:
    """
    Build a synthetic network. Attach asset_value and security_level attributes.
    Roles: 'workstation', 'server', 'router', 'firewall'
    """
    if kind == "erdos":
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    else:
        # default: Barabasi-Albert (scale-free), often useful to mimic networks
        G = nx.barabasi_albert_graph(n, m, seed=seed)

    rng = np.random.default_rng(seed)
    roles = ["workstation", "server", "router", "firewall"]
    for node in G.nodes():
        # assign role by probability (fewer servers, fewer firewalls)
        r = rng.choice(roles, p=[0.7, 0.15, 0.1, 0.05])
        # asset_value: servers higher, workstations low
        value = {"workstation": 1, "server": 10, "router": 5, "firewall": 7}[r]
        # security level: 1-10 higher means better defended
        sec = int(rng.integers(3, 9)) if r == "server" else int(rng.integers(1, 8))
        G.nodes[node]["role"] = r
        G.nodes[node]["asset_value"] = value
        G.nodes[node]["security_level"] = sec

    # default edge weight (defense cost) maybe derived from node security levels
    for u, v in G.edges():
        # make defense cost slightly higher if either node is firewall/router
        base = 1.0
        if G.nodes[u]["role"] in ("firewall", "router") or G.nodes[v]["role"] in ("firewall", "router"):
            base = 2.0
        G.edges[u, v]["defense_cost"] = base + rng.random()  # add small noise
    return G

# ---------- Metrics ----------
def compute_centralities(G: nx.Graph) -> Dict[str, Dict[int, float]]:
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=200)
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}
    return {"degree": deg, "betweenness": bet, "closeness": clo, "eigenvector": eig}

def detect_communities(G: nx.Graph) -> Dict[int, int]:
    # greedy modularity communities from networkx
    communities = nx_community.greedy_modularity_communities(G)
    membership = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[node] = cid
    return membership

def compute_vulnerability_scores(G: nx.Graph, centralities: Dict[str, Dict[int, float]],
                                 weights: Dict[str, float] = None) -> Dict[int, float]:
    """
    Combine normalized centrality measures + asset_value + security level + articulation point flag into a score.
    Higher score = more vulnerable/attractive to attackers (subject to chosen interpretation).
    """
    if weights is None:
        weights = {"betweenness": 0.35, "closeness": 0.15, "degree": 0.2, "eigenvector": 0.1,
                   "asset_value": 0.1, "security_weakness": 0.08, "articulation": 0.02}

    nodes = list(G.nodes())
    bet = np.array([centralities["betweenness"].get(n, 0.0) for n in nodes])
    clo = np.array([centralities["closeness"].get(n, 0.0) for n in nodes])
    deg = np.array([centralities["degree"].get(n, 0.0) for n in nodes])
    eig = np.array([centralities["eigenvector"].get(n, 0.0) for n in nodes])
    asset = np.array([G.nodes[n].get("asset_value", 1) for n in nodes])
    sec = np.array([G.nodes[n].get("security_level", 5) for n in nodes])

    # normalize
    n_bet = minmax_norm(bet)
    n_clo = minmax_norm(clo)
    n_deg = minmax_norm(deg)
    n_eig = minmax_norm(eig)
    n_asset = minmax_norm(asset)
    n_secweak = 1.0 - minmax_norm(sec)  # high security -> less weakness

    art_points = set(nx.articulation_points(G))
    art_flag = np.array([1.0 if n in art_points else 0.0 for n in nodes])

    score = (weights["betweenness"] * n_bet +
             weights["closeness"] * n_clo +
             weights["degree"] * n_deg +
             weights["eigenvector"] * n_eig +
             weights["asset_value"] * n_asset +
             weights["security_weakness"] * n_secweak +
             weights["articulation"] * art_flag)

    return {n: float(s) for n, s in zip(nodes, score)}

# ---------- Attack & robustness simulations ----------
def simulate_node_removals(G: nx.Graph, remove_order: List[int]) -> pd.DataFrame:
    """
    Remove nodes in sequence and record largest connected component size after each removal.
    remove_order: list of nodes to remove (targeted or random)
    """
    G = G.copy()
    records = []
    for i, node in enumerate(remove_order):
        if node in G:
            G.remove_node(node)
        # largest connected component size (by nodes)
        if len(G) == 0:
            lcc = 0
        else:
            lcc = len(max(nx.connected_components(G), key=len))
        records.append({"step": i + 1, "removed_node": node, "remaining_nodes": G.number_of_nodes(), "largest_cc": lcc})
    return pd.DataFrame(records)

def targeted_vs_random_attack(G: nx.Graph, vuln_scores: Dict[int, float], top_k: int = None) -> Dict[str, pd.DataFrame]:
    nodes_sorted = sorted(vuln_scores.items(), key=lambda x: x[1], reverse=True)
    targeted_order = [n for n, _ in nodes_sorted]
    rng = np.random.default_rng(123)
    random_order = rng.permutation(list(G.nodes())).tolist()

    data_targeted = simulate_node_removals(G, targeted_order if top_k is None else targeted_order[:top_k])
    data_random = simulate_node_removals(G, random_order if top_k is None else random_order[:top_k])
    return {"targeted": data_targeted, "random": data_random}

# ---------- Attack path analysis ----------
def find_attack_paths(G: nx.Graph, attackers: List[int], targets: List[int], weight_attr="defense_cost", k: int = 3) -> Dict[Tuple[int,int], List[Any]]:
    """
    For each attacker-target pair, find up to k shortest paths (by sum of edge weight_attr)
    """
    paths = {}
    # ensure graph is weighted; if missing, default to 1
    for a in attackers:
        for t in targets:
            if a not in G or t not in G:
                paths[(a, t)] = []
                continue
            try:
                path = nx.shortest_path(G, source=a, target=t, weight=weight_attr)
                paths[(a, t)] = [path]
            except nx.NetworkXNoPath:
                paths[(a, t)] = []
    return paths

# ---------- Reporting ----------
def export_node_table(G: nx.Graph, centralities: Dict[str, Dict[int, float]], communities: Dict[int, int], vuln: Dict[int, float], out_csv="reports/node_metrics.csv"):
    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "role": G.nodes[n].get("role"),
            "asset_value": G.nodes[n].get("asset_value"),
            "security_level": G.nodes[n].get("security_level"),
            "community": communities.get(n, -1),
            "degree": centralities["degree"].get(n, 0.0),
            "betweenness": centralities["betweenness"].get(n, 0.0),
            "closeness": centralities["closeness"].get(n, 0.0),
            "eigenvector": centralities["eigenvector"].get(n, 0.0),
            "vulnerability_score": vuln.get(n, 0.0)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def plot_largest_cc(df_targeted: pd.DataFrame, df_random: pd.DataFrame, out="figures/robustness.png"):
    plt.figure(figsize=(8,5))
    plt.plot(df_targeted["step"], df_targeted["largest_cc"], label="targeted")
    plt.plot(df_random["step"], df_random["largest_cc"], label="random")
    plt.xlabel("removal step")
    plt.ylabel("largest connected component size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# ---------- Demo / CLI ----------
def demo_workflow():
    ensure_dirs()
    G = build_synthetic_network(kind="barabasi", n=300, m=3)
    centralities = compute_centralities(G)
    communities = detect_communities(G)
    vuln = compute_vulnerability_scores(G, centralities)
    df_nodes = export_node_table(G, centralities, communities, vuln)
    sim = targeted_vs_random_attack(G, vuln, top_k=100)
    plot_largest_cc(sim["targeted"], sim["random"])
    # also export example paths from top attacker node(s) to top asset servers
    top_attackers = sorted(vuln, key=vuln.get, reverse=True)[:3]
    # choose high value targets (servers)
    targets = [n for n, d in G.nodes(data=True) if d.get("role") == "server"][:5]
    paths = find_attack_paths(G, top_attackers, targets)
    with open("reports/example_paths.json", "w") as f:
        json.dump({f"{a}->{t}": p for (a,t), p in paths.items()}, f, indent=2)
    print("Demo done. Outputs in reports/ and figures/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run demo workflow")
    args = parser.parse_args()
    if args.demo:
        demo_workflow()

if __name__ == "__main__":
    main()
