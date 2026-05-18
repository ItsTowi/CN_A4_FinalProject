"""
simulation.py
=============
Parametrizable simulation pipeline for "Listen what you won't".

Two phases in a single call to `run_simulation`:
  1. BURN-IN   — HK dynamics on the original network until convergence.
                 Agents form opinion clusters determined by epsilon.
  2. REWIRING  — Homophilic rewiring interleaved with HK steps.
                 Discordant edges are cut and replaced with concordant ones,
                 driving the network into isolated echo chambers.

All phases record opinion snapshots and key metrics at every step,
making the result easy to plot from a notebook.

Typical usage
-------------
    from network_gen import make_network
    from simulation import run_simulation

    G = make_network("BA", n=300, m=3, opinion="uniform", seed=0)

    result = run_simulation(
        G,
        epsilon             = 0.25,
        rewire              = True,
        p_rewire            = 0.05,
        rewire_steps        = 80,
        hk_steps_per_rewire = 5,
        seed                = 0,
    )

    # result["snapshots"] -> list of (label, opinion_array) tuples
    # result["metrics"]   -> dict of metric time-series
    # result["G"]         -> final state of the graph
"""

import copy
import numpy as np
import networkx as nx

from network_gen import get_opinions, set_opinions
from hk_dynamics  import hk_step, hk_run_history, opinion_variance, n_clusters, polarization_index

try:
    from community import best_partition as _louvain_partition
    from community import modularity     as _louvain_modularity
    _LOUVAIN = True
except ImportError:
    _LOUVAIN = False


# ---------------------------------------------------------------------------
# Metric snapshot
# ---------------------------------------------------------------------------

def compute_metrics(G):
    """
    Compute a snapshot of network + opinion metrics at the current state.

    Returns
    -------
    dict with keys:
        opinion_variance, polarization_index, n_opinion_clusters,
        modularity, n_communities, mean_degree
    """
    opinions = get_opinions(G)
    metrics = {
        "opinion_variance"   : opinion_variance(opinions),
        "polarization_index" : polarization_index(opinions),
        "n_opinion_clusters" : n_clusters(opinions),
        "mean_degree"        : float(np.mean([d for _, d in G.degree()])),
    }

    if _LOUVAIN and G.number_of_edges() > 0:
        partition = _louvain_partition(G)
        metrics["modularity"]    = _louvain_modularity(partition, G)
        metrics["n_communities"] = len(set(partition.values()))
        # Store partition on graph so visualization can use it
        nx.set_node_attributes(G, partition, "community")
    else:
        metrics["modularity"]    = float("nan")
        metrics["n_communities"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Phase 1 — Burn-in: plain HK until convergence
# ---------------------------------------------------------------------------

def _phase_burnin(G, epsilon, max_steps, tol, record_every):
    """
    Run HK dynamics until convergence and record opinion history.

    After this phase, opinions have settled into clusters — the number and
    position of clusters depends on epsilon and the network topology.
    """
    history, converged, steps = hk_run_history(
        G, epsilon, max_steps=max_steps, tol=tol, record_every=record_every
    )
    # Compute metrics for each recorded snapshot
    metrics_list = []
    for op_snap in history:
        set_opinions(G, op_snap)
        metrics_list.append(compute_metrics(G))
    # Restore final state
    set_opinions(G, history[-1])

    return {
        "name"      : "burn_in",
        "history"   : history,
        "metrics"   : metrics_list,
        "converged" : converged,
        "steps"     : steps,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Homophilic rewiring interleaved with HK
# ---------------------------------------------------------------------------

def _homophilic_rewire_step(G, epsilon, p_rewire, rng):
    """
    One round of homophilic rewiring on graph G (modified in place).

    For every edge (i, j) where |opinion_i − opinion_j| > epsilon:
      - with probability p_rewire, remove the edge
      - reconnect i to a random node k that is NOT yet a neighbour
        and whose opinion IS within epsilon of i's opinion

    This mechanism mimics social-media algorithms that promote content from
    like-minded accounts: over time, cross-opinion links disappear and the
    network fragments into opinion-homogeneous communities (echo chambers).

    Parameters
    ----------
    G        : nx.Graph  (modified in place)
    epsilon  : float     bounded-confidence threshold (same as HK)
    p_rewire : float     probability of rewiring each discordant edge per round
    rng      : np.random.Generator
    """
    opinions      = {n: G.nodes[n]["opinion"] for n in G.nodes()}
    all_nodes     = list(G.nodes())
    edges_to_remove = []
    edges_to_add    = []

    for i, j in list(G.edges()):
        if abs(opinions[i] - opinions[j]) > epsilon:
            if rng.random() < p_rewire:
                edges_to_remove.append((i, j))
                # Find a compatible replacement neighbour for i
                compatible = [
                    k for k in all_nodes
                    if k != i and k != j
                    and not G.has_edge(i, k)
                    and abs(opinions[i] - opinions[k]) <= epsilon
                ]
                if compatible:
                    k_new = compatible[rng.integers(len(compatible))]
                    edges_to_add.append((i, k_new))

    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)


def _phase_rewiring(G, epsilon, p_rewire, rewire_steps,
                    hk_steps_per_rewire, tol, record_every, rng):
    """
    Alternate homophilic rewiring and short HK runs for `rewire_steps` rounds.

    Each round:
      (a) rewire discordant edges homophilically
      (b) run HK for hk_steps_per_rewire steps (partial convergence)

    The combination accelerates echo-chamber formation: rewiring makes
    neighbourhoods more homogeneous, which in turn allows HK to push
    opinions even closer together.
    """
    history      = [get_opinions(G).copy()]
    metrics_list = [compute_metrics(G)]

    for _ in range(rewire_steps):
        # (a) structural change
        _homophilic_rewire_step(G, epsilon, p_rewire, rng)
        # (b) opinion update
        for _ in range(hk_steps_per_rewire):
            new_op, delta = hk_step(G, epsilon)
            set_opinions(G, new_op)
            if delta < tol:
                break

        history.append(get_opinions(G).copy())
        metrics_list.append(compute_metrics(G))

    return {
        "name"    : "rewiring",
        "history" : history,
        "metrics" : metrics_list,
        "steps"   : rewire_steps,
    }


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_simulation(
    G,
    epsilon             = 0.25,
    # --- Rewiring phase ---
    rewire              = True,
    p_rewire            = 0.05,
    rewire_steps        = 60,
    hk_steps_per_rewire = 5,
    # --- Repair / depolarization phase ---
    repair              = False,
    repair_strategy     = None,
    repair_steps        = 60,
    hk_steps_per_repair = 5,
    # --- HK convergence ---
    max_hk_steps        = 500,
    tol                 = 1e-6,
    record_every        = 1,
    # --- Misc ---
    seed                = None,
):
    """
    Full two-phase simulation: HK burn-in → homophilic rewiring.

    Parameters
    ----------
    G : nx.Graph
        Input graph with "opinion" attribute on every node.
        The original graph is NOT modified (deep copy used internally).

    epsilon : float
        Bounded-confidence threshold for HK (0 < epsilon <= 1).
        Controls how "open-minded" agents are.
        Lower  → more clusters / more polarization.
        Higher → consensus.

    rewire : bool
        If True, run the homophilic rewiring phase after burn-in (default True).
        Set to False to observe pure HK dynamics without structural change.

    p_rewire : float
        Probability of rewiring each discordant edge per rewiring step.
        Higher → echo chambers form faster.

    rewire_steps : int
        Number of rewiring rounds.
        Higher → stronger final fragmentation.

    hk_steps_per_rewire : int
        HK steps applied after each rewiring round (default 5).

    repair : bool
        If True, run the depolarization phase after rewiring (default False).
        Requires ``repair_strategy`` to be set.

    repair_strategy : BaseRepairStrategy or None
        An instance of a repair strategy (subclass of BaseRepairStrategy).
        Only used when ``repair=True``.

        Example:
            from repair import MediatorStrategy
            strategy = MediatorStrategy(n_mediators=5)
            result = run_simulation(G, ..., repair=True,
                                    repair_strategy=strategy)

    repair_steps : int
        Number of repair rounds (default 60).

    hk_steps_per_repair : int
        HK steps applied after each repair round (default 5).

    max_hk_steps : int
        Maximum HK steps in the burn-in phase.

    tol : float
        HK convergence tolerance (stop when max opinion change < tol).

    record_every : int
        Record a snapshot every N steps (default 1 = every step).
        Increase to reduce memory usage for large networks.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        {
          "G"        : nx.Graph,           # final graph state
          "phases"   : list[dict],         # one dict per phase
          "snapshots": list[(str, array)], # opinions at key moments
          "metrics"  : dict[str, list],    # time-series of every metric
        }

    Examples
    --------
    Pure HK (no rewiring):
    >>> result = run_simulation(G, epsilon=0.3, rewire=False)

    HK + homophilic rewiring:
    >>> result = run_simulation(G, epsilon=0.25, rewire=True,
    ...                         p_rewire=0.05, rewire_steps=80)

    Full pipeline (burn-in + rewiring + repair):
    >>> from repair import MediatorStrategy
    >>> result = run_simulation(
    ...     G, epsilon=0.15, rewire=True, p_rewire=0.15, rewire_steps=120,
    ...     repair=True, repair_strategy=MediatorStrategy(n_mediators=5),
    ...     repair_steps=60,
    ... )

    Exploring epsilon:
    >>> for eps in [0.1, 0.2, 0.3, 0.4]:
    ...     result = run_simulation(G, epsilon=eps, rewire=False)
    ...     final_var = result["metrics"]["opinion_variance"][-1]
    ...     print(f"eps={eps}  variance={final_var:.4f}")
    """
    rng    = np.random.default_rng(seed)
    G      = copy.deepcopy(G)   # never modify the caller's graph
    phases = []

    # ---- Phase 1: Burn-in HK ---------------------------------------------
    phase1 = _phase_burnin(G, epsilon, max_hk_steps, tol, record_every)
    phases.append(phase1)

    # ---- Phase 2: Rewiring + HK ------------------------------------------
    if rewire:
        phase2 = _phase_rewiring(
            G, epsilon, p_rewire, rewire_steps,
            hk_steps_per_rewire, tol, record_every, rng
        )
        phases.append(phase2)

    # ---- Phase 3: Repair / depolarization --------------------------------
    if repair:
        if repair_strategy is None:
            raise ValueError(
                "repair=True requires a repair_strategy. "
                "Pass an instance of a BaseRepairStrategy subclass.\n"
                "Example:\n"
                "    from repair import MediatorStrategy\n"
                "    repair_strategy=MediatorStrategy(n_mediators=5)"
            )
        # Lazy import to avoid circular dependency
        from repair import run_repair_phase as _run_repair_phase
        phase3 = _run_repair_phase(
            G                   = G,
            strategy            = repair_strategy,
            epsilon             = epsilon,
            repair_steps        = repair_steps,
            hk_steps_per_repair = hk_steps_per_repair,
            tol                 = tol,
            seed                = seed,
        )
        phases.append(phase3)

    # ---- Aggregate all metrics into flat time-series ---------------------
    all_metrics = {}
    for phase in phases:
        for m_snap in phase["metrics"]:
            for key, val in m_snap.items():
                all_metrics.setdefault(key, []).append(val)

    # ---- Named snapshots for easy plotting -------------------------------
    snapshots = [("initial", phases[0]["history"][0].copy())]
    snapshots.append(("after_burnin", phases[0]["history"][-1].copy()))
    if rewire and len(phases) >= 2:
        snapshots.append(("after_rewiring", phases[1]["history"][-1].copy()))
    if repair:
        # repair phase is always the last one
        snapshots.append(("after_repair", phases[-1]["history"][-1].copy()))

    return {
        "G"        : G,
        "phases"   : phases,
        "snapshots": snapshots,
        "metrics"  : all_metrics,
    }
