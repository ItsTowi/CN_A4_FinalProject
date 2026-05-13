"""
hk_dynamics.py
==============
Hegselmann-Krause (HK) bounded-confidence opinion dynamics.

Each agent i updates its opinion as the average of all neighbours j
(including i itself) whose opinion is within epsilon of i's current opinion.

    x_i(t+1) = mean{ x_j(t) : j in N(i) union {i}, |x_i(t) - x_j(t)| <= eps }

Convergence is declared when the maximum opinion change in a step is below `tol`.

Typical usage from a notebook
------------------------------
    from network_gen import make_network
    from hk_dynamics import hk_run

    G = make_network("BA", n=200, m=3, opinion="uniform", seed=42)
    opinions_history, converged, n_steps = hk_run(G, epsilon=0.3)
    # G.nodes[i]["opinion"] now holds the final (converged) opinions
    # opinions_history[t] is the array of opinions at step t
"""

import numpy as np
import networkx as nx
from network_gen import get_opinions, set_opinions


# ---------------------------------------------------------------------------
# Core HK step
# ---------------------------------------------------------------------------

def hk_step(G, epsilon):
    """
    Apply one synchronous HK update to graph G.

    Each node i computes the average opinion of its closed neighbourhood
    (itself + graph neighbours) restricted to those within epsilon.
    If no neighbour (including self) is within epsilon, the opinion is
    unchanged (the agent is already isolated).

    Parameters
    ----------
    G : nx.Graph
        Graph with "opinion" attribute on every node.
    epsilon : float
        Bounded-confidence threshold in (0, 1].

    Returns
    -------
    new_opinions : np.ndarray, shape (n,)
        Updated opinion for each node (in node-iteration order).
    delta : float
        Maximum absolute opinion change (useful for convergence check).
    """
    nodes = list(G.nodes())
    old_op = {n: G.nodes[n]["opinion"] for n in nodes}
    new_op = np.empty(len(nodes))

    for k, i in enumerate(nodes):
        xi = old_op[i]
        # Closed neighbourhood: node itself + graph neighbours
        candidates = [i] + list(G.neighbors(i))
        in_bounds = [old_op[j] for j in candidates
                     if abs(old_op[j] - xi) <= epsilon]
        new_op[k] = np.mean(in_bounds) if in_bounds else xi

    delta = max(abs(new_op[k] - old_op[nodes[k]]) for k in range(len(nodes)))
    return new_op, delta


# ---------------------------------------------------------------------------
# Full run (no history — fast)
# ---------------------------------------------------------------------------

def hk_run(G, epsilon, max_steps=500, tol=1e-6):
    """
    Run HK dynamics until convergence or max_steps, updating G in place.

    Parameters
    ----------
    G : nx.Graph
        Graph with "opinion" attribute on every node (modified in place).
    epsilon : float
        Bounded-confidence threshold.
    max_steps : int
        Maximum number of update steps.
    tol : float
        Convergence tolerance: stop when max opinion change < tol.

    Returns
    -------
    converged : bool
        True if convergence criterion was met before max_steps.
    n_steps : int
        Number of steps executed.

    Example
    -------
    >>> converged, steps = hk_run(G, epsilon=0.3)
    >>> print(f"Converged: {converged} in {steps} steps")
    """
    for step in range(max_steps):
        new_op, delta = hk_step(G, epsilon)
        set_opinions(G, new_op)
        if delta < tol:
            return True, step + 1
    return False, max_steps


# ---------------------------------------------------------------------------
# Full run WITH history (slower but needed for visualization)
# ---------------------------------------------------------------------------

def hk_run_history(G, epsilon, max_steps=500, tol=1e-6, record_every=1):
    """
    Run HK dynamics while recording opinion snapshots at each step.

    Parameters
    ----------
    G : nx.Graph
        Graph with "opinion" attribute on every node (modified in place).
    epsilon : float
        Bounded-confidence threshold.
    max_steps : int
        Maximum number of update steps.
    tol : float
        Convergence tolerance.
    record_every : int
        Record a snapshot every this many steps (default 1 = every step).

    Returns
    -------
    history : list of np.ndarray
        history[t] is an array of shape (n_nodes,) with opinions at step t.
        history[0] is the initial state.
    converged : bool
    n_steps : int

    Example
    -------
    >>> history, converged, steps = hk_run_history(G, epsilon=0.3)
    >>> # opinions at step 10:
    >>> opinions_t10 = history[10]
    """
    history = [get_opinions(G).copy()]

    for step in range(max_steps):
        new_op, delta = hk_step(G, epsilon)
        set_opinions(G, new_op)

        if (step + 1) % record_every == 0:
            history.append(new_op.copy())

        if delta < tol:
            # Make sure the final state is always in history
            if len(history) == 0 or not np.array_equal(history[-1], new_op):
                history.append(new_op.copy())
            return history, True, step + 1

    return history, False, max_steps


# ---------------------------------------------------------------------------
# Convenience metrics derived from an opinion array
# ---------------------------------------------------------------------------

def opinion_variance(opinions):
    """Global variance of the opinion distribution."""
    return float(np.var(opinions))


def n_clusters(opinions, tol=1e-3):
    """
    Estimate the number of opinion clusters at convergence.

    Clusters are contiguous groups of agents whose opinions differ by < tol.
    Works best on a *converged* opinion array.

    Parameters
    ----------
    opinions : array-like
    tol : float
        Two opinions are in the same cluster if |x_i - x_j| < tol.

    Returns
    -------
    int
    """
    sorted_op = np.sort(opinions)
    gaps = np.diff(sorted_op)
    return int(np.sum(gaps >= tol)) + 1


def polarization_index(opinions):
    """
    Simple polarization index: variance of opinions normalised to [0, 1].

    Returns 0 for full consensus, 1 for maximum polarization (all agents at
    0 or 1 with equal split).

    Returns
    -------
    float in [0, 1]
    """
    # Maximum possible variance for opinions in [0,1] is 0.25 (half at 0, half at 1)
    return float(np.var(opinions) / 0.25)
