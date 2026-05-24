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

def robust_polarization_index(opinions, tol=1e-3, min_size_pct=0.05):
    """
    Calcula el índice de polarización (varianza normalizada) basándose únicamente
    en los clústeres robustos significativos. Elimina el ruido de los nodos congelados.
    
    Devuelve un float entre 0 y 1.
    """
    n_nodes = len(opinions)
    min_nodes_required = max(1, int(n_nodes * min_size_pct))
    
    # 1. Identificar y etiquetar los clústeres (Misma lógica que n_robust_clusters)
    sorted_indices = np.argsort(opinions)
    sorted_op = opinions[sorted_indices]
    gaps = np.diff(sorted_op)
    
    cluster_ids = np.zeros(n_nodes, dtype=int)
    current_cluster = 0
    for i, gap in enumerate(gaps):
        if gap >= tol:
            current_cluster += 1
        cluster_ids[i+1] = current_cluster
        
    # 2. Contar el tamaño de cada clúster y calcular sus medias
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    
    # Filtrar cuáles son los clústeres que superan la masa crítica
    valid_clusters = unique_clusters[counts >= min_nodes_required]
    
    # Si la tolerancia aisló a todo el mundo (ej. epsilon=0) y ningún clúster es "grande",
    # devolvemos la varianza normalizada estándar para no romper la gráfica.
    if len(valid_clusters) == 0:
        return float(np.var(opinions) / 0.25)
        
    # 3. Construir el nuevo vector de opiniones "limpio"
    # Solo meteremos las opiniones promediadas de los clústeres que sí son robustos
    robust_opinions = []
    
    for c_id in valid_clusters:
        # Encontramos los nodos que pertenecen a este clúster específico
        nodos_del_cluster = sorted_op[cluster_ids == c_id]
        # Calculamos la opinión media exacta de este bando
        cluster_mean = np.mean(nodos_del_cluster)
        # Añadimos tantos puntos como nodos tenga el clúster para mantener el peso estadístico
        robust_opinions.extend([cluster_mean] * len(nodos_del_cluster))
        
    # 4. Calcular la varianza de este nuevo sistema limpio y normalizar a [0, 1]
    robust_var = np.var(robust_opinions)
    
    return float(robust_var / 0.25)

def n_robust_clusters(opinions, tol=1e-3, min_size_pct=0.05):
    """
    Cuenta el número de clústeres significativos eliminando nodos disidentes aislados.
    Si el sistema está masivamente fragmentado (como en epsilon cercano a 0), 
    devuelve el número total de microclústeres reales.
    """
    n_nodes = len(opinions)
    min_nodes_required = max(1, int(n_nodes * min_size_pct))
    
    # 1. Agrupar las opiniones que estén muy juntas
    sorted_op = np.sort(opinions)
    gaps = np.diff(sorted_op)
    cluster_ids = np.zeros(n_nodes, dtype=int)
    
    current_cluster = 0
    for i, gap in enumerate(gaps):
        if gap >= tol:
            current_cluster += 1
        cluster_ids[i+1] = current_cluster
        
    # 2. Contar cuántos nodos tiene cada clúster
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    
    # 3. FILTRADO INTELIGENTE
    robust_clusters = 0
    for count in counts:
        if count >= min_nodes_required:
            robust_clusters += 1
            
    # SI NO HAY NINGÚN CLÚSTER GRANDE: Significa que la sociedad está atomizada 
    # (fragmentación total). Devolvemos el número real de microclústeres detectados.
    if robust_clusters == 0:
        return len(unique_clusters)
        
    return robust_clusters

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
