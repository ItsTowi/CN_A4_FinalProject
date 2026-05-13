"""
network_gen.py
==============
Network generation utilities for the "Listen what you won't" project.

Provides functions to create NetworkX graphs (Erdos-Renyi, Barabasi-Albert,
Watts-Strogatz) with opinion attributes already assigned to every node.
Opinion values are floats in [0, 1].

Typical usage from a notebook
------------------------------
    from network_gen import make_network

    G = make_network("BA", n=200, m=3, opinion="uniform", seed=42)
    # Every node now has G.nodes[i]["opinion"] in [0, 1]
"""

import random
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Low-level opinion initializers
# ---------------------------------------------------------------------------

def _assign_uniform(G, seed=None):
    """Assign opinions drawn uniformly from [0, 1]."""
    rng = np.random.default_rng(seed)
    for node in G.nodes():
        G.nodes[node]["opinion"] = float(rng.uniform(0.0, 1.0))


def _assign_bimodal(G, frac_low=0.5, mu_low=0.2, mu_high=0.8,
                    sigma=0.05, seed=None):
    """
    Assign opinions from a bimodal Gaussian mixture.

    Parameters
    ----------
    frac_low : float
        Fraction of nodes placed in the lower cluster (default 0.5).
    mu_low, mu_high : float
        Centres of the two clusters.
    sigma : float
        Std-dev of each Gaussian.
    seed : int or None
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n_low = int(len(nodes) * frac_low)
    indices = rng.permutation(len(nodes))
    for k, node in enumerate(nodes):
        if indices[k] < n_low:
            op = rng.normal(mu_low, sigma)
        else:
            op = rng.normal(mu_high, sigma)
        G.nodes[node]["opinion"] = float(np.clip(op, 0.0, 1.0))


def _assign_clustered(G, n_clusters=3, sigma=0.04, seed=None):
    """
    Assign opinions from *n_clusters* evenly-spaced Gaussian clusters.

    Parameters
    ----------
    n_clusters : int
        Number of opinion clusters (default 3).
    sigma : float
        Std-dev of each cluster.
    seed : int or None
    """
    rng = np.random.default_rng(seed)
    centres = np.linspace(0.1, 0.9, n_clusters)
    nodes = list(G.nodes())
    for node in nodes:
        c = centres[rng.integers(n_clusters)]
        G.nodes[node]["opinion"] = float(np.clip(rng.normal(c, sigma), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Network generators
# ---------------------------------------------------------------------------

def make_erdos_renyi(n=200, p=0.05, opinion="uniform",
                     opinion_kwargs=None, seed=None):
    """
    Erdos-Renyi G(n, p) random graph with opinion attributes.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability.
    opinion : str
        One of {"uniform", "bimodal", "clustered"}.
    opinion_kwargs : dict or None
        Extra keyword arguments forwarded to the opinion initializer.
    seed : int or None

    Returns
    -------
    G : nx.Graph
    """
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    _apply_opinion(G, opinion, opinion_kwargs, seed)
    G.graph["model"] = "ER"
    G.graph["params"] = {"n": n, "p": p}
    return G


def make_barabasi_albert(n=200, m=3, opinion="uniform",
                         opinion_kwargs=None, seed=None):
    """
    Barabasi-Albert preferential-attachment graph with opinion attributes.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Edges added per new node.
    opinion : str
        One of {"uniform", "bimodal", "clustered"}.
    opinion_kwargs : dict or None
    seed : int or None

    Returns
    -------
    G : nx.Graph
    """
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    _apply_opinion(G, opinion, opinion_kwargs, seed)
    G.graph["model"] = "BA"
    G.graph["params"] = {"n": n, "m": m}
    return G


def make_watts_strogatz(n=200, k=6, p=0.1, opinion="uniform",
                        opinion_kwargs=None, seed=None):
    """
    Watts-Strogatz small-world graph with opinion attributes.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is connected to k nearest neighbours in ring topology.
    p : float
        Probability of rewiring each edge.
    opinion : str
        One of {"uniform", "bimodal", "clustered"}.
    opinion_kwargs : dict or None
    seed : int or None

    Returns
    -------
    G : nx.Graph
    """
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    _apply_opinion(G, opinion, opinion_kwargs, seed)
    G.graph["model"] = "WS"
    G.graph["params"] = {"n": n, "k": k, "p": p}
    return G


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def make_network(model="BA", opinion="uniform", opinion_kwargs=None,
                 seed=None, **kwargs):
    """
    Unified factory function — the recommended way to create a network.

    Parameters
    ----------
    model : str
        Network model: "ER", "BA", or "WS" (case-insensitive).
    opinion : str
        Opinion distribution: "uniform", "bimodal", or "clustered".
    opinion_kwargs : dict or None
        Extra keyword arguments forwarded to the opinion initializer.
        Examples:
            opinion="bimodal",  opinion_kwargs={"frac_low": 0.4, "mu_low": 0.15}
            opinion="clustered", opinion_kwargs={"n_clusters": 4}
    seed : int or None
        Global random seed (passed to both nx generator and opinion assigner).
    **kwargs
        Model-specific parameters:
            ER  -> n, p
            BA  -> n, m
            WS  -> n, k, p

    Returns
    -------
    G : nx.Graph
        Graph with node attribute "opinion" in [0, 1] on every node.

    Examples
    --------
    >>> G = make_network("BA", n=300, m=4, opinion="bimodal", seed=0)
    >>> G = make_network("ER", n=200, p=0.04, opinion="uniform", seed=1)
    >>> G = make_network("WS", n=150, k=6, p=0.1, opinion="clustered",
    ...                  opinion_kwargs={"n_clusters": 3}, seed=2)
    """
    model = model.upper()
    builders = {
        "ER": make_erdos_renyi,
        "BA": make_barabasi_albert,
        "WS": make_watts_strogatz,
    }
    if model not in builders:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(builders)}.")
    return builders[model](opinion=opinion, opinion_kwargs=opinion_kwargs,
                           seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_opinion(G, opinion, opinion_kwargs, seed):
    """Dispatch to the right opinion initializer."""
    kw = opinion_kwargs or {}
    opinion = opinion.lower()
    if opinion == "uniform":
        _assign_uniform(G, seed=seed, **kw)
    elif opinion == "bimodal":
        _assign_bimodal(G, seed=seed, **kw)
    elif opinion == "clustered":
        _assign_clustered(G, seed=seed, **kw)
    else:
        raise ValueError(
            f"Unknown opinion distribution '{opinion}'. "
            "Choose from {'uniform', 'bimodal', 'clustered'}."
        )


def get_opinions(G):
    """Return a NumPy array of opinions in node order."""
    return np.array([G.nodes[n]["opinion"] for n in G.nodes()])


def set_opinions(G, opinions):
    """
    Write a NumPy array of opinion values back into the graph.

    Parameters
    ----------
    G : nx.Graph
    opinions : array-like, shape (n_nodes,)
    """
    for node, op in zip(G.nodes(), opinions):
        G.nodes[node]["opinion"] = float(op)
