"""
visualization.py
================
Plotting utilities for "Listen what you won't".

All functions accept an optional `ax` (or `axes`) argument so they can be
embedded inside notebook subplots. When called without `ax` they create
their own figure and call plt.show().

Key functions
-------------
plot_network_opinion(G, ...)
    Draw the graph coloured by each node's continuous opinion value.

plot_network_community(G, ...)
    Draw the graph coloured by community membership (requires python-louvain).

plot_snapshots(G, snapshots, ...)
    Grid of network plots — one panel per (label, opinion_array) snapshot
    from run_simulation(). Ideal for showing the time evolution.

plot_metrics(result, ...)
    Multi-panel line chart of opinion variance, polarization index,
    modularity and number of communities over all simulation steps.

plot_opinion_histogram(opinions_or_history, ...)
    Histogram (or overlaid histograms) of opinion distributions.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

try:
    import community as community_louvain
    _LOUVAIN = True
except ImportError:
    _LOUVAIN = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OPINION_CMAP = "RdYlGn_r"   # red=0 (extreme left), green=1 (extreme right)
_COMMUNITY_PALETTES = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def _get_layout(G, pos, layout):
    """Return node positions dict."""
    if pos is not None:
        return pos
    layouts = {
        "spring"    : nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular"  : nx.circular_layout,
        "spectral"  : nx.spectral_layout,
    }
    fn = layouts.get(layout, nx.spring_layout)
    try:
        return fn(G, seed=42)
    except TypeError:
        return fn(G)


def _opinion_colors(opinions, cmap=_OPINION_CMAP):
    """Map opinion values in [0,1] to RGBA colours."""
    mapper = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap)
    return [mapper.to_rgba(op) for op in opinions]


def _community_colors(G):
    """Map community IDs to distinct colours."""
    comm_attr = nx.get_node_attributes(G, "community")
    if not comm_attr:
        return None, None
    nodes = list(G.nodes())
    comm_ids = [comm_attr.get(n, 0) for n in nodes]
    unique   = sorted(set(comm_ids))
    palette  = _COMMUNITY_PALETTES[:len(unique)]
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique)}
    colors = [color_map[cid] for cid in comm_ids]
    return colors, color_map


# ---------------------------------------------------------------------------
# 1. Network coloured by opinion
# ---------------------------------------------------------------------------

def plot_network_opinion(G, opinions=None, title="Opinion distribution",
                         layout="spring", pos=None, node_size=40,
                         cmap=_OPINION_CMAP, ax=None, figsize=(7, 6)):
    """
    Draw the graph with nodes coloured by their opinion value.

    Parameters
    ----------
    G        : nx.Graph
    opinions : array-like or None
        If None, reads "opinion" attribute from G.
        If provided, temporarily overrides the stored opinions for plotting.
    title    : str
    layout   : str
        One of {"spring", "kamada_kawai", "circular", "spectral"}.
    pos      : dict or None
        Pre-computed position dict. Overrides `layout` if given.
    node_size : int
    cmap     : str  matplotlib colormap name.
    ax       : matplotlib.axes.Axes or None
    figsize  : tuple

    Returns
    -------
    pos : dict
        Node positions (reuse across calls for consistent layout).

    Example
    -------
    >>> pos = plot_network_opinion(G, title="After burn-in")
    >>> plot_network_opinion(G_rewired, pos=pos, title="After rewiring")
    """
    if opinions is None:
        opinions = np.array([G.nodes[n]["opinion"] for n in G.nodes()])
    else:
        opinions = np.asarray(opinions)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    pos = _get_layout(G, pos, layout)
    colors = _opinion_colors(opinions, cmap=cmap)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#aaaaaa",
                           width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=node_size, linewidths=0.3,
                           edgecolors="#555555")
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # Colorbar
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Opinion", fontsize=10)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return pos


# ---------------------------------------------------------------------------
# 2. Network coloured by community
# ---------------------------------------------------------------------------

def plot_network_community(G, title="Community structure",
                            layout="spring", pos=None, node_size=40,
                            ax=None, figsize=(7, 6)):
    """
    Draw the graph with nodes coloured by community membership.

    Requires that `community` node attribute is set (done automatically by
    `compute_metrics` in simulation.py, which calls python-louvain).

    Parameters
    ----------
    G        : nx.Graph
    title    : str
    layout   : str
    pos      : dict or None
    node_size : int
    ax       : matplotlib.axes.Axes or None
    figsize  : tuple

    Returns
    -------
    pos : dict

    Example
    -------
    >>> pos = plot_network_community(G_rewired, title="Echo chambers")
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    pos = _get_layout(G, pos, layout)
    colors, color_map = _community_colors(G)

    if colors is None:
        # Fallback: colour by opinion
        ax.set_title(f"{title} (no community data — showing opinion)", fontsize=13)
        return plot_network_opinion(G, title=title, pos=pos,
                                    node_size=node_size, ax=ax)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#aaaaaa",
                           width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=node_size, linewidths=0.3,
                           edgecolors="#555555")
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # Legend
    patches = [
        mpl.patches.Patch(color=col, label=f"Community {cid}")
        for cid, col in color_map.items()
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.7, ncol=2)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return pos


# ---------------------------------------------------------------------------
# 3. Snapshot grid — opinion + community side by side at multiple time points
# ---------------------------------------------------------------------------

def plot_snapshots(G, snapshots, layout="spring", node_size=35,
                   color_by="opinion", figsize=None):
    """
    Plot a grid of network snapshots at key moments in the simulation.

    Parameters
    ----------
    G         : nx.Graph
        Final graph (used for structure; opinions are swapped per snapshot).
    snapshots : list of (str, np.ndarray)
        As returned by run_simulation()["snapshots"]:
        [(label, opinion_array), ...]
    layout    : str
    node_size : int
    color_by  : str
        "opinion" | "community" | "both"
        "both" shows opinion on top row, community on bottom row.
    figsize   : tuple or None
        Auto-computed if None.

    Example
    -------
    >>> result = run_simulation(G, epsilon=0.25, rewire=True)
    >>> plot_snapshots(result["G"], result["snapshots"])
    """
    import copy
    n_snap = len(snapshots)
    if n_snap == 0:
        print("No snapshots to plot.")
        return

    n_rows = 2 if color_by == "both" else 1
    if figsize is None:
        figsize = (4.5 * n_snap, 4.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_snap, figsize=figsize,
                              squeeze=False)
    fig.suptitle("Opinion dynamics — snapshots", fontsize=14, y=1.01)

    # Compute layout once using first snapshot
    G_tmp = copy.deepcopy(G)
    set_first = snapshots[0][1]
    for node, op in zip(G_tmp.nodes(), set_first):
        G_tmp.nodes[node]["opinion"] = float(op)
    pos = _get_layout(G_tmp, None, layout)

    for col, (label, op_array) in enumerate(snapshots):
        G_snap = copy.deepcopy(G)
        for node, op in zip(G_snap.nodes(), op_array):
            G_snap.nodes[node]["opinion"] = float(op)

        # Top row (or only row) — opinion
        if color_by in ("opinion", "both"):
            ax = axes[0][col]
            plot_network_opinion(G_snap, opinions=op_array,
                                  title=label.replace("_", " ").title(),
                                  pos=pos, node_size=node_size, ax=ax)

        # Bottom row — community
        if color_by == "both":
            ax = axes[1][col]
            plot_network_community(G_snap,
                                    title=f"{label} (community)",
                                    pos=pos, node_size=node_size, ax=ax)
        elif color_by == "community":
            ax = axes[0][col]
            plot_network_community(G_snap,
                                    title=label.replace("_", " ").title(),
                                    pos=pos, node_size=node_size, ax=ax)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Metrics over time
# ---------------------------------------------------------------------------

def plot_metrics(result, figsize=(13, 9)):
    """
    Multi-panel line chart of key metrics across all simulation steps.

    Reads result["metrics"] as produced by run_simulation().
    Draws vertical dashed lines at phase boundaries.

    Parameters
    ----------
    result  : dict  (output of run_simulation)
    figsize : tuple

    Example
    -------
    >>> result = run_simulation(G, epsilon=0.25, rewire=True, repair="mediators")
    >>> plot_metrics(result)
    """
    metrics = result["metrics"]
    phases  = result["phases"]

    keys_to_plot = [
        ("opinion_variance",   "Opinion Variance",     "#2166ac"),
        ("polarization_index", "Polarization Index",   "#d6604d"),
        ("modularity",         "Modularity",           "#1a9641"),
        ("n_communities",      "# Communities",        "#7b3294"),
    ]
    keys_to_plot = [(k, lbl, c) for k, lbl, c in keys_to_plot
                    if k in metrics and any(not np.isnan(v)
                                            for v in metrics[k])]

    n_plots = len(keys_to_plot)
    if n_plots == 0:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=False)
    if n_plots == 1:
        axes = [axes]

    # Compute phase boundary positions
    boundaries = []
    cumulative = 0
    for phase in phases[:-1]:
        cumulative += len(phase["history"])
        boundaries.append(cumulative)

    for ax, (key, label, color) in zip(axes, keys_to_plot):
        vals = metrics[key]
        steps = list(range(len(vals)))
        ax.plot(steps, vals, color=color, linewidth=2)
        ax.set_xlabel("Simulation step", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Phase boundaries
        phase_labels = [p["name"].replace("_", " ").title() for p in phases]
        colors_phase = ["#3182bd", "#e6550d", "#31a354"]
        prev = 0
        for i, (b, pl, pc) in enumerate(zip(
                boundaries + [len(steps)],
                phase_labels,
                colors_phase
        )):
            ax.axvspan(prev, b, alpha=0.06, color=pc, label=pl)
            ax.axvline(x=b, color=pc, linestyle="--", linewidth=1, alpha=0.7)
            prev = b

    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Simulation metrics over time", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Opinion histogram
# ---------------------------------------------------------------------------

def plot_opinion_histogram(opinions_dict, bins=30, figsize=(10, 4),
                            title="Opinion distribution"):
    """
    Overlaid histograms of opinion distributions.

    Parameters
    ----------
    opinions_dict : dict {label: array-like}
        Each key is a legend label and value is an opinion array.
        Example: {"initial": op0, "after rewiring": op1}
    bins    : int
    figsize : tuple
    title   : str

    Example
    -------
    >>> plot_opinion_histogram({
    ...     "Initial"        : result["snapshots"][0][1],
    ...     "After rewiring" : result["snapshots"][2][1],
    ... })
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, opinions) in enumerate(opinions_dict.items()):
        ax.hist(np.asarray(opinions), bins=bins, range=(0, 1),
                alpha=0.55, color=colors[i % len(colors)],
                label=label, density=True, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Opinion value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Helper — used in plot_snapshots to set opinions without importing simulation
# ---------------------------------------------------------------------------

def _tmp_set_opinions(G, opinions):
    for node, op in zip(G.nodes(), opinions):
        G.nodes[node]["opinion"] = float(op)
