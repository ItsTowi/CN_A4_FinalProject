# Listen what you won't — Code Documentation

**Complex Networks · MESIIA**  
Philippe Lemoine · Pol Pujol · Antonio Iglesias

---

## Overview

This project simulates how social-media-style homophilic rewiring drives a
connected network into echo chambers, and then tests strategies to reverse
that fragmentation. The code is split into **four Python modules** (easy to
import in any Jupyter notebook) and one or more experiment notebooks.

```
CN_A4_FinalProject/
├── network_gen.py      # Network + opinion generation
├── hk_dynamics.py      # Hegselmann-Krause opinion model
├── simulation.py       # Full parametrizable pipeline
├── visualization.py    # All plotting utilities
└── README.md           # This file
```

---

## Installation

```bash
pip install networkx numpy scipy matplotlib seaborn python-louvain
```

> `python-louvain` provides community detection and modularity.  
> If it is not installed, modularity metrics are silently skipped.

---

## Module 1 — `network_gen.py`

### Purpose
Creates NetworkX graphs of three types (Erdős–Rényi, Barabási–Albert,
Watts–Strogatz) and assigns each node an `opinion` attribute in `[0, 1]`.

### Main function: `make_network`

```python
from network_gen import make_network

G = make_network(
    model          = "BA",        # "ER" | "BA" | "WS"
    n              = 300,         # number of nodes
    m              = 3,           # BA only: edges per new node
    opinion        = "uniform",   # "uniform" | "bimodal" | "clustered"
    opinion_kwargs = None,        # extra args to the opinion initializer
    seed           = 42,
)
```

Every node `i` now has `G.nodes[i]["opinion"]` ∈ [0, 1].

### Opinion distributions

| Name | Description | Extra kwargs |
|------|-------------|--------------|
| `"uniform"` | Uniform draw from [0, 1] | — |
| `"bimodal"` | Two Gaussian clusters (polarized start) | `frac_low`, `mu_low`, `mu_high`, `sigma` |
| `"clustered"` | N evenly-spaced Gaussian clusters | `n_clusters`, `sigma` |

**Examples:**

```python
# Uniformly distributed opinions on a BA graph
G = make_network("BA", n=200, m=3, opinion="uniform", seed=0)

# Two opinion camps (already polarized initial state)
G = make_network("ER", n=300, p=0.04,
                 opinion="bimodal",
                 opinion_kwargs={"frac_low": 0.45, "mu_low": 0.15, "mu_high": 0.85},
                 seed=1)

# Three clusters on a small-world graph
G = make_network("WS", n=200, k=6, p=0.1,
                 opinion="clustered",
                 opinion_kwargs={"n_clusters": 3},
                 seed=2)
```

### Utility functions

```python
from network_gen import get_opinions, set_opinions

opinions = get_opinions(G)       # np.ndarray of shape (n,)
set_opinions(G, new_opinions)    # write array back into graph
```

---

## Module 2 — `hk_dynamics.py`

### Purpose
Implements the **Hegselmann-Krause (HK) bounded-confidence** opinion model.

### The HK rule

At each discrete time step, every agent `i` updates its opinion by averaging
over all its network neighbours (including itself) whose opinion is within
`epsilon` of its own:

```
x_i(t+1) = mean{ x_j(t) : j ∈ N(i) ∪ {i},  |x_i(t) − x_j(t)| ≤ ε }
```

If no such neighbour exists, `x_i` stays unchanged.

### Key parameter: `epsilon` (ε)

| ε value | Expected behaviour |
|---------|--------------------|
| ε < 0.2 | Strong fragmentation — many isolated clusters |
| ε ≈ 0.3 | Moderate — 2–3 clusters typical |
| ε > 0.5 | Convergence to consensus |

### Functions

#### `hk_step(G, epsilon)` — single update
```python
from hk_dynamics import hk_step

new_opinions, delta = hk_step(G, epsilon=0.3)
# new_opinions: np.ndarray with the updated values
# delta: max absolute opinion change (use for convergence check)
```

#### `hk_run(G, epsilon)` — run to convergence (fast, no history)
```python
from hk_dynamics import hk_run

converged, n_steps = hk_run(G, epsilon=0.3, max_steps=500, tol=1e-6)
# G is modified in place
```

#### `hk_run_history(G, epsilon)` — run to convergence with snapshots
```python
from hk_dynamics import hk_run_history

history, converged, n_steps = hk_run_history(G, epsilon=0.3)
# history: list of np.ndarray — opinions at every step
# history[0] = initial state, history[-1] = converged state
```

#### Opinion metrics
```python
from hk_dynamics import opinion_variance, n_clusters, polarization_index

opinions = get_opinions(G)

opinion_variance(opinions)   # np.var — 0 means consensus
n_clusters(opinions)         # number of distinct opinion groups after convergence
polarization_index(opinions) # variance / 0.25 — normalized to [0, 1]
```

---

## Module 3 — `simulation.py`

### Purpose
Orchestrates the full experiment as three sequential phases, recording
opinions and network metrics at every step.

### The three phases

```
Phase 1 — Burn-in
    Run HK until convergence on the original graph.
    Agents form initial opinion clusters.

Phase 2 — Rewiring
    Alternate homophilic rewiring with short HK runs.
    Discordant edges are removed and replaced with concordant ones.
    Echo chambers form; modularity rises; opinion variance grows.

Phase 3 — Repair (optional)
    Apply a repair strategy, then run HK again.
    Goal: bring the network back towards consensus.
```

### Main function: `run_simulation`

```python
from network_gen import make_network
from simulation import run_simulation

G = make_network("BA", n=300, m=3, opinion="uniform", seed=0)

result = run_simulation(
    G,

    # HK threshold — most important parameter
    epsilon             = 0.25,

    # Phase 2 — rewiring
    rewire              = True,
    p_rewire            = 0.05,      # prob. of rewiring each discordant edge
    rewire_steps        = 80,        # number of rewiring rounds
    hk_steps_per_rewire = 5,         # HK steps after each rewiring round

    # Phase 3 — repair (set to None to skip)
    repair              = "mediators",
    repair_kwargs       = {"n_mediators": 20},

    # Convergence
    max_hk_steps        = 500,
    tol                 = 1e-6,

    seed                = 0,
)
```

> **Important:** `run_simulation` always works on a **deep copy** of `G`.
> Your original graph is never modified.

### Return value

`result` is a dict:

| Key | Type | Contents |
|-----|------|----------|
| `"G"` | `nx.Graph` | Final state of the graph |
| `"phases"` | `list[dict]` | One dict per phase (history, metrics, convergence info) |
| `"snapshots"` | `list[(str, np.ndarray)]` | Opinions at key moments — ready for `plot_snapshots` |
| `"metrics"` | `dict[str, list]` | Time-series of every metric across all steps |

**Accessing snapshots:**
```python
for label, opinions in result["snapshots"]:
    print(label, opinions.mean())
# initial, after_burnin, after_rewiring, after_repair
```

**Accessing phase histories:**
```python
burnin_history  = result["phases"][0]["history"]   # list of np.ndarray
rewiring_metrics = result["phases"][1]["metrics"]  # list of metric dicts
```

### Repair strategies

| Strategy | Description | kwargs |
|----------|-------------|--------|
| `"mediators"` | Add neutral nodes (opinion ≈ 0.5) that bridge communities | `n_mediators` (int), `opinion` (float) |
| `"forced_links"` | Force N random cross-community edges | `n_links` (int) |
| `"reduce_epsilon"` | Gradually widen ε from current to `epsilon_end` | `epsilon_end` (float), `steps` (int) |

```python
# Example — forced cross-community links
result = run_simulation(G, epsilon=0.25, rewire=True,
                        repair="forced_links",
                        repair_kwargs={"n_links": 30})

# Example — gradual epsilon relaxation
result = run_simulation(G, epsilon=0.25, rewire=True,
                        repair="reduce_epsilon",
                        repair_kwargs={"epsilon_end": 0.55, "steps": 10})
```

### Metrics recorded at every step

| Metric | Description |
|--------|-------------|
| `opinion_variance` | Global variance of opinion values |
| `polarization_index` | Variance normalised to [0, 1] |
| `n_opinion_clusters` | Number of distinct opinion clusters |
| `modularity` | Louvain modularity of graph partition |
| `n_communities` | Number of Louvain communities |
| `mean_degree` | Average node degree |

---

## Module 4 — `visualization.py`

### Purpose
All plotting. Functions return `pos` (node positions) so you can reuse
the same layout across multiple plots for visual consistency.

### `plot_network_opinion` — colour by opinion
```python
from visualization import plot_network_opinion

pos = plot_network_opinion(
    G,
    opinions  = None,    # if None, reads from G.nodes[i]["opinion"]
    title     = "After burn-in",
    layout    = "spring",  # "spring" | "kamada_kawai" | "circular" | "spectral"
    pos       = None,      # reuse a previous pos dict for consistent layout
    node_size = 40,
    cmap      = "RdYlGn_r",
)
# Returns pos — pass to next call to keep nodes in same positions
```

### `plot_network_community` — colour by community
```python
from visualization import plot_network_community

# Requires community attribute set by compute_metrics / run_simulation
pos = plot_network_community(G, title="Echo chambers after rewiring")
```

### `plot_snapshots` — grid of snapshots over time
```python
from visualization import plot_snapshots

plot_snapshots(
    result["G"],
    result["snapshots"],
    layout    = "spring",
    node_size = 35,
    color_by  = "both",   # "opinion" | "community" | "both"
)
```
This is the **main visualization call** — it shows you the full evolution
in one figure: initial → burn-in → rewiring → repair.

### `plot_metrics` — time series of network metrics
```python
from visualization import plot_metrics

plot_metrics(result)
# Shows: opinion variance, polarization, modularity, # communities
# Draws vertical lines at phase boundaries (burn-in / rewiring / repair)
```

### `plot_opinion_histogram` — opinion distributions
```python
from visualization import plot_opinion_histogram

plot_opinion_histogram({
    "Initial"        : result["snapshots"][0][1],
    "After rewiring" : result["snapshots"][2][1],
    "After repair"   : result["snapshots"][3][1],
})
```

---

## Putting it all together — minimal notebook example

```python
import matplotlib.pyplot as plt
from network_gen   import make_network
from simulation    import run_simulation
from visualization import plot_snapshots, plot_metrics, plot_opinion_histogram

# 1. Generate network
G = make_network("BA", n=300, m=3, opinion="uniform", seed=0)

# 2. Run full experiment
result = run_simulation(
    G,
    epsilon      = 0.25,
    rewire       = True,
    p_rewire     = 0.05,
    rewire_steps = 80,
    repair       = "mediators",
    repair_kwargs = {"n_mediators": 20},
    seed         = 0,
)

# 3. Visualise
plot_snapshots(result["G"], result["snapshots"], color_by="both")
plot_metrics(result)
plot_opinion_histogram({
    label: op for label, op in result["snapshots"]
})
```

---

## Parameter reference — cheat sheet

```
epsilon       ∈ (0, 1]   HK confidence threshold      (lower → more polarization)
p_rewire      ∈ [0, 1]   Prob. of rewiring per edge   (higher → faster echo chambers)
rewire_steps  ∈ N        Rewiring rounds               (more → stronger fragmentation)
n_mediators   ∈ N        Mediator nodes to add         (more → stronger bridge effect)
n_links       ∈ N        Forced cross-community edges  (more → stronger bridge effect)
```

---

## Suggested experiment structure

1. **Baseline polarization** — vary `epsilon` from 0.1 to 0.5, no rewiring.
   Observe how HK alone creates clusters depending on ε.

2. **Echo chamber formation** — fix `epsilon=0.25`, vary `p_rewire` and
   `rewire_steps`. Track modularity and opinion variance over time.

3. **Repair comparison** — apply all three repair strategies under identical
   rewiring conditions. Compare final modularity and polarization index.

4. **Network topology effect** — run the same experiment on ER, BA, and WS
   graphs. Does the network model affect how fast echo chambers form, or how
   recoverable they are?
