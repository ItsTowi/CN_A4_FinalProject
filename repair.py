import abc
import numpy as np
import networkx as nx

from network_gen import get_opinions, set_opinions
from hk_dynamics  import hk_step
from simulation   import compute_metrics


class BaseRepairStrategy(abc.ABC):
    @abc.abstractmethod
    def step(self, G: nx.Graph, epsilon: float, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

class MediatorStrategy(BaseRepairStrategy):
    def __init__(self,
                 n_mediators      : int   = 5,
                 sigma_mode       : str   = "uniform",
                 sigma_value      : float = 0.1,
                 k_per_community  : int   = 3):
        self.n_mediators     = n_mediators
        self.sigma_mode      = sigma_mode
        self.sigma_value     = sigma_value
        self.k_per_community = k_per_community
        self._initialized    = False

    def step(self, G: nx.Graph, epsilon: float, rng: np.random.Generator) -> None:
        if not self._initialized:
            self._add_mediators(G, rng)
            self._assign_sigma(G)
            self._initialized = True

    def _add_mediators(self, G: nx.Graph, rng: np.random.Generator) -> None:
        # Detectar comunidades
        try:
            from community import best_partition as _louvain
            partition = _louvain(G)
        except Exception:
            partition = {}
            for cid, comp in enumerate(nx.connected_components(G)):
                for n in comp:
                    partition[n] = cid

        # Hub de cada comunidad = nodo de mayor grado dentro de ella
        communities: dict = {}
        for node, cid in partition.items():
            communities.setdefault(cid, []).append(node)

        hub_list = [
            max(members, key=lambda n: G.degree(n))
            for members in communities.values()
        ]

        # Para cada comunidad, elegir k_per_community nodos de contacto:
        contact_nodes = []
        for members in communities.values():
            sorted_by_degree = sorted(members, key=lambda n: G.degree(n), reverse=True)
            k = min(self.k_per_community, len(sorted_by_degree))
            contact_nodes.append(sorted_by_degree[:k])

        # Añadir mediadores y conectarlos a los nodos de contacto de cada comunidad
        next_id = max(G.nodes()) + 1
        for k in range(self.n_mediators):
            m = next_id + k
            G.add_node(m, opinion=0.5, is_mediator=True, sigma=0.0)
            for community_contacts in contact_nodes:
                for node in community_contacts:
                    G.add_edge(m, node)


    def _assign_sigma(self, G: nx.Graph) -> None:
        """Asigna el índice de convencimiento σ_i a cada nodo normal."""
        normal = [n for n in G.nodes()
                  if not G.nodes[n].get("is_mediator", False)]

        if self.sigma_mode == "uniform":
            for n in normal:
                G.nodes[n]["sigma"] = self.sigma_value

        elif self.sigma_mode == "inv_degree":
            degrees = {n: G.degree(n) for n in normal}
            d_max   = max(degrees.values()) if degrees else 1
            for n in normal:
                G.nodes[n]["sigma"] = 1.0 - degrees[n] / d_max

        else:
            raise ValueError(
                f"sigma_mode desconocido: '{self.sigma_mode}'. "
                "Usa 'uniform' o 'inv_degree'."
            )


def _hk_step_repair(G: nx.Graph, epsilon: float) -> float:
    nodes  = list(G.nodes())
    old_op = {n: G.nodes[n]["opinion"] for n in nodes}
    delta  = 0.0

    new_opinions = {}
    for i in nodes:
        if G.nodes[i].get("is_mediator", False):
            new_opinions[i] = 0.5
            continue

        xi    = old_op[i]
        sigma = G.nodes[i].get("sigma", 1.0)

        candidates = [i] + list(G.neighbors(i))
        in_bounds  = [old_op[j] for j in candidates
                      if abs(old_op[j] - xi) <= epsilon
                      or G.nodes[j].get("is_mediator", False)]
        hk_avg = np.mean(in_bounds) if in_bounds else xi

        x_new          = (1 - sigma) * xi + sigma * hk_avg
        new_opinions[i] = x_new
        delta           = max(delta, abs(x_new - xi))

    for n, val in new_opinions.items():
        G.nodes[n]["opinion"] = val

    return delta


def run_repair_phase(
    G,
    strategy,
    epsilon             = 0.15,
    repair_steps        = 60,
    hk_steps_per_repair = 5,
    tol                 = 1e-6,
    seed                = None,
):
    rng          = np.random.default_rng(seed)
    history      = [get_opinions(G).copy()]
    metrics_list = [compute_metrics(G)]

    for _ in range(repair_steps):
        # (a) Intervención: añade mediadores (solo 1ª vez) y asigna sigma
        strategy.step(G, epsilon, rng)

        # (b) HK modificado: mediadores fijos, sigma-ponderado para el resto
        for _ in range(hk_steps_per_repair):
            delta = _hk_step_repair(G, epsilon)
            if delta < tol:
                break

        history.append(get_opinions(G).copy())
        metrics_list.append(compute_metrics(G))

    return {
        "name"    : "repair",
        "strategy": str(strategy),
        "history" : history,
        "metrics" : metrics_list,
        "steps"   : repair_steps,
        "G"       : G,
    }