"""Deux graphes UPFD côte à côte (Real / Fake) : taille et couleur = degré pour voir les hubs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PathCollection
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD

from data_loader import LABEL_NAMES, _patch_upfd_google_drive_ids, default_data_root
from feature_engineering import pyg_to_networkx_undirected


def project_root() -> Path:
    return Path(__file__).resolve().parent


def pick_one_graph_per_class(
    dataset: UPFD,
    label_real: int = 0,
    label_fake: int = 1,
) -> Tuple[Optional[Data], Optional[Data]]:
    """Premier graphe Real et premier Fake trouvés dans l’ordre du dataset."""
    real_data: Optional[Data] = None
    fake_data: Optional[Data] = None
    for i in range(len(dataset)):
        data = dataset[i]
        y = int(data.y.view(-1)[0].item())
        if y == label_real and real_data is None:
            real_data = data
        elif y == label_fake and fake_data is None:
            fake_data = data
        if real_data is not None and fake_data is not None:
            break
    return real_data, fake_data


def layout_positions(G: nx.Graph, seed: int = 42) -> Dict[int, np.ndarray]:
    """Layout spring pour l’affichage."""
    return nx.spring_layout(G, seed=seed, k=2 / np.sqrt(max(G.number_of_nodes(), 1)))


def degree_to_size_and_color(
    G: nx.Graph,
    deg_global_max: float,
    size_min: float = 120.0,
    size_max: float = 2200.0,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Tailles et degrés pour le dessin (même échelle max sur les deux graphes)."""
    nodes = list(G.nodes())
    degs = np.array([G.degree(n) for n in nodes], dtype=np.float64)
    denom = max(float(deg_global_max), 1e-6)
    alpha = degs / denom
    sizes = size_min + (size_max - size_min) * alpha
    return nodes, degs, sizes


def draw_graph_panel(
    ax: plt.Axes,
    G: nx.Graph,
    title: str,
    subtitle: str,
    pos: Dict[int, np.ndarray],
    deg_global_max: float,
    cmap_name: str = "magma",
) -> PathCollection:
    """Un panneau matplotlib (graphe + colorbar degré)."""
    _, _, sizes = degree_to_size_and_color(G, deg_global_max)
    node_list = list(G.nodes())
    degs_nodes = np.array([G.degree(n) for n in node_list])

    norm = mcolors.Normalize(vmin=0.0, vmax=float(deg_global_max))
    cmap = plt.get_cmap(cmap_name)
    node_colors = cmap(norm(degs_nodes))

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        width=1.0,
        alpha=0.55,
        arrows=False,
    )
    coll = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=node_list,
        node_size=sizes,
        node_color=node_colors,
        linewidths=0.6,
        edgecolors="#1e293b",
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Degré du nœud", rotation=270, labelpad=18)
    return coll


def build_comparison_figure(
    G_real: nx.Graph,
    G_fake: nx.Graph,
    out_path: Path,
    show: bool = True,
) -> None:
    """Figure 1×2 → fichier + plt.show si possible."""
    deg_max_real = max(dict(G_real.degree()).values()) if G_real.number_of_nodes() else 0
    deg_max_fake = max(dict(G_fake.degree()).values()) if G_fake.number_of_nodes() else 0
    deg_global_max = float(max(deg_max_real, deg_max_fake, 1))

    pos_real = layout_positions(G_real, seed=41)
    pos_fake = layout_positions(G_fake, seed=43)

    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
            "font.family": "sans-serif",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), constrained_layout=True)
    fig.suptitle(
        "Propagation d’information sur Twitter (UPFD — GossipCop)\n"
        "Taille et couleur proportionnelles au degré (hubs plus grands, plus visibles)",
        fontsize=15,
        fontweight="semibold",
        color="#0f172a",
    )

    n_r, m_r = G_real.number_of_nodes(), G_real.number_of_edges()
    n_f, m_f = G_fake.number_of_nodes(), G_fake.number_of_edges()

    draw_graph_panel(
        axes[0],
        G_real,
        "Organique — Real",
        f"{n_r} nœuds, {m_r} arêtes (non orientées) · Δ_max = {deg_max_real}",
        pos_real,
        deg_global_max,
        cmap_name="YlGnBu",
    )
    draw_graph_panel(
        axes[1],
        G_fake,
        "Désinformation — Fake",
        f"{n_f} nœuds, {m_f} arêtes (non orientées) · Δ_max = {deg_max_fake}",
        pos_fake,
        deg_global_max,
        cmap_name="YlOrRd",
    )

    fig.text(
        0.5,
        0.02,
        "Échelle commune : le degré maximal pris sur les deux graphes permet de comparer "
        "visuellement l’intensité des hubs (centralités locales).",
        ha="center",
        fontsize=10,
        color="#475569",
        style="italic",
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure enregistrée : {out_path.resolve()}")

    if show:
        backend = plt.get_backend().lower()
        if "agg" in backend:
            print(
                "Backend Matplotlib non interactif : ouvrez comparaison_graphes.png "
                "pour voir la figure (ou lancez sans MPLBACKEND=Agg pour plt.show())."
            )
        else:
            try:
                plt.show()
            except Exception as exc:  # pragma: no cover — environnement sans display
                print(
                    f"Affichage interactif indisponible ({type(exc).__name__}) : "
                    "figure sauvegardée uniquement."
                )

    plt.close(fig)


def main() -> None:
    root = default_data_root()
    _patch_upfd_google_drive_ids()

    test_ds = UPFD(str(root), "gossipcop", "bert", split="test")
    real_data, fake_data = pick_one_graph_per_class(test_ds)
    if real_data is None or fake_data is None:
        raise RuntimeError(
            "Impossible de trouver un graphe Real et un graphe Fake dans le split test."
        )

    G_real = pyg_to_networkx_undirected(real_data)
    G_fake = pyg_to_networkx_undirected(fake_data)

    out_png = project_root() / "comparaison_graphes.png"
    build_comparison_figure(G_real, G_fake, out_png, show=True)

    print("\nRésumé des exemples choisis :")
    print(f"  Real : label={int(real_data.y)} ({LABEL_NAMES.get(0, '')})")
    print(f"  Fake : label={int(fake_data.y)} ({LABEL_NAMES.get(1, '')})")


if __name__ == "__main__":
    main()
