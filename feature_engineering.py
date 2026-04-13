"""Features topologiques globales (NetworkX) → DataFrame pour la baseline Random Forest.

Graphes en non orienté (comme souvent avec ToUndirected sur UPFD). Les arbres UPFD
ont souvent un clustering moyen à 0 ; densité / degré max restent utiles."""

from __future__ import annotations

import argparse
from typing import Any, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.utils import to_networkx

from data_loader import LABEL_NAMES, default_data_root, load_upfd_splits


def pyg_to_networkx_undirected(data: Data) -> nx.Graph:
    """PyG Data → NetworkX non orienté (self-loops retirées)."""
    return to_networkx(
        data,
        to_undirected=True,
        remove_self_loops=True,
    )


def graph_topology_metrics(G: nx.Graph) -> dict[str, float]:
    """Densité, clustering moyen, assortativité degré, degré max. NaN si assortativité indéfinie."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = float(nx.density(G))
    clustering_mean = float(nx.average_clustering(G))

    assortativity_degree: float
    if m == 0 or n < 2:
        assortativity_degree = float(np.nan)
    else:
        try:
            assortativity_degree = float(nx.degree_assortativity_coefficient(G))
        except (nx.NetworkXError, ZeroDivisionError):
            assortativity_degree = float(np.nan)

    if n == 0:
        max_degree = float(np.nan)
    else:
        max_degree = float(max(dict(G.degree()).values()))

    return {
        "density": density,
        "clustering_mean": clustering_mean,
        "assortativity_degree": assortativity_degree,
        "max_degree": max_degree,
    }


def _gather_indexed_graphs(
    datasets: Sequence[Tuple[str, UPFD]],
) -> List[Tuple[str, int, Data]]:
    """Liste (nom_split, index, Data) pour tout le corpus chargé."""
    out: List[Tuple[str, int, Data]] = []
    for split_name, ds in datasets:
        for idx in range(len(ds)):
            out.append((split_name, idx, ds[idx]))
    return out


def stratified_subsample(
    items: Sequence[Tuple[str, int, Data]],
    n_samples: int,
    random_state: int,
) -> List[Tuple[str, int, Data]]:
    """Sous-échantillon stratifié sur y (équilibre des classes)."""
    if n_samples <= 0:
        return []

    rng = np.random.default_rng(random_state)
    labels = np.array([int(t[2].y) for t in items], dtype=np.int64)
    classes = np.unique(labels)
    n_classes = len(classes)

    base = n_samples // n_classes
    remainder = n_samples % n_classes

    selected: List[Tuple[str, int, Data]] = []
    for k, c in enumerate(classes):
        pool_idx = np.nonzero(labels == c)[0]
        n_take = base + (1 if k < remainder else 0)
        n_take = min(n_take, int(pool_idx.size))
        if n_take == 0:
            continue
        pick = rng.choice(pool_idx, size=n_take, replace=False)
        for j in pick:
            selected.append(items[int(j)])

    rng.shuffle(selected)
    return selected


def build_topology_dataframe(
    train: UPFD,
    val: UPFD,
    test: UPFD,
    n_samples: int = 800,
    random_state: int = 42,
) -> pd.DataFrame:
    """DataFrame : split, graph_index, 4 métriques, label, label_name."""
    all_items = _gather_indexed_graphs(
        (("train", train), ("val", val), ("test", test))
    )
    if n_samples >= len(all_items):
        chosen = list(all_items)
    else:
        chosen = stratified_subsample(all_items, n_samples, random_state)

    rows: List[dict[str, Any]] = []
    for split_name, local_idx, data in chosen:
        G = pyg_to_networkx_undirected(data)
        metrics = graph_topology_metrics(G)
        label = int(data.y.view(-1)[0].item())
        rows.append(
            {
                "split": split_name,
                "graph_index": local_idx,
                **metrics,
                "label": label,
                "label_name": LABEL_NAMES.get(label, f"class_{label}"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction de features topologiques UPFD → DataFrame."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=800,
        help="Taille du sous-échantillon stratifié (par défaut : 800).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine pseudo-aléatoire.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Si non vide, chemin CSV de sortie (ex. data/topology_features.csv).",
    )
    args = parser.parse_args()

    root = default_data_root()
    train, val, test = load_upfd_splits(root, name="gossipcop", feature="bert")

    df = build_topology_dataframe(
        train,
        val,
        test,
        n_samples=args.n_samples,
        random_state=args.seed,
    )

    print(f"DataFrame : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print("\nAperçu :")
    print(df.head().to_string())
    print("\nStatistiques descriptives (colonnes numériques) :")
    print(df.describe().to_string())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nFichier écrit : {args.output}")


if __name__ == "__main__":
    main()
