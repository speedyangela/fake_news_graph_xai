"""Chargement UPFD via PyG (graphes de propagation Twitter, tâche classification de graphes)."""

from __future__ import annotations

import os.path as osp
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import UPFD


def _patch_upfd_google_drive_ids() -> None:
    # Anciennes versions de PyG ont des liens Drive morts (404) ; IDs à jour côté pyg-team.
    UPFD.file_ids.update(
        {
            "politifact": "1toou2GO0agoY_OS54LaCWEECQfe93nuq",
            "gossipcop": "1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g",
        }
    )

# UPFD brut : 0 = real, 1 = fake (souvent). PyG peut ré-encoder en 0..C-1.
LABEL_NAMES: Dict[int, str] = {
    0: "Real (organique / vérifié)",
    1: "Fake (fausse information)",
}


def _project_root() -> Path:
    """Répertoire du dépôt (dossier parent de ce fichier)."""
    return Path(__file__).resolve().parent


def default_data_root() -> Path:
    """Racine des données : <projet>/data/UPFD."""
    return _project_root() / "data" / "UPFD"


@dataclass
class SplitStats:
    """Stats par split UPFD."""

    name: str
    num_graphs: int
    mean_num_nodes: float
    class_counts: Dict[int, int]

    def __str__(self) -> str:
        lines = [
            f"  [{self.name}]",
            f"    Graphes : {self.num_graphs}",
            f"    Nombre moyen de nœuds par graphe : {self.mean_num_nodes:.2f}",
            "    Répartition des classes (indice PyG → effectif) :",
        ]
        for cls, cnt in sorted(self.class_counts.items()):
            name = LABEL_NAMES.get(cls, f"classe {cls}")
            lines.append(f"      {cls} ({name}) : {cnt}")
        return "\n".join(lines)


def _count_classes(labels: torch.Tensor) -> Dict[int, int]:
    """Effectifs par classe."""
    labels_np = labels.view(-1).numpy()
    unique, counts = np.unique(labels_np, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def _mean_num_nodes(dataset: Dataset) -> float:
    """Nombre moyen de nœuds par graphe."""
    if len(dataset) == 0:
        return float("nan")
    total = sum(data.num_nodes for data in dataset)
    return float(total) / len(dataset)


def load_upfd_splits(
    root: Path | str,
    name: str = "gossipcop",
    feature: str = "bert",
) -> Tuple[UPFD, UPFD, UPFD]:
    """Train / val / test UPFD. ``root`` : ex. data/UPFD ; ``name`` politifact|gossipcop ; ``feature`` profile|spacy|bert|content."""
    root_str = str(root)
    _patch_upfd_google_drive_ids()
    train = UPFD(root_str, name, feature, split="train")
    val = UPFD(root_str, name, feature, split="val")
    test = UPFD(root_str, name, feature, split="test")
    return train, val, test


def raw_label_semantics(train_ds: UPFD) -> str:
    """Résumé des valeurs dans graph_labels.npy si le fichier brut existe."""
    raw_path = osp.join(train_ds.raw_dir, "graph_labels.npy")
    if not osp.isfile(raw_path):
        return (
            "(Fichier brut non encore présent — lancez le script une première "
            "fois pour télécharger et traiter les données.)"
        )
    raw = np.load(raw_path)
    uniq = np.unique(raw)
    parts = [f"Valeurs distinctes dans graph_labels.npy : {uniq.tolist()}"]
    parts.append(
        "Après traitement PyG, la classe c correspond à la c-ième valeur "
        "triée parmi ces entiers (encodage contigu 0 … C-1)."
    )
    return "\n    ".join(parts)


def compute_split_statistics(train: UPFD, val: UPFD, test: UPFD) -> List[SplitStats]:
    """Stats par split."""
    out: List[SplitStats] = []
    for split_name, ds in ("train", train), ("val", val), ("test", test):
        labels = torch.stack([ds[i].y for i in range(len(ds))], dim=0)
        out.append(
            SplitStats(
                name=split_name,
                num_graphs=len(ds),
                mean_num_nodes=_mean_num_nodes(ds),
                class_counts=_count_classes(labels),
            )
        )
    return out


def print_global_report(split_stats: List[SplitStats], train_ds: UPFD) -> None:
    """Affiche récap global + détail par split."""
    total_graphs = sum(s.num_graphs for s in split_stats)
    weighted_nodes = sum(s.mean_num_nodes * s.num_graphs for s in split_stats)
    mean_nodes_global = weighted_nodes / total_graphs if total_graphs else float("nan")

    merged_counts: Dict[int, int] = {}
    for s in split_stats:
        for cls, cnt in s.class_counts.items():
            merged_counts[cls] = merged_counts.get(cls, 0) + cnt

    print("=" * 72)
    print("UPFD — rapport d'ingestion (PyTorch Geometric)")
    print("=" * 72)
    print(f"Nombre total de graphes (train + val + test) : {total_graphs}")
    print(
        "Nombre moyen de nœuds par graphe (moyenne globale pondérée) : "
        f"{mean_nodes_global:.2f}"
    )
    print("\nRépartition globale des classes (Fake vs Real, indices PyG) :")
    for cls in sorted(merged_counts):
        cnt = merged_counts[cls]
        pct = 100.0 * cnt / total_graphs if total_graphs else 0.0
        label = LABEL_NAMES.get(cls, f"classe {cls}")
        print(f"  • {label}")
        print(f"      effectif = {cnt}  ({pct:.2f} % du total)")

    print("\nLecture du fichier brut d'étiquettes (si disponible) :")
    print(f"    {raw_label_semantics(train_ds)}")

    print("\nDétail par split :")
    for s in split_stats:
        print(s)
    print("=" * 72)


MANUAL_UPFD_HINT = """
Si le téléchargement automatique échoue (quota Drive, pare-feu, etc.) :
  1. Télécharger l'archive UPFD depuis le dossier partagé du projet GNN-FakeNews
     (lien « Google Drive » dans leur README).
  2. Décompresser le contenu du zip dans :
 <racine>/gossipcop/raw/
     de sorte que ``graph_labels.npy``, ``A.txt``, ``new_bert_feature.npz``, etc.
     soient directement présents dans ``raw``.
  3. Relancer ce script : le pré-traitement PyG construira ``processed/bert/``.
"""


def main() -> None:
    root = default_data_root()
    root.mkdir(parents=True, exist_ok=True)

    print(
        f"Chargement UPFD (gossipcop, features bert) depuis :\n  {root}\n"
        "Premier lancement : téléchargement et pré-traitement peuvent prendre "
        "plusieurs minutes et nécessiter ~1,2 Go d'espace disque.\n"
    )

    try:
        train, val, test = load_upfd_splits(root, name="gossipcop", feature="bert")
    except urllib.error.HTTPError as exc:
        print(f"Échec du téléchargement HTTP ({exc.code}) : {exc.reason}", file=sys.stderr)
        print(MANUAL_UPFD_HINT, file=sys.stderr)
        raise SystemExit(1) from exc

    stats = compute_split_statistics(train, val, test)
    print_global_report(stats, train)

    sample = train[0]
    print("\nExemple de premier graphe (split train) :")
    print(f"  x.shape       = {tuple(sample.x.shape)}  (nœuds × dimension BERT)")
    print(f"  edge_index    = {tuple(sample.edge_index.shape)}  (2 × |E|)")
    print(f"  y (étiquette) = {int(sample.y)}")


if __name__ == "__main__":
    main()
