"""Baseline : Random Forest sur les 4 features topologiques uniquement (pas de BERT dans X)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import LABEL_NAMES, default_data_root, load_upfd_splits
from feature_engineering import build_topology_dataframe

TOPOLOGY_FEATURES: List[str] = [
    "density",
    "clustering_mean",
    "assortativity_degree",
    "max_degree",
]


def load_features_dataframe(
    csv_path: Path | None,
    n_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """CSV fourni, sinon recalcule via feature_engineering."""
    if csv_path is not None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
        return pd.read_csv(csv_path)

    root = default_data_root()
    train, val, test = load_upfd_splits(root, name="gossipcop", feature="bert")
    return build_topology_dataframe(
        train,
        val,
        test,
        n_samples=n_samples,
        random_state=random_state,
    )


def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """X = colonnes topo, y = label."""
    missing = [c for c in TOPOLOGY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")
    if "label" not in df.columns:
        raise ValueError("Colonne cible 'label' absente du DataFrame.")
    X = df[TOPOLOGY_FEATURES].copy()
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


def make_baseline_pipeline(random_state: int) -> Pipeline:
    """Médiane sur les NaN (assortativité) puis RandomForest."""
    imputer = SimpleImputer(strategy="median")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("imputer", imputer), ("rf", clf)])


def print_confusion_matrix_pretty(cm: np.ndarray) -> None:
    """Matrice de confusion (lignes = vérité, colonnes = prédit)."""
    row_labels = [f"Vrai {i} ({LABEL_NAMES.get(i, '')})" for i in range(cm.shape[0])]
    col_labels = [f"Préd {j}" for j in range(cm.shape[1])]
    print("\nMatrice de confusion (lignes = vérité terrain, colonnes = prédiction) :")
    print(f"{'':>32}", end="")
    for c in col_labels:
        print(f"{c:>14}", end="")
    print()
    for i, row_name in enumerate(row_labels):
        print(f"{row_name:>32}", end="")
        for j in range(cm.shape[1]):
            print(f"{cm[i, j]:>14}", end="")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline RandomForest sur features topologiques (UPFD)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Chemin vers un CSV de l'étape 2 ; si vide, recalcule le DataFrame.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5464,
        help="Taille de l'échantillon si recalcul (défaut : tout le corpus).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine (reproductibilité train/test + forêt).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test (défaut : 0.2).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    df = load_features_dataframe(csv_path, args.n_samples, args.seed)
    X, y = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = make_baseline_pipeline(random_state=args.seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_fake = f1_score(y_test, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("=" * 64)
    print("Baseline — RandomForest sur features topologiques uniquement")
    print("=" * 64)
    print(f"Échantillon : {len(df)} graphes | Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"Features : {TOPOLOGY_FEATURES}")
    print()
    print(f"Accuracy (test)     : {acc:.4f}")
    print(f"F1-score (classe 1 = Fake, test) : {f1_fake:.4f}")
    print_confusion_matrix_pretty(cm)
    print(
        "\nInterprétation rapide : [0,0] vrais négatifs (Real→Real), [1,1] vrais "
        "positifs (Fake→Fake) ; hors diagonale = confusions."
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
