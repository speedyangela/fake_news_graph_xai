"""GCN sur UPFD : message passing + max pooling + features racine (comme l’exemple PyG upfd). Entrées : edge_index, x (BERT)."""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.transforms import ToUndirected

from data_loader import _patch_upfd_google_drive_ids, default_data_root


def root_node_indices(batch: Tensor) -> Tensor:
    """Index du premier nœud de chaque graphe dans un batch PyG (racine UPFD = début du bloc)."""
    if batch.numel() == 0:
        return batch.new_zeros(0, dtype=torch.long)
    diff = batch[1:] - batch[:-1]
    starts = (diff != 0).nonzero(as_tuple=False).view(-1)
    return torch.cat([starts.new_zeros(1), starts + 1], dim=0)


class PropagationGCN(nn.Module):
    """2× GCNConv, max pool sur le graphe, concat avec x racine → logits (CrossEntropy côté loss)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.lin_root = nn.Linear(in_channels, hidden_channels)
        self.lin_out = nn.Linear(2 * hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x0 = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        h_pool = global_max_pool(x, batch)
        roots = root_node_indices(batch)
        h_root = F.relu(self.lin_root(x0[roots]))

        h = torch.cat([h_pool, h_root], dim=-1)
        return self.lin_out(h)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> float:
    """Une epoch : perte moyenne par graphe."""
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_graphs = 0

    for data in loader:
        data = data.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index, data.batch)
        loss = criterion(logits, data.y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * data.num_graphs
        n_graphs += data.num_graphs

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Accuracy et F1 (fake = classe 1)."""
    model.eval()
    preds: List[Tensor] = []
    labels: List[Tensor] = []

    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)
        preds.append(pred.cpu())
        labels.append(data.y.cpu().view(-1))

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1))
    return acc, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="GCN PyG pour UPFD (gossipcop, bert).")
    parser.add_argument("--epochs", type=int, default=40, help="Nombre d’epochs (30–50 conseillé).")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de mini-batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Pas Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Décroissance L2.")
    parser.add_argument("--hidden", type=int, default=128, help="Dimension cachée GCN.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout entre conv1 et conv2.")
    parser.add_argument("--seed", type=int, default=42, help="Graine reproductibilité.")
    args = parser.parse_args()

    epochs = max(1, min(args.epochs, 50))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = ToUndirected()

    root = default_data_root()
    _patch_upfd_google_drive_ids()
    train_dataset = UPFD(str(root), "gossipcop", "bert", split="train", transform=transform)
    val_dataset = UPFD(str(root), "gossipcop", "bert", split="val", transform=transform)
    test_dataset = UPFD(str(root), "gossipcop", "bert", split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PropagationGCN(
        in_channels=train_dataset.num_features,
        hidden_channels=args.hidden,
        num_classes=int(train_dataset.num_classes),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print("=" * 72)
    print("Entraînement GCN — UPFD gossipcop (features BERT, graphes non dirigés)")
    print("=" * 72)
    print(f"Périphérique : {device}")
    print(f"Epochs : {epochs} | batch_size : {args.batch_size} | hidden : {args.hidden}")
    print(f"Train : {len(train_dataset)} graphes | Test : {len(test_dataset)} graphes")
    print()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )

        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                val_loss = run_epoch(
                    model, val_loader, None, criterion, device, train=False
                )
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}"
            )

    test_acc, test_f1 = evaluate_metrics(model, test_loader, device)

    print()
    print("— Métriques finales (jeu de test) —")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1-score (classe 1 = Fake, pos_label=1) : {test_f1:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
