from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CHECKPOINT_PATH = Path("mlp_breast_cancer.pth")
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Rend l'experience plus reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Selectionne le GPU si disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BreastCancerMLP(nn.Module):
    """
    Perceptron multicouche personnalise.

    La derniere couche retourne un logit brut. On n'ajoute donc pas de sigmoid
    dans forward(), car BCEWithLogitsLoss applique une sigmoid numeriquement stable.
    """

    def __init__(self, input_dim: int = 30, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def init_xavier_uniform(module: nn.Module) -> None:
    """Initialisation Xavier uniforme des couches lineaires, biais a zero."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def load_data(batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Charge Breast Cancer, normalise les variables et cree train/val/test."""
    dataset = load_breast_cancer()
    x = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    def make_loader(features: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        tensors = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    metadata = {
        "input_dim": int(x.shape[1]),
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "scaler_mean": scaler.mean_.astype(np.float32).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
    }
    return make_loader(x_train, y_train, True), make_loader(x_val, y_val, False), make_loader(x_test, y_test, False), metadata


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * x_batch.size(0)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_targets).astype(int)
    y_pred = (np.concatenate(all_probs) >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def inspect_model(model: nn.Module) -> None:
    """Affiche les noms et dimensions des parametres, comme demande dans la fiche."""
    print("\nParametres du modele:")
    for name, param in model.named_parameters():
        print(f"- {name:20s} shape={tuple(param.shape)} requires_grad={param.requires_grad}")
    print("\nCles du state_dict:")
    print(list(model.state_dict().keys()))


def main() -> None:
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader, metadata = load_data()

    model = BreastCancerMLP(input_dim=metadata["input_dim"]).to(device)
    model.apply(init_xavier_uniform)
    inspect_model(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    epochs = 120

    print(f"\nDevice utilise: {device}")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nEvaluation finale sur test:")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1']:.4f}")
    print("Matrice de confusion:")
    print(test_metrics["confusion_matrix"])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "test_metrics": {key: value for key, value in test_metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        },
        CHECKPOINT_PATH,
    )
    print(f"\nPoids sauvegardes dans {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
