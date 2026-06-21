"""
Partie 2 - CNN inspire de LeNet sur CIFAR-10.

Le script applique les bonnes pratiques CNN vues dans la fiche :
- conservation de la structure spatiale des images ;
- convolutions locales avec partage de poids ;
- padding, pooling et couches fully-connected finales ;
- CrossEntropyLoss pour une classification multi-classe ;
- visualisation des cartes de caracteristiques de la premiere convolution.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset


CHECKPOINT_PATH = Path("lenet_cifar.pth")
FEATURE_MAP_PATH = Path("feature_maps_conv1.png")
RANDOM_SEED = 42
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNetCIFAR(nn.Module):
    """
    Variante de LeNet adaptee a CIFAR-10.

    Entree : 3 x 32 x 32
    conv1 + pool : 16 x 16 x 16
    conv2 + pool : 32 x 8 x 8
    conv3 + pool : 64 x 4 x 4
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
        )

    def forward_features_first_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les cartes de caracteristiques apres conv1 + ReLU."""
        return self.relu(self.conv1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.forward_features_first_layer(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.classifier(x)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_transform, test_transform


def load_data(batch_size: int = 128, max_train_samples: int | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Telecharge CIFAR-10 si necessaire et cree train/val/test."""
    train_transform, test_transform = get_transforms()
    train_full = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    val_full = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=test_transform)
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    indices = np.arange(len(train_full))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(indices)
    val_size = 5000
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


@torch.no_grad()
def save_first_layer_feature_maps(model: LeNetCIFAR, loader: DataLoader, device: torch.device, output_path: Path = FEATURE_MAP_PATH) -> None:
    """Genere une grille des cartes de caracteristiques de conv1 pour une image CIFAR-10."""
    model.eval()
    images, labels = next(iter(loader))
    image = images[:1].to(device)
    feature_maps = model.forward_features_first_layer(image).squeeze(0).cpu()

    cols = 4
    rows = int(np.ceil(feature_maps.size(0) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, axis in enumerate(axes):
        axis.axis("off")
        if idx < feature_maps.size(0):
            axis.imshow(feature_maps[idx], cmap="viridis")
            axis.set_title(f"Carte {idx + 1}")

    fig.suptitle(f"Cartes de caracteristiques conv1 - classe: {CIFAR10_CLASSES[int(labels[0])]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Cartes de caracteristiques sauvegardees dans {output_path.resolve()}")


def main() -> None:
    set_seed()
    device = get_device()
    train_loader, val_loader, test_loader = load_data()

    model = LeNetCIFAR(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_state = None
    best_val_acc = -1.0
    epochs = 20

    print(f"Device utilise: {device}")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nEvaluation finale sur test:")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"F1 macro : {test_metrics['f1_macro']:.4f}")
    print("Matrice de confusion:")
    print(test_metrics["confusion_matrix"])

    save_first_layer_feature_maps(model, test_loader, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CIFAR10_CLASSES,
            "test_metrics": {key: value for key, value in test_metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        },
        CHECKPOINT_PATH,
    )
    print(f"Poids sauvegardes dans {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
