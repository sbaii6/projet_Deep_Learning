"""
Partie 4 - Modèle hybride CNN + LSTM (Late Fusion) avec apprentissage.

Ce script démontre :
1. L'extraction spatiale via un CNN sur chaque frame.
2. La modélisation temporelle via un LSTM sur la séquence de frames.
3. Une boucle d'entraînement complète sur des données vidéo synthétiques.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class ExtracteurSpatialCNN(nn.Module):
    """Bloc CNN pour extraire les caractéristiques spatiales d'une image."""
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x


class ModeleLSTM(nn.Module):
    """LSTM pour modéliser l'évolution temporelle des caractéristiques."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regresseur = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x)
        # On récupère le dernier état caché temporel
        dernier_etat = output[:, -1, :]
        return self.regresseur(dernier_etat)


class ModeleHybrideCNNLSTM(nn.Module):
    """Architecture Hybride finale : fusionne le CNN et le LSTM."""
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        cnn_feature_size: int = 64 * 4 * 4,
        lstm_input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.extracteur_spatial = ExtracteurSpatialCNN(in_channels=in_channels)
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_feature_size, lstm_input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.classifieur = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = x.shape
        
        # 1. Fusionner batch et temps pour le CNN
        images = x.reshape(batch_size * sequence_length, channels, height, width)
        cartes = self.extracteur_spatial(images)
        vecteurs = self.projection(cartes)
        
        # 2. Reconstruire la séquence pour le LSTM
        sequence_visuelle = vecteurs.reshape(batch_size, sequence_length, -1)
        output, (hidden, cell) = self.lstm(sequence_visuelle)
        dernier_etat = output[:, -1, :]
        
        return self.classifieur(dernier_etat)


# --- SECTION APPRENTISSAGE (TRAINING) ---

class VideoDataset(Dataset):
    """Générateur de données vidéo synthétiques pour l'entraînement."""
    def __init__(self, num_samples: int = 200, seq_len: int = 10, num_classes: int = 10):
        # On génère des tenseurs aléatoires simulant des vidéos (frames 32x32 RGB)
        self.data = torch.randn(num_samples, seq_len, 3, 32, 32)
        # On assigne des étiquettes (classes de 0 à 9)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def entrainer_modele_hybride() -> None:
    print("🚀 Initialisation de l'entraînement du modèle hybride CNN+LSTM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Préparation des données
    dataset = VideoDataset(num_samples=400, seq_len=10, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Initialisation du modèle, de la perte et de l'optimiseur
    model = ModeleHybrideCNNLSTM(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Boucle d'entraînement (Training Loop)
    epochs = 5
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct_preds = 0
        
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            
            # Remise à zéro des gradients
            optimizer.zero_grad()
            
            # Propagation avant (Forward)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Rétropropagation (Backward)
            loss.backward()
            
            # Gradient Clipping pour stabiliser le LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Mise à jour des poids
            optimizer.step()
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_preds += (predictions == labels).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / len(dataset)
        print(f"Époque [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
    # 4. Sauvegarde des poids entraînés
    print("\n💾 Sauvegarde des poids entraînés pour Streamlit...")
    torch.save({
        "model_state_dict": model.state_dict()
    }, "hybride_model.pth")
    print("✅ Entraînement terminé ! Le fichier 'hybride_model.pth' est prêt.")


if __name__ == "__main__":
    entrainer_modele_hybride()