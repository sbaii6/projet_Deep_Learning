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
        images = x.reshape(batch_size * sequence_length, channels, height, width)
        cartes = self.extracteur_spatial(images)
        vecteurs = self.projection(cartes)
        sequence_visuelle = vecteurs.reshape(batch_size, sequence_length, -1)
        output, (hidden, cell) = self.lstm(sequence_visuelle)
        dernier_etat = output[:, -1, :]
        return self.classifieur(dernier_etat)


class VideoDataset(Dataset):
    """Génère des vidéos soit très sombres (Classe 0), soit très claires (Classe 1)."""
    def __init__(self, num_samples: int = 300, seq_len: int = 10, num_classes: int = 2):
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.data = torch.zeros(num_samples, seq_len, 3, 32, 32)
        
        for i in range(num_samples):
            if self.labels[i] == 0:
                # Classe 0 : Vidéo Sombre (valeurs de pixels basses proches de 0)
                self.data[i] = torch.abs(torch.randn(seq_len, 3, 32, 32) * 0.1)
            else:
                # Classe 1 : Vidéo Lumineuse (valeurs de pixels hautes proches de 1)
                self.data[i] = torch.clamp(torch.ones(seq_len, 3, 32, 32) - torch.abs(torch.randn(seq_len, 3, 32, 32) * 0.1), 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def entrainer_modele_hybride() -> None:
    print("🚀 Entraînement du modèle sur l'analyse de la luminosité (2 classes)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoDataset(num_samples=300, seq_len=10, num_classes=2)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = ModeleHybrideCNNLSTM(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        model.train()
        total_loss, correct = 0.0, 0
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            
        print(f"Époque [{epoch+1}/5] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {correct/len(dataset):.2%}")
        
    torch.save({"model_state_dict": model.state_dict()}, "hybride_model.pth")
    print("✅ Modèle sauvegardé avec succès pour deux classes (0: Sombre, 1: Lumineux) !")


if __name__ == "__main__":
    entrainer_modele_hybride()