"""
Partie 4 - Modeles LSTM et hybride CNN + LSTM.

Ce fichier sert de support clair pour expliquer l'architecture LSTM :
- le LSTM lit une sequence dans l'ordre temporel ;
- a chaque pas de temps, ses portes (forget, input, output) decident quoi oublier,
  quoi ajouter dans la memoire C_t, et quoi exposer dans l'etat cache h_t ;
- pour une sequence d'images, le CNN extrait d'abord les informations spatiales
  de chaque image, puis le LSTM modelise leur evolution dans le temps.
"""

from __future__ import annotations

import torch
from torch import nn


class ModeleLSTM(nn.Module):
    """
    LSTM simple pour series temporelles.

    Entree attendue :
        x.shape = (batch_size, sequence_length, input_size)

    Exemple :
        - batch_size = nombre de series dans un lot
        - sequence_length = nombre de pas de temps
        - input_size = nombre de variables observees a chaque pas

    Sortie :
        logits.shape = (batch_size, output_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        # nn.LSTM implemente directement les equations vues sur le schema :
        # f_t = porte d'oubli, i_t = porte d'entree, g_t = contenu candidat,
        # o_t = porte de sortie, C_t = memoire longue, h_t = etat cache.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # La tete finale transforme la representation temporelle en prediction.
        # output_size=1 convient a une regression ou une classification binaire.
        self.regresseur = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output contient h_t pour tous les pas de temps :
        # output.shape = (batch_size, sequence_length, hidden_size)
        output, (hidden, cell) = self.lstm(x)

        # Pour predire toute la serie, on utilise le dernier etat temporel h_T.
        # Avec plusieurs couches, output[:, -1, :] correspond a la derniere couche.
        dernier_etat = output[:, -1, :]

        # La cellule cell represente C_t, la memoire longue du LSTM.
        # Elle est calculee par PyTorch mais n'est pas necessaire ici pour predire.
        return self.regresseur(dernier_etat)


class ExtracteurSpatialCNN(nn.Module):
    """
    Bloc CNN inspire de la partie 2.

    Il conserve la logique de LeNetCIFAR :
    Conv2d -> ReLU -> MaxPool, repete trois fois.

    Pour une image CIFAR 32x32 :
        entree :  3 x 32 x 32
        sortie : 64 x 4 x 4
    """

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
    """
    Modele hybride pour sequences d'images.

    Idee principale :
        1. Le CNN analyse chaque image separement pour extraire les formes,
           textures et motifs spatiaux.
        2. Le LSTM recoit ensuite la suite de ces vecteurs visuels pour apprendre
           la dynamique temporelle.

    Entree attendue :
        x.shape = (batch_size, sequence_length, channels, height, width)

    Exemple :
        une video courte de 10 images CIFAR :
        x.shape = (batch_size, 10, 3, 32, 32)
    """

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

        # Projection compacte : on transforme la carte CNN aplatie en vecteur
        # temporel donne au LSTM. Cela evite d'envoyer un vecteur trop grand.
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

        # Pour une classification video/sequence, on classe le dernier etat h_T.
        self.classifieur = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = x.shape

        # On fusionne batch et temps pour appliquer le meme CNN a chaque image.
        # (B, T, C, H, W) devient (B*T, C, H, W).
        images = x.reshape(batch_size * sequence_length, channels, height, width)

        # Extraction spatiale : chaque image devient une carte de caracteristiques.
        cartes = self.extracteur_spatial(images)

        # Projection : chaque carte CNN devient un vecteur.
        vecteurs = self.projection(cartes)

        # On reconstruit la dimension temporelle pour le LSTM :
        # (B*T, F) devient (B, T, F).
        sequence_visuelle = vecteurs.reshape(batch_size, sequence_length, -1)

        # Le LSTM apprend comment les caracteristiques visuelles evoluent.
        output, (hidden, cell) = self.lstm(sequence_visuelle)
        dernier_etat = output[:, -1, :]

        return self.classifieur(dernier_etat)


if __name__ == "__main__":
    # Mini-test de dimensions pour verifier que les deux modeles fonctionnent.
    serie = torch.randn(8, 12, 5)  # 8 series, 12 pas de temps, 5 variables
    modele_lstm = ModeleLSTM(input_size=5, hidden_size=64, output_size=1)
    print("Sortie LSTM:", modele_lstm(serie).shape)

    video = torch.randn(4, 10, 3, 32, 32)  # 4 videos, 10 images, RGB 32x32
    modele_hybride = ModeleHybrideCNNLSTM(num_classes=10)
    print("Sortie hybride CNN+LSTM:", modele_hybride(video).shape)
