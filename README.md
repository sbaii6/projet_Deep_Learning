# Projet Deep Learning & MLOps

Projet réalisé dans le cadre du module **Deep Learning** à l'EMSI.

##  Objectif

Comparer plusieurs architectures de Deep Learning sur différents types de données :

* Données tabulaires (MLP)
* Images (CNN)
* Séquences temporelles (LSTM, GRU, Seq2Seq)
* Vidéos (CNN + LSTM)

##  Modèles Implémentés

### MLP

Classification du cancer du sein à partir de données médicales.

### CNN

Classification d'images du dataset CIFAR-10.

### LSTM / GRU / Seq2Seq

Analyse de séquences et traduction automatique.

### CNN + LSTM

Reconnaissance d'actions à partir de séquences d'images.

##  Dashboard Streamlit

Interface permettant :

* Visualisation des performances
* Courbes Accuracy / Loss
* Matrices de confusion
* Prédictions interactives

##  Technologies

* Python
* PyTorch
* Streamlit
* NumPy
* Pandas
* Matplotlib

##  Installation

### Cloner le projet

```bash
git clone https://github.com/Abderrahmane122k/Projet-DeepLearning-MLOps.git
cd Projet-DeepLearning-MLOps
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Entraîner les modèles

```bash
python partie1_mlp.py
python partie2_cnn.py
python partie3_seq2seq.py
python partie4_lstm_hybride.py
```

### Lancer l'interface Streamlit

```bash
streamlit run app.py
```

## Structure du Projet

```text
Projet-DeepLearning-MLOps/
│
├── partie1_mlp.py
├── partie2_cnn.py
├── partie3_seq2seq.py
├── partie4_lstm_hybride.py
├── app.py
├── requirements.txt
└── README.md
```

##  Auteur

**Abderrahmane Sbaii**
EMSI - 4IIR
Année universitaire 2025-2026
