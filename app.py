from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import cv2

from partie1_mlp import BreastCancerMLP
from partie2_cnn import CIFAR10_CLASSES, LeNetCIFAR
from partie3_seq2seq import Vocabulary, build_model
from partie4_lstm_hybride import ModeleHybrideCNNLSTM


MLP_PATH = Path("mlp_breast_cancer.pth")
CNN_PATH = Path("lenet_cifar.pth")
SEQ2SEQ_PATH = Path("seq2seq_model.pth")
HYBRIDE_PATH = Path("hybride_model.pth")


st.set_page_config(page_title="Projet Deep Learning EMSI", page_icon="DL", layout="wide")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_mlp() -> tuple[BreastCancerMLP, dict]:
    checkpoint = torch.load(MLP_PATH, map_location=device(), weights_only=False)
    metadata = checkpoint["metadata"]
    model = BreastCancerMLP(input_dim=metadata["input_dim"]).to(device())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


@st.cache_resource
def load_cnn() -> tuple[LeNetCIFAR, tuple[str, ...]]:
    checkpoint = torch.load(CNN_PATH, map_location=device(), weights_only=False)
    classes = tuple(checkpoint.get("classes", CIFAR10_CLASSES))
    model = LeNetCIFAR(num_classes=len(classes)).to(device())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


@st.cache_resource
def load_seq2seq():
    checkpoint = torch.load(SEQ2SEQ_PATH, map_location=device(), weights_only=False)
    src_vocab = Vocabulary(checkpoint["src_token_to_idx"], checkpoint["src_idx_to_token"])
    tgt_vocab = Vocabulary(checkpoint["tgt_token_to_idx"], checkpoint["tgt_idx_to_token"])
    
    saved_cell_type = checkpoint.get("cell_type", "GRU")
    
    model = build_model(
        src_vocab_size=len(src_vocab.idx_to_token),
        tgt_vocab_size=len(tgt_vocab.idx_to_token),
        device=device(),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        cell_type=saved_cell_type,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab


@st.cache_resource
def load_hybride():
    checkpoint = torch.load(HYBRIDE_PATH, map_location=device(), weights_only=False)
    # ⚠️ Changement vital ici : num_classes=2 pour correspondre au nouvel entraînement
    model = ModeleHybrideCNNLSTM(num_classes=2).to(device())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def checkpoint_missing(path: Path) -> bool:
    if path.exists():
        return False
    st.warning(f"Checkpoint introuvable: `{path}`. Lancez d'abord le script d'entrainement correspondant.")
    return True


def page_mlp() -> None:
    st.title("MLP - Breast Cancer Wisconsin")
    if checkpoint_missing(MLP_PATH):
        return

    model, metadata = load_mlp()
    feature_names = metadata["feature_names"]
    mean = np.asarray(metadata["scaler_mean"], dtype=np.float32)
    scale = np.asarray(metadata["scaler_scale"], dtype=np.float32)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Donnees patient synthetiques")
        randomize = st.button("Generer de fausses donnees")
        if randomize or "mlp_values" not in st.session_state:
            rng = np.random.default_rng()
            st.session_state["mlp_values"] = rng.normal(loc=mean, scale=scale).astype(np.float32)

        values = []
        for idx, name in enumerate(feature_names):
            values.append(
                st.number_input(
                    name,
                    value=float(st.session_state["mlp_values"][idx]),
                    step=float(max(scale[idx] * 0.10, 0.01)),
                    format="%.4f",
                )
            )

    with col_right:
        raw = np.asarray(values, dtype=np.float32)
        standardized = ((raw - mean) / scale).reshape(1, -1)
        x = torch.tensor(standardized, dtype=torch.float32, device=device())
        with torch.no_grad():
            logit = model(x)
            probability = torch.sigmoid(logit).item()
        pred_idx = int(probability >= 0.5)
        label = metadata["target_names"][pred_idx]

        st.subheader("Prediction")
        st.metric("Classe predite", label)
        st.metric("Probabilite classe positive", f"{probability:.2%}")
        st.caption("Le modele retourne un logit brut; la probabilite est calculee par sigmoid en inference.")
        st.dataframe(pd.DataFrame({"feature": feature_names, "value": raw}), use_container_width=True)


def page_cnn() -> None:
    st.title("CNN LeNet - CIFAR-10")
    if checkpoint_missing(CNN_PATH):
        return

    model, classes = load_cnn()
    uploaded = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    if uploaded is None:
        st.info("Importez une image pour lancer la prediction CIFAR-10.")
        return

    image = Image.open(uploaded).convert("RGB")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.image(image, caption="Image importee", use_container_width=True)

    x = transform(image).unsqueeze(0).to(device())
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probabilities.argmax())
        feature_maps = model.forward_features_first_layer(x).squeeze(0).cpu()

    with col_right:
        st.subheader("Prediction")
        st.metric("Classe predite", classes[pred_idx])
        chart_data = pd.DataFrame({"classe": classes, "probabilite": probabilities}).sort_values("probabilite", ascending=False)
        st.bar_chart(chart_data.set_index("classe"))

    st.subheader("Cartes de caracteristiques de la premiere convolution")
    cols = st.columns(4)
    for idx in range(min(8, feature_maps.shape[0])):
        fmap = feature_maps[idx].numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        cols[idx % 4].image(fmap, caption=f"Carte {idx + 1}", clamp=True, use_container_width=True)


def page_seq2seq() -> None:
    st.title("Seq2Seq - Traduction automatique")
    if checkpoint_missing(SEQ2SEQ_PATH):
        return

    model, src_vocab, tgt_vocab = load_seq2seq()
    sentence = st.text_input("Texte source en anglais", value="i love deep learning")
    max_len = st.slider("Longueur maximale generee", min_value=3, max_value=30, value=12)

    if st.button("Traduire"):
        src_ids = src_vocab.encode(sentence)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device())
        pred_ids = model.translate(src_tensor, max_len=max_len)
        translation = tgt_vocab.decode(pred_ids)

        st.subheader("Traduction")
        st.write(translation if translation else "(sequence vide)")
        st.caption("Inference auto-regressive sans teacher forcing, avec decodage glouton.")


def page_hybride() -> None:
    st.title("Hybride CNN + LSTM - Séquence Vidéo")
    if checkpoint_missing(HYBRIDE_PATH):
        st.info("💡 Astuce : Exécutez d'abord `python partie4_lstm_hybride.py` pour générer le modèle.")
        return

    model = load_hybride()
    
    st.write("Le modèle découpe votre vidéo pour extraire un tenseur 5D : `(batch, frames, canaux, hauteur, largeur)`")
    
    uploaded_video = st.file_uploader("Importez une courte vidéo (.mp4, .mov)", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("Lancer l'analyse du flux vidéo"):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            frames = []
            transform = transforms.ToTensor() 
            
            while len(frames) < 10:
                ret, frame = cap.read()
                if not ret:
                    break 
                
                frame_resized = cv2.resize(frame, (32, 32))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                tensor_frame = transform(frame_rgb)
                frames.append(tensor_frame)
                
            cap.release()
            
            if len(frames) < 10:
                st.error(f"La vidéo est trop courte ! Il faut au moins 10 frames (trouvé {len(frames)}).")
                return
            
            video_tensor = torch.stack(frames) 
            video_reelle = video_tensor.unsqueeze(0).to(device())
            
            with torch.no_grad():
                logits = model(video_reelle)
                probabilities = torch.softmax(logits, dim=1)
                classe = logits.argmax().item()
                probabilite = probabilities[0][classe].item()
                
            st.success(f"Analyse réussie ! Votre vidéo a été convertie en tenseur {list(video_reelle.shape)} et a traversé le modèle.")
            
            col1, col2 = st.columns(2)
            interpretation = "Vidéo Sombre / Nuit" if classe == 0 else "Vidéo Lumineuse / Jour"
            
            col1.metric("Analyse de la scène", interpretation)
            col2.metric("Confiance du modèle", f"{probabilite:.2%}")


def main() -> None:
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Modele", ["MLP tabulaire", "CNN image", "Seq2Seq texte", "Hybride vidéo"])
    st.sidebar.caption(f"Device: {device()}")

    if choice == "MLP tabulaire":
        page_mlp()
    elif choice == "CNN image":
        page_cnn()
    elif choice == "Seq2Seq texte":
        page_seq2seq()
    else:
        page_hybride()


if __name__ == "__main__":
    main()