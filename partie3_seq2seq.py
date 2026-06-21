
"""
Partie 3 - Systeme Seq2Seq GRU pour traduction automatique.

Le script definit une architecture encodeur-decodeur recurrente :
- tokenisation simple et vocabulaire avec tokens speciaux ;
- encodeur/decodeur recurrent configurable en GRU ou LSTM ;
- attention additive de Bahdanau pour mieux conserver le contexte source ;
- teacher forcing pendant l'entrainement ;
- CrossEntropyLoss(ignore_index=PAD) pour ignorer le padding ;
- gradient clipping pour stabiliser la retropropagation a travers le temps ;
- sauvegarde du state_dict et des vocabulaires dans seq2seq_model.pth.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


CHECKPOINT_PATH = Path("seq2seq_model.pth")
DATASET_PATH = Path("fra.txt")
RANDOM_SEED = 42
CELL_TYPE = "LSTM"  # Choisir "GRU" ou "LSTM" sans changer le reste du code.
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_cell_type(cell_type: str) -> str:
    """Valide le type de cellule recurrente demande : GRU ou LSTM."""
    normalized = cell_type.upper()
    if normalized not in {"GRU", "LSTM"}:
        raise ValueError("cell_type doit etre 'GRU' ou 'LSTM'.")
    return normalized


def tokenize(text: str) -> list[str]:
    """Tokenisation volontairement simple pour un corpus pedagogique."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z<> ]+", " ", text)
    return [token for token in text.split() if token]


def resolve_dataset_path(path: Path = DATASET_PATH) -> Path:
    """
    Localise le fichier fra.txt du dataset Tatoeba anglais-francais.

    Le fichier peut etre place :
    - directement a la racine du projet : fra.txt
    - ou dans le dossier data : data/fra.txt
    """
    candidates = [path, Path("data") / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset introuvable. Placez le fichier Tatoeba 'fra.txt' "
        "dans la racine du projet ou dans le dossier data/."
    )


def load_translation_pairs(path: Path = DATASET_PATH, max_pairs: int = 15000, max_tokens: int = 12) -> list[tuple[str, str]]:
    """
    Charge un vrai corpus parallele anglais-francais depuis fra.txt.

    Format attendu du fichier Tatoeba simplifie :
        phrase anglaise<TAB>phrase francaise

    Certaines versions contiennent une troisieme colonne de metadonnees ;
    elle est ignoree automatiquement.
    """
    dataset_path = resolve_dataset_path(path)
    pairs: list[tuple[str, str]] = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            columns = line.strip().split("\t")
            if len(columns) < 2:
                continue

            src_sentence, tgt_sentence = columns[0], columns[1]
            src_tokens = tokenize(src_sentence)
            tgt_tokens = tokenize(tgt_sentence)

            # Filtrage pedagogique : phrases courtes pour un entrainement CPU rapide.
            if not src_tokens or not tgt_tokens:
                continue
            if len(src_tokens) > max_tokens or len(tgt_tokens) > max_tokens:
                continue

            pairs.append((src_sentence, tgt_sentence))
            if len(pairs) >= max_pairs:
                break

    if not pairs:
        raise ValueError("Aucune paire valide n'a ete chargee depuis le fichier fra.txt.")

    return pairs


def split_pairs(
    pairs: list[tuple[str, str]],
    train_ratio: float = 0.80,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Melange le corpus puis separe train/test pour une evaluation finale honnete."""
    shuffled = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split_index], shuffled[split_index:]


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]
    idx_to_token: list[str]

    @classmethod
    def build(cls, sentences: list[str]) -> "Vocabulary":
        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        tokens = sorted({token for sentence in sentences for token in tokenize(sentence)})
        idx_to_token = specials + [token for token in tokens if token not in specials]
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    def encode(self, sentence: str, add_sos: bool = True, add_eos: bool = True) -> list[int]:
        ids = []
        if add_sos:
            ids.append(self.token_to_idx[SOS_TOKEN])
        ids.extend(self.token_to_idx.get(token, self.token_to_idx[UNK_TOKEN]) for token in tokenize(sentence))
        if add_eos:
            ids.append(self.token_to_idx[EOS_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            token = self.idx_to_token[int(idx)]
            if token == EOS_TOKEN:
                break
            if token not in {PAD_TOKEN, SOS_TOKEN}:
                tokens.append(token)
        return " ".join(tokens)


class TranslationDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> None:
        self.examples = [
            (
                torch.tensor(src_vocab.encode(src), dtype=torch.long),
                torch.tensor(tgt_vocab.encode(tgt), dtype=torch.long),
            )
            for src, tgt in pairs
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


class EncoderGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.10,
        cell_type: str = "GRU",
    ) -> None:
        super().__init__()
        self.cell_type = normalize_cell_type(cell_type)
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)

        rnn_class = nn.GRU if self.cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src: torch.Tensor):
        embedded = self.embedding(src)
        # encoder_outputs contient un etat h_t pour chaque mot source.
        # Ces etats servent de "memoire" dans laquelle l'attention va chercher.
        encoder_outputs, hidden = self.rnn(embedded)
        return encoder_outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Attention additive de Bahdanau.

    A chaque pas du decodeur, on compare :
    - l'etat courant du decodeur ;
    - tous les etats produits par l'encodeur.

    Le resultat est une distribution de poids indiquant quels mots source sont
    les plus utiles pour produire le prochain mot cible.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_projection = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor], encoder_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pour LSTM, hidden est un couple (h_n, c_n). L'attention utilise h_n,
        # car c_n est la memoire interne longue et n'a pas la meme interpretation.
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # hidden.shape = (num_layers, batch_size, hidden_dim)
        # On utilise la derniere couche du decodeur pour calculer l'alignement.
        decoder_hidden = hidden[-1].unsqueeze(1)

        # energie.shape = (batch_size, src_len, hidden_dim)
        energie = torch.tanh(
            self.encoder_projection(encoder_outputs)
            + self.decoder_projection(decoder_hidden)
        )

        # scores.shape = (batch_size, src_len)
        scores = self.score(energie).squeeze(2)

        # Les positions PAD ne doivent pas recevoir d'attention.
        scores = scores.masked_fill(mask == 0, -1e9)
        return torch.softmax(scores, dim=1)


class DecoderGRU(nn.Module):
    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.10,
        cell_type: str = "GRU",
    ) -> None:
        super().__init__()
        self.cell_type = normalize_cell_type(cell_type)
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)

        # GRU et LSTM recoivent les memes dimensions d'entree et hidden_size.
        # La difference de portes est geree par PyTorch dans la cellule choisie.
        rnn_class = nn.GRU if self.cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim * 2 + embedding_dim, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        # attention_weights.shape = (batch_size, src_len)
        attention_weights = self.attention(hidden, encoder_outputs, mask)

        # context est la somme ponderee des etats encodeur.
        # Il resume les mots source les plus importants pour ce pas de generation.
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        # La cellule recurrente du decodeur recoit le mot precedent + le contexte.
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        prediction_input = torch.cat(
            (output.squeeze(1), context.squeeze(1), embedded.squeeze(1)),
            dim=1,
        )
        logits = self.fc_out(prediction_input)
        return logits, hidden, attention_weights


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder: EncoderGRU, decoder: DecoderGRU, sos_idx: int, eos_idx: int, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Masque les tokens PAD pour eviter que l'attention les utilise."""
        return (src != 0).to(self.device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Retourne les logits pour chaque pas cible.

        A chaque pas, le decodeur recoit soit le vrai token precedent
        (teacher forcing), soit sa propre prediction precedente.
        """
        batch_size, tgt_len = tgt.shape
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, output_dim, device=self.device)

        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.create_src_mask(src)
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            logits, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = logits
            predicted_token = logits.argmax(dim=1)
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = tgt[:, t] if use_teacher else predicted_token

        return outputs

    @torch.no_grad()
    def translate(self, src: torch.Tensor, max_len: int = 20) -> list[int]:
        self.eval()
        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.create_src_mask(src)
        input_token = torch.tensor([self.sos_idx], dtype=torch.long, device=self.device)
        generated: list[int] = []

        for _ in range(max_len):
            logits, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            next_token = int(logits.argmax(dim=1).item())
            if next_token == self.eos_idx:
                break
            generated.append(next_token)
            input_token = torch.tensor([next_token], dtype=torch.long, device=self.device)

        return generated


def build_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    cell_type: str = CELL_TYPE,
) -> Seq2SeqGRU:
    cell_type = normalize_cell_type(cell_type)
    encoder = EncoderGRU(src_vocab_size, embedding_dim, hidden_dim, cell_type=cell_type)
    decoder = DecoderGRU(tgt_vocab_size, embedding_dim, hidden_dim, cell_type=cell_type)
    return Seq2SeqGRU(encoder, decoder, sos_idx=1, eos_idx=2, device=device).to(device)


def train_one_epoch(model: Seq2SeqGRU, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        logits = model(src, tgt, teacher_forcing_ratio=0.60)

        # On ignore t=0 (<sos>) : la cible utile commence au token 1.
        loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * src.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_bleu(
    model: Seq2SeqGRU,
    pairs: list[tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 20,
) -> float:
    """
    Calcule le score BLEU corpus sur le jeu de test.

    BLEU compare les tokens generes par le modele avec la traduction de reference.
    Le smoothing evite un score nul trop brutal sur les petites phrases.
    """
    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []
    smoothing = SmoothingFunction().method1

    model.eval()
    for src_sentence, tgt_sentence in pairs:
        src_tensor = torch.tensor([src_vocab.encode(src_sentence)], dtype=torch.long, device=device)
        predicted_ids = model.translate(src_tensor, max_len=max_len)
        predicted_sentence = tgt_vocab.decode(predicted_ids)

        reference_tokens = tokenize(tgt_sentence)
        hypothesis_tokens = tokenize(predicted_sentence)
        if not hypothesis_tokens:
            hypothesis_tokens = [UNK_TOKEN]

        references.append([reference_tokens])
        hypotheses.append(hypothesis_tokens)

    return corpus_bleu(references, hypotheses, smoothing_function=smoothing)


def main() -> None:
    set_seed()
    device = get_device()

    pairs = load_translation_pairs()
    train_pairs, test_pairs = split_pairs(pairs)

    src_sentences = [src for src, _ in train_pairs]
    tgt_sentences = [tgt for _, tgt in train_pairs]
    src_vocab = Vocabulary.build(src_sentences)
    tgt_vocab = Vocabulary.build(tgt_sentences)

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

    model = build_model(len(src_vocab.idx_to_token), len(tgt_vocab.idx_to_token), device, cell_type=CELL_TYPE)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.token_to_idx[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    print(f"Device utilise: {device}")
    print(f"Paires chargees: {len(pairs)} | train: {len(train_pairs)} | test: {len(test_pairs)}")
    epochs = 30
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | loss={loss:.4f}")

    bleu = evaluate_bleu(model, test_pairs, src_vocab, tgt_vocab, device)
    print("\nEvaluation finale sur test:")
    print(f"BLEU : {bleu:.4f}")

    print("\nExemples de traduction:")
    for sentence, reference in test_pairs[:3]:
        src_tensor = torch.tensor([src_vocab.encode(sentence)], dtype=torch.long, device=device)
        pred_ids = model.translate(src_tensor)
        print(f"{sentence!r} -> prediction={tgt_vocab.decode(pred_ids)!r} | reference={reference!r}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "src_token_to_idx": src_vocab.token_to_idx,
            "src_idx_to_token": src_vocab.idx_to_token,
            "tgt_token_to_idx": tgt_vocab.token_to_idx,
            "tgt_idx_to_token": tgt_vocab.idx_to_token,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "architecture": f"Seq2Seq{normalize_cell_type(CELL_TYPE)}_BahdanauAttention",
            "cell_type": normalize_cell_type(CELL_TYPE),
            "dataset_path": str(resolve_dataset_path()),
            "bleu": bleu,
            "special_tokens": {
                "pad": PAD_TOKEN,
                "sos": SOS_TOKEN,
                "eos": EOS_TOKEN,
                "unk": UNK_TOKEN,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"\nArchitecture et poids sauvegardes dans {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
