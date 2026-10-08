"""Classe PyTorch Dataset para os sinais de ECG pré-processados.

Carrega os arquivos ``.npz`` produzidos por ``preprocess.py`` (arrays já
truncados/normalizados) e os expõe como tensores para o DataLoader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Dataset de ECG multi-rótulo a partir de um arquivo ``.npz``.

    Cada item é uma tupla ``(signal, label)`` onde ``signal`` é um tensor
    float32 de forma (12, 15000) e ``label`` um tensor float32 de forma (9,).

    Parameters
    ----------
    npz_path:
        Caminho para ``train.npz`` / ``val.npz`` / ``test.npz``.
    in_memory:
        Se True (padrão) carrega os arrays inteiros em memória. Para conjuntos
        muito grandes, use ``mmap`` deixando ``in_memory=False``.
    augment:
        Se True, aplica o aumento de dados de Zhang et al. (2021) a cada item
        (usar só no treino): ver ``augment_signal``.
    """

    def __init__(self, npz_path: str | Path, in_memory: bool = True,
                 augment: bool = False) -> None:
        self.npz_path = Path(npz_path)
        mmap = None if in_memory else "r"
        data = np.load(self.npz_path, mmap_mode=mmap)
        self.X = data["X"]
        self.Y = data["Y"]
        self.record_ids = data["record_ids"]
        self.lengths = data["lengths"] if "lengths" in data.files else None
        self.augment = augment
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X e Y têm números de amostras diferentes")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        if self.augment:
            length = int(self.lengths[idx]) if self.lengths is not None else x.shape[1]
            x = augment_signal(x, length)
        y = torch.from_numpy(np.asarray(self.Y[idx], dtype=np.float32))
        return x, y

    # -- utilidades ------------------------------------------------------- #
    def labels_array(self) -> np.ndarray:
        """Devolve todos os rótulos (N, 9) como array numpy."""
        return np.asarray(self.Y)

    def class_pos_counts(self) -> np.ndarray:
        """Número de exemplos positivos por classe (para pesos do BCE)."""
        return np.asarray(self.Y).sum(axis=0)

    def get_record_id(self, idx: int) -> str:
        return str(self.record_ids[idx])


# Probabilidade de aplicar cada transformação: o código de referência usa
# ``np.random.randn() > 0.5``, i.e. P(Z > 0,5) ≈ 0,31.
AUG_PROB = 0.3085
SCALE_SIGMA = 0.1     # fator de escala ~ N(1, 0,1) por derivação
SHIFT_MAX = 0.1       # deslocamento da linha de base ~ U(-0,1; 0,1) por derivação


def augment_signal(x: torch.Tensor, length: int) -> torch.Tensor:
    """Aumento de dados de Zhang et al. (2021), sobre o sinal real (sem padding).

    - Escala: multiplica cada derivação por um fator ~ N(1, 0,1).
    - Deslocamento: soma a cada derivação uma constante ~ U(-0,1; 0,1).

    Na referência o deslocamento é de ±0,02 mV no sinal bruto; como aqui o
    sinal já está em z-score (unidades de desvio padrão), usa-se ±0,1, que
    corresponde a ~0,02 mV para derivações com desvio de ~0,2 mV. Usa o
    gerador do torch para que cada worker do DataLoader tenha sorteios próprios.
    """
    n_leads = x.shape[0]
    if torch.rand(1).item() < AUG_PROB:
        x[:, :length] *= 1 + SCALE_SIGMA * torch.randn(n_leads, 1)
    if torch.rand(1).item() < AUG_PROB:
        x[:, :length] += SHIFT_MAX * (2 * torch.rand(n_leads, 1) - 1)
    return x


def compute_pos_weight(dataset: ECGDataset) -> torch.Tensor:
    """Calcula ``pos_weight`` do BCE: (n_neg / n_pos) por classe.

    Implementa pesos inversamente proporcionais à frequência de cada classe
    no conjunto de treino, conforme a seção 4.2 da proposta.
    """
    Y = dataset.labels_array()
    n = Y.shape[0]
    pos = Y.sum(axis=0)
    neg = n - pos
    # evita divisão por zero em classes ausentes
    pos = np.clip(pos, 1.0, None)
    weight = neg / pos
    return torch.tensor(weight, dtype=torch.float32)
