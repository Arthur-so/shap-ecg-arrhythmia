"""Configuração comum do GradientSHAP: referência (baselines), sementes e sorteios.

Todos os módulos que geram mapas (``explain.py``, ``sensitivity.py``,
``random_logit.py`` e os testes de referência) usam estas funções, para que um
mapa refeito por qualquer um deles seja idêntico ao mapa salvo.

Configuração adotada no TCC desde 26/10/2026 (opção D da verificação da
referência, ``fichamentos/verificacao-referencia-gradientshap``):
  - referência ``balanceada``: 4 ECGs de rótulo único de cada uma das 9 classes
    do treino (36), sorteados com a semente do projeto;
  - 200 sorteios por mapa, como média de 4 blocos de 50 (cabe na memória da
    GPU e equivale estatisticamente a 200 sorteios de uma vez);
  - semente própria de cada mapa, ``map_seed``; o bloco j usa
    ``semente + j * CHUNK_STEP``.
A configuração anterior (32 ECGs sorteados do treino, 50 sorteios) continua
disponível com ``reference="treino"`` e ``chunks=1``.
"""
from __future__ import annotations

import numpy as np
import torch

from src.config import CLASSES

REFERENCES = ["balanceada", "treino", "normal"]
CHUNK_STEP = 7_919  # distância entre as sementes dos blocos de sorteios


def map_seed(seed: int, test_index: int, class_index: int) -> int:
    """Semente de um mapa (ECG do teste × classe explicada)."""
    return seed * 100_003 + test_index * 11 + class_index


def random_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    """``n`` ECGs sorteados do treino, sem reposição (referência ``treino``)."""
    return np.random.default_rng(seed).choice(n_total, size=min(n, n_total), replace=False)


def normal_indices(Y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` ECGs com rótulo único SNR (referência ``normal``)."""
    snr = CLASSES.index("SNR")
    pool = np.flatnonzero((Y[:, snr] == 1) & (Y.sum(axis=1) == 1))
    return np.random.default_rng(seed).choice(pool, size=min(n, pool.size), replace=False)


def balanced_indices(Y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """``per_class`` ECGs de rótulo único de cada classe (referência ``balanceada``)."""
    rng = np.random.default_rng(seed)
    single = Y.sum(axis=1) == 1
    idx = [rng.choice(np.flatnonzero(single & (Y[:, k] == 1)), size=per_class, replace=False)
           for k in range(Y.shape[1])]
    return np.concatenate(idx)


def reference_indices(Y: np.ndarray, reference: str, seed: int,
                      n_baselines: int = 32, per_class: int = 4) -> np.ndarray:
    """Índices, no treino, dos ECGs de referência."""
    if reference == "balanceada":
        return balanced_indices(Y, per_class, seed)
    if reference == "treino":
        return random_indices(len(Y), n_baselines, seed)
    if reference == "normal":
        return normal_indices(Y, n_baselines, seed)
    raise ValueError(f"referência desconhecida: {reference}")


class Explainer:
    """GradientSHAP com referência fixa e mapas como média de blocos de sorteios."""

    def __init__(self, model: torch.nn.Module, X_train: np.ndarray, Y_train: np.ndarray,
                 device: torch.device, reference: str = "balanceada", seed: int = 42,
                 n_baselines: int = 32, per_class: int = 4, n_samples: int = 50,
                 chunks: int = 4, stdevs: float = 0.09) -> None:
        from captum.attr import GradientShap

        self.idx = reference_indices(Y_train, reference, seed, n_baselines, per_class)
        self.baselines = torch.from_numpy(X_train[self.idx].astype(np.float32)).to(device)
        self.counts = Y_train[self.idx].sum(axis=0).astype(int)
        self.gs = GradientShap(model)
        self.device = device
        self.reference = reference
        self.n_samples, self.chunks, self.stdevs = n_samples, chunks, stdevs

    def describe(self) -> str:
        comp = " ".join(f"{c}={k}" for c, k in zip(CLASSES, self.counts))
        return (f"referência {self.reference} ({len(self.idx)} ECGs: {comp}); "
                f"{self.n_samples * self.chunks} sorteios ({self.chunks} bloco(s) de {self.n_samples})")

    def block(self, x: np.ndarray, c: int, seed: int) -> np.ndarray:
        """Um bloco de ``n_samples`` sorteios com a semente ``seed``."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        xt = torch.from_numpy(x[None].astype(np.float32)).to(self.device).requires_grad_(True)
        attr = self.gs.attribute(xt, baselines=self.baselines, target=int(c),
                                 n_samples=self.n_samples, stdevs=self.stdevs)
        return attr.detach().cpu().numpy()[0].astype(np.float64)

    def __call__(self, x: np.ndarray, c: int, seed: int) -> np.ndarray:
        """Mapa (12, T): média de ``chunks`` blocos, sementes ``seed + j·CHUNK_STEP``."""
        return np.mean([self.block(x, c, seed + j * CHUNK_STEP) for j in range(self.chunks)],
                       axis=0)


def add_args(p) -> None:
    """Argumentos de linha de comando comuns aos módulos que geram mapas."""
    p.add_argument("--reference", type=str, default="balanceada", choices=REFERENCES,
                   help="ECGs de referência do GradientSHAP (padrão: balanceada)")
    p.add_argument("--n-baselines", type=int, default=32,
                   help="tamanho das referências treino e normal")
    p.add_argument("--per-class", type=int, default=4,
                   help="ECGs de cada classe na referência balanceada")
    p.add_argument("--n-samples", type=int, default=50, help="sorteios por bloco")
    p.add_argument("--chunks", type=int, default=4, help="blocos por mapa (4 × 50 = 200)")
    p.add_argument("--stdevs", type=float, default=0.09, help="desvio do ruído do GradientSHAP")
