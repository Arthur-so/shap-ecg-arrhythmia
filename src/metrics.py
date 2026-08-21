"""Métricas de desempenho multi-rótulo (F1 por classe e macro).

Compartilhado entre ``train.py`` (early stopping / relatório final) e
``evaluate.py`` (F1 por classe para a análise de correlação da seção 4.5).
"""
from __future__ import annotations

import numpy as np

from src.config import CLASSES


def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converte probabilidades (N, C) em predições binárias por limiar."""
    return (probs >= threshold).astype(np.int64)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """F1-score por classe para rótulos binários multi-rótulo.

    Implementação direta (sem dependência dura de sklearn) para portabilidade.
    Retorna vetor (C,) com o F1 de cada classe.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = np.sum((y_pred == 1) & (y_true == 1), axis=0)
    fp = np.sum((y_pred == 1) & (y_true == 0), axis=0)
    fn = np.sum((y_pred == 0) & (y_true == 1), axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                          where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float),
                       where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(denom), where=denom > 0)
    return f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Média macro do F1 por classe."""
    return float(np.mean(per_class_f1(y_true, y_pred)))


def f1_report(y_true: np.ndarray, probs: np.ndarray,
              threshold: float = 0.5) -> dict[str, float]:
    """Dicionário {classe: f1} + 'macro' a partir de probabilidades."""
    y_pred = binarize(probs, threshold)
    f1s = per_class_f1(y_true, y_pred)
    report = {cls: float(f1s[i]) for i, cls in enumerate(CLASSES)}
    report["macro"] = float(np.mean(f1s))
    return report
