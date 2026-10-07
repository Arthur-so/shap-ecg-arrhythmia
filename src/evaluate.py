"""Avaliação quantitativa das explicações com o toolkit Quantus (seção 4.4).

Calcula, por amostra e depois agregado por classe diagnóstica, as métricas de:

  Faithfulness: Faithfulness Correlation, Faithfulness Estimate, Selectivity,
                SensitivityN, Infidelity, Sufficiency.
  Robustness:   Local Lipschitz Estimate, Max-Sensitivity, Avg-Sensitivity,
                Continuity, Relative Input Stability (RIS),
                Relative Output Stability (ROS).

A Consistency (Dasgupta et al., 2022) é calculada em ``consistency.py``: a
discretização padrão da ``quantus.Consistency`` (sinal dos 5 primeiros valores
do mapa, após valor absoluto) dava a mesma explicação a todos os mapas.

As métricas de robustez requerem gerar novas explicações sob perturbação;
para isso é fornecida uma ``explain_func`` que reproduz o GradientSHAP usado
em ``explain.py`` (mesmos baselines de treino).

Ao final salva:
  - ``results/quantus_metrics.csv``: média de cada métrica por classe.
  - ``results/quality_vs_f1.csv``: métricas de qualidade + F1 por classe,
    base para a análise de correlação da seção 4.5.

Uso:
    python -m src.evaluate \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --attr-dir results/attributions \
        --results-dir results \
        --max-per-class 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import CLASSES, SEED
from src.data.dataset import ECGDataset
from src.device import get_device
from src.model.resnet34_1d import build_model


# --------------------------------------------------------------------------- #
# Modelo e função de explicação (para métricas de robustez)
# --------------------------------------------------------------------------- #
class _FlexModel(torch.nn.Module):
    """Envolve o modelo 1D aceitando também entradas pseudo-2D.

    Algumas métricas do Quantus foram escritas para imagens e inserem uma
    dimensão unitária no sinal: a Infidelity espera (N, 1, 12, L) e a
    Continuity expande internamente para (N, 12, 1, L). Este wrapper achata
    qualquer entrada 4D de volta para (N, derivações, tempo) antes de chamar
    o modelo ResNet34 1D real, sem afetar as entradas 3D normais.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        return self.model(x)


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def make_explain_func(torch_model, baselines: torch.Tensor, device,
                      n_samples: int = 20, stdevs: float = 0.09):
    """Cria uma explain_func compatível com Quantus usando GradientSHAP.

    Assinatura exigida pelo Quantus: ``f(model, inputs, targets, **kwargs)``
    retornando atribuições numpy com a mesma forma de ``inputs``.
    """
    from captum.attr import GradientShap

    gradient_shap = GradientShap(torch_model)

    def explain_func(model, inputs, targets, **kwargs):  # noqa: ARG001
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        x.requires_grad_(True)
        tgt = torch.as_tensor(targets, dtype=torch.long, device=device)
        # adapta os baselines ao shape da entrada (métricas como Continuity
        # inserem uma dimensão unitária); o nº de elementos é o mesmo.
        b = baselines
        if tuple(b.shape[1:]) != tuple(x.shape[1:]):
            b = b.reshape(b.shape[0], *x.shape[1:])
        attr = gradient_shap.attribute(
            x, baselines=b, target=tgt,
            n_samples=n_samples, stdevs=stdevs,
        )
        return attr.detach().cpu().numpy().astype(np.float32)

    return explain_func


# --------------------------------------------------------------------------- #
# Definição das métricas Quantus
# --------------------------------------------------------------------------- #
def build_metrics(explain_func):
    """Instancia os objetos de métrica do Quantus.

    Retorna dois dicionários (faithfulness, robustness) {nome: métrica}.
    Import tardio para não exigir Quantus em ambientes só de estruturação.
    """
    import quantus

    faithfulness = {
        "FaithfulnessCorrelation": quantus.FaithfulnessCorrelation(
            nr_runs=10, subset_size=224, return_aggregate=False,
            disable_warnings=True),
        # features_in_step precisa dividir o nº de pontos temporais (15000);
        # 500 -> 30 passos. 224 (usado antes) não divide 15000.
        "FaithfulnessEstimate": quantus.FaithfulnessEstimate(
            features_in_step=500, disable_warnings=True),
        "Selectivity": quantus.Selectivity(
            patch_size=200, disable_warnings=True),
        "SensitivityN": quantus.SensitivityN(
            features_in_step=500, n_max_percentage=0.8, disable_warnings=True),
        # Patches "só no tempo": com o sinal como imagem (N, 1, 12, L), a altura é
        # 12 (derivações) e a largura é o tempo. patch_size=12 divide exatamente a
        # altura (1 bloco = todas as 12 derivações) e a largura (15000/12=1250),
        # sem padding. Cada patch cobre as 12 derivações de uma vez × 12 instantes
        # (~24 ms), deslizando apenas no tempo — não agrupa derivações como
        # vizinhas espaciais.
        "Infidelity": quantus.Infidelity(
            perturb_baseline="uniform", n_perturb_samples=5,
            perturb_patch_sizes=[12], disable_warnings=True),
        "Sufficiency": quantus.Sufficiency(disable_warnings=True),
    }
    robustness = {
        "LocalLipschitzEstimate": quantus.LocalLipschitzEstimate(
            nr_samples=10, disable_warnings=True),
        "MaxSensitivity": quantus.MaxSensitivity(
            nr_samples=10, disable_warnings=True),
        "AvgSensitivity": quantus.AvgSensitivity(
            nr_samples=10, disable_warnings=True),
        "Continuity": quantus.Continuity(disable_warnings=True),
        "RelativeInputStability": quantus.RelativeInputStability(
            nr_samples=10, disable_warnings=True),
        "RelativeOutputStability": quantus.RelativeOutputStability(
            nr_samples=10, disable_warnings=True),
    }
    # explain_func é usada pelas métricas de robustez
    for m in robustness.values():
        m.explain_func = explain_func
    return faithfulness, robustness


def _run_metric(metric, model, x, y, a, device, explain_func=None) -> float:
    """Executa uma métrica Quantus e devolve a média dos scores.

    Encapsula em try/except para que a falha de uma métrica não interrompa a
    avaliação das demais (retorna NaN nesse caso).
    """
    kwargs = dict(model=model, x_batch=x, y_batch=y, a_batch=a,
                  device=str(device), channel_first=True)
    if explain_func is not None:
        kwargs["explain_func"] = explain_func
    try:
        scores = metric(**kwargs)
        arr = np.asarray(scores, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")
    except Exception as exc:  # noqa: BLE001
        print(f"      [erro] {type(metric).__name__}: {exc}")
        return float("nan")


def run_all_metrics(flex_model, x_batch, y_batch, a_batch,
                    faithfulness, robustness, explain_func, device,
                    scores: dict[str, float] | None = None,
                    on_update=None) -> dict[str, float]:
    """Roda todas as métricas sobre um lote e devolve {nome: score médio}.

    A Infidelity é a única que exige o sinal em formato de imagem 2D
    (N, 1, 12, L); as demais usam o formato 1D nativo (N, 12, L).

    ``scores`` pode ser um dict pré-existente que será preenchido no local
    (permitindo que o chamador o observe antes do término); ``on_update`` é
    chamado após cada métrica calculada, para persistir resultados parciais.
    """
    if scores is None:
        scores = {}
    # Infidelity em formato de imagem (N, 1, 12, L): altura = 12 derivações,
    # largura = tempo. Combinado com patch_size >= 12 (ver build_metrics), cada
    # patch cobre as 12 derivações de uma vez e desliza só no tempo.
    x4 = x_batch[:, None, :, :]   # (N, 1, 12, L)
    a4 = a_batch[:, None, :, :]
    for name, metric in faithfulness.items():
        if name == "Infidelity":
            scores[name] = _run_metric(metric, flex_model, x4, y_batch, a4, device)
        else:
            scores[name] = _run_metric(metric, flex_model, x_batch, y_batch, a_batch, device)
        print(f"    {name:>26}: {scores[name]:.4f}")
        if on_update is not None:
            on_update()
    for name, metric in robustness.items():
        scores[name] = _run_metric(metric, flex_model, x_batch, y_batch, a_batch,
                                   device, explain_func=explain_func)
        print(f"    {name:>26}: {scores[name]:.4f}")
        if on_update is not None:
            on_update()
    return scores


# --------------------------------------------------------------------------- #
# Avaliação por classe
# --------------------------------------------------------------------------- #
def evaluate(args: argparse.Namespace) -> None:
    device = get_device()

    data_dir = Path(args.data_dir)
    train_ds = ECGDataset(data_dir / "train.npz")
    test_ds = ECGDataset(data_dir / "test.npz")
    model = load_model(Path(args.checkpoint), device)
    flex_model = _FlexModel(model).to(device).eval()

    # baselines (mesma lógica de explain.py) para a explain_func da robustez
    rng = np.random.default_rng(args.seed)
    bidx = rng.choice(len(train_ds), size=min(args.n_baselines, len(train_ds)),
                      replace=False)
    baselines = torch.from_numpy(
        np.stack([train_ds.X[i] for i in bidx]).astype(np.float32)).to(device)
    explain_func = make_explain_func(flex_model, baselines, device,
                                     n_samples=args.n_samples, stdevs=args.stdevs)

    faithfulness, robustness = build_metrics(explain_func)
    excluded = {m.strip().lower() for m in args.exclude.split(",") if m.strip()}
    if excluded:
        for d in (faithfulness, robustness):
            for name in list(d):
                if name.lower() in excluded:
                    del d[name]
        print(f"[info] métricas excluídas: {sorted(excluded)}")
    all_metric_names = list(faithfulness) + list(robustness)

    attr_dir = Path(args.attr_dir)
    results_dir = Path(args.results_dir)
    per_class_results: dict[str, dict[str, float]] = {}

    def _save(verbose: bool = False) -> None:
        """Grava os CSVs a partir do estado atual (parcial ou final).

        Chamado após cada métrica, de modo que um encerramento abrupto do
        processo (ex.: SIGKILL por falta de memória numa métrica pesada) não
        descarte o que já foi calculado nas classes/métricas anteriores.
        """
        _save_metrics_csv(per_class_results, all_metric_names,
                          results_dir / "quantus_metrics.csv", verbose=verbose)
        _merge_with_f1(per_class_results, all_metric_names, results_dir,
                       verbose=verbose)

    for c, cls in enumerate(CLASSES):
        attr_path = attr_dir / f"{cls}.npz"
        if not attr_path.exists():
            print(f"[skip] {cls}: sem atribuições ({attr_path})")
            continue
        data = np.load(attr_path)
        a_batch = data["attributions"].astype(np.float32)
        test_indices = data["test_indices"]

        # limita nº de amostras por classe (custo do Quantus é alto)
        if args.max_per_class and len(a_batch) > args.max_per_class:
            a_batch = a_batch[: args.max_per_class]
            test_indices = test_indices[: args.max_per_class]

        x_batch = np.stack([test_ds.X[i] for i in test_indices]).astype(np.float32)
        y_batch = np.full(len(x_batch), c, dtype=np.int64)  # classe-alvo
        print(f"\n[classe {cls}] {len(x_batch)} amostras")

        # dict preenchido incrementalmente e persistido a cada métrica
        scores: dict[str, float] = {}
        per_class_results[cls] = scores
        run_all_metrics(
            flex_model, x_batch, y_batch, a_batch,
            faithfulness, robustness, explain_func, device,
            scores=scores, on_update=_save)

    _save(verbose=True)
    print("[ok] avaliação concluída")


def _atomic_write_csv(path: Path, header, rows) -> None:
    """Escreve o CSV de forma atômica (arquivo tmp + ``os.replace``).

    Assim nunca fica um CSV truncado caso o processo seja morto no meio de
    uma gravação — o destino ou tem a versão anterior íntegra ou a nova.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def _save_metrics_csv(results, metric_names, path: Path,
                      verbose: bool = True) -> None:
    header = ["classe"] + metric_names
    rows = [[cls] + [f"{results[cls].get(m, float('nan')):.6f}"
                     for m in metric_names]
            for cls in CLASSES if cls in results]
    _atomic_write_csv(path, header, rows)
    if verbose:
        print(f"\n[salvo] métricas Quantus -> {path}")


def _merge_with_f1(results, metric_names, results_dir: Path,
                   verbose: bool = True) -> None:
    """Combina métricas de qualidade com o F1 por classe (seção 4.5)."""
    f1_path = results_dir / "f1_scores.json"
    f1_test: dict[str, float] = {}
    if f1_path.exists():
        with open(f1_path) as f:
            data = json.load(f)
        f1_test = data.get("test_f1", data.get("val_f1", {}))
    elif verbose:
        print(f"[aviso] {f1_path} não encontrado; F1 ficará vazio no merge")

    out = results_dir / "quality_vs_f1.csv"
    header = ["classe", "f1"] + metric_names
    rows = [[cls, f"{f1_test.get(cls, ''):}"]
            + [f"{results[cls].get(m, float('nan')):.6f}" for m in metric_names]
            for cls in CLASSES if cls in results]
    _atomic_write_csv(out, header, rows)
    if verbose:
        print(f"[salvo] qualidade vs F1 -> {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Avaliação Quantus das explicações")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--max-per-class", type=int, default=50,
                   help="nº máximo de amostras por classe (custo do Quantus)")
    p.add_argument("--n-baselines", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--stdevs", type=float, default=0.09)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--exclude", type=str, default="",
                   help="métricas a pular, separadas por vírgula (ex.: Infidelity)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
