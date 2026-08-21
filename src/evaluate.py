"""Avaliação quantitativa das explicações com o toolkit Quantus (seção 4.4).

Calcula, por amostra e depois agregado por classe diagnóstica, as métricas de:

  Faithfulness: Faithfulness Correlation, Faithfulness Estimate, Selectivity,
                SensitivityN, Infidelity, Sufficiency.
  Robustness:   Local Lipschitz Estimate, Max-Sensitivity, Avg-Sensitivity,
                Continuity, Consistency, Relative Input Stability (RIS),
                Relative Output Stability (ROS).

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
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import CLASSES, SEED
from src.data.dataset import ECGDataset
from src.model.resnet34_1d import build_model


# --------------------------------------------------------------------------- #
# Modelo e função de explicação (para métricas de robustez)
# --------------------------------------------------------------------------- #
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
        attr = gradient_shap.attribute(
            x, baselines=baselines, target=tgt,
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
        "FaithfulnessEstimate": quantus.FaithfulnessEstimate(
            features_in_step=224, disable_warnings=True),
        "Selectivity": quantus.Selectivity(
            patch_size=224, disable_warnings=True),
        "SensitivityN": quantus.SensitivityN(
            features_in_step=224, n_max_percentage=0.8, disable_warnings=True),
        "Infidelity": quantus.Infidelity(
            perturb_baseline="uniform", n_perturb_samples=5,
            disable_warnings=True),
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
        "Consistency": quantus.Consistency(disable_warnings=True),
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


# --------------------------------------------------------------------------- #
# Avaliação por classe
# --------------------------------------------------------------------------- #
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] dispositivo: {device}")

    data_dir = Path(args.data_dir)
    train_ds = ECGDataset(data_dir / "train.npz")
    test_ds = ECGDataset(data_dir / "test.npz")
    model = load_model(Path(args.checkpoint), device)

    # baselines (mesma lógica de explain.py) para a explain_func da robustez
    rng = np.random.default_rng(args.seed)
    bidx = rng.choice(len(train_ds), size=min(args.n_baselines, len(train_ds)),
                      replace=False)
    baselines = torch.from_numpy(
        np.stack([train_ds.X[i] for i in bidx]).astype(np.float32)).to(device)
    explain_func = make_explain_func(model, baselines, device,
                                     n_samples=args.n_samples, stdevs=args.stdevs)

    faithfulness, robustness = build_metrics(explain_func)
    all_metric_names = list(faithfulness) + list(robustness)

    attr_dir = Path(args.attr_dir)
    per_class_results: dict[str, dict[str, float]] = {}

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

        scores: dict[str, float] = {}
        for name, metric in faithfulness.items():
            scores[name] = _run_metric(metric, model, x_batch, y_batch,
                                       a_batch, device)
            print(f"    {name:>26}: {scores[name]:.4f}")
        for name, metric in robustness.items():
            scores[name] = _run_metric(metric, model, x_batch, y_batch,
                                       a_batch, device, explain_func=explain_func)
            print(f"    {name:>26}: {scores[name]:.4f}")
        per_class_results[cls] = scores

    _save_metrics_csv(per_class_results, all_metric_names,
                      Path(args.results_dir) / "quantus_metrics.csv")
    _merge_with_f1(per_class_results, all_metric_names, Path(args.results_dir))


def _save_metrics_csv(results, metric_names, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["classe"] + metric_names)
        for cls in CLASSES:
            if cls not in results:
                continue
            row = [cls] + [f"{results[cls].get(m, float('nan')):.6f}"
                           for m in metric_names]
            writer.writerow(row)
    print(f"\n[salvo] métricas Quantus -> {path}")


def _merge_with_f1(results, metric_names, results_dir: Path) -> None:
    """Combina métricas de qualidade com o F1 por classe (seção 4.5)."""
    f1_path = results_dir / "f1_scores.json"
    f1_test: dict[str, float] = {}
    if f1_path.exists():
        with open(f1_path) as f:
            data = json.load(f)
        f1_test = data.get("test_f1", data.get("val_f1", {}))
    else:
        print(f"[aviso] {f1_path} não encontrado; F1 ficará vazio no merge")

    out = results_dir / "quality_vs_f1.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["classe", "f1"] + metric_names)
        for cls in CLASSES:
            if cls not in results:
                continue
            f1 = f1_test.get(cls, "")
            row = [cls, f"{f1:}"] + [f"{results[cls].get(m, float('nan')):.6f}"
                                     for m in metric_names]
            writer.writerow(row)
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
