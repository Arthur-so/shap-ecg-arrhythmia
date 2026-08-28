"""Geração de explicações GradientSHAP via Captum (seção 4.3 da proposta).

O GradientSHAP é aplicado apenas às amostras corretamente classificadas do
conjunto de teste, em relação à classe predita, usando como distribuição de
referência (baseline) amostras extraídas aleatoriamente do conjunto de treino.

Para cada amostra e cada classe positiva corretamente predita (pred=1 e
rótulo verdadeiro=1) é gerado um mapa de atribuição de dimensão (12, 15000).
Os mapas são salvos organizados por classe em ``<out-dir>/<CLASSE>.npz``,
cada arquivo contendo:
    - ``attributions``: (M, 12, 15000) float32
    - ``test_indices``: (M,) índices na partição de teste
    - ``record_ids``:   (M,) identificadores dos registros

Uso:
    python -m src.explain \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --out-dir results/attributions \
        --n-baselines 32 --threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import CLASSES, SEED
from src.data.dataset import ECGDataset
from src.device import get_device
from src.model.resnet34_1d import build_model


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def sample_baselines(train_ds: ECGDataset, n: int, device: torch.device,
                     seed: int = SEED) -> torch.Tensor:
    """Amostra ``n`` sinais de treino como distribuição de referência."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_ds), size=min(n, len(train_ds)), replace=False)
    baselines = np.stack([train_ds.X[i] for i in idx]).astype(np.float32)
    return torch.from_numpy(baselines).to(device)


def generate_explanations(args: argparse.Namespace) -> None:
    from captum.attr import GradientShap

    device = get_device()

    data_dir = Path(args.data_dir)
    train_ds = ECGDataset(data_dir / "train.npz")
    test_ds = ECGDataset(data_dir / "test.npz")
    print(f"[info] teste: {len(test_ds)} amostras")

    model = load_model(Path(args.checkpoint), device)
    baselines = sample_baselines(train_ds, args.n_baselines, device, args.seed)
    print(f"[info] {baselines.shape[0]} baselines amostrados do treino")

    gradient_shap = GradientShap(model)

    # acumuladores por classe
    acc_attr: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    acc_idx: dict[str, list[int]] = {c: [] for c in CLASSES}
    acc_rec: dict[str, list[str]] = {c: [] for c in CLASSES}

    n_correct = 0
    for i in range(len(test_ds)):
        x = torch.from_numpy(np.asarray(test_ds.X[i], dtype=np.float32))
        x = x.unsqueeze(0).to(device)  # (1, 12, 15000)
        y_true = np.asarray(test_ds.Y[i])
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()[0]
        pred = (probs >= args.threshold).astype(int)

        # classes positivas corretamente preditas nesta amostra
        correct_classes = np.where((pred == 1) & (y_true == 1))[0]
        if len(correct_classes) == 0:
            continue
        n_correct += 1

        for c in correct_classes:
            x.requires_grad_(True)
            attr = gradient_shap.attribute(
                x, baselines=baselines, target=int(c),
                n_samples=args.n_samples, stdevs=args.stdevs,
            )
            attr_np = attr.detach().cpu().numpy()[0].astype(np.float32)
            cls = CLASSES[c]
            acc_attr[cls].append(attr_np)
            acc_idx[cls].append(i)
            acc_rec[cls].append(test_ds.get_record_id(i))

        if (i + 1) % args.log_every == 0:
            print(f"[progresso] {i + 1}/{len(test_ds)} amostras processadas")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[info] {n_correct} amostras com ao menos 1 classe correta")
    for cls in CLASSES:
        if not acc_attr[cls]:
            print(f"  {cls:>5}: 0 explicações (nenhuma amostra correta)")
            continue
        attrs = np.stack(acc_attr[cls]).astype(np.float32)
        np.savez_compressed(
            out_dir / f"{cls}.npz",
            attributions=attrs,
            test_indices=np.asarray(acc_idx[cls], dtype=np.int64),
            record_ids=np.asarray(acc_rec[cls]),
        )
        print(f"  {cls:>5}: {attrs.shape[0]} explicações -> {out_dir / f'{cls}.npz'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GradientSHAP via Captum (CPSC2018)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--out-dir", type=str, default="results/attributions")
    p.add_argument("--n-baselines", type=int, default=32,
                   help="nº de amostras de treino usadas como baseline")
    p.add_argument("--n-samples", type=int, default=50,
                   help="nº de amostras de ruído do GradientSHAP")
    p.add_argument("--stdevs", type=float, default=0.09,
                   help="desvio do ruído gaussiano do GradientSHAP")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    generate_explanations(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
