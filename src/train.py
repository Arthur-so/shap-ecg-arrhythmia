"""Treino da ResNet34 1D no CPSC2018 (seção 4.2 da proposta).

Hiperparâmetros e protocolo de Zhang et al. (2021) / código de referência
(github.com/onlyzdd/ecg-diagnosis):
- Otimizador Adam, taxa de aprendizado 1e-4, batch 32, até 40 épocas.
- Aumento de dados no treino (escala e deslocamento da linha de base).
- Limiar de decisão por classe, escolhido para maximizar o F1 na validação;
  os mesmos limiares são aplicados ao teste e salvos no checkpoint.
- Perda BCEWithLogitsLoss com ``pos_weight`` inversamente proporcional à
  frequência de cada classe no treino (seção 4.2 da proposta); com
  ``--no-pos-weight`` usa BCE simples, como no código de referência.
- Early stopping monitorando o F1-macro na validação (com os limiares
  otimizados); o melhor modelo é preservado.
- Ao final, salva o F1-score por classe (validação e teste), com os limiares
  otimizados e, para referência, com limiar fixo 0,5, em ``results/``.

Uso:
    python -m src.train \
        --data-dir data/processed \
        --out-dir checkpoints \
        --results-dir results \
        --epochs 40 --lr 1e-4 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CLASSES, SEED
from src.data.dataset import ECGDataset, compute_pos_weight
from src.device import get_device
from src.metrics import f1_report, macro_f1, optimal_thresholds
from src.model.resnet34_1d import build_model


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_probs(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Roda o modelo sobre um loader e devolve (y_true, probs)."""
    model.eval()
    all_true, all_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_prob.append(probs)
        all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_prob)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def train(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.out_dir) / "resnet34_1d_best.pt"
    if getattr(args, "skip_if_exists", False) and ckpt_path.exists():
        print(f"[skip] checkpoint já existe: {ckpt_path} — pulando treino")
        return
    set_seed(args.seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    train_ds = ECGDataset(data_dir / "train.npz", augment=not args.no_augment)
    val_ds = ECGDataset(data_dir / "val.npz")
    print(f"[info] treino: {len(train_ds)} | val: {len(val_ds)} | "
          f"lr={args.lr} | aumento de dados={'não' if args.no_augment else 'sim'} | "
          f"BCE {'simples' if args.no_pos_weight else 'ponderada (pos_weight)'}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(dropout=args.dropout).to(device)
    pos_weight = None if args.no_pos_weight else compute_pos_weight(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "resnet34_1d_best.pt"

    best_f1 = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, probs = evaluate_probs(model, val_loader, device)
        thresholds = _thresholds(args, y_true, probs)
        val_macro = macro_f1(y_true, (probs >= thresholds).astype(int))
        val_macro_05 = macro_f1(y_true, (probs >= 0.5).astype(int))
        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] loss={train_loss:.4f} "
              f"val_macroF1={val_macro:.4f} (limiar 0,5: {val_macro_05:.4f}) ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1": val_macro, "val_macro_f1_05": val_macro_05})

        if val_macro > best_f1:
            best_f1 = val_macro
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_macro_f1": val_macro,
                        "classes": CLASSES, "arch": model.arch,
                        "thresholds": thresholds.tolist()}, ckpt_path)
            print(f"          [*] melhor modelo salvo (F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stopping] sem melhora há {args.patience} épocas")
                break

    # --- relatório final por classe (carrega melhor checkpoint) ---
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    thresholds = np.asarray(ckpt["thresholds"])
    print("\n[limiares por classe (validação)] "
          + " ".join(f"{c}={th:.2f}" for c, th in zip(CLASSES, thresholds)))
    y_true, probs = evaluate_probs(model, val_loader, device)
    val_report = f1_report(y_true, probs, threshold=thresholds)
    _print_report("validação", val_report, f1_report(y_true, probs, threshold=0.5))

    results = {"config": {"epochs": args.epochs, "lr": args.lr,
                          "batch_size": args.batch_size, "patience": args.patience,
                          "dropout": args.dropout, "augment": not args.no_augment,
                          "pos_weight": not args.no_pos_weight,
                          "fixed_threshold": args.fixed_threshold, "arch": model.arch},
               "val_f1": val_report,
               "val_f1_limiar_05": f1_report(y_true, probs, threshold=0.5),
               "thresholds": dict(zip(CLASSES, map(float, thresholds))),
               "history": history,
               "best_epoch": ckpt["epoch"], "best_val_macro_f1": best_f1}

    # avalia no teste se existir (para a comparação com Quantus)
    test_path = data_dir / "test.npz"
    if test_path.exists():
        test_ds = ECGDataset(test_path)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        y_true_t, probs_t = evaluate_probs(model, test_loader, device)
        test_report = f1_report(y_true_t, probs_t, threshold=thresholds)
        test_05 = f1_report(y_true_t, probs_t, threshold=0.5)
        results["test_f1"] = test_report
        results["test_f1_limiar_05"] = test_05
        _print_report("teste", test_report, test_05)

    out_json = results_dir / "f1_scores.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[salvo] métricas -> {out_json}")

    # CSV enxuto por classe (usado por evaluate.py)
    _save_f1_csv(results, results_dir / "f1_per_class.csv")


def _thresholds(args: argparse.Namespace, y_true: np.ndarray,
                probs: np.ndarray) -> np.ndarray:
    """Limiares por classe: otimizados na validação, ou fixos se pedido."""
    if args.fixed_threshold is not None:
        return np.full(y_true.shape[1], args.fixed_threshold)
    return optimal_thresholds(y_true, probs)


def _print_report(name: str, report: dict, report_05: dict) -> None:
    print(f"\n[F1 por classe — {name}]   (limiares por classe | limiar 0,5)")
    for cls in CLASSES + ["macro"]:
        print(f"  {cls:>5}: {report[cls]:.4f} | {report_05[cls]:.4f}")


def _save_f1_csv(results: dict, path: Path) -> None:
    import csv

    val = results.get("val_f1", {})
    test = results.get("test_f1", {})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["classe", "f1_val", "f1_test"])
        for cls in CLASSES:
            writer.writerow([cls, f"{val.get(cls, ''):}", f"{test.get(cls, ''):}"])
    print(f"[salvo] F1 por classe -> {path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Treino da ResNet34 1D (CPSC2018)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10,
                   help="épocas sem melhora no F1-macro antes de parar")
    p.add_argument("--fixed-threshold", type=float, default=None,
                   help="usa um limiar fixo (ex.: 0.5) em vez de otimizar por classe")
    p.add_argument("--no-pos-weight", action="store_true",
                   help="BCE sem pesos por classe (código de referência de Zhang et al.)")
    p.add_argument("--no-augment", action="store_true",
                   help="desliga o aumento de dados no treino")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-if-exists", action="store_true",
                   help="pula o treino se o checkpoint já existir em --out-dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
