"""Faithfulness Correlation das explicações GradientSHAP (Bhatt, Weller & Moura, 2020).

Implementa a Definição 3 do artigo — correlação de Pearson, para cada exemplo,
entre a soma das atribuições de subconjuntos aleatórios de atributos e a queda
na saída do modelo (logit) quando esses atributos são substituídos por uma
baseline — com atributos adequados ao ECG. Substitui a
``quantus.FaithfulnessCorrelation``, cuja configuração padrão (224 pontos
isolados entre 180 000, substituídos pelo mínimo do próprio sinal) gerava
artefatos e correlação nula (ver fichamento de Bhatt et al., 2020).

Protocolo:
  1. Atributos = 36 regiões do batimento (12 derivações × pré-QRS/QRS/ST-T),
     as mesmas de ``consistency.py``. A importância de uma região é a soma do
     mapa SHAP nela; retirá-la é substituir todos os seus pontos (em todos os
     batimentos) pela baseline. Batimentos incompletos e o preenchimento
     (padding) não são sorteados.
  2. Para cada tamanho |S| e cada exemplo: sorteiam-se ``--runs`` subconjuntos
     de |S| regiões; para cada um, soma das atribuições × queda
     f(x) − f(x_[S = baseline]) do logit da classe explicada.
  3. Correlação de Pearson por exemplo; média por classe. Os mesmos
     subconjuntos são usados nas duas baselines.
  4. Baselines do artigo (§7): ``zero`` (em z-score, a média de cada
     derivação) e ``mean`` (média de cada ponto derivação × instante no treino).

Ao final salva:
  - ``results/faithfulness_correlation.csv``: média por classe, baseline e |S|.
  - ``results/faithfulness_correlation_mapas.csv``: correlação de cada mapa.

Uso:
    python -m src.faithfulness_correlation \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --attr-dir results/attributions \
        --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.config import CLASSES, SEED
from src.consistency import REF_LEAD, REGIONS, _real_length, detect_r_peaks, region_mask


# --------------------------------------------------------------------------- #
# Atributos (regiões) e núcleo da métrica
# --------------------------------------------------------------------------- #
def region_index(x: np.ndarray, length: int) -> list[tuple[int, np.ndarray]]:
    """Lista de atributos (derivação, índices de tempo) com ao menos um ponto."""
    mask = region_mask(detect_r_peaks(x[REF_LEAD, :length]), length, x.shape[1])
    regions = []
    for lead in range(x.shape[0]):
        for r in range(len(REGIONS)):
            t = np.flatnonzero(mask == r)
            if t.size:
                regions.append((lead, t))
    return regions


def sample_subsets(n_regions: int, size: int, runs: int,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """``runs`` subconjuntos de ``size`` regiões, sem reposição dentro de cada um."""
    return [rng.choice(n_regions, size, replace=False) for _ in range(runs)]


def faithfulness_correlation(x: np.ndarray, attr: np.ndarray,
                             regions: list[tuple[int, np.ndarray]],
                             subsets: list[np.ndarray], baseline: np.ndarray,
                             predict: Callable[[np.ndarray], np.ndarray],
                             batch_size: int = 64) -> float:
    """Definição 3 de Bhatt et al. (2020) para um exemplo.

    ``predict`` recebe um lote (N, 12, T) e devolve o logit da classe
    explicada (N,). Retorna NaN se as somas ou as quedas forem constantes.
    """
    sums = np.array([sum(attr[regions[k][0], regions[k][1]].sum() for k in s)
                     for s in subsets])
    f_x = float(predict(x[None])[0])
    drops = []
    for i in range(0, len(subsets), batch_size):
        batch = np.repeat(x[None], len(subsets[i:i + batch_size]), axis=0)
        for b, s in enumerate(subsets[i:i + batch_size]):
            for k in s:
                lead, t = regions[k]
                batch[b, lead, t] = baseline[lead, t]
        drops.append(f_x - predict(batch))
    drops = np.concatenate(drops)
    if np.std(sums) == 0 or np.std(drops) == 0:
        return float("nan")
    return float(np.corrcoef(sums, drops)[0, 1])


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _train_mean(data_dir: Path, x_shape: tuple[int, ...]) -> np.ndarray:
    """Média de cada ponto (derivação × instante) no conjunto de treino."""
    X = np.load(data_dir / "train.npz")["X"]
    mean = np.zeros(x_shape, dtype=np.float64)
    for i in range(0, len(X), 256):
        mean += X[i:i + 256].astype(np.float64).sum(axis=0)
    return (mean / len(X)).astype(np.float32)


def run(args: argparse.Namespace) -> None:
    from src.device import get_device
    from src.model.resnet34_1d import load_checkpoint_model

    device = get_device()
    model, _ = load_checkpoint_model(Path(args.checkpoint), device)
    data_dir = Path(args.data_dir)
    test = np.load(data_dir / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None

    baselines = {}
    if "zero" in args.baselines:
        baselines["zero"] = np.zeros(X.shape[1:], dtype=np.float32)
    if "mean" in args.baselines:
        print("[info] calculando a média do treino (baseline 'mean')")
        baselines["mean"] = _train_mean(data_dir, X.shape[1:])

    rng = np.random.default_rng(args.seed)
    region_cache: dict[int, list] = {}
    per_map = []  # (record_id, classe, baseline, |S|, correlação)
    attr_dir = Path(args.attr_dir)
    for c, cls in enumerate(CLASSES):
        path = attr_dir / f"{cls}.npz"
        if not path.exists():
            print(f"[skip] {cls}: sem atribuições ({path})")
            continue
        data = np.load(path)
        rids = data["record_ids"] if "record_ids" in data.files else data["test_indices"]

        @torch.no_grad()
        def predict(batch: np.ndarray, c: int = c) -> np.ndarray:
            inp = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(device)
            return model(inp)[:, c].float().cpu().numpy()

        for a, ti, rid in zip(data["attributions"], data["test_indices"], rids):
            ti = int(ti)
            x = np.asarray(X[ti], dtype=np.float32)
            if ti not in region_cache:
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                region_cache[ti] = region_index(x, length)
            regions = region_cache[ti]
            a = np.asarray(a, dtype=np.float64)
            for size in args.subset_sizes:
                if len(regions) < size:
                    for b in baselines:
                        per_map.append((str(rid), cls, b, size, float("nan")))
                    continue
                subsets = sample_subsets(len(regions), size, args.runs, rng)
                for b, base in baselines.items():
                    corr = faithfulness_correlation(x, a, regions, subsets, base, predict,
                                                    batch_size=args.batch_size)
                    per_map.append((str(rid), cls, b, size, corr))
        print(f"  {cls:>5}: {len(data['attributions'])} mapas")

    rows = _aggregate(per_map, list(baselines), args.subset_sizes)
    _print_table(rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "faithfulness_correlation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")
    out_m = results_dir / "faithfulness_correlation_mapas.csv"
    with open(out_m, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "classe", "baseline", "subset_size", "faithfulness_correlation"])
        for rid, cls, b, size, corr in per_map:
            w.writerow([rid, cls, b, size, f"{corr:.4f}"])
    print(f"[salvo] {out_m}")


def _aggregate(per_map: list, baselines: list[str], sizes: list[int]) -> list[dict]:
    rows = []
    for b in baselines:
        for size in sizes:
            for cls in CLASSES + ["global"]:
                v = np.array([r[4] for r in per_map
                              if r[2] == b and r[3] == size and (cls == "global" or r[1] == cls)])
                if v.size == 0:
                    continue
                ok = v[np.isfinite(v)]
                rows.append({"baseline": b, "subset_size": size, "classe": cls,
                             "n": int(v.size), "n_nan": int(v.size - ok.size),
                             "media": float(ok.mean()) if ok.size else float("nan"),
                             "desvio": float(ok.std()) if ok.size else float("nan")})
    return rows


def _print_table(rows: list[dict]) -> None:
    for b in dict.fromkeys(r["baseline"] for r in rows):
        sizes = list(dict.fromkeys(r["subset_size"] for r in rows if r["baseline"] == b))
        print(f"\n[baseline {b}] Faithfulness Correlation média por classe")
        print(f"{'classe':>7} " + " ".join(f"{'|S|=' + str(s):>9}" for s in sizes))
        for cls in CLASSES + ["global"]:
            vals = [next((r for r in rows if r["baseline"] == b and r["subset_size"] == s
                          and r["classe"] == cls), None) for s in sizes]
            if all(v is None for v in vals):
                continue
            print(f"{cls:>7} " + " ".join(f"{v['media']:9.3f}" if v else f"{'-':>9}" for v in vals))
    nan = sum(r["n_nan"] for r in rows if r["classe"] == "global")
    print(f"\n[info] valores indefinidos (NaN) em todas as configurações: {nan}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Faithfulness Correlation (regiões do batimento)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--subset-sizes", type=int, nargs="+", default=[4, 9, 18],
                   help="tamanhos |S| (nº de regiões de 36 retiradas de uma vez)")
    p.add_argument("--runs", type=int, default=50, help="subconjuntos sorteados por exemplo")
    p.add_argument("--baselines", nargs="+", default=["zero", "mean"], choices=["zero", "mean"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
