"""Robustez dos limites das regiões do batimento (mapas da versão 2).

Recalcula, em 9 configurações de regiões (janela do QRS 140/180/220 ms ×
fim do ST-T em 40/45/50% do RR), duas métricas que não exigem rodar o modelo:
a Consistency (k = 1, 2, 3) e a complexidade (número efetivo de regiões).
A janela do QRS cresce 20 ms de cada lado: R − 60/+80, R − 80/+100 (base) e
R − 100/+120 ms. Uso::

    python -m src.regioes_robustez --attr-dir output/modeloB_balanceada200 \
        --out results/modeloB_v2/regioes_robustez.json
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

from src.complexity import entropy
from src.config import CLASSES
from src.consistency import (REF_LEAD, chance_level, detect_r_peaks, discretize,
                             local_consistency, region_mask, summarize)

QRS = {"140ms": (0.060, 0.080), "180ms": (0.080, 0.100), "220ms": (0.100, 0.120)}
ST_FRACS = (0.40, 0.45, 0.50)


def run(args: argparse.Namespace) -> None:
    test = np.load(Path(args.data_dir) / "test.npz")
    X, L = test["X"], test["lengths"]
    maps, preds, tis = [], [], []
    for c, cls in enumerate(CLASSES):
        d = np.load(Path(args.attr_dir) / f"{cls}.npz")
        for a, ti in zip(d["attributions"], d["test_indices"]):
            maps.append(a)
            preds.append(c)
            tis.append(int(ti))
    preds = np.asarray(preds)
    peaks = {ti: detect_r_peaks(X[ti][REF_LEAD, :int(L[ti])]) for ti in set(tis)}

    out = []
    for (qrs, (pre, post)), st in product(QRS.items(), ST_FRACS):
        S = [summarize(a.astype(np.float64),
                       region_mask(peaks[ti], int(L[ti]), a.shape[1],
                                   qrs_pre=pre, qrs_post=post, st_frac=st))
             for a, ti in zip(maps, tis)]
        nef = np.array([np.exp(entropy(s)) for s in S])
        row = {"qrs": qrs, "st": st,
               "n_efetivo": {cls: float(nef[preds == c].mean()) for c, cls in enumerate(CLASSES)},
               "n_efetivo_global": float(nef.mean())}
        for k in (1, 2, 3):
            e = [discretize(s, k) for s in S]
            loc = local_consistency(e, preds)
            ch = chance_level(e, preds, len(CLASSES))
            row[f"k{k}"] = {"global": float(loc.mean()),
                            "per_classe": {cls: float(loc[preds == c].mean())
                                           for c, cls in enumerate(CLASSES)},
                            "acaso": {cls: float(ch[c]) for c, cls in enumerate(CLASSES)}}
        out.append(row)
        print(f"QRS {qrs} ST {st:.2f}: Consistency k=2 {row['k2']['global']:.3f} | "
              f"n_efetivo {row['n_efetivo_global']:.1f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"[salvo] {args.out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robustez dos limites das regiões")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--attr-dir", type=str, default="output/modeloB_balanceada200")
    p.add_argument("--out", type=str, default="results/modeloB_v2/regioes_robustez.json")
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
