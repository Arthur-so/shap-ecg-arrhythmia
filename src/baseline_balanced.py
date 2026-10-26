"""Referência balanceada por classe para o GradientSHAP, com 200 sorteios.

Continuação de ``baseline_check.py``. Lá, a referência normal (32 ECGs SNR)
mudou os mapas moderadamente, mas tira o sentido da classe SNR; e a referência
atual (32 ECGs sorteados do treino) ficou desbalanceada (BRD 14, BRE 0, STE 0),
de modo que outro sorteio já muda os mapas. Aqui se testa uma referência
balanceada: o mesmo número de ECGs de cada uma das 9 classes (rótulo único,
treino), que não privilegia nenhuma classe e mantém SNR interpretável.

Para cada mapa (ECG × classe explicada), com as mesmas sementes de
``baseline_check.py`` (semente s do mapa; blocos de 50 sorteios com sementes
s + j·7919):
  - ``atual``          : 32 baselines de ``explain.py``, 50 sorteios;
  - ``atual200``       : idem, 200 sorteios (média de 4 blocos; o 1.º é ``atual``);
  - ``balanceada200``  : ``--per-class`` ECGs de cada classe, 200 sorteios;
  - ``balanceada200'`` : idem a partir da semente s+1 (piso com 200 sorteios).
Distâncias (ℓ2 entre mapas com norma 1, nas 36 regiões do batimento):
  - ``d_atual_200``         : atual × atual200        (só o efeito dos sorteios)
  - ``d_balanceada``        : atual × balanceada200   (referência e sorteios)
  - ``d_ref_200``           : atual200 × balanceada200 (só o efeito da referência, ambos com 200)
  - ``d_piso_balanceada200``: balanceada200 × balanceada200' (piso com 200 sorteios)
Também registra a soma de cada mapa e a região mais importante.

Com ``--save-dir``, salva os mapas ``atual200`` e ``balanceada200`` (formato de
``explain.py``) em subpastas, para recalcular as métricas sem gerar os mapas.

Ao final salva:
  - ``results/baseline_balanced.csv``: médias por classe.
  - ``results/baseline_balanced_mapas.csv``: valores de cada mapa.

Uso:
    python -m src.baseline_balanced \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --attr-dir results/attributions \
        --results-dir results \
        --save-dir results/attributions_200
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config import CLASSES, SEED
from src.consistency import LEADS, REGIONS, _real_length
from src.faithfulness_correlation import region_index
from src.gshap import CHUNK_STEP, balanced_indices, random_indices
from src.sensitivity import region_vector, unit_distance

PAIRS = [("atual_200", "atual", "atual200"),
         ("balanceada", "atual", "balanceada200"),
         ("ref_200", "atual200", "balanceada200"),
         ("piso_balanceada200", "balanceada200", "balanceada200'")]
SUMS = ["atual", "atual200", "balanceada200"]
SAVED = ["atual200", "balanceada200"]


def run(args: argparse.Namespace) -> None:
    from captum.attr import GradientShap

    from src.device import get_device
    from src.model.resnet34_1d import load_checkpoint_model

    device = get_device()
    model, _ = load_checkpoint_model(Path(args.checkpoint), device)
    data_dir = Path(args.data_dir)
    train = np.load(data_dir / "train.npz")
    Xtr, Ytr = train["X"], train["Y"]
    sets = {"atual": random_indices(len(Xtr), args.n_baselines, args.seed),
            "balanceada": balanced_indices(Ytr, args.per_class, args.seed)}
    for name, idx in sets.items():
        counts = Ytr[idx].sum(axis=0).astype(int)
        print(f"[baselines] {name:>10} ({len(idx)}): " +
              " ".join(f"{c}={k}" for c, k in zip(CLASSES, counts)))
    bases = {name: torch.from_numpy(Xtr[idx].astype(np.float32)).to(device)
             for name, idx in sets.items()}
    gradient_shap = GradientShap(model)
    test = np.load(data_dir / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None
    print(f"[info] blocos de {args.n_samples} sorteios; mapas de {args.n_samples * args.chunks} "
          f"sorteios = média de {args.chunks} blocos")

    def explain(x: np.ndarray, c: int, seed: int, base: str) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        xt = torch.from_numpy(x[None].astype(np.float32)).to(device).requires_grad_(True)
        attr = gradient_shap.attribute(xt, baselines=bases[base], target=c,
                                       n_samples=args.n_samples, stdevs=args.stdevs)
        return attr.detach().cpu().numpy()[0].astype(np.float64)

    def explain_more(x: np.ndarray, c: int, seed: int, base: str,
                     first: np.ndarray | None = None) -> np.ndarray:
        """Média de ``args.chunks`` blocos; ``first`` (opcional) é o bloco da semente ``seed``."""
        if first is None:
            first = explain(x, c, seed, base)
        blocks = [first] + [explain(x, c, seed + j * CHUNK_STEP, base) for j in range(1, args.chunks)]
        return np.mean(blocks, axis=0)

    attr_dir = Path(args.attr_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        for k in SAVED:
            (save_dir / k).mkdir(parents=True, exist_ok=True)
    out_m = results_dir / "baseline_balanced_mapas.csv"
    cols = ["record_id", "classe"] + [f"d_{p[0]}" for p in PAIRS] + \
           [f"soma_{k}" for k in SUMS] + [f"top1_{k}" for k in SUMS]
    fh = open(out_m, "w", newline="")  # escrito mapa a mapa (resultado parcial se cair)
    writer = csv.DictWriter(fh, fieldnames=cols)
    writer.writeheader()
    per_map: list[dict] = []
    region_cache: dict[int, list] = {}
    t_start, n_done = time.time(), 0
    n_total = sum(len(np.load(attr_dir / f"{c}.npz")["test_indices"])
                  for c in CLASSES if (attr_dir / f"{c}.npz").exists())
    if args.limit:
        n_total = min(n_total, args.limit * len(CLASSES))

    for c, cls in enumerate(CLASSES):
        path = attr_dir / f"{cls}.npz"
        if not path.exists():
            print(f"[skip] {cls}: sem atribuições ({path})")
            continue
        data = np.load(path)
        rids = data["record_ids"] if "record_ids" in data.files else data["test_indices"]
        items = list(zip(data["test_indices"], rids))
        if args.limit:
            items = items[: args.limit]
        kept = {k: [] for k in SAVED}
        kept_idx, kept_rids = [], []
        for ti, rid in items:
            ti = int(ti)
            x = np.asarray(X[ti], dtype=np.float32)
            if ti not in region_cache:
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                region_cache[ti] = region_index(x, length, with_labels=True)
            regions = region_cache[ti]
            seed = args.seed * 100_003 + ti * 11 + c  # mesma referência de sensitivity.py
            maps = {"atual": explain(x, c, seed, "atual")}
            maps["atual200"] = explain_more(x, c, seed, "atual", maps["atual"])
            maps["balanceada200"] = explain_more(x, c, seed, "balanceada")
            maps["balanceada200'"] = explain_more(x, c, seed + 1, "balanceada")
            g = {k: region_vector(m, regions) for k, m in maps.items()}
            row = {"record_id": str(rid), "classe": cls}
            for name, a, b in PAIRS:
                row[f"d_{name}"] = unit_distance(g[a], g[b])
            for k in SUMS:
                row[f"soma_{k}"] = float(maps[k].sum())
                lead, _, r = regions[int(np.argmax(np.abs(g[k])))]
                row[f"top1_{k}"] = f"{LEADS[lead]} {REGIONS[r]}"
            per_map.append(row)
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            if save_dir:
                for k in SAVED:
                    kept[k].append(maps[k].astype(np.float32))
                kept_idx.append(ti)
                kept_rids.append(str(rid))
            n_done += 1
            if n_done % args.log_every == 0 or n_done == n_total:
                el = time.time() - t_start
                print(f"[progresso] {n_done}/{n_total} mapas, {el / 60:.1f} min, "
                      f"restam ~{el / n_done * (n_total - n_done) / 60:.0f} min", flush=True)
        if save_dir and kept_idx:
            for k in SAVED:
                np.savez_compressed(save_dir / k / f"{cls}.npz", attributions=np.stack(kept[k]),
                                    test_indices=np.asarray(kept_idx, dtype=np.int64),
                                    record_ids=np.asarray(kept_rids))
        print(f"  {cls:>5}: {len(items)} mapas")
    fh.close()
    print(f"[salvo] {out_m}")
    if save_dir:
        print(f"[salvo] mapas {', '.join(SAVED)} em {save_dir}")

    rows = _aggregate(per_map)
    _print_table(rows)
    out = results_dir / "baseline_balanced.csv"
    with open(out, "w", newline="") as fh2:
        w = csv.DictWriter(fh2, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")


def _aggregate(per_map: list[dict]) -> list[dict]:
    rows = []
    for cls in CLASSES + ["global"]:
        sel = [r for r in per_map if cls == "global" or r["classe"] == cls]
        if not sel:
            continue
        row = {"classe": cls, "n": len(sel)}
        for s in [f"d_{p[0]}" for p in PAIRS] + [f"soma_{k}" for k in SUMS]:
            v = np.array([r[s] for r in sel], dtype=np.float64)
            v = v[np.isfinite(v)]
            row[f"{s}_media"] = float(v.mean()) if v.size else float("nan")
            row[f"{s}_desvio"] = float(v.std()) if v.size else float("nan")
        for k in ["atual200", "balanceada200"]:
            row[f"top1_igual_{k}"] = float(np.mean([r["top1_atual"] == r[f"top1_{k}"] for r in sel]))
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\nDistâncias médias (36 regiões, norma 1) e soma média do mapa")
    labels = ["at×at200", "at×bal200", "at200×bal", "piso bal"]
    print(f"{'classe':>7} {'n':>4} " + " ".join(f"{h:>9}" for h in labels) +
          f" {'soma at.':>9} {'soma bal.':>9}")
    for r in rows:
        print(f"{r['classe']:>7} {r['n']:4d} " +
              " ".join(f"{r[f'd_{p[0]}_media']:9.3f}" for p in PAIRS) +
              f" {r['soma_atual_media']:9.2f} {r['soma_balanceada200_media']:9.2f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Referência balanceada por classe (GradientSHAP, 200 sorteios)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions",
                   help="mapas salvos (só para saber quais ECG × classe avaliar)")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-dir", type=str, default="",
                   help="se informado, salva os mapas atual200 e balanceada200 em subpastas")
    p.add_argument("--n-baselines", type=int, default=32, help="tamanho da referência atual")
    p.add_argument("--per-class", type=int, default=4, help="ECGs de cada classe na referência balanceada")
    p.add_argument("--n-samples", type=int, default=50, help="sorteios por bloco")
    p.add_argument("--chunks", type=int, default=4, help="blocos por mapa (4 × 50 = 200)")
    p.add_argument("--stdevs", type=float, default=0.09)
    p.add_argument("--limit", type=int, default=0, help=">0: só os N primeiros mapas de cada classe")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
