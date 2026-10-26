"""Verificação da referência (baselines) e do número de sorteios do GradientSHAP.

O GradientSHAP explica a diferença entre a saída do modelo para o ECG e a
saída média para um conjunto de ECGs de referência (baselines). Em
``explain.py`` esse conjunto são 32 ECGs sorteados do treino (semente do
projeto), uma mistura de classes em que BRD ficou super-representado e BRE e
STE ausentes. Clinicamente, a referência natural seria o ECG normal: o mapa
diria o que, em relação a um coração saudável, levou o modelo à classe. Além
disso, o mapa é a média de só 50 sorteios, e o erro dessa média (o piso) é
grande frente às diferenças medidas pelas métricas.

Este módulo mede, para cada mapa (ECG × classe explicada), quanto o mapa muda
com a referência e quanto o erro cai com mais sorteios. Mapas comparados
(semente s do mapa, a mesma de ``sensitivity.py``):
  - ``atual``       : 32 baselines de ``explain.py``, 50 sorteios, semente s;
  - ``atual'``      : idem com semente s+1;
  - ``normal``      : 32 ECGs do treino com rótulo único SNR, 50 sorteios, semente s;
  - ``normal'``     : idem com semente s+1;
  - ``aleatoria``   : outros 32 ECGs sorteados do treino (controle), 50 sorteios, semente s;
  - ``normal200``   : referência normal, 200 sorteios (média de 4 blocos de 50; o 1.º é ``normal``);
  - ``normal200'``  : idem a partir de ``normal'``.
Os 200 sorteios são feitos como média de 4 execuções de 50 com sementes
diferentes, o que equivale a uma média de 200 sorteios e cabe na memória da GPU.

Distâncias (ℓ2 entre mapas com norma 1, nas 36 regiões do batimento):
  - ``d_piso``           : atual × atual'          (piso atual, 50 sorteios)
  - ``d_normal``         : atual × normal          (efeito da referência normal)
  - ``d_aleatoria``      : atual × aleatoria       (controle: só trocar o conjunto)
  - ``d_normal200``      : atual × normal200       (efeito da referência, com o mapa normal mais preciso)
  - ``d_piso_normal``    : normal × normal'        (piso com a referência normal, 50 sorteios)
  - ``d_piso_normal200`` : normal200 × normal200'  (piso com a referência normal, 200 sorteios)
  - ``d_normal_50_200``  : normal × normal200      (quanto o mapa de 50 se afasta do de 200)
Também registra a soma de cada mapa (que acompanha a saída do ECG menos a saída
média das baselines) e a região mais importante.

Com ``--save-dir``, salva os mapas ``normal200`` no mesmo formato de
``explain.py`` (um ``<CLASSE>.npz`` por classe), para recalcular as métricas
sem gerar os mapas de novo.

Ao final salva:
  - ``results/baseline_check.csv``: médias por classe.
  - ``results/baseline_check_mapas.csv``: valores de cada mapa.

Uso:
    python -m src.baseline_check \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --attr-dir results/attributions \
        --results-dir results \
        --save-dir results/attributions_normal200
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
from src.gshap import CHUNK_STEP, normal_indices, random_indices
from src.sensitivity import region_vector, unit_distance

# (nome, mapa 1, mapa 2)
PAIRS = [("piso", "atual", "atual'"),
         ("normal", "atual", "normal"),
         ("aleatoria", "atual", "aleatoria"),
         ("normal200", "atual", "normal200"),
         ("piso_normal", "normal", "normal'"),
         ("piso_normal200", "normal200", "normal200'"),
         ("normal_50_200", "normal", "normal200")]
SUMS = ["atual", "normal", "normal200", "aleatoria"]
TOPS = ["atual", "normal", "normal200"]


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
            "normal": normal_indices(Ytr, args.n_baselines, args.seed),
            "aleatoria": random_indices(len(Xtr), args.n_baselines, args.seed + 1)}
    for name, idx in sets.items():
        counts = Ytr[idx].sum(axis=0).astype(int)
        print(f"[baselines] {name:>9}: " + " ".join(f"{c}={k}" for c, k in zip(CLASSES, counts)))
    bases = {name: torch.from_numpy(Xtr[idx].astype(np.float32)).to(device)
             for name, idx in sets.items()}
    gradient_shap = GradientShap(model)
    test = np.load(data_dir / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None
    print(f"[info] {args.n_samples} sorteios por mapa; mapa normal mais preciso com "
          f"{args.n_samples * args.chunks} ({args.chunks} blocos)")

    def explain(x: np.ndarray, c: int, seed: int, base: str) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        xt = torch.from_numpy(x[None].astype(np.float32)).to(device).requires_grad_(True)
        attr = gradient_shap.attribute(xt, baselines=bases[base], target=c,
                                       n_samples=args.n_samples, stdevs=args.stdevs)
        return attr.detach().cpu().numpy()[0].astype(np.float64)

    def explain_more(x: np.ndarray, c: int, seed: int, base: str, first: np.ndarray) -> np.ndarray:
        """Média de ``args.chunks`` blocos de sorteios; ``first`` é o bloco da semente ``seed``."""
        blocks = [first] + [explain(x, c, seed + j * CHUNK_STEP, base) for j in range(1, args.chunks)]
        return np.mean(blocks, axis=0)

    attr_dir = Path(args.attr_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    out_m = results_dir / "baseline_check_mapas.csv"
    cols = ["record_id", "classe"] + [f"d_{p[0]}" for p in PAIRS] + \
           [f"soma_{k}" for k in SUMS] + [f"top1_{k}" for k in TOPS]
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
        kept_maps, kept_idx, kept_rids = [], [], []
        for ti, rid in items:
            ti = int(ti)
            x = np.asarray(X[ti], dtype=np.float32)
            if ti not in region_cache:
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                region_cache[ti] = region_index(x, length, with_labels=True)
            regions = region_cache[ti]
            seed = args.seed * 100_003 + ti * 11 + c  # mesma referência de sensitivity.py
            maps = {"atual": explain(x, c, seed, "atual"),
                    "atual'": explain(x, c, seed + 1, "atual"),
                    "normal": explain(x, c, seed, "normal"),
                    "normal'": explain(x, c, seed + 1, "normal"),
                    "aleatoria": explain(x, c, seed, "aleatoria")}
            maps["normal200"] = explain_more(x, c, seed, "normal", maps["normal"])
            maps["normal200'"] = explain_more(x, c, seed + 1, "normal", maps["normal'"])
            g = {k: region_vector(m, regions) for k, m in maps.items()}
            row = {"record_id": str(rid), "classe": cls}
            for name, a, b in PAIRS:
                row[f"d_{name}"] = unit_distance(g[a], g[b])
            for k in SUMS:
                row[f"soma_{k}"] = float(maps[k].sum())
            for k in TOPS:
                lead, _, r = regions[int(np.argmax(np.abs(g[k])))]
                row[f"top1_{k}"] = f"{LEADS[lead]} {REGIONS[r]}"
            per_map.append(row)
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            if save_dir:
                kept_maps.append(maps["normal200"].astype(np.float32))
                kept_idx.append(ti)
                kept_rids.append(str(rid))
            n_done += 1
            if n_done % args.log_every == 0 or n_done == n_total:
                el = time.time() - t_start
                print(f"[progresso] {n_done}/{n_total} mapas, {el / 60:.1f} min, "
                      f"restam ~{el / n_done * (n_total - n_done) / 60:.0f} min", flush=True)
        if save_dir and kept_maps:
            np.savez_compressed(save_dir / f"{cls}.npz", attributions=np.stack(kept_maps),
                                test_indices=np.asarray(kept_idx, dtype=np.int64),
                                record_ids=np.asarray(kept_rids))
        print(f"  {cls:>5}: {len(items)} mapas")
    fh.close()
    print(f"[salvo] {out_m}")
    if save_dir:
        print(f"[salvo] mapas normal200 em {save_dir}")

    rows = _aggregate(per_map)
    _print_table(rows)
    out = results_dir / "baseline_check.csv"
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
        for k in ["normal", "normal200"]:
            row[f"top1_igual_{k}"] = float(np.mean([r["top1_atual"] == r[f"top1_{k}"] for r in sel]))
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\nDistâncias médias (36 regiões, norma 1) e soma média do mapa")
    head = ["piso", "normal", "aleatoria", "normal200", "piso_normal", "piso_normal200",
            "normal_50_200"]
    labels = ["piso", "normal", "aleat.", "normal200", "piso nor.", "piso n200", "n50×n200"]
    print(f"{'classe':>7} {'n':>4} " + " ".join(f"{h:>9}" for h in labels) +
          f" {'soma at.':>9} {'soma n200':>9}")
    for r in rows:
        print(f"{r['classe']:>7} {r['n']:4d} " +
              " ".join(f"{r[f'd_{h}_media']:9.3f}" for h in head) +
              f" {r['soma_atual_media']:9.2f} {r['soma_normal200_media']:9.2f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verificação da referência e dos sorteios do GradientSHAP")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions",
                   help="mapas salvos (só para saber quais ECG × classe avaliar)")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-dir", type=str, default="",
                   help="se informado, salva os mapas normal200 (formato de explain.py)")
    p.add_argument("--n-baselines", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=50, help="sorteios por bloco")
    p.add_argument("--chunks", type=int, default=4,
                   help="blocos de --n-samples no mapa normal mais preciso (4 × 50 = 200)")
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
