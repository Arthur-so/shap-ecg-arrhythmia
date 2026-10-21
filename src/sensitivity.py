"""Max-Sensitivity e Avg-Sensitivity das explicações GradientSHAP.

Max-Sensitivity: Yeh et al. (2019), Definição 3.1. Avg-Sensitivity: Bhatt,
Weller & Moura (2020), Definição 2. Substitui ``quantus.MaxSensitivity`` e
``quantus.AvgSensitivity``, que no TCC comparavam o mapa salvo (50 amostras do
GradientSHAP) com mapas refeitos com 20 amostras e outra semente, somavam ruído
também ao preenchimento e não descartavam perturbações que mudam a predição
(ver fichamento de Yeh et al., 2019).

Protocolo, para cada mapa (ECG × classe explicada):
  1. Explicação de referência refeita com a configuração de ``explain.py``
     (mesmas baselines do treino, ``n_samples`` e ``stdevs``) e uma semente
     fixa por mapa.
  2. ``--n-perturb`` perturbações: ruído uniforme U(−r, r) somado a cada ponto
     do trecho real do ECG (vizinhança ℓ∞ de raio r, como em Yeh et al.); o
     preenchimento não é tocado.
  3. Cada ECG perturbado é explicado com a MESMA semente da referência
     (mesmos sorteios de baselines e ruído do GradientSHAP), de modo que a
     diferença entre os mapas reflete só a perturbação.
  4. Perturbações que tiram a classe explicada da predição (probabilidade
     abaixo do limiar da classe) são descartadas, como em Bhatt et al.
  5. Distância ℓ2 entre explicações normalizadas para norma 1 (Yeh et al.,
     Bhatt et al.), calculada nas 36 regiões do batimento (principal, como nas
     demais métricas) e ponto a ponto (referência).
  6. Max = maior distância; Avg = média das distâncias, sobre as perturbações
     válidas do mapa.
  7. Piso: distância entre a referência e outra explicação do mesmo ECG, sem
     perturbação e com outra semente — a variação própria do GradientSHAP.

Ao final salva:
  - ``results/sensitivity.csv``: médias por classe (Max, Avg, piso, descartes).
  - ``results/sensitivity_mapas.csv``: valores de cada mapa.

Uso:
    python -m src.sensitivity \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --attr-dir results/attributions \
        --results-dir results
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
from src.consistency import _real_length
from src.faithfulness_correlation import region_index

DISTANCES = ["regioes", "pontos"]


# --------------------------------------------------------------------------- #
# Núcleo
# --------------------------------------------------------------------------- #
def unit_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância ℓ2 entre ``a`` e ``b`` normalizados para norma 1."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.linalg.norm(a / na - b / nb))


def region_vector(attr: np.ndarray, regions: list[tuple]) -> np.ndarray:
    """Soma do mapa em cada região -> vetor com uma importância por região."""
    return np.array([attr[r[0], r[1]].sum() for r in regions])


def distances(a: np.ndarray, b: np.ndarray, regions: list[tuple]) -> dict[str, float]:
    return {"regioes": unit_distance(region_vector(a, regions), region_vector(b, regions)),
            "pontos": unit_distance(a.ravel(), b.ravel())}


def perturb(x: np.ndarray, length: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Soma ruído U(−r, r) a cada ponto do trecho real (preenchimento intacto)."""
    xp = x.copy()
    xp[:, :length] += rng.uniform(-radius, radius, size=(x.shape[0], length)).astype(x.dtype)
    return xp


def summarize(values: list[float]) -> tuple[float, float]:
    """(Max, Avg) das distâncias válidas; NaN se não houver nenhuma."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.max()), float(v.mean())


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    from captum.attr import GradientShap

    from src.data.dataset import ECGDataset
    from src.device import get_device
    from src.explain import sample_baselines
    from src.model.resnet34_1d import load_checkpoint_model

    device = get_device()
    model, ckpt = load_checkpoint_model(Path(args.checkpoint), device)
    thresholds = np.asarray(ckpt.get("thresholds", [0.5] * len(CLASSES)), dtype=np.float64)
    data_dir = Path(args.data_dir)
    baselines = sample_baselines(ECGDataset(data_dir / "train.npz"), args.n_baselines,
                                 device, args.seed)
    gradient_shap = GradientShap(model)
    test = np.load(data_dir / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None

    def explain(x: np.ndarray, c: int, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        xt = torch.from_numpy(x[None].astype(np.float32)).to(device).requires_grad_(True)
        attr = gradient_shap.attribute(xt, baselines=baselines, target=c,
                                       n_samples=args.n_samples, stdevs=args.stdevs)
        return attr.detach().cpu().numpy()[0].astype(np.float64)

    @torch.no_grad()
    def prob(x: np.ndarray, c: int) -> float:
        z = model(torch.from_numpy(x[None].astype(np.float32)).to(device))[0, c]
        return float(torch.sigmoid(z))

    rng = np.random.default_rng(args.seed)
    attr_dir = Path(args.attr_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_m = results_dir / "sensitivity_mapas.csv"
    cols = ["record_id", "classe", "n_validas", "n_descartadas"]
    for dname in DISTANCES:
        cols += [f"max_{dname}", f"avg_{dname}", f"piso_{dname}"]
    per_map: list[dict] = []
    fh = open(out_m, "w", newline="")  # escrito mapa a mapa (resultado parcial se cair)
    writer = csv.DictWriter(fh, fieldnames=cols)
    writer.writeheader()
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
        for ti, rid in items:
            ti = int(ti)
            x = np.asarray(X[ti], dtype=np.float32)
            length = int(lengths[ti]) if lengths is not None else _real_length(x)
            if ti not in region_cache:
                region_cache[ti] = region_index(x, length)
            regions = region_cache[ti]
            seed = args.seed * 100_003 + ti * 11 + c
            ref = explain(x, c, seed)
            floor = distances(ref, explain(x, c, seed + 1), regions)
            dists = {dname: [] for dname in DISTANCES}
            n_drop = 0
            for _ in range(args.n_perturb):
                xp = perturb(x, length, args.radius, rng)
                if prob(xp, c) < thresholds[c]:
                    n_drop += 1
                    continue
                d = distances(ref, explain(xp, c, seed), regions)
                for dname in DISTANCES:
                    dists[dname].append(d[dname])
            row = {"record_id": str(rid), "classe": cls,
                   "n_validas": args.n_perturb - n_drop, "n_descartadas": n_drop}
            for dname in DISTANCES:
                mx, avg = summarize(dists[dname])
                row.update({f"max_{dname}": mx, f"avg_{dname}": avg,
                            f"piso_{dname}": floor[dname]})
            per_map.append(row)
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            n_done += 1
            if n_done % args.log_every == 0 or n_done == n_total:
                el = time.time() - t_start
                print(f"[progresso] {n_done}/{n_total} mapas, {el / 60:.1f} min, "
                      f"restam ~{el / n_done * (n_total - n_done) / 60:.0f} min", flush=True)
        print(f"  {cls:>5}: {len(items)} mapas")
    fh.close()
    print(f"[salvo] {out_m}")

    rows = _aggregate(per_map)
    _print_table(rows)
    out = results_dir / "sensitivity.csv"
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
        row = {"classe": cls, "n": len(sel),
               "n_sem_validas": sum(r["n_validas"] == 0 for r in sel),
               "frac_descartadas": float(np.mean([r["n_descartadas"] / max(r["n_validas"]
                                                  + r["n_descartadas"], 1) for r in sel]))}
        for dname in DISTANCES:
            for kind in ("max", "avg", "piso"):
                v = np.array([r[f"{kind}_{dname}"] for r in sel], dtype=np.float64)
                v = v[np.isfinite(v)]
                row[f"{kind}_{dname}"] = float(v.mean()) if v.size else float("nan")
                row[f"{kind}_{dname}_desvio"] = float(v.std()) if v.size else float("nan")
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\nSensibilidade média por classe (distância ℓ2 entre explicações com norma 1)")
    print(f"{'classe':>7} {'n':>4}  {'Max reg':>8} {'Avg reg':>8} {'piso reg':>8}  "
          f"{'Max pts':>8} {'Avg pts':>8} {'piso pts':>8}  {'descart.':>8}")
    for r in rows:
        print(f"{r['classe']:>7} {r['n']:>4}  {r['max_regioes']:8.3f} {r['avg_regioes']:8.3f} "
              f"{r['piso_regioes']:8.3f}  {r['max_pontos']:8.3f} {r['avg_pontos']:8.3f} "
              f"{r['piso_pontos']:8.3f}  {r['frac_descartadas']:8.1%}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Max/Avg-Sensitivity (GradientSHAP, ECG)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions",
                   help="usado só para saber quais mapas (ECG × classe) avaliar")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--radius", type=float, default=0.1,
                   help="raio r do ruído uniforme U(-r, r), em unidades de z-score")
    p.add_argument("--n-perturb", type=int, default=10, help="perturbações por mapa")
    p.add_argument("--n-baselines", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=50)
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
