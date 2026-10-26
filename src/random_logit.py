"""Random Logit das explicações GradientSHAP (Sixt, Granz & Landgraf, 2020).

O artigo (§3, p. 5) compara o mapa que explica o logit verdadeiro com o mapa
de um logit sorteado entre as demais classes: se forem quase iguais, a
explicação é insensível à classe. Substitui a ``quantus.RandomLogit``, que
compararia o mapa salvo com um mapa novo gerado com outros sorteios do
GradientSHAP, usaria SSIM (medida de imagens) sobre os 180 000 pontos com o
preenchimento e sortearia a outra classe sem distinguir as presentes no ECG
(ver fichamento de Sixt et al., 2020).

Protocolo, para cada mapa (ECG × classe explicada c):
  1. Mapa de referência de c refeito com a configuração de ``explain.py``
     (mesmas baselines do treino, ``n_samples`` e ``stdevs``) e a semente fixa
     por mapa de ``sensitivity.py`` (a mesma referência da sensibilidade).
  2. Mapa de cada uma das outras 8 classes c' no mesmo ECG, com a MESMA
     semente (mesmos sorteios de baselines e ruído), para que a diferença
     venha só da classe.
  3. Distância ℓ2 entre os mapas normalizados para norma 1, nas 36 regiões do
     batimento (principal) e ponto a ponto (referência), como na sensibilidade.
  4. Piso: distância entre a referência e outro mapa de c com outra semente.
     Uma distância entre classes só indica especificidade se passar do piso.
  5. As classes c' ausentes nos rótulos do ECG formam o resultado principal;
     as presentes (ECG com mais de um diagnóstico) são registradas à parte,
     pois podem usar legitimamente a mesma evidência.

Ao final salva:
  - ``results/random_logit.csv``: médias por classe explicada.
  - ``results/random_logit_mapas.csv``: valores de cada mapa.
  - ``results/random_logit_pares.csv``: cada par (mapa, outra classe).
  - ``results/random_logit_matriz.csv``: distância média nas regiões por par
    (classe explicada, outra classe ausente).

Uso:
    python -m src.random_logit \
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
from src.gshap import add_args, map_seed
from src.sensitivity import DISTANCES, distances


def _mean(values: list[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def summarize_map(pairs: list[dict], floor: float) -> dict:
    """Resumo de um mapa a partir dos seus pares (uma linha por outra classe)."""
    absent = [p for p in pairs if not p["presente"]]
    present = [p for p in pairs if p["presente"]]
    d_abs = [p["d_regioes"] for p in absent]
    row = {"n_ausentes": len(absent), "n_presentes": len(present),
           "d_ausentes": _mean(d_abs),
           "d_min_ausentes": float(np.min(d_abs)) if d_abs else float("nan"),
           "mais_proxima": absent[int(np.argmin(d_abs))]["outra"] if d_abs else "",
           "d_presentes": _mean([p["d_regioes"] for p in present]),
           "d_ausentes_pontos": _mean([p["d_pontos"] for p in absent]),
           "piso_regioes": floor}
    row["razao_piso"] = row["d_ausentes"] / floor if floor > 0 else float("nan")
    row["frac_acima_piso"] = (float(np.mean(np.asarray(d_abs) > floor)) if d_abs
                              else float("nan"))
    return row


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    from src.data.dataset import ECGDataset
    from src.device import get_device
    from src.gshap import Explainer
    from src.model.resnet34_1d import load_checkpoint_model

    device = get_device()
    model, _ = load_checkpoint_model(Path(args.checkpoint), device)
    data_dir = Path(args.data_dir)
    train = ECGDataset(data_dir / "train.npz")
    explain = Explainer(model, train.X, train.Y, device, reference=args.reference,
                        seed=args.seed, n_baselines=args.n_baselines, per_class=args.per_class,
                        n_samples=args.n_samples, chunks=args.chunks, stdevs=args.stdevs)
    print(f"[info] {explain.describe()}")
    test = np.load(data_dir / "test.npz")
    X, Y = test["X"], test["Y"]
    lengths = test["lengths"] if "lengths" in test.files else None

    @torch.no_grad()
    def logits(x: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(x[None].astype(np.float32)).to(device))[0].float().cpu().numpy()

    attr_dir = Path(args.attr_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pair_cols = ["record_id", "classe", "outra", "presente", "logit_outra"] + \
                [f"d_{d}" for d in DISTANCES]
    map_cols = ["record_id", "classe", "logit_classe", "n_ausentes", "n_presentes",
                "d_ausentes", "d_min_ausentes", "mais_proxima", "d_presentes",
                "d_ausentes_pontos", "piso_regioes", "piso_pontos", "razao_piso",
                "frac_acima_piso"]
    # escritos mapa a mapa (resultado parcial se cair)
    fh_p = open(results_dir / "random_logit_pares.csv", "w", newline="")
    fh_m = open(results_dir / "random_logit_mapas.csv", "w", newline="")
    w_p = csv.DictWriter(fh_p, fieldnames=pair_cols)
    w_m = csv.DictWriter(fh_m, fieldnames=map_cols)
    w_p.writeheader()
    w_m.writeheader()
    fmt = lambda r: {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()}

    per_map: list[dict] = []
    all_pairs: list[dict] = []
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
            if ti not in region_cache:
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                region_cache[ti] = region_index(x, length)
            regions = region_cache[ti]
            z = logits(x)
            seed = map_seed(args.seed, ti, c)  # mesma semente de explain.py e sensitivity.py
            ref = explain(x, c, seed)
            floor = distances(ref, explain(x, c, seed + 1), regions)
            pairs = []
            for k, other in enumerate(CLASSES):
                if k == c:
                    continue
                d = distances(ref, explain(x, k, seed), regions)
                pairs.append({"record_id": str(rid), "classe": cls, "outra": other,
                              "presente": int(Y[ti, k]), "logit_outra": float(z[k]),
                              **{f"d_{dn}": d[dn] for dn in DISTANCES}})
            row = {"record_id": str(rid), "classe": cls, "logit_classe": float(z[c]),
                   **summarize_map(pairs, floor["regioes"]), "piso_pontos": floor["pontos"]}
            per_map.append(row)
            all_pairs += pairs
            w_m.writerow(fmt({k: row[k] for k in map_cols}))
            w_p.writerows(fmt(p) for p in pairs)
            fh_m.flush()
            fh_p.flush()
            n_done += 1
            if n_done % args.log_every == 0 or n_done == n_total:
                el = time.time() - t_start
                print(f"[progresso] {n_done}/{n_total} mapas, {el / 60:.1f} min, "
                      f"restam ~{el / n_done * (n_total - n_done) / 60:.0f} min", flush=True)
        print(f"  {cls:>5}: {len(items)} mapas")
    fh_p.close()
    fh_m.close()
    print(f"[salvo] {results_dir / 'random_logit_pares.csv'}")
    print(f"[salvo] {results_dir / 'random_logit_mapas.csv'}")

    rows = _aggregate(per_map, all_pairs)
    _print_table(rows)
    out = results_dir / "random_logit.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")
    out_mx = results_dir / "random_logit_matriz.csv"
    with open(out_mx, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["classe"] + CLASSES)
        for cls in CLASSES:
            w.writerow([cls] + [f"{_mean([p['d_regioes'] for p in all_pairs if p['classe'] == cls and p['outra'] == o and not p['presente']]):.4f}"
                                for o in CLASSES])
    print(f"[salvo] {out_mx}")


def _aggregate(per_map: list[dict], pairs: list[dict]) -> list[dict]:
    rows = []
    for cls in CLASSES + ["global"]:
        sel = [r for r in per_map if cls == "global" or r["classe"] == cls]
        if not sel:
            continue
        row = {"classe": cls, "n": len(sel)}
        for s in ["d_ausentes", "d_min_ausentes", "piso_regioes", "razao_piso",
                  "frac_acima_piso", "d_ausentes_pontos", "piso_pontos"]:
            v = np.array([r[s] for r in sel], dtype=np.float64)
            v = v[np.isfinite(v)]
            row[f"{s}_media"] = float(v.mean()) if v.size else float("nan")
            row[f"{s}_desvio"] = float(v.std()) if v.size else float("nan")
        pres = [p["d_regioes"] for p in pairs
                if p["presente"] and (cls == "global" or p["classe"] == cls)]
        row["n_pares_presentes"] = len(pres)
        row["d_presentes_media"] = _mean(pres)
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\nRandom Logit por classe explicada (36 regiões; outras classes ausentes no ECG)")
    print(f"{'classe':>7} {'n':>4} {'dist.':>6} {'mín.':>6} {'piso':>6} {'razão':>6} "
          f"{'>piso':>6} {'pres.':>6}")
    for r in rows:
        print(f"{r['classe']:>7} {r['n']:4d} {r['d_ausentes_media']:6.3f} "
              f"{r['d_min_ausentes_media']:6.3f} {r['piso_regioes_media']:6.3f} "
              f"{r['razao_piso_media']:6.2f} {r['frac_acima_piso_media']:6.1%} "
              f"{r['d_presentes_media']:6.3f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random Logit (regiões do batimento)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions",
                   help="mapas salvos (só para saber quais ECG × classe avaliar)")
    p.add_argument("--results-dir", type=str, default="results")
    add_args(p)
    p.add_argument("--limit", type=int, default=0, help=">0: só os N primeiros mapas de cada classe")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
