"""eMPRT das explicações GradientSHAP (Hedström et al., 2023, Definição 3).

O Efficient Model Parameter Randomisation Test compara a explicação do modelo
treinado (e) com a do modelo de pesos totalmente sorteados (ê) pelo aumento
relativo da complexidade, q = (ξ(ê) − ξ(e)) / ξ(e). Se o mapa descreve o que o
modelo aprendeu, o modelo aleatório deve dar um mapa mais espalhado (q > 0).
Substitui a ``quantus.EfficientMPRT``, que calcula a entropia de um histograma
dos 180 000 pontos com o preenchimento (ver documento de seleção de métricas).

Protocolo, para cada mapa (ECG × classe explicada c):
  1. e = mapa salvo da versão 2 (mesma configuração de ``src/gshap.py``). O
     primeiro mapa de cada classe é refeito no modelo treinado para confirmar
     que reproduz o salvo (na GPU; a CPU sorteia outros números).
  2. Para cada semente de randomização: cópia da ResNet34 com os pesos de
     todas as camadas sorteados de novo (``reset_parameters``, como no
     Quantus); ê refeito com a MESMA semente do mapa e a mesma referência, de
     modo que só o modelo muda.
  3. ξ = entropia nas 36 regiões do batimento (``complexity.entropy``); q e o
     teto do mapa, (ln n − ξ(e)) / ξ(e), que limita q.
  4. Complementos:
       - distância ℓ2 entre e e ê normalizados, com sinal (MPRT original) e só
         com as magnitudes (Adebayo et al., 2018, §3.1: a randomização pode
         mudar o sinal e manter as regiões de maior magnitude), ambas
         comparadas com o piso do GradientSHAP (``sensitivity_mapas.csv``);
       - correlação de Spearman, entre as regiões, da |importância| com a
         amplitude do sinal em relação à referência, média de Σ|x − b|
         (Adebayo et al., §5.2: entrada × gradiente tende a devolver a entrada);
       - se as 2 regiões principais são QRS e se a principal coincide;
       - verificação de aderência: q com a entropia do artigo (Equação 4,
         histograma de 100 faixas dos valores do mapa, com sinal), calculada
         nos pontos do trecho real do ECG, sem o preenchimento.

Ao final salva:
  - ``results/emprt.csv``: médias por classe explicada.
  - ``results/emprt_mapas.csv``: valores de cada mapa (média das sementes).
  - ``results/emprt_modelos.csv``: cada par (mapa, semente de randomização).
  - ``results/emprt_regioes.npz``: vetores das 36 regiões (e, ê e amplitude),
    para refazer qualquer análise sem GPU.

Uso:
    python -m src.emprt \
        --data-dir data/processed \
        --checkpoint checkpoints/resnet34_1d_best.pt \
        --attr-dir results/attributions \
        --results-dir results \
        --sens-mapas results/sensitivity_mapas.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.complexity import _load_floor, entropy
from src.config import CLASSES, SEED
from src.consistency import REGIONS, _real_length
from src.faithfulness_correlation import region_index
from src.gshap import add_args, map_seed
from src.sensitivity import region_vector, unit_distance

QRS = REGIONS.index("QRS")
N_SLOTS = 12 * len(REGIONS)  # posição fixa de cada região no .npz
STATS = ["entropia_e", "entropia_r", "q", "teto", "q_teto", "d_sinal", "razao_sinal",
         "acima_piso_sinal", "d_abs", "razao_abs", "rho_amp_e", "rho_amp_r",
         "top2_qrs_e", "top2_qrs_r", "top1_igual", "hist_e", "hist_r", "q_hist"]
HIST_BINS = 100  # B do artigo (Suplemento 6.1)


# --------------------------------------------------------------------------- #
# Núcleo
# --------------------------------------------------------------------------- #
def randomize(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Cópia do modelo com os pesos de todas as camadas sorteados de novo."""
    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    for m in rnd.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return rnd.eval()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def amplitude(x: np.ndarray, baselines: np.ndarray, regions: list[tuple]) -> np.ndarray:
    """Média, entre os ECGs de referência, de Σ|x − b| em cada região."""
    return np.array([np.abs(x[lead, t][None] - baselines[:, lead, t]).sum(axis=1).mean()
                     for lead, t, _ in regions])


def hist_entropy(values: np.ndarray, bins: int = HIST_BINS) -> float:
    """Equação 4 do artigo: entropia do histograma dos valores do mapa (com sinal)."""
    c, _ = np.histogram(values, bins=bins)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


def top2_qrs(g: np.ndarray, labels: np.ndarray) -> float:
    return float(np.all(labels[np.argsort(np.abs(g))[-2:]] == QRS))


def compare(g_e: np.ndarray, g_r: np.ndarray, amp: np.ndarray, labels: np.ndarray,
            floor: float, hist_e: float, hist_r: float) -> dict[str, float]:
    """Medidas de um par (mapa do modelo treinado, mapa do modelo aleatório)."""
    h_e, h_r = entropy(g_e), entropy(g_r)
    ceiling = (np.log(len(g_e)) - h_e) / h_e
    q = (h_r - h_e) / h_e
    d_s = unit_distance(g_e, g_r)
    d_a = unit_distance(np.abs(g_e), np.abs(g_r))
    ok = floor > 0
    return {"entropia_e": h_e, "entropia_r": h_r, "q": q, "teto": ceiling,
            "q_teto": q / ceiling if ceiling > 0 else float("nan"),
            "d_sinal": d_s, "razao_sinal": d_s / floor if ok else float("nan"),
            "acima_piso_sinal": float(d_s > floor) if ok else float("nan"),
            "d_abs": d_a, "razao_abs": d_a / floor if ok else float("nan"),
            "rho_amp_e": spearman(np.abs(g_e), amp), "rho_amp_r": spearman(np.abs(g_r), amp),
            "top2_qrs_e": top2_qrs(g_e, labels), "top2_qrs_r": top2_qrs(g_r, labels),
            "top1_igual": float(np.argmax(np.abs(g_e)) == np.argmax(np.abs(g_r))),
            "hist_e": hist_e, "hist_r": hist_r, "q_hist": (hist_r - hist_e) / hist_e}


def _slots(regions: list[tuple], g: np.ndarray) -> np.ndarray:
    out = np.full(N_SLOTS, np.nan)
    out[[lead * len(REGIONS) + r for lead, _, r in regions]] = g
    return out


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
    rand_seeds = [int(s) for s in args.rand_seeds.split(",")]
    make = lambda m: Explainer(m, train.X, train.Y, device, reference=args.reference,
                               seed=args.seed, n_baselines=args.n_baselines,
                               per_class=args.per_class, n_samples=args.n_samples,
                               chunks=args.chunks, stdevs=args.stdevs)
    trained = make(model)
    explainers = [make(randomize(model, s)) for s in rand_seeds]
    print(f"[info] {explainers[0].describe()}; sementes de randomização {rand_seeds}")
    baselines = explainers[0].baselines.cpu().numpy().astype(np.float64)
    test = np.load(data_dir / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None
    floor = _load_floor(Path(args.sens_mapas))
    _check_outputs(model, explainers, X, device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pair_cols = ["record_id", "classe", "semente_modelo"] + STATS
    map_cols = ["record_id", "classe", "n_regioes", "piso_regioes"] + STATS
    # escritos mapa a mapa (resultado parcial se cair)
    fh_p = open(results_dir / "emprt_modelos.csv", "w", newline="")
    fh_m = open(results_dir / "emprt_mapas.csv", "w", newline="")
    w_p = csv.DictWriter(fh_p, fieldnames=pair_cols)
    w_m = csv.DictWriter(fh_m, fieldnames=map_cols)
    w_p.writeheader()
    w_m.writeheader()
    fmt = lambda r: {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()}

    attr_dir = Path(args.attr_dir)
    per_map: list[dict] = []
    saved = {"record_id": [], "classe": [], "g_e": [], "g_r": [], "amplitude": []}
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
        items = list(zip(data["attributions"], data["test_indices"], rids))
        if args.limit:
            items = items[: args.limit]
        for a, ti, rid in items:
            ti = int(ti)
            x = np.asarray(X[ti], dtype=np.float32)
            if ti not in region_cache:
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                regions = region_index(x, length, with_labels=True)
                region_cache[ti] = (regions, length, np.array([r for _, _, r in regions]),
                                    amplitude(x.astype(np.float64), baselines, regions))
            regions, length, labels, amp = region_cache[ti]
            g_e = region_vector(np.asarray(a, dtype=np.float64), regions)
            h_e = hist_entropy(np.asarray(a, dtype=np.float64)[:, :length])
            fl = floor.get((str(rid), cls), float("nan"))
            seed = map_seed(args.seed, ti, c)  # mesma semente do mapa salvo
            if args.check_repro and ti == int(items[0][1]):
                d = unit_distance(g_e, region_vector(trained(x, c, seed), regions))
                print(f"[verificação] {cls}: mapa refeito no modelo treinado × mapa salvo, "
                      f"distância {d:.4f} (≈ 0 se a configuração é a mesma)")
            rows, g_rs = [], []
            for s, explain in zip(rand_seeds, explainers):
                m_r = explain(x, c, seed)
                g_r = region_vector(m_r, regions)
                g_rs.append(_slots(regions, g_r))
                row = {"record_id": str(rid), "classe": cls, "semente_modelo": s,
                       **compare(g_e, g_r, amp, labels, fl, h_e,
                                 hist_entropy(m_r[:, :length]))}
                rows.append(row)
                w_p.writerow(fmt(row))
            row = {"record_id": str(rid), "classe": cls, "n_regioes": len(regions),
                   "piso_regioes": fl,
                   **{k: float(np.nanmean([r[k] for r in rows])) for k in STATS}}
            per_map.append(row)
            w_m.writerow(fmt(row))
            fh_m.flush()
            fh_p.flush()
            saved["record_id"].append(str(rid))
            saved["classe"].append(cls)
            saved["g_e"].append(_slots(regions, g_e))
            saved["g_r"].append(np.stack(g_rs))
            saved["amplitude"].append(_slots(regions, amp))
            n_done += 1
            if n_done % args.log_every == 0 or n_done == n_total:
                el = time.time() - t_start
                print(f"[progresso] {n_done}/{n_total} mapas, {el / 60:.1f} min, "
                      f"restam ~{el / n_done * (n_total - n_done) / 60:.0f} min", flush=True)
        print(f"  {cls:>5}: {len(items)} mapas")
    fh_p.close()
    fh_m.close()
    print(f"[salvo] {results_dir / 'emprt_modelos.csv'}")
    print(f"[salvo] {results_dir / 'emprt_mapas.csv'}")
    out_npz = results_dir / "emprt_regioes.npz"
    np.savez_compressed(out_npz, sementes_modelo=np.array(rand_seeds),
                        **{k: np.asarray(v) for k, v in saved.items()})
    print(f"[salvo] {out_npz}")

    rows = _aggregate(per_map)
    _print_table(rows)
    out = results_dir / "emprt.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")


@torch.no_grad()
def _check_outputs(model, explainers, X, device, n: int = 64) -> None:
    """Verificação: o modelo aleatório deve perder a estrutura das saídas."""
    xb = torch.from_numpy(np.asarray(X[:n], dtype=np.float32)).to(device)
    for name, m in [("treinado", model)] + [(f"aleatório {i}", e.gs.forward_func)
                                           for i, e in enumerate(explainers)]:
        z = m(xb).float().cpu().numpy()
        top = np.bincount(z.argmax(axis=1), minlength=len(CLASSES))
        print(f"[verificação] {name:>12}: |logit| médio {np.abs(z).mean():.3f}, "
              f"desvio entre ECGs {z.std(axis=0).mean():.3f}, classe de maior logit {top.tolist()}")


def _aggregate(per_map: list[dict]) -> list[dict]:
    rows = []
    for cls in CLASSES + ["global"]:
        sel = [r for r in per_map if cls == "global" or r["classe"] == cls]
        if not sel:
            continue
        row = {"classe": cls, "n": len(sel)}
        for s in ["piso_regioes"] + STATS:
            v = np.array([r[s] for r in sel], dtype=np.float64)
            v = v[np.isfinite(v)]
            row[f"{s}_media"] = float(v.mean()) if v.size else float("nan")
            row[f"{s}_desvio"] = float(v.std()) if v.size else float("nan")
        row["frac_q_positivo"] = float(np.mean([r["q"] > 0 for r in sel]))
        row["frac_q_hist_positivo"] = float(np.mean([r["q_hist"] > 0 for r in sel]))
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\neMPRT por classe explicada (36 regiões; média das sementes de randomização)")
    print(f"{'classe':>7} {'n':>4} {'q':>7} {'teto':>6} {'q/teto':>7} {'q>0':>6} "
          f"{'d sinal':>8} {'d |·|':>6} {'piso':>6} {'ρ amp e':>8} {'ρ amp ê':>8} "
          f"{'QRS e':>6} {'QRS ê':>6} {'q hist':>7}")
    for r in rows:
        print(f"{r['classe']:>7} {r['n']:4d} {r['q_media']:7.3f} {r['teto_media']:6.3f} "
              f"{r['q_teto_media']:7.2f} {r['frac_q_positivo']:6.1%} "
              f"{r['d_sinal_media']:8.3f} {r['d_abs_media']:6.3f} {r['piso_regioes_media']:6.3f} "
              f"{r['rho_amp_e_media']:8.2f} {r['rho_amp_r_media']:8.2f} "
              f"{r['top2_qrs_e_media']:6.1%} {r['top2_qrs_r_media']:6.1%} "
              f"{r['q_hist_media']:7.3f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="eMPRT (regiões do batimento)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions",
                   help="mapas salvos do modelo treinado (e)")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--sens-mapas", type=str, default="results/sensitivity_mapas.csv",
                   help="sensitivity_mapas.csv com o piso de cada mapa (opcional)")
    p.add_argument("--rand-seeds", type=str, default="0,1,2",
                   help="sementes da randomização dos pesos, separadas por vírgula")
    p.add_argument("--no-check-repro", dest="check_repro", action="store_false",
                   help="não refazer o 1º mapa de cada classe no modelo treinado")
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
