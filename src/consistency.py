"""Consistency das explicações GradientSHAP (Dasgupta, Frost & Moshkovitz, 2022).

Implementa a medida de *consistency* do artigo — "instâncias que recebem a
mesma explicação devem receber a mesma predição" — com uma discretização das
explicações adequada ao ECG. Substitui a ``quantus.Consistency``, cuja
discretização padrão (sinal dos 5 *primeiros* valores do mapa, após valor
absoluto) atribuía a mesma explicação a todos os mapas.

Protocolo:
  1. Regiões por batimento, ancoradas nos picos R (derivação II):
       QRS     = [R - 80 ms, R + 100 ms)
       ST-T    = [R + 100 ms, R + 0,45·RR)
       pré-QRS = [R + 0,45·RR, R_seguinte - 80 ms)
     Só batimentos completos (entre dois picos R) entram; o trecho de
     preenchimento (padding) é excluído.
  2. Cada mapa (12, 15000) é resumido em 36 valores: soma da atribuição por
     derivação × região, acumulada em todos os batimentos.
  3. Discretização ψ "sinal das k regiões de maior |atribuição|" (análoga ao
     Sign-of-top-5 do artigo, Apêndice D.5), com k = 2 por padrão.
  4. Consistency local (Def. 1) com o estimador do artigo (§4.1): fração das
     *outras* instâncias com a mesma explicação discretizada que têm a mesma
     predição; instância sem par recebe 0. A predição f é a classe que a
     explicação justifica. Todas as classes formam uma única população.
  5. Por classe: média das consistencies locais, o valor esperado ao acaso
     (classes embaralhadas entre os mapas) e o escore ajustado
     (observado − acaso) / (1 − acaso).

Ao final salva:
  - ``results/consistency.csv``: escores por classe (+ F1, se disponível).
  - ``results/consistency_explanations.csv``: explicação discretizada e
    consistency local de cada mapa.

Uso:
    python -m src.consistency \
        --data-dir data/processed \
        --attr-dir results/attributions \
        --results-dir results --k 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.config import CLASSES, SAMPLING_RATE

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
REGIONS = ["pré-QRS", "QRS", "ST-T"]
REF_LEAD = LEADS.index("II")


# --------------------------------------------------------------------------- #
# Regiões do batimento
# --------------------------------------------------------------------------- #
def detect_r_peaks(lead: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """Índices dos picos R de uma derivação (neurokit2)."""
    import neurokit2 as nk

    clean = nk.ecg_clean(lead, sampling_rate=fs)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def region_mask(r_peaks: np.ndarray, length: int, total: int,
                fs: int = SAMPLING_RATE, qrs_pre: float = 0.080,
                qrs_post: float = 0.100, st_frac: float = 0.45) -> np.ndarray:
    """Rótulo de região por amostra: 0 = pré-QRS, 1 = QRS, 2 = ST-T, -1 = fora.

    ``length`` é o nº de amostras reais (antes do padding) e ``total`` o
    comprimento do mapa. Só batimentos completos são rotulados.
    """
    pre, post = int(qrs_pre * fs), int(qrs_post * fs)
    mask = np.full(total, -1, dtype=np.int8)
    for r0, r1 in zip(r_peaks[:-1], r_peaks[1:]):
        q0, q1 = max(r0 - pre, 0), r0 + post
        s1 = max(r0 + int(st_frac * (r1 - r0)), q1)
        mask[q0:q1] = 1
        mask[q1:s1] = 2
        mask[s1:r1 - pre] = 0
    mask[length:] = -1
    return mask


def summarize(attr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Soma a atribuição (12, T) por derivação × região -> vetor (36,).

    Ordem: derivação-major, i.e. índice = 3·derivação + região.
    """
    return np.stack([attr[:, mask == r].sum(axis=1) for r in range(len(REGIONS))],
                    axis=1).ravel()


# --------------------------------------------------------------------------- #
# Discretização e estimador
# --------------------------------------------------------------------------- #
def discretize(summary: np.ndarray, k: int) -> tuple[tuple[int, int], ...]:
    """ψ: conjunto (região, sinal) das k regiões de maior |atribuição|."""
    idx = np.argsort(-np.abs(summary), kind="stable")[:k]
    return tuple(sorted((int(i), int(np.sign(summary[i]))) for i in idx))


def explanation_name(expl: tuple[tuple[int, int], ...]) -> str:
    """Texto legível, ex.: 'V1·QRS(+) | aVR·QRS(+)'."""
    return " | ".join(
        f"{LEADS[i // len(REGIONS)]}·{REGIONS[i % len(REGIONS)]}({'+' if s > 0 else '−'})"
        for i, s in expl)


def local_consistency(expls: list, preds: np.ndarray) -> np.ndarray:
    """Estimador do artigo (§4.1): para cada instância, fração das *outras*
    com a mesma explicação que têm a mesma predição; 0 se não há outra."""
    groups: dict = defaultdict(list)
    for i, e in enumerate(expls):
        groups[e].append(i)
    scores = np.zeros(len(expls))
    for idx in groups.values():
        if len(idx) < 2:
            continue
        p = preds[idx]
        for i in idx:
            scores[i] = (np.sum(p == preds[i]) - 1) / (len(idx) - 1)
    return scores


def chance_level(expls: list, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Consistency esperada por classe se as predições fossem embaralhadas
    entre os mapas (estrutura de explicações fixa).

    Para uma instância da classe c num grupo com ≥ 2 membros, a fração
    esperada de outros membros da classe c é (n_c − 1)/(n − 1); instâncias
    sem par valem 0. Logo acaso_c = q · (n_c − 1)/(n − 1), com q = fração de
    instâncias que têm par.
    """
    n = len(preds)
    sizes: dict = defaultdict(int)
    for e in expls:
        sizes[e] += 1
    q = sum(1 for e in expls if sizes[e] > 1) / n
    counts = np.bincount(preds, minlength=n_classes)
    return q * np.maximum(counts - 1, 0) / max(n - 1, 1)


def adjusted(observed: np.ndarray, chance: np.ndarray) -> np.ndarray:
    """(observado − acaso) / (1 − acaso): 0 = acaso, 1 = perfeito."""
    return (observed - chance) / np.where(chance < 1, 1 - chance, np.nan)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _real_length(x: np.ndarray) -> int:
    """Fallback para .npz antigos sem ``lengths``: o padding (zeros antes do
    z-score) vira um trecho final constante em todas as derivações."""
    diff = np.any(x != x[:, -1:], axis=0)
    nz = np.flatnonzero(diff)
    return int(nz[-1]) + 1 if nz.size else x.shape[1]


def run(args: argparse.Namespace) -> None:
    test = np.load(Path(args.data_dir) / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None
    if lengths is None:
        print("[aviso] test.npz sem 'lengths' (pré-processamento antigo); "
              "estimando o fim do sinal real pelo trecho final constante")

    attr_dir = Path(args.attr_dir)
    summaries, preds, rec_ids = [], [], []
    peaks_cache: dict[int, tuple[np.ndarray, int]] = {}
    for c, cls in enumerate(CLASSES):
        path = attr_dir / f"{cls}.npz"
        if not path.exists():
            print(f"[skip] {cls}: sem atribuições ({path})")
            continue
        data = np.load(path)
        A, idxs = data["attributions"], data["test_indices"]
        rids = data["record_ids"] if "record_ids" in data.files else idxs
        for a, ti, rid in zip(A, idxs, rids):
            ti = int(ti)
            if ti not in peaks_cache:
                x = X[ti]
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                peaks_cache[ti] = (detect_r_peaks(x[REF_LEAD, :length]), length)
            r, length = peaks_cache[ti]
            mask = region_mask(r, length, a.shape[1],
                               qrs_pre=args.qrs_pre, qrs_post=args.qrs_post,
                               st_frac=args.st_frac)
            summaries.append(summarize(a.astype(np.float64), mask))
            preds.append(c)
            rec_ids.append(str(rid))
        print(f"  {cls:>5}: {len(A)} mapas")

    preds_arr = np.asarray(preds)
    expls = [discretize(s, args.k) for s in summaries]
    local = local_consistency(expls, preds_arr)
    chance = chance_level(expls, preds_arr, len(CLASSES))
    sizes = Counter(expls)
    n_unique = sum(1 for e in expls if sizes[e] == 1)
    print(f"\n[info] {len(expls)} mapas, k={args.k}: {len(set(expls))} explicações "
          f"distintas, {n_unique / len(expls):.1%} sem par (unicidade)")

    f1 = _load_f1(Path(args.results_dir))
    rows = []
    for c, cls in enumerate(CLASSES):
        sel = preds_arr == c
        if not sel.any():
            continue
        obs = float(local[sel].mean())
        rows.append({"classe": cls, "n": int(sel.sum()),
                     "consistency": obs, "acaso": float(chance[c]),
                     "consistency_ajustada": float(adjusted(np.array(obs), np.array(chance[c]))),
                     "f1": f1.get(cls, "")})
    glob_obs = float(local.mean())
    glob_chance = float(np.mean(chance[preds_arr]))
    print(f"[info] Consistency global = {glob_obs:.3f} | acaso = {glob_chance:.3f} | "
          f"ajustada = {float(adjusted(np.array(glob_obs), np.array(glob_chance))):.3f}")
    _print_table(rows)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "consistency.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")

    out_e = results_dir / "consistency_explanations.csv"
    with open(out_e, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "classe", "explicacao", "consistency_local"])
        for rid, p, e, s in zip(rec_ids, preds_arr, expls, local):
            w.writerow([rid, CLASSES[p], explanation_name(e), f"{s:.4f}"])
    print(f"[salvo] {out_e}")


def _load_f1(results_dir: Path) -> dict[str, float]:
    path = results_dir / "f1_scores.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return data.get("test_f1", data.get("val_f1", {}))


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'classe':>6} {'n':>4} {'consist.':>8} {'acaso':>6} {'ajust.':>7} {'F1':>6}")
    for r in rows:
        f1 = f"{r['f1']:.3f}" if r["f1"] != "" else "-"
        print(f"{r['classe']:>6} {r['n']:4d} {r['consistency']:8.3f} {r['acaso']:6.3f} "
              f"{r['consistency_ajustada']:7.3f} {f1:>6}")
    with_f1 = [r for r in rows if r["f1"] != ""]
    if len(with_f1) >= 3:
        from scipy.stats import pearsonr, spearmanr

        f1 = [r["f1"] for r in with_f1]
        for key in ("consistency", "consistency_ajustada"):
            v = [r[key] for r in with_f1]
            print(f"  F1 × {key}: Pearson={pearsonr(f1, v)[0]:.2f} "
                  f"Spearman={spearmanr(f1, v)[0]:.2f} (n={len(v)})")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Consistency das explicações (regiões do batimento)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--attr-dir", type=str, default="results/attributions")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--k", type=int, default=2,
                   help="nº de regiões (de 36) que formam a explicação discretizada")
    p.add_argument("--qrs-pre", type=float, default=0.080, help="início do QRS antes do R (s)")
    p.add_argument("--qrs-post", type=float, default=0.100, help="fim do QRS após o R (s)")
    p.add_argument("--st-frac", type=float, default=0.45,
                   help="fim do ST-T como fração do intervalo RR")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
