"""Complexidade das explicações GradientSHAP (Bhatt, Weller & Moura, 2020).

Implementa a Definição 4 do artigo — entropia das frações de importância
P(i) = |g_i| / Σ_j |g_j| — com atributos adequados ao ECG. Substitui a
``quantus.Complexity``, que no ECG trata cada um dos 180 000 pontos do mapa
como atributo, inclui o preenchimento (padding) e acaba medindo a duração do
exame (ver ``analise-complexidade`` no fichamento de Bhatt et al., 2020).

Protocolo, para cada mapa (ECG × classe explicada):
  1. Atributos = 36 regiões do batimento (12 derivações × pré-QRS/QRS/ST-T),
     as mesmas de ``consistency.py``, ``faithfulness_correlation.py`` e
     ``faithfulness_estimate.py``. A importância de uma região é a soma do
     mapa SHAP nela; batimentos incompletos e o preenchimento ficam de fora.
  2. Entropia H das frações |importância| / soma (principal) e número efetivo
     de regiões e^H (quantas regiões, todas com a mesma importância, dariam a
     mesma entropia; vai de 1 a 36).
  3. Verificação: índice de Gini dos valores absolutos (a ``quantus.Sparseness``
     de Chalasani et al., 2020, na mesma fórmula), de 0 (uniforme) a
     (n − 1)/n (tudo em uma região).
  4. Apoio à leitura: fração da importância nas 3 regiões mais importantes e
     região mais importante.
  5. Contexto opcional: piso do GradientSHAP de cada mapa, lido do
     ``sensitivity_mapas.csv`` (distância entre duas sementes), que indica
     quanto do espalhamento pode vir do erro de estimativa.

Não chama o modelo: usa só os mapas salvos e roda na CPU.

Ao final salva:
  - ``results/complexity.csv``: média e desvio por classe.
  - ``results/complexity_mapas.csv``: valores de cada mapa.

Uso:
    python -m src.complexity \
        --data-dir data/processed \
        --attr-dir results/attributions \
        --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.config import CLASSES
from src.consistency import LEADS, REGIONS, _real_length
from src.faithfulness_correlation import region_index
from src.sensitivity import region_vector

STATS = ["entropia", "n_efetivo", "gini", "top3_frac", "piso_regioes"]


# --------------------------------------------------------------------------- #
# Núcleo
# --------------------------------------------------------------------------- #
def entropy(g: np.ndarray) -> float:
    """Definição 4 de Bhatt et al. (2020): entropia de |g| / Σ|g|."""
    a = np.abs(np.asarray(g, dtype=np.float64))
    if a.sum() == 0:
        return float("nan")
    p = a / a.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def gini(g: np.ndarray) -> float:
    """Índice de Gini de |g| (fórmula da ``quantus.Sparseness``, sem o 1e-7)."""
    a = np.sort(np.abs(np.asarray(g, dtype=np.float64)))
    n = a.size
    if a.sum() == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * a).sum() / (n * a.sum()))


def complexity(g: np.ndarray) -> dict[str, float]:
    """Complexidade de um vetor de importâncias por região."""
    h = entropy(g)
    share = np.abs(g) / np.abs(g).sum()
    return {"entropia": h, "n_efetivo": float(np.exp(h)), "gini": gini(g),
            "top3_frac": float(np.sort(share)[-3:].sum()), "top1": int(np.argmax(share)),
            "top1_frac": float(share.max())}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _load_floor(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        print(f"[info] sem piso do GradientSHAP ({path}); coluna fica vazia")
        return {}
    with open(path, newline="") as fh:
        return {(r["record_id"], r["classe"]): float(r["piso_regioes"])
                for r in csv.DictReader(fh)}


def run(args: argparse.Namespace) -> None:
    test = np.load(Path(args.data_dir) / "test.npz")
    X = test["X"]
    lengths = test["lengths"] if "lengths" in test.files else None
    floor = _load_floor(Path(args.sens_mapas))

    region_cache: dict[int, list] = {}
    per_map = []
    attr_dir = Path(args.attr_dir)
    for cls in CLASSES:
        path = attr_dir / f"{cls}.npz"
        if not path.exists():
            print(f"[skip] {cls}: sem atribuições ({path})")
            continue
        data = np.load(path)
        rids = data["record_ids"] if "record_ids" in data.files else data["test_indices"]
        for a, ti, rid in zip(data["attributions"], data["test_indices"], rids):
            ti = int(ti)
            if ti not in region_cache:
                x = np.asarray(X[ti], dtype=np.float32)
                length = int(lengths[ti]) if lengths is not None else _real_length(x)
                region_cache[ti] = region_index(x, length, with_labels=True)
            regions = region_cache[ti]
            m = complexity(region_vector(np.asarray(a, dtype=np.float64), regions))
            lead, _, r = regions[m.pop("top1")]
            per_map.append({"record_id": str(rid), "classe": cls, "n_regioes": len(regions),
                            **m, "top1_regiao": f"{LEADS[lead]} {REGIONS[r]}",
                            "piso_regioes": floor.get((str(rid), cls), float("nan"))})
        print(f"  {cls:>5}: {len(data['attributions'])} mapas")

    rows = _aggregate(per_map)
    _print_table(rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "complexity.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")
    out_m = results_dir / "complexity_mapas.csv"
    with open(out_m, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_map[0]))
        w.writeheader()
        for r in per_map:
            w.writerow({k: f"{v:.5f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"[salvo] {out_m}")


def _aggregate(per_map: list[dict]) -> list[dict]:
    rows = []
    for cls in CLASSES + ["global"]:
        sel = [r for r in per_map if cls == "global" or r["classe"] == cls]
        if not sel:
            continue
        row = {"classe": cls, "n": len(sel)}
        for s in STATS:
            v = np.array([r[s] for r in sel], dtype=np.float64)
            v = v[np.isfinite(v)]
            row[f"{s}_media"] = float(v.mean()) if v.size else float("nan")
            row[f"{s}_desvio"] = float(v.std()) if v.size else float("nan")
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\nComplexidade por classe (36 regiões; da mais concentrada para a mais espalhada)")
    print(f"{'classe':>7} {'n':>4} {'H':>6} {'nº efet.':>9} {'Gini':>6} {'top 3':>6} {'piso':>6}")
    by_class = sorted((r for r in rows if r["classe"] != "global"),
                      key=lambda r: r["n_efetivo_media"])
    for r in by_class + [r for r in rows if r["classe"] == "global"]:
        print(f"{r['classe']:>7} {r['n']:4d} {r['entropia_media']:6.3f} "
              f"{r['n_efetivo_media']:9.1f} {r['gini_media']:6.3f} "
              f"{r['top3_frac_media']:6.1%} {r['piso_regioes_media']:6.3f}")
    by_gini = [r["classe"] for r in sorted(by_class, key=lambda r: -r["gini_media"])]
    print("\n[verificação] ordem pela entropia: " + " ".join(r["classe"] for r in by_class))
    print("[verificação] ordem pelo Gini:     " + " ".join(by_gini))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Complexidade (regiões do batimento)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--attr-dir", type=str, default="results/attributions")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--sens-mapas", type=str, default="results/sensitivity_mapas.csv",
                   help="sensitivity_mapas.csv com o piso de cada mapa (opcional)")
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
