"""Faithfulness Estimate das explicações GradientSHAP (Alvarez-Melis & Jaakkola, 2018).

Implementa o §5.3 do artigo — para cada exemplo, cada atributo é retirado
isoladamente e calcula-se a correlação de Pearson entre a importância que a
explicação lhe dá e a queda na saída do modelo — com atributos adequados ao
ECG. Substitui a ``quantus.FaithfulnessEstimate``, cuja configuração usada
(grupos de 500 pontos em ordem de atribuição, trocados pelo mínimo do próprio
sinal) media o dano causado ao sinal e não a retirada de informação (ver
fichamento de Alvarez-Melis & Jaakkola, 2018).

Protocolo:
  1. Atributos = 36 regiões do batimento (12 derivações × pré-QRS/QRS/ST-T),
     as mesmas de ``consistency.py`` e ``faithfulness_correlation.py``. A
     importância de uma região é a soma do mapa SHAP nela; retirá-la é
     substituir todos os seus pontos (em todos os batimentos) pela baseline.
     Batimentos incompletos e o preenchimento (padding) ficam de fora.
  2. Cada região é retirada isoladamente (uma retirada por região): soma das
     atribuições × queda f(x) − f(x_[região = baseline]).
  3. Correlação de Pearson por exemplo; média por classe. Equivale à
     Faithfulness Correlation com subconjuntos de uma região, percorrendo
     todas as regiões em vez de sorteá-las.
  4. Saídas: ``logit`` da classe explicada (principal; a probabilidade satura
     com o modelo confiante) e ``prob`` = sigmoide do logit (a saída do
     artigo, reportada como verificação).
  5. Baselines do artigo (“valores médios”): ``zero`` (em z-score, a média de
     cada derivação) e ``mean`` (média de cada ponto derivação × instante no
     treino).

Ao final salva:
  - ``results/faithfulness_estimate.csv``: média por classe, baseline e saída.
  - ``results/faithfulness_estimate_mapas.csv``: correlação de cada mapa.
  - ``results/faithfulness_estimate_regioes.csv``: cada retirada (região,
    soma das atribuições, quedas do logit e da probabilidade).

Uso:
    python -m src.faithfulness_estimate \
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

from src.config import CLASSES
from src.consistency import LEADS, REGIONS, _real_length
from src.faithfulness_correlation import (_train_mean, pearson, perturbed_outputs,
                                          region_index, subset_sums)

OUTPUTS = ["logit", "prob"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def region_drops(x: np.ndarray, attr: np.ndarray, regions: list[tuple],
                 baseline: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
                 batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retira cada região isoladamente.

    ``predict`` recebe um lote (N, 12, T) e devolve o logit da classe
    explicada (N,). Retorna (soma das atribuições, queda do logit, queda da
    probabilidade), um valor por região.
    """
    singles = [np.array([k]) for k in range(len(regions))]
    sums = subset_sums(attr, regions, singles)
    f_x = float(predict(x[None])[0])
    z = perturbed_outputs(x, regions, singles, baseline, predict, batch_size)
    return sums, f_x - z, _sigmoid(f_x) - _sigmoid(z)


def faithfulness_estimate(x: np.ndarray, attr: np.ndarray, regions: list[tuple],
                          baseline: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
                          batch_size: int = 64) -> dict[str, float]:
    """§5.3 de Alvarez-Melis & Jaakkola (2018) para um exemplo, nas duas saídas."""
    sums, d_logit, d_prob = region_drops(x, attr, regions, baseline, predict, batch_size)
    return {"logit": pearson(sums, d_logit), "prob": pearson(sums, d_prob)}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
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

    region_cache: dict[int, list] = {}
    per_map = []     # (record_id, classe, baseline, saída, correlação)
    per_region = []  # uma linha por retirada
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
                region_cache[ti] = region_index(x, length, with_labels=True)
            regions = region_cache[ti]
            a = np.asarray(a, dtype=np.float64)
            for b, base in baselines.items():
                sums, d_logit, d_prob = region_drops(x, a, regions, base, predict,
                                                     batch_size=args.batch_size)
                per_map.append((str(rid), cls, b, "logit", pearson(sums, d_logit)))
                per_map.append((str(rid), cls, b, "prob", pearson(sums, d_prob)))
                for (lead, _, r), s, dl, dp in zip(regions, sums, d_logit, d_prob):
                    per_region.append([str(rid), cls, b, LEADS[lead], REGIONS[r],
                                       f"{s:.6g}", f"{dl:.6g}", f"{dp:.6g}"])
        print(f"  {cls:>5}: {len(data['attributions'])} mapas")

    rows = _aggregate(per_map, list(baselines))
    _print_table(rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "faithfulness_estimate.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[salvo] {out}")
    out_m = results_dir / "faithfulness_estimate_mapas.csv"
    with open(out_m, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "classe", "baseline", "saida", "faithfulness_estimate"])
        for rid, cls, b, o, corr in per_map:
            w.writerow([rid, cls, b, o, f"{corr:.4f}"])
    print(f"[salvo] {out_m}")
    out_r = results_dir / "faithfulness_estimate_regioes.csv"
    with open(out_r, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "classe", "baseline", "derivacao", "regiao",
                    "soma_atrib", "queda_logit", "queda_prob"])
        w.writerows(per_region)
    print(f"[salvo] {out_r}")


def _aggregate(per_map: list, baselines: list[str]) -> list[dict]:
    rows = []
    for b in baselines:
        for o in OUTPUTS:
            for cls in CLASSES + ["global"]:
                v = np.array([r[4] for r in per_map
                              if r[2] == b and r[3] == o and (cls == "global" or r[1] == cls)])
                if v.size == 0:
                    continue
                ok = v[np.isfinite(v)]
                rows.append({"baseline": b, "saida": o, "classe": cls,
                             "n": int(v.size), "n_nan": int(v.size - ok.size),
                             "media": float(ok.mean()) if ok.size else float("nan"),
                             "desvio": float(ok.std()) if ok.size else float("nan")})
    return rows


def _print_table(rows: list[dict]) -> None:
    cols = list(dict.fromkeys((r["baseline"], r["saida"]) for r in rows))
    print("\nFaithfulness Estimate média por classe")
    print(f"{'classe':>7} " + " ".join(f"{b + '/' + o:>11}" for b, o in cols))
    for cls in CLASSES + ["global"]:
        vals = [next((r for r in rows if (r["baseline"], r["saida"]) == bo
                      and r["classe"] == cls), None) for bo in cols]
        if all(v is None for v in vals):
            continue
        print(f"{cls:>7} " + " ".join(f"{v['media']:11.3f}" if v else f"{'-':>11}" for v in vals))
    nan = sum(r["n_nan"] for r in rows if r["classe"] == "global")
    print(f"\n[info] valores indefinidos (NaN) em todas as configurações: {nan}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Faithfulness Estimate (regiões do batimento)")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet34_1d_best.pt")
    p.add_argument("--attr-dir", type=str, default="results/attributions")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--baselines", nargs="+", default=["zero", "mean"], choices=["zero", "mean"])
    p.add_argument("--batch-size", type=int, default=64)
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
