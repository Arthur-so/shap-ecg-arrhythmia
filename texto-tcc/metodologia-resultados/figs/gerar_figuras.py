"""Figuras do texto de Metodologia e Resultados (texto-tcc/metodologia-resultados).

Gera, em PDF, as figuras feitas a partir dos dados do projeto:
  - treino.pdf: perda de treino e F1-macro da validação por época (modelo B);
  - limiar.pdf: F1 da validação em função do limiar, para três classes;
  - gshap.pdf: um sorteio do GradientSHAP com sinais reais e o mapa final;
  - regioes.pdf: picos R, segmentos do batimento e limites alternativos;
  - agregacao.pdf: mapa de um ECG de BRD e o vetor das 36 regiões;
  - retirada.pdf: retirada das regiões de QRS de V1 (substituição por zero).

As figuras esquemáticas (fluxogramas, arquitetura, piso) estão em TikZ no
próprio main.tex.

Uso (na raiz do repositório):
    PYTHONPATH=. python texto-tcc/metodologia-resultados/figs/gerar_figuras.py
``limiar.pdf`` precisa de ``output/modeloB/val_probs.npz`` (probabilidades do
modelo B na validação); sem ele, essa figura é pulada.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.config import CLASSES, SAMPLING_RATE as FS, SEED
from src.consistency import LEADS, REGIONS, detect_r_peaks, region_mask, summarize
from src.gshap import reference_indices
from src.metrics import per_class_f1

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ATTR = ROOT / "output" / "modeloB_balanceada200"
EXEMPLO = ("BRD", 20)  # (classe, posição no arquivo de mapas): registro A0808, 80 bpm

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COR = {"pré-QRS": "#4C78A8", "QRS": "#E45756", "ST-T": "#54A24B"}
SINAL = "#222222"


def salvar(fig, nome: str) -> None:
    fig.savefig(OUT / nome)
    plt.close(fig)
    print(f"[salvo] {OUT / nome}")


@lru_cache(maxsize=1)
def exemplo():
    """ECG, comprimento real, picos R, máscara e mapa do exemplo de BRD."""
    cls, j = EXEMPLO
    z = np.load(ATTR / f"{cls}.npz")
    i = int(z["test_indices"][j])
    te = np.load(ROOT / "data" / "processed" / "test.npz")
    x, n = te["X"][i], int(te["lengths"][i])
    r = detect_r_peaks(x[LEADS.index("II"), :n])
    mask = region_mask(r, n, x.shape[1])
    return x, n, r, mask, z["attributions"][j], str(z["record_ids"][j])


def sombrear(ax, mask, t0, t1, alpha=0.18):
    """Pinta os segmentos da máscara entre as amostras t0 e t1."""
    seg = mask[t0:t1]
    ini = 0
    for k in range(1, len(seg) + 1):
        if k == len(seg) or seg[k] != seg[ini]:
            if seg[ini] >= 0:
                ax.axvspan((t0 + ini) / FS, (t0 + k) / FS, color=COR[REGIONS[seg[ini]]],
                           alpha=alpha, lw=0)
            ini = k


# --------------------------------------------------------------------------- #
def fig_treino() -> None:
    d = json.load(open(ROOT / "results" / "ablacao_bce" / "B_bce_simples" / "f1_scores.json"))
    h = d["history"]
    ep = [e["epoch"] for e in h]
    best = d["best_epoch"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.3))
    a.plot(ep, [e["train_loss"] for e in h], color=SINAL, lw=1.2)
    a.set_yscale("log")
    a.set_xlabel("época")
    a.set_ylabel("perda BCE no treino")
    a.set_title("(a) perda de treino")
    b.plot(ep, [e["val_macro_f1"] for e in h], color="#2F5D8A", lw=1.2,
           label="limiares por classe")
    b.plot(ep, [e["val_macro_f1_05"] for e in h], color="#9E9E9E", lw=1.0, ls="--",
           label="limiar fixo 0,5")
    b.set_xlabel("época")
    b.set_ylabel("F1-macro na validação")
    b.set_title("(b) F1-macro na validação")
    b.set_ylim(0, 0.9)
    b.legend(loc="lower right", frameon=False, bbox_to_anchor=(1, 0.02))
    for ax in (a, b):
        ax.axvline(best, color="#E45756", lw=0.9, ls=":")
        ax.axvline(ep[-1], color="#9E9E9E", lw=0.9, ls=":")
    b.annotate(f"melhor época ({best})\nF1 = {d['best_val_macro_f1']:.3f}".replace(".", ","),
               xy=(best, d["best_val_macro_f1"]), xytext=(12, 0.42),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7.5)
    b.text(ep[-1] - 0.6, 0.55, f"parada\n({ep[-1]})", ha="right", fontsize=7.5, color="#666666")
    fig.tight_layout(w_pad=2.0)
    salvar(fig, "treino.pdf")


def fig_limiar() -> None:
    path = ROOT / "output" / "modeloB" / "val_probs.npz"
    if not path.exists():
        print(f"[pulado] limiar.pdf: falta {path}")
        return
    v = np.load(path)
    probs, y = v["probs"], v["Y"]
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    cores = ["#2F5D8A", "#E45756", "#54A24B"]
    for cor, cls in zip(cores, ["AF", "IAVB", "STD"]):
        c = CLASSES.index(cls)
        pred = (probs[:, c][None, :] >= grid[:, None]).astype(np.int64)
        f1 = per_class_f1(np.repeat(y[:, c][None, :], len(grid), 0).T, pred.T)
        k = int(np.argmax(f1))
        ax.plot(grid, f1, color=cor, lw=1.2, label=cls)
        ax.plot(grid[k], f1[k], "o", color=cor, ms=4)
        ax.annotate(f"{grid[k]:.2f}".replace(".", ","), (grid[k], f1[k]),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5, color=cor)
    ax.axvline(0.5, color="#9E9E9E", lw=0.8, ls="--")
    ax.set_xlabel("limiar de decisão")
    ax.set_ylabel("F1 na validação")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower center", ncol=3)
    salvar(fig, "limiar.pdf")


def fig_gshap() -> None:
    x, n, r, mask, attr, rec = exemplo()
    tr = np.load(ROOT / "data" / "processed" / "train.npz")
    base = tr["X"][reference_indices(tr["Y"], "balanceada", SEED)[0]]
    d = LEADS.index("II")
    t0, t1 = 0, int(2.5 * FS)
    t = np.arange(t0, t1) / FS
    rng = np.random.default_rng(0)
    xt = x[d, t0:t1] + rng.normal(0, 0.09, t1 - t0)
    alpha = 0.4
    xp = base[d, t0:t1] + alpha * (xt - base[d, t0:t1])
    linhas = [
        (x[d, t0:t1], r"(a) ECG explicado $x$"),
        (base[d, t0:t1], r"(b) referência sorteada $r$ (1 dos 36 ECGs)"),
        (xt, r"(c) ECG com ruído gaussiano $\tilde{x} = x + \varepsilon$, desvio 0,09"),
        (xp, r"(d) ponto do caminho $x' = r + \alpha(\tilde{x} - r)$, com $\alpha = 0{,}4$"),
    ]
    fig, axs = plt.subplots(5, 1, figsize=(6.3, 5.4), sharex=True)
    for ax, (y, tit) in zip(axs, linhas):
        ax.plot(t, y, color=SINAL, lw=0.8)
        ax.set_title(tit, loc="left")
        ax.set_ylabel("z-score")
    a = attr[d, t0:t1]
    lim = np.abs(a).max()
    axs[4].fill_between(t, a, 0, where=a >= 0, color="#E45756", lw=0)
    axs[4].fill_between(t, a, 0, where=a < 0, color="#4C78A8", lw=0)
    axs[4].set_ylim(-lim * 1.1, lim * 1.1)
    axs[4].set_title(r"(e) mapa final $\phi$ (média de 200 sorteios de $\nabla f(x') \odot (\tilde{x} - r)$)",
                     loc="left")
    axs[4].set_ylabel("atribuição")
    axs[4].set_xlabel("tempo (s)")
    fig.tight_layout(h_pad=0.4)
    salvar(fig, "gshap.pdf")


def fig_regioes() -> None:
    x, n, r, mask, _, rec = exemplo()
    d = LEADS.index("II")
    t0, t1 = 0, int(3.2 * FS)
    fig = plt.figure(figsize=(6.3, 5.0))
    a = fig.add_axes([0.08, 0.60, 0.9, 0.33])
    b = fig.add_axes([0.08, 0.08, 0.9, 0.33])
    sombrear(a, mask, t0, t1)
    a.plot(np.arange(t0, t1) / FS, x[d, t0:t1], color=SINAL, lw=0.8)
    rr = r[(r >= t0) & (r < t1)]
    a.plot(rr / FS, x[d, rr], "v", color="black", ms=4)
    a.set_xlim(t0 / FS, t1 / FS)
    a.set_xlabel("tempo (s)")
    a.set_ylabel("derivação II (z-score)")
    a.set_title(f"(a) picos R detectados e segmentos de cada batimento completo (registro {rec})",
                loc="left")
    a.text(r[0] / FS / 2, a.get_ylim()[0] * 0.85, "fora\n(incompleto)", ha="center",
           fontsize=7, color="#666666")
    hand = [plt.Rectangle((0, 0), 1, 1, color=COR[k], alpha=0.35) for k in REGIONS]
    a.legend(hand, REGIONS, ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.08))

    # (b) um batimento ampliado, com os limites da base e os alternativos
    k = 1
    r0, r1 = int(r[k]), int(r[k + 1])
    rrs = (r1 - r0) / FS
    s0, s1 = r0 - int(0.25 * FS), r1 + int(0.15 * FS)
    tb = (np.arange(s0, s1) - r0) / FS * 1000
    sombrear_ms = mask[s0:s1]
    ini = 0
    for q in range(1, len(sombrear_ms) + 1):
        if q == len(sombrear_ms) or sombrear_ms[q] != sombrear_ms[ini]:
            if sombrear_ms[ini] >= 0:
                b.axvspan(tb[ini], tb[q - 1], color=COR[REGIONS[sombrear_ms[ini]]], alpha=0.18, lw=0)
            ini = q
    b.plot(tb, x[d, s0:s1], color=SINAL, lw=0.9)
    lims = [(-80, "base"), (100, "base"), (0.45 * rrs * 1000, "base"), (rrs * 1000 - 80, "base")]
    alts = [-60, -100, 80, 120, 0.40 * rrs * 1000, 0.50 * rrs * 1000]
    yl = b.get_ylim()
    for v, _ in lims:
        b.axvline(v, color="black", lw=0.9)
    for v in alts:
        b.axvline(v, color="black", lw=0.7, ls=(0, (2, 2)))
    txt = [(-80, "R − 80 ms"), (100, "R + 100 ms"), (0.45 * rrs * 1000, "R + 0,45·RR"),
           (rrs * 1000 - 80, "R seguinte − 80 ms")]
    for v, s in txt:
        b.text(v, yl[1] + 0.02 * (yl[1] - yl[0]), s, rotation=0, ha="center", va="bottom", fontsize=7)
    b.set_xlabel("tempo em relação ao pico R (ms)")
    b.set_ylabel("derivação II (z-score)")
    b.set_title(f"(b) um batimento ampliado (RR = {rrs * 1000:.0f} ms): limites da base (contínuos) "
                "e alternativos do teste de robustez (tracejados)", loc="left", y=1.10)
    salvar(fig, "regioes.pdf")


def fig_agregacao() -> None:
    x, n, r, mask, attr, rec = exemplo()
    d = LEADS.index("V1")
    t0, t1 = 0, int(3.2 * FS)
    s = summarize(attr, mask)
    fig = plt.figure(figsize=(6.3, 3.6))
    a1 = fig.add_axes([0.07, 0.62, 0.55, 0.30])
    a2 = fig.add_axes([0.07, 0.15, 0.55, 0.40], sharex=a1)
    h = fig.add_axes([0.73, 0.12, 0.17, 0.80])
    t = np.arange(t0, t1) / FS
    for ax in (a1, a2):
        sombrear(ax, mask, t0, t1)
    a1.plot(t, x[d, t0:t1], color=SINAL, lw=0.8)
    a1.set_ylabel("V1 (z-score)")
    a1.set_title("(a) sinal e mapa de V1 (classe BRD)", loc="left")
    plt.setp(a1.get_xticklabels(), visible=False)
    a = attr[d, t0:t1]
    a2.fill_between(t, a, 0, where=a >= 0, color="#E45756", lw=0)
    a2.fill_between(t, a, 0, where=a < 0, color="#4C78A8", lw=0)
    a2.set_ylabel("atribuição")
    a2.set_xlabel("tempo (s)")
    a2.set_xlim(t0 / FS, t1 / FS)
    m = s.reshape(12, 3)
    lim = np.abs(m).max()
    im = h.imshow(m, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    h.set_xticks(range(3), ["pré-\nQRS", "QRS", "ST-T"], fontsize=7)
    h.set_yticks(range(12), LEADS, fontsize=7)
    for (i, j), v in np.ndenumerate(m):
        h.text(j, i, f"{v + 0.0:.1f}".replace("-0.0", "0.0").replace(".", ",").replace("-", "−"), ha="center", va="center",
               fontsize=6, color="white" if abs(v) > 0.55 * lim else "black")
    h.set_title("(b) 36 regiões\n(soma nos batimentos)", fontsize=8)
    h.spines[:].set_visible(False)
    salvar(fig, "agregacao.pdf")


def fig_retirada() -> None:
    x, n, r, mask, _, rec = exemplo()
    d = LEADS.index("V1")
    t0, t1 = 0, int(3.2 * FS)
    y = x[d].copy()
    y[mask == REGIONS.index("QRS")] = 0.0
    t = np.arange(t0, t1) / FS
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    sombrear(ax, mask, t0, t1)
    ax.plot(t, x[d, t0:t1], color="#9E9E9E", lw=0.8, label="original")
    ax.plot(t, y[t0:t1], color=SINAL, lw=0.9, label="região V1·QRS retirada (zero)")
    ax.set_xlim(t0 / FS, t1 / FS)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("V1 (z-score)")
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.18))
    salvar(fig, "retirada.pdf")


if __name__ == "__main__":
    fig_treino()
    fig_limiar()
    fig_gshap()
    fig_regioes()
    fig_agregacao()
    fig_retirada()
