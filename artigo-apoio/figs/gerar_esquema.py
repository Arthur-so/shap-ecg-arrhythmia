"""Figura esquemática dos segmentos do batimento (esquema_regioes.pdf).

Usa um ECG sintético (NeuroKit2), sem dados da base, para mostrar como os três
segmentos são definidos a partir dos picos R: o intervalo RR, a janela do QRS e
o fim do ST-T como fração do RR (com a correção de Bazett).

Uso (de qualquer pasta; grava em artigo/figs):
    python3 gerar_esquema.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "artigo" / "figs"
FS = 500
HR = 75
COR = {"pré-QRS": "#4C78A8", "QRS": "#E45756", "ST-T": "#54A24B"}

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ecg = nk.ecg_simulate(duration=4, sampling_rate=FS, heart_rate=HR, noise=0, random_state=3,
                      method="ecgsyn")
ecg = (ecg - np.mean(ecg)) / np.std(ecg)
_, info = nk.ecg_peaks(ecg, sampling_rate=FS)
R = np.asarray(info["ECG_R_Peaks"])
R = R[(R > int(0.3 * FS))][:3]  # três picos R: dois batimentos completos
rr = int(np.median(np.diff(R)))
t = np.arange(len(ecg)) / FS


def ms(n):  # milissegundos -> amostras
    return int(round(n * FS / 1000))


fig = plt.figure(figsize=(7.1, 5.4))
ga = fig.add_axes([0.04, 0.58, 0.94, 0.34])
gb = fig.add_axes([0.06, 0.08, 0.40, 0.30])
gc = fig.add_axes([0.58, 0.08, 0.40, 0.30])

# ---------------------------------------------------------------- (a) segmentos
lo, hi = max(R[0] - ms(300), 0), min(R[2] + ms(250), len(ecg))
ga.plot(t[lo:hi], ecg[lo:hi], color="#222222", lw=1.0)
r0, r1 = R[0], R[1]
ymin = ecg[lo:hi].min()
top = ecg[R].max()
y_lane = ymin - 0.35          # base do sombreado; abaixo fica a faixa branca dos rótulos
y_lab = ymin - 0.50           # altura dos rótulos dos limites
y_rr = ymin - 1.75            # altura da seta do intervalo RR
ylow, yhigh = y_rr - 0.8, top + 0.7
fr = (y_lane - ylow) / (yhigh - ylow)
for k in range(2):
    ra, rb = R[k], R[k + 1]
    segs = [("QRS", ra - ms(80), ra + ms(100)),
            ("ST-T", ra + ms(100), ra + int(0.45 * (rb - ra))),
            ("pré-QRS", ra + int(0.45 * (rb - ra)), rb - ms(80))]
    for nome, a0, a1 in segs:
        ga.axvspan(t[a0], t[a1], ymin=fr, ymax=1, color=COR[nome], alpha=0.20, lw=0)
for r in R:
    ga.text(t[r], ecg[r] + 0.2, "pico R", ha="center", fontsize=7.5)
ga.annotate("", xy=(t[r0], y_rr), xytext=(t[r1], y_rr), arrowprops=dict(arrowstyle="<->", lw=0.9))
ga.text((t[r0] + t[r1]) / 2, y_rr - 0.12, "intervalo RR (de um pico R ao seguinte)", ha="center",
        va="top", fontsize=8)
# rótulos das ondas do primeiro batimento
p_idx = np.argmax(ecg[r1 - ms(260):r1 - ms(110)]) + r1 - ms(260)
t_idx = np.argmax(ecg[r0 + ms(100):r0 + ms(450)]) + r0 + ms(100)
ga.text(t[p_idx], ecg[p_idx] + 0.18, "onda P", ha="center", fontsize=7.5)
ga.text(t[t_idx], ecg[t_idx] + 0.18, "onda T", ha="center", fontsize=7.5)
s_idx = r0 + ms(135)
ga.text(t[s_idx], ecg[s_idx] + 0.12, "ST", ha="center", va="bottom", fontsize=7.5)
# limites dos segmentos do primeiro batimento: linha tracejada e rótulo na faixa branca,
# todos na mesma altura e encostados na sua linha (o primeiro à esquerda, os outros à direita)
lims = [(r0 - ms(80), "R − 80 ms ", "right"), (r0 + ms(100), " R + 100 ms", "left"),
        (r0 + int(0.45 * (r1 - r0)), " R + 0,45·RR", "left"),
        (r1 - ms(80), " R seguinte − 80 ms", "left")]
for v, rot, ha in lims:
    ga.plot([t[v], t[v]], [ymin - 1.0, yhigh], color="black", lw=0.6, ls=(0, (2, 2)))
    ga.text(t[v], y_lab, rot, ha=ha, va="top", fontsize=6.5)
ga.set_xlim(t[lo], t[hi - 1])
ga.set_ylim(ylow, yhigh)
ga.set_yticks([])
ga.set_xlabel("tempo (s)")
ga.set_title("(a) um batimento completo, de um pico R ao seguinte, dividido em três segmentos",
             loc="left")
hand = [plt.Rectangle((0, 0), 1, 1, color=COR[k], alpha=0.35) for k in ("pré-QRS", "QRS", "ST-T")]
ga.legend(hand, ["pré-QRS (onda P e intervalo PR)", "QRS", "ST-T (segmento ST e onda T)"],
          ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5)

# ---------------------------------------------------------------- (b) janela do QRS
r0 = R[1]
a0, a1 = r0 - ms(150), r0 + ms(250)
tm = (np.arange(a0, a1) - r0) / FS * 1000
gb.plot(tm, ecg[a0:a1], color="#222222", lw=1.0)
gb.axvspan(-80, 100, ymin=0.36, ymax=1, color=COR["QRS"], alpha=0.20, lw=0)
gb.plot([0, 0], [ecg[a0:a1].min() - 0.3, ecg[R].max()], color="black", lw=0.6, ls=(0, (2, 2)))
gb.text(0, ecg[R].max() + 0.25, "pico R", ha="center", fontsize=7)
ymin_b = ecg[a0:a1].min()
yb = ymin_b - 2.3
gb.annotate("", xy=(-80, yb), xytext=(100, yb), arrowprops=dict(arrowstyle="<->", lw=0.9))
gb.text(10, yb - 0.3, "janela do QRS: 180 ms", ha="center", va="top", fontsize=8)
yb2 = ymin_b - 0.8
gb.annotate("", xy=(-50, yb2), xytext=(70, yb2), arrowprops=dict(arrowstyle="<->", lw=0.9,
                                                                color="#444444"))
gb.text(10, yb2 - 0.3, "QRS de 120 ms", ha="center", va="top",
        fontsize=7, color="#444444")
gb.set_xlim(-150, 250)
gb.set_ylim(yb - 1.0, ecg[R].max() + 0.7)
gb.set_yticks([])
gb.set_xlabel("tempo em relação ao pico R (ms)")
gb.set_title("(b) janela do QRS: 80 ms antes e 100 ms depois", loc="left")

# ---------------------------------------------------------------- (c) fim do ST-T
bpm = np.linspace(40, 130, 200)
rr_s = 60 / bpm
frac = 0.44 / np.sqrt(rr_s)
gc.axhspan(0.40, 0.50, color=COR["ST-T"], alpha=0.18, lw=0)
gc.axhline(0.45, color=COR["ST-T"], lw=1.0, ls=(0, (3, 2)))
gc.plot(bpm, frac, color="#222222", lw=1.3)
for b_, nome, dy in [(60, "60 bpm", -0.045), (77, "77 bpm\n(mediana)", -0.055),
                     (100, "100 bpm", -0.065)]:
    f_ = 0.44 / np.sqrt(60 / b_)
    gc.plot([b_], [f_], "o", color="black", ms=3.5)
    gc.text(b_, f_ + dy * 0.0 + 0.012, f"{f_:.2f}", ha="center", va="bottom", fontsize=7)
    gc.text(b_, 0.345, nome, ha="center", va="bottom", fontsize=6.5)
gc.text(131, 0.455, "limite do ST-T usado: 0,45", ha="right", va="bottom", fontsize=7, color=COR["ST-T"])
gc.text(131, 0.405, "teste de robustez: 0,40 a 0,50", ha="right", va="bottom", fontsize=6.5,
        color=COR["ST-T"])
gc.set_xlim(40, 131)
gc.set_ylim(0.32, 0.66)
gc.set_xlabel("frequência cardíaca (bpm)")
gc.set_ylabel("fim do QT como fração do RR")
gc.set_title(r"(c) $QT_c = 0{,}44$ s pela fórmula de Bazett", loc="left")

fig.savefig(OUT / "esquema_regioes.pdf")
fig.savefig("/tmp/esquema_v10.png", dpi=170)
print("ok", rr, "RR amostras ->", 60 / (rr / FS), "bpm")
