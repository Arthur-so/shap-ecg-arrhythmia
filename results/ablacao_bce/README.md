# Ablação — BCE ponderada × BCE simples

- **Código:** commit `9cf200c`, notebook `notebooks/kaggle_ablation.ipynb`.
- **Execução:** Kaggle (GPU T4), 08–09/10/2026, uma única execução: pré-processamento
  uma vez e os dois treinos em sequência.
- **Configuração comum:** lr 1e-4, batch 32, até 60 épocas, early stopping com
  paciência 10, kernel 7, pooling média+máximo, aumento de dados, limiar por classe
  otimizado no F1 da validação. Só muda a perda:
  - **A:** BCE com `pos_weight` = negativos/positivos de cada classe no treino.
  - **B:** BCE simples (como no código de referência de Zhang et al., 2021).

| | Melhor época (parou em) | F1 val | **F1 teste (limiares)** | F1 teste (0,5) |
|---|---|---|---|---|
| A — ponderada | 50 (60) | 0,844 | **0,816** | 0,785 |
| B — simples | 36 (46) | 0,849 | **0,819** | 0,806 |
| Run 3 — ponderada, 40 épocas | 39 (40) | 0,846 | 0,803 | 0,771 |
| Zhang et al. (2021), média de 10 folds | — | — | 0,813 | — |

F1 por classe no teste (limiares por classe):

| | SNR | AF | IAVB | BRE | BRD | CAP | CVP | STD | STE |
|---|---|---|---|---|---|---|---|---|---|
| A | 0,823 | 0,923 | 0,909 | 0,816 | 0,918 | 0,736 | 0,812 | 0,771 | 0,636 |
| B | 0,796 | 0,931 | 0,884 | 0,870 | 0,927 | 0,750 | 0,837 | 0,778 | 0,596 |

**Conclusão:** com limiares por classe, as duas perdas são equivalentes (diferença de
0,003, abaixo da variação entre execuções com a mesma configuração, ≈ 0,01 — ver A ×
run 3). Adotou-se a **BCE simples (B)**: mais fiel à referência, probabilidades mais
bem calibradas (melhor F1 com limiar 0,5) e convergência mais rápida. A partir do
commit seguinte, `train.py` usa BCE simples por padrão (`--pos-weight` reativa a
ponderação) e 60 épocas.

Os checkpoints estão na saída do notebook `arthurso/kaggle-ablation` no Kaggle, em
`/kaggle/working/ablation/<experimento>/checkpoints/resnet34_1d_best.pt`.
