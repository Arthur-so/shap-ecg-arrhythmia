# Run 3 — treino alinhado a Zhang et al. (2021), 40 épocas

- **Código:** commit `56f2375` (notebook `kaggle_runner.ipynb` com `EPOCHS = 40`,
  `LR = 1e-4`, `RUN_SHAP = False`, `RUN_QUANTUS = False`).
- **Execução:** Kaggle (GPU T4), 08/10/2026, ~105 s por época. Treino do zero;
  parou pelo limite de 40 épocas (melhor época: 39). Sem SHAP nem Consistency.
- **Mudanças em relação ao run 2:** lr 1e-4 (antes 1e-3), kernel 7 nos blocos
  (antes 3), pooling média+máximo (antes só média), aumento de dados (escala e
  deslocamento de linha de base) e limiar por classe otimizado no F1 da validação.
  Mantidos: z-score por derivação, BCE com `pos_weight`, divisão 70/20/10.

| F1-macro | Validação | Teste |
|---|---|---|
| Limiares por classe | 0,846 | **0,803** |
| Limiar fixo 0,5 | 0,789 | 0,771 |
| Run 2 (limiar 0,5) | 0,641 | 0,641 |
| Zhang et al. (2021), média de 10 folds | — | 0,813 |

F1 por classe (teste, limiares por classe): SNR 0,804 · AF 0,921 · IAVB 0,932 ·
BRE 0,809 · BRD 0,921 · CAP 0,683 · CVP 0,833 · STD 0,731 · STE 0,591.

Limiares (validação): SNR 0,95 · AF 0,83 · IAVB 0,87 · BRE 0,83 · BRD 0,94 ·
CAP 0,79 · CVP 0,93 · STD 0,37 · STE 0,97.

`f1_scores.json` contém também o histórico por época (F1 da validação com os
limiares otimizados e com 0,5). O checkpoint deste run está na saída da versão
correspondente do notebook no Kaggle (não há Kaggle Dataset de reuso).
