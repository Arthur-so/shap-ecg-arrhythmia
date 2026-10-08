# Run 2 — 9 classes (CVP corrigida) + Consistency por regiões

- **Código:** commit `eac5f24` (notebook `kaggle_runner.ipynb`, `RUN_QUANTUS = False`).
- **Execução:** Kaggle (GPU), 07/10/2026. Treino do zero; early stopping na época 20,
  melhor época 10. F1-macro: validação 0,641 · teste 0,641.
- **Artefatos pesados** (checkpoint + atribuições + F1): Kaggle Dataset privado
  `arthurso/shap-ecg-run2` (reuso: `PRETRAINED = '/kaggle/input/shap-ecg-run2'`).

| Arquivo | Conteúdo |
|---|---|
| `f1_scores.json`, `f1_per_class.csv` | F1 por classe (validação e teste) e histórico de treino |
| `consistency.csv` | Consistency por classe: observada, acaso, ajustada e F1 (k = 2) |
| `consistency_explanations.csv` | Explicação discretizada e consistency local de cada mapa (651) |
| `consistency_sensibilidade.json` | Sensibilidade: k ∈ {1,2,3} × fim do ST-T ∈ {40,45,50}% do RR × janela do QRS ∈ {140,180,220} ms |

A análise de sensibilidade foi feita localmente com as funções de `src/consistency.py`
sobre as mesmas atribuições (picos R calculados uma vez por ECG).
