# Modelo B — referência balanceada por classe e referência atual, com 200 sorteios

- **Código:** `src/baseline_balanced.py` (commit `4d07014`), continuação de `src/baseline_check.py`.
- **Execução:** Kaggle (GPU T4), versão 22 do `shap-ecg-arrhythmia-runner`, 25–26/10/2026, ~85 min,
  com modelo e mapas do dataset `arthurso/shap-ecg-modelob`. Nenhum erro.
- **Configuração:** 622 mapas, mesmas sementes de `baseline_check.py`; 200 sorteios = média de 4
  blocos de 50. Referências: atual (32 do treino, semente 42) e balanceada (4 ECGs de rótulo único de
  cada classe, 36, semente 42). Distância ℓ2 entre mapas com norma 1 nas 36 regiões.
- **Mapas** `atual200` e `balanceada200` fora do git, em `output/modeloB_atual200/` e
  `output/modeloB_balanceada200/`.
- **Análise completa:** `fichamentos/verificacao-referencia-gradientshap/main.pdf` (Parte IV).

| Arquivo | Conteúdo |
|---|---|
| `baseline_balanced.csv` | Por classe: atual × atual200, atual × balanceada200, atual200 × balanceada200, piso da balanceada (200), somas, região principal igual |
| `baseline_balanced_mapas.csv` | Valores de cada mapa |
| `previa_200/consistency_*.csv`, `complexity_*.csv` | Consistency (k=2) e complexidade recalculadas com os mapas atual200 e balanceada200 |
| `previa_200/distancias_200_mapas.csv` | Distâncias entre os mapas de 200 sorteios: atual × normal (CB), atual × balanceada (CD), balanceada × normal (DB), mapa salvo × atual200 (AC) |

Resumo: só os sorteios mudam pouco os mapas (0,21) e as métricas (Consistency 0,42 → 0,44); a
referência balanceada muda os mapas além do ruído, mas pouco (0,31 contra piso de 0,18), mantém SNR
interpretável e corrige o excesso de BRD; Consistency 0,46; complexidade igual.
