# Modelo B — versão 2 dos mapas (referência balanceada, 200 sorteios)

Desde 26/10/2026 o TCC usa a **opção D** da verificação da referência
(`fichamentos/verificacao-referencia-gradientshap/main.pdf`): mesmos modelo B e 622 casos (ECG ×
classe corretamente predita), com os mapas GradientSHAP gerados assim (`src/gshap.py`):

- **Referência:** 4 ECGs de rótulo único de cada uma das 9 classes do treino (36), semente 42.
- **Sorteios:** 200 por mapa, como média de 4 blocos de 50; semente própria por mapa
  (`map_seed`), bloco j com semente + j·7919; stdevs 0,09.
- **Mapas:** gerados no Kaggle (versão 22 do runner, `src/baseline_balanced.py`, commit `4d07014`);
  publicados no dataset `arthurso/shap-ecg-modelob-v2` (com o checkpoint, o F1 e os dados
  processados); cópia local fora do git em `output/modeloB_balanceada200/`.

Os resultados da versão 1 (32 ECGs sorteados do treino, 50 sorteios) continuam nas pastas
`results/modeloB_*` como histórico.

| Pasta | Métrica | Execução |
|---|---|---|
| `consistency/` | Consistency (k = 2) | Local (CPU), 26/10/2026 |
| `complexity/` | Complexidade (entropia por regiões; Gini como verificação) | Local (CPU), 26/10/2026; o piso será acrescentado após a sensibilidade |
| `faithcorr/`, `faithest/`, `sens/`, `rlogit/` | Demais métricas | Kaggle (pendente) |
