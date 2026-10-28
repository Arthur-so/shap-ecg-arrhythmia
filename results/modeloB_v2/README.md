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
| `complexity/` | Complexidade (entropia por regiões; Gini como verificação), com o piso da sensibilidade v2 | Local (CPU), 26/10/2026; piso acrescentado em 27/10/2026 |
| `faithcorr/` | Faithfulness Correlation (\|S\| = 4, 9, 18; baselines zero e média) | Kaggle, versão 23 do runner, 26/10/2026 |
| `faithest/` | Faithfulness Estimate (36 retiradas; logit e probabilidade) | Kaggle, versão 23 do runner, 26/10/2026 |
| `rlogit/` | Random Logit (mapas das 9 classes com a mesma semente; piso) | Kaggle, versão 23 do runner, 26/10/2026, 264 min |
| `sens/` | Max/Avg-Sensitivity (10 perturbações de raio 0,1; piso) | Kaggle, versão 24 do runner, 27/10/2026, 329 min |
| `emprt/` | eMPRT (modelo com pesos sorteados, 3 sementes; entropia por regiões, distâncias contra o piso, relação com a amplitude) | Kaggle, versão 25 do runner, 27/10/2026, 77 min |

Os arquivos de cada pasta têm o mesmo formato das pastas da versão 1 (ver os README de
`results/modeloB_faithcorr/`, `modeloB_faithest/`, `modeloB_rlogit/` e `modeloB_sens/`). Nenhum valor indefinido.

`emprt/` não tem versão 1. Arquivos: `emprt.csv` (média e desvio por classe), `emprt_mapas.csv`
(cada mapa, média das 3 sementes), `emprt_modelos.csv` (cada mapa × semente de randomização) e
`emprt_regioes.npz` (vetores das 36 regiões do mapa treinado `g_e`, dos 3 mapas do modelo aleatório
`g_r` e da amplitude Σ|x − x'| por região; posição = derivação·3 + região, NaN onde a região não existe).
Leitura em `fichamentos/hedstrom-2023-emprt/main.pdf`. A subpasta `emprt/histograma/` tem a verificação
com a entropia do artigo (histograma de 100 faixas dos valores; colunas `hist_e`, `hist_r`, `q_hist`), feita em
20 mapas por classe e uma semente (Kaggle, versão 26 do runner, 27/10/2026).

Resumo (média global):

| Métrica | Versão 2 | Versão 1 |
|---|---|---|
| Consistency | 0,455 | 0,416 |
| Complexidade (n.º efetivo de regiões) | 24,1 | 24,2 |
| Faithfulness Correlation (\|S\| = 9, baseline zero) | 0,600 | 0,606 |
| Faithfulness Estimate (logit, baseline zero) | 0,611 | 0,615 |
| Random Logit: distância às classes ausentes | 1,595 | 1,574 |
| Random Logit: piso | 0,183 | 0,355 |
| Max-Sensitivity (regiões) | 0,045 | 0,072 |
| Avg-Sensitivity (regiões) | 0,036 | 0,058 |
| Piso da sensibilidade (regiões) | 0,183 | 0,355 |
| eMPRT: q (aumento relativo da entropia) | −0,015 (37% > 0) | — |
| eMPRT: distância treinado × aleatório (com sinal / magnitudes) | 1,40 / 0,94 | — |
| eMPRT: piso | 0,183 | — |
| eMPRT, entropia do artigo (histograma; 174 mapas): q | −0,31 (14% > 0) | — |
