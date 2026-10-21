# Modelo B — Faithfulness Estimate por regiões do batimento

- **Métrica:** `src/faithfulness_estimate.py` (commit `76c0dff`), §5.3 de
  Alvarez-Melis & Jaakkola (2018) com atributos = 36 regiões do batimento.
- **Execução:** Kaggle (GPU T4), versão 18 do `shap-ecg-arrhythmia-runner`, 20/10/2026,
  com modelo e mapas do dataset `arthurso/shap-ecg-modelob` (treino e SHAP pulados).
- **Configuração:** 622 mapas (predições corretas do teste); cada uma das 36 regiões
  retirada isoladamente; baselines `zero` e `mean` (média do treino); saídas `logit`
  (principal) e `prob` (sigmoide do logit, verificação). Nenhum valor indefinido (NaN).

| Arquivo | Conteúdo |
|---|---|
| `faithfulness_estimate.csv` | Média e desvio por classe, baseline e saída (n e n_nan) |
| `faithfulness_estimate_mapas.csv` | Correlação de cada mapa em cada configuração |
| `faithfulness_estimate_regioes.csv` | Cada retirada: derivação, região, soma das atribuições, quedas do logit e da probabilidade |
