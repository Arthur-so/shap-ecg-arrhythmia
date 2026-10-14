# Modelo B — Faithfulness Correlation por regiões do batimento

- **Métrica:** `src/faithfulness_correlation.py` (commit `baf7d4f`), Definição 3 de
  Bhatt, Weller & Moura (2020) com atributos = 36 regiões do batimento.
- **Execução:** Kaggle (GPU T4), versão 17 do `shap-ecg-arrhythmia-runner`, 14/10/2026,
  com modelo e mapas do dataset `arthurso/shap-ecg-modelob` (treino e SHAP pulados).
- **Configuração:** 622 mapas (predições corretas do teste), 50 subconjuntos por mapa,
  |S| ∈ {4, 9, 18} regiões, baselines `zero` e `mean` (média do treino), mesmos
  subconjuntos nas duas baselines. Nenhum valor indefinido (NaN).

| Arquivo | Conteúdo |
|---|---|
| `faithfulness_correlation.csv` | Média e desvio por classe, baseline e \|S\| (n e n_nan) |
| `faithfulness_correlation_mapas.csv` | Correlação de cada mapa em cada configuração |
