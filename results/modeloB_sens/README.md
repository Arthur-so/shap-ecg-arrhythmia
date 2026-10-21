# Modelo B — Max-Sensitivity e Avg-Sensitivity

- **Métrica:** `src/sensitivity.py` (commit `ecc19ce`): Max-Sensitivity de Yeh et al. (2019,
  Def. 3.1) e Avg-Sensitivity de Bhatt et al. (2020, Def. 2).
- **Execução:** Kaggle (GPU T4), versão 19 do `shap-ecg-arrhythmia-runner`, 21/10/2026, 83 min,
  com modelo e mapas do dataset `arthurso/shap-ecg-modelob`.
- **Configuração:** 622 mapas (predições corretas do teste); 10 perturbações por mapa com ruído
  U(-0,1; 0,1) só no trecho real; GradientSHAP refeito com a configuração de `explain.py`
  (32 baselines, 50 amostras, stdevs 0,09) e a mesma semente na referência e nos ECGs
  perturbados; perturbações que mudam a predição descartadas (1 de 6 220); distância ℓ2 entre
  explicações com norma 1, nas 36 regiões e ponto a ponto; piso = mesma distância entre duas
  execuções do GradientSHAP com sementes diferentes, sem perturbação.

| Arquivo | Conteúdo |
|---|---|
| `sensitivity.csv` | Por classe: Max, Avg e piso (regiões e pontos), média e desvio, descartes |
| `sensitivity_mapas.csv` | Valores de cada mapa |
