# Modelo B — Random Logit por regiões do batimento

- **Métrica:** `src/random_logit.py` (commit `3a81dec`), Random Logit de Sixt et al. (2020, §3, p. 5):
  o mapa da classe explicada é comparado com o mapa de cada outra classe no mesmo ECG.
- **Execução:** Kaggle (GPU T4), versão 20 do `shap-ecg-arrhythmia-runner`, 23/10/2026, 68 min,
  com modelo e mapas do dataset `arthurso/shap-ecg-modelob`. Nenhum erro, nenhum NaN.
- **Configuração:** 622 mapas (predições corretas do teste); GradientSHAP refeito com a configuração
  de `explain.py` (32 baselines, 50 sorteios, stdevs 0,09) e a mesma semente da sensibilidade, igual
  nos mapas das 9 classes; distância ℓ2 entre mapas com norma 1, nas 36 regiões e ponto a ponto;
  resultado principal com as classes ausentes nos rótulos do ECG (4 886 pares), presentes à parte
  (90 pares); piso = mesma distância entre duas sementes, sem trocar a classe.

| Arquivo | Conteúdo |
|---|---|
| `random_logit.csv` | Por classe explicada: distância às ausentes (média, mínima), piso, razão, fração acima do piso, distância às presentes |
| `random_logit_mapas.csv` | Valores de cada mapa, com a classe ausente mais parecida |
| `random_logit_pares.csv` | Cada par (mapa, outra classe): presente/ausente, logit da outra classe, distâncias |
| `random_logit_matriz.csv` | Distância média nas regiões por par (classe explicada × classe ausente) |
