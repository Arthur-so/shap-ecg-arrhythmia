# Modelo B — Complexidade por regiões do batimento

- **Métrica:** `src/complexity.py`, Definição 4 de Bhatt et al. (2020) (entropia das frações
  de |importância|) com atributos = 36 regiões do batimento; índice de Gini (fórmula da
  `quantus.Sparseness`) como verificação.
- **Execução:** local, na CPU, 23/10/2026, cerca de 10 s (não chama o modelo), com os mapas
  do modelo B (`output/modeloB/results/attributions`, os mesmos do dataset
  `arthurso/shap-ecg-modelob`).
- **Configuração:** 622 mapas (predições corretas do teste); importância da região = soma do
  mapa na região; número efetivo de regiões = e^H (1 a 36); piso do GradientSHAP de cada mapa
  lido de `results/modeloB_sens/sensitivity_mapas.csv`, como contexto.

```
python -m src.complexity --attr-dir output/modeloB/results/attributions \
    --results-dir results/modeloB_complexity \
    --sens-mapas results/modeloB_sens/sensitivity_mapas.csv
```

| Arquivo | Conteúdo |
|---|---|
| `complexity.csv` | Por classe: média e desvio da entropia, nº efetivo, Gini, fração nas 3 maiores regiões e piso |
| `complexity_mapas.csv` | Valores de cada mapa, com a região mais importante |
