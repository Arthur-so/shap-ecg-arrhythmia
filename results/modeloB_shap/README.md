# Modelo B (BCE simples) — SHAP + Consistency

- **Modelo:** experimento B da ablação (`results/ablacao_bce/`): BCE simples, lr 1e-4,
  kernel 7, pooling média+máximo, aumento de dados, 60 épocas (melhor = 36), limiares por
  classe. F1-macro teste **0,819**.
- **Execução:** Kaggle (GPU T4), versão 16 do `shap-ecg-arrhythmia-runner`, 09/10/2026,
  código `d6c6e0c`, publicada pela CLI com `PRETRAINED` apontando para
  `kaggle-ablation/ablation/B_bce_simples` (treino pulado) e `RUN_SHAP = True`.
- **SHAP:** 583 ECGs do teste com ao menos uma classe corretamente predita (limiares do
  checkpoint) → 622 mapas. Os mapas ficam na saída da versão 16 no Kaggle.

| Arquivo | Conteúdo |
|---|---|
| `f1_scores.json` | F1 do modelo B (cópia do checkpoint reaproveitado) |
| `consistency.csv` | Consistency por classe (k = 2): observada, acaso, ajustada, F1 |
| `consistency_explanations.csv` | Explicação discretizada e consistency local de cada mapa |
| `consistency_sensibilidade.json` | k ∈ {1,2,3} × fim do ST-T ∈ {40,45,50}% do RR × janela do QRS ∈ {140,180,220} ms |

**Resumo (configuração base: QRS 180 ms, ST-T até 45% do RR):**

| k | Explicações distintas | Sem par | Consistency global | Esperado ao acaso |
|---|---|---|---|---|
| 1 | 44 | 1,9% | 0,436 | 0,15 |
| 2 | 196 | 17,5% | 0,416 | 0,13 |
| 3 | 376 | 46,1% | 0,307 | — |

Com k = 3 quase metade dos mapas fica sem par (não verificável); k = 2 já deixa 17,5%.
Comparação com o F1 feita de forma descritiva (posições por classe), sem coeficientes
de correlação: as 9 classes não são observações independentes nem uma amostra de uma
população de classes. Com k = 2, BRD fica em 1º e CAP e STE nas duas últimas posições
em todas as 9 combinações de regiões. O JSON de sensibilidade traz, por configuração,
a Consistency por classe (`per_classe`), o valor esperado ao acaso (`acaso_por_classe`)
e, só como registro, a versão ajustada e coeficientes de correlação.
A sensibilidade foi calculada localmente com `src/consistency.py` sobre os mapas da
versão 16 e o `data/processed/test.npz` local (record_ids conferidos: 0 divergências).
