# Modelo B — verificação da referência (baselines) e dos sorteios do GradientSHAP

- **Código:** `src/baseline_check.py` (commit `c3e5420`).
- **Execução:** Kaggle (GPU T4), versão 21 do `shap-ecg-arrhythmia-runner`, 23/10/2026, 72 min,
  com modelo e mapas do dataset `arthurso/shap-ecg-modelob`. Nenhum erro.
- **Configuração:** 622 mapas; mesma semente do GradientSHAP em todas as versões; distância ℓ2
  entre mapas com norma 1 nas 36 regiões.
  - `atual`: 32 baselines de `explain.py` (sorteio do treino, semente 42: SNR=2 AF=4 IAVB=5 BRE=0
    BRD=14 CAP=2 CVP=5 STD=1 STE=0), 50 sorteios;
  - `normal`: 32 ECGs do treino com rótulo único SNR (semente 42), 50 sorteios;
  - `aleatoria`: outros 32 do treino (semente 43: SNR=6 AF=2 IAVB=6 BRE=1 BRD=7 CAP=3 CVP=2 STD=5
    STE=1), 50 sorteios;
  - `normal200`: referência normal com 200 sorteios (média de 4 blocos de 50).
- **Mapas `normal200`:** salvos fora do git em `output/modeloB_normal200/` (formato de `explain.py`).

## Resultado (médias, 36 regiões)

| Classe | Piso atual (50) | Atual × aleatória | Atual × normal (50) | Atual × normal (200) | Piso normal (50) | Piso normal (200) | Região principal igual (normal 200) |
|---|---|---|---|---|---|---|---|
| SNR | 0,412 | 0,522 | 0,993 | 1,011 | 0,512 | 0,260 | 29% |
| AF | 0,348 | 0,373 | 0,412 | 0,371 | 0,311 | 0,165 | 67% |
| IAVB | 0,358 | 0,362 | 0,400 | 0,374 | 0,303 | 0,152 | 66% |
| BRE | 0,233 | 0,301 | 0,372 | 0,318 | 0,300 | 0,150 | 65% |
| BRD | 0,317 | 0,362 | 0,408 | 0,390 | 0,245 | 0,119 | 92% |
| CAP | 0,455 | 0,516 | 0,541 | 0,488 | 0,413 | 0,209 | 47% |
| CVP | 0,321 | 0,358 | 0,372 | 0,341 | 0,290 | 0,150 | 76% |
| STD | 0,398 | 0,461 | 0,500 | 0,456 | 0,389 | 0,197 | 45% |
| STE | 0,353 | 0,408 | 0,436 | 0,423 | 0,355 | 0,152 | 57% |
| Global | 0,355 | 0,404 | 0,497 | 0,471 | 0,331 | 0,167 | 66% |

- **200 sorteios cortam o piso pela metade** (0,331 → 0,167), como previsto (erro ∝ 1/√n).
- **A referência normal muda os mapas além do ruído, mas moderadamente:** sem SNR, atual × normal (200)
  = 0,393 (cosseno médio 0,92), acima do piso atual (0,347) em 69% dos mapas; trocar só o sorteio do
  conjunto (aleatória) já muda 0,387.
- **SNR perde sentido com a referência normal:** soma média do mapa −1,5 (contra 16,2 na atual).

## Prévia das métricas sem modelo com os mapas `normal200` (`previa_normal200/`)

Consistency (k=2) e complexidade recalculadas localmente com os mapas `normal200` (mudam referência e
sorteios ao mesmo tempo). Consistency global 0,540 (ajustada 0,472); complexidade global 24,3 regiões
efetivas. Prévia, não resultado oficial: a decisão de adotar a nova referência é do orientador.
