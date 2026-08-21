# Avaliação das Explicações SHAP em Redes Neurais Profundas para Classificação de Arritmias Cardíacas em ECG

Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação, UNIFESP.

**Autor:** Arthur Santos de Oliveira  ·  **Orientador:** Prof. Dr. Luis Augusto Martins Pereira

## Objetivo

Treinar uma **ResNet34 1D** para classificação multi-rótulo de arritmias no
**CPSC2018**, gerar explicações **GradientSHAP** (Captum) sobre o conjunto de
teste, avaliá-las com o toolkit **Quantus** (Faithfulness e Robustness por
classe) e comparar essas métricas ao **F1-score por classe** do modelo,
investigando se a qualidade das explicações acompanha o desempenho preditivo.

O contexto teórico completo está em [`PROPOSTA_TCC.txt`](PROPOSTA_TCC.txt).

## Classes diagnósticas (CPSC2018)

`SNR`, `AF`, `IAVB`, `BRE`, `BRD`, `CAP`, `CVP`, `STD`, `STE` — 9 condições,
representação multi-rótulo (BCE).

## Estrutura do repositório

```
src/
├── config.py             # constantes globais (classes, 12x15000, seed)
├── metrics.py            # F1 por classe / macro
├── data/
│   ├── download.py       # download/organização do CPSC2018
│   ├── preprocess.py     # truncamento/padding, z-score, split estratificado
│   └── dataset.py        # PyTorch Dataset + pos_weight do BCE
├── model/
│   └── resnet34_1d.py    # arquitetura ResNet34 1D
├── train.py              # treino (Adam, BCE ponderada, early stopping F1-macro)
├── explain.py            # GradientSHAP via Captum
└── evaluate.py           # métricas Quantus (Faithfulness/Robustness)
notebooks/
└── kaggle_runner.ipynb   # runner fino para o Kaggle
kaggle/
└── kernel-metadata.json  # metadata para `kaggle kernels push`
results/                  # métricas agregadas, F1 por classe, csvs
```

> O treinamento pesado (GPU) roda no **Kaggle Notebooks**. Toda a lógica fica
> em módulos `.py` de `src/`; o notebook apenas clona/atualiza o repositório e
> chama os scripts via linha de comando. Dados brutos, dados pré-processados e
> checkpoints **não** são versionados (ver `.gitignore`).

## Fluxo de execução

```bash
# 1. Dependências
pip install -r requirements.txt

# 2. Dados (só baixa de fato com --confirm; pode ser grande)
python -m src.data.download --raw-dir data/raw --confirm

# 3. Pré-processamento -> data/processed/{train,val,test}.npz
python -m src.data.preprocess --raw-dir data/raw --out-dir data/processed

# 4. Treino -> checkpoints/ e results/f1_scores.json
python -m src.train --data-dir data/processed --out-dir checkpoints \
    --results-dir results --epochs 50 --lr 1e-3 --batch-size 32

# 5. Explicações GradientSHAP -> results/attributions/<CLASSE>.npz
python -m src.explain --data-dir data/processed \
    --checkpoint checkpoints/resnet34_1d_best.pt --out-dir results/attributions

# 6. Avaliação Quantus -> results/quantus_metrics.csv e quality_vs_f1.csv
python -m src.evaluate --data-dir data/processed \
    --checkpoint checkpoints/resnet34_1d_best.pt \
    --attr-dir results/attributions --results-dir results
```

Todos os scripts expõem `--help` com a lista completa de argumentos.

## Reprodutibilidade

- Semente fixa (`SEED = 42` em `src/config.py`) para split, baselines e treino.
- Split estratificado por classe 70% / 20% / 10%, com a partição de teste
  isolada até a avaliação final.

## Protocolo (Zhang et al., 2021)

- Sinais truncados/preenchidos até 15.000 amostras (30 s @ 500 Hz), 12 derivações.
- Normalização z-score por derivação.
- Adam + `BCEWithLogitsLoss` com `pos_weight` inverso à frequência de classe.
- Early stopping monitorando F1-macro na validação.
