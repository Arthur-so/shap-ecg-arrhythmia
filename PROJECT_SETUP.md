# TCC — Avaliação das Explicações SHAP em Redes Neurais Profundas para Classificação de Arritmias Cardíacas em ECG

Autor: Arthur Santos de Oliveira | Orientador: Prof. Dr. Luis Augusto Martins Pereira
UNIFESP — Bacharelado em Ciência da Computação

Este arquivo contém as instruções para o Claude Code executar a estruturação
inicial do projeto de implementação do TCC. O contexto teórico completo do
trabalho está em `PROPOSTA_TCC.txt`, no mesmo diretório — leia esse arquivo
antes de gerar qualquer código, pois ele define arquitetura, dataset,
pré-processamento e métricas exatas a serem usadas.

## Objetivo geral do projeto

Treinar uma ResNet34 1D para classificação multi-rótulo de arritmias no
CPSC2018, gerar explicações SHAP (via GradientSHAP/Captum) sobre o conjunto
de teste, avaliá-las com o toolkit Quantus (Faithfulness e Robustness por
classe) e comparar essas métricas ao F1-score por classe do modelo.

## Fluxo de trabalho (importante)

O treinamento pesado (GPU) será feito no **Kaggle Notebooks**, não localmente.
Por isso o código deve ser escrito de forma que:

- Toda a lógica (modelo, preprocessing, dataset, treino, geração de SHAP,
  avaliação Quantus) fique em módulos `.py` dentro de `src/`, nunca em células
  de notebook.
- O notebook do Kaggle é apenas um "runner fino": clona/atualiza o
  repositório e chama scripts de `src/` via linha de comando.
- Dados brutos/pré-processados e checkpoints de modelo NÃO vão para o
  repositório git — apenas código, configs e resultados agregados (métricas,
  logs, tabelas).

## Passo a passo a executar

1. **Estruturar o repositório**
   Criar a seguinte estrutura, com um `.gitignore` apropriado (excluindo
   `data/`, `checkpoints/`, `*.pt`, `__pycache__/`, etc.):

   ```
   .
   ├── src/
   │   ├── data/
   │   │   ├── download.py        # baixa/organiza o CPSC2018
   │   │   ├── preprocess.py      # truncamento/padding, z-score, split
   │   │   └── dataset.py         # classe PyTorch Dataset
   │   ├── model/
   │   │   └── resnet34_1d.py     # arquitetura da ResNet34 1D
   │   ├── train.py               # loop de treino com early stopping
   │   ├── explain.py             # geração de GradientSHAP via Captum
   │   └── evaluate.py            # métricas Quantus (Faithfulness/Robustness)
   ├── notebooks/
   │   └── kaggle_runner.ipynb    # célula de git pull + chamada dos scripts
   ├── results/                   # métricas, F1 por classe, csvs de saída
   ├── kaggle/
   │   └── kernel-metadata.json   # metadata para `kaggle kernels push`
   ├── requirements.txt
   ├── README.md
   └── .gitignore
   ```

2. **`requirements.txt`**
   Incluir no mínimo: `torch`, `numpy`, `scipy`, `wfdb` (leitura de sinais
   PhysioNet), `scikit-learn`, `captum`, `quantus`, `pandas`, `tqdm`.

3. **Pré-processamento (`src/data/preprocess.py`)**
   Implementar conforme a seção 4.1.1 da proposta:
   - Truncamento/padding com zeros até 15.000 amostras por derivação.
   - Normalização z-score por derivação (média e desvio padrão por canal).
   - Split estratificado por classe: 70% treino / 20% validação / 10% teste,
     com seed fixa para reprodutibilidade.
   - Registros multi-rótulo mantidos com representação nativa (compatível com
     BCE).

4. **Modelo (`src/model/resnet34_1d.py`)**
   ResNet34 1D com 4 blocos residuais empilhados, cada um com 2 convoluções
   1D, 2 batch norms, dropout e skip connection, seguindo Zhang et al.
   (2021)/He et al. (2016). Saída: 9 valores com ativação sigmoide (uma por
   classe diagnóstica: SNR, AF, IAVB, BRE, BRD, CAP, CVP, STD, STE).

5. **Treino (`src/train.py`)**
   Otimizador Adam, loss BCE com pesos inversamente proporcionais à
   frequência de classe no treino, early stopping monitorando F1-macro na
   validação. Deve aceitar argumentos de linha de comando (epochs, lr,
   batch_size, caminho dos dados, diretório de saída dos checkpoints/logs) e
   salvar o F1-score por classe ao final em `results/`.

6. **Explicações (`src/explain.py`)**
   GradientSHAP via Captum, aplicado apenas às amostras corretamente
   classificadas do conjunto de teste, em relação à classe predita, usando
   baselines amostrados aleatoriamente do conjunto de treino. Salvar os mapas
   de atribuição (15.000 × 12 por amostra) de forma organizada por classe.

7. **Avaliação (`src/evaluate.py`)**
   Usar Quantus para calcular, por amostra e depois agregado por classe:
   - Faithfulness: Faithfulness Correlation, Faithfulness Estimate,
     Selectivity, SensitivityN, Infidelity, Sufficiency.
   - Robustness: Local Lipschitz Estimate, Max-Sensitivity, Avg-Sensitivity,
     Continuity, Consistency, Relative Input Stability (RIS), Relative Output
     Stability (ROS).
   Salvar tabela final com médias por classe + F1-score correspondente, para
   a análise de correlação descrita na seção 4.5 da proposta.

8. **Notebook Kaggle (`notebooks/kaggle_runner.ipynb`)**
   Célula 1: clonar/atualizar o repositório (`git clone` ou `git pull`).
   Célula 2: instalar `requirements.txt`.
   Células seguintes: chamar `python src/train.py ...`,
   `python src/explain.py ...`, `python src/evaluate.py ...` via `!`.

9. **`kaggle/kernel-metadata.json`**
   Configurar para permitir `kaggle kernels push`/`kaggle kernels output` a
   partir do terminal local, evitando upload manual pela interface web.

10. **Git**
    Inicializar repositório git local, criar `.gitignore`, primeiro commit
    com a estrutura acima. Perguntar ao usuário a URL do remoto GitHub antes
    de tentar dar push (não assumir que o repositório remoto já existe).

## Não fazer nesta etapa

- Não baixar o dataset completo automaticamente sem confirmação (pode ser
  grande e a rede do ambiente pode ter restrições).
- Não rodar treino real aqui — este ambiente não tem GPU. Apenas validar que
  os scripts rodam estruturalmente (ex.: forward pass com tensor sintético).
- Não subir dados/checkpoints para o git.
