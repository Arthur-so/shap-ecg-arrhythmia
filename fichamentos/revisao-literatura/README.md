# Revisão da literatura — avaliação de explicações SHAP em séries temporais e ECG

Pasta criada em 28/10–03/11/2026. Ordem sugerida de leitura:

1. `guia-de-leitura/main.pdf` — resumo dos fichamentos completos, ordem de leitura, conexões com o TCC e glossário (AUC, AUPRC, MoRF, ROAD...).
2. `relatorio/main.pdf` — relatório completo da revisão (fonte: `reports/Avaliação de SHAP em séries temporais.md`; notas em `research_notes/`).
3. `completos/` — fichamentos completos (Partes I–III, citações conferidas no original).
4. `curtos/` — fichamentos curtos, para consulta.

O texto inicial de Metodologia e Resultados está em `texto-tcc/metodologia-resultados/main.pdf`.

## Fichamentos completos

| Pasta | Referência | Leitura | Páginas |
|---|---|---|---|
| `arends-2026-atribuicao-ecg` | ARENDS, B. K. O.; VAN AMSTERDAM, W. A. C.; VAN DER HARST, P.; VAN SMEDEN, M.; VAN ES, R.; VAN DE LEUR, R. R. *Signal or noise? Evaluating commonly used attribution methods for explaining deep neural networks in electrocardiogram classification* | íntegra (versão publicada, texto completo do PMC12980500, sem paginação; material suplementar .docx com Tabelas S1–S3 e Figuras S1–S5) | 10 |
| `baer-2025-dados-sinteticos-classe` | BAER, G.; GRAU, I.; ZHANG, C.; VAN GORP, P. *Why Do Class-Dependent Evaluation Effects Occur with Time Series Feature Attributions? A Synthetic Data Investigation* | íntegra (arXiv v2, 24/7/2025, 16 páginas com apêndice) | 9 |
| `baer-2025-efeitos-dependentes-classe` | BAER, G.; GRAU, I.; ZHANG, C.; VAN GORP, P. *Class-Dependent Perturbation Effects in Evaluating Time Series Attributions* | íntegra (arXiv v2, 1/4/2025, 24 páginas com apêndice) | 8 |
| `gumpfer-2026-avaliacao-diretrizes-ecg` | GUMPFER, N.; GUCKERT, M.; SOSSALLA, S.; AẞMUS, B.; HANNIG, J. *Beyond Local Inspection: Global, Guideline-Grounded Evaluation of Post-hoc XAI Methods for ECG Classification* | íntegra do texto principal (arXiv v1, 17 páginas); material suplementar (Métodos Suplementares, Tabelas S1–S3, Figuras S1–S2) não incluído no PDF do arXiv e não consultado | 8 |
| `hollig-2023-xtsc-bench` | HÖLLIG, J.; THOMA, S.; GRIMM, F. *XTSC-Bench: Quantitative Benchmarking for Explainers on Time Series Classification* | íntegra (arXiv:2310.14957v1, 10 páginas, com o apêndice) | 8 |
| `ismail-2020-benchmark-series` | ISMAIL, A. A.; GUNADY, M.; CORRADA BRAVO, H.; FEIZI, S. *Benchmarking Deep Learning Interpretability in Time Series Predictions* | íntegra (arXiv:2010.13924v1, 32 páginas, com o material suplementar) | 9 |
| `mercier-2022-benchmark-atribuicao` | MERCIER, D.; BHATT, J.; DENGEL, A.; AHMED, S. *Time to Focus: A Comprehensive Benchmark Using Time Series Attribution Methods* | íntegra (arXiv:2202.03759v1, 12 páginas) | 8 |
| `ravindran-2023-eeg-verdade-simulada` | SUJATHA RAVINDRAN, A.; CONTRERAS-VIDAL, J. *An empirical comparison of deep learning explainability approaches for EEG using simulated ground truth* | íntegra (versão publicada, PDF da Scientific Reports, 20 páginas) | 9 |
| `storas-2025-mapas-calor-ecg` | STORÅS, A. M.; MÆLAND, S.; ISAKSEN, J. L.; HICKS, S. A.; THAMBAWITA, V.; GRAFF, C.; HAMMER, H. L.; HALVORSEN, P.; RIEGLER, M. A.; KANTERS, J. K. *Evaluating gradient-based explanation methods for neural network ECG analysis using heatmaps* | íntegra do artigo (versão publicada, 10 páginas); os apêndices A.1 a A.4, publicados como material suplementar, não foram acessados | 8 |
| `suh-2024-atribuicao-ecg` | SUH, J.; KIM, J.; KWON, S.; JUNG, E.; AHN, H.-J.; LEE, K.-Y.; CHOI, E.-K.; RHEE, W. *Visual interpretation of deep learning model in ECG classification: A comprehensive evaluation of feature attribution methods* | íntegra (versão publicada, 20 páginas, com os apêndices B a F; o material suplementar on-line não foi lido) | 10 |
| `turbe-2023-avaliacao-series` | TURBÉ, H.; BJELOGRLIC, M.; LOVIS, C.; MENGALDO, G. *Evaluation of post-hoc interpretability methods in time-series classification* | íntegra da versão publicada (acesso aberto; PDF de 24 páginas: artigo, Métodos, Extended Data e Reporting Summary); as Supplementary Information (hiperparâmetros, métricas de classificação, resultados do FordA e a derivação teórica da Seção 4) não foram lidas | 9 |
| `wagner-2024-auditoria-ecg` | WAGNER, P.; MEHARI, T.; HAVERKAMP, W.; STRODTHOFF, N. *Explaining deep learning for ECG analysis: Building blocks for auditing and knowledge discovery* | íntegra (arXiv v2, 25 páginas, com os apêndices; versão publicada não consultada) | 9 |

## Fichamentos curtos

| Pasta | Referência | Páginas |
|---|---|---|
| `atanasova-2020-diagnostico-texto` | ATANASOVA, P.; SIMONSEN, J. G.; LIOMA, C.; AUGENSTEIN, I. *A Diagnostic Study of Explainability Techniques for Text Classification* | 3 |
| `balagopalan-2022-justica-explicacoes` | BALAGOPALAN, A.; ZHANG, H.; HAMIDIEH, K.; HARTVIGSEN, T.; RUDZICZ, F.; GHASSEMI, M. *The Road to Explainability is Paved with Bias: Measuring the Fairness of Explanations* | 3 |
| `crabbe-2021-dynamask` | CRABBÉ, J.; VAN DER SCHAAR, M. *Explaining Time Series Predictions with Dynamic Masks* | 3 |
| `cui-2023-interpretacao-eeg` | CUI, J.; YUAN, L.; WANG, Z.; LI, R.; JIANG, T. *Towards best practice of interpreting deep learning models for EEG-based brain computer interfaces* | 3 |
| `dai-2022-disparidade-qualidade-explicacoes` | DAI, J.; UPADHYAY, S.; AÏVODJI, U.; BACH, S. H.; LAKKARAJU, H. *Fairness via Explanation Quality: Evaluating Disparities in the Quality of Post hoc Explanations* | 3 |
| `fresz-2024-metricas-classificacao-imagens` | FRESZ, B.; LOERCHER, L.; HUBER, M. F. *Classification Metrics for Image Explanations: Towards Building Reliable XAI-Evaluations* | 3 |
| `goettling-2024-xecgarch` | GOETTLING, M.; HAMMER, A.; MALBERG, H.; SCHMIDT, M. *xECGArch: a trustworthy deep learning architecture for interpretable ECG analysis considering short-term and long-term features* | 3 |
| `hicks-2021-ecgradcam-descoberta` | HICKS, S. A.; ISAKSEN, J. L.; THAMBAWITA, V.; GHOUSE, J.; AHLBERG, G.; LINNEBERG, A.; GRARUP, N.; STRÜMKE, I.; ELLERVIK, C.; OLESEN, M. S.; HANSEN, T.; GRAFF, C.; HOLSTEIN-RATHLOU, N.-H.; HALVORSEN, P.; MALECKAR, M. M.; RIEGLER, M. A.; KANTERS, J. K. *Explaining deep neural networks for knowledge discovery in electrocardiogram analysis* | 3 |
| `hooker-2019-roar-benchmark` | HOOKER, S.; ERHAN, D.; KINDERMANS, P.-J.; KIM, B. *A Benchmark for Interpretability Methods in Deep Neural Networks* | 3 |
| `jang-2026-execg-framework` | JANG, J.-H.; JO, Y.-Y. *ExECG: An Explainable AI Framework for ECG models* | 3 |
| `karimi-2023-explicacao-predicao-causal` | KARIMI, A.-H.; MUANDET, K.; KORNBLITH, S.; SCHÖLKOPF, B.; KIM, B. *On the Relationship Between Explanation and Prediction: A Causal View* | 3 |
| `krishna-2024-problema-desacordo` | KRISHNA, S.; HAN, T.; GU, A.; WU, S.; JABBARI, S.; LAKKARAJU, H. *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective* | 3 |
| `mancini-2024-parkinson-fala` | MANCINI, E.; PAISSAN, F.; TORRONI, P.; RAVANELLI, M.; SUBAKAN, C. *Investigating the Effectiveness of Explainability Methods in Parkinson's Detection from Speech* | 3 |
| `mayortorres-2023-eeg-roar` | MAYOR-TORRES, J. M.; MEDINA-DEVILLIERS, S.; CLARKSON, T.; LERNER, M. D.; RICCARDI, G. *Evaluation of Interpretability for Deep Learning algorithms in EEG Emotion Recognition: A case study in Autism* | 3 |
| `nauta-2023-revisao-avaliacao-xai` | NAUTA, M.; TRIENES, J.; PATHAK, S.; NGUYEN, E.; PETERS, M.; SCHMITT, Y.; SCHLÖTTERER, J.; VAN KEULEN, M.; SEIFERT, C. *From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI* | 3 |
| `nguyen-2024-amee-recomendacao` | NGUYEN, T. T.; LE NGUYEN, T.; IFRIM, G. *Robust Explainer Recommendation for Time Series Classification* | 3 |
| `peter-2026-frameworks-xai-series` | PETER, L.; GUMPFER, N.; FISCHER, J.; SEIFERT, C.; HANNIG, J. *Software Frameworks for Explainable AI in Time Series Classification: A Systematic Review* | 3 |
| `rojat-2021-survey-xai-series` | ROJAT, T.; PUGET, R.; FILLIAT, D.; DEL SER, J.; GELIN, R.; DÍAZ-RODRÍGUEZ, N. *Explainable Artificial Intelligence (XAI) on Time Series Data: A Survey* | 3 |
| `schlegel-2019-avaliacao-xai-series` | SCHLEGEL, U.; ARNOUT, H.; EL-ASSADY, M.; OELKE, D.; KEIM, D. A. *Towards a Rigorous Evaluation of XAI Methods on Time Series* | 3 |
| `serramazza-2023-explicacoes-multivariadas` | SERRAMAZZA, D. I.; NGUYEN, T. T.; LE NGUYEN, T.; IFRIM, G. *Evaluating Explanation Methods for Multivariate Time Series Classification* | 3 |
| `serramazza-2024-interprettime-mascaras` | SERRAMAZZA, D. I.; LE NGUYEN, T.; IFRIM, G. *Improving the Evaluation and Actionability of Explanation Methods for Multivariate Time Series Classification* | 3 |
| `simic-2025-perturbacao-series` | ŠIMIĆ, I.; VEAS, E.; SABOL, V. *A comprehensive analysis of perturbation methods in explainable AI feature attribution validation for neural time series classifiers* | 3 |
| `storas-2023-mapas-calor-ecg` | STORÅS, A. M.; ANDERSEN, O. E.; LOCKHART, S.; THIELEMANN, R.; GNESIN, F.; THAMBAWITA, V.; HICKS, S. A.; KANTERS, J. K.; STRÜMKE, I.; HALVORSEN, P.; RIEGLER, M. A. *Usefulness of Heat Map Explanations for Deep-Learning-Based Electrocardiogram Analysis* | 3 |
| `suh-2022-preprint-atribuicao-ecg` | SUH, J.; KIM, J.; JUNG, E.; RHEE, W. *Evaluating Feature Attribution Methods for Electrocardiogram* | 3 |
| `tempel-2024-shap-reconhecimento-atividades` | TEMPEL, F.; IHLEN, E. A. F.; ADDE, L.; STRÜMKE, I. *Explaining Human Activity Recognition with SHAP: Validating Insights with Perturbation and Quantitative Measures* | 3 |
| `zhang-2021-metricas-ppg` | ZHANG, O.; DING, C.; PEREIRA, T.; XIAO, R.; GADHOUMI, K.; MEISEL, K.; LEE, R. J.; CHEN, Y.; HU, X. *Explainability Metrics of Deep Convolutional Networks for Photoplethysmography Quality Assessment* | 3 |
| `zhou-2022-atribuicao-verdade-referencia` | ZHOU, Y.; BOOTH, S.; RIBEIRO, M. T.; SHAH, J. *Do Feature Attribution Methods Correctly Attribute Features?* | 3 |

Planejados mas não feitos (interrompidos pelo limite de uso; dispensados): fichamentos completos de Fouladgar 2022, Zhang 2021, Saporta 2022, Löffler 2025 e MetaQuantus 2023, e 18 fichamentos curtos. Os fichamentos das métricas do TCC (Bhatt, Alvarez-Melis, Yeh, Dasgupta, Sixt, Hedström) estão nas pastas próprias em `fichamentos/`.
