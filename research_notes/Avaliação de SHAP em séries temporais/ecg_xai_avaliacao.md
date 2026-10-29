# Avaliação quantitativa de explicações (SHAP e atribuições) em classificação de ECG

Notas de pesquisa (busca feita em 27/10/2026). Legenda: **íntegra = sim** quando o texto completo foi lido (total ou leitura dirigida ao texto extraído); **parcial** quando só parte do texto foi lida; **não (só resumo)** quando só resumo, metadados ou resumos secundários estavam acessíveis. Os trabalhos marcados "não" devem ser baixados pelo estudante antes de qualquer citação de detalhe.

## 1. Quais trabalhos de ECG avaliam explicações com métricas quantitativas, e quais métricas?

### Takeaway
A avaliação quantitativa em ECG ainda é rara e heterogênea. Os trabalhos encontrados usam variantes de deleção/perturbação (curvas MoRF/LeRF, top-k/bottom-k, ROAD modificado, interpolação), randomização em cascata, autoconsistência entre sementes, localização contra máscaras (batimentos anômalos, máscaras derivadas de diretrizes) e entropia espectral. **Nenhum trabalho de ECG encontrado usa Quantus nem as métricas específicas do TCC** (Faithfulness Correlation, Faithfulness Estimate, Max/Avg-Sensitivity, Complexity de Bhatt, Consistency de Dasgupta, Random Logit, eMPRT). Os mais comparáveis ao TCC são Arends et al. 2026 (12 métodos, GradientSHAP incluído, randomização e perturbação), Turbé et al. 2023 (CPSC/RBBB, GradientSHAP com baseline de 50 amostras), Gumpfer et al. 2026 (13 métodos, GradSHAP, segmentos clínicos por patologia) e Suh et al. 2022/2024 (deleção MoRF/LeRF em janelas).

### Cited Findings

#### Panorama
- Uma revisão de avaliação de XAI em cardiologia relata que 43% dos estudos não avaliam a explicação, 37% só comparam com a literatura, 11% usam clínicos e 11% usam proxies ou análise estatística — Salih et al. 2024, *Artificial Intelligence Review* 57(9), DOI 10.1007/s10462-024-10852-w; **íntegra: não (só resumo)**. O total de estudos revisados não foi confirmado (o número "213" apareceu só num resumo secundário). — [PMC/PubMed 39132011](https://pmc.ncbi.nlm.nih.gov/articles/pmid/39132011); [Springer](https://link.springer.com/article/10.1007/s10462-024-10852-w)
- Quantus organiza as métricas em seis grupos (fidelidade, robustez, localização, complexidade, randomização, axiomáticas). As buscas por "Quantus" com ECG, PTB-XL e eletrocardiograma não retornaram nenhum estudo de ECG. — [Quantus GitHub](https://github.com/understandable-machine-intelligence-lab/Quantus)

#### Catálogo (mais comparáveis ao TCC primeiro)

**(A) Arends BKO, van Amsterdam WAC, van der Harst P, van Smeden M, van Es R, van de Leur RR. 2026. "Signal or noise? Evaluating commonly used attribution methods for explaining deep neural networks in electrocardiogram classification." *European Heart Journal – Digital Health* 7(3):ztag038. DOI 10.1093/ehjdh/ztag038. Íntegra: parcial (~86% do texto via WebFetch, sem o material suplementar).** — [OUP](https://academic.oup.com/ehjdh/article/7/3/ztag038/8512917)
- Dados: 873.710 ECGs de 12 derivações (batimento mediano) de 192.257 pacientes do UMC Utrecht; nove classes (ritmo sinusal, FA, bradicardia e taquicardia sinusal, BAV de 1º grau, BRD, BRE, baixa voltagem do QRS, HVE); teste de 1.351 ECGs.
- Modelo: ResNet-1D convolucional de 22 camadas, kernel 7, oito derivações de entrada (I, II, V1–V6). Um classificador binário por classe × 10 sementes = 90 modelos. Todas as AUROC > 0,95. AUPRC média > 0,85 em oito classes, mas só 0,64 na bradicardia sinusal.
- Métodos (Captum): Saliency, SmoothGrad, Gradient×Input, Guided Backprop, GradCAM, Guided GradCAM, Shapley sampling, DeepLift, IG, IG+SmoothGrad, **GradientSHAP**, DeepLiftSHAP.
- Métricas:
  1. Similaridade entre métodos (Pearson após filtro Savitzky–Golay).
  2. Autoconsistência entre modelos de sementes diferentes.
  3. Randomização em cascata das camadas, a partir da saída, num modelo de FA.
  4. Perturbação no estilo ROAD modificado ("remove and debias"): substitui o 1% de pontos mais relevantes por passo, em 30 passos (30% do sinal), com interpolação linear entre vizinhos mais ruído gaussiano (σ = 0,25). A unidade é o ponto amostral, sem segmentação por onda ou derivação.
- Resultados:
  - Correlação entre métodos baseados em baseline de 0,34 (DeepLiftSHAP) a 0,96 (IG+SmoothGrad × GradientSHAP).
  - Autoconsistência média de 0,52 ± 0,14, de 0,29 (Guided GradCAM) a 0,82 (SmoothGrad).
  - Na randomização, a maioria dos métodos cai para correlação ≈ 0, acompanhando a AUROC. Saliency, SmoothGrad e Guided Backprop caem mais devagar e não chegam a zero.
  - Na perturbação, a AUROC cai para ≈ 0,50 com SmoothGrad, ≈ 0,65 com DeepLiftSHAP, G×I e Saliency, ≈ 0,75 com a maioria dos outros e só até 0,92 com GradCAM.
- Conclusão (verbatim): "Despite extensive testing, these methods fail to meet key XAI criteria"; "If attribution methods are still used, they should be treated as exploratory tools rather than definitive explanations."

**(B) Turbé H, Bjelogrlic M, Lovis C, Mengaldo G. 2023. "Evaluation of post-hoc interpretability methods in time-series classification." *Nature Machine Intelligence* 5(3):250–260. DOI 10.1038/s42256-023-00620-w; arXiv 2202.05656. Íntegra: parcial (~87% do HTML do arXiv).** — [arXiv HTML](https://arxiv.org/html/2202.05656v2)
- Dados: subconjunto CPSC da PhysioNet/CinC Challenge 2020, tarefa binária BRD (6.877 ECGs: 1.857 BRD), cada derivação reduzida a batimento médio.
- Modelos: Bi-LSTM, CNN, Transformer.
- Métodos: DeepLift, **GradShap**, IG, KernelShap, DeepLiftShap, Shapley Value Sampling.
- Métricas:
  - AUC-S̃top (área da curva top-k) e F1-S̃ (média harmônica do desempenho top e bottom).
  - A corrupção troca pontos por valores de N(0,1). Ela também é aplicada no treino, o que evita o deslocamento de distribuição sem retreinar.
  - Baselines: média por instante de tempo (IG, DeepLift, Shapley, KernelShap) e **50 amostras aleatórias do teste (GradShap, DeepLiftShap)**.
- Resultados no ECG: os melhores foram Shapley (Bi-LSTM 0,348; Transformer 0,619) e DeepLiftShap (CNN 0,465). KernelShap foi o pior em todas as arquiteturas, perto do aleatório (CNN: 0,025 contra 0,047 aleatório).
- Conclusões: "Shapley Value Sampling consistently outperforms the other interpretability methods" (exceto CNN no ECG); o modelo "relies almost entirely on a single lead" para BRD.

**(C) Gumpfer N, Guckert M, Sossalla S, Aßmus B, Hannig J. 2026. "Beyond Local Inspection: Global, Guideline-Grounded Evaluation of Post-hoc XAI Methods for ECG Classification." arXiv:2607.24035 (preprint). Íntegra: sim (texto extraído do PDF, leitura dirigida).** — [arXiv](https://arxiv.org/pdf/2607.24035)
- Dados: PTB-XL, quatro CNNs binárias (BAV, isquemia, BRD, BRE), AUROC 0,93–0,99, só verdadeiros positivos, batimentos centrados no R.
- Métodos: 13 gradientes, entre eles Gradient, G×Input, SmoothGrad, VarGrad, IG, DeepSHAP, **GradSHAP**, LRP-ε/αβ e variantes SIGN.
- Métricas:
  - Correlação de Spearman entre |relevância| e |amplitude|.
  - Coverage: fração da relevância positiva dentro de máscaras derivadas de diretrizes (intervalo PR para BAV, ST nas precordiais para isquemia, QRS para BRD e BRE).
  - Coverage normalizada pelo acaso (NCov) e pior caso entre patologias (NCovmin).
- Resultados:
  - Correlação média com a amplitude: IG 0,69; LRP-ε 0,67; SmoothGrad×Input 0,65; G×I 0,57; **GradSHAP 0,33**; DeepSHAP 0,32.
  - NCov do GradSHAP: BAV 32,6%, isquemia −5,5%, BRD 80,9%, BRE 75,8% (média 45,9%).
  - DeepSHAP chega a −37,7% na isquemia, IG a −14,6% e LRP-ε a −54,0%, com só 4,6% da relevância no ST contra 63,8% no LRP-ε/SIGN.
  - Nove dos 13 métodos ficam abaixo do acaso em pelo menos uma patologia.
- Conclusão (verbatim): "Integrated Gradients, DeepSHAP, GradSHAP, and LRP-ε, tie relevance to input magnitude through input multiplication or baseline subtraction and shift attribution toward high-amplitude waveform components"; "Popularity is therefore a poor guide to reliability". Os autores avisam que avaliam plausibilidade e não fidelidade: "A high coverage value nevertheless does not certify faithfulness".

**(D) Suh J, Kim J, Jung E, Rhee W. "Evaluating Feature Attribution Methods for Electrocardiogram." arXiv:2211.12702 (v1 2022, v2 out/2024; venue não identificado). Íntegra: sim (HTML do arXiv).** — [arXiv HTML](https://arxiv.org/html/2211.12702)
- Dados: Icentia11K, uma derivação, 250 Hz, janelas de ≈ 8 s, classes normal/PAC/PVC.
- Modelo: ResNet-18 1D, acurácia 0,90.
- Métodos: 11, entre eles **DeepSHAP**, KernelSHAP, LIME, IG, LRP e Grad-CAM.
- Métricas:
  - Localização: IoU dos top-n pontos com os batimentos anômalos anotados.
  - Pointing game.
  - Degradation score: área entre as curvas MoRF e LeRF, com janelas de 16 amostras substituídas pela média da janela.
  - Só entram PAC/PVC preditos com p > 0,9.
- Resultados (localização / pointing / degradação):
  - Grad-CAM 0,485 / 0,705 / 0,888, o melhor;
  - IG 0,224 / 0,684 / 0,885;
  - DeepSHAP 0,213 / 0,556 / 0,883;
  - KernelSHAP 0,116 / 0,398 / 0,237;
  - aleatório 0,109 / 0,186 / 0,004.
- Conclusão: "Grad-CAM outperforms the other methods including the ones that are known to perform well in the image domain."

**(E) Wagner P, Mehari T, Haverkamp W, Strodthoff N. 2024. "Explaining deep learning for ECG analysis: Building blocks for auditing and knowledge discovery." *Computers in Biology and Medicine* 176:108525. arXiv 2305.17043. Íntegra: parcial (~70% do HTML, que cobre Métodos, Resultados e Discussão).** — [arXiv HTML](https://arxiv.org/html/2305.17043v2)
- Dados e modelos: PTB-XL (23 subdiagnósticos, 100 Hz, recortes de 2,5 s); LeNet-1D e xresnet1d50.
- Métodos: GradCAM, Saliency, IG, LRP.
- Sanity check: modelos de regressão que preveem as amplitudes de P, R e T em cada derivação. A especificidade espacial mede a fração da relevância na derivação-alvo. A especificidade temporal usa janelas de −300 a +500 ms em torno do R.
- Agregação clínica: um U-Net segmenta 24 segmentos (ondas por derivação), a relevância é ponderada por segmento e há TCAV com conceitos clínicos.
- Resultado: "saliency is the single XAI method, which satisfies this condition in all six experiments". IG, LRP e GradCAM colocam muita relevância no pico R mesmo quando ele não é o alvo.
- Conclusões (§4): "GradCAM, the most widely used attribution method in the field of ECG analysis, fails to satisfy the sanity check." Não há métricas de deleção nem de fidelidade por perturbação.

**(F) Goettling M, Hammer A, Malberg H, Schmidt M. 2024. "xECGArch: a trustworthy deep learning architecture for interpretable ECG analysis considering short-term and long-term features." *Scientific Reports* 14. DOI 10.1038/s41598-024-63656-x. Íntegra: sim (PMC).** — [PMC11161651](https://pmc.ncbi.nlm.nih.gov/articles/PMC11161651/)
- Dados: detecção de FA na derivação II, 10 s; 9.854 ECGs de PTB-XL, Georgia, CPSC2018 e Chapman.
- Modelo: duas CNNs, uma de curto e outra de longo prazo.
- Métodos: 13, entre eles SHAP, IG, LRP, DTD e GradCAM.
- Métrica: AUC relativa da perda contra a fração de amostras perturbadas, por ordem de relevância. Compara substituição por **interpolação linear** (proposta) com substituição por **zero**, e usa ordem aleatória como controle.
- Resultados com interpolação:
  - DTD 0,546 (curto) e 0,817 (longo);
  - LRP-αβ 0,248 e 0,852;
  - aleatório 0,129 e 0,087.
- Com zeros, o ranking muda: G×I é o melhor (0,529 e 0,712).
- Os autores dizem que a substituição por zero introduz ruído porque o modelo nunca viu saltos abruptos para zero.

**(G) Storås AM, Mæland S, Isaksen JL, Hicks SA, Thambawita V, Graff C, Hammer HL, Halvorsen P, Riegler MA, Kanters JK. 2025. "Evaluating gradient-based explanation methods for neural network ECG analysis using heatmaps." *JAMIA* 32(1):79–88. DOI 10.1093/jamia/ocae280. Íntegra: não (só resumo; texto pago).** — [OUP](https://academic.oup.com/jamia/article/32/1/79/7876016)
- Modelo: ResNet que regride intervalos e amplitudes do ECG.
- Métodos: nove, incluindo **Gradient SHAP**, IG, DeepLIFT, SmoothGrad e GradCAM.
- Avaliação: especialistas médicos (qualitativa) mais uma avaliação objetiva por perturbação.
- Resumo: "No single explanation method consistently outperformed the other methods… considerable disagreement between the human expert evaluation and the objective evaluation by perturbation. The best explanation method depended on the ECG measure."
- Baseline, janelas e números não foram verificados.

**(H) Abdullah, Fatima Z, Abadian A, Sánchez Mejorada CG, Torres Ruiz MJ, Quintero Téllez R. 2026. "Integration of Machine Learning Techniques in ECG-Based Multiclass Arrhythmia Classification with Explainability Analysis." *Biosensors* 16(6):326. DOI 10.3390/bios16060326. Íntegra: parcial (~63%; conclusão não lida).** — [PMC13296925](https://pmc.ncbi.nlm.nih.gov/articles/PMC13296925/)
- Dados e modelo: MIT-BIH, cinco classes AAMI, batimentos de 187 amostras, CNN própria.
- Métodos: Grad-CAM e IG.
- Métricas e resultados:
  - Inserção AUC 0,89 e deleção 0,21, contra 0,51 e 0,48 com atribuição aleatória.
  - Randomização de parâmetros: Spearman cai de 1,00 para 0,18.
  - Ruído de 10 dB: a fração da atribuição no QRS passa de 61,4% para 58,9%.
- A substituição usada na deleção não está descrita. Há inconsistências internas nos F1 por classe.

**(I) Kelly DA, Blake N. 2026. "Quantifying Explainable AI-introduced signal noise on ECG data with Spectral Entropy." arXiv:2606.24974 (aceito no EUSIPCO 2026). Íntegra: sim (HTML).** — [arXiv HTML](https://arxiv.org/html/2606.24974)
- Dados e modelo: MIT-BIH, 17 classes, CNN-1D de 17 camadas.
- Métodos: Grad-CAM, LIME, DeepLIFT, **GradientShap**, KernelShap, ReX.
- Métrica: entropia de Shannon da densidade espectral do mapa, só com valores positivos. É uma métrica de "ruído" e se aproxima da ideia de complexidade.
- Média ± DP nas classes:
  - GradientShap 9,16 ± 0,33;
  - KernelShap 10,21 ± 0,04;
  - DeepLIFT 9,04 ± 0,39;
  - Grad-CAM 1,07 ± 0,58;
  - ReX 1,93 ± 1,20.
- Conclusão: "Spectral entropy can be used alongside other quantitative measures".

**(J) Lopes JF, Bressan FT, Bressan RS, Sanches DS. 2026. "Comparative Evaluation of Deep Learning Models and Explainability Methods for ECG Heartbeat Classification." *RECIMA21* 7(10). DOI 10.47820/recima21.v7i10.9152. Íntegra: não (só resumo).** — [RECIMA21](https://recima21.com.br/recima21/en/article/view/9152)
- Dados e modelos: MIT-BIH, cinco classes, 1D-CNN (Macro-F1 0,868) e CNN-BiLSTM-Attention (0,814).
- Métodos: Grad-CAM-1D, IG e atenção temporal.
- Resultado: IG teve "positive mean faithfulness gains across all classes for both architectures"; Grad-CAM e a atenção foram mais dependentes da classe. Concordância entre métodos baixa a moderada.

**(K) Neves I, Folgado D, Santos S, Barandas M, Campagner A, Ronzio L, Cabitza F, Gamboa H. 2021. "Interpretable heartbeat classification using local model-agnostic explanations on ECGs." *Computers in Biology and Medicine* 133:104393. DOI 10.1016/j.compbiomed.2021.104393. Íntegra: não (só resumo; PDFs institucionais bloqueados com 403/anti-bot).** — [BOA Unimib](https://boa.unimib.it/handle/10281/324847)
- Dados: MIT-BIH.
- Métodos: LIME/SHAP agnósticos, com o sinal e sua derivada para dar dependência temporal.
- Avaliação: queda de desempenho ao perturbar, índice de Jaccard 1-D contra subsequências de um modelo interpretável e estudo com usuários.
- Um resumo secundário (não verificado) diz que a substituição por média e por zero deu fidelidade semelhante no LIME.

**(L) Beck J, John A. 2025. "Explainable AI (XAI) for Arrhythmia detection from electrocardiograms." arXiv:2508.17294 (preprint). Íntegra: sim (HTML).** — [arXiv HTML](https://arxiv.org/html/2508.17294v1)
- Dados e modelo: MIT-BIH mais o 12-lead de Zheng et al.; CNN-1D; batimentos de −200 a +400 ms do R.
- Métodos: permutation SHAP, KernelSHAP, **GradientSHAP**, DeepLIFT/DeepSHAP.
- Avaliação **sem métricas quantitativas** de explicação: análise visual e cinco clínicos.
- KernelSHAP e permutation geram mapas "cluttered". As explicações da classe mais fraca (batimento atrial prematuro) não se alinham à clínica.

#### Trabalhos frequentemente citados que *não* fazem avaliação quantitativa da explicação (verificado)
- **Zhang D, Yang S, Yuan X, Zhang P. 2021. "Interpretable deep learning for automatic diagnosis of 12-lead electrocardiogram." *iScience* 24(4):102373. DOI 10.1016/j.isci.2021.102373. Íntegra: sim (PMC, sem suplemento).** — [PMC8082080](https://pmc.ncbi.nlm.nih.gov/articles/PMC8082080)
  - CPSC2018, nove classes, CNN-1D (a arquitetura está no suplemento; código em github.com/onlyzdd/ecg-diagnosis).
  - SHAP por paciente e "contribution rate" por derivação e por classe.
  - Nenhuma métrica de fidelidade. Validação qualitativa por um clínico.
  - F1 por classe: SNR 0,805, AF 0,919, IAVB 0,864, LBBB 0,866, RBBB 0,926, PAC 0,735, PVC 0,851, STD 0,814, STE 0,535 (média 0,813).
  - Verbatim: "the deep learning model could make wrong predictions, and the SHAP method could learn wrong interpretations."
- **Bender T, Beinecke JM, … Spicher N, et al. 2023. "Analysis of a Deep Learning Model for 12-Lead ECG Classification Reveals Learned Features Similar to Diagnostic Criteria." *IEEE JBHI*. DOI 10.1109/JBHI.2023.3271858; arXiv 2211.01738. Íntegra: sim (PDF do arXiv, leitura dirigida).** — [arXiv](https://arxiv.org/pdf/2211.01738)
  - Modelo pré-treinado de Ribeiro et al. sobre CPSC2018 e PTB-XL.
  - Métodos: IG e quatro variantes de LRP.
  - Propõe "quantitative methods" que são **agregações** (relevância média por classe, por derivação e por "average beat"), não métricas de fidelidade.
  - Conclui que "IG gives most distinct results" e que o modelo "learned features similar to cardiology textbook knowledge" (§ Discussão 1 e 4).
  - Perturbação e oclusão são citadas, mas não usadas.
- **Strodthoff N, Wagner P, Schaeffter T, Samek W. 2021. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE JBHI*. DOI 10.1109/JBHI.2020.3022989; arXiv 2004.13701. Íntegra: sim (seção de interpretabilidade).** — [arXiv](https://arxiv.org/pdf/2004.13701)
  - LRP-ε (ε = 0,1) em ResNet, com dois exemplos (PVC e marca-passo) só visuais.
  - Propõe como trabalho futuro uma "more extensive, statistical analysis of the attribution maps".
- **Hicks SA, Isaksen JL, Thambawita V, … Riegler MA, Kanters JK. 2021. "Explaining deep neural networks for knowledge discovery in electrocardiogram analysis." *Scientific Reports* 11:10949. DOI 10.1038/s41598-021-90285-5. Íntegra: sim (PMC).** — [PMC8154909](https://pmc.ncbi.nlm.nih.gov/articles/PMC8154909/)
  - Propõe ECGradCAM, com mapas validados de forma qualitativa.
  - O único teste quantitativo é o "wave blanking": ondas substituídas por interpolação linear. Na classificação de sexo, sem o QRS a acurácia vai a 62,4%; sem a onda P, a 87,5%; sem a onda T, a 79,8%.
- **Storås AM et al. 2023. "Usefulness of Heat Map Explanations for Deep-Learning-Based Electrocardiogram Analysis." *Diagnostics* 13(14):2345. DOI 10.3390/diagnostics13142345. Íntegra: sim (PMC).** — [PMC10378376](https://pmc.ncbi.nlm.nih.gov/articles/PMC10378376/)
  - Grad-CAM avaliado só por dois médicos.
  - "the explanations were not found useful by the domain experts".
- **Jo YY, Kwon JM, et al. (autoria não confirmada nas fontes acessadas) 2021. "Explainable artificial intelligence to detect atrial fibrillation using electrocardiogram." *International Journal of Cardiology* 328:104–110. DOI 10.1016/j.ijcard.2020.11.053. Íntegra: não (só resumo).** — [AMiner](https://www.aminer.org/pub/5fcb5b25d4150a363cc9762f/explainable-artificial-intelligence-to-detect-atrial-fibrillation-using-electrocardiogram)
  - Módulos explicativos de ritmo irregular e ausência de onda P, com "explainability AUC" de 0,961–0,993 e 0,983–0,993.
  - Externamente validado em PTB-XL, Chapman e PhysioNet. Como esse AUC foi calculado não foi verificado.
- **Jang JH, Jo YY. 2026. "ExECG: An Explainable AI Framework for ECG models." arXiv 2605.19258. Íntegra: sim (leitura dirigida).** — [arXiv](https://arxiv.org/pdf/2605.19258)
  - Framework de execução e visualização (PTB-XL, MIMIC-IV-ECG) com atribuições, contrafactuais e TCAV.
  - Nenhuma métrica de qualidade.

### Inferences
- O conjunto de métricas do TCC (FaithCorr, FE, Max/Avg-Sens, Complexity, Consistency, Random Logit, eMPRT) não aparece em nenhum estudo de ECG encontrado. A literatura de ECG usa sobretudo deleção/perturbação ordenada (Suh, Goettling, Arends, Turbé, Abdullah), randomização (Arends, Abdullah, Wagner com sanity check próprio) e localização ou plausibilidade (Suh, Gumpfer). Isso sustenta apresentar a bateria do TCC como contribuição metodológica, citando Arends 2026 e Gumpfer 2026 como as avaliações mais amplas e recentes.
- Os estudos mais próximos do desenho do TCC são estes:
  - Turbé 2023: mesma fonte de dados (CPSC) e GradShap com baseline de amostras.
  - Arends 2026: ResNet-1D, nove classes, GradientSHAP, randomização e perturbação.
  - Gumpfer 2026: agregação por segmentos clínicos (PR/ST/QRS), análoga às 36 regiões do TCC.
  - Zhang 2021: mesmo CPSC2018 com nove classes, F1 por classe e SHAP, mas sem avaliação.

### Gaps
- Não foram localizados os trabalhos "Mortensen et al." (ECG XAI evaluation), "Anand et al. 2022" e "Ganeshkumar et al. 2021". As buscas pelo título dado não os encontraram, e não convém citá-los sem verificar título e DOI.
- Ficaram sem detalhe (texto pago ou bloqueado): Storås JAMIA 2025 (baseline e números da perturbação), Neves 2021 (valores de Jaccard e de queda de desempenho), Jo 2021 (método do "explainability AUC"), Lopes 2026 (definição de fidelidade) e a revisão de Salih 2024 (número de estudos).
- Dobiczek et al. (ICCS 2026, CineECG) aparece só em snippet: Dice 0,56 contra 0,47 em 20 casos anotados por especialistas. Não verificado — [ICCS PDF](https://www.iccs-meeting.org/archive/iccs2026/papers/167860449.pdf).
- Schlegel et al. 2019 (arXiv 1909.07082) e Schlegel & Keim 2023 (arXiv 2307.05104) são avaliações por perturbação em séries temporais, mas não foi confirmado se usam conjuntos de ECG. — [arXiv 1909.07082](https://arxiv.org/abs/1909.07082v2); [arXiv 2307.05104](https://arxiv.org/abs/2307.05104v1)

## 2. Como segmentaram o ECG e escolheram o baseline de perturbação?

### Takeaway
Não há padrão. Os baselines vão de zero (criticado), média da janela e N(0,1) até interpolação linear, com ou sem ruído, e a escolha muda o ranking dos métodos. A segmentação varia de ponto a ponto (Arends, Turbé) e janelas fixas (Suh: 16 amostras) até batimentos centrados no R e segmentos de onda por derivação (Wagner: 24 segmentos via U-Net; Gumpfer: máscaras PR/ST/QRS; Bender: "average beats"). Nenhum estudo usou exatamente pré-QRS / QRS / ST-T × 12 derivações, mas Wagner e Gumpfer são os análogos diretos.

### Cited Findings
- Goettling 2024: a interpolação linear foi proposta porque a substituição por zero "introduces noise" (o modelo nunca viu saltos para zero), e trocar interpolação por zero muda o melhor método de DTD/LRP-αβ para G×I. Limitação admitida: "The perturbation scheme of interpolation is limited in that it reduces information to a baseline". — [PMC11161651](https://pmc.ncbi.nlm.nih.gov/articles/PMC11161651/)
- Arends 2026: ROAD modificado com interpolação linear dos vizinhos mais ruído gaussiano (σ = 0,25), 1% dos pontos por passo, até 30%. A descrição do ruído é ambígua entre o texto ("small amount of Gaussian noise") e a legenda da figura 4 ("multiplied with Gaussian noise"). — [OUP](https://academic.oup.com/ehjdh/article/7/3/ztag038/8512917)
- Turbé 2023: substituição por N(0,1) aplicada também no treino, para manter a distribuição sem retreinar. Baselines de atribuição: média temporal (IG, DeepLift) e **50 amostras aleatórias do teste para GradShap e DeepLiftShap**. Entrada em batimento médio por derivação. — [arXiv](https://arxiv.org/html/2202.05656v2)
- Suh 2022/2024: janelas de 16 amostras (≈ 64 ms a 250 Hz) substituídas pela **média da janela**, nas ordens MoRF e LeRF. — [arXiv](https://arxiv.org/html/2211.12702)
- Hicks 2021: "blanking" de ondas inteiras (P, QRS, T) por interpolação linear. É uma perturbação por segmento fisiológico. — [PMC8154909](https://pmc.ncbi.nlm.nih.gov/articles/PMC8154909/)
- Wagner 2024: segmentação por U-Net 2D em 24 segmentos, com relevância ponderada pelas máscaras suaves e somada por derivação e segmento. Batimentos de −300 a +500 ms do R, que é detectado em V1–V4. — [arXiv](https://arxiv.org/html/2305.17043v2)
- Gumpfer 2026: batimentos centrados no R, histogramas de relevância agregados, máscaras temporais derivadas de diretrizes e aplicadas a todas as derivações. Análise de sensibilidade com deslocamento das bordas das máscaras, sem mudar o ranking. — [arXiv](https://arxiv.org/pdf/2607.24035)
- Bender 2023: "average beats", com o mesmo índice de batimento aplicado ao sinal e à relevância, por classe e por derivação. — [arXiv](https://arxiv.org/pdf/2211.01738)
- Kelly & Blake 2026: avaliam o mapa inteiro no domínio da frequência e descartam atribuições negativas. — [arXiv](https://arxiv.org/html/2606.24974)

### Inferences
- A escolha do TCC (36 regiões, 12 derivações × pré-QRS / QRS / ST-T) fica entre a granularidade de Wagner (24 segmentos de onda) e a de Gumpfer (três tipos de máscara). Ela pode ser justificada como agregação fisiológica no estilo "glocal". Como Goettling e Arends mostram que o baseline muda o ranking, convém relatar no TCC a sensibilidade das métricas de fidelidade ao baseline de substituição.
- A observação do TCC no eMPRT (o mapa do modelo aleatório se concentra no QRS) é coerente com Gumpfer: métodos com baseline ou input-multiplication, entre eles o GradSHAP, acompanham a amplitude (Spearman médio 0,33 no GradSHAP e 0,69 no IG). Também é coerente com Wagner: IG, LRP e GradCAM marcam o pico R mesmo quando ele não é o alvo. Isso sugere que parte da relevância no QRS é um artefato de amplitude do método e não depende só do modelo.

### Gaps
- Nenhum estudo de ECG encontrado compara de forma sistemática zero × média × ruído × interpolação no mesmo modelo, além de Goettling (zero × interpolação) e da observação secundária de Neves (zero ≈ média no LIME, não verificada).

## 3. O que encontraram e concluíram?

### Takeaway
A conclusão dominante é cautelosa ou negativa:
- os métodos discordam entre si (Arends, Storås);
- as avaliações por perturbação e por especialistas divergem (Storås);
- métodos populares falham em sanity checks (GradCAM em Wagner) ou seguem a amplitude em vez da clínica (Gumpfer);
- nenhum método vence em todos os cenários (Storås, Goettling).

Os métodos da família SHAP saem mistos: KernelSHAP vai mal de forma consistente (Turbé, Suh, Beck & John, Kelly & Blake); GradientSHAP e DeepSHAP ficam no meio, bons em degradação, mas ruidosos e guiados pela amplitude.

### Cited Findings
- Arends 2026: autoconsistência média de 0,52 e queda heterogênea na perturbação. Conclusão: as explicações por atribuição devem ser "treated as exploratory tools rather than definitive explanations". — [OUP](https://academic.oup.com/ehjdh/article/7/3/ztag038/8512917)
- Storås 2025: "The best explanation method depended on the ECG measure"; recomenda "Several explanation methods should be used". — [OUP](https://academic.oup.com/jamia/article/32/1/79/7876016)
- Wagner 2024: só o saliency passou nos seis sanity checks. "The outcome of the sanity check not only call into question results achieved using attribution methods that did not pass the sanity check" (§4). — [arXiv](https://arxiv.org/html/2305.17043v2)
- Gumpfer 2026: nove de 13 métodos ficam abaixo do acaso em ao menos uma patologia. GradSHAP tem NCovmin de −5,5% (isquemia) e DeepSHAP de −37,7%. As variantes SIGN ficam acima do acaso em tudo. — [arXiv](https://arxiv.org/pdf/2607.24035)
- Turbé 2023: Shapley Value Sampling é o melhor no geral e KernelShap é o pior. GradShap fica intermediário. — [arXiv](https://arxiv.org/html/2202.05656v2)
- Suh 2022/2024: Grad-CAM é 18% melhor que o segundo (IG). DeepSHAP vai bem só na degradação (0,883) e LIME/KernelSHAP vão mal. — [arXiv](https://arxiv.org/html/2211.12702)
- Goettling 2024: DTD é o mais confiável (+24% sobre o segundo, segundo o resumo). — [PMC11161651](https://pmc.ncbi.nlm.nih.gov/articles/PMC11161651/)
- Kelly & Blake 2026: GradientShap e KernelShap têm entropia espectral alta (≈ 9–10 bits), perto de ruído, contra ≈ 1 do Grad-CAM. — [arXiv](https://arxiv.org/html/2606.24974)
- Bender 2023: IG e LRP recuperam critérios clínicos de FA (ausência de P) e BRE (QRS largo) nas médias por batimento. É uma análise de plausibilidade, sem fidelidade. — [arXiv](https://arxiv.org/pdf/2211.01738)
- Storås 2023: os médicos não acharam úteis os mapas Grad-CAM. — [PMC10378376](https://pmc.ncbi.nlm.nih.gov/articles/PMC10378376/)

### Inferences
- O achado do TCC de que a robustez acompanha a fidelidade, com o piso do GradientSHAP dominando, combina com a entropia alta do GradientShap em Kelly & Blake e com o comportamento intermediário em Turbé e Gumpfer. Vale citar esses trabalhos ao discutir a complexidade e o piso.
- A divergência entre plausibilidade (localização ou clínica) e fidelidade (perturbação), mostrada por Storås 2025 e reconhecida por Gumpfer, apoia a decisão do TCC de deixar a localização de fora e concentrar-se em fidelidade, robustez, complexidade e randomização.

### Gaps
- Nenhum estudo de ECG comparou o GradientSHAP com métricas de robustez (Max/Avg-Sensitivity) nem com Random Logit ou eMPRT. Não há evidência publicada para comparar diretamente com esses resultados do TCC.

## 4. Algum trabalho relacionou a qualidade da explicação ao desempenho por classe (F1/AUC)?

### Takeaway
**Nenhum trabalho encontrado testa de forma explícita (correlação ou regressão) se a qualidade da explicação acompanha o F1 ou a AUC por classe.** Existem só indícios parciais:
- Arends relata autoconsistência por classe e afirma que o comportamento é parecido em tarefas de AUPRC alta e baixa;
- Gumpfer mostra variação forte por patologia com modelos de AUROC parecida;
- Lopes relata fidelidade dependente da classe para Grad-CAM e atenção;
- Beck & John notam, de forma qualitativa, que a classe mais fraca tem as piores explicações.

A pergunta do TCC é portanto uma lacuna real na literatura de ECG.

### Cited Findings
- Arends 2026: autoconsistência média por classe de 0,54 (baixa voltagem do QRS, HVE) a 0,62 (BAV de 1º grau); na classe sinusal, 0,57 ± 0,13. Os autores dizem que houve comportamento semelhante em tarefas com AUPRC maior e menor (a menor foi 0,64, na bradicardia sinusal) e atribuem os achados aos métodos, não à acurácia do classificador. Não há perturbação nem randomização por classe. — [OUP](https://academic.oup.com/ehjdh/article/7/3/ztag038/8512917)
- Gumpfer 2026: todos os modelos têm AUROC de 0,93 a 0,99 (escolhidos "ensuring that subsequent analyses assessed explanation quality rather than model accuracy limitations"). Mesmo assim a NCov varia muito por patologia; no GradSHAP, de −5,5% (isquemia) a 80,9% (BRD). A variação vem do tipo de critério (baixa × alta amplitude), não do desempenho. — [arXiv](https://arxiv.org/pdf/2607.24035)
- Lopes 2026: IG teve ganho de fidelidade positivo "across all classes", enquanto Grad-CAM e atenção foram "more class-dependent". Só resumo, sem números por classe nem F1 por classe ligados à explicação. — [RECIMA21](https://recima21.com.br/recima21/en/article/view/9152)
- Beck & John 2025: a classe de batimento atrial prematuro teve o pior desempenho e explicações desalinhadas da clínica. É uma observação qualitativa. — [arXiv](https://arxiv.org/html/2508.17294v1)
- Kelly & Blake 2026: a entropia espectral varia pouco entre as 17 classes na maioria dos métodos. Os autores leem isso como um "self-noise" quase constante de cada método, ou seja, mais dependente do método que da classe. — [arXiv](https://arxiv.org/html/2606.24974)
- Suh 2022/2024 e Abdullah 2026 avaliam só exemplos corretamente classificados ou de alta confiança, o que neutraliza por desenho o efeito do desempenho. Abdullah relata F1 por classe (S 0,79, F 0,77), mas não liga esses valores às métricas de explicação. — [arXiv](https://arxiv.org/html/2211.12702); [PMC13296925](https://pmc.ncbi.nlm.nih.gov/articles/PMC13296925/)
- Zhang 2021 dá o F1 por classe no CPSC2018 (STE 0,535 é o pior; os autores o associam à discordância entre médicos), sem avaliar a qualidade do SHAP por classe. — [PMC8082080](https://pmc.ncbi.nlm.nih.gov/articles/PMC8082080)

### Inferences
- A evidência disponível (Arends, Kelly & Blake, Gumpfer) indica que a qualidade da explicação depende mais do método e da natureza do critério diagnóstico (amplitude da região relevante) do que do desempenho do classificador. Se o TCC encontrar associação fraca entre as métricas e o F1 por classe, isso é consistente com a literatura. Se encontrar associação forte, seria um achado novo, e convém controlar o tipo de critério da classe: classes de QRS largo (BRE, BRD, PVC) × de baixa amplitude (STD, STE, PAC/P-wave).
- Zhang 2021 é a referência natural de F1 por classe no mesmo conjunto (CPSC2018, nove classes) para situar o modelo B do TCC.

### Gaps
- Não foi encontrado nenhum estudo de ECG (nem de séries temporais fisiológicas) que calcule correlação entre métricas de fidelidade, robustez, complexidade ou randomização e o F1 ou a AUC por classe.
- O texto completo de Storås 2025 pode trazer resultados por medida de ECG (intervalos e amplitudes), o que é análogo a "por classe" numa tarefa de regressão, mas não foi acessado.
