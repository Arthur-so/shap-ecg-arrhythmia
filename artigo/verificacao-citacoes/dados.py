"""Dados da verificação das citações do artigo (gera main.tex com gerar.py).

Cada referência: número no artigo, chave BibTeX, referência curta, fonte
consultada e lista de ocorrências (seção do artigo, afirmação, trecho do
original e página). Os trechos foram copiados do texto extraído do original
(pdftotext ou texto completo do Europe PMC) e conferidos por busca automática.
"""

FONTES = {
    "local": "PDF original em ~/Documentos/tcc/artigos-relacionados",
    "arxiv": "PDF do arXiv (baixado em 12/11/2026)",
    "epmc": "Texto completo no Europe PMC (sem paginação; indicada a seção)",
    "web": "Página do periódico/repositório consultada por ferramenta de leitura web (texto resumido, não o PDF)",
    "meta": "Só metadados (Europe PMC); citação de software, sem afirmação sobre o conteúdo",
}

REFS = [
 (1, "zhang2021", "Zhang et al. (2021), iScience 24:102373", "local (consistency.pdf = versão publicada, 22 p.) + código no GitHub", [
  ("Introdução, 1.º parágrafo", "Doenças cardiovasculares são a principal causa de morte; o ECG é simples, confiável e não invasivo; 12 derivações registradas por eletrodos na superfície do corpo.",
   "Cardiovascular diseases (CVDs) are the leading cause of death and produce immense health and economic burdens in the United States and globally [...] The electrocardiogram (ECG) is a simple, reliable, and non-invasive approach for monitoring patients' heart activity and diagnosing cardiac arrhythmias. A standard ECG has 12 leads including 6 limb leads (I, II, III, aVR, aVL, aVF) and 6 chest leads (V1, V2, V3, V4, V5, V6) recorded from electrodes on the body surface.", "p. 2"),
  ("Introdução, 1.º parágrafo", "Interpretar o ECG é difícil mesmo para cardiologistas experientes; o número de exames é grande; o diagnóstico automático é tema ativo.",
   "Accurately interpreting the ECG for a patient with concurrent cardiac arrhythmias is challenging even for an experienced cardiologist [...] An estimated 300 million ECGs are recorded worldwide annually [...] Therefore, accurate and automatic diagnosis of ECG signals has become a hot research interest.", "p. 2"),
  ("Introdução, 3.º parágrafo", "Os modelos são considerados caixas-pretas; as decisões não são rastreáveis; isso pesa na prática clínica.",
   "deep learning models are usually considered to be black boxes. Due to the multi-layer nonlinear structure, the decisions made by deep learning models are not traceable by humans. However, understanding the model's behavior when making predictions is as crucial as the accuracy of predictions in many applications, especially in clinical practice.", "p. 14-15"),
  ("Introdução, 2.º parágrafo", "ResNet unidimensional de 34 camadas obteve F1 médio de 0,813 no CPSC2018, com nove classes.",
   "the network consists of 34 layers. [...] the average F1 score was 0.813 with an average precision of 0.821 and an average recall of 0.812.", "p. 13 e p. 4"),
  ("Introdução, 2.º parágrafo", "F1 acima de 0,9 em AF e BRD e de 0,535 em STE, atribuído à discordância entre médicos.",
   "the deep model performed best on AF and RBBB classification with an F1 score of over 0.9. However, we also observed the F1 score of STE is low as 0.535 which may be due to the significant physician disagreement in diagnosing STE from ECGs", "p. 4"),
  ("Introdução, 3.º parágrafo", "O SHAP já foi usado para explicar a rede residual no CPSC2018, por paciente e por derivação.",
   "we applied the SHapley Additive exPlanations (SHAP) method (Lundberg and Lee, 2017) to interpret the model's predictions at both the patient level and population level.", "p. 3"),
  ("Metodologia II-A", "Duração dos registros de 6 a 60 s.",
   "The CPSC2018 database comprises multi-label 12-lead ECGs with varying durations between 6 s and 60 s.", "p. 13"),
  ("Metodologia II-A", "Comprimento de 15 000 amostras (30 s); a referência mantém os últimos 30 s.",
   "setting nsteps to 15000 (duration of 30 s, sampling rate of 500 Hz) achieved the best performance. For ECGs with a duration of more than 30 s, they will be cropped and the last 30 s ECG data are kept. Otherwise, they will be padded to 30 s with zeros.", "p. 13"),
  ("Metodologia II-A", "A referência não normaliza o sinal.",
   "Conferido no código (dataset.py do repositório onlyzdd/ecg-diagnosis, consultado em 07/11/2026): não há z-score nem outra normalização; o artigo não descreve normalização.", "código"),
  ("Metodologia II-B e Fig. 2", "Arquitetura ResNet34-1D (blocos residuais, pooling de média e máximo, saída sigmoide).",
   "4 stacked residual blocks are used to extract deep features. Within each residual block, there are two 1D convolutional (Conv1d) layers, two batch normalization (BatchNorm1d) layers, 1 dropout (Dropout) layer, and two rectified linear unit (ReLU) activation layers.", "p. 13"),
  ("Metodologia II-B", "Adam, taxa 1e-4, lote 32, dropout 0,2, máximo de 30 épocas; aumento por escala e deslocamento.",
   "Adam optimizer is used as the optimization method and cross-entropy as the loss function to train the model. [...] the learning rate is 0.0001; the batch size is 32; the maximum number of epochs is 30; the kernel size of 1D CNNs is 15; the dropout rate of dropout layers is 0.2. || we applied scaling and shifting for data augmentation during the training phase.", "p. 14 e p. 13"),
  ("Metodologia II-B", "Entropia cruzada binária sem ponderação, como na referência.",
   "O artigo diz só cross-entropy (p. 14); o código (main.py) usa nn.BCEWithLogitsLoss() sem pesos (consultado em 07/11/2026).", "p. 14 + código"),
  ("Metodologia II-B", "Limiar por classe escolhido na validação pelo melhor F1 e aplicado ao teste.",
   "The optimal threshold of each class is selected to achieve the best F1 score on the validation dataset. Then the selected thresholds are applied to the test dataset to produce results.", "p. 14"),
  ("Metodologia II-D", "Zhang et al. resumem os valores SHAP pela soma ao longo do tempo.",
   "the contribution [...] of lead k for diagnostic class i is defined as the sum of shap values", "p. 15"),
  ("Resultados III-A e Tabela III (F1)", "F1 médio 0,813 (10 dobras) e F1 por classe da referência.",
   "Tabela 1 do artigo: SNR 0,805; AF 0,919; IAVB 0,864; LBBB 0,866; RBBB 0,926; PAC 0,735; PVC 0,851; STD 0,814; STE 0,535; AVG 0,813.", "p. 6"),
 ]),
 (2, "lundberg2017", "Lundberg e Lee (2017), NeurIPS", "arxiv (1705.07874v2)", [
  ("Introdução, 3.º parágrafo", "No SHAP, a soma das atribuições reproduz a diferença entre a saída e um valor de base.",
   "local accuracy requires the explanation model to at least match the output of f for the simplified input x' (which corresponds to the original input x). || They explain how to get from the base value E[f (z)] that would be predicted if we did not know any features to the current output f (x).", "p. 4 e p. 5"),
  ("Metodologia II-C", "O GradientSHAP aproxima os valores SHAP.",
   "These are the Shapley values of a conditional expectation function of the original model", "p. 4 (definição dos valores SHAP; a aproximação por gradientes vem de [16])"),
 ]),
 (3, "adebayo2018", "Adebayo et al. (2018), NeurIPS", "local (MPRT.pdf, arXiv v3)", [
  ("Introdução, 4.º parágrafo (duas vezes)", "Inspeção visual é guia fraco; métricas de randomização verificam se a explicação depende do modelo.",
   "The edge detector does not depend on model or training data, and yet produces results that bear visual similarity with saliency maps. This goes to show that visual inspection is a poor guide in judging whether an explanation is sensitive to the underlying model and data.", "p. 2"),
  ("Discussão IV-C", "Em gradiente × entrada, a randomização altera o sinal, mas a estrutura da entrada permanece.",
   "re-initialization disrupts the sign of the map, so that the spearman rank correlation without absolute values goes to zero || For gradient input and integrated gradients, we also observe visual changes in the masks obtained, particularly, in the sign of the attributions. Despite this, the input structure is still clearly prevalent in the masks.", "p. 5 e p. 7"),
 ]),
 (4, "hedstrom2023quantus", "Hedström et al. (2023), JMLR (Quantus)", "arxiv (2202.06861v3)", [
  ("Introdução, 4.º parágrafo", "Métricas quantitativas organizadas em categorias que medem propriedades diferentes.",
   "for six different evaluation categories [...] Faithfulness Robustness Localisation Complexity Axiomatic Randomisation", "p. 2 (Tabela 1)"),
  ("Metodologia II-F", "O projeto inicial previa métricas de fidelidade e robustez calculadas pela biblioteca Quantus.",
   "Referência à biblioteca. O projeto inicial é o TCC1 (~/Documentos/tcc/entregas/tcc1.pdf, seção 2.5); o motivo do descarte das versões ponto a ponto é resultado do próprio TCC.", "--"),
 ]),
 (5, "bhatt2020", "Bhatt, Weller e Moura (2020), IJCAI", "local (faithcorr.pdf)", [
  ("Introdução, 4.º parágrafo", "Fidelidade: os atributos importantes devem ser os que mais alteram a saída quando retirados.",
   "when we set particular features xs to a baseline value x̄s, the change in predictor's output should be proportional to the sum of attribution scores", "p. 2 (seção 3.2)"),
  ("Introdução, 4.º parágrafo", "Complexidade: a importância concentrada em poucos atributos.",
   "The simplest explanation would be concentrated on one feature. We define complexity as the entropy of P_g.", "p. 2"),
  ("Introdução, 5.º parágrafo e Tabela II (métricas)", "Faithfulness Correlation: correlação entre soma das atribuições de um subconjunto e a queda da saída.",
   "Definition 3 (Faithfulness). Given a predictor f, an explanation function g, a point x, and a subset size |S|, we define the faithfulness of g to f at x as [...] (correlação, entre subconjuntos S, da soma das atribuições de S com a queda da saída ao trocar S pela referência)", "p. 2"),
  ("Tabela II (métricas)", "Avg-Sensitivity.",
   "We follow the convention of the former and define max sensitivity and average sensitivity in the neighborhood of a point of interest x. [...] Definition 1 (Max Sensitivity) [...] Definition 2 (Average Sensitivity)", "p. 2"),
  ("Tabela II (métricas)", "Complexity como entropia das frações de importância.",
   "If every feature had equal attribution, the explanation would be complex (even if it is faithful). The simplest explanation would be concentrated on one feature. We define complexity as the entropy of P_g.", "p. 2"),
 ]),
 (6, "suh2024", "Suh et al. (2024), Comput. Biol. Med. 182:109088", "local (computers-in-biology-and-medicine.pdf)", [
  ("Introdução, 5.º parágrafo", "Faithfulness Correlation usada para comparar onze métodos em quatro bases de ECG.",
   "we evaluated a total of eleven widely-used feature attribution methods in our experiments || we conducted automatic evaluations using four datasets: MITDB, SVDB, INCARTDB, and ICENTIA11K. || three perturbation-based metrics - Region Perturbation with MoRF [...], Region Perturbation with LeRF [...], and Faithfulness Correlation", "p. 2, p. 6 e p. 4"),
  ("Discussão IV-A", "Faithfulness Correlation do DeepSHAP entre 0,41 e 0,63, a maior entre os onze métodos.",
   "Tabela 4 e Tabelas F.1-F.3: DeepSHAP 0.409 (1) no ICENTIA11K, 0.591 (1) no MITDB, 0.634 (1) no SVDB, 0.532 (1) no INCARTDB; entre parênteses, a posição.", "p. 8 e p. 17-18"),
 ]),
 (7, "arends2026", "Arends et al. (2026), Eur. Heart J. Digit. Health 7(3):ztag038", "epmc (PMC12980500)", [
  ("Introdução, 6.º parágrafo", "Estudo com nove classes e modelos com AUPRC média acima de 0,85 em oito delas.",
   "All models achieved a high discriminatory performance with a mean AUPRC above 0.85 for eight out of nine classes, with the exception of sinus bradycardia which had a mean AUPRC of 0.64.", "Resultados"),
  ("Discussão IV-C", "Perda rápida de correlação quando os pesos são randomizados.",
   "Randomizing model weights led to rapid loss of correlation, although some methods did not converge to zero.", "Resumo"),
  ("Discussão IV-E (Limitações)", "Mapas de modelos com sementes diferentes concordam apenas moderadamente.",
   "Self-consistency across random model initializations was moderate for most methods (mean correlation 0.41-0.65).", "Resumo"),
 ]),
 (8, "gumpfer2026", "Gumpfer et al. (2026), preprint arXiv:2607.24035", "arxiv", [
  ("Introdução, 1.º parágrafo", "Amplitudes das ondas diferem muito, com o pico R dominante.",
   "Because ECG amplitudes differ markedly between waveform components, most notably the dominant R-peak, these methods may over-attribute large-amplitude regions even when the diagnostically critical information lies in low-amplitude regions such as the PR interval or the ST segment", "p. 2"),
  ("Introdução, 1.º parágrafo", "Diagnósticos na forma do QRS (bloqueios de ramo) ou em segmentos de baixa amplitude (isquemia, segmento ST).",
   "two categories of ECG diagnostic criteria, low-amplitude intervals / segments near the isoelectric line (atrioventricular block, myocardial ischemia) and high-amplitude QRS morphology patterns (right and left bundle branch block). || For ischemia, LRP-ε assigns only 4.6% of relevance to the ST segment", "p. 3 e p. 1"),
  ("Introdução, 5.º parágrafo", "Explicações seguiram a amplitude e deixaram de cobrir regiões de baixa amplitude importantes.",
   "Their explanations often follow signal amplitude rather than clinical relevance, with mean Spearman correlations up to 0.69, leading them to overlook diagnostically decisive low-amplitude regions.", "p. 1"),
  ("Introdução, 5.º parágrafo e Discussão IV-A", "Explicações confrontadas com regiões das diretrizes; métodos que multiplicam pela entrada ou subtraem referência deslocam a relevância para a alta amplitude.",
   "Methods commonly applied in ECG analysis [...], among them Integrated Gradients, DeepSHAP, GradSHAP, and LRP-ε, tie relevance to input magnitude through input multiplication or baseline subtraction and shift attribution toward high-amplitude waveform components", "p. 9"),
  ("Discussão IV-A", "Esses métodos, entre eles o GradientSHAP, reforçaram a alta amplitude e falharam no ST da isquemia.",
   "Integrated Gradients, DeepSHAP, GradSHAP, and LRP-z/ε similarly reinforce high-amplitude regions through baselinedependent contrast shifts. The coverage divergence, where these methods fail on ischemia despite reasonable mean values, is the more diagnostic evidence", "p. 7"),
 ]),
 (9, "baer2025a", "Baer et al. (2025a), xAI 2025, CCIS 2579", "arxiv (2502.17022)", [
  ("Introdução, 5.º parágrafo", "A eficácia das métricas de perturbação varia muito entre as classes.",
   "we reveal previously overlooked class-dependent effects in these metrics: they show varying effectiveness across classes, achieving strong results for some while remaining less sensitive to others.", "p. 1"),
  ("Metodologia II-F", "O valor usado para retirar um trecho pode afetar as classes de modo diferente.",
   "perturbation effectiveness can vary substantially between classes; [...] optimal perturbation strategies can vary considerably across datasets and model architectures", "Conclusão"),
  ("Discussão IV-A", "Recomendação de análise estratificada por classe.",
   "First, supplementing aggregate metrics with more detailed class-stratified analysis can help identify when dominant classes disproportionately influence results.", "Conclusão"),
 ]),
 (10, "baer2025b", "Baer et al. (2025b), preprint arXiv:2506.11790", "arxiv", [
  ("Introdução, 5.º parágrafo, e Discussão IV-A", "Variações de amplitude ou duração entre classes bastam para efeitos dependentes da classe; perturbação e verdade conhecida discordam.",
   "class-dependent effects emerge with both evaluation approaches, even in simple scenarios with temporally localized features, triggered by basic variations in feature amplitude or temporal extent between classes. Most critically, we find that perturbation-based and ground truth metrics frequently yield contradictory assessments of attribution quality across classes", "p. 1"),
 ]),
 (11, "hollig2023", "Höllig, Thoma e Grimm (2023), ICMLA", "arxiv (2310.14957)", [
  ("Introdução, 6.º parágrafo", "O XTSC-Bench só avalia classificadores com acurácia acima de 90%.",
   "we evaluate the explainers' capabilities on complexity, reliability, robustness, and faithfulness for all classifiers with an accuracy of over 90%", "p. 4"),
  ("Metodologia II-F", "O valor de substituição pode carregar informação em séries temporais.",
   "replacing features with standard techniques (class means or zeros) might be relevant information in time series.", "p. 2"),
 ]),
 (12, "liu2018", "Liu et al. (2018), J. Med. Imaging Health Inform. 8(7):1368-1373", "web (via [13])", [
  ("Metodologia II-A", "Origem da base CPSC2018.",
   "Texto original não acessível. A referência completa e a origem da base foram conferidas em Perez Alday et al. [13]: \"The first source is the China Physiological Signal Challenge 2018 (CPSC2018)\" (Liu et al., 2018). Os números (6 877 registros, 500 Hz) foram conferidos nos próprios dados do TCC.", "--"),
 ]),
 (13, "perezalday2020", "Perez Alday et al. (2020), Physiol. Meas. 41(12):124003", "web (PMC8015789)", [
  ("Metodologia II-A", "Distribuição do desafio PhysioNet/CinC 2020; 6 877 registros de treino do CPSC; 500 Hz.",
   "Tabela 2 do artigo: CPSC Training, 6 877 registros, 500 Hz (conferido na página do PMC por ferramenta de leitura web, 12/11/2026).", "Tabela 2"),
 ]),
 (14, "he2016", "He et al. (2016), CVPR", "arxiv (1512.03385)", [
  ("Metodologia II-B", "Cada bloco aprende um resíduo em relação à entrada, somada por um atalho.",
   "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. || the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers", "p. 1 e p. 2"),
 ]),
 (15, "lipton2014", "Lipton, Elkan e Naryanaswamy (2014), ECML PKDD", "arxiv (1402.1892)", [
  ("Metodologia II-B", "O limiar que maximiza o F1 em geral não é 0,5 e depende da classe.",
   "For any classifier that produces a real-valued output, we derive the relationship between the best achievable F1 score and the decision-making threshold that achieves this optimum. As a special case, if the classifier outputs are well-calibrated conditional probabilities, then the optimal threshold is half the optimal F1 score.", "p. 1"),
 ]),
 (16, "kokhlikyan2020", "Kokhlikyan et al. (2020), Captum, arXiv:2009.07896", "arxiv", [
  ("Introdução, 3.º parágrafo, e Metodologia II-C", "Implementação do GradientSHAP no Captum; aproximação por gradientes esperados multiplicados por (x - referência).",
   "Examples of such attribution functions are Integrated Gradients, DeepLift, SHAP variant of DeepLift and Gradient SHAP aka Expected Gradients [5]. The mathematical formulations of those methods multiply the resulting saliency maps by x − x0.", "p. 6"),
  ("Metodologia II-C (sem citação)", "O desvio 0,09 é o do exemplo de uso do Captum.",
   "README do repositório pytorch/captum: gs.attribute(input, stdevs=0.09, n_samples=4, baselines=baseline_dist, target=0, return_convergence_delta=True) (consultado em 07/11/2026).", "README"),
 ]),
 (17, "sundararajan2017", "Sundararajan, Taly e Yan (2017), ICML", "arxiv (1703.01365)", [
  ("Metodologia II-C", "Gradientes calculados ao longo do caminho entre referência e entrada.",
   "We consider the straightline path [...] from the baseline x' to the input x, and compute the gradients at all points along the path. Integrated gradients are obtained by cumulating these gradients.", "p. 3"),
 ]),
 (18, "turbe2023", "Turbé et al. (2023), Nat. Mach. Intell. 5:250-260", "local", [
  ("Introdução, 3.º parágrafo", "Métodos post hoc explicam um modelo já treinado.",
   "post-hoc interpretability refers to how a trained model makes predictions and uses the input features it is given.", "p. 2"),
  ("Metodologia II-C", "Referência como distribuição: 50 ECGs na mesma base.",
   "GradShap and DeepLiftShap uses a distribution of baseline and this baseline was constructed by taking 50 random samples from the test set. || The dataset was narrowed down to the CPSC subset, which included 6,877 ECGs annotated for nine cardiovascular diseases.", "p. 9"),
  ("Metodologia II-D", "Janela de 0,35 s antes e 0,55 s depois do pico R.",
   "The beats centred around the R-peaks are then extracted from the ECG by taking 0.35 before and 0.55 s after the R-peak.", "p. 9"),
  ("Discussão IV-A", "A relevância funciona mais como ordem do que como contribuição proporcional.",
   "The attributed relevance instead acts more as a ranking of the most important time steps among themselves.", "p. 6"),
 ]),
 (19, "smilkov2017", "Smilkov et al. (2017), SmoothGrad", "arxiv (1706.03825)", [
  ("Metodologia II-C", "O ruído suaviza oscilações locais do gradiente.",
   "the derivative of the function Sc may fluctuate sharply at small scales. In other words, the apparent noise one sees in a sensitivity map may be due to essentially meaningless local variations in partial derivatives.", "p. 2"),
 ]),
 (20, "cui2023", "Cui et al. (2023), Front. Comput. Neurosci. 17:1232925", "epmc (PMC10470463)", [
  ("Metodologia II-D e Discussão IV-E (Limitações)", "Perturbar trechos contíguos introduz menos ruído de alta frequência.",
   "we perturbed the input sample locally in small patches with fixed length n instead of n random points from different parts of the sample. In this way, we can evaluate the local accuracy of the contribution map while introducing less high-frequency noise to EEG signals.", "seção 4.2.1 (p. 5 do PDF publicado)"),
 ]),
 (21, "serramazza2024", "Serramazza, Le Nguyen e Ifrim (2024), arXiv:2406.12507", "arxiv", [
  ("Metodologia II-D", "Agrupar pontos consecutivos melhorou as métricas de perturbação.",
   "We also improved the AUC scores of attribution methods, while reducing running time, by using chunking, i.e., grouping consecutive features in each channel.", "p. 16"),
 ]),
 (22, "simic2025", "Šimić, Veas e Sabol (2025), Sci. Rep. 15:26607", "epmc (PMC12284047)", [
  ("Metodologia II-D", "Perturbações mínimas mudam bruscamente a predição por deslocamento de distribuição; recomendação de regiões de tamanho fixo.",
   "This resulted in abrupt changes in a classifiers predicted class probability from 1 to 0 after minimal perturbations, regardless of the features' actual relevance, due to shifts in the data distribution. || a fixed region size has to be selected, which according to the results of our previous paper should neither be too large, nor too small (e.g., 2-5% of input length).", "Introdução; Metodologia"),
 ]),
 (23, "wagner2024", "Wagner et al. (2024), Comput. Biol. Med. 176:108525", "arxiv (2305.17043v2)", [
  ("Metodologia II-D", "A agregação entre batimentos reduz o ruído dos métodos de gradiente.",
   "Gradient noise, a major drawback of saliency methods, can be very effectively reduced by aggregating attributions across beats and samples.", "p. 14"),
  ("Metodologia II-D", "Janela de 300 ms antes e 500 ms depois do pico R.",
   "We then extract median beats from the signal by extracting the signal 300 ms before until 500 ms after each identified R-peaks [...]. We proceed similarly for the attribution maps", "p. 5"),
  ("Metodologia II-D", "Wagner et al. usam a média por segmento.",
   "we calculate the dot product of the attributions of the i-th channel with the segmentation outputs (normalized by their sum, so that they sum up to 1) corresponding to the m-th segment.", "p. 5"),
  ("Discussão IV-C", "O viés coloca a relevância no pico R mesmo em tarefas sobre as ondas P e T.",
   "IG and LRP attribute the largest fraction of relevance to the R-peak even if the R-peak is not directly relevant for the given task (e.g., for predicting the P-wave or T-wave amplitude).", "p. 8"),
 ]),
 (24, "makowski2021", "Makowski et al. (2021), Behav. Res. Methods 53(4):1689-1696", "meta", [
  ("Metodologia II-D", "Picos R detectados com o NeuroKit2.",
   "Citação de software (biblioteca usada no código, versão 0.2.13). Metadados conferidos no Europe PMC (DOI 10.3758/s13428-020-01516-y).", "--"),
 ]),
 (25, "hicks2021", "Hicks et al. (2021), Sci. Rep. 11:10949", "epmc (PMC8154909)", [
  ("Metodologia II-D", "Bloqueios de ramo definidos por QRS de 120 ms ou mais.",
   "we tested the prediction errors on subgroups with bundle branch blocks (QRS >= 120 ms) or first-degree AV block (PR > 220 ms).", "Resultados"),
 ]),
 (26, "vandenberk2016", "Vandenberk et al. (2016), J. Am. Heart Assoc. 5(6):e003264", "epmc (PMC4937268)", [
  ("Metodologia II-D", "Bazett é a correção mais usada e sobrecorrige em frequências altas.",
   "The current clinical standard is the most widely used Bazett formula, but with this formula, there is a known overcorrection at high heart rates and undercorrection at lower heart rates.", "Introdução"),
 ]),
 (27, "fouladgar2022", "Fouladgar, Alirezaie e Främling (2022), IEEE Access 10", "local", [
  ("Metodologia II-E", "Sem normalização, escores de escala menor podem parecer menos sensíveis só pela escala.",
   "IG and LIME provide a lower sensitivity scale than SG in all the metrics and settings, potentially due to the lower scale of important scores generated by IG and LIME", "p. 1 (p. 23995 da revista)"),
 ]),
 (28, "alvarezmelis2018", "Alvarez-Melis e Jaakkola (2018), NeurIPS", "local (faithest.pdf, arXiv v2)", [
  ("Tabela II (métricas)", "Faithfulness Estimate: retirar cada atributo e correlacionar com a queda.",
   "a common approach to measuring the faithfulness of relevance scores with respect to the model they are explaining relies on a proxy notion of importance: observing the effect of removing features on the model's prediction.", "p. 6 (seção 5.3)"),
 ]),
 (29, "yeh2019", "Yeh et al. (2019), NeurIPS", "local (Avg-Max-Sensitivity.pdf, arXiv v4)", [
  ("Introdução, 4.º parágrafo", "Robustez: a explicação deve mudar pouco quando a entrada muda pouco.",
   "the sensitivity of an explanation, which measures the degree to which the explanation is affected by insignificant perturbations from the test point.", "p. 2"),
  ("Tabela II (métricas)", "Max-Sensitivity: maior mudança da explicação numa vizinhança.",
   "we consider a closely related measure, we term max-sensitivity, that measures the maximum change in the explanation with a small perturbation of the input x.", "p. 5 (Definição 3.1)"),
 ]),
 (30, "dasgupta2022", "Dasgupta, Frost e Moshkovitz (2022), ICML", "arxiv (2202.00734v1)", [
  ("Tabela II (métricas) e Discussão IV-B", "Consistency: instâncias com a mesma explicação devem ter a mesma predição; usa as predições do modelo.",
   "consider the set of instances that are assigned this explanation [...] If π is a good explanation, then we would hope that these instances all have the same predicted label", "p. 4-5 (Definição 1)"),
 ]),
 (31, "sixt2020", "Sixt, Granz e Landgraf (2020), ICML", "local (RandomLogitTest.pdf, arXiv v7)", [
  ("Introdução, 4.º parágrafo, Tabela II (métricas) e Discussão IV-B", "Random Logit: comparar mapas da classe verdadeira e de outra classe; motivado por mapas insensíveis à classe.",
   "Random Logit: We display the difference of saliency maps explaining the ground-truth and a random logit [...]. As the logit value is responsible for the predicted class, the saliency maps should change. || Our findings show that many modified BP methods are prone to class-insensitive explanations", "p. 5 e p. 2"),
 ]),
 (32, "hedstrom2023emprt", "Hedström et al. (2023), XAI in Action @ NeurIPS (eMPRT)", "local (EfficientSmoothMPRT.pdf)", [
  ("Tabela II (métricas)", "eMPRT: aumento relativo da complexidade entre o modelo original e o totalmente randomizado.",
   "This test effectively removes the layer-by-layer pairwise comparison between e and ê and instead compute the relative rise in explanation complexity using only two model states, i.e., the original- and fully randomised model.", "p. 4-5 (Definição 3)"),
 ]),
 (33, "mercier2022", "Mercier et al. (2022), ICAART", "arxiv (2202.03759)", [
  ("Discussão IV-A", "Métodos baseados em SHAP entre os de maior sensibilidade; gradientes ruidosos e fracos nesse critério.",
   "Results show larger values for Lime and Shap-based approaches. || Generally, gradient-based methods are fast, show high Infidelity, label dependency but are noisy, not continuous, and suffer concerning the Sensitivity.", "p. 6 (Tabela 6) e p. 11"),
 ]),
 (34, "ravindran2023", "Sujatha Ravindran e Contreras-Vidal (2023), Sci. Rep. 13:17709", "epmc (PMC10584975)", [
  ("Discussão IV-B", "Em EEG simulado, Input × Gradient foi preciso e específico da classe e do modelo.",
   "DeepLift, Input × Gradient, and LRP was found to be both accurate as well as robust in all three cases (spatial, temporal, and spectral).", "Discussão (robustez = não ser insensível à randomização de rótulos e pesos)"),
 ]),
 (35, "hedstrom2023metaquantus", "Hedström et al. (2023), TMLR (MetaQuantus)", "arxiv (2302.07265)", [
  ("Metodologia II-F", "Não se recomenda estimar a qualidade das explicações por uma única categoria.",
   "Relying on only one category to estimate explanation quality is therefore not recommended.", "p. 13"),
  ("Discussão IV-C", "A complexidade foi a categoria menos confiável; não se recomenda usar uma única categoria.",
   "the complexity category has the lowest overall MC scores across the tested datasets || Relying on only one category to estimate explanation quality is therefore not recommended.", "p. 12 e p. 13"),
 ]),
 (36, "ismail2020", "Ismail et al. (2020), NeurIPS", "arxiv (2010.13924)", [
  ("Introdução, 6.º parágrafo", "A comparação de Ismail et al. relata resultados de modelos com acurácia acima de 95%.",
   "The results reported in the following section are for models that produce accuracy above 95% in the classification task.", "p. 5"),
  ("Discussão IV-E (Limitações)", "Retirar atributos muda a distribuição dos dados de teste.",
   "eliminating features changes the test data distribution violating the assumption that both training and testing data are independent and identically distributed (i.i.d.).", "p. 4"),
 ]),
 (37, "storas2025", "Storås et al. (2025), JAMIA 32(1):79-88", "local", [
  ("Discussão IV-E (Limitações)", "A plausibilidade avaliada por especialistas pode discordar da avaliação objetiva.",
   "Differences between objective measurements and experts' assessments may in part owe to differences in how experts would analyze the ECG and how ML models analyze the ECG.", "p. 8"),
 ]),
]

CORRECOES = [
 ("Suh et al. [6]", "O texto dizia que a Faithfulness Correlation foi calculada em cinco bases e ponto a ponto.", "São quatro bases na avaliação automática (a quinta, PTB-XL, só na avaliação humana), e o artigo não deixa claro se o sorteio é ponto a ponto. Corrigido para quatro bases, sem afirmar ponto a ponto."),
 ("Arends et al. [7]", "O texto dizia que o comportamento das explicações foi descrito como semelhante entre classes de AUPRC alta e baixa.", "O artigo não diz isso. Mantido só o fato verificável (AUPRC média acima de 0,85 em oito das nove classes) e retirada a frase da Análise que atribuía ao artigo um tratamento descritivo por classe."),
 ("Arends et al. [7]", "O texto dizia correlação próxima de zero após a randomização.", "O artigo diz perda rápida de correlação, mas nem todos os métodos chegam a zero. Corrigido."),
 ("Gumpfer et al. [8]", "O texto dizia que o GradientSHAP cobriu bem o QRS nos bloqueios de ramo e ficou abaixo do acaso no ST (valor de -5,5% do fichamento).", "O valor está numa tabela que não foi possível conferir no texto extraído. Substituído pela frase verificável da p. 7 (falham na isquemia)."),
 ("Baer et al. [10]", "O texto dizia que a métrica chega a ordenar as classes ao contrário da verdade.", "O artigo diz que as métricas de perturbação e as baseadas na verdade conhecida frequentemente dão avaliações contraditórias. Corrigido para essa formulação."),
 ("Baer et al. [9]", "O texto dizia que os autores recomendam não ler a diferença entre classes como diferença de qualidade.", "A recomendação literal é complementar as médias agregadas com análise estratificada por classe. Corrigido."),
 ("Mercier et al. [33]", "O texto dizia que o GradientSHAP teve a maior sensibilidade entre os métodos de gradiente em seis conjuntos.", "Formulação do fichamento a partir de tabelas; substituída pelas frases literais do artigo (p. 6 e p. 11)."),
 ("Šimić et al. [22]", "O texto atribuía ao artigo que agrupar pontos melhora as métricas.", "O artigo trata de perturbações e do tamanho das regiões; a afirmação de melhora ficou só com Serramazza et al. [21]."),
 ("Fouladgar et al. [27]", "O texto dizia que a normalização evita o efeito de escala.", "O artigo só aponta o efeito de escala como causa provável; o texto agora diz isso, sem atribuir ao artigo a recomendação de normalizar."),
 ("Surawicz et al. (2009)", "Citada no texto de Metodologia e Resultados para a duração do QRS.", "Texto completo inacessível. Substituída por Hicks et al. [25], que definem bloqueio de ramo por QRS de 120 ms ou mais."),
 ("Bazett (1920)", "Citada como fonte da fórmula.", "Original inacessível. A fórmula e a sua limitação são citadas por Vandenberk et al. [26]."),
 ("Zhang et al. [1]", "O fichamento listava três autores (versão arXiv).", "A versão publicada na iScience tem quatro autores (Zhang, Yang, Yuan e Zhang); a referência usa a versão publicada."),
 ("Cui et al. [20]", "Nas limitações, o texto dizia que retirar regiões contíguas cria menos descontinuidades.", "O artigo fala em introduzir menos ruído de alta frequência; o texto passou a usar essa formulação."),
 ("Arends et al. [7]", "Na primeira versão da Introdução, o estudo era descrito pelos experimentos de concordância, randomização e perturbação.", "Na reescrita da Introdução, ficou só o dado usado no argumento (AUPRC média acima de 0,85 em oito classes); a concordância entre sementes continua citada nas limitações."),
 ("Introdução do TCC1", "Dados epidemiológicos (GBD 2025, World Heart Federation 2023) e a revisão de Manimaran et al. (2025).", "Textos não acessíveis para conferência; não foram usados no artigo."),
]
