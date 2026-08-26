"""Configurações globais e constantes compartilhadas do projeto.

Centraliza parâmetros do protocolo (Zhang et al., 2021) usados por vários
módulos: número de derivações, comprimento fixo do sinal, nomes/ordem das
classes diagnósticas e semente de reprodutibilidade.
"""
from __future__ import annotations

# --- Sinal de ECG (CPSC2018) ---
NUM_LEADS = 12          # derivações do ECG padrão
SAMPLING_RATE = 500     # Hz
SIGNAL_LENGTH = 15000   # amostras (= 30 s a 500 Hz), truncamento/padding

# --- Classes diagnósticas ---
# A ordem segue a codificação oficial do CPSC2018 (rótulos 1..9 do
# REFERENCE.csv), preservada em todo o pipeline (modelo, SHAP, avaliação).
CLASSES = ["SNR", "AF", "IAVB", "BRE", "BRD", "CAP", "CVP", "STD", "STE"]
NUM_CLASSES = len(CLASSES)

# Nomes por extenso (para relatórios/tabelas)
CLASS_NAMES = {
    "SNR": "Ritmo sinusal normal",
    "AF": "Fibrilacao atrial",
    "IAVB": "Bloqueio AV de 1o grau",
    "BRE": "Bloqueio de ramo esquerdo",
    "BRD": "Bloqueio de ramo direito",
    "CAP": "Contracao atrial prematura",
    "CVP": "Contracao ventricular prematura",
    "STD": "Depressao do segmento ST",
    "STE": "Elevacao do segmento ST",
}

# Mapeia o código inteiro do REFERENCE.csv (1..9) -> índice da classe (0..8)
CPSC_CODE_TO_INDEX = {i + 1: i for i in range(NUM_CLASSES)}

# Mapeia códigos SNOMED CT -> índice da classe (0..8). Necessário porque a
# distribuição do CPSC2018 no Kaggle/PhysioNet (formato PhysioNet-CinC 2020)
# não traz o REFERENCE.csv original; os rótulos ficam no campo "#Dx:" de cada
# cabeçalho .hea como códigos SNOMED (possivelmente múltiplos, separados por
# vírgula). Inclui os códigos equivalentes usados no desafio CinC 2020 (ex.:
# RBBB completo, batimentos ventriculares/supraventriculares prematuros) para
# não perder rótulos.
SNOMED_TO_INDEX = {
    # SNR — ritmo sinusal normal
    "426783006": 0,
    # AF — fibrilação atrial
    "164889003": 1,
    # IAVB — bloqueio AV de 1º grau
    "270492004": 2,
    # BRE — bloqueio de ramo esquerdo (LBBB)
    "164909002": 3,
    # BRD — bloqueio de ramo direito (RBBB e RBBB completo)
    "59118001": 4,
    "713427006": 4,
    # CAP — contração atrial prematura (PAC e batimentos supraventriculares prematuros)
    "284470004": 5,
    "63593006": 5,
    # CVP — contração ventricular prematura (PVC e batimentos ventriculares prematuros)
    "427172004": 6,
    "17338001": 6,
    # STD — depressão do segmento ST
    "429622005": 7,
    # STE — elevação do segmento ST
    "164931005": 8,
}

# --- Reprodutibilidade ---
SEED = 42

# --- Split (estratificado por classe) ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
