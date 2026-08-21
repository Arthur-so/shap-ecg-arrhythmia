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

# --- Reprodutibilidade ---
SEED = 42

# --- Split (estratificado por classe) ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
