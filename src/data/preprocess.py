"""Pré-processamento do CPSC2018 (seção 4.1.1 da proposta).

Protocolo (Zhang et al., 2021):
  1. Leitura dos registros WFDB de 12 derivações + rótulos do REFERENCE.csv.
  2. Truncamento/padding com zeros até ``SIGNAL_LENGTH`` (15.000) amostras.
  3. Normalização z-score por derivação (média/desvio por canal).
  4. Split estratificado por classe 70/20/10 com semente fixa.
  5. Registros multi-rótulo mantidos em representação nativa (vetor binário
     de 9 posições, compatível com BCE).

Saída: ``train.npz``, ``val.npz`` e ``test.npz`` no diretório processado,
cada um com arrays ``X`` (N, 12, 15000) float32, ``Y`` (N, 9) float32 e
``record_ids`` (N,) str.

Uso:
    python -m src.data.preprocess --raw-dir data/raw --out-dir data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.config import (
    CPSC_CODE_TO_INDEX,
    NUM_CLASSES,
    NUM_LEADS,
    SEED,
    SIGNAL_LENGTH,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)


# --------------------------------------------------------------------------- #
# Leitura de rótulos e sinais
# --------------------------------------------------------------------------- #
def load_reference(reference_csv: Path) -> dict[str, np.ndarray]:
    """Lê o REFERENCE.csv e devolve {record_id: vetor binário (9,)}.

    O REFERENCE.csv tem cabeçalho ``Recording,First_label,Second_label,
    Third_label`` com códigos inteiros 1..9 (Second/Third podem estar vazios).
    """
    import csv

    labels: dict[str, np.ndarray] = {}
    with open(reference_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # descarta cabeçalho
        del header
        for row in reader:
            if not row or not row[0]:
                continue
            rec = row[0].strip()
            y = np.zeros(NUM_CLASSES, dtype=np.float32)
            for code_str in row[1:4]:
                code_str = code_str.strip()
                if code_str and code_str.lower() != "nan":
                    code = int(float(code_str))
                    if code in CPSC_CODE_TO_INDEX:
                        y[CPSC_CODE_TO_INDEX[code]] = 1.0
            labels[rec] = y
    return labels


def read_signal(record_path: Path) -> np.ndarray:
    """Lê um registro WFDB e devolve array (12, T) float32.

    Requer o pacote ``wfdb``. ``record_path`` é o caminho sem extensão.
    """
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    sig = np.asarray(record.p_signal, dtype=np.float32)  # (T, 12)
    return sig.T  # (12, T)


# --------------------------------------------------------------------------- #
# Transformações do sinal
# --------------------------------------------------------------------------- #
def fix_length(signal: np.ndarray, length: int = SIGNAL_LENGTH) -> np.ndarray:
    """Trunca ou preenche com zeros até ``length`` amostras por derivação.

    ``signal`` tem forma (n_leads, T). Retorna (n_leads, length).
    """
    n_leads, t = signal.shape
    if t >= length:
        return signal[:, :length]
    out = np.zeros((n_leads, length), dtype=signal.dtype)
    out[:, :t] = signal
    return out


def zscore_per_lead(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalização z-score por derivação (canal), independente por linha."""
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return (signal - mean) / (std + eps)


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """Aplica fix_length + zscore_per_lead a um sinal (n_leads, T)."""
    return zscore_per_lead(fix_length(signal))


# --------------------------------------------------------------------------- #
# Split estratificado multi-rótulo
# --------------------------------------------------------------------------- #
def stratified_multilabel_split(
    Y: np.ndarray,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide índices em treino/val/teste preservando proporção por classe.

    Estratégia gulosa para multi-rótulo (iterative stratification simplificada):
    processa as classes da mais rara para a mais frequente e distribui seus
    registros ainda não alocados respeitando as cotas de cada partição. Isso
    aproxima a proporção de cada classe nos três subconjuntos mesmo com
    registros multi-rótulo, e é determinístico dada a semente.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    fracs = {"train": train_frac, "val": val_frac, "test": test_frac}

    assigned = np.full(n, -1, dtype=np.int64)  # -1 = não alocado
    part_index = {"train": 0, "val": 1, "test": 2}
    counts = {p: 0 for p in fracs}  # nº de registros já em cada partição

    class_freq = Y.sum(axis=0)
    class_order = np.argsort(class_freq)  # rara -> frequente

    for c in class_order:
        idx = np.where((Y[:, c] == 1) & (assigned == -1))[0]
        rng.shuffle(idx)
        # cotas-alvo para esta classe em cada partição
        n_c = len(idx)
        target = {p: fracs[p] * n_c for p in fracs}
        filled = {p: 0 for p in fracs}
        for i in idx:
            # escolhe a partição mais "faminta" (maior déficit relativo)
            deficit = {p: target[p] - filled[p] for p in fracs}
            p = max(deficit, key=lambda k: deficit[k])
            assigned[i] = part_index[p]
            filled[p] += 1
            counts[p] += 1

    # registros restantes (sem nenhum rótulo ou não alcançados) por proporção
    leftover = np.where(assigned == -1)[0]
    rng.shuffle(leftover)
    for i in leftover:
        totals = sum(counts.values()) or 1
        deficit = {p: fracs[p] - counts[p] / totals for p in fracs}
        p = max(deficit, key=lambda k: deficit[k])
        assigned[i] = part_index[p]
        counts[p] += 1

    train_idx = np.where(assigned == 0)[0]
    val_idx = np.where(assigned == 1)[0]
    test_idx = np.where(assigned == 2)[0]
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def _discover_records(raw_dir: Path, labels: dict[str, np.ndarray]) -> list[tuple[str, Path]]:
    """Associa cada record_id do REFERENCE a seu caminho WFDB (.hea)."""
    hea_by_stem = {p.stem: p.with_suffix("") for p in raw_dir.rglob("*.hea")}
    records: list[tuple[str, Path]] = []
    missing = 0
    for rec in labels:
        if rec in hea_by_stem:
            records.append((rec, hea_by_stem[rec]))
        else:
            missing += 1
    if missing:
        print(f"[aviso] {missing} registros do REFERENCE sem arquivo .hea correspondente")
    return sorted(records, key=lambda t: t[0])


def build_dataset(raw_dir: Path, out_dir: Path, limit: int | None = None) -> None:
    """Executa o pipeline completo e salva os splits em ``out_dir``."""
    from tqdm import tqdm

    reference_csv = next(raw_dir.rglob("REFERENCE.csv"))
    labels = load_reference(reference_csv)
    print(f"[info] {len(labels)} rótulos lidos de {reference_csv}")

    records = _discover_records(raw_dir, labels)
    if limit is not None:
        records = records[:limit]
    print(f"[info] {len(records)} registros a processar")

    X = np.zeros((len(records), NUM_LEADS, SIGNAL_LENGTH), dtype=np.float32)
    Y = np.zeros((len(records), NUM_CLASSES), dtype=np.float32)
    record_ids: list[str] = []

    for i, (rec, path) in enumerate(tqdm(records, desc="preprocess")):
        sig = read_signal(path)
        X[i] = preprocess_signal(sig)
        Y[i] = labels[rec]
        record_ids.append(rec)

    record_ids_arr = np.asarray(record_ids)
    train_idx, val_idx, test_idx = stratified_multilabel_split(Y)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X=X[idx],
            Y=Y[idx],
            record_ids=record_ids_arr[idx],
        )
        print(f"[salvo] {name}: {len(idx)} registros -> {out_dir / f'{name}.npz'}")

    _report_split(Y, train_idx, val_idx, test_idx)


def _report_split(Y, train_idx, val_idx, test_idx) -> None:
    from src.config import CLASSES

    print("\n[distribuição por classe (nº registros)]")
    header = f"{'classe':>6} | {'treino':>7} | {'val':>6} | {'teste':>6}"
    print(header)
    print("-" * len(header))
    for c, name in enumerate(CLASSES):
        tr = int(Y[train_idx, c].sum())
        va = int(Y[val_idx, c].sum())
        te = int(Y[test_idx, c].sum())
        print(f"{name:>6} | {tr:>7} | {va:>6} | {te:>6}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pré-processamento do CPSC2018")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None,
                        help="processa apenas os N primeiros registros (debug)")
    args = parser.parse_args(argv)
    build_dataset(args.raw_dir, args.out_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
