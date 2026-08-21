"""Download e organização do conjunto de dados CPSC2018.

O CPSC2018 (China Physiological Signal Challenge 2018) é composto por 6.877
registros de ECG de 12 derivações (formato WFDB: pares .hea/.mat) mais um
arquivo REFERENCE.csv com os rótulos diagnósticos (1..9) por registro.

Este script NÃO baixa o dataset automaticamente sem confirmação explícita
(``--confirm``), pois o arquivo é grande e o ambiente de execução pode ter
restrições de rede. Sem ``--confirm`` ele apenas valida/organiza um diretório
de dados brutos já existente.

Uso:
    python -m src.data.download --raw-dir data/raw            # apenas valida
    python -m src.data.download --raw-dir data/raw --confirm  # baixa de fato
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# URLs oficiais do CPSC2018 (training set liberado pelo desafio ICBEB/CPSC).
# São mantidas aqui como referência; o download só ocorre com --confirm.
CPSC2018_URLS = [
    "http://2018.icbeb.org/file/REFERENCE.csv",
    "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip",
    "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip",
    "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip",
]


def _download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    out = dest_dir / fname
    if out.exists():
        print(f"[skip] já existe: {out}")
        return out
    print(f"[download] {url} -> {out}")
    urlretrieve(url, out)  # noqa: S310 (URL fixa e confiável do desafio)
    return out


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_cpsc2018(raw_dir: Path) -> None:
    """Baixa e extrai os arquivos do CPSC2018 em ``raw_dir``."""
    for url in CPSC2018_URLS:
        path = _download(url, raw_dir)
        if path.suffix == ".zip":
            _extract_zip(path, raw_dir)


def summarize_raw(raw_dir: Path) -> dict[str, int]:
    """Conta registros WFDB (.hea) e verifica presença do REFERENCE.csv."""
    hea = list(raw_dir.rglob("*.hea"))
    mat = list(raw_dir.rglob("*.mat"))
    ref = list(raw_dir.rglob("REFERENCE.csv"))
    summary = {
        "hea_files": len(hea),
        "mat_files": len(mat),
        "reference_csv": len(ref),
    }
    print("[resumo]", summary)
    if not ref:
        print("[aviso] REFERENCE.csv não encontrado em", raw_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/organização do CPSC2018")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="diretório de destino dos dados brutos")
    parser.add_argument("--confirm", action="store_true",
                        help="confirma o download real dos arquivos (grande)")
    args = parser.parse_args(argv)

    if args.confirm:
        download_cpsc2018(args.raw_dir)
    else:
        print("[info] execução sem --confirm: apenas validando diretório.")
        print("[info] use --confirm para baixar o dataset de fato.")

    if args.raw_dir.exists():
        summarize_raw(args.raw_dir)
    else:
        print(f"[info] diretório {args.raw_dir} ainda não existe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
