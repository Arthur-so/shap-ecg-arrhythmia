"""Monta a pasta de textos dos originais usada por verificar.py.

Baixa os PDFs do arXiv e os textos completos do Europe PMC e extrai com o
pdftotext os PDFs que estão em ~/Documentos/tcc/artigos-relacionados. Os
textos não são versionados (direitos autorais).

Uso: python3 baixar_originais.py <pasta de saída>
     python3 verificar.py <pasta de saída>
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ARXIV = {
    "gumpfer": "2607.24035", "mercier": "2202.03759", "ismail": "2010.13924",
    "hollig": "2310.14957", "sundararajan": "1703.01365", "captum": "2009.07896",
    "quantus": "2202.06861", "wagner": "2305.17043v2", "baer_a": "2502.17022",
    "baer_b": "2506.11790", "he": "1512.03385", "smilkov": "1706.03825",
    "lipton": "1402.1892", "metaquantus": "2302.07265", "lundberg": "1705.07874v2",
    "dasgupta": "2202.00734v1", "serramazza24": "2406.12507",
}
EPMC = {
    "arends": "PMC12980500", "ravindran": "PMC10584975", "hicks": "PMC8154909",
    "simic": "PMC12284047", "cui": "PMC10470463", "vandenberk": "PMC4937268",
}
LOCAIS = {
    "suh": "computers-in-biology-and-medicine.pdf",
    "turbe": "evaluating-of-post-hoc-interpretability-methods-in-time-series-classification.pdf",
    "storas": "Evaluating gradient-based explanation methods for neural.pdf",
    "zhang": "consistency.pdf",
    "bhatt": "faithcorr.pdf",
    "alvarez": "faithest.pdf",
    "yeh": "Avg-Max-Sensitivity.pdf",
    "sixt_local": "RandomLogitTest.pdf",
    "emprt": "EfficientSmoothMPRT.pdf",
    "adebayo": "MPRT.pdf",
    "fouladgar": "Metrics_and_Evaluations_of_Time_Series_Explanations_An_Application_in_Affect_Computing.pdf",
}
PASTA_LOCAL = Path.home() / "Documentos/tcc/artigos-relacionados"


def main(saida: str) -> int:
    out = Path(saida)
    out.mkdir(parents=True, exist_ok=True)
    for nome, ident in ARXIV.items():
        pdf = out / f"{nome}.pdf"
        if not pdf.exists():
            urllib.request.urlretrieve(f"https://arxiv.org/pdf/{ident}", pdf)
        subprocess.run(["pdftotext", str(pdf), str(out / f"{nome}.txt")], check=True)
    for nome, pmc in EPMC.items():
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML"
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        texto = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))
        (out / f"{nome}.txt").write_text(texto)
    for nome, arquivo in LOCAIS.items():
        subprocess.run(["pdftotext", str(PASTA_LOCAL / arquivo), str(out / f"{nome}.txt")], check=True)
    print(f"textos em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
