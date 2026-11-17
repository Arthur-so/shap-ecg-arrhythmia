"""Confere, por busca automática, se cada trecho citado em dados.py está no
texto extraído do original (pdftotext ou texto completo do Europe PMC).

Uso: python3 verificar.py <pasta com os .txt dos originais>
Os textos dos originais não são versionados (direitos autorais); a lista de
arquivos esperados está em ARQUIVOS.
"""
import re
import sys
import unicodedata

from dados import REFS

ARQUIVOS = {
    "zhang2021": "zhang", "lundberg2017": "lundberg", "adebayo2018": "adebayo",
    "hedstrom2023quantus": "quantus", "bhatt2020": "bhatt", "suh2024": "suh",
    "arends2026": "arends", "gumpfer2026": "gumpfer", "baer2025a": "baer_a",
    "baer2025b": "baer_b", "hollig2023": "hollig", "he2016": "he",
    "lipton2014": "lipton", "kokhlikyan2020": "captum",
    "sundararajan2017": "sundararajan", "turbe2023": "turbe",
    "smilkov2017": "smilkov", "cui2023": "cui", "serramazza2024": "serramazza24",
    "simic2025": "simic", "wagner2024": "wagner", "hicks2021": "hicks",
    "vandenberk2016": "vandenberk", "fouladgar2022": "fouladgar",
    "alvarezmelis2018": "alvarez", "yeh2019": "yeh", "dasgupta2022": "dasgupta",
    "sixt2020": "sixt_local", "hedstrom2023emprt": "emprt", "mercier2022": "mercier",
    "ravindran2023": "ravindran", "hedstrom2023metaquantus": "metaquantus",
    "ismail2020": "ismail", "storas2025": "storas",
}
PT = re.compile(r"[ãçõáéíóúâê]|^(Tabela|Conferido|O artigo|README|Citação|Texto|Referência)")


def norm(x: str) -> str:
    x = unicodedata.normalize("NFKC", x)
    # o pdftotext transcreve o apóstrofo de x' (linha) como "x0"
    x = re.sub(r"(?<![A-Za-z])([a-z])['′]", r"\g<1>0", x)
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&amp;", "&"), ("’", "'")):
        x = x.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "", x.lower())


def main(pasta: str) -> int:
    falhas = 0
    for n, key, _short, _src, occ in REFS:
        if key not in ARQUIVOS:
            print(f"[{n}] {key}: sem texto local (ver fonte na tabela)")
            continue
        texto = norm(open(f"{pasta}/{ARQUIVOS[key]}.txt", errors="ignore").read())
        for loc, _claim, quote, _pg in occ:
            for parte in re.split(r"\|\||\[\.\.\.\]", quote):
                parte = parte.strip()
                if len(parte) < 25 or PT.search(parte):
                    continue
                p = norm(parte)
                blocos = [p[i:i + 40] for i in range(0, max(1, len(p) - 40), 40)]
                faltam = [b for b in blocos if b not in texto]
                if faltam:
                    falhas += 1
                    print(f"[{n}] {key} | {loc}: {len(faltam)}/{len(blocos)} blocos não achados: {parte[:80]}")
    print("Todos os trechos encontrados." if not falhas else f"{falhas} trecho(s) com falha.")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
