"""Gera main.tex (verificação das citações) a partir de dados.py, com o
preâmbulo do template de relatório da Unifesp (preambulo.tex)."""
import re

from dados import CORRECOES, FONTES, REFS


def esc(t: str) -> str:
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("$", "\\$"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        t = t.replace(a, b)
    t = t.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    t = t.replace(" || ", " \\textbar\\textbar{} ")
    return t


def fonte(src: str) -> str:
    chave = src.split(" ")[0]
    return esc(src if chave not in FONTES else f"{FONTES[chave]} {src[len(chave):].strip()}")


partes = [open("preambulo.tex").read(), r"""\usepackage[brazil]{babel}
\usepackage{newunicodechar}
\usepackage{longtable}
\usepackage{array}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{ê}{\^e}
\newunicodechar{′}{\ensuremath{'}}
\renewcommand{\arraystretch}{1.15}

\begin{document}

\begin{center}
    {\LARGE \textbf{Verificação das citações do artigo}} \\[0.35em]
    {\large Avaliação das Explicações SHAP em Redes Neurais Profundas para Classificação de Arritmias Cardíacas em ECG} \\[0.3em]
    \textbf{Autor:} Arthur Santos de Oliveira (RA: 156297) \quad \textbf{Orientador:} Prof.\ Dr.\ Luis Augusto Martins Pereira \\[0.2em]
    \textbf{Data:} 13/11/2026 \quad \textbf{Repositório:} \url{https://github.com/Arthur-so/shap-ecg-arrhythmia}
\end{center}
\vspace{-0.5em}
\hrule
\vspace{0.8em}

\section{Como a verificação foi feita}
Para cada citação do artigo (\texttt{artigo/main.pdf}), este documento indica \textbf{onde} ela aparece, \textbf{o que} o artigo afirma com base nela e o \textbf{trecho do trabalho citado} que sustenta a afirmação, com a página. Os números entre colchetes são os do artigo.

\begin{itemize}
\item \textbf{Fonte consultada:} sempre o texto original, e não só o fichamento: o PDF em \texttt{\textasciitilde/Documentos/tcc/artigos-relacionados}, o PDF do arXiv ou o texto completo do Europe PMC. As exceções estão indicadas na coluna ``Fonte'' da Tabela~\ref{tab:resumo}.
\item \textbf{Conferência automática:} todos os trechos em inglês desta lista foram buscados no texto extraído do original pelo programa \texttt{verificar.py} (nesta pasta), que compara o trecho, em blocos, com o texto do PDF; todos foram encontrados. Fórmulas foram omitidas com [...], e a linha de $x'$, que o \texttt{pdftotext} transcreve como ``x0'', foi tratada como equivalente.
\item \textbf{Páginas:} numeração do PDF consultado. Textos do Europe PMC não têm paginação; nesses casos, indica-se a seção.
\item \textbf{Trechos múltiplos:} separados por ``\textbar\textbar''. Comentários do TCC aparecem em português.
\end{itemize}

\section{Resumo por referência}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.6cm}>{\raggedright\arraybackslash}p{6.0cm}>{\raggedright\arraybackslash}p{8.0cm}>{\centering\arraybackslash}p{1.2cm}}
\caption{Referências do artigo, fonte consultada e número de ocorrências no texto.}\label{tab:resumo}\\
\toprule
\textbf{N.º} & \textbf{Referência} & \textbf{Fonte consultada} & \textbf{Ocorr.} \\
\midrule
\endfirsthead
\toprule
\textbf{N.º} & \textbf{Referência} & \textbf{Fonte consultada} & \textbf{Ocorr.} \\
\midrule
\endhead
"""]
for n, key, short, src, occ in REFS:
    partes.append(f"{{[{n}]}} & {esc(short)} & {fonte(src)} & {len(occ)} \\\\\n")
partes.append("\\bottomrule\n\\end{longtable}\n\n\\section{Citações, uma a uma}\n")
for n, key, short, src, occ in REFS:
    partes.append(f"\\subsection*{{[{n}] {esc(short)}}}\n")
    partes.append("\\begin{longtable}{>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{4.6cm}>{\\raggedright\\arraybackslash}p{7.6cm}>{\\raggedright\\arraybackslash}p{1.4cm}}\n")
    partes.append("\\toprule\n\\textbf{Onde} & \\textbf{Afirmação no artigo} & \\textbf{Trecho do original} & \\textbf{Página} \\\\\n\\midrule\n")
    for loc, claim, quote, pg in occ:
        q = esc(quote)
        if not re.match(r"^(Tabela|Conferido|O artigo|README|Citação|Texto|Referência)", quote):
            q = "\\textit{``" + q + "''}"
        partes.append(f"{esc(loc)} & {esc(claim)} & {q} & {esc(pg)} \\\\\n\\addlinespace\n")
    partes.append("\\bottomrule\n\\end{longtable}\n")

partes.append(r"""
\section{Correções feitas durante a verificação}
A conferência no original mostrou formulações que diziam mais do que os trabalhos sustentam, em geral herdadas de resumos dos fichamentos. Todas foram corrigidas no artigo antes desta versão.
\begin{longtable}{>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{6.2cm}>{\raggedright\arraybackslash}p{7.2cm}}
\toprule
\textbf{Referência} & \textbf{O que estava escrito} & \textbf{O que o original sustenta e a correção} \\
\midrule
""")
for r, antes, depois in CORRECOES:
    partes.append(f"{esc(r)} & {esc(antes)} & {esc(depois)} \\\\\n\\addlinespace\n")
partes.append(r"""\bottomrule
\end{longtable}

\section{Afirmações sem citação}
Os números do próprio estudo (F1, valores das métricas, piso, frequência cardíaca, número de batimentos, limiares) não têm citação: vêm dos arquivos de resultados do repositório, em \texttt{results/modeloB\_v2/} (métricas, piso e robustez dos limites), \texttt{results/ablacao\_bce/} (treino, F1 e ablação) e dos scripts em \texttt{src/}. A afirmação de que a delineação das ondas pelo NeuroKit2 falhou em ECGs de AF e de bloqueio de ramo vem do relatório de status de 09/10/2026 (\texttt{status do projeto/relatorio-status-2026-10-09}).

\end{document}
""")
open("main.tex", "w").write("".join(partes))
print("main.tex gerado")
