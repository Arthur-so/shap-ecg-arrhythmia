# Artigo do TCC

Artigo no template da Unifesp (IEEEtran, duas colunas), em português.

| Arquivo | Conteúdo |
|---|---|
| `main.pdf` | Artigo compilado (10 páginas) |
| `main.tex` | Título, autores, resumo, palavras-chave e inclusão das seções |
| `Secoes/` | Introdução, Metodologia, Resultados, Discussão e Conclusão |
| `referencias.bib` | 37 referências, com metadados conferidos no original ou no Europe PMC |
| `figs/` | Figuras: PDFs gerados por `texto-tcc/metodologia-resultados/figs/gerar_figuras.py` e esquemas em TikZ (`pipeline.tex`, `resnet.tex`, `piso.tex`) |
| `IEEEtran.cls` | Classe do template da universidade (corrigido um texto solto, `TCC`, após `\def\indexname{Index}`, que gerava erro de compilação) |
| `IEEEtran.bst` | Estilo de referências IEEE (versão 1.14, do CTAN; não vinha com o template) |
| `verificacao-citacoes/` | Verificação de cada citação: onde aparece, o que afirma e o trecho do original (ver abaixo) |

## Compilar

```
cd artigo
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Verificação das citações

`verificacao-citacoes/main.pdf` lista, para cada citação do artigo, a seção em que aparece, a afirmação
feita, o trecho literal do trabalho citado e a página, além das correções feitas durante a conferência.

- `dados.py`: os dados da verificação (editar aqui ao mudar o artigo).
- `gerar.py`: gera `main.tex` a partir de `dados.py` (`python3 gerar.py`).
- `verificar.py`: confere se cada trecho está no texto extraído do original
  (`python3 verificar.py <pasta com os .txt>`). Os textos dos originais não ficam no repositório;
  os PDFs estão em `~/Documentos/tcc/artigos-relacionados` ou são baixados do arXiv e do Europe PMC
  (lista de arquivos em `verificar.py`).
