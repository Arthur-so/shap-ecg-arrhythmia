# Apoio ao artigo do TCC

A pasta `artigo/` tem só o necessário para compilar o artigo (é a que vai para o Overleaf).
Esta pasta guarda o restante.

| Arquivo | Conteúdo |
|---|---|
| `artigo.pdf` | Última versão compilada do artigo (copiar para cá depois de compilar) |
| `revisao/` | PDFs anotados pelo aluno em cada rodada de revisão |
| `verificacao-citacoes/` | Verificação de cada citação: onde aparece, o que afirma e o trecho do original |
| `figs/gerar_esquema.py` | Gera `artigo/figs/esquema_regioes.pdf` (ECG sintético) |

As demais figuras do artigo são geradas por `texto-tcc/metodologia-resultados/figs/gerar_figuras.py`
e copiadas para `artigo/figs/`.

## Compilar o artigo localmente

```
cd artigo
pdflatex main && bibtex main && pdflatex main && pdflatex main
cp main.pdf ../artigo-apoio/artigo.pdf
```

Os arquivos gerados (`main.pdf`, `.aux`, `.bbl`, `.log`…) ficam fora do git. No Overleaf, enviar a
pasta `artigo/` inteira; `IEEEtran.cls` é a classe do template da universidade (com a correção de um
texto solto, `TCC`, após `\def\indexname{Index}`), e `IEEEtran.bst` é a versão 1.14 do CTAN.

## Verificação das citações

`verificacao-citacoes/main.pdf` lista, para cada citação do artigo, a seção em que aparece, a afirmação
feita, o trecho literal do trabalho citado e a página, além das correções feitas durante a conferência.

- `dados.py`: os dados da verificação (editar aqui ao mudar o artigo).
- `gerar.py`: gera `main.tex` a partir de `dados.py` (`python3 gerar.py`).
- `baixar_originais.py`: monta a pasta com os textos dos originais
  (`python3 baixar_originais.py <pasta>`), baixando do arXiv e do Europe PMC e extraindo os PDFs de
  `~/Documentos/tcc/artigos-relacionados`. Os textos não ficam no repositório (direitos autorais).
- `verificar.py`: confere se cada trecho está no texto do original (`python3 verificar.py <pasta>`).
