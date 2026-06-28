# Análise de Artigos Científicos com PLN

Este projeto foi desenvolvido para a disciplina de Introdução à Inteligência Artificial (IIA) do Departamento de Informática da Universidade Estadual de Maringá (UEM). O objetivo principal é utilizar técnicas de Processamento de Linguagem Natural (PLN) para analisar textos científicos. O tema selecionado pela equipe (Eduardo, Janaina e Nuno) foi "Bioinformática", utilizando artigos da base de dados IEEE Xplore.

## Sobre o Projeto

O sistema lê e processa 12 artigos científicos em formato PDF. As etapas de execução do pipeline incluem:

**Etapa 1:** Leitura de arquivos PDF em um diretório e pré-processamento textual, englobando a remoção de stop-words e lematização. A etapa também identifica os 10 termos mais citados e extrai as referências bibliográficas.

**Etapa 2:** Extração de informações específicas diretamente do texto, como objetivo, problema, método e contribuições da pesquisa.

**Etapa 3:** Salvamento dos dados extraídos na forma de uma ontologia de artigo científico no formato JSON-LD.

**Visualizações:** Geração de nuvens de palavras, gráficos de barras de frequência, heatmaps de coocorrência e análise da evolução temporal dos termos ao longo dos anos.

**Etapa 4:** Avaliação de desempenho do sistema de extração de informações em comparação com anotações manuais.


## Pré-requisitos

* É necessário ter o Python versão 3.10 ou superior instalado para suportar as tipagens utilizadas no código.
* O gerenciador de pacotes `pip` deve estar atualizado no seu ambiente.

## Configuração e Instalação

* Organize os arquivos do projeto mantendo o arquivo `main.py` e a pasta `src/` na raiz do diretório.
* Crie uma pasta chamada `artigos/` e adicione os arquivos PDF dentro dela.
* Instale as bibliotecas requeridas executando o comando `pip install -r requirements.txt` no terminal.
* As dependências do projeto incluem ferramentas como `PyMuPDF` para a leitura dos documentos e `nltk` para a tokenização e análise textual.
* Também serão instalados o `matplotlib`, o `numpy` e a biblioteca `wordcloud` para a geração dos gráficos e recursos visuais.

## Execução do Sistema

* Para que a Etapa 4 funcione, é obrigatório criar um arquivo chamado `gabarito.json` na raiz do projeto contendo as anotações manuais de 4 artigos.
* Os nomes das chaves preenchidas dentro do arquivo `gabarito.json` precisam ser idênticos aos nomes dos arquivos PDF armazenados na pasta `artigos/`.
* Com o ambiente configurado, execute o comando `python main.py` na raiz do projeto para rodar a aplicação.
* O pipeline unificado orquestrado pelo `main.py` roda as etapas sequencialmente na ordem: Etapa 1, Etapa 2, Etapa 3, Visualizações e Etapa 4.

## Estrutura de Saída

Após a execução completa do projeto, os dados gerados serão automaticamente organizados dentro do diretório `saida/`:

* A pasta `jsonld/` conterá a ontologia individualizada de cada artigo e um arquivo unificado do corpus consolidado.
* A pasta `visualizacoes/` armazenará todas as imagens produzidas, como a nuvem de palavras dos trabalhos futuros e o ranking de termos frequentes.
* A pasta `avaliacao/` abrigará os resultados visuais e numéricos da avaliação de desempenho, separados por campos de extração.