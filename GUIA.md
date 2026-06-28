# Guia de Execução — Trabalho de IIA/PLN
**Universidade Estadual de Maringá | Bioinformática | IEEE Xplore**

---

## Pré-requisitos

- Python **3.10 ou superior** (para suportar `int | None` como tipo)
- pip atualizado

Verifique sua versão:
```bash
python --version
pip --version
```

---

## 1. Estrutura esperada do projeto

Antes de rodar, organize as pastas assim:

```
projeto/
├── main.py
├── requirements.txt
├── gabarito.json          ← Janaina preenche (veja seção 4)
├── artigos/               ← coloque os 12 PDFs aqui
└── src/
    ├── etapa1.py
    ├── etapa2.py
    ├── etapa3.py
    ├── etapa4.py
    └── visualizacoes.py
```

A pasta `saida/` é criada automaticamente ao rodar.

---

## 2. Instalar dependências

Na pasta raiz do projeto, rode:

```bash
pip install -r requirements.txt
```

O que será instalado:
- `pdfplumber` — leitura dos PDFs
- `nltk` — tokenização, stopwords, lematização, n-gramas
- `matplotlib` — gráficos e heatmaps
- `numpy` — operações numéricas nos gráficos
- `wordcloud` — nuvens de palavras

> **Dica:** se quiser usar um ambiente virtual (recomendado):
> ```bash
> python -m venv venv
> # Windows:
> venv\Scripts\activate
> # Mac/Linux:
> source venv/bin/activate
> pip install -r requirements.txt
> ```

---

## 3. Adicionar os artigos

Coloque os **12 PDFs** dentro da pasta `artigos/`.

**Dica de nomenclatura:** inclua o ano no nome do arquivo para que o gráfico temporal funcione corretamente. Exemplos de nomes bons:

```
artigos/
├── protein_prediction_2021.pdf
├── genomic_network_2022.pdf
├── essential_proteins_vit_2023.pdf
...
```

---

## 4. Preencher o gabarito (Janaina)

A etapa 4 precisa de um arquivo `gabarito.json` na raiz do projeto com anotações manuais de **4 artigos**.

Use o arquivo `gabarito_exemplo.json` como modelo:

1. Copie e renomeie: `gabarito_exemplo.json` → `gabarito.json`
2. Substitua os nomes das chaves pelo nome **exato** dos PDFs (incluindo `.pdf`)
3. Preencha cada campo com um trecho representativo copiado do artigo

Exemplo de gabarito preenchido:
```json
{
  "protein_prediction_2021.pdf": {
    "objetivo": "The objective of this paper is to propose a deep learning framework for predicting essential proteins in PPI networks.",
    "problema": "Existing methods rely on single-omics data and fail to capture complex biological interactions.",
    "metodo": "We applied a Vision Transformer trained on fused PPI and subcellular localization data using an outer product operation.",
    "contribuicao": "Our study contributes to bioinformatics by introducing EPViT, a novel framework that outperforms existing methods on three PPI datasets."
  },
  "artigo2.pdf": { ... },
  "artigo3.pdf": { ... },
  "artigo4.pdf": { ... }
}
```

> O nome da chave precisa ser **idêntico** ao nome do arquivo na pasta `artigos/`, incluindo maiúsculas e extensão.

---

## 5. Rodar o projeto

Com tudo configurado, execute na raiz do projeto:

```bash
python main.py
```

O pipeline roda em ordem: Etapa 1 → 2 → 3 → Visualizações → 4.

---

## 6. O que será gerado

```
saida/
├── jsonld/
│   ├── protein_prediction_2021_pdf.jsonld   ← ontologia de cada artigo
│   ├── genomic_network_2022_pdf.jsonld
│   ├── ...
│   └── corpus_consolidado.jsonld            ← todos os artigos em um único arquivo
│
├── visualizacoes/
│   ├── nuvem_geral.png                      ← nuvem de palavras do corpus
│   ├── nuvem_<artigo>.png                   ← nuvem por artigo
│   ├── barras_top_termos.png                ← top 20 termos mais frequentes
│   ├── heatmap_coocorrencia.png             ← frequência dos top termos por artigo
│   ├── temporal_termos.png                  ← evolução dos termos por ano
│   ├── termos_trabalhos_futuros.png         ← termos de "future work"
│   └── nuvem_futuros.png                    ← nuvem de trabalhos futuros
│
└── avaliacao/
    ├── avaliacao_por_campo.png              ← P, R, F1 por campo (barras)
    ├── f1_por_artigo.png                    ← heatmap F1 por artigo × campo
    └── resultados_avaliacao.json            ← números completos em JSON
```

---

## 7. Rodar apenas uma etapa (opcional)

Cada módulo pode ser executado de forma isolada para testes:

```bash
# Só a etapa 1
python -m src.etapa1

# Só a etapa 2 (requer ter rodado a 1 antes — integração via main.py)
# Rode o main.py normalmente; as etapas são encadeadas
```

---

## 8. Problemas comuns

**Erro: `No module named 'pdfplumber'`**
→ As dependências não foram instaladas. Rode `pip install -r requirements.txt`.

**Erro: `[Errno 2] No such file or directory: './artigos'`**
→ Crie a pasta `artigos/` e coloque os PDFs dentro.

**Nuvem de palavras não gerada**
→ O pacote `wordcloud` pode falhar em alguns ambientes. Instale manualmente:
```bash
pip install wordcloud
```

**Gráfico temporal vazio**
→ O ano não foi detectado nos nomes dos arquivos. Renomeie os PDFs incluindo o ano (ex: `artigo_2022.pdf`).

**Etapa 4 pulada / gabarito não encontrado**
→ O arquivo `gabarito.json` precisa estar na **raiz do projeto** (mesma pasta que o `main.py`).

**NLTK reclamando de recursos faltando**
→ Na primeira execução os recursos são baixados automaticamente. Se falhar (sem internet), rode manualmente:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## 9. Divisão de responsabilidades (referência)

| Integrante | Arquivos |
|---|---|
| **Nuno** | `src/etapa1.py` + parte do `src/etapa2.py` (objetivo e problema) |
| **Eduardo** | `src/etapa2.py` (método e contribuição) + `src/etapa3.py` |
| **Janaina** | `gabarito.json` + `src/etapa4.py` + slides |