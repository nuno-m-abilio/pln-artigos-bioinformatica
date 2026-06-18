import os
import re
from collections import Counter

import nltk
import pdfplumber
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Download dos recursos NLTK necessários ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def baixar_recursos_nltk():
    recursos = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for r in recursos:
        nltk.download(r, quiet=True)


# Leitura de PDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ler_pdf(caminho_pdf: str) -> str:
    """Extrai todo o texto de um arquivo PDF usando pdfplumber."""
    texto = ""
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                conteudo = pagina.extract_text()
                if conteudo:
                    texto += conteudo + "\n"
    except Exception as e:
        print(f"  [ERRO] Não foi possível ler '{caminho_pdf}': {e}")
    return texto


def ler_pdfs_do_diretorio(diretorio: str) -> dict[str, str]:
    """
    Lê todos os PDFs de um diretório.
    Retorna dicionário {nome_arquivo: texto_completo}.
    """
    artigos:dict[str, str] = {}
    arquivos_pdf = [f for f in os.listdir(diretorio) if f.lower().endswith(".pdf")]

    if not arquivos_pdf:
        print(f"[AVISO] Nenhum PDF encontrado em '{diretorio}'")
        return artigos

    print(f"[INFO] {len(arquivos_pdf)} PDF(s) encontrado(s) em '{diretorio}'")

    for nome_arquivo in sorted(arquivos_pdf):
        caminho = os.path.join(diretorio, nome_arquivo)
        print(f"  Lendo: {nome_arquivo}")
        texto = ler_pdf(caminho)
        if texto.strip():
            artigos[nome_arquivo] = texto

    return artigos


# Separação: corpo do artigo x referências ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Padrões comuns de início da seção de referências
_PADROES_REFERENCIAS = [
    r"\breferences\b",
    r"\bbibliography\b",
    r"\bbibliograf[íi]a\b",
    r"\breferências\b",
    r"\bliterature cited\b",
    r"\bworks cited\b",
]

_REGEX_REFERENCIAS = re.compile(
    "|".join(_PADROES_REFERENCIAS),
    flags=re.IGNORECASE,
)


def separar_corpo_e_referencias(texto: str) -> tuple[str, str]:
    """
    Divide o texto em (corpo_do_artigo, secao_referencias).
    Busca a ÚLTIMA ocorrência do cabeçalho de referências para evitar
    falsos positivos em menções no corpo do texto.
    """
    matches = list(_REGEX_REFERENCIAS.finditer(texto))

    if not matches:
        return texto, ""

    # Usa a última ocorrência (a seção real de referências fica no final)
    ultimo_match = matches[-1]
    inicio_refs = ultimo_match.start()

    corpo = texto[:inicio_refs].strip()
    referencias = texto[inicio_refs:].strip()

    return corpo, referencias


# ── Extração de referências individuais ───────────────────────────────────────

def extrair_referencias(secao_referencias: str) -> list[str]:
    """
    Extrai referências individuais da seção de referências.
    Suporta dois formatos comuns:
      - Numeradas:  [1] Autor, Título...  ou  1. Autor, Título...
      - Por autor:  Sobrenome, I. (Ano). Título...
    """
    if not secao_referencias:
        return []

    linhas = secao_referencias.splitlines()

    # Remove a linha de cabeçalho (ex: "References", "Bibliography")
    if linhas and _REGEX_REFERENCIAS.search(linhas[0]):
        linhas = linhas[1:]

    referencias = []
    ref_atual = ""

    # Padrões de início de nova referência
    inicio_numerada = re.compile(r"^\s*(\[\d+\]|\d+[\.\)])\s+\S")
    inicio_autor = re.compile(r"^\s*[A-Z][a-zA-Zá-ú\-]+,\s+[A-Z]")

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        nova_ref = inicio_numerada.match(linha) or inicio_autor.match(linha)

        if nova_ref:
            if ref_atual.strip():
                referencias.append(ref_atual.strip())
            ref_atual = linha
        else:
            ref_atual += " " + linha

    if ref_atual.strip():
        referencias.append(ref_atual.strip())

    # Filtra entradas muito curtas (ruído de extração)
    referencias = [r for r in referencias if len(r) > 20]

    return referencias


# ── Pré-processamento ──────────────────────────────────────────────────────────

_STOP_WORDS = set(stopwords.words("english"))

# Stop words extras comuns em artigos científicos que poluem os resultados
_STOP_WORDS_EXTRAS = {
    "et", "al", "fig", "figure", "table", "also", "show", "shown",
    "use", "used", "using", "based", "however", "thus", "therefore",
    "paper", "study", "result", "results", "proposed", "method",
    "approach", "work", "article", "section", "ieee", "doi", "http",
    "https", "www", "e", "g", "i",
}

_STOP_WORDS.update(_STOP_WORDS_EXTRAS)

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()


def _get_wordnet_pos(tag: str) -> str:
    """Converte tag do POS tagger do NLTK para formato do WordNet."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocessar_texto(
    texto: str,
    lematizar: bool = True,
    aplicar_stemming: bool = False,
) -> list[str]:
    """
    Pré-processa o texto e retorna lista de tokens limpos.

    Passos:
      1. Tokenização
      2. Lowercase + filtra apenas palavras alfabéticas com 3+ chars
      3. Remove stop words
      4. Lematização (padrão) e/ou stemming (opcional)
    """
    tokens = word_tokenize(texto.lower())

    # Mantém apenas palavras alfabéticas com pelo menos 3 caracteres
    tokens = [t for t in tokens if t.isalpha() and len(t) >= 3]

    # Remove stop words
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    if lematizar:
        # POS tagging melhora a qualidade da lematização
        tags = nltk.pos_tag(tokens)
        tokens = [
            _LEMMATIZER.lemmatize(token, _get_wordnet_pos(tag))
            for token, tag in tags
        ]

    if aplicar_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens


# ── Modelos de linguagem ───────────────────────────────────────────────────────

def bag_of_words(tokens: list[str]) -> dict[str, int]:
    """Retorna dicionário {termo: frequência} — Bag of Words."""
    return dict(Counter(tokens))


def ngramas(tokens: list[str], n: int) -> list[tuple]:
    """Gera lista de n-gramas a partir dos tokens."""
    return list(nltk.ngrams(tokens, n))


def contar_ngramas(tokens: list[str], n: int) -> dict[tuple, int]:
    """Retorna dicionário {n-grama: frequência}."""
    return dict(Counter(ngramas(tokens, n)))


def top_n_termos(contagem: dict, n: int = 10) -> list[tuple]:
    """Retorna os N termos/n-gramas mais frequentes como lista de (termo, freq)."""
    return Counter(contagem).most_common(n)


# ── Processamento completo de um artigo ───────────────────────────────────────

def processar_artigo(nome: str, texto_completo: str) -> dict:
    """
    Executa todo o pipeline da Etapa 1 para um único artigo.
    Retorna dicionário com todos os resultados.
    """
    print(f"\n  Processando: {nome}")

    # 1. Separar corpo e referências
    corpo, secao_refs = separar_corpo_e_referencias(texto_completo)
    print(f"    Corpo: {len(corpo)} chars | Refs: {len(secao_refs)} chars")

    # 2. Extrair referências individuais
    lista_referencias = extrair_referencias(secao_refs)
    print(f"    Referências extraídas: {len(lista_referencias)}")

    # 3. Pré-processar apenas o corpo
    tokens = preprocessar_texto(corpo, lematizar=True, aplicar_stemming=False)
    print(f"    Tokens após pré-processamento: {len(tokens)}")

    # 4. Bag of Words (unigramas)
    bow = bag_of_words(tokens)

    # 5. Bigramas e trigramas
    contagem_bigramas = contar_ngramas(tokens, 2)
    contagem_trigramas = contar_ngramas(tokens, 3)

    # 6. Top 10 termos (unigramas)
    top10_unigramas = top_n_termos(bow, 10)

    # 7. Top 10 bigramas e trigramas
    top10_bigramas = top_n_termos(contagem_bigramas, 10)
    top10_trigramas = top_n_termos(contagem_trigramas, 10)

    return {
        "nome": nome,
        "corpo": corpo,
        "secao_referencias": secao_refs,
        "referencias": lista_referencias,
        "tokens": tokens,
        "bow": bow,
        "top10_unigramas": top10_unigramas,
        "top10_bigramas": top10_bigramas,
        "top10_trigramas": top10_trigramas,
    }


# ── Agregação entre todos os artigos ──────────────────────────────────────────

def top_termos_globais(resultados: list[dict], n: int = 10) -> list[tuple]:
    """
    Agrega os BoW de todos os artigos e retorna os N termos mais frequentes
    considerando o corpus inteiro.
    """
    contador_global = Counter()
    for r in resultados:
        contador_global.update(r["bow"])
    return contador_global.most_common(n)


# ── Exibição formatada dos resultados ─────────────────────────────────────────

def exibir_resultado(resultado: dict):
    nome = resultado["nome"]
    print(f"\n{'='*60}")
    print(f"  ARTIGO: {nome}")
    print(f"{'='*60}")

    print("\n  TOP 10 UNIGRAMAS:")
    for i, (termo, freq) in enumerate(resultado["top10_unigramas"], 1):
        print(f"    {i:2}. {termo:<25} {freq} ocorrências")

    print("\n  TOP 10 BIGRAMAS:")
    for i, (bigrama, freq) in enumerate(resultado["top10_bigramas"], 1):
        print(f"    {i:2}. {' '.join(bigrama):<35} {freq} ocorrências")

    print("\n  TOP 10 TRIGRAMAS:")
    for i, (trigrama, freq) in enumerate(resultado["top10_trigramas"], 1):
        print(f"    {i:2}. {' '.join(trigrama):<45} {freq} ocorrências")

    print(f"\n  REFERÊNCIAS EXTRAÍDAS ({len(resultado['referencias'])}):")
    for i, ref in enumerate(resultado["referencias"][:5], 1):
        print(f"    [{i}] {ref[:100]}{'...' if len(ref) > 100 else ''}")
    if len(resultado["referencias"]) > 5:
        print(f"    ... e mais {len(resultado['referencias']) - 5} referência(s)")


# ── Função principal da etapa ──────────────────────────────────────────────────

def executar_etapa1(diretorio_artigos: str) -> list[dict]:
    """
    Ponto de entrada da Etapa 1.
    Recebe o caminho do diretório com os PDFs.
    Retorna lista de dicionários com os resultados de cada artigo.
    """
    baixar_recursos_nltk()

    print("\n" + "="*60)
    print("  ETAPA 1: Leitura, Pré-processamento e Modelos de Linguagem")
    print("="*60)

    # Lê todos os PDFs
    artigos_texto = ler_pdfs_do_diretorio(diretorio_artigos)

    if not artigos_texto:
        print("[ERRO] Nenhum artigo para processar.")
        return []

    # Processa cada artigo
    resultados = []
    for nome, texto in artigos_texto.items():
        resultado = processar_artigo(nome, texto)
        resultados.append(resultado)
        exibir_resultado(resultado)

    # Top 10 global (corpus inteiro)
    print("\n" + "="*60)
    print("  TOP 10 TERMOS — CORPUS COMPLETO (todos os artigos)")
    print("="*60)
    top_global = top_termos_globais(resultados, 10)
    for i, (termo, freq) in enumerate(top_global, 1):
        print(f"  {i:2}. {termo:<25} {freq} ocorrências")

    return resultados


if __name__ == "__main__":
    # Altere o caminho abaixo para o diretório com seus PDFs
    DIRETORIO = "./artigos"
    executar_etapa1(DIRETORIO)