import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # backend sem GUI para ambientes sem display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from wordcloud import WordCloud
    _WORDCLOUD_OK = True
except ImportError:
    _WORDCLOUD_OK = False
    print("[AVISO] wordcloud não instalada. Nuvens de palavras serão puladas.")


# Regex de trabalhos futuros ─────────────────────────────────────────────────

_RE_FUTURO = re.compile(
    r"(future (work|research|study|studies|direction|plan)|"
    r"in the future|as future|future investigation|"
    r"we (plan|intend|will) to|can be (extended|explored|improved)|"
    r"remains? (as an? )?(open|future|interesting)|"
    r"should be (investigated|explored|studied) in the future)",
    flags=re.IGNORECASE,
)

_RE_ANO = re.compile(r"\b(19|20)(\d{2})\b")


# Utilitários ────────────────────────────────────────────────────────────────

def _garantir_diretorio(caminho: str):
    os.makedirs(caminho, exist_ok=True)


def _salvar_figura(fig, caminho: str):
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [VIZ] Salvo: {caminho}")


def _inferir_ano(nome_arquivo: str) -> int | None:
    m = _RE_ANO.search(nome_arquivo)
    return int(m.group()) if m else None


# 1. Nuvem de palavras ───────────────────────────────────────────────────────

def nuvem_de_palavras(
    resultados: list[dict],
    diretorio_saida: str,
    por_artigo: bool = False,
):
    """Gera nuvem de palavras geral e, opcionalmente, por artigo."""
    _garantir_diretorio(diretorio_saida)

    if not _WORDCLOUD_OK:
        return

    # Geral: agrega todos os tokens
    contador_global = Counter()
    for r in resultados:
        contador_global.update(r.get("bow", {}))

    if contador_global:
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            max_words=100,
            colormap="viridis",
        ).generate_from_frequencies(contador_global)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Nuvem de Palavras — Corpus Completo", fontsize=16, pad=12)
        _salvar_figura(fig, os.path.join(diretorio_saida, "nuvem_geral.png"))

    # Por artigo
    if por_artigo:
        for r in resultados:
            bow = r.get("bow", {})
            if not bow:
                continue
            wc = WordCloud(
                width=800, height=400,
                background_color="white",
                max_words=60,
                colormap="plasma",
            ).generate_from_frequencies(bow)

            slug = re.sub(r"[^a-z0-9]", "_", r["nome"].lower())[:40]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Nuvem de Palavras — {r['nome']}", fontsize=12, pad=8)
            _salvar_figura(fig, os.path.join(diretorio_saida, f"nuvem_{slug}.png"))


# 2. Barras de frequência dos top termos ────────────────────────────────────

def barras_top_termos(resultados: list[dict], diretorio_saida: str, top_n: int = 20):
    """Gráfico de barras horizontal dos N termos mais frequentes no corpus."""
    _garantir_diretorio(diretorio_saida)

    contador = Counter()
    for r in resultados:
        contador.update(r.get("bow", {}))

    if not contador:
        return

    itens = contador.most_common(top_n)
    termos = [t for t, _ in reversed(itens)]
    freqs = [f for _, f in reversed(itens)]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(termos, freqs, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(termos))))
    ax.set_xlabel("Frequência (corpus completo)")
    ax.set_title(f"Top {top_n} Termos Mais Frequentes", fontsize=13)
    ax.bar_label(bars, padding=3, fontsize=8)
    plt.tight_layout()
    _salvar_figura(fig, os.path.join(diretorio_saida, "barras_top_termos.png"))


# 3. Heatmap de coocorrência de termos entre artigos ────────────────────────

def heatmap_coocorrencia(resultados: list[dict], diretorio_saida: str, top_n: int = 15):
    """
    Heatmap N×M onde linhas = artigos e colunas = top termos do corpus.
    Valor = frequência do termo naquele artigo (normalizada por artigo).
    """
    _garantir_diretorio(diretorio_saida)

    # Termos globais mais frequentes
    contador = Counter()
    for r in resultados:
        contador.update(r.get("bow", {}))

    top_termos = [t for t, _ in contador.most_common(top_n)]
    if not top_termos or not resultados:
        return

    nomes_artigos = [r["nome"][:30] for r in resultados]
    matriz = np.zeros((len(resultados), len(top_termos)))

    for i, r in enumerate(resultados):
        bow = r.get("bow", {})
        total = sum(bow.values()) or 1
        for j, termo in enumerate(top_termos):
            matriz[i, j] = bow.get(termo, 0) / total * 1000  # por mil

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.7), max(6, len(resultados) * 0.6)))
    im = ax.imshow(matriz, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top_termos)))
    ax.set_xticklabels(top_termos, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(resultados)))
    ax.set_yticklabels(nomes_artigos, fontsize=8)
    ax.set_title(f"Heatmap de Coocorrência — Top {top_n} Termos por Artigo", fontsize=12)
    plt.colorbar(im, ax=ax, label="Frequência por mil tokens")
    plt.tight_layout()
    _salvar_figura(fig, os.path.join(diretorio_saida, "heatmap_coocorrencia.png"))


# 4. Evolução temporal dos termos ───────────────────────────────────────────

def grafico_temporal(resultados: list[dict], diretorio_saida: str, top_n: int = 8):
    """
    Agrupa artigos por ano e mostra frequência relativa dos top termos ao longo do tempo.
    Requer que o nome do arquivo contenha um ano (ex: 2021).
    """
    _garantir_diretorio(diretorio_saida)

    por_ano: dict[int, Counter] = {}
    for r in resultados:
        ano = _inferir_ano(r.get("nome", ""))
        if ano is None:
            continue
        por_ano.setdefault(ano, Counter()).update(r.get("bow", {}))

    if len(por_ano) < 2:
        print("    [VIZ] Dados insuficientes para gráfico temporal (< 2 anos distintos nos nomes dos arquivos).")
        return

    # Top termos globais para traçar
    contador_global = Counter()
    for c in por_ano.values():
        contador_global.update(c)
    top_termos = [t for t, _ in contador_global.most_common(top_n)]

    anos = sorted(por_ano.keys())
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.get_cmap("tab10", len(top_termos))

    for idx, termo in enumerate(top_termos):
        freqs = []
        for ano in anos:
            total = sum(por_ano[ano].values()) or 1
            freqs.append(por_ano[ano].get(termo, 0) / total * 1000)
        ax.plot(anos, freqs, marker="o", label=termo, color=cmap(idx), linewidth=1.8)

    ax.set_xlabel("Ano de Publicação")
    ax.set_ylabel("Frequência por mil tokens")
    ax.set_title(f"Evolução Temporal dos Top {top_n} Termos", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xticks(anos)
    plt.tight_layout()
    _salvar_figura(fig, os.path.join(diretorio_saida, "temporal_termos.png"))


# 5. Termos de trabalhos futuros ────────────────────────────────────────────

def extrair_trabalhos_futuros(resultados: list[dict]) -> dict[str, list[str]]:
    """
    Extrai sentenças sobre trabalhos futuros da seção de conclusão de cada artigo.
    Retorna dict {nome_artigo: [sentenças]}.
    """
    from nltk.tokenize import sent_tokenize

    futuros = {}
    for r in resultados:
        corpo = r.get("corpo", "")
        # Busca nas últimas 20% do corpo (normalmente onde fica a conclusão)
        trecho_final = corpo[int(len(corpo) * 0.7):]
        sentencas = sent_tokenize(trecho_final)
        encontradas = [s.strip() for s in sentencas if _RE_FUTURO.search(s)]
        if encontradas:
            futuros[r["nome"]] = encontradas

    return futuros


def grafico_termos_futuros(resultados: list[dict], diretorio_saida: str):
    """Nuvem ou barras de frequência dos termos nas sentenças de trabalhos futuros."""
    _garantir_diretorio(diretorio_saida)

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop = set(stopwords.words("english"))
    futuros = extrair_trabalhos_futuros(resultados)

    todos_tokens = []
    for sentencas in futuros.values():
        for s in sentencas:
            tokens = [t.lower() for t in word_tokenize(s) if t.isalpha() and len(t) > 3]
            todos_tokens.extend([t for t in tokens if t not in stop])

    if not todos_tokens:
        print("    [VIZ] Nenhum trecho de trabalhos futuros encontrado.")
        return

    contador = Counter(todos_tokens)
    top = contador.most_common(20)
    termos = [t for t, _ in reversed(top)]
    freqs = [f for _, f in reversed(top)]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(termos, freqs, color="#4CAF50")
    ax.set_xlabel("Frequência")
    ax.set_title("Termos Mais Frequentes em 'Trabalhos Futuros'", fontsize=13)
    plt.tight_layout()
    _salvar_figura(fig, os.path.join(diretorio_saida, "termos_trabalhos_futuros.png"))

    # Nuvem de futuros
    if _WORDCLOUD_OK and contador:
        wc = WordCloud(
            width=900, height=450,
            background_color="white",
            colormap="Greens",
            max_words=60,
        ).generate_from_frequencies(contador)
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        ax2.set_title("Nuvem — Termos de Trabalhos Futuros", fontsize=13)
        _salvar_figura(fig2, os.path.join(diretorio_saida, "nuvem_futuros.png"))

    # Exibe quais artigos tiveram trechos encontrados
    print(f"\n  Artigos com trechos de trabalhos futuros detectados: {len(futuros)}")
    for nome, sents in futuros.items():
        print(f"    {nome}: {len(sents)} sentença(s)")

    return futuros


# Ponto de entrada ───────────────────────────────────────────────────────────

def executar_visualizacoes(resultados: list[dict], diretorio_saida: str = "./saida/visualizacoes"):
    """Executa todas as visualizações."""
    print("\n" + "="*60)
    print("  VISUALIZAÇÕES")
    print("="*60)

    nuvem_de_palavras(resultados, diretorio_saida, por_artigo=True)
    barras_top_termos(resultados, diretorio_saida, top_n=20)
    heatmap_coocorrencia(resultados, diretorio_saida, top_n=15)
    grafico_temporal(resultados, diretorio_saida, top_n=8)
    grafico_termos_futuros(resultados, diretorio_saida)

    print(f"\n  [OK] Visualizações salvas em '{diretorio_saida}'")