"""
Etapa 4 — Avaliação de Desempenho do Sistema de Extração
Responsável: Janaina

Método de avaliação: sobreposição de palavras (word overlap)
Métricas: Precisão, Recall e F1 por campo (objetivo, problema, metodo, contribuicao)

Como funciona:
  - O gabarito (anotação manual) é um arquivo JSON com os campos corretos de 4 artigos
  - O sistema extrai os campos automaticamente (Etapa 2)
  - Compara-se as palavras do trecho extraído com as do gabarito (ignorando stopwords)
  - Calcula Precisão, Recall e F1 com base nas palavras em comum

Formato do gabarito (gabarito.json):
{
  "nome_do_arquivo.pdf": {
    "objetivo": "texto do objetivo conforme o artigo",
    "problema": "texto do problema conforme o artigo",
    "metodo": "texto do método conforme o artigo",
    "contribuicao": "texto da contribuição conforme o artigo"
  },
  ...
}
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ── Configuração ───────────────────────────────────────────────────────────────

CAMINHO_GABARITO = "./gabarito.json"
CAMPOS = ["objetivo", "problema", "metodo", "contribuicao"]

_STOP_WORDS = set(stopwords.words("english"))
_STOP_WORDS_EXTRAS = {
    "et", "al", "fig", "figure", "table", "also", "show", "shown",
    "use", "used", "using", "based", "however", "thus", "therefore",
    "paper", "study", "result", "results", "proposed", "method",
    "approach", "work", "article", "section",
}
_STOP_WORDS.update(_STOP_WORDS_EXTRAS)


# ── Utilitários ────────────────────────────────────────────────────────────────

def tokenizar_para_avaliacao(texto: str) -> set[str]:
    """
    Tokeniza e limpa um texto para comparação.
    Retorna conjunto de palavras sem stopwords e com 3+ chars.
    """
    if not texto:
        return set()
    tokens = word_tokenize(texto.lower())
    return {t for t in tokens if t.isalpha() and len(t) >= 3 and t not in _STOP_WORDS}


def calcular_metricas(
    tokens_extraidos: set[str],
    tokens_gabarito: set[str],
) -> dict[str, float]:
    """
    Calcula Precisão, Recall e F1 por sobreposição de palavras.
    """
    if not tokens_extraidos and not tokens_gabarito:
        return {"precisao": 1.0, "recall": 1.0, "f1": 1.0}
    if not tokens_extraidos:
        return {"precisao": 0.0, "recall": 0.0, "f1": 0.0}
    if not tokens_gabarito:
        return {"precisao": 0.0, "recall": 0.0, "f1": 0.0}

    intersecao = tokens_extraidos & tokens_gabarito
    precisao = len(intersecao) / len(tokens_extraidos)
    recall = len(intersecao) / len(tokens_gabarito)

    if precisao + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precisao * recall / (precisao + recall)

    return {"precisao": round(precisao, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def juntar_trechos(trechos: list[str]) -> str:
    """Une lista de trechos extraídos em um único texto para comparação."""
    return " ".join(trechos)


# ── Avaliação principal ────────────────────────────────────────────────────────

def avaliar_artigo(
    nome: str,
    extracao: dict,
    gabarito_artigo: dict,
) -> dict[str, dict[str, float]]:
    """
    Avalia a extração de um artigo contra o gabarito.
    Retorna dict {campo: {precisao, recall, f1}}.
    """
    resultados = {}
    for campo in CAMPOS:
        trechos_extraidos = extracao.get(campo, [])
        texto_extraido = juntar_trechos(trechos_extraidos)
        texto_gabarito = gabarito_artigo.get(campo, "")

        tokens_ext = tokenizar_para_avaliacao(texto_extraido)
        tokens_gab = tokenizar_para_avaliacao(texto_gabarito)

        metricas = calcular_metricas(tokens_ext, tokens_gab)
        resultados[campo] = metricas

    return resultados


def avaliar_sistema(resultados_etapa2: list[dict], gabarito: dict) -> dict:
    """
    Avalia o sistema completo comparando os resultados da Etapa 2
    com o gabarito manual.

    Retorna:
      {
        "por_artigo": {nome: {campo: {precisao, recall, f1}}},
        "media_por_campo": {campo: {precisao, recall, f1}},
        "media_global": {precisao, recall, f1},
      }
    """
    por_artigo = {}
    acumulado: dict[str, list] = defaultdict(lambda: defaultdict(list))

    for resultado in resultados_etapa2:
        nome = resultado["nome"]
        if nome not in gabarito:
            print(f"  [AVISO] '{nome}' não encontrado no gabarito. Pulando.")
            continue

        extracao = resultado.get("extracao", {})
        gabarito_artigo = gabarito[nome]
        metricas_artigo = avaliar_artigo(nome, extracao, gabarito_artigo)
        por_artigo[nome] = metricas_artigo

        for campo, m in metricas_artigo.items():
            for metrica, valor in m.items():
                acumulado[campo][metrica].append(valor)

    # Média por campo
    media_por_campo = {}
    for campo in CAMPOS:
        if campo in acumulado:
            media_por_campo[campo] = {
                "precisao": round(sum(acumulado[campo]["precisao"]) / len(acumulado[campo]["precisao"]), 4),
                "recall": round(sum(acumulado[campo]["recall"]) / len(acumulado[campo]["recall"]), 4),
                "f1": round(sum(acumulado[campo]["f1"]) / len(acumulado[campo]["f1"]), 4),
            }
        else:
            media_por_campo[campo] = {"precisao": 0.0, "recall": 0.0, "f1": 0.0}

    # Média global
    todas_precisoes = [v for c in acumulado.values() for v in c["precisao"]]
    todos_recalls = [v for c in acumulado.values() for v in c["recall"]]
    todos_f1s = [v for c in acumulado.values() for v in c["f1"]]

    media_global = {
        "precisao": round(sum(todas_precisoes) / len(todas_precisoes), 4) if todas_precisoes else 0.0,
        "recall": round(sum(todos_recalls) / len(todos_recalls), 4) if todos_recalls else 0.0,
        "f1": round(sum(todos_f1s) / len(todos_f1s), 4) if todos_f1s else 0.0,
    }

    return {
        "por_artigo": por_artigo,
        "media_por_campo": media_por_campo,
        "media_global": media_global,
    }


# ── Visualizações de avaliação ─────────────────────────────────────────────────

def plotar_metricas_por_campo(avaliacao: dict, diretorio_saida: str):
    """Gráfico de barras agrupadas: P, R, F1 por campo."""
    os.makedirs(diretorio_saida, exist_ok=True)

    media = avaliacao["media_por_campo"]
    campos = list(media.keys())
    precisoes = [media[c]["precisao"] for c in campos]
    recalls = [media[c]["recall"] for c in campos]
    f1s = [media[c]["f1"] for c in campos]

    x = np.arange(len(campos))
    largura = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - largura, precisoes, largura, label="Precisão", color="#2196F3")
    ax.bar(x, recalls, largura, label="Recall", color="#4CAF50")
    ax.bar(x + largura, f1s, largura, label="F1", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("Avaliação de Desempenho por Campo Extraído")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in campos])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adiciona valores nas barras
    for bar in ax.containers:
        ax.bar_label(bar, fmt="%.2f", padding=2, fontsize=8)

    plt.tight_layout()
    caminho = os.path.join(diretorio_saida, "avaliacao_por_campo.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [AVALIAÇÃO] Gráfico salvo: {caminho}")


def plotar_f1_por_artigo(avaliacao: dict, diretorio_saida: str):
    """Heatmap de F1 por artigo × campo."""
    os.makedirs(diretorio_saida, exist_ok=True)

    por_artigo = avaliacao["por_artigo"]
    if not por_artigo:
        return

    artigos = list(por_artigo.keys())
    campos = CAMPOS

    matriz = np.array([
        [por_artigo[a].get(c, {}).get("f1", 0.0) for c in campos]
        for a in artigos
    ])

    fig, ax = plt.subplots(figsize=(8, max(4, len(artigos) * 0.7)))
    im = ax.imshow(matriz, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(campos)))
    ax.set_xticklabels([c.capitalize() for c in campos])
    ax.set_yticks(range(len(artigos)))
    ax.set_yticklabels([a[:35] for a in artigos], fontsize=8)
    ax.set_title("F1-Score por Artigo e Campo", fontsize=12)

    # Adiciona valores nas células
    for i in range(len(artigos)):
        for j in range(len(campos)):
            ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="F1-Score")
    plt.tight_layout()
    caminho = os.path.join(diretorio_saida, "f1_por_artigo.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [AVALIAÇÃO] Heatmap salvo: {caminho}")


# ── Exibição no terminal ───────────────────────────────────────────────────────

def exibir_avaliacao(avaliacao: dict):
    print("\n" + "="*60)
    print("  ETAPA 4 — AVALIAÇÃO DE DESEMPENHO")
    print("="*60)

    print("\n  Resultados por artigo:")
    for nome, campos in avaliacao["por_artigo"].items():
        print(f"\n  {nome}")
        print(f"  {'Campo':<15} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*47}")
        for campo, m in campos.items():
            print(f"  {campo.capitalize():<15} {m['precisao']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print("\n  Média por campo (todos os artigos avaliados):")
    print(f"  {'Campo':<15} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*47}")
    for campo, m in avaliacao["media_por_campo"].items():
        print(f"  {campo.capitalize():<15} {m['precisao']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    g = avaliacao["media_global"]
    print(f"\n  Média Global → Precisão: {g['precisao']:.4f} | Recall: {g['recall']:.4f} | F1: {g['f1']:.4f}")


# ── Ponto de entrada da etapa ──────────────────────────────────────────────────

def executar_etapa4(
    resultados_etapa2: list[dict],
    caminho_gabarito: str = CAMINHO_GABARITO,
    diretorio_saida: str = "./saida/avaliacao",
) -> dict:
    """
    Executa a Etapa 4 — Avaliação de Desempenho.
    Requer o arquivo gabarito.json preenchido manualmente.
    """
    print("\n" + "="*60)
    print("  ETAPA 4: Avaliação de Desempenho")
    print("="*60)

    if not os.path.exists(caminho_gabarito):
        print(f"  [AVISO] Gabarito não encontrado em '{caminho_gabarito}'.")
        print("  Crie o arquivo gabarito.json com as anotações manuais dos 4 artigos.")
        print("  Formato esperado:")
        print('  { "nome_arquivo.pdf": { "objetivo": "...", "problema": "...", "metodo": "...", "contribuicao": "..." } }')
        return {}

    with open(caminho_gabarito, encoding="utf-8") as f:
        gabarito = json.load(f)

    print(f"  Gabarito carregado: {len(gabarito)} artigo(s) anotado(s).")

    avaliacao = avaliar_sistema(resultados_etapa2, gabarito)
    exibir_avaliacao(avaliacao)
    plotar_metricas_por_campo(avaliacao, diretorio_saida)
    plotar_f1_por_artigo(avaliacao, diretorio_saida)

    # Salva os resultados em JSON
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_resultado = os.path.join(diretorio_saida, "resultados_avaliacao.json")
    with open(caminho_resultado, "w", encoding="utf-8") as f:
        json.dump(avaliacao, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] Resultados salvos em '{caminho_resultado}'")

    return avaliacao