"""
Etapa 3 — Ontologia de Artigo Científico em JSON-LD
Modela cada artigo como instância da classe ArtigoCientifico e serializa
em JSON-LD com @context customizado.
Gera: um arquivo por artigo + um arquivo consolidado com todos.
"""

import json
import os
import re
from datetime import datetime


# ── @context da ontologia ──────────────────────────────────────────────────────

CONTEXT = {
    "@vocab": "http://example.org/ontologia-artigo#",
    "schema": "http://schema.org/",
    "titulo": "schema:name",
    "autores": "schema:author",
    "ano": "schema:datePublished",
    "fonte": "schema:isPartOf",
    "Artigo": "schema:ScholarlyArticle",
    "Objetivo": "http://example.org/ontologia-artigo#Objective",
    "Problema": "http://example.org/ontologia-artigo#ResearchProblem",
    "Metodologia": "http://example.org/ontologia-artigo#Methodology",
    "Contribuicao": "http://example.org/ontologia-artigo#Contribution",
    "Referencia": "http://example.org/ontologia-artigo#BibliographicReference",
    "Termo": "http://example.org/ontologia-artigo#Term",
    "temObjetivo": {
        "@id": "http://example.org/ontologia-artigo#temObjetivo",
        "@type": "@id",
        "@container": "@set",
    },
    "temProblema": {
        "@id": "http://example.org/ontologia-artigo#temProblema",
        "@type": "@id",
        "@container": "@set",
    },
    "usaMetodologia": {
        "@id": "http://example.org/ontologia-artigo#usaMetodologia",
        "@type": "@id",
        "@container": "@set",
    },
    "temContribuicao": {
        "@id": "http://example.org/ontologia-artigo#temContribuicao",
        "@type": "@id",
        "@container": "@set",
    },
    "citaReferencia": {
        "@id": "http://example.org/ontologia-artigo#citaReferencia",
        "@type": "@id",
        "@container": "@set",
    },
    "mencionaTermo": {
        "@id": "http://example.org/ontologia-artigo#mencionaTermo",
        "@type": "@id",
        "@container": "@set",
    },
    "textoExtraido": "http://example.org/ontologia-artigo#textoExtraido",
    "frequencia": "http://example.org/ontologia-artigo#frequencia",
    "nomeTermo": "http://example.org/ontologia-artigo#nomeTermo",
    "confianca": "http://example.org/ontologia-artigo#confianca",
    "totalTokens": "http://example.org/ontologia-artigo#totalTokens",
    "totalReferencias": "http://example.org/ontologia-artigo#totalReferencias",
    "dataProcessamento": "http://example.org/ontologia-artigo#dataProcessamento",
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _slugify(texto: str) -> str:
    """Converte string para slug seguro para usar em @id."""
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = texto.strip("_")
    return texto[:50]


def _extrair_ano_do_nome(nome_arquivo: str) -> int | None:
    """Tenta inferir o ano de publicação a partir do nome do arquivo PDF."""
    match = re.search(r"\b(19|20)\d{2}\b", nome_arquivo)
    return int(match.group()) if match else None


def _extrair_titulo_do_nome(nome_arquivo: str) -> str:
    """Usa o nome do arquivo (sem extensão) como título aproximado."""
    titulo = os.path.splitext(nome_arquivo)[0]
    titulo = re.sub(r"[_\-]+", " ", titulo)
    return titulo.strip().title()


def _confianca_campo(trechos: list[str]) -> str:
    """Estima confiança da extração com base na quantidade de trechos."""
    if not trechos:
        return "baixa"
    if len(trechos) == 1:
        return "media"
    return "alta"


# ── Modelagem do artigo como JSON-LD ─────────────────────────────────────────

def modelar_artigo_jsonld(resultado: dict) -> dict:
    """
    Recebe o dicionário consolidado (Etapas 1 + 2) de um artigo e
    retorna um grafo JSON-LD completo representando esse artigo.
    """
    nome = resultado.get("nome", "artigo_desconhecido")
    slug = _slugify(nome)
    extracao = resultado.get("extracao", {})
    top10 = resultado.get("top10_unigramas", [])
    referencias = resultado.get("referencias", [])
    tokens = resultado.get("tokens", [])

    # IDs base
    id_artigo = f"_:artigo_{slug}"
    grafo: list[dict] = []

    # ── Nós de Objetivo ──────────────────────────────────────────────────────
    ids_objetivo = []
    for i, trecho in enumerate(extracao.get("objetivo", []), 1):
        node_id = f"_:obj_{slug}_{i}"
        grafo.append({
            "@id": node_id,
            "@type": "Objetivo",
            "textoExtraido": trecho,
            "confianca": _confianca_campo(extracao.get("objetivo", [])),
        })
        ids_objetivo.append(node_id)

    # ── Nós de Problema ──────────────────────────────────────────────────────
    ids_problema = []
    for i, trecho in enumerate(extracao.get("problema", []), 1):
        node_id = f"_:prob_{slug}_{i}"
        grafo.append({
            "@id": node_id,
            "@type": "Problema",
            "textoExtraido": trecho,
            "confianca": _confianca_campo(extracao.get("problema", [])),
        })
        ids_problema.append(node_id)

    # ── Nós de Metodologia ───────────────────────────────────────────────────
    ids_metodo = []
    for i, trecho in enumerate(extracao.get("metodo", []), 1):
        node_id = f"_:met_{slug}_{i}"
        grafo.append({
            "@id": node_id,
            "@type": "Metodologia",
            "textoExtraido": trecho,
            "confianca": _confianca_campo(extracao.get("metodo", [])),
        })
        ids_metodo.append(node_id)

    # ── Nós de Contribuição ──────────────────────────────────────────────────
    ids_contribuicao = []
    for i, trecho in enumerate(extracao.get("contribuicao", []), 1):
        node_id = f"_:contrib_{slug}_{i}"
        grafo.append({
            "@id": node_id,
            "@type": "Contribuicao",
            "textoExtraido": trecho,
            "confianca": _confianca_campo(extracao.get("contribuicao", [])),
        })
        ids_contribuicao.append(node_id)

    # ── Nós de Referência ────────────────────────────────────────────────────
    ids_referencias = []
    for i, ref in enumerate(referencias, 1):
        node_id = f"_:ref_{slug}_{i}"
        grafo.append({
            "@id": node_id,
            "@type": "Referencia",
            "textoExtraido": ref,
        })
        ids_referencias.append(node_id)

    # ── Nós de Termos ────────────────────────────────────────────────────────
    ids_termos = []
    for termo, freq in top10:
        node_id = f"_:termo_{slug}_{_slugify(termo)}"
        grafo.append({
            "@id": node_id,
            "@type": "Termo",
            "nomeTermo": termo,
            "frequencia": freq,
        })
        ids_termos.append(node_id)

    # ── Nó principal do Artigo ───────────────────────────────────────────────
    no_artigo: dict = {
        "@id": id_artigo,
        "@type": "Artigo",
        "titulo": _extrair_titulo_do_nome(nome),
        "autores": [],
        "ano": _extrair_ano_do_nome(nome),
        "fonte": "IEEE Xplore",
        "totalTokens": len(tokens),
        "totalReferencias": len(referencias),
        "dataProcessamento": datetime.now().isoformat(timespec="seconds"),
    }

    if ids_objetivo:
        no_artigo["temObjetivo"] = ids_objetivo
    if ids_problema:
        no_artigo["temProblema"] = ids_problema
    if ids_metodo:
        no_artigo["usaMetodologia"] = ids_metodo
    if ids_contribuicao:
        no_artigo["temContribuicao"] = ids_contribuicao
    if ids_referencias:
        no_artigo["citaReferencia"] = ids_referencias
    if ids_termos:
        no_artigo["mencionaTermo"] = ids_termos

    grafo.insert(0, no_artigo)

    return {
        "@context": CONTEXT,
        "@graph": grafo,
    }


# ── Salvar arquivo individual ─────────────────────────────────────────────────

def salvar_jsonld_artigo(resultado: dict, diretorio_saida: str) -> str:
    """
    Serializa um artigo em JSON-LD e salva no diretório de saída.
    Retorna o caminho do arquivo gerado.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    nome = resultado.get("nome", "artigo")
    slug = _slugify(nome)
    caminho = os.path.join(diretorio_saida, f"{slug}.jsonld")

    jsonld = modelar_artigo_jsonld(resultado)

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(jsonld, f, ensure_ascii=False, indent=2)

    print(f"    [JSON-LD] Salvo: {caminho}")
    return caminho


# ── Salvar arquivo consolidado ────────────────────────────────────────────────

def salvar_jsonld_consolidado(resultados: list[dict], diretorio_saida: str) -> str:
    """
    Gera um único arquivo JSON-LD com todos os artigos no mesmo @graph.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho = os.path.join(diretorio_saida, "corpus_consolidado.jsonld")

    grafo_total = []
    for resultado in resultados:
        jsonld = modelar_artigo_jsonld(resultado)
        grafo_total.extend(jsonld["@graph"])

    consolidado = {
        "@context": CONTEXT,
        "@graph": grafo_total,
    }

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(consolidado, f, ensure_ascii=False, indent=2)

    print(f"\n  [JSON-LD] Corpus consolidado salvo: {caminho}")
    return caminho


# ── Ponto de entrada da etapa ─────────────────────────────────────────────────

def executar_etapa3(
    resultados: list[dict],
    diretorio_saida: str = "./saida/jsonld",
) -> list[str]:
    """
    Executa a Etapa 3 para todos os artigos.
    Retorna lista de caminhos dos arquivos gerados.
    """
    print("\n" + "="*60)
    print("  ETAPA 3: Serialização em Ontologia JSON-LD")
    print("="*60)

    arquivos = []
    for resultado in resultados:
        caminho = salvar_jsonld_artigo(resultado, diretorio_saida)
        arquivos.append(caminho)

    consolidado = salvar_jsonld_consolidado(resultados, diretorio_saida)
    arquivos.append(consolidado)

    print(f"\n  Total de arquivos gerados: {len(arquivos)}")
    return arquivos