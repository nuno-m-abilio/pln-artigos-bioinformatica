"""
Etapa 2 — Extração de Informações do Artigo Científico (ATUALIZADO)
Abordagem: regex avançado + heurísticas de contexto (sem ML)
Campos extraídos: Objetivo, Problema, Método, Contribuição
"""

import re
from nltk.tokenize import sent_tokenize

# ── Padrões de extração melhorados (Tolerância a advérbios e espaços) ───────────

_PADROES_OBJETIVO = [
    r"the\s+(main\s+|primary\s+)?objective(s)?\s+of\s+(this|the)\s+(paper|article|study|work|research)",
    r"this\s+(paper|article|study|work|research)\s+(?:\w+\s+){0,3}(aims?|intends?|seeks?|proposes?)",
    r"we\s+(?:\w+\s+){0,3}(propose|present|introduce|aim|seek|develop|describe|design)",
    r"the\s+(aim|purpose|goal)(s)?\s+of\s+(this|the)\s+(paper|article|study|work|research)",
    r"in\s+this\s+(paper|article|study|work|research)[,\s]+we\s+(?:\w+\s+){0,3}(propose|present|focus|aim)",
    r"this\s+(paper|article|study|work|research)\s+(?:\w+\s+){0,3}(presents?|describes?|proposes?|introduces?)",
    r"our\s+(main\s+)?(goal|aim|objective|purpose)\s+(is|was)",
    r"we\s+(?:\w+\s+){0,3}(investigate|examine|analyze|evaluate|explore|focus\s+on)",
]

_PADROES_PROBLEMA = [
    r"the\s+(main\s+|key\s+|fundamental\s+)?(problem|challenge|issue|limitation|difficulty|drawback)",
    r"this\s+(paper|article|study|work|research)\s+(?:\w+\s+){0,3}(addresses?|tackles?|overcomes?|solves?)",
    r"(existing|current|previous|traditional)\s+(method|approach|technique|model|system)s?\s+(?:\w+\s+){0,3}(fail|suffer|struggle|lack|limit|cannot)",
    r"(however|unfortunately|despite|although)[,\s].{0,100}?(problem|challenge|issue|limitation|drawback|lack)",
    r"a\s+(key|major|fundamental|critical|significant)\s+(problem|challenge|limitation|issue|drawback)",
    r"(little|limited|insufficient|lack\s+of)\s+(attention|research|work|study|investigation)",
    r"(it\s+is|remains?)\s+(unclear|unknown|difficult|challenging)",
    r"(motivat|inspir)(ed|es?|ing)\s+by",
]

_PADROES_METODO = [
    r"we\s+(?:\w+\s+){0,3}(use[d]?|employ[ed]?|apply|applied|adopt[ed]?|implement[ed]?|conduct[ed]?)",
    r"(the\s+)?(method|methodology|approach|technique|framework|model|system|algorithm)\s+(used|proposed|presented|applied|adopted)",
    r"(experiment|survey|interview|questionnaire|case\s+study|content\s+analysis)",
    r"(data|dataset)s?\s+(was|were|is|are)\s+(collect|gather|obtain|extract)(ed)?",
    r"(train|test|evaluat|validat)(ed|ing|ion)\s+using",
    r"(using|based\s+on|by\s+means\s+of)\s+(?:a\s+|the\s+)?(method|approach|model|framework)",
    r"(our|the)\s+(proposed|presented|described)\s+(method|approach|framework|model|system|algorithm)",
    r"methodology\s+consists\s+of",
]

_PADROES_CONTRIBUICAO = [
    r"(this\s+(paper|article|study|work|research)|we)\s+(?:\w+\s+){0,3}contributes?\s+to",
    r"our\s+(main\s+)?(contribution|contributions)",
    r"the\s+(main\s+)?(contribution|contributions)\s+(of\s+this|is|are)",
    r"(novel|new|first)\s+(approach|method|framework|model|algorithm|contribution)",
    r"(to\s+the\s+best\s+of\s+our\s+knowledge)",
    r"(demonstrate|show)\s+that\s+.{0,80}(better|superior|outperform|improve|advance)",
    r"(our|the)\s+(study|work|paper|article|approach)\s+(demonstrate|show|reveal|prove)s?",
]


def _compilar_padroes(lista: list[str]) -> re.Pattern:
    return re.compile("|".join(lista), flags=re.IGNORECASE)

_RE_OBJETIVO = _compilar_padroes(_PADROES_OBJETIVO)
_RE_PROBLEMA = _compilar_padroes(_PADROES_PROBLEMA)
_RE_METODO = _compilar_padroes(_PADROES_METODO)
_RE_CONTRIBUICAO = _compilar_padroes(_PADROES_CONTRIBUICAO)

# ── Janela de contexto ─────────────────────────────────────────────────────────

def _extrair_com_janela(
    sentencas: list[str],
    padrao: re.Pattern,
    janela: int = 2,
    max_resultados: int = 3,
) -> list[str]:
    trechos = []
    visto = set()

    for i, sent in enumerate(sentencas):
        if padrao.search(sent):
            fim = min(i + janela + 1, len(sentencas))
            trecho = " ".join(sentencas[i:fim]).strip()
            chave = sent[:60]
            if chave not in visto:
                visto.add(chave)
                trechos.append(trecho)
            if len(trechos) >= max_resultados:
                break

    return trechos

# ── Heurísticas de seção ───────────────────────────────────────────────────────

# Melhorado para capturar numerações antes do título, ex: "1. Introduction" ou "I. Background"
_RE_INTRO = re.compile(
    r"^(?:\d{1,2}\.?|[IVX]+\.?)?\s*(introduction|background|motivation|overview)",
    flags=re.IGNORECASE,
)

_RE_METODO_SEC = re.compile(
    r"^(?:\d{1,2}\.?|[IVX]+\.?)?\s*(method|methodology|approach|materials?\s+and\s+methods?|experiment|framework|system|proposed)",
    flags=re.IGNORECASE,
)

_RE_CONCLUSAO_SEC = re.compile(
    r"^(?:\d{1,2}\.?|[IVX]+\.?)?\s*(conclusion|discussion|future\s+work|summary)",
    flags=re.IGNORECASE,
)


def _dividir_em_secoes(texto: str) -> dict[str, str]:
    linhas = texto.splitlines()
    secoes: dict[str, list[str]] = {"inicio": []}
    secao_atual = "inicio"

    for linha in linhas:
        stripped = linha.strip()
        if 2 < len(stripped) < 80 and re.match(r"^[IVX\d\.\s]*[A-Z]", stripped):
            if _RE_INTRO.search(stripped):
                secao_atual = "introducao"
                secoes.setdefault(secao_atual, [])
            elif _RE_METODO_SEC.search(stripped):
                secao_atual = "metodologia"
                secoes.setdefault(secao_atual, [])
            elif _RE_CONCLUSAO_SEC.search(stripped):
                secao_atual = "conclusao"
                secoes.setdefault(secao_atual, [])
            else:
                secoes.setdefault(secao_atual, []).append(linha)
        else:
            secoes.setdefault(secao_atual, []).append(linha)

    return {k: "\n".join(v) for k, v in secoes.items()}


# ── Extração principal ─────────────────────────────────────────────────────────

def extrair_informacoes(corpo: str) -> dict:
    secoes = _dividir_em_secoes(corpo)

    # Prioridade de seção para cada campo
    texto_objetivo_prob = secoes.get("introducao", corpo[:5000])
    texto_metodo = secoes.get("metodologia", corpo)
    texto_contribuicao = secoes.get("conclusao", "") + " " + secoes.get("introducao", "")

    # LIMPEZA CRÍTICA: Remover espaços duplos e quebras de linha antes de tokenizar
    texto_objetivo_prob = re.sub(r'\s+', ' ', texto_objetivo_prob)
    texto_metodo = re.sub(r'\s+', ' ', texto_metodo)
    texto_contribuicao = re.sub(r'\s+', ' ', texto_contribuicao)
    corpo_limpo = re.sub(r'\s+', ' ', corpo)

    # Tokenizar em sentenças
    sents_intro = sent_tokenize(texto_objetivo_prob)
    sents_corpo = sent_tokenize(corpo_limpo)
    sents_metodo = sent_tokenize(texto_metodo)
    sents_contribuicao = sent_tokenize(texto_contribuicao) if texto_contribuicao.strip() else sents_corpo

    # Extrações
    objetivo = _extrair_com_janela(sents_intro, _RE_OBJETIVO, janela=2, max_resultados=3)
    if not objetivo:
        objetivo = _extrair_com_janela(sents_corpo[:60], _RE_OBJETIVO, janela=2, max_resultados=2)

    problema = _extrair_com_janela(sents_intro, _RE_PROBLEMA, janela=2, max_resultados=3)
    if not problema:
        problema = _extrair_com_janela(sents_corpo[:60], _RE_PROBLEMA, janela=2, max_resultados=2)

    metodo = _extrair_com_janela(sents_metodo, _RE_METODO, janela=2, max_resultados=3)
    if not metodo:
        metodo = _extrair_com_janela(sents_corpo, _RE_METODO, janela=2, max_resultados=3)

    contribuicao = _extrair_com_janela(sents_contribuicao, _RE_CONTRIBUICAO, janela=2, max_resultados=3)
    if not contribuicao:
        contribuicao = _extrair_com_janela(sents_corpo, _RE_CONTRIBUICAO, janela=2, max_resultados=3)

    return {
        "objetivo": objetivo,
        "problema": problema,
        "metodo": metodo,
        "contribuicao": contribuicao,
    }

# ── Exibição ───────────────────────────────────────────────────────────────────

def exibir_extracao(nome: str, extracao: dict):
    print(f"\n{'='*60}")
    print(f"  ETAPA 2 — EXTRAÇÃO: {nome}")
    print(f"{'='*60}")

    campos = [
        ("OBJETIVO", "objetivo"),
        ("PROBLEMA", "problema"),
        ("MÉTODO", "metodo"),
        ("CONTRIBUIÇÃO", "contribuicao"),
    ]

    for label, chave in campos:
        print(f"\n  {label}:")
        trechos = extracao.get(chave, [])
        if trechos:
            for i, t in enumerate(trechos, 1):
                print(f"    [{i}] {t[:300]}{'...' if len(t) > 300 else ''}")
        else:
            print("    [não encontrado]")

# ── Ponto de entrada da etapa ──────────────────────────────────────────────────

def executar_etapa2(resultados_etapa1: list[dict]) -> list[dict]:
    print("\n" + "="*60)
    print("  ETAPA 2: Extração de Informações (Objetivo, Problema, Método, Contribuição)")
    print("="*60)

    for resultado in resultados_etapa1:
        nome = resultado["nome"]
        corpo = resultado.get("corpo", "")
        extracao = extrair_informacoes(corpo)
        resultado["extracao"] = extracao
        exibir_extracao(nome, extracao)

    return resultados_etapa1