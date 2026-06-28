import re
from nltk.tokenize import sent_tokenize


# Padrões de extração ────────────────────────────────────────────────────────

_PADROES_OBJETIVO = [
    r"the (main\s+)?objective(s)? of (this|the) (paper|article|study|work|research)",
    r"this (paper|article|study|work|research) (aims?|intends?|seeks?|proposes?)",
    r"we (propose|present|introduce|aim|seek|develop|describe|design)",
    r"the (aim|purpose|goal)(s)? of (this|the) (paper|article|study|work|research)",
    r"in this (paper|article|study|work|research),?\s+we",
    r"this (paper|article|study|work|research) (presents?|describes?|proposes?|introduces?)",
    r"our (goal|aim|objective|purpose) (is|was)",
    r"we (investigate|examine|analyze|evaluate|explore|focus on)",
]

_PADROES_PROBLEMA = [
    r"the (main\s+)?(problem|challenge|issue|limitation|difficulty|drawback)",
    r"this (paper|article|study|work|research) (addresses?|tackles?|overcomes?|solves?)",
    r"(existing|current|previous|traditional) (method|approach|technique|model|system)s? (fail|suffer|struggle|lack|limit|cannot)",
    r"(however|unfortunately|despite|although)[,\s].{0,80}(problem|challenge|issue|limitation)",
    r"a (key|major|fundamental|critical|significant) (problem|challenge|limitation|issue|drawback)",
    r"(little|limited|insufficient|lack of) (attention|research|work|study|investigation)",
    r"(it is|remains?) (unclear|unknown|difficult|challenging)",
    r"(motivat|inspir)(ed|es?|ing) by",
]

_PADROES_METODO = [
    r"we (use[d]?|employ[ed]?|apply|applied|adopt[ed]?|implement[ed]?|conduct[ed]?)",
    r"(the\s+)?(method|methodology|approach|technique|framework|model|system|algorithm)",
    r"(experiment|survey|interview|questionnaire|case study|content analysis|dataset)",
    r"(data|dataset)s? (was|were|is|are) (collect|gather|obtain|extract)(ed)?",
    r"(train|test|evaluat|validat)(ed|ing|ion)",
    r"(using|based on|by means of)",
    r"(our|the) (proposed|presented|described) (method|approach|framework|model|system|algorithm)",
    r"(cross[- ]?validation|k[- ]?fold|benchmark)",
]

_PADROES_CONTRIBUICAO = [
    r"(this (paper|article|study|work|research)|we) contributes? to",
    r"our (main\s+)?(contribution|contributions)",
    r"the (main\s+)?(contribution|contributions) (of this|is|are)",
    r"(novel|new|first) (approach|method|framework|model|algorithm|contribution)",
    r"(to the best of our knowledge)",
    r"(demonstrate|show) that .{0,60}(better|superior|outperform|improve|advance)",
    r"(our|the) (study|work|paper|article|approach) (demonstrate|show|reveal|prove)s?",
]


def _compilar_padroes(lista: list[str]) -> re.Pattern:
    return re.compile("|".join(lista), flags=re.IGNORECASE)


_RE_OBJETIVO = _compilar_padroes(_PADROES_OBJETIVO)
_RE_PROBLEMA = _compilar_padroes(_PADROES_PROBLEMA)
_RE_METODO = _compilar_padroes(_PADROES_METODO)
_RE_CONTRIBUICAO = _compilar_padroes(_PADROES_CONTRIBUICAO)


# Janela de contexto ─────────────────────────────────────────────────────────

def _extrair_com_janela(
    sentencas: list[str],
    padrao: re.Pattern,
    janela: int = 2,
    max_resultados: int = 3,
) -> list[str]:
    """
    Percorre a lista de sentenças buscando matches do padrão.
    Para cada match, retorna a sentença + as próximas `janela` sentenças.
    Limita a `max_resultados` trechos para não poluir a saída.
    """
    trechos = []
    visto = set()

    for i, sent in enumerate(sentencas):
        if padrao.search(sent):
            fim = min(i + janela + 1, len(sentencas))
            trecho = " ".join(sentencas[i:fim]).strip()
            # Evita duplicatas muito próximas (mesma frase inicial)
            chave = sent[:60]
            if chave not in visto:
                visto.add(chave)
                trechos.append(trecho)
            if len(trechos) >= max_resultados:
                break

    return trechos


# Heurísticas de seção ───────────────────────────────────────────────────────

_RE_INTRO = re.compile(
    r"(introduction|background|motivation|overview)",
    flags=re.IGNORECASE,
)

_RE_METODO_SEC = re.compile(
    r"(method|methodology|approach|materials?\s+and\s+methods?|experiment|framework|system|proposed)",
    flags=re.IGNORECASE,
)

_RE_CONCLUSAO_SEC = re.compile(
    r"(conclusion|discussion|future\s+work|summary)",
    flags=re.IGNORECASE,
)


def _dividir_em_secoes(texto: str) -> dict[str, str]:
    """
    Tenta dividir o corpo do artigo em grandes seções por heurística de
    cabeçalhos em linha própria com poucos caracteres.
    Retorna dict {nome_secao_lower: texto_da_secao}.
    """
    # Detecta linhas curtas que parecem cabeçalhos de seção
    linhas = texto.splitlines()
    secoes: dict[str, list[str]] = {"inicio": []}
    secao_atual = "inicio"

    for linha in linhas:
        stripped = linha.strip()
        # Linha curta (< 60 chars) que parece ser um título de seção
        if 2 < len(stripped) < 60 and re.match(r"^[IVX\d\.\s]*[A-Z]", stripped):
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


# Extração principal ─────────────────────────────────────────────────────────

def extrair_informacoes(corpo: str) -> dict:
    """
    Extrai Objetivo, Problema, Método e Contribuição do corpo do artigo.
    Retorna dicionário com listas de trechos para cada campo.
    """
    secoes = _dividir_em_secoes(corpo)

    # Prioridade de seção para cada campo
    texto_objetivo_prob = secoes.get("introducao", corpo[:5000])
    texto_metodo = secoes.get("metodologia", corpo)
    texto_contribuicao = secoes.get("conclusao", "") + "\n" + secoes.get("introducao", "")

    # Tokenizar em sentenças cada bloco relevante
    sents_intro = sent_tokenize(texto_objetivo_prob)
    sents_corpo = sent_tokenize(corpo)
    sents_metodo = sent_tokenize(texto_metodo)
    sents_contribuicao = sent_tokenize(texto_contribuicao) if texto_contribuicao.strip() else sents_corpo

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


# Exibição ───────────────────────────────────────────────────────────────────

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


# Ponto de entrada da etapa ──────────────────────────────────────────────────

def executar_etapa2(resultados_etapa1: list[dict]) -> list[dict]:
    """
    Recebe os resultados da Etapa 1 e adiciona as extrações da Etapa 2.
    Retorna lista de dicionários com campo 'extracao' adicionado.
    """
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