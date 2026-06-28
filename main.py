"""
Trabalho de IIA — Processamento de Linguagem Natural
Tema: Bioinformática (IEEE Xplore)
Alunos: Eduardo, Janaina e Nuno
"""

import sys

from src.etapa1 import executar_etapa1
from src.etapa2 import executar_etapa2
from src.etapa3 import executar_etapa3
from src.etapa4 import executar_etapa4
from src.visualizacoes import executar_visualizacoes

# ── Configuração ───────────────────────────────────────────────────────────────

CAMINHO_GABARITO = "./gabarito.json"
SAIDA_AVALIACAO = "./saida/avaliacao"

DIRETORIO_ARTIGOS = "./artigos"
DIRETORIO_SAIDA_JSONLD = "./saida/jsonld"
DIRETORIO_SAIDA_VIZ = "./saida/visualizacoes"


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  TRABALHO DE IA - PROCESSAMENTO DE LINGUAGEM NATURAL")
    print("  Universidade Estadual de Maringá — Depto. de Informática")
    print("  Tema: Bioinformática | Base: IEEE Xplore")
    print("  Alunos: Eduardo, Janaina e Nuno")
    print("="*60)

    # ── Etapa 1: Leitura, pré-processamento e modelos de linguagem ─────────────
    resultados = executar_etapa1(DIRETORIO_ARTIGOS)

    if not resultados:
        print("\n[ERRO] Etapa 1 não retornou resultados. Verifique o diretório de artigos.")
        sys.exit(1)

    print(f"\n[OK] Etapa 1 concluída — {len(resultados)} artigo(s) processado(s).")

    # ── Etapa 2: Extração de informações ───────────────────────────────────────
    resultados = executar_etapa2(resultados)
    print("\n[OK] Etapa 2 concluída.")

    # ── Etapa 3: Serialização em ontologia JSON-LD ─────────────────────────────
    arquivos_jsonld = executar_etapa3(resultados, DIRETORIO_SAIDA_JSONLD)
    print(f"\n[OK] Etapa 3 concluída — {len(arquivos_jsonld)} arquivo(s) JSON-LD gerado(s).")

    # ── Visualizações ──────────────────────────────────────────────────────────
    executar_visualizacoes(resultados, DIRETORIO_SAIDA_VIZ)
    print("\n[OK] Visualizações concluídas.")

    # ── Etapa 4 (Janaina) ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ETAPA 4: Avaliação de Desempenho")
    executar_etapa4(resultados, CAMINHO_GABARITO , SAIDA_AVALIACAO)
    print("="*60)

    print("\n" + "="*60)
    print("  Pipeline concluído com sucesso!")
    print("="*60)


if __name__ == "__main__":
    main()