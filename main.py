import sys
from src.etapa1 import executar_etapa1

DIRETORIO_ARTIGOS = "./artigos"

def main():
    print("\n" + "="*60)
    print("  TRABALHO DE IA - PROCESSAMENTO DE LINGUAGEM NATURAL")
    print("  Alunos: Eduardo, Janaina e Nuno.")
    print("="*60)

    # Etapa 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    resultados_etapa1 = executar_etapa1(DIRETORIO_ARTIGOS)

    if not resultados_etapa1:
        print("\n[ERRO] Etapa 1 não retornou resultados. Verifique o diretório de artigos.")
        sys.exit(1)

    print("\n[OK] Etapa 1 concluída.")
    print(f"     {len(resultados_etapa1)} artigo(s) processado(s).")
    
    print("\n[INFO] Etapas 2, 3 e 4 serão integradas aqui conforme forem implementadas.")


if __name__ == "__main__":
    main()