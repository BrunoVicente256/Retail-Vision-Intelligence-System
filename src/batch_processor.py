"""
batch_processor.py
------------------
Orquestração PURA do dataset de inspeções. Não contém lógica de visão,
prompts, nem comunicação com a API — delega tudo no shelf_inspector.

Responsabilidades EXCLUSIVAS deste módulo:
  - Percorrer todas as pastas de imagens de forma determinística (ordenada)
  - Aplicar pausa de rate limiting SÓ entre chamadas REAIS à API
    (cache hits não dormem — são instantâneos e não consomem quota)
  - Distinguir erros de QUOTA (esgotamento diário) de erros de PARSING
  - Fallback gracioso: se a quota esgotar, continua a servir imagens em cache
  - Agregar todos os resultados em cache/vision/dataset_final.json

EXECUÇÃO (a partir da RAIZ do projeto, não de dentro de src/):
    python src/batch_processor.py

    O Python adiciona src/ ao sys.path (imports planos resolvem),
    enquanto o CWD continua a ser a raiz (images/ e cache/ resolvem).
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from shelf_inspector import inspecionar_prateleira, esta_em_cache
from utils.api_client import gemini_client, MODEL_NAME
from utils.console import configurar_consola

configurar_consola()

# Ordem das categorias (nomes das pastas em images/).
CATEGORIAS = ["normal", "empty", "dirty", "ambiguous", "planogram"]

# Estratégia de prompting usada na construção do dataset base.
ESTRATEGIA = "cot"

# Extensões de imagem aceites.
EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg")

# Pausa entre chamadas REAIS à API (segundos).
# NOTA: o api_client já impõe o limite duro de 15 req/min (janela deslizante).
# Esta pausa é apenas um amortecedor secundário para suavizar o tráfego e
# reduzir a probabilidade de 429 no free tier. 4s ≈ ritmo de 15/min.
# Cache hits NÃO sofrem esta pausa.
PAUSA_ENTRE_CHAMADAS_REAIS = 4.0

# Caminho do índice agregado final.
CAMINHO_INDICE = Path("cache/vision/dataset_final.json")

# ==========================================
# RECOLHA DETERMINÍSTICA DE FICHEIROS
# ==========================================

def _listar_imagens(pasta: Path) -> list[Path]:
    """
    Devolve as imagens de uma pasta por ordem alfabética determinística.
    os.listdir tem ordem arbitrária — ordenar é essencial para
    reprodutibilidade (requisito explícito do enunciado).
    """
    if not pasta.exists():
        return []
    ficheiros = [
        p for p in pasta.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_VALIDAS
    ]
    return sorted(ficheiros, key=lambda p: p.name.lower())


# ==========================================
# PROCESSAMENTO EM LOTE
# ==========================================

def processar_todas_as_imagens(base_dir: str = "data/images") -> dict:
    """
    Percorre todas as categorias, inspeciona cada imagem e agrega os
    resultados num único índice persistido em disco.

    Returns:
        O dicionário do índice agregado (também escrito em CAMINHO_INDICE).
    """
    base = Path(base_dir)

    inspecoes: dict = {}      # sucessos: id_imagem -> registo
    falhas: list = []         # falhas: {caminho, categoria, motivo}
    stats = {cat: 0 for cat in CATEGORIAS}
    cache_hits = 0
    chamadas_reais = 0
    erros_parsing = 0
    erros_quota = 0

    # Controla o espaçamento: só dormimos ANTES de uma chamada real que
    # não seja a primeira da sessão.
    ja_houve_chamada_real = False

    print("=" * 60)
    print("   PROCESSAMENTO EM LOTE — RETAIL VISION".center(60))
    print("=" * 60)
    print(f"  Estratégia : {ESTRATEGIA}")
    print(f"  Modelo     : {MODEL_NAME}")
    print(f"  Pausa real : {PAUSA_ENTRE_CHAMADAS_REAIS}s (só entre chamadas à API)")
    print("=" * 60)

    for categoria in CATEGORIAS:
        pasta = base / categoria
        imagens = _listar_imagens(pasta)

        if not imagens:
            print(f"\n[AVISO] Sem imagens em: {pasta}")
            continue

        print(f"\n---> /{categoria}  ({len(imagens)} imagens)")

        for caminho in imagens:
            caminho_str = str(caminho)
            id_imagem = f"{categoria}__{caminho.name}"

            em_cache = esta_em_cache(caminho_str, ESTRATEGIA)

            # --- FALLBACK GRACIOSO ---
            # Quota esgotada + imagem não cacheada → nem tentamos chamar a API.
            # O sistema continua a processar as restantes imagens em cache.
            if not em_cache and gemini_client.quota_esgotada:
                print(f"  [QUOTA] Saltada (sem cache, quota esgotada): {caminho.name}")
                falhas.append({
                    "caminho": caminho_str,
                    "categoria": categoria,
                    "motivo": "quota_esgotada",
                })
                erros_quota += 1
                continue

            # --- ESPAÇAMENTO ENTRE CHAMADAS REAIS ---
            # Só dorme se esta vai ser uma chamada real E já houve outra antes.
            if not em_cache and ja_houve_chamada_real:
                print(f"  [RATE] A aguardar {PAUSA_ENTRE_CHAMADAS_REAIS}s entre chamadas reais...")
                time.sleep(PAUSA_ENTRE_CHAMADAS_REAIS)

            # --- INSPEÇÃO (cache hit ou chamada real, decidido lá dentro) ---
            resultado = inspecionar_prateleira(caminho_str, tipo_prompt=ESTRATEGIA)

            # Se não estava em cache, uma tentativa real ocorreu (sucesso ou não).
            if not em_cache:
                ja_houve_chamada_real = True

            # --- CLASSIFICAÇÃO DO RESULTADO ---
            if resultado is not None:
                inspecoes[id_imagem] = {
                    "caminho": caminho_str,
                    "categoria_real": categoria,
                    "origem": "cache" if em_cache else "api",
                    "analise_ia": resultado,
                }
                stats[categoria] += 1
                if em_cache:
                    cache_hits += 1
                    print(f"  [CACHE] {caminho.name}  →  {resultado.get('overall_status')}")
                else:
                    chamadas_reais += 1
                    print(f"  [API]   {caminho.name}  →  {resultado.get('overall_status')}")
            else:
                # Resultado None: distinguir quota de parsing via a flag do singleton.
                if gemini_client.quota_esgotada:
                    motivo = "quota_esgotada"
                    erros_quota += 1
                    print(f"  [QUOTA] Falha por quota esgotada: {caminho.name}")
                else:
                    motivo = "parsing_ou_erro_api"
                    erros_parsing += 1
                    print(f"  [ERRO]  Parsing/API falhou: {caminho.name}")
                falhas.append({
                    "caminho": caminho_str,
                    "categoria": categoria,
                    "motivo": motivo,
                })

    # ==========================================
    # AGREGAÇÃO E PERSISTÊNCIA
    # ==========================================
    indice = {
        "metadata": {
            "gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "estrategia": ESTRATEGIA,
            "modelo": MODEL_NAME,
            "total_imagens": len(inspecoes) + len(falhas),
            "sucessos": len(inspecoes),
            "cache_hits": cache_hits,
            "chamadas_reais": chamadas_reais,
            "erros_parsing": erros_parsing,
            "erros_quota": erros_quota,
            "por_categoria": stats,
        },
        "inspecoes": inspecoes,
        "falhas": falhas,
    }

    CAMINHO_INDICE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMINHO_INDICE, "w", encoding="utf-8") as f:
        json.dump(indice, f, indent=4, ensure_ascii=False)

    # ==========================================
    # RESUMO
    # ==========================================
    print("\n" + "=" * 60)
    print("   RESUMO".center(60))
    print("=" * 60)
    print(f"  Sucessos          : {len(inspecoes)}")
    print(f"    ├─ cache hits    : {cache_hits}")
    print(f"    └─ chamadas API  : {chamadas_reais}")
    print(f"  Erros de parsing  : {erros_parsing}")
    print(f"  Erros de quota    : {erros_quota}")
    print("  Por categoria:")
    for cat in CATEGORIAS:
        print(f"    /{cat:<10}: {stats[cat]}")
    print("=" * 60)
    print(f"  Índice agregado   : {CAMINHO_INDICE}")
    if gemini_client.quota_esgotada:
        print("  [!] Quota diária esgotada — corre de novo amanhã para")
        print("      completar as imagens saltadas (as já feitas ficam em cache).")
    print("=" * 60)

    return indice


if __name__ == "__main__":
    processar_todas_as_imagens()