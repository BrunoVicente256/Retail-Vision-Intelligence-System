"""
rag_memory.py
-------------
Componente 3 do Retail Vision Intelligence System.
Responsabilidades EXCLUSIVAS deste módulo:
  - Gerar summaries ricos de inspecções via LLM
  - Indexar inspecções no ChromaDB com estratégia híbrida
  - Suportar queries em linguagem natural com retrieval semântico
  - Sintetizar respostas com referências explícitas (inspection_id, data)
  - Implementar e comparar duas estratégias de chunking (Recall@3)

Estratégias de chunking implementadas:
  A) HÍBRIDO (principal): summary LLM como texto + metadados estruturados
  B) RECORD COMPLETO: concatenação de todos os campos do JSON

Depende de:
  - utils/api_client.py  → síntese de respostas via LLM
  - sentence-transformers → embeddings locais, gratuitos, suportam PT
  - chromadb             → vector store persistente em disco

Instalação:
  pip install chromadb sentence-transformers
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.api_client import gemini_client

# ==========================================
# CONFIGURAÇÃO
# ==========================================
logger = logging.getLogger(__name__)

# Directórios de persistência
VECTORSTORE_DIR   = Path("vectorstore")
INSPECTIONS_DIR   = Path("data/inspections")

# Nomes das colecções ChromaDB (uma por estratégia)
COLECAO_HIBRIDA   = "inspections_hybrid"
COLECAO_COMPLETA  = "inspections_full"

# Modelo de embeddings: multilingual, gratuito, corre localmente
EMBEDDING_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Número de resultados por defeito no retrieval
TOP_K_DEFAULT     = 3


# ==========================================
# 1. INICIALIZAÇÃO (lazy — só carrega quando necessário)
# ==========================================

_chroma_client     = None
_embedding_model   = None
_colecao_hibrida   = None
_colecao_completa  = None


def _get_chroma():
    """Inicializa o cliente ChromaDB de forma lazy."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
            logger.info("[RAG] ChromaDB inicializado em: %s", VECTORSTORE_DIR)
        except ImportError:
            raise ImportError(
                "ChromaDB não instalado. Corre: pip install chromadb"
            )
    return _chroma_client


def _get_embedding_model():
    """Carrega o modelo de embeddings de forma lazy."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[RAG] A carregar modelo de embeddings: %s", EMBEDDING_MODEL)
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("[RAG] Modelo de embeddings carregado.")
        except ImportError:
            raise ImportError(
                "sentence-transformers não instalado. "
                "Corre: pip install sentence-transformers"
            )
    return _embedding_model


def _get_colecao(nome: str):
    """Obtém ou cria uma colecção ChromaDB pelo nome."""
    client = _get_chroma()
    return client.get_or_create_collection(
        name=nome,
        metadata={"hnsw:space": "cosine"}  # similaridade cosseno
    )


def _get_colecao_hibrida():
    global _colecao_hibrida
    if _colecao_hibrida is None:
        _colecao_hibrida = _get_colecao(COLECAO_HIBRIDA)
    return _colecao_hibrida


def _get_colecao_completa():
    global _colecao_completa
    if _colecao_completa is None:
        _colecao_completa = _get_colecao(COLECAO_COMPLETA)
    return _colecao_completa


def _gerar_embedding(texto: str) -> list:
    """Gera embedding para um texto."""
    modelo = _get_embedding_model()
    return modelo.encode(texto, normalize_embeddings=True).tolist()


# ==========================================
# 2. GERAÇÃO DE SUMMARIES (via LLM)
# ==========================================

_PROMPT_SUMMARY = """
Gera um summary de inspecção de prateleira rico em detalhes semânticos.
O summary será indexado numa base de dados vectorial para recuperação futura.
Deve conter TODOS os detalhes relevantes de forma natural e densa.

DADOS DA INSPECÇÃO:
- ID: {inspection_id}
- Data/Hora: {timestamp}
- Zona: {zone_id}
- Estado geral: {overall_status}
- Fill rate: {fill_rate_pct}
- Produtos detectados: {products}
- Issues encontrados: {issues_texto}
- Raciocínio do modelo: {reasoning}

INSTRUÇÕES PARA O SUMMARY:
1. Começa com a zona, data e estado geral
2. Descreve o fill rate de forma específica (ex: "72% de ocupação")
3. Menciona TODOS os produtos visíveis por categoria
4. Descreve CADA issue com localização e severidade
5. Inclui padrões temporais se relevantes (hora do dia)
6. Usa linguagem natural densa — este texto vai ser pesquisado semanticamente

EXEMPLO DE BOM SUMMARY:
"Inspecção da prateleira Z_S3 na terça-feira às 15h revelou estado warning.
Fill rate de 72%, abaixo do normal para este horário. Produto de limpeza
(detergente líquido) fora de posição na secção central — violação de planograma
de severidade média. Embalagem danificada detectada no lado direito da prateleira
inferior. Produtos de higiene pessoal presentes no lado esquerdo sem issues.
Duas lacunas visíveis na prateleira do meio sugerem início de rutura de stock."

DEVOLVE APENAS o texto do summary, sem JSON, sem prefixos.
"""


def _gerar_summary_llm(inspeccao: dict) -> str:
    """
    Gera um summary rico via LLM para indexação semântica.
    Fallback para summary sintético se a API falhar.
    """
    issues = inspeccao.get("issues", [])
    issues_texto = "; ".join([
        f"{i.get('type')} em {i.get('location', '?')} "
        f"(severidade {i.get('severity', '?')}, "
        f"confiança {i.get('confidence', 0):.0%})"
        for i in issues
    ]) or "nenhum issue detectado"

    # Extrai hora do timestamp para contexto temporal
    ts = inspeccao.get("timestamp", "")
    try:
        hora = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M")
        data = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%d/%m/%Y")
    except (ValueError, AttributeError):
        hora = "hora desconhecida"
        data = "data desconhecida"

    prompt = _PROMPT_SUMMARY.format(
        inspection_id=inspeccao.get("inspection_id", "?"),
        timestamp=f"{data} às {hora}",
        zone_id=inspeccao.get("zone_id", "?"),
        overall_status=inspeccao.get("overall_status", "?"),
        fill_rate_pct=f"{inspeccao.get('shelf_fill_rate', 0):.0%}",
        products=", ".join(inspeccao.get("products_detected", [])) or "não identificados",
        issues_texto=issues_texto,
        reasoning=inspeccao.get("model_reasoning", "")[:300]
    )

    try:
        response = gemini_client.gerar_conteudo(
            contents=[prompt],
            temperature=0.1,                   # ligeira criatividade para summaries
            response_mime_type="text/plain"
        )
        summary = response.text.strip()
        if len(summary) > 50:                  # validação mínima
            return summary
    except RuntimeError as e:
        logger.warning("[RAG] API indisponível para summary, usando fallback: %s", e)

    # Fallback sintético (sem LLM) — ainda semanticamente útil
    return _gerar_summary_sintetico(inspeccao, issues_texto, data, hora)


def _summary_rapido(inspeccao: dict) -> str:
    """
    Summary sintético sem qualquer chamada à API (instantâneo, sem custo de
    quota). Reconstrói issues_texto/data/hora e delega em _gerar_summary_sintetico.
    Usado na indexação ao vivo da interface CLI.
    """
    issues = inspeccao.get("issues", [])
    issues_texto = "; ".join([
        f"{i.get('type')} em {i.get('location', '?')} "
        f"(severidade {i.get('severity', '?')})"
        for i in issues
    ]) or "nenhum issue detectado"

    ts = inspeccao.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        hora, data = dt.strftime("%H:%M"), dt.strftime("%d/%m/%Y")
    except (ValueError, AttributeError):
        hora, data = "hora desconhecida", "data desconhecida"

    return _gerar_summary_sintetico(inspeccao, issues_texto, data, hora)


def _gerar_summary_sintetico(inspeccao: dict, issues_texto: str,
                              data: str, hora: str) -> str:
    """Summary gerado sem LLM — usado como fallback."""
    issues = inspeccao.get("issues", [])
    produtos = ", ".join(inspeccao.get("products_detected", [])) or "produtos não identificados"
    return (
        f"Inspecção {inspeccao.get('inspection_id', '?')} da zona "
        f"{inspeccao.get('zone_id', '?')} em {data} às {hora}. "
        f"Estado: {inspeccao.get('overall_status', '?')}. "
        f"Fill rate: {inspeccao.get('shelf_fill_rate', 0):.0%}. "
        f"Produtos: {produtos}. "
        f"Issues ({len(issues)}): {issues_texto}."
    )


# ==========================================
# 3. CONSTRUÇÃO DOS CHUNKS
# ==========================================

def _construir_chunk_hibrido(inspeccao: dict, summary: str) -> dict:
    """
    ESTRATÉGIA A — HÍBRIDA:
    Texto indexado = summary rico (gerado pela LLM)
    Metadados = campos estruturados para filtragem pre-retrieval

    Vantagem: retrieval semântico sobre texto natural + filtragem
    eficiente por zona/data/status sem custo de embedding.
    """
    ts = inspeccao.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        data_iso  = dt.strftime("%Y-%m-%d")
        hora_int  = dt.hour
        dia_semana = dt.strftime("%A")  # Monday, Tuesday, etc.
    except (ValueError, AttributeError):
        data_iso  = "1970-01-01"
        hora_int  = 0
        dia_semana = "unknown"

    issues = inspeccao.get("issues", [])
    tipos_issues = list({i.get("type", "") for i in issues if i.get("type")})

    return {
        "texto": summary,
        "metadados": {
            # Campos de identidade
            "inspection_id": inspeccao.get("inspection_id", ""),
            "image_path":    inspeccao.get("image_path", ""),
            # Campos temporais (para queries como "última semana", "sextas")
            "data":          data_iso,
            "hora":          hora_int,
            "dia_semana":    dia_semana,
            # Campos operacionais (para filtragem directa)
            "zone_id":       inspeccao.get("zone_id", ""),
            "status":        inspeccao.get("overall_status", ""),
            "fill_rate":     float(inspeccao.get("shelf_fill_rate", 0.0)),
            "n_issues":      len(issues),
            "tipos_issues":  json.dumps(tipos_issues),   # ChromaDB não aceita listas
        }
    }


def _construir_chunk_completo(inspeccao: dict) -> dict:
    """
    ESTRATÉGIA B — RECORD COMPLETO:
    Texto indexado = concatenação de todos os campos relevantes do JSON
    Sem metadados estruturados separados.

    Vantagem: simples de implementar.
    Desvantagem: o embedding representa uma média de todo o conteúdo —
    queries específicas podem perder-se no ruído de campos irrelevantes.
    Comparar Recall@3 com estratégia A no relatório.
    """
    issues = inspeccao.get("issues", [])
    issues_str = " | ".join([
        f"{i.get('type')} {i.get('location', '')} {i.get('severity', '')} {i.get('description', '')}"
        for i in issues
    ]) or "sem issues"

    texto_completo = (
        f"inspection_id={inspeccao.get('inspection_id', '')} "
        f"timestamp={inspeccao.get('timestamp', '')} "
        f"zone_id={inspeccao.get('zone_id', '')} "
        f"overall_status={inspeccao.get('overall_status', '')} "
        f"shelf_fill_rate={inspeccao.get('shelf_fill_rate', 0):.2f} "
        f"products_detected={' '.join(inspeccao.get('products_detected', []))} "
        f"issues={issues_str} "
        f"model_reasoning={inspeccao.get('model_reasoning', '')[:400]}"
    )

    return {
        "texto": texto_completo,
        "metadados": {
            "inspection_id": inspeccao.get("inspection_id", ""),
            "zone_id":       inspeccao.get("zone_id", ""),
            "status":        inspeccao.get("overall_status", ""),
            "data":          inspeccao.get("timestamp", "")[:10],
        }
    }


# ==========================================
# 4. INDEXAÇÃO
# ==========================================

def indexar_inspeccao(inspeccao: dict, estrategia: str = "hybrid",
                      gerar_summary_llm: bool = True) -> bool:
    """
    Indexa uma inspecção na(s) colecção(ões) ChromaDB.

    Args:
        inspeccao: JSON de inspecção do shelf_inspector.
        estrategia: "hybrid" | "full" | "both"
                    "both" indexa nas duas (para comparação de Recall@3).
        gerar_summary_llm: Se True (default), gera o summary rico via LLM
                    (consome quota). Se False, usa o summary sintético —
                    instantâneo e sem chamadas à API. A interface CLI usa
                    False na indexação ao vivo para preservar a quota de
                    visão; o bulk-indexing do dataset usa True (qualidade).

    Returns:
        True se indexado com sucesso, False em caso de erro.
    """
    inspection_id = inspeccao.get("inspection_id", "")
    if not inspection_id:
        logger.error("[RAG] Inspecção sem inspection_id — não indexada.")
        return False

    logger.info("[RAG] A indexar inspecção: %s", inspection_id)

    # Gera summary uma vez (partilhado entre estratégias)
    summary = _gerar_summary_llm(inspeccao) if gerar_summary_llm \
        else _summary_rapido(inspeccao)
    logger.info("[RAG] Summary gerado (%d chars).", len(summary))

    sucesso = True

    if estrategia in ("hybrid", "both"):
        sucesso &= _indexar_chunk(
            _get_colecao_hibrida(),
            inspection_id,
            _construir_chunk_hibrido(inspeccao, summary)
        )

    if estrategia in ("full", "both"):
        sucesso &= _indexar_chunk(
            _get_colecao_completa(),
            inspection_id,
            _construir_chunk_completo(inspeccao)
        )

    # Persiste o JSON da inspecção em data/inspections/
    _persistir_inspeccao(inspeccao)

    return sucesso


def _indexar_chunk(colecao, doc_id: str, chunk: dict) -> bool:
    """Indexa um chunk numa colecção ChromaDB."""
    try:
        embedding = _gerar_embedding(chunk["texto"])

        # upsert: actualiza se já existir (idempotente — seguro correr múltiplas vezes)
        colecao.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk["texto"]],
            metadatas=[chunk["metadados"]]
        )
        logger.info("[RAG] Chunk indexado em '%s': %s", colecao.name, doc_id)
        return True
    except Exception as e:
        logger.error("[RAG] Erro ao indexar %s em '%s': %s",
                     doc_id, colecao.name, e)
        return False


def _persistir_inspeccao(inspeccao: dict) -> None:
    """Guarda o JSON completo da inspecção em data/inspections/."""
    INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    insp_id = inspeccao.get("inspection_id", "unknown")
    caminho = INSPECTIONS_DIR / f"{insp_id}.json"
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(inspeccao, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logger.warning("[RAG] Não foi possível persistir inspecção: %s", e)


def indexar_batch(inspeccoes: list, estrategia: str = "both") -> dict:
    """
    Indexa uma lista de inspecções em lote.
    Usado para indexar o dataset_final.json gerado pelo batch_processor.

    Args:
        inspeccoes: Lista de dicionários de inspecção.
        estrategia: "hybrid" | "full" | "both"

    Returns:
        Dicionário com estatísticas: {sucesso, falha, total}
    """
    stats = {"sucesso": 0, "falha": 0, "total": len(inspeccoes)}

    for i, inspeccao in enumerate(inspeccoes, 1):
        logger.info("[RAG] Indexando %d/%d...", i, stats["total"])
        ok = indexar_inspeccao(inspeccao, estrategia=estrategia)
        if ok:
            stats["sucesso"] += 1
        else:
            stats["falha"] += 1

    logger.info(
        "[RAG] Batch indexado: %d sucesso, %d falha.",
        stats["sucesso"], stats["falha"]
    )
    return stats


def indexar_a_partir_de_dataset(
    caminho_dataset: str = "cache/vision/dataset_final.json",
    estrategia: str = "both"
) -> dict:
    """
    Carrega o dataset_final.json do batch_processor e indexa todas as inspecções.
    Ponto de entrada conveniente para indexar o dataset completo.
    """
    path = Path(caminho_dataset)
    if not path.exists():
        logger.error("[RAG] Dataset não encontrado: %s", caminho_dataset)
        return {"sucesso": 0, "falha": 0, "total": 0}

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    inspecoes_dict = dataset.get("inspecoes", {})
    inspeccoes = [
        entry["analise_ia"]
        for entry in inspecoes_dict.values()
        if entry.get("analise_ia") is not None
    ]

    logger.info("[RAG] A indexar %d inspecções do dataset...", len(inspeccoes))
    return indexar_batch(inspeccoes, estrategia=estrategia)


# ==========================================
# 5. RETRIEVAL E SÍNTESE
# ==========================================

_PROMPT_SINTESE = """
És um assistente especialista em gestão de retalho com acesso ao histórico
de inspecções de prateleiras.

PERGUNTA DO GESTOR: "{query}"

INSPECÇÕES RELEVANTES RECUPERADAS (ordenadas por relevância):
{contexto}

INSTRUÇÕES:
1. Responde directamente à pergunta usando APENAS os dados das inspecções acima
2. Cita SEMPRE os inspection_id e datas relevantes na resposta
3. Se identificares padrões (ex: sempre às terças, sempre na mesma zona), menciona-os
4. Se os dados forem insuficientes para responder, diz claramente
5. Responde em português, de forma concisa e accionável

RESPOSTA:
"""


def _formatar_contexto(resultados: list) -> str:
    """Formata os resultados do retrieval para o prompt de síntese."""
    partes = []
    for i, res in enumerate(resultados, 1):
        meta = res.get("metadados", {})
        partes.append(
            f"[{i}] inspection_id={meta.get('inspection_id', '?')} | "
            f"zona={meta.get('zone_id', '?')} | "
            f"data={meta.get('data', '?')} | "
            f"status={meta.get('status', '?')} | "
            f"fill_rate={meta.get('fill_rate', '?')}\n"
            f"    {res.get('documento', '')[:300]}"
        )
    return "\n\n".join(partes)


def _executar_retrieval(
    query: str,
    colecao,
    k: int = TOP_K_DEFAULT,
    filtros: dict | None = None
) -> list:
    """
    Executa retrieval por similaridade numa colecção ChromaDB.

    Args:
        query: Texto da pergunta em linguagem natural.
        colecao: Colecção ChromaDB a pesquisar.
        k: Número de resultados a devolver.
        filtros: Metadados para filtragem pre-retrieval (ex: {"zone_id": "Z_EMPTY"}).

    Returns:
        Lista de dicionários com documento, metadados e distância.
    """
    embedding_query = _gerar_embedding(query)

    kwargs = {
        "query_embeddings": [embedding_query],
        "n_results": min(k, colecao.count() or 1),
        "include": ["documents", "metadatas", "distances"]
    }
    if filtros:
        kwargs["where"] = filtros

    try:
        resultados = colecao.query(**kwargs)
    except Exception as e:
        logger.error("[RAG] Erro no retrieval: %s", e)
        return []

    # Normaliza o formato de saída
    docs      = resultados.get("documents", [[]])[0]
    metas     = resultados.get("metadatas", [[]])[0]
    distancias = resultados.get("distances", [[]])[0]

    return [
        {
            "documento":  doc,
            "metadados":  meta,
            "distancia":  dist,
            "relevancia": round(1 - dist, 3)  # cosine distance → similarity
        }
        for doc, meta, dist in zip(docs, metas, distancias)
    ]


def query_memoria(
    pergunta: str,
    k: int = TOP_K_DEFAULT,
    estrategia: str = "hybrid",
    filtros: dict | None = None,
    sintetizar: bool = True
) -> dict:
    """
    Responde a uma pergunta em linguagem natural usando o histórico de inspecções.

    Args:
        pergunta: Pergunta do gestor em linguagem natural.
        k: Número de documentos a recuperar (default: 3).
        estrategia: "hybrid" | "full" — qual colecção usar.
        filtros: Metadados para filtragem pre-retrieval opcional.
        sintetizar: Se True, usa LLM para sintetizar resposta.
                    Se False, devolve apenas os documentos recuperados.

    Returns:
        Dicionário com:
          - resposta: texto sintetizado pela LLM
          - chunks_recuperados: lista de documentos relevantes
          - inspection_ids_citados: IDs das inspecções referenciadas
    """
    logger.info("[RAG] Query: '%s'", pergunta[:60])

    colecao = _get_colecao_hibrida() if estrategia == "hybrid" else _get_colecao_completa()

    if colecao.count() == 0:
        return {
            "resposta": "Ainda não há inspecções indexadas na memória do sistema.",
            "chunks_recuperados": [],
            "inspection_ids_citados": []
        }

    chunks = _executar_retrieval(pergunta, colecao, k=k, filtros=filtros)

    if not chunks:
        return {
            "resposta": "Não foram encontradas inspecções relevantes para esta query.",
            "chunks_recuperados": [],
            "inspection_ids_citados": []
        }

    ids_citados = [
        c["metadados"].get("inspection_id", "")
        for c in chunks
        if c["metadados"].get("inspection_id")
    ]

    if not sintetizar:
        return {
            "resposta": None,
            "chunks_recuperados": chunks,
            "inspection_ids_citados": ids_citados
        }

    # Síntese via LLM
    contexto = _formatar_contexto(chunks)
    prompt   = _PROMPT_SINTESE.format(query=pergunta, contexto=contexto)

    try:
        response = gemini_client.gerar_conteudo(
            contents=[prompt],
            temperature=0.1,
            response_mime_type="text/plain"
        )
        resposta_texto = response.text.strip()
    except RuntimeError as e:
        logger.warning("[RAG] API indisponível para síntese: %s", e)
        # Fallback: devolve os documentos directamente sem síntese
        resposta_texto = (
            "Inspecções relevantes encontradas:\n" +
            "\n".join([
                f"- {c['metadados'].get('inspection_id', '?')} "
                f"({c['metadados'].get('zona', c['metadados'].get('zone_id', '?'))}, "
                f"{c['metadados'].get('data', '?')}): "
                f"{c['documento'][:150]}..."
                for c in chunks
            ])
        )

    return {
        "resposta": resposta_texto,
        "chunks_recuperados": chunks,
        "inspection_ids_citados": ids_citados
    }


# ==========================================
# 6. QUERIES OBRIGATÓRIAS DO ENUNCIADO
# ==========================================

def ultima_vez_zona_com_problema(zone_id: str) -> dict:
    """
    'Quando foi a última vez que a zona X teve problemas de prateleira vazia?'
    Filtra por zona e pesquisa por empty_shelf semanticamente.
    """
    return query_memoria(
        pergunta=f"Quando foi a última vez que a zona {zone_id} teve prateleira vazia ou rutura de stock?",
        filtros={"zone_id": zone_id},
        k=TOP_K_DEFAULT
    )


def zonas_com_mais_issues_planograma(dias: int = 14) -> dict:
    """
    'Que zonas tiveram mais issues de planograma nas últimas N semanas?'
    """
    return query_memoria(
        pergunta=f"Que zonas tiveram mais violações de planograma, produto errado ou etiqueta ausente nas últimas {dias} dias?",
        k=5  # mais resultados para agregar por zona
    )


def padroes_por_dia_semana(dia: str) -> dict:
    """
    'Existe algum padrão nos problemas detectados às X?'
    Ex: dia = 'sextas-feiras à tarde'
    """
    return query_memoria(
        pergunta=f"Existe algum padrão nos problemas de prateleira detectados {dia}? "
                 f"Que issues são mais frequentes nesse período?",
        k=5
    )


def regras_mais_disparadas(periodo: str = "este mês") -> dict:
    """
    'Que regras foram mais frequentemente disparadas este mês?'
    Consulta os logs de execução (não o ChromaDB).
    """
    log_path = Path("data/rules/execution_logs.json")
    if not log_path.exists():
        return {
            "resposta": "Ainda não há logs de execução de regras.",
            "chunks_recuperados": [],
            "inspection_ids_citados": []
        }

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "resposta": "Erro ao ler logs de execução.",
            "chunks_recuperados": [],
            "inspection_ids_citados": []
        }

    from collections import Counter
    contagem = Counter()
    for entrada in logs:
        for detalhe in entrada.get("detalhes", []):
            if detalhe.get("disparou"):
                contagem[detalhe.get("rule_id", "?")] += 1

    if not contagem:
        return {
            "resposta": "Nenhuma regra disparou ainda.",
            "chunks_recuperados": [],
            "inspection_ids_citados": []
        }

    ranking = contagem.most_common(5)
    resposta = f"Regras mais disparadas ({periodo}):\n" + "\n".join([
        f"  {i+1}. {rule_id}: {n} vez(es)"
        for i, (rule_id, n) in enumerate(ranking)
    ])

    return {
        "resposta": resposta,
        "chunks_recuperados": [],
        "inspection_ids_citados": []
    }


# ==========================================
# 7. AVALIAÇÃO — RECALL@K
# ==========================================

def avaliar_recall_k(
    queries_gt: list,
    k: int = TOP_K_DEFAULT,
    estrategia: str = "hybrid"
) -> dict:
    """
    Calcula Recall@K para uma estratégia de chunking.
    Usado no harness de avaliação (evaluate.py) e na comparação do relatório.

    Args:
        queries_gt: Lista de dicionários com:
                    {
                      "query": "texto da pergunta",
                      "inspection_ids_relevantes": ["INS_...", "INS_..."]
                    }
        k: Janela de recuperação (default: 3).
        estrategia: "hybrid" | "full"

    Returns:
        {
          "recall_at_k": float,   # % queries onde doc relevante está no top-k
          "k": int,
          "estrategia": str,
          "total_queries": int,
          "hits": int,
          "detalhes": list        # por query: hit/miss e IDs recuperados
        }
    """
    hits = 0
    detalhes = []

    for item in queries_gt:
        query         = item["query"]
        ids_relevantes = set(item.get("inspection_ids_relevantes", []))

        resultado = query_memoria(
            query, k=k, estrategia=estrategia, sintetizar=False
        )
        ids_recuperados = set(resultado.get("inspection_ids_citados", []))

        hit = bool(ids_relevantes.intersection(ids_recuperados))
        if hit:
            hits += 1

        detalhes.append({
            "query": query,
            "hit": hit,
            "ids_relevantes": list(ids_relevantes),
            "ids_recuperados": list(ids_recuperados)
        })

    total = len(queries_gt)
    recall = hits / total if total > 0 else 0.0

    logger.info(
        "[RAG] Recall@%d (%s): %.1f%% (%d/%d)",
        k, estrategia, recall * 100, hits, total
    )

    return {
        "recall_at_k": round(recall, 3),
        "k": k,
        "estrategia": estrategia,
        "total_queries": total,
        "hits": hits,
        "detalhes": detalhes
    }


# ==========================================
# 8. UTILITÁRIOS
# ==========================================

def estatisticas_vectorstore() -> dict:
    """Devolve estatísticas das colecções indexadas."""
    try:
        hibrida  = _get_colecao_hibrida().count()
        completa = _get_colecao_completa().count()
    except Exception:
        hibrida, completa = 0, 0

    return {
        "colecao_hibrida":  hibrida,
        "colecao_completa": completa,
        "vectorstore_dir":  str(VECTORSTORE_DIR)
    }


def limpar_vectorstore(estrategia: str = "both") -> None:
    """
    Limpa a(s) colecção(ões) do ChromaDB.
    CUIDADO: operação irreversível — requer re-indexação completa.
    """
    client = _get_chroma()
    if estrategia in ("hybrid", "both"):
        try:
            client.delete_collection(COLECAO_HIBRIDA)
            logger.warning("[RAG] Colecção '%s' eliminada.", COLECAO_HIBRIDA)
        except Exception:
            pass
    if estrategia in ("full", "both"):
        try:
            client.delete_collection(COLECAO_COMPLETA)
            logger.warning("[RAG] Colecção '%s' eliminada.", COLECAO_COMPLETA)
        except Exception:
            pass
    # Repõe as referências globais
    global _colecao_hibrida, _colecao_completa
    _colecao_hibrida  = None
    _colecao_completa = None


# ==========================================
# 9. EXECUÇÃO DE TESTE
# ==========================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.console import configurar_consola
    configurar_consola()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("\n" + "=" * 65)
    print("   TESTE DO RAG MEMORY".center(65))
    print("=" * 65)

    # --- Inspecção sintética para teste ---
    inspeccao_teste = {
        "inspection_id": "INS_TEST_RAG_001",
        "timestamp": "2026-06-10T15:30:00Z",
        "image_path": "images/empty/teste.jpg",
        "zone_id": "Z_EMPTY",
        "overall_status": "critical",
        "shelf_fill_rate": 0.38,
        "issues": [
            {
                "issue_id": "ISS_001",
                "type": "empty_shelf",
                "location": "prateleira inferior, lado esquerdo",
                "severity": "high",
                "description": "Rutura profunda de stock — fundo da prateleira visível",
                "confidence": 0.91,
                "affected_area_pct": 0.55
            }
        ],
        "products_detected": ["detergentes", "produtos de limpeza"],
        "model_reasoning": (
            "Prateleira de produtos de limpeza com rutura grave no lado esquerdo inferior. "
            "O fundo metálico da prateleira está visível numa área de aproximadamente 55%. "
            "Fill rate estimado em 38%, muito abaixo do esperado para este horário. "
            "Classificado como critical pela extensão e profundidade da rutura."
        )
    }

    # Teste 1: Indexação
    print("\n[TESTE 1] Indexação de inspecção sintética")
    ok = indexar_inspeccao(inspeccao_teste, estrategia="both")
    print(f"  Indexado com sucesso: {ok}")
    stats = estatisticas_vectorstore()
    print(f"  Colecção híbrida : {stats['colecao_hibrida']} documentos")
    print(f"  Colecção completa: {stats['colecao_completa']} documentos")

    # Teste 2: Query obrigatória 1
    print("\n" + "-" * 65)
    print("\n[TESTE 2] Query: última vez que Z_EMPTY teve problemas")
    r = ultima_vez_zona_com_problema("Z_EMPTY")
    print(f"  Resposta: {r['resposta'][:200]}")
    print(f"  IDs citados: {r['inspection_ids_citados']}")

    # Teste 3: Query livre
    print("\n" + "-" * 65)
    print("\n[TESTE 3] Query livre em linguagem natural")
    r2 = query_memoria("Que zonas tiveram ruturas de stock graves?")
    print(f"  Resposta: {r2['resposta'][:200]}")
    print(f"  Chunks recuperados: {len(r2['chunks_recuperados'])}")
    for c in r2["chunks_recuperados"]:
        print(f"    relevância={c['relevancia']:.3f} | "
              f"id={c['metadados'].get('inspection_id', '?')}")

    # Teste 4: Recall@3 com ground truth mínimo
    print("\n" + "-" * 65)
    print("\n[TESTE 4] Recall@3 (ground truth mínimo)")
    gt_minimo = [
        {
            "query": "prateleira vazia com rutura de stock",
            "inspection_ids_relevantes": ["INS_TEST_RAG_001"]
        },
        {
            "query": "zona Z_EMPTY com problemas críticos",
            "inspection_ids_relevantes": ["INS_TEST_RAG_001"]
        }
    ]
    for est in ("hybrid", "full"):
        r_k = avaliar_recall_k(gt_minimo, k=3, estrategia=est)
        print(f"  Recall@3 [{est:6}]: {r_k['recall_at_k']:.0%} "
              f"({r_k['hits']}/{r_k['total_queries']} hits)")

    # Teste 5: Regras mais disparadas
    print("\n" + "-" * 65)
    print("\n[TESTE 5] Regras mais disparadas")
    r3 = regras_mais_disparadas()
    print(f"  {r3['resposta']}")

    print("\n" + "=" * 65)
    print("[CONCLUÍDO]")