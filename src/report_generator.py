"""
report_generator.py
-------------------
Componente 4 do Retail Vision Intelligence System.
Responsabilidades EXCLUSIVAS deste módulo:
  - Gerar relatórios de inspecção em Markdown por sessão
  - Agregar resultados de múltiplas inspecções numa sessão
  - Integrar contexto histórico do RAG (rag_memory.py)
  - Integrar alertas disparados pelo rule_engine.py
  - Garantir as 6 secções obrigatórias do enunciado

Estrutura obrigatória do relatório (enunciado secção 7):
  1. Sumário executivo       (máx. 150 palavras)
  2. Problemas por zona
  3. Regras disparadas
  4. Contexto histórico RAG  (com inspection_id + data)
  5. Recomendações           (máx. 5, ordenadas por urgência)
  6. Integração com trajectória (placeholder — não implementado)

Output: reports/report_YYYYMMDD_HHMMSS.md

Depende de:
  - utils/api_client.py → geração do relatório via LLM
  - rag_memory.py       → contexto histórico
  - rule_engine.py      → alertas disparados
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from utils.api_client import gemini_client
from rag_memory import query_memoria, estatisticas_vectorstore
from rule_engine import executar_regras, listar_regras

# ==========================================
# CONFIGURAÇÃO
# ==========================================
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
MAX_PALAVRAS_SUMARIO = 150
MAX_RECOMENDACOES    = 5


# ==========================================
# 1. PROMPTS
# ==========================================

_PROMPT_SUMARIO = """
Escreve um sumário executivo de uma sessão de inspecção de prateleiras de retalho.
O sumário deve ter NO MÁXIMO {max_palavras} palavras. Linguagem directa e accionável.

DADOS DA SESSÃO:
- Data/hora: {data_hora}
- Zonas inspeccionadas: {n_zonas}
- Total de inspecções: {n_inspeccoes}
- Issues críticos: {n_criticos}
- Issues warning: {n_warnings}
- Issues ok: {n_ok}
- Fill rate médio: {fill_rate_medio}
- Tipos de issues mais frequentes: {tipos_frequentes}

INSTRUÇÕES:
1. Primeira frase: estado geral da sessão (bom/preocupante/crítico)
2. Menciona o número de zonas e issues críticos
3. Destaca o problema mais grave se existir
4. Termina com uma acção imediata se necessário
5. MÁXIMO {max_palavras} palavras — conta as palavras antes de responder

DEVOLVE APENAS o texto do sumário, sem títulos nem formatação Markdown.
"""

_PROMPT_RECOMENDACOES = """
Gera até {max_rec} recomendações accionáveis para uma sessão de inspecção de retalho.
Cada recomendação deve ser específica o suficiente para ser executada sem interpretação adicional.

PROBLEMAS DETECTADOS NA SESSÃO:
{problemas_resumo}

ALERTAS DE REGRAS DISPARADOS:
{alertas_resumo}

CONTEXTO HISTÓRICO (padrões passados):
{contexto_historico}

INSTRUÇÕES:
1. Ordena por urgência (mais urgente primeiro)
2. Cada recomendação: acção concreta + zona + motivo
3. Máximo {max_rec} recomendações
4. Formato de cada linha: "N. [URGÊNCIA] Acção específica (Zona X — motivo)"
   Urgência: CRÍTICO | URGENTE | NORMAL
5. Não repitas recomendações semelhantes

DEVOLVE APENAS a lista numerada, sem texto adicional.
"""

_PROMPT_CONTEXTO_HISTORICO = """
Analisa o histórico de inspecções e identifica padrões relevantes para este relatório.

QUERY CONTEXTUAL: {query}

INSPECÇÕES HISTÓRICAS RECUPERADAS:
{chunks}

INSTRUÇÕES:
1. Identifica padrões recorrentes (mesma zona, mesmo problema, mesmo horário)
2. Compara a sessão actual com o histórico
3. Menciona SEMPRE os inspection_id e datas das inspecções referenciadas
4. Máximo 3 parágrafos
5. Se não há padrões claros, diz isso directamente

DEVOLVE APENAS o texto de análise, sem títulos.
"""


# ==========================================
# 2. AGREGAÇÃO DE DADOS DA SESSÃO
# ==========================================

def _agregar_sessao(inspeccoes: list) -> dict:
    """
    Agrega métricas de uma lista de inspecções numa sessão.
    Devolve um dicionário com estatísticas consolidadas.
    """
    if not inspeccoes:
        return {}

    n_criticos = sum(1 for i in inspeccoes if i.get("overall_status") == "critical")
    n_warnings = sum(1 for i in inspeccoes if i.get("overall_status") == "warning")
    n_ok       = sum(1 for i in inspeccoes if i.get("overall_status") == "ok")

    fill_rates = [
        i.get("shelf_fill_rate", 0)
        for i in inspeccoes
        if isinstance(i.get("shelf_fill_rate"), (int, float))
    ]
    fill_medio = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

    # Contagem de tipos de issues
    from collections import Counter
    contador_tipos = Counter(
        issue.get("type", "other")
        for insp in inspeccoes
        for issue in insp.get("issues", [])
    )

    # Agrupa por zona
    zonas = {}
    for insp in inspeccoes:
        zona = insp.get("zone_id", "Z_UNKNOWN")
        if zona not in zonas:
            zonas[zona] = []
        zonas[zona].append(insp)

    return {
        "n_inspeccoes":     len(inspeccoes),
        "n_zonas":          len(zonas),
        "n_criticos":       n_criticos,
        "n_warnings":       n_warnings,
        "n_ok":             n_ok,
        "fill_rate_medio":  round(fill_medio, 3),
        "tipos_frequentes": contador_tipos.most_common(3),
        "por_zona":         zonas,
        "todos_issues":     [
            issue
            for insp in inspeccoes
            for issue in insp.get("issues", [])
        ]
    }


def _contar_palavras(texto: str) -> int:
    """Conta palavras num texto."""
    return len(texto.split())


def _truncar_sumario(texto: str, max_palavras: int) -> str:
    """
    Garante que o sumário não excede max_palavras.
    Se exceder, corta na última frase completa antes do limite.
    """
    palavras = texto.split()
    if len(palavras) <= max_palavras:
        return texto

    # Corta no limite e tenta terminar numa frase completa
    texto_cortado = " ".join(palavras[:max_palavras])
    ultimo_ponto  = max(
        texto_cortado.rfind("."),
        texto_cortado.rfind("!"),
        texto_cortado.rfind("?")
    )
    if ultimo_ponto > len(texto_cortado) // 2:
        return texto_cortado[:ultimo_ponto + 1]
    return texto_cortado + "..."


# ==========================================
# 3. GERAÇÃO DE CADA SECÇÃO
# ==========================================

def _gerar_sumario_executivo(agregado: dict, data_hora: str) -> str:
    """Secção 1 — Sumário executivo (máx. 150 palavras)."""
    tipos_str = ", ".join([
        f"{t} ({n}x)" for t, n in agregado.get("tipos_frequentes", [])
    ]) or "nenhum issue"

    prompt = _PROMPT_SUMARIO.format(
        max_palavras=MAX_PALAVRAS_SUMARIO,
        data_hora=data_hora,
        n_zonas=agregado.get("n_zonas", 0),
        n_inspeccoes=agregado.get("n_inspeccoes", 0),
        n_criticos=agregado.get("n_criticos", 0),
        n_warnings=agregado.get("n_warnings", 0),
        n_ok=agregado.get("n_ok", 0),
        fill_rate_medio=f"{agregado.get('fill_rate_medio', 0):.0%}",
        tipos_frequentes=tipos_str
    )

    try:
        response = gemini_client.gerar_conteudo(
            contents=[prompt],
            temperature=0.1,
            response_mime_type="text/plain"
        )
        sumario = response.text.strip()
    except RuntimeError as e:
        logger.warning("[REPORT] API indisponível para sumário: %s", e)
        sumario = (
            f"Sessão de {data_hora}: {agregado.get('n_inspeccoes', 0)} inspecções "
            f"em {agregado.get('n_zonas', 0)} zona(s). "
            f"{agregado.get('n_criticos', 0)} crítico(s), "
            f"{agregado.get('n_warnings', 0)} warning(s). "
            f"Fill rate médio: {agregado.get('fill_rate_medio', 0):.0%}."
        )

    # Garante o limite de palavras
    sumario = _truncar_sumario(sumario, MAX_PALAVRAS_SUMARIO)
    n_palavras = _contar_palavras(sumario)

    return f"{sumario}\n\n*({n_palavras} palavras)*"


def _gerar_problemas_por_zona(agregado: dict, contexto_rag: dict) -> str:
    """Secção 2 — Problemas por zona com comparação histórica."""
    linhas = []
    por_zona = agregado.get("por_zona", {})

    if not por_zona:
        return "_Nenhuma zona inspeccionada nesta sessão._"

    for zona, inspeccoes in sorted(por_zona.items()):
        linhas.append(f"### {zona}")

        for insp in inspeccoes:
            status     = insp.get("overall_status", "?")
            fill       = insp.get("shelf_fill_rate", 0)
            insp_id    = insp.get("inspection_id", "?")
            issues     = insp.get("issues", [])
            ts         = insp.get("timestamp", "")

            # Emoji de status para leitura rápida
            emoji = {"ok": "✅", "warning": "⚠️", "critical": "🔴"}.get(status, "❓")

            linhas.append(
                f"\n**{emoji} {insp_id}** | {ts[:16].replace('T', ' ')} | "
                f"Fill rate: {fill:.0%} | Status: `{status}`"
            )

            if issues:
                linhas.append("\n| Issue | Localização | Severidade | Confiança |")
                linhas.append("|-------|------------|------------|-----------|")
                for issue in issues:
                    linhas.append(
                        f"| {issue.get('type', '?')} "
                        f"| {issue.get('location', '?')} "
                        f"| {issue.get('severity', '?')} "
                        f"| {issue.get('confidence', 0):.0%} |"
                    )
            else:
                linhas.append("\n_Sem issues detectados._")

            # Contexto histórico desta zona (do RAG)
            chunks_zona = [
                c for c in contexto_rag.get("chunks_recuperados", [])
                if c.get("metadados", {}).get("zone_id") == zona
                and c.get("metadados", {}).get("inspection_id") != insp_id
            ]
            if chunks_zona:
                melhor = chunks_zona[0]
                meta   = melhor.get("metadados", {})
                linhas.append(
                    f"\n> **Histórico:** Inspecção anterior `{meta.get('inspection_id', '?')}` "
                    f"em {meta.get('data', '?')} — status `{meta.get('status', '?')}`, "
                    f"fill rate {meta.get('fill_rate', 0):.0%}."
                )

        linhas.append("")  # linha em branco entre zonas

    return "\n".join(linhas)


def _gerar_regras_disparadas(alertas: list) -> str:
    """Secção 3 — Regras disparadas com dados concretos."""
    if not alertas:
        return "_Nenhuma regra foi activada nesta sessão._"

    linhas = []
    for alerta in alertas:
        nivel  = alerta.get("alert_level", "info").upper()
        emoji  = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(nivel, "❓")
        linhas.append(
            f"- {emoji} **{alerta.get('rule_id', '?')}** "
            f"[{nivel}] — Zona: `{alerta.get('zona', '?')}`  \n"
            f"  {alerta.get('message', '')}  \n"
            f"  *(Inspecção: `{alerta.get('inspection_id', '?')}` "
            f"| {alerta.get('timestamp', '')[:16].replace('T', ' ')})*"
        )

    return "\n".join(linhas)


def _gerar_contexto_historico(inspeccoes: list, agregado: dict) -> tuple:
    """
    Secção 4 — Contexto histórico com referências explícitas.
    Devolve (texto_markdown, chunks_recuperados) para reutilização na secção 2.
    """
    # Constrói query contextual baseada nos problemas da sessão
    zonas_com_issues = [
        zona for zona, insps in agregado.get("por_zona", {}).items()
        if any(i.get("overall_status") != "ok" for i in insps)
    ]

    if zonas_com_issues:
        query = (
            f"Histórico de problemas nas zonas {', '.join(zonas_com_issues)}: "
            f"ruturas de stock, produtos danificados, fill rate baixo."
        )
    else:
        query = "Padrões de problemas recentes em todas as zonas da loja."

    resultado_rag = query_memoria(query, k=5, sintetizar=True)

    chunks = resultado_rag.get("chunks_recuperados", [])
    ids_citados = resultado_rag.get("inspection_ids_citados", [])

    if not chunks:
        return "_Ainda não há histórico suficiente para identificar padrões._", resultado_rag

    # Texto sintetizado pela LLM
    texto_sintetizado = resultado_rag.get("resposta") or ""

    # Adiciona referências explícitas
    refs = []
    for c in chunks:
        meta = c.get("metadados", {})
        insp_id = meta.get("inspection_id", "")
        if insp_id:
            refs.append(
                f"`{insp_id}` ({meta.get('zone_id', '?')}, "
                f"{meta.get('data', '?')}, "
                f"status: {meta.get('status', '?')})"
            )

    texto_final = texto_sintetizado
    if refs:
        texto_final += "\n\n**Inspecções referenciadas:** " + " | ".join(refs)

    return texto_final, resultado_rag


def _gerar_recomendacoes(agregado: dict, alertas: list,
                          contexto_historico: str) -> str:
    """Secção 5 — Máx. 5 recomendações ordenadas por urgência."""
    # Resume os problemas para o prompt
    problemas = []
    for zona, insps in agregado.get("por_zona", {}).items():
        for insp in insps:
            if insp.get("overall_status") != "ok":
                for issue in insp.get("issues", []):
                    problemas.append(
                        f"Zona {zona}: {issue.get('type')} "
                        f"({issue.get('severity')}) — "
                        f"{issue.get('description', '')[:80]}"
                    )

    if not problemas and not alertas:
        return "_Sem recomendações — todas as prateleiras estão conformes._"

    alertas_resumo = "\n".join([
        f"- [{a.get('alert_level', '?').upper()}] {a.get('message', '')}"
        for a in alertas
    ]) or "Nenhum alerta activo."

    prompt = _PROMPT_RECOMENDACOES.format(
        max_rec=MAX_RECOMENDACOES,
        problemas_resumo="\n".join(problemas) or "Sem problemas críticos.",
        alertas_resumo=alertas_resumo,
        contexto_historico=contexto_historico[:500]
    )

    try:
        response = gemini_client.gerar_conteudo(
            contents=[prompt],
            temperature=0.1,
            response_mime_type="text/plain"
        )
        recomendacoes = response.text.strip()
    except RuntimeError as e:
        logger.warning("[REPORT] API indisponível para recomendações: %s", e)
        # Fallback: gera recomendações básicas sem LLM
        recomendacoes = _recomendacoes_fallback(problemas, alertas)

    return recomendacoes


def _recomendacoes_fallback(problemas: list, alertas: list) -> str:
    """Recomendações básicas geradas sem LLM."""
    recs = []
    criticos = [p for p in problemas if "high" in p]
    medios   = [p for p in problemas if "medium" in p]

    for i, p in enumerate(criticos[:2], 1):
        recs.append(f"{i}. [CRÍTICO] Resolver imediatamente: {p}")
    for i, p in enumerate(medios[:2], len(criticos[:2]) + 1):
        recs.append(f"{i}. [URGENTE] Verificar: {p}")
    for i, a in enumerate(alertas[:1], len(recs) + 1):
        recs.append(f"{i}. [NORMAL] {a.get('message', '')}")

    return "\n".join(recs) if recs else "1. [NORMAL] Manter monitorização regular."


# ==========================================
# 4. MONTAGEM DO RELATÓRIO COMPLETO
# ==========================================

def _montar_markdown(
    data_hora: str,
    session_id: str,
    sumario: str,
    problemas_zona: str,
    regras_disparadas: str,
    contexto_historico: str,
    recomendacoes: str
) -> str:
    """Monta o ficheiro Markdown final com as 6 secções obrigatórias."""
    return f"""# Relatório de Inspecção de Prateleiras
**Sessão:** `{session_id}`  
**Gerado em:** {data_hora}  
**Sistema:** Retail Vision Intelligence System

---

## 1. Sumário Executivo

{sumario}

---

## 2. Problemas por Zona

{problemas_zona}

---

## 3. Regras Disparadas

{regras_disparadas}

---

## 4. Contexto Histórico

{contexto_historico}

---

## 5. Recomendações

{recomendacoes}

---

## 6. Integração com Trajectória

> *Integração com dados de trajectória do Projeto 1 não implementada nesta versão.*  
> *Para activar: indexar dados de afluência por zona/período no RAG e correlacionar*  
> *com fill rate baixo para distinguir rutura por procura elevada vs falha de reposição.*

---
*Relatório gerado automaticamente pelo Retail Vision Intelligence System*
"""


# ==========================================
# 5. FUNÇÃO PRINCIPAL
# ==========================================

def gerar_relatorio(
    inspeccoes: list,
    session_id: str | None = None,
    guardar: bool = True
) -> dict:
    """
    Gera um relatório completo de uma sessão de inspecção.

    Args:
        inspeccoes: Lista de JSONs de inspecção (do shelf_inspector).
        session_id: ID da sessão. Se None, gerado automaticamente.
        guardar: Se True, guarda o ficheiro .md em reports/.

    Returns:
        Dicionário com:
          - session_id: str
          - caminho_ficheiro: str (None se guardar=False)
          - markdown: str (conteúdo completo)
          - alertas: list (alertas disparados)
          - n_palavras_sumario: int
    """
    if not inspeccoes:
        logger.warning("[REPORT] Nenhuma inspecção para gerar relatório.")
        return {
            "session_id": session_id,
            "caminho_ficheiro": None,
            "markdown": "# Relatório Vazio\n\nNenhuma inspecção nesta sessão.",
            "alertas": [],
            "n_palavras_sumario": 0
        }

    # --- Metadados da sessão ---
    agora      = datetime.now(timezone.utc)
    data_hora  = agora.strftime("%d/%m/%Y às %H:%M UTC")
    session_id = session_id or f"SESSION_{agora.strftime('%Y%m%d_%H%M%S')}"

    logger.info("[REPORT] A gerar relatório %s (%d inspecções)...",
                session_id, len(inspeccoes))

    # --- Agregação ---
    agregado = _agregar_sessao(inspeccoes)

    # --- Alertas de regras (executa para cada inspecção) ---
    todos_alertas = []
    for insp in inspeccoes:
        alertas_insp = executar_regras(insp)
        todos_alertas.extend(alertas_insp)

    # Remove alertas duplicados pelo mesmo rule_id + inspection_id
    vistos = set()
    alertas_unicos = []
    for a in todos_alertas:
        chave = (a.get("rule_id"), a.get("inspection_id"))
        if chave not in vistos:
            vistos.add(chave)
            alertas_unicos.append(a)

    # --- Contexto histórico (RAG) ---
    contexto_historico_texto, resultado_rag = _gerar_contexto_historico(
        inspeccoes, agregado
    )

    # --- Geração de cada secção ---
    sumario = _gerar_sumario_executivo(agregado, data_hora)

    problemas_zona = _gerar_problemas_por_zona(agregado, resultado_rag)

    regras_disparadas = _gerar_regras_disparadas(alertas_unicos)

    recomendacoes = _gerar_recomendacoes(
        agregado, alertas_unicos, contexto_historico_texto
    )

    # --- Montagem ---
    markdown = _montar_markdown(
        data_hora=data_hora,
        session_id=session_id,
        sumario=sumario,
        problemas_zona=problemas_zona,
        regras_disparadas=regras_disparadas,
        contexto_historico=contexto_historico_texto,
        recomendacoes=recomendacoes
    )

    # --- Persistência ---
    caminho = None
    if guardar:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        nome_ficheiro = f"report_{agora.strftime('%Y%m%d_%H%M%S')}.md"
        caminho = REPORTS_DIR / nome_ficheiro
        try:
            with open(caminho, "w", encoding="utf-8") as f:
                f.write(markdown)
            logger.info("[REPORT] Relatório guardado: %s", caminho)
        except IOError as e:
            logger.error("[REPORT] Erro ao guardar relatório: %s", e)
            caminho = None

    n_palavras = _contar_palavras(
        sumario.split("*(")[0].strip()  # exclui o contador de palavras
    )

    return {
        "session_id":        session_id,
        "caminho_ficheiro":  str(caminho) if caminho else None,
        "markdown":          markdown,
        "alertas":           alertas_unicos,
        "n_palavras_sumario": n_palavras
    }


def gerar_relatorio_de_dataset(
    caminho_dataset: str = "cache/vision/dataset_final.json",
    session_id: str | None = None
) -> dict:
    """
    Gera relatório directamente a partir do dataset_final.json
    produzido pelo batch_processor.
    Ponto de entrada conveniente para o comando `report --session today`.
    """
    path = Path(caminho_dataset)
    if not path.exists():
        logger.error("[REPORT] Dataset não encontrado: %s", caminho_dataset)
        return {"erro": f"Dataset não encontrado: {caminho_dataset}"}

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    inspeccoes = [
        entry["analise_ia"]
        for entry in dataset.get("inspecoes", {}).values()
        if entry.get("analise_ia") is not None
    ]

    if not inspeccoes:
        logger.warning("[REPORT] Nenhuma inspecção válida no dataset.")
        return {"erro": "Nenhuma inspecção válida no dataset."}

    return gerar_relatorio(inspeccoes, session_id=session_id)


# ==========================================
# 6. EXECUÇÃO DE TESTE
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
    print("   TESTE DO REPORT GENERATOR".center(65))
    print("=" * 65)

    # Inspecções sintéticas para teste
    inspeccoes_teste = [
        {
            "inspection_id": "INS_TEST_RPT_001",
            "timestamp": "2026-06-10T10:15:00Z",
            "image_path": "images/empty/teste1.jpg",
            "zone_id": "Z_EMPTY",
            "overall_status": "critical",
            "shelf_fill_rate": 0.32,
            "issues": [{
                "issue_id": "ISS_001",
                "type": "empty_shelf",
                "location": "prateleira inferior, lado esquerdo",
                "severity": "high",
                "description": "Rutura profunda — fundo da prateleira visível",
                "confidence": 0.91,
                "affected_area_pct": 0.60
            }],
            "products_detected": ["detergentes", "limpeza"],
            "model_reasoning": "Fill rate de 32% na zona Z_EMPTY. Rutura grave detectada."
        },
        {
            "inspection_id": "INS_TEST_RPT_002",
            "timestamp": "2026-06-10T10:20:00Z",
            "image_path": "images/normal/teste2.jpg",
            "zone_id": "Z_NORMAL",
            "overall_status": "ok",
            "shelf_fill_rate": 0.92,
            "issues": [],
            "products_detected": ["bolachas", "cereais", "snacks"],
            "model_reasoning": "Prateleira bem abastecida. Sem issues detectados."
        },
        {
            "inspection_id": "INS_TEST_RPT_003",
            "timestamp": "2026-06-10T10:25:00Z",
            "image_path": "images/dirty/teste3.jpg",
            "zone_id": "Z_DIRTY",
            "overall_status": "warning",
            "shelf_fill_rate": 0.71,
            "issues": [{
                "issue_id": "ISS_001",
                "type": "damaged",
                "location": "prateleira do meio",
                "severity": "medium",
                "description": "Embalagem amarfanhada visível",
                "confidence": 0.78,
                "affected_area_pct": 0.15
            }],
            "products_detected": ["iogurtes", "laticínios"],
            "model_reasoning": "Prateleira com embalagem danificada no meio."
        }
    ]

    print(f"\nA gerar relatório para {len(inspeccoes_teste)} inspecções...")
    resultado = gerar_relatorio(inspeccoes_teste, session_id="SESSION_TESTE")

    print(f"\n  Session ID     : {resultado['session_id']}")
    print(f"  Ficheiro       : {resultado['caminho_ficheiro']}")
    print(f"  Alertas        : {len(resultado['alertas'])}")
    print(f"  Palavras sumário: {resultado['n_palavras_sumario']} / {MAX_PALAVRAS_SUMARIO}")

    # Mostra as primeiras linhas do relatório
    print("\n--- PRÉVIA DO RELATÓRIO (primeiras 30 linhas) ---")
    linhas = resultado["markdown"].split("\n")
    for linha in linhas[:30]:
        print(linha)
    if len(linhas) > 30:
        print(f"  ... ({len(linhas) - 30} linhas adicionais)")

    print("\n[CONCLUÍDO]")