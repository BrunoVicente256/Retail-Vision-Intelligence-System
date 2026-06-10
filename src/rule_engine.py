"""
rule_engine.py
--------------
Componente 2 do Retail Vision Intelligence System.
Responsabilidades EXCLUSIVAS deste módulo:
  - Converter regras em linguagem natural para JSON estruturado (via LLM)
  - Detectar ambiguidades e pedir clarificação ANTES de guardar
  - Executar regras automaticamente após cada inspecção
  - Persistir regras em disco (data/rules/rules.json)
  - Gerar logs de execução por regra

Depende de:
  - utils/api_client.py  → comunicação com API (rate limit, retry, backoff)

NÃO depende de shelf_inspector.py — recebe o JSON de inspecção já processado.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from utils.api_client import gemini_client

# ==========================================
# CONFIGURAÇÃO
# ==========================================
logger = logging.getLogger(__name__)

RULES_PATH = Path("data/rules/rules.json")

ALERT_LEVELS_VALIDOS = {"info", "warning", "critical"}
SEVERITY_VALIDOS     = {"low", "medium", "high"}
LOCATION_VALIDOS     = {"bottom", "middle", "top", "any"}
ISSUE_TYPES_VALIDOS  = {
    "empty_shelf", "wrong_product", "damaged",
    "misaligned", "label_missing", "other"
}


# ==========================================
# 1. PERSISTÊNCIA EM DISCO
# ==========================================

def _carregar_regras() -> list:
    if not RULES_PATH.exists():
        return []
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
            return dados if isinstance(dados, list) else []
    except (json.JSONDecodeError, IOError) as e:
        logger.error("[RULES] Erro ao carregar regras: %s", e)
        return []


def _guardar_regras(regras: list) -> None:
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(regras, f, indent=4, ensure_ascii=False)


def _gerar_rule_id() -> str:
    regras = _carregar_regras()
    if not regras:
        return "RULE_001"
    numeros = []
    for r in regras:
        match = re.search(r"RULE_(\d+)", r.get("rule_id", ""))
        if match:
            numeros.append(int(match.group(1)))
    proximo = max(numeros) + 1 if numeros else 1
    return f"RULE_{str(proximo).zfill(3)}"


# ==========================================
# 2. PROMPTS PARA A LLM
# ==========================================

_PROMPT_CONVERTER_REGRA = """
Es um sistema especialista em processar regras de negocio para retalho.
A tua tarefa e converter uma regra em linguagem natural para um JSON estruturado.

REGRA DO GESTOR: "{texto_regra}"

CONTEXTO DO SISTEMA:
- Zonas: Z_NORMAL, Z_DIRTY, Z_EMPTY, Z_PLANOGRAM, Z_AMBIGUOUS (null para todas)
- Tipos de issue: empty_shelf, wrong_product, damaged, misaligned, label_missing, other
- Severidades: low, medium, high
- Niveis de alerta: info, warning, critical
- Localizacao: bottom, middle, top, any

INSTRUCOES:
1. Analisa e identifica TODAS as condicoes mencionadas
2. Para cada dimensao em falta ou ambigua, regista em "ambiguities"
3. Quando assumires algo, regista em "assumptions"
4. Se ha ambiguidades -> "is_valid": false
5. Se clara e completa -> "is_valid": true

DIMENSOES A VERIFICAR (todas as 4 devem estar claras para is_valid=true):
  A) THRESHOLD: fill_rate especificado? "vazia" = 0% ou abaixo de X%?
  B) SCOPE: todas as zonas ou especificas?
  C) URGENCIA: nivel de alerta? (info/warning/critical)
  D) TEMPO: sempre ou so em determinados horarios?

DEVOLVE EXCLUSIVAMENTE este JSON valido:
{{
    "rule_id": "{rule_id}",
    "created_at": "{created_at}",
    "natural_language": "{texto_regra}",
    "description": "<reformulacao clara em portugues formal>",
    "conditions": {{
        "zone_filter": <lista de zone_ids ou null para todas>,
        "time_filter": <{{"hours_start": int, "hours_end": int}} ou null>,
        "issue_types": <lista de tipos ou null para qualquer>,
        "severity_threshold": <"low"|"medium"|"high" ou null>,
        "fill_rate_threshold": <float 0.0-1.0 ou null>,
        "location_filter": <"bottom"|"middle"|"top"|"any">
    }},
    "action": {{
        "alert_level": <"info"|"warning"|"critical">,
        "notification_message": "<template com {{zona}}, {{fill_rate}}, {{issues}} como placeholders>"
    }},
    "validation": {{
        "is_valid": <true ou false>,
        "ambiguities": [<lista de strings com cada ambiguidade>],
        "assumptions": [<lista de strings com cada pressuposto>]
    }}
}}
"""

_PROMPT_RESOLVER_AMBIGUIDADE = """
Es um sistema especialista em regras de negocio para retalho.
Uma regra foi criada com ambiguidades por resolver.

REGRA ORIGINAL: "{texto_regra}"

REGRA ACTUAL (JSON):
{regra_actual}

RESPOSTAS DO GESTOR:
{respostas}

Com base nas respostas:
1. Preenche os campos indefinidos
2. Remove ambiguidades resolvidas
3. Adiciona as clarificacoes a "assumptions"
4. Se todas resolvidas -> "is_valid": true
5. Se ainda restam -> "is_valid": false

DEVOLVE EXCLUSIVAMENTE o JSON completo e actualizado.
"""


# ==========================================
# 3. PARSING E NORMALIZACAO
# ==========================================

def _parsear_regra_json(texto: str):
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    texto_limpo = re.sub(r'```(?:json)?\s*', '', texto).strip()
    try:
        return json.loads(texto_limpo)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.error("[RULE ENGINE] Nao foi possivel parsear JSON da resposta.")
    return None


def _normalizar_regra(regra: dict) -> dict:
    if not isinstance(regra.get("conditions"), dict):
        regra["conditions"] = {}
    cond = regra["conditions"]
    cond.setdefault("zone_filter", None)
    cond.setdefault("time_filter", None)
    cond.setdefault("issue_types", None)
    cond.setdefault("severity_threshold", None)
    cond.setdefault("fill_rate_threshold", None)
    cond.setdefault("location_filter", "any")

    loc = str(cond.get("location_filter", "any")).lower()
    cond["location_filter"] = loc if loc in LOCATION_VALIDOS else "any"

    sev = cond.get("severity_threshold")
    if sev and str(sev).lower() not in SEVERITY_VALIDOS:
        cond["severity_threshold"] = None

    frt = cond.get("fill_rate_threshold")
    if frt is not None:
        try:
            cond["fill_rate_threshold"] = max(0.0, min(1.0, float(frt)))
        except (TypeError, ValueError):
            cond["fill_rate_threshold"] = None

    if not isinstance(regra.get("action"), dict):
        regra["action"] = {}
    action = regra["action"]
    alert = str(action.get("alert_level", "info")).lower()
    action["alert_level"] = alert if alert in ALERT_LEVELS_VALIDOS else "info"
    action.setdefault("notification_message", "Regra {rule_id} disparou na zona {zona}.")

    if not isinstance(regra.get("validation"), dict):
        regra["validation"] = {}
    val = regra["validation"]
    val.setdefault("is_valid", False)
    val.setdefault("ambiguities", [])
    val.setdefault("assumptions", [])

    regra.setdefault("description", regra.get("natural_language", ""))
    return regra


# ==========================================
# 4. CRIACAO DE REGRAS (MODO INTERACTIVO)
# ==========================================

def criar_regra(texto_natural: str) -> dict:
    """
    Converte texto natural em regra JSON.
    Se is_valid=True -> guarda em disco imediatamente.
    Se is_valid=False -> devolve sem guardar (ha ambiguidades).
    """
    rule_id    = _gerar_rule_id()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    prompt = _PROMPT_CONVERTER_REGRA.format(
        texto_regra=texto_natural,
        rule_id=rule_id,
        created_at=created_at
    )

    logger.info("[RULE ENGINE] A converter regra: '%s'", texto_natural[:60])

    try:
        response = gemini_client.gerar_conteudo(
            contents=[prompt],
            temperature=0.0,
            response_mime_type="application/json"
        )
    except RuntimeError as e:
        logger.error("[RULE ENGINE] Falha na API: %s", e)
        return _regra_erro(rule_id, created_at, texto_natural, str(e))

    regra = _parsear_regra_json(response.text)
    if regra is None:
        return _regra_erro(rule_id, created_at, texto_natural,
                           "Nao foi possivel processar. Tenta reformular a regra.")

    regra = _normalizar_regra(regra)
    regra["rule_id"]          = rule_id
    regra["created_at"]       = created_at
    regra["natural_language"] = texto_natural

    if regra["validation"]["is_valid"]:
        regras = _carregar_regras()
        regras.append(regra)
        _guardar_regras(regras)
        logger.info("[RULE ENGINE] Regra %s guardada.", rule_id)
    else:
        n = len(regra["validation"]["ambiguities"])
        logger.info("[RULE ENGINE] Regra %s tem %d ambiguidade(s).", rule_id, n)

    return regra


def _regra_erro(rule_id: str, created_at: str,
                texto: str, motivo: str) -> dict:
    """Devolve uma regra inválida com o motivo do erro como ambiguidade."""
    return {
        "rule_id": rule_id,
        "created_at": created_at,
        "natural_language": texto,
        "description": texto,
        "conditions": {},
        "action": {"alert_level": "info", "notification_message": ""},
        "validation": {
            "is_valid": False,
            "ambiguities": [motivo],
            "assumptions": []
        }
    }


def responder_ambiguidade(regra_pendente: dict, respostas: str) -> dict:
    """
    Processa respostas do gestor a ambiguidades de uma regra pendente.
    Se todas resolvidas -> guarda em disco e devolve is_valid=True.
    """
    prompt = _PROMPT_RESOLVER_AMBIGUIDADE.format(
        texto_regra=regra_pendente.get("natural_language", ""),
        regra_actual=json.dumps(regra_pendente, ensure_ascii=False, indent=2),
        respostas=respostas
    )

    logger.info("[RULE ENGINE] A resolver ambiguidades de %s...",
                regra_pendente.get("rule_id"))

    try:
        response = gemini_client.gerar_conteudo(
            contents=[prompt],
            temperature=0.0,
            response_mime_type="application/json"
        )
    except RuntimeError as e:
        logger.error("[RULE ENGINE] Falha na API: %s", e)
        return regra_pendente

    regra = _parsear_regra_json(response.text)
    if regra is None:
        return regra_pendente

    regra = _normalizar_regra(regra)
    # Preserva identidade original
    regra["rule_id"]          = regra_pendente["rule_id"]
    regra["created_at"]       = regra_pendente["created_at"]
    regra["natural_language"] = regra_pendente["natural_language"]

    if regra["validation"]["is_valid"]:
        regras = _carregar_regras()
        regras = [r for r in regras if r.get("rule_id") != regra["rule_id"]]
        regras.append(regra)
        _guardar_regras(regras)
        logger.info("[RULE ENGINE] Regra %s clarificada e guardada.", regra["rule_id"])

    return regra


# ==========================================
# 5. EXECUCAO DE REGRAS
# ==========================================

def _avaliar_condicoes(regra: dict, inspeccao: dict) -> bool:
    """
    Avalia se UMA regra dispara para UMA inspeccao.
    Logica AND: TODAS as condicoes especificadas devem ser satisfeitas.
    """
    cond = regra.get("conditions", {})

    # Filtro de zona
    zone_filter = cond.get("zone_filter")
    if zone_filter:
        if inspeccao.get("zone_id", "") not in zone_filter:
            return False

    # Filtro de horario
    time_filter = cond.get("time_filter")
    if time_filter:
        hora = datetime.now().hour
        if not (time_filter.get("hours_start", 0) <= hora
                <= time_filter.get("hours_end", 23)):
            return False

    # Filtro de fill_rate (dispara se fill_rate ABAIXO do threshold)
    frt = cond.get("fill_rate_threshold")
    if frt is not None:
        if inspeccao.get("shelf_fill_rate", 1.0) >= frt:
            return False

    # Filtro de tipos de issue (dispara se ALGUM tipo esta presente)
    issue_types = cond.get("issue_types")
    if issue_types:
        tipos = {i.get("type") for i in inspeccao.get("issues", [])}
        if not tipos.intersection(set(issue_types)):
            return False

    # Filtro de severidade (dispara se ALGUM issue tem severidade >= threshold)
    sev_threshold = cond.get("severity_threshold")
    if sev_threshold:
        ordem = {"low": 0, "medium": 1, "high": 2}
        nivel_min = ordem.get(sev_threshold, 0)
        severidades = [
            ordem.get(i.get("severity", "low"), 0)
            for i in inspeccao.get("issues", [])
        ]
        if not any(s >= nivel_min for s in severidades):
            return False

    # Filtro de localizacao
    loc = cond.get("location_filter", "any")
    if loc and loc != "any":
        localizacoes = [
            i.get("location", "").lower()
            for i in inspeccao.get("issues", [])
        ]
        if not any(loc in l for l in localizacoes):
            return False

    return True


def _gerar_notificacao(regra: dict, inspeccao: dict) -> str:
    template = regra.get("action", {}).get(
        "notification_message",
        "Regra {rule_id} disparou na zona {zona}."
    )
    issues_str = "; ".join([
        f"{i.get('type')} ({i.get('severity')})"
        for i in inspeccao.get("issues", [])
    ]) or "nenhum issue especifico"

    try:
        return template.format(
            rule_id=regra.get("rule_id", ""),
            zona=inspeccao.get("zone_id", "desconhecida"),
            fill_rate=f"{inspeccao.get('shelf_fill_rate', 0):.0%}",
            issues=issues_str,
            status=inspeccao.get("overall_status", ""),
            timestamp=inspeccao.get("timestamp", "")
        )
    except KeyError:
        # Template com placeholders desconhecidos -- devolve mensagem segura
        return (f"[{regra.get('action', {}).get('alert_level', 'info').upper()}] "
                f"{regra.get('rule_id')} disparou na zona "
                f"{inspeccao.get('zone_id', '?')}.")


def executar_regras(inspeccao: dict) -> list:
    """
    Executa todas as regras validas contra uma inspeccao.
    Chamado automaticamente apos cada inspeccao pelo pipeline.

    Returns:
        Lista de alertas gerados (vazia se nenhuma regra disparar).
    """
    regras = _carregar_regras()
    validas = [r for r in regras if r.get("validation", {}).get("is_valid")]

    if not validas:
        logger.info("[RULE ENGINE] Nenhuma regra valida para executar.")
        return []

    alertas = []
    logs    = []

    for regra in validas:
        rule_id  = regra.get("rule_id", "?")
        disparou = _avaliar_condicoes(regra, inspeccao)
        ts       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        log = {
            "rule_id": rule_id,
            "inspection_id": inspeccao.get("inspection_id"),
            "zona": inspeccao.get("zone_id"),
            "disparou": disparou,
            "timestamp": ts
        }

        if disparou:
            mensagem    = _gerar_notificacao(regra, inspeccao)
            alert_level = regra.get("action", {}).get("alert_level", "info")
            alertas.append({
                "rule_id": rule_id,
                "alert_level": alert_level,
                "message": mensagem,
                "zona": inspeccao.get("zone_id"),
                "inspection_id": inspeccao.get("inspection_id"),
                "timestamp": ts
            })
            log["mensagem"] = mensagem
            logger.warning("[ALERTA %s] %s | %s | %s",
                           alert_level.upper(), rule_id,
                           inspeccao.get("zone_id"), mensagem)
        else:
            logger.info("[RULE ENGINE] %s nao disparou para %s",
                        rule_id, inspeccao.get("inspection_id"))

        logs.append(log)

    _guardar_log_execucao(logs, inspeccao.get("inspection_id", "unknown"))
    return alertas


def _guardar_log_execucao(logs: list, inspection_id: str) -> None:
    log_path = Path("data/rules/execution_logs.json")
    historico = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                historico = json.load(f)
        except (json.JSONDecodeError, IOError):
            historico = []

    historico.append({
        "inspection_id": inspection_id,
        "executado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regras_avaliadas": len(logs),
        "regras_disparadas": sum(1 for l in logs if l["disparou"]),
        "detalhes": logs
    })

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(historico, f, indent=4, ensure_ascii=False)


# ==========================================
# 6. CRUD DE REGRAS
# ==========================================

def listar_regras() -> list:
    """Devolve todas as regras guardadas em disco."""
    return _carregar_regras()


def eliminar_regra(rule_id: str) -> bool:
    """Remove uma regra pelo ID. Retorna True se encontrada e removida."""
    regras = _carregar_regras()
    filtradas = [r for r in regras if r.get("rule_id") != rule_id]
    if len(filtradas) == len(regras):
        logger.warning("[RULE ENGINE] Regra %s nao encontrada.", rule_id)
        return False
    _guardar_regras(filtradas)
    logger.info("[RULE ENGINE] Regra %s eliminada.", rule_id)
    return True


def testar_regra(rule_id: str, inspeccao: dict) -> dict:
    """
    Dry-run de uma regra contra uma inspeccao.
    NAO gera alertas reais nem logs de execucao.
    """
    regras = _carregar_regras()
    regra  = next((r for r in regras if r.get("rule_id") == rule_id), None)

    if regra is None:
        return {"erro": f"Regra {rule_id} nao encontrada."}

    if not regra.get("validation", {}).get("is_valid"):
        return {
            "rule_id": rule_id,
            "dispararia": False,
            "motivo": "Regra invalida (ambiguidades por resolver).",
            "ambiguidades": regra.get("validation", {}).get("ambiguities", [])
        }

    disparou = _avaliar_condicoes(regra, inspeccao)
    resultado = {
        "rule_id": rule_id,
        "dispararia": disparou,
        "zona_testada": inspeccao.get("zone_id"),
        "fill_rate": inspeccao.get("shelf_fill_rate"),
        "issues_presentes": [i.get("type") for i in inspeccao.get("issues", [])]
    }
    if disparou:
        resultado["mensagem_que_geraria"] = _gerar_notificacao(regra, inspeccao)
    return resultado


def obter_regra(rule_id: str):
    """Devolve uma regra pelo ID ou None se nao existir."""
    return next((r for r in _carregar_regras()
                 if r.get("rule_id") == rule_id), None)


# ==========================================
# 7. TESTE
# ==========================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.console import configurar_consola
    configurar_consola()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 60)
    print("   TESTE DO RULE ENGINE".center(60))
    print("=" * 60)

    # Teste 1: Regra clara
    print("\n[TESTE 1] Regra clara e completa")
    r1 = criar_regra(
        "Se o fill rate da zona Z_EMPTY estiver abaixo de 50%, "
        "gera um alerta critico imediatamente."
    )
    print(f"  is_valid    : {r1['validation']['is_valid']}")
    print(f"  rule_id     : {r1['rule_id']}")
    print(f"  description : {r1.get('description', '')[:80]}")
    if r1["validation"]["ambiguities"]:
        print(f"  ambiguidades: {r1['validation']['ambiguities']}")

    print("\n" + "-" * 60)

    # Teste 2: Regra ambigua
    print("\n[TESTE 2] Regra ambigua")
    r2 = criar_regra("Avisa-me quando a prateleira estiver vazia.")
    print(f"  is_valid     : {r2['validation']['is_valid']}")
    print("  ambiguidades :")
    for a in r2["validation"]["ambiguities"]:
        print(f"    - {a}")

    if not r2["validation"]["is_valid"]:
        print("\n  [Gestor responde...]")
        r2b = responder_ambiguidade(
            r2,
            "Vazia = fill rate abaixo de 20%. "
            "Aplica-se a todas as zonas. "
            "Nivel: warning. Sempre, sem restricao de horario."
        )
        print(f"  is_valid apos resolucao: {r2b['validation']['is_valid']}")

    print("\n" + "-" * 60)

    # Teste 3: Execucao sobre inspeccao sintetica
    print("\n[TESTE 3] Execucao de regras")
    inspeccao_teste = {
        "inspection_id": "INS_TEST_001",
        "zone_id": "Z_EMPTY",
        "overall_status": "critical",
        "shelf_fill_rate": 0.35,
        "issues": [{
            "issue_id": "ISS_001",
            "type": "empty_shelf",
            "severity": "high",
            "location": "prateleira inferior",
            "description": "Rutura de stock",
            "confidence": 0.92,
            "affected_area_pct": 0.65
        }],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    alertas = executar_regras(inspeccao_teste)
    print(f"  Alertas gerados: {len(alertas)}")
    for a in alertas:
        print(f"  [{a['alert_level'].upper()}] {a['rule_id']}: {a['message']}")

    print("\n" + "-" * 60)

    # Teste 4: Listagem
    print("\n[TESTE 4] Regras em disco")
    for r in listar_regras():
        s = "OK" if r["validation"]["is_valid"] else "PENDENTE"
        print(f"  [{s}] {r['rule_id']}: {r.get('description', '')[:60]}")

    print("\n[CONCLUIDO]")