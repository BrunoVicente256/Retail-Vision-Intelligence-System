"""
shelf_inspector.py
------------------
Componente 1 do Retail Vision Intelligence System.
Responsabilidades EXCLUSIVAS deste módulo:
  - Gerir as 3 estratégias de prompting (zero_shot, cot, few_shot)
  - Cache local por hash MD5 (imagem + estratégia)
  - Validação e parsing defensivo do JSON de resposta
  - Derivar zone_id automaticamente a partir da pasta da imagem

Depende de:
  - utils/api_client.py  → comunicação com API (rate limit, retry, backoff)
"""

import os
import hashlib
import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Importa o singleton partilhado — rate limiter global à sessão
from utils.api_client import gemini_client, MODEL_NAME

# ==========================================
# CONFIGURAÇÃO
# ==========================================
logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/vision")

# Mapeamento de nome de pasta → zone_id legível
# Adiciona aqui novas categorias se o dataset crescer
CATEGORIA_PARA_ZONE_ID = {
    "normal":    "Z_NORMAL",
    "dirty":     "Z_DIRTY",
    "empty":     "Z_EMPTY",
    "planogram": "Z_PLANOGRAM",
    "ambiguous": "Z_AMBIGUOUS",
}
ZONE_ID_DEFAULT = "Z_UNKNOWN"


# ==========================================
# 1. UTILITÁRIOS DE CACHE
# ==========================================

def _gerar_hash_cache(caminho_imagem: str, tipo_prompt: str, modelo: str) -> str:
    """
    Gera um hash MD5 determinístico baseado em:
      - Conteúdo binário do ficheiro (não o nome — o nome pode mudar)
      - Estratégia de prompting usada
      - Modelo usado (ex: flash vs flash-lite)

    O modelo FAZ PARTE da chave: trocar de tier (ou comparar tiers na
    mesma sessão) tem de produzir entradas de cache distintas, senão um
    tier serviria silenciosamente os resultados do outro.

    Dois ficheiros com o mesmo conteúdo mas nomes diferentes produzem o
    MESMO hash — comportamento correto para evitar chamadas duplicadas.
    """
    hasher = hashlib.md5()
    with open(caminho_imagem, 'rb') as f:
        hasher.update(f.read())
    hasher.update(tipo_prompt.encode('utf-8'))
    hasher.update(modelo.encode('utf-8'))
    return hasher.hexdigest()


def _ler_cache(hash_id: str) -> dict | None:
    """Tenta ler um resultado do cache. Retorna None se não existir."""
    caminho = CACHE_DIR / f"{hash_id}.json"
    if caminho.exists():
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("[CACHE] Ficheiro corrompido, ignorado: %s | %s", caminho, e)
            return None
    return None


def _escrever_cache(hash_id: str, dados: dict) -> None:
    """Persiste o resultado em disco."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    caminho = CACHE_DIR / f"{hash_id}.json"
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)


def esta_em_cache(caminho_imagem: str, tipo_prompt: str = "cot",
                  modelo: str | None = None) -> bool:
    """
    Indica se já existe resultado em cache para (imagem, estratégia, modelo),
    SEM chamar a API e SEM efeitos colaterais.

    Usado por orquestradores (ex: batch_processor) para saber de antemão
    se uma chamada REAL à API vai acontecer — e só nesse caso aplicar
    pausas de rate limiting. Reutiliza _gerar_hash_cache como fonte única
    de verdade para o hashing, evitando duplicar a lógica.
    """
    if not os.path.exists(caminho_imagem):
        return False
    modelo = modelo or MODEL_NAME
    hash_id = _gerar_hash_cache(caminho_imagem, tipo_prompt, modelo)
    return (CACHE_DIR / f"{hash_id}.json").exists()


# ==========================================
# 2. DERIVAÇÃO DE METADADOS
# ==========================================

def _derivar_zone_id(caminho_imagem: str) -> str:
    """
    Extrai o zone_id a partir do nome da pasta imediatamente acima da imagem.
    Ex: images/dirty/foto.jpg  →  Z_DIRTY
        images/normal/foto.jpg →  Z_NORMAL
        qualquer_outra_pasta/  →  Z_UNKNOWN
    """
    pasta = Path(caminho_imagem).parent.name.lower()
    return CATEGORIA_PARA_ZONE_ID.get(pasta, ZONE_ID_DEFAULT)


def _gerar_inspection_id(hash_id: str) -> str:
    """
    Gera um inspection_id determinístico e único.
    Usa os primeiros 8 chars do hash para garantir que o mesmo ficheiro
    + estratégia produz sempre o mesmo ID (importante para o RAG).
    Formato: INS_YYYYMMDD_HHMMSS_XXXXXXXX
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sufixo = hash_id[:8].upper()
    return f"INS_{timestamp}_{sufixo}"


# ==========================================
# 3. PARSING E VALIDAÇÃO DO JSON
# ==========================================

# Campos obrigatórios no JSON de resposta (validação mínima)
CAMPOS_OBRIGATORIOS = {
    "inspection_id", "timestamp", "image_path", "zone_id",
    "overall_status", "issues", "shelf_fill_rate",
    "products_detected", "model_reasoning"
}

OVERALL_STATUS_VALIDOS = {"ok", "warning", "critical"}
ISSUE_TYPES_VALIDOS = {
    "empty_shelf", "wrong_product", "damaged",
    "misaligned", "label_missing", "other"
}
SEVERITY_VALIDOS = {"low", "medium", "high"}


def _parsear_resposta(texto_resposta: str) -> dict | None:
    """
    Parsing defensivo da resposta da API.
    Tenta múltiplas estratégias de limpeza antes de desistir.
    """
    # Tentativa 1: parsing direto
    try:
        return json.loads(texto_resposta)
    except json.JSONDecodeError:
        pass

    # Tentativa 2: remover blocos de código Markdown (```json ... ```)
    texto_limpo = re.sub(r'```(?:json)?\s*', '', texto_resposta).strip()
    try:
        return json.loads(texto_limpo)
    except json.JSONDecodeError:
        pass

    # Tentativa 3: extrair o primeiro objeto JSON encontrado no texto
    match = re.search(r'\{.*\}', texto_resposta, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.error("[PARSING] Não foi possível extrair JSON válido da resposta.")
    logger.debug("[PARSING] Resposta raw: %s", texto_resposta[:500])
    return None


def _validar_e_normalizar(dados: dict, caminho_imagem: str,
                           inspection_id: str, zone_id: str,
                           timestamp: str) -> dict:
    """
    Garante que o JSON tem todos os campos obrigatórios e valores válidos.
    Preenche campos em falta com valores seguros em vez de crashar.
    Normaliza overall_status e severidades para os valores permitidos.
    """
    # Campos de identidade — são sempre os do sistema, não do modelo
    dados["inspection_id"] = inspection_id
    dados["image_path"] = caminho_imagem
    dados["zone_id"] = zone_id
    dados["timestamp"] = timestamp

    # overall_status: normaliza para minúsculas e valida
    status_raw = str(dados.get("overall_status", "")).lower().strip()
    dados["overall_status"] = status_raw if status_raw in OVERALL_STATUS_VALIDOS else "warning"

    # shelf_fill_rate: garante float entre 0 e 1
    try:
        fill = float(dados.get("shelf_fill_rate", 0.0))
        dados["shelf_fill_rate"] = max(0.0, min(1.0, fill))
    except (TypeError, ValueError):
        dados["shelf_fill_rate"] = 0.0

    # products_detected: garante lista
    if not isinstance(dados.get("products_detected"), list):
        dados["products_detected"] = []

    # model_reasoning: garante string não-vazia
    if not dados.get("model_reasoning"):
        dados["model_reasoning"] = "[model_reasoning não gerado pelo modelo]"

    # issues: valida e normaliza cada issue
    issues_raw = dados.get("issues", [])
    if not isinstance(issues_raw, list):
        issues_raw = []

    issues_normalizados = []
    for i, issue in enumerate(issues_raw):
        if not isinstance(issue, dict):
            continue

        # issue_id sequencial se não existir
        if not issue.get("issue_id"):
            issue["issue_id"] = f"ISS_{str(i+1).zfill(3)}"

        # type: normaliza
        tipo_raw = str(issue.get("type", "other")).lower().strip()
        issue["type"] = tipo_raw if tipo_raw in ISSUE_TYPES_VALIDOS else "other"

        # severity: normaliza
        sev_raw = str(issue.get("severity", "low")).lower().strip()
        issue["severity"] = sev_raw if sev_raw in SEVERITY_VALIDOS else "low"

        # confidence e affected_area_pct: garante floats em [0, 1]
        for campo in ("confidence", "affected_area_pct"):
            try:
                val = float(issue.get(campo, 0.0))
                issue[campo] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                issue[campo] = 0.0

        # location e description: garante strings
        issue["location"] = str(issue.get("location", "não especificado"))
        issue["description"] = str(issue.get("description", "sem descrição"))

        issues_normalizados.append(issue)

    dados["issues"] = issues_normalizados
    return dados


# ==========================================
# 4. SCHEMA BASE INJETADO NO PROMPT
# ==========================================

def _get_schema_instrucao(inspection_id: str, timestamp: str,
                           image_path: str, zone_id: str) -> str:
    """
    Schema JSON com instruções claras em vez de placeholders ambíguos.
    Os pipes (ok|warning|critical) foram substituídos por instruções
    explícitas para evitar que o modelo os copie literalmente.
    """
    return f"""
DEVOLVE EXCLUSIVAMENTE um objeto JSON válido, sem blocos de código Markdown, sem texto antes ou depois.

{{
    "inspection_id": "{inspection_id}",
    "timestamp": "{timestamp}",
    "image_path": "{image_path}",
    "zone_id": "{zone_id}",
    "overall_status": "<escolhe exatamente uma das opções: ok, warning, critical>",
    "issues": [
        {{
            "issue_id": "ISS_001",
            "type": "<escolhe: empty_shelf, wrong_product, damaged, misaligned, label_missing, other>",
            "location": "<descrição da localização na prateleira, ex: prateleira inferior, lado esquerdo>",
            "severity": "<escolhe: low, medium, high>",
            "description": "<descrição factual do problema observado na imagem>",
            "confidence": <float entre 0.0 e 1.0 representando a tua certeza neste issue>,
            "affected_area_pct": <float entre 0.0 e 1.0 da percentagem da prateleira afetada>
        }}
    ],
    "shelf_fill_rate": <float entre 0.0 e 1.0 estimando a percentagem da prateleira preenchida com produto>,
    "products_detected": ["<lista de categorias de produto visíveis, ex: bebidas, laticínios, snacks>"],
    "model_reasoning": "<cadeia de raciocínio explícita que justifica todas as classificações acima>"
}}

REGRAS CRÍTICAS:
- Se não houver issues, devolve "issues": []
- O campo "model_reasoning" é OBRIGATÓRIO e deve ter pelo menos 3 frases explicando o teu raciocínio
- Não inventes produtos ou problemas que não sejam claramente visíveis na imagem

ESTIMATIVA DE FILL RATE (não uses valores por defeito):
- NÃO assumas 0.95 nem qualquer valor "típico". Estima a partir da imagem REAL.
- Olha prateleira a prateleira: que fração de cada uma tem produto vs. espaço vazio? Faz a média.
- "shelf_fill_rate" de 1.0 = 100% cheia; 0.0 = completamente vazia.

DETEÇÃO DE PRATELEIRA VAZIA (o recall é prioritário):
- Qualquer ponto onde se veja o FUNDO ou o SUPORTE da prateleira, um slot vazio, ou uma
  lacuna profunda conta como "empty_shelf" — MESMO que o resto da prateleira esteja cheio.
- Examina ativamente o topo, o fundo, os cantos e o interior das caixas de exposição.
- Faltar 1-2 unidades só na FRENTE (com produto visível atrás) NÃO é issue. Mas em caso de
  dúvida entre "frente em falta" e "rutura", se vês o fundo da prateleira → classifica como empty_shelf.

CALIBRAÇÃO DE SEVERIDADE (evita falsos positivos):
- Produto presente mas apenas ligeiramente torto NÃO é issue — ignora desalinhamentos cosméticos.
- Só reporta "misaligned" se o produto estiver tombado, a bloquear outros, ou claramente fora do sítio.
- "overall_status" só é "warning"/"critical" se houver issues REAIS; prateleira cheia e organizada é "ok".
"""


# ==========================================
# 5. AS 3 ESTRATÉGIAS DE PROMPTING
# ==========================================

def _prompt_zero_shot(inspection_id: str, timestamp: str,
                      image_path: str, zone_id: str) -> str:
    """
    ESTRATÉGIA A — Zero-Shot Direto.
    Instrução direta sem exemplos nem guia de raciocínio.
    Testa a capacidade inata do modelo para a tarefa.
    """
    instrucao = (
        "És um sistema especialista de Visão Computacional para Retalho. "
        "Analisa a imagem submetida e classifica o estado da prateleira. "
        "Sê factual: só reporta o que é claramente visível na imagem.\n\n"
    )
    return instrucao + _get_schema_instrucao(inspection_id, timestamp, image_path, zone_id)


def _prompt_cot(inspection_id: str, timestamp: str,
                image_path: str, zone_id: str) -> str:
    """
    ESTRATÉGIA B — Chain-of-Thought Visual.
    Força o modelo a raciocinar região a região antes de classificar.
    Produz o model_reasoning mais rico e verificável.
    """
    instrucao = """És um sistema especialista de Visão Computacional para Retalho.
Analisa a imagem seguindo OBRIGATORIAMENTE este processo passo a passo.
Regista todo este raciocínio no campo "model_reasoning" do JSON final.

PASSO 1 — INVENTÁRIO DE PRODUTOS:
Que tipos de produtos estão presentes? Estão organizados por categoria lógica?

PASSO 2 — AVALIAÇÃO DE PREENCHIMENTO:
Estima a percentagem de espaço ocupado por produto (shelf_fill_rate).
Distingue: faltar unidades na frente (reposição normal, não é issue)
vs. buracos profundos até ao fundo da prateleira (rutura de stock, é issue).

PASSO 3 — DETEÇÃO DE ANOMALIAS (analisa cada região da prateleira):
Procura ativamente por:
  - RUTURA (empty_shelf): espaço vazio profundo, etiqueta sem produto
  - DANO/SUJIDADE (damaged): embalagens amarfanhadas, rasgadas, tombadas, lixo
  - VIOLAÇÃO (wrong_product): produto fora da categoria lógica da prateleira
  - DESALINHAMENTO (misaligned): produto ligeiramente fora de posição
  - ETIQUETA AUSENTE (label_missing): espaço sem etiqueta de preço/produto

PASSO 4 — CLASSIFICAÇÃO DE RISCO:
  - 0 anomalias → overall_status: "ok"
  - 1 a 2 anomalias menores → overall_status: "warning"
  - Rutura grave, dano extenso, ou múltiplas anomalias → overall_status: "critical"

Após completar os 4 passos, gera o JSON estruturado.\n\n"""
    return instrucao + _get_schema_instrucao(inspection_id, timestamp, image_path, zone_id)


def _prompt_few_shot(inspection_id: str, timestamp: str,
                     image_path: str, zone_id: str) -> str:
    """
    ESTRATÉGIA C — Few-Shot com Exemplos Textuais.
    Usa descrições textuais de análises anteriores como calibração.
    Não passa imagens adicionais (poupa quota de API).
    """
    instrucao = """És um sistema especialista de Visão Computacional para Retalho.
Usa os exemplos abaixo como padrão de calibração para a tua análise.

━━━ EXEMPLO 1: PRATELEIRA SAUDÁVEL ━━━
Situação: Prateleira bem organizada, produtos alinhados, sem buracos visíveis.
Pode faltar 1-2 unidades na frente (reposição normal).
Resultado esperado:
  - overall_status: "ok"
  - issues: []
  - shelf_fill_rate: ~0.90 a 1.0
  - model_reasoning: "Prateleira com bom nível de stock. Produtos organizados por categoria.
    Não foram detetadas ruturas, danos ou violações. Pequenas lacunas na frente são normais
    e não constituem issue. Fill rate estimado em 92%."

━━━ EXEMPLO 2: RUTURA DE STOCK ━━━
Situação: Buraco profundo visível (vê-se o fundo ou o suporte metálico da prateleira).
Resultado esperado:
  - overall_status: "critical" (se grande) ou "warning" (se pequeno)
  - issues: [{{ type: "empty_shelf", severity: "high", confidence: 0.9 }}]
  - shelf_fill_rate: ~0.40 a 0.65
  - model_reasoning: "Detetado buraco profundo na secção central. O fundo da prateleira
    está visível, indicando rutura de stock e não apenas falta de frente de produto.
    Área afetada estimada em 35%. Classificado como critical pela extensão da rutura."

━━━ EXEMPLO 3: VIOLAÇÃO DE PLANOGRAMA ━━━
Situação: Produto visivelmente fora da categoria lógica da prateleira.
Resultado esperado:
  - overall_status: "warning"
  - issues: [{{ type: "wrong_product", severity: "medium", confidence: 0.75 }}]
  - shelf_fill_rate: ~0.85
  - model_reasoning: "Prateleira de snacks com um produto de limpeza mal colocado
    na extremidade direita. Produto fora de categoria identificado por diferença
    visual clara de embalagem. Restante prateleira conforme."

━━━ EXEMPLO 4: PRATELEIRA CAÓTICA / DANIFICADA ━━━
Situação: Embalagens amarfanhadas, rasgadas, produtos tombados, ou lixo visível.
Resultado esperado:
  - overall_status: "critical"
  - issues: [
      {{ type: "damaged", severity: "high" }},
      {{ type: "misaligned", severity: "medium" }}
    ]
  - shelf_fill_rate: ~0.50 a 0.75
  - model_reasoning: "Prateleira em estado crítico. Múltiplas embalagens com dano
    físico visível (amarfanhadas/rasgadas). Produtos tombados uns sobre os outros.
    Estado exige intervenção humana imediata para limpeza e reorganização."

━━━ AGORA ANALISA A IMAGEM FORNECIDA ━━━
Aplica o mesmo nível de detalhe dos exemplos acima.\n\n"""
    return instrucao + _get_schema_instrucao(inspection_id, timestamp, image_path, zone_id)


# Roteador de estratégias
_ESTRATEGIAS = {
    "zero_shot": _prompt_zero_shot,
    "cot":       _prompt_cot,
    "few_shot":  _prompt_few_shot,
}


# ==========================================
# 6. FUNÇÃO PRINCIPAL DE INSPEÇÃO
# ==========================================

def inspecionar_prateleira(
    caminho_imagem: str,
    tipo_prompt: str = "cot",
    zone_id: str | None = None,
    modelo: str | None = None
) -> dict | None:
    """
    Analisa uma imagem de prateleira e devolve um JSON estruturado.

    Args:
        caminho_imagem: Caminho para o ficheiro de imagem.
        tipo_prompt: Estratégia de prompting ('zero_shot', 'cot', 'few_shot').
        zone_id: Identificador da zona. Se None, é derivado automaticamente
                 do nome da pasta (ex: images/dirty/ → Z_DIRTY).
        modelo: Override do modelo Gemini. Se None, usa o default da sessão
                (MODEL_NAME). Faz parte da chave de cache.

    Returns:
        Dicionário com a análise estruturada, ou None em caso de falha.
    """
    if tipo_prompt not in _ESTRATEGIAS:
        raise ValueError(
            f"tipo_prompt inválido: '{tipo_prompt}'. "
            f"Opções: {list(_ESTRATEGIAS.keys())}"
        )

    if not os.path.exists(caminho_imagem):
        logger.error("[INSPETOR] Ficheiro não encontrado: %s", caminho_imagem)
        return None

    # --- Metadados determinísticos ---
    modelo     = modelo or MODEL_NAME
    hash_id    = _gerar_hash_cache(caminho_imagem, tipo_prompt, modelo)
    zone_id    = zone_id or _derivar_zone_id(caminho_imagem)
    timestamp  = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    insp_id    = _gerar_inspection_id(hash_id)

    logger.info(
        "[INSPETOR] %s | estratégia=%s | zone=%s",
        Path(caminho_imagem).name, tipo_prompt, zone_id
    )

    # --- Verificar cache ---
    cached = _ler_cache(hash_id)
    if cached is not None:
        logger.info("[CACHE HIT] Resultado recuperado do disco.")
        return cached

    logger.info("[CACHE MISS] A chamar API...")

    # --- Construir prompt e carregar imagem ---
    construir_prompt = _ESTRATEGIAS[tipo_prompt]
    prompt = construir_prompt(insp_id, timestamp, caminho_imagem, zone_id)

    try:
        imagem = Image.open(caminho_imagem)
    except Exception as e:
        logger.error("[INSPETOR] Erro ao abrir imagem: %s", e)
        return None

    # --- Chamada à API (rate limit e retry geridos pelo api_client) ---
    try:
        response = gemini_client.gerar_conteudo(contents=[prompt, imagem], model=modelo)
    except RuntimeError as e:
        # Quota esgotada ou retries esgotados — falha graciosa
        logger.error("[INSPETOR] %s", str(e))
        return None

    # --- Parsing defensivo ---
    dados = _parsear_resposta(response.text)
    if dados is None:
        logger.error(
            "[INSPETOR] Resposta não parseável para: %s", caminho_imagem
        )
        return None

    # --- Validação e normalização ---
    dados = _validar_e_normalizar(dados, caminho_imagem, insp_id, zone_id, timestamp)

    # --- Persistir em cache ---
    _escrever_cache(hash_id, dados)

    return dados


# ==========================================
# 7. EXECUÇÃO DE TESTE (as 3 estratégias)
# ==========================================

if __name__ == "__main__":
    import sys

    # Permite passar o caminho da imagem como argumento:
    # python shelf_inspector.py images/dirty/foto.jpg
    if len(sys.argv) > 1:
        imagem_teste = sys.argv[1]
    else:
        # Fallback: primeira imagem que encontrar em qualquer subpasta de data/images/
        imagem_teste = None
        for pasta in ["normal", "dirty", "empty", "planogram", "ambiguous"]:
            p = Path(f"data/images/{pasta}")
            if p.exists():
                ficheiros = list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png"))
                if ficheiros:
                    imagem_teste = str(ficheiros[0])
                    break

    if not imagem_teste or not os.path.exists(imagem_teste):
        print("[AVISO] Nenhuma imagem de teste encontrada. "
              "Passa o caminho como argumento: python shelf_inspector.py <caminho>")
        sys.exit(1)

    print("\n" + "="*60)
    print("   TESTE COMPARATIVO — 3 ESTRATÉGIAS DE PROMPTING")
    print("="*60)
    print(f"Imagem: {imagem_teste}\n")

    resultados = {}
    estrategias = ["zero_shot", "cot", "few_shot"]

    for estrategia in estrategias:
        print(f"\n>>> Estratégia: {estrategia.upper()}")
        resultado = inspecionar_prateleira(imagem_teste, tipo_prompt=estrategia)

        if resultado:
            resultados[estrategia] = resultado
            print(f"  Status:       {resultado.get('overall_status')}")
            print(f"  Fill Rate:    {resultado.get('shelf_fill_rate'):.0%}")
            print(f"  Issues:       {len(resultado.get('issues', []))}")
            print(f"  Zone:         {resultado.get('zone_id')}")
            print(f"  Reasoning:    {resultado.get('model_reasoning', '')[:120]}...")
        else:
            print(f"  [FALHA] Não foi possível obter resultado para esta estratégia.")

        print("-" * 60)

    print("\n[CONCLUÍDO] Resultados guardados em cache/vision/")