"""
llm_judge.py
------------
LLM-as-Judge do Retail Vision Intelligence System (enunciado §9.3).

Avaliador automático que usa o próprio Gemini como juiz: recebe um output do
sistema + um critério de avaliação e devolve uma pontuação com justificação.
Alimenta três métricas que NÃO são computáveis programaticamente (§9.2):

  - Answer Relevance (RAG)  : a resposta responde à query do gestor?
  - Faithfulness (RAG)      : as afirmações da resposta são suportadas pelos
                              chunks recuperados? (anti-alucinação no RAG)
  - Hallucination Rate (vis): MULTIMODAL — o juiz vê a IMAGEM e verifica se as
                              afirmações do campo 'description' são realmente
                              visíveis (não inventadas).

Também suporta avaliação qualitativa genérica de relatórios.

------------------------------------------------------------------------
DECISÃO DE MODELO — preservação de quota
------------------------------------------------------------------------
O juiz corre por defeito em JUDGE_MODEL = 'gemini-2.5-flash', que tem um
bucket de quota SEPARADO do 'gemini-2.5-flash-lite' usado na análise de
visão (ver memória runtime-environment). Assim, avaliar não consome os
preciosos 20/dia do flash-lite reservados para o evaluate.py.

NOTA META-ANÁLISE (§9.3): cada avaliador devolve score + justificação para
permitir comparar a concordância juiz vs. humano (componente do relatório).
Reprodutibilidade: temperature=0.

Depende de:
  - utils/api_client.py → chamada à API (com override de modelo)
  - PIL                 → carregar a imagem para o juízo multimodal
"""

import json
import re
import logging
from pathlib import Path

from PIL import Image

from utils.api_client import gemini_client

logger = logging.getLogger(__name__)

# Modelo do juiz — bucket de quota separado do flash-lite (ver docstring).
JUDGE_MODEL = "gemini-2.5-flash"


# ==========================================
# 1. NÚCLEO — chamada ao juiz + parsing
# ==========================================

def _parsear_json(texto: str) -> dict | None:
    """Parsing defensivo a 3 níveis (igual ao resto do sistema)."""
    if not texto:
        return None
    for candidato in (
        texto,
        re.sub(r"```(?:json)?\s*", "", texto).strip(),
    ):
        try:
            return json.loads(candidato)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _julgar(prompt: str, imagem: Image.Image | None = None,
            modelo: str | None = None) -> dict | None:
    """
    Envia um prompt de avaliação ao juiz e devolve o JSON parseado.
    Devolve None em caso de falha (quota esgotada, parsing) — graciosamente.
    """
    contents = [prompt, imagem] if imagem is not None else [prompt]
    try:
        response = gemini_client.gerar_conteudo(
            contents=contents,
            temperature=0.0,                       # determinismo (§11)
            response_mime_type="application/json",
            model=modelo or JUDGE_MODEL,
        )
    except RuntimeError as e:
        logger.warning("[JUDGE] API indisponível: %s", e)
        return None

    dados = _parsear_json(response.text)
    if dados is None:
        logger.warning("[JUDGE] Resposta do juiz não parseável.")
    return dados


def _clamp_score(valor, lo: int = 1, hi: int = 5) -> int:
    """Garante um score inteiro dentro da escala."""
    try:
        return max(lo, min(hi, int(round(float(valor)))))
    except (TypeError, ValueError):
        return lo


# ==========================================
# 2. PROMPTS DOS AVALIADORES (versionados — anexo do relatório)
# ==========================================

_PROMPT_ANSWER_RELEVANCE = """
És um avaliador rigoroso de sistemas de pergunta-resposta para gestão de retalho.
A tua tarefa é avaliar se a RESPOSTA responde efetivamente à PERGUNTA do gestor.

PERGUNTA DO GESTOR:
{query}

RESPOSTA DO SISTEMA:
{resposta}

CRITÉRIO (Answer Relevance):
Avalia numa escala de 1 a 5 se a resposta é relevante e responde à pergunta:
  1 = não responde de todo / fora do tópico
  2 = aborda o tema mas não responde ao que foi perguntado
  3 = responde parcialmente
  4 = responde bem, com pequenas lacunas
  5 = responde de forma completa, direta e acionável

Ignora a correção factual (isso é avaliado noutra métrica) — avalia só a RELEVÂNCIA.

DEVOLVE EXCLUSIVAMENTE este JSON:
{{"score": <1-5>, "justificacao": "<2 frases a explicar o score>"}}
"""

_PROMPT_FAITHFULNESS = """
És um avaliador rigoroso de fidelidade factual em sistemas RAG de retalho.
Avalia se as afirmações da RESPOSTA são SUPORTADAS pelo CONTEXTO recuperado.
Uma resposta fiel só afirma o que está no contexto — não inventa nem extrapola.

PERGUNTA:
{query}

CONTEXTO RECUPERADO (única fonte de verdade permitida):
{contexto}

RESPOSTA A AVALIAR:
{resposta}

INSTRUÇÕES:
1. Identifica as afirmações factuais distintas na resposta.
2. Para cada uma, verifica se é suportada pelo contexto acima.
3. n_afirmacoes = total de afirmações; n_suportadas = quantas têm suporte.

DEVOLVE EXCLUSIVAMENTE este JSON:
{{"score": <1-5>, "n_afirmacoes": <int>, "n_suportadas": <int>,
  "justificacao": "<2 frases; identifica qualquer afirmação não suportada>"}}
"""

_PROMPT_HALLUCINATION = """
És um auditor visual rigoroso. Observa atentamente a IMAGEM fornecida.
Outro modelo analisou esta prateleira e produziu a DESCRIÇÃO de um problema.
A tua tarefa é verificar se essa descrição é REALMENTE VISÍVEL na imagem,
ou se é uma alucinação (afirmação plausível mas não fundamentada no que se vê).

DESCRIÇÃO GERADA PELO OUTRO MODELO:
"{descricao}"

INSTRUÇÕES:
- Olha a imagem e decide: o que a descrição afirma está mesmo visível?
- "verificavel": true se o que é descrito se observa na imagem; false se não
  há evidência visual (alucinação).
- Sê exigente: na dúvida sem evidência clara, marca como NÃO verificável.

DEVOLVE EXCLUSIVAMENTE este JSON:
{{"verificavel": <true|false>, "score": <1-5 de confiança no teu juízo>,
  "justificacao": "<1-2 frases sobre o que vês ou não vês>"}}
"""

_PROMPT_GENERICO = """
És um avaliador especialista. Avalia o OUTPUT abaixo segundo o CRITÉRIO dado.

CRITÉRIO DE AVALIAÇÃO:
{criterio}

OUTPUT A AVALIAR:
{output}

DEVOLVE EXCLUSIVAMENTE este JSON:
{{"score": <1-5>, "justificacao": "<2-3 frases a fundamentar o score>"}}
"""


# ==========================================
# 3. AVALIADORES PÚBLICOS
# ==========================================

def avaliar_answer_relevance(query: str, resposta: str) -> dict:
    """RAG Answer Relevance: a resposta responde à query? (score 1-5)."""
    prompt = _PROMPT_ANSWER_RELEVANCE.format(query=query, resposta=resposta)
    r = _julgar(prompt)
    if r is None:
        return {"score": None, "justificacao": "[juiz indisponível]", "ok": False}
    return {
        "score": _clamp_score(r.get("score")),
        "justificacao": str(r.get("justificacao", "")),
        "ok": True,
    }


def avaliar_faithfulness(query: str, resposta: str, chunks: list) -> dict:
    """
    RAG Faithfulness: % de afirmações da resposta suportadas pelos chunks.
    chunks: lista de dicts (do query_memoria) ou de strings.
    """
    contexto = _formatar_chunks(chunks)
    prompt = _PROMPT_FAITHFULNESS.format(
        query=query, contexto=contexto, resposta=resposta
    )
    r = _julgar(prompt)
    if r is None:
        return {"score": None, "faithfulness": None,
                "justificacao": "[juiz indisponível]", "ok": False}

    n_af = max(0, int(r.get("n_afirmacoes", 0) or 0))
    n_sup = max(0, int(r.get("n_suportadas", 0) or 0))
    faithfulness = round(n_sup / n_af, 3) if n_af else None
    return {
        "score": _clamp_score(r.get("score")),
        "n_afirmacoes": n_af,
        "n_suportadas": n_sup,
        "faithfulness": faithfulness,
        "justificacao": str(r.get("justificacao", "")),
        "ok": True,
    }


def avaliar_hallucination(descricao: str, caminho_imagem: str) -> dict:
    """
    Hallucination (visual, MULTIMODAL): o juiz vê a imagem e decide se a
    'description' é verificável. Devolve {verificavel, score, justificacao}.
    """
    if not descricao or not descricao.strip():
        return {"verificavel": True, "score": 5,
                "justificacao": "[sem descrição para avaliar]", "ok": True}
    try:
        imagem = Image.open(caminho_imagem)
    except Exception as e:
        logger.warning("[JUDGE] Não foi possível abrir imagem %s: %s", caminho_imagem, e)
        return {"verificavel": None, "score": None,
                "justificacao": "[imagem indisponível]", "ok": False}

    prompt = _PROMPT_HALLUCINATION.format(descricao=descricao)
    r = _julgar(prompt, imagem=imagem)
    if r is None:
        return {"verificavel": None, "score": None,
                "justificacao": "[juiz indisponível]", "ok": False}
    return {
        "verificavel": bool(r.get("verificavel", False)),
        "score": _clamp_score(r.get("score")),
        "justificacao": str(r.get("justificacao", "")),
        "ok": True,
    }


def avaliar_generico(output: str, criterio: str) -> dict:
    """Avaliação qualitativa genérica (ex: qualidade de um relatório)."""
    prompt = _PROMPT_GENERICO.format(criterio=criterio, output=output)
    r = _julgar(prompt)
    if r is None:
        return {"score": None, "justificacao": "[juiz indisponível]", "ok": False}
    return {
        "score": _clamp_score(r.get("score")),
        "justificacao": str(r.get("justificacao", "")),
        "ok": True,
    }


def _formatar_chunks(chunks: list) -> str:
    """Formata chunks (dicts do RAG ou strings) para o prompt de faithfulness."""
    partes = []
    for i, c in enumerate(chunks, 1):
        if isinstance(c, dict):
            doc = c.get("documento") or c.get("texto") or ""
            meta = c.get("metadados", {})
            idp = meta.get("inspection_id", "?")
            partes.append(f"[{i}] ({idp}) {doc[:300]}")
        else:
            partes.append(f"[{i}] {str(c)[:300]}")
    return "\n".join(partes) or "(sem contexto)"


# ==========================================
# 4. AGREGADORES (para o evaluate.py / relatório)
# ==========================================

def taxa_alucinacao(predicoes: list) -> dict:
    """
    Calcula a Hallucination Rate sobre uma lista de predições de inspeção.
    Para cada issue de cada predição, o juiz vê a imagem e verifica a 'description'.

    predicoes: lista de dicts de inspeção (com 'image_path' e 'issues').

    Returns:
        {hallucination_rate, n_descricoes, n_alucinadas, n_avaliadas, detalhes}
    ATENÇÃO: faz 1 chamada multimodal por description — quota-pesado.
    """
    n_aval = n_aluc = 0
    detalhes = []

    for insp in predicoes:
        img = insp.get("image_path", "")
        for issue in insp.get("issues", []):
            desc = issue.get("description", "")
            if not desc:
                continue
            veredicto = avaliar_hallucination(desc, img)
            if not veredicto.get("ok"):
                continue  # juiz indisponível → não conta
            n_aval += 1
            alucinou = not veredicto.get("verificavel", True)
            if alucinou:
                n_aluc += 1
            detalhes.append({
                "inspection_id": insp.get("inspection_id"),
                "descricao": desc[:120],
                "verificavel": veredicto.get("verificavel"),
                "justificacao": veredicto.get("justificacao"),
            })

    taxa = round(n_aluc / n_aval, 4) if n_aval else None
    return {
        "hallucination_rate": taxa,
        "n_descricoes_avaliadas": n_aval,
        "n_alucinadas": n_aluc,
        "detalhes": detalhes,
    }


def avaliar_respostas_rag(itens: list) -> dict:
    """
    Avalia em lote Answer Relevance + Faithfulness de respostas RAG.
    itens: [{"query":..., "resposta":..., "chunks":[...]}, ...]

    Returns média das métricas + detalhes por item.
    """
    relevancias, faiths, detalhes = [], [], []
    for it in itens:
        ar = avaliar_answer_relevance(it["query"], it["resposta"])
        ff = avaliar_faithfulness(it["query"], it["resposta"], it.get("chunks", []))
        if ar.get("score") is not None:
            relevancias.append(ar["score"])
        if ff.get("faithfulness") is not None:
            faiths.append(ff["faithfulness"])
        detalhes.append({"query": it["query"],
                         "answer_relevance": ar, "faithfulness": ff})

    return {
        "answer_relevance_medio": round(sum(relevancias) / len(relevancias), 3) if relevancias else None,
        "faithfulness_media": round(sum(faiths) / len(faiths), 3) if faiths else None,
        "n_avaliadas": len(itens),
        "detalhes": detalhes,
    }


# ==========================================
# 5. TESTE
# ==========================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.console import configurar_consola
    configurar_consola()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 64)
    print("   TESTE DO LLM-AS-JUDGE".center(64))
    print("=" * 64)
    print(f"  Modelo do juiz: {JUDGE_MODEL}\n")

    # Teste 1: Answer Relevance (1 chamada)
    print("[TESTE 1] Answer Relevance")
    r = avaliar_answer_relevance(
        query="Quando foi a última vez que a zona Z_EMPTY teve rutura de stock?",
        resposta="A zona Z_EMPTY teve uma rutura crítica na inspeção INS_001 "
                 "a 10/06/2026, com fill rate de 38%."
    )
    print(f"  score={r.get('score')} | {r.get('justificacao')}")

    # Teste 2: Answer Relevance — resposta irrelevante (deve dar score baixo)
    print("\n[TESTE 2] Answer Relevance (resposta evasiva)")
    r2 = avaliar_answer_relevance(
        query="Quantas zonas estão em estado crítico?",
        resposta="As prateleiras de retalho são importantes para as vendas."
    )
    print(f"  score={r2.get('score')} | {r2.get('justificacao')}")

    # Teste 3: Faithfulness
    print("\n[TESTE 3] Faithfulness")
    r3 = avaliar_faithfulness(
        query="A zona Z_EMPTY teve problemas?",
        resposta="Sim, a Z_EMPTY teve rutura crítica e também um incêndio.",
        chunks=[{"documento": "Z_EMPTY: rutura de stock, fill rate 38%, status critical",
                 "metadados": {"inspection_id": "INS_001"}}]
    )
    print(f"  faithfulness={r3.get('faithfulness')} "
          f"({r3.get('n_suportadas')}/{r3.get('n_afirmacoes')}) | {r3.get('justificacao')}")

    print("\n[CONCLUÍDO]")
