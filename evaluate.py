"""
evaluate.py
-----------
Harness de avaliação do Retail Vision Intelligence System (enunciado §9).

EXECUÇÃO (o comando que o professor corre na defesa):
    python evaluate.py --images-dir test_images/ --output evaluation_report.json

O professor fornece test_images/ com 10 imagens NÃO vistas. O ground truth
dessas imagens é procurado automaticamente (ver --ground-truth abaixo).

EXECUÇÃO DE AUTO-TESTE (sem gastar quota — usa o cache já existente):
    python evaluate.py --images-dir images --ground-truth data/ground_truth.json --output eval_dev.json

------------------------------------------------------------------------
MÉTRICAS IMPLEMENTADAS (enunciado §9.2)
------------------------------------------------------------------------
Análise visual:
  - Issue Detection Rate (recall) : % de issues do GT corretamente identificados
  - False Positive Rate           : % de issues reportados que não existem no GT
  - Severity Accuracy             : % de issues casados com severidade correta
  - JSON Parse Rate               : % de respostas do modelo parseáveis
  - Hallucination Rate            : DEPENDE do LLM-as-Judge (M3) — null até lá

Extras (fortalecem a análise de limitações no relatório):
  - status_accuracy               : overall_status vs true_status
  - fill_rate_mae                 : erro médio absoluto do shelf_fill_rate
  - por_tipo                      : recall discriminado por tipo de issue
                                    (essencial: só empty_shelf tem amostra robusta)

RAG (§9.2) e Rule Engine (§9.2) são avaliados em secções opcionais que NÃO
consomem quota (Recall@3 usa só embeddings locais; rule correctness usa dados
sintéticos). Ativam-se com --rag-eval / --rules-eval.

------------------------------------------------------------------------
DECISÃO DE MATCHING (P5) — documentada para a defesa oral
------------------------------------------------------------------------
O matching predito↔ground-truth é feito ao nível do TIPO de issue, por imagem.
Um issue de GT do tipo T conta como detetado se a predição tiver pelo menos um
issue do tipo T na MESMA imagem (matching por multiconjunto de tipos).
Justificação: o campo 'location' é texto livre em português tanto no GT como na
predição — casá-lo programaticamente seria frágil e introduziria ruído nas
métricas. O tipo é a dimensão robusta e é o que o enunciado mede (recall de
deteção). A severidade é comparada nos pares casados por tipo.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

# --- Permite importar os módulos de src/ a partir da raiz ---
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.console import configurar_consola
from shelf_inspector import inspecionar_prateleira, esta_em_cache
from utils.api_client import MODEL_NAME, gemini_client

configurar_consola()  # consola UTF-8 segura (demo-crítico)

logger = logging.getLogger(__name__)

EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg")
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
TODOS_OS_TIPOS = [
    "empty_shelf", "wrong_product", "damaged",
    "misaligned", "label_missing", "other",
]


# ==========================================
# 1. RECOLHA DE IMAGENS E GROUND TRUTH
# ==========================================

def _normalizar_caminho(caminho: str) -> str:
    """Normaliza separadores para '/' — Windows usa '\\', o GT usa '/'."""
    return str(caminho).replace("\\", "/")


def _listar_imagens(images_dir: Path) -> list[Path]:
    """Recolhe todas as imagens de uma pasta (recursivo), ordenadas."""
    if not images_dir.exists():
        return []
    ficheiros = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSOES_VALIDAS
    ]
    return sorted(ficheiros, key=lambda p: _normalizar_caminho(p).lower())


def _carregar_ground_truth(images_dir: Path, gt_arg: str | None) -> dict:
    """
    Carrega o ground truth. Ordem de procura:
      1. --ground-truth explícito, se fornecido
      2. <images-dir>/ground_truth.json  (convenção do professor)
      3. data/ground_truth.json          (o nosso, para auto-teste)

    Devolve o dicionário 'annotations' (chave = caminho relativo da imagem).
    Devolve {} se não houver GT — as métricas dependentes de GT ficam a null.
    """
    candidatos = []
    if gt_arg:
        candidatos.append(Path(gt_arg))
    candidatos.append(images_dir / "ground_truth.json")
    candidatos.append(Path("data/ground_truth.json"))

    for caminho in candidatos:
        if caminho.exists():
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    dados = json.load(f)
                # Aceita tanto {"annotations": {...}} como {...} plano
                annotations = dados.get("annotations", dados)
                logger.info("[EVAL] Ground truth carregado de: %s (%d entradas)",
                            caminho, len(annotations))
                return annotations
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("[EVAL] GT ilegível em %s: %s", caminho, e)

    logger.warning("[EVAL] Nenhum ground truth encontrado — "
                   "métricas dependentes de GT ficarão a null.")
    return {}


def _procurar_gt_para_imagem(caminho_imagem: Path, gt: dict) -> dict | None:
    """
    Encontra a entrada de GT para uma imagem.
    Tenta primeiro o caminho relativo normalizado, depois o nome de ficheiro
    (robusto a diferenças de diretório entre images/ e test_images/).
    """
    if not gt:
        return None

    alvo_path = _normalizar_caminho(caminho_imagem)
    # 1) match por caminho completo normalizado (sufixo, para tolerar prefixos)
    for chave, valor in gt.items():
        chave_norm = _normalizar_caminho(chave)
        if alvo_path.endswith(chave_norm) or chave_norm.endswith(alvo_path):
            return valor

    # 2) match por nome de ficheiro (os nomes são longos e únicos)
    nome = caminho_imagem.name
    for chave, valor in gt.items():
        if Path(_normalizar_caminho(chave)).name == nome:
            return valor

    return None


# ==========================================
# 2. MATCHING DE ISSUES (decisão P5)
# ==========================================

def _casar_issues_por_tipo(pred_issues: list, gt_issues: list) -> dict:
    """
    Casa issues preditos com issues do GT ao nível do tipo (multiconjunto).

    Returns:
        {
          "gt_total": int,            # nº de issues no GT
          "pred_total": int,          # nº de issues preditos
          "matched": int,             # nº de issues do GT detetados
          "fp": int,                  # nº de issues preditos sem correspondência
          "pares_severidade": [(sev_gt, sev_pred), ...],  # pares casados por tipo
          "por_tipo": {tipo: {"gt": n, "matched": n}, ...}
        }
    """
    gt_por_tipo   = defaultdict(list)
    pred_por_tipo = defaultdict(list)
    for it in gt_issues:
        gt_por_tipo[it.get("type", "other")].append(it)
    for it in pred_issues:
        pred_por_tipo[it.get("type", "other")].append(it)

    matched = 0
    pares_severidade = []
    por_tipo = {}

    tipos = set(gt_por_tipo) | set(pred_por_tipo)
    for tipo in tipos:
        gts   = gt_por_tipo.get(tipo, [])
        preds = pred_por_tipo.get(tipo, [])
        n_match = min(len(gts), len(preds))
        matched += n_match

        if gts:
            por_tipo[tipo] = {"gt": len(gts), "matched": n_match}

        # Pares de severidade: empareja os primeiros n_match de cada lado,
        # ordenados por rank de severidade para um emparelhamento estável e justo.
        gts_ord   = sorted(gts,   key=lambda x: SEVERITY_RANK.get(x.get("severity", "low"), 0))
        preds_ord = sorted(preds, key=lambda x: SEVERITY_RANK.get(x.get("severity", "low"), 0))
        for i in range(n_match):
            pares_severidade.append((
                gts_ord[i].get("severity", "low"),
                preds_ord[i].get("severity", "low"),
            ))

    gt_total   = len(gt_issues)
    pred_total = len(pred_issues)
    fp = max(0, pred_total - matched)

    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "matched": matched,
        "fp": fp,
        "pares_severidade": pares_severidade,
        "por_tipo": por_tipo,
    }


# ==========================================
# 3. AVALIAÇÃO VISUAL
# ==========================================

def avaliar_visual(images_dir: Path, gt: dict, estrategia: str) -> dict:
    """
    Corre a inspeção (cache-first) sobre todas as imagens e calcula as
    métricas visuais da §9.2 contra o ground truth disponível.
    """
    imagens = _listar_imagens(images_dir)
    if not imagens:
        return {"erro": f"Nenhuma imagem encontrada em {images_dir}"}

    print(f"\n[EVAL] {len(imagens)} imagens em {images_dir} | estratégia={estrategia}")
    print(f"[EVAL] Modelo: {MODEL_NAME} | cache-first (imagens vistas não gastam quota)\n")

    # Acumuladores globais
    n_parse_ok = 0
    n_tentadas = 0
    gt_total = pred_total = matched_total = fp_total = 0
    sev_pares = []
    status_ok = status_aval = 0
    fill_erros = []
    por_tipo_acc = defaultdict(lambda: {"gt": 0, "matched": 0})
    detalhes = []
    predicoes_raw = {}

    for caminho in imagens:
        caminho_str = _normalizar_caminho(caminho)
        em_cache = esta_em_cache(caminho_str, estrategia)

        # Fallback gracioso: quota esgotada + sem cache → não conta como parse-fail
        if not em_cache and gemini_client.quota_esgotada:
            print(f"  [QUOTA] saltada (sem cache): {caminho.name}")
            detalhes.append({"imagem": caminho_str, "estado": "saltada_quota"})
            continue

        n_tentadas += 1
        resultado = inspecionar_prateleira(caminho_str, tipo_prompt=estrategia)

        if resultado is None:
            # None = parse falhou OU quota/erro de API. Só conta como parse-fail
            # se NÃO foi quota (a quota é uma limitação externa, não do modelo).
            if gemini_client.quota_esgotada:
                n_tentadas -= 1
                print(f"  [QUOTA] saltada (sem cache): {caminho.name}")
                detalhes.append({"imagem": caminho_str, "estado": "saltada_quota"})
            else:
                print(f"  [PARSE-FAIL] {caminho.name}")
                detalhes.append({"imagem": caminho_str, "estado": "parse_fail"})
            continue

        n_parse_ok += 1
        predicoes_raw[caminho_str] = resultado
        origem = "cache" if em_cache else "api"
        print(f"  [{origem.upper():5}] {caminho.name}  →  {resultado.get('overall_status')}")

        entrada_gt = _procurar_gt_para_imagem(caminho, gt)
        det = {
            "imagem": caminho_str,
            "estado": "ok",
            "origem": origem,
            "status_pred": resultado.get("overall_status"),
            "fill_pred": resultado.get("shelf_fill_rate"),
            "issues_pred": [i.get("type") for i in resultado.get("issues", [])],
            "tem_gt": entrada_gt is not None,
        }

        if entrada_gt is not None:
            gt_issues   = entrada_gt.get("issues", [])
            pred_issues = resultado.get("issues", [])
            m = _casar_issues_por_tipo(pred_issues, gt_issues)

            gt_total      += m["gt_total"]
            pred_total    += m["pred_total"]
            matched_total += m["matched"]
            fp_total      += m["fp"]
            sev_pares.extend(m["pares_severidade"])
            for tipo, v in m["por_tipo"].items():
                por_tipo_acc[tipo]["gt"]      += v["gt"]
                por_tipo_acc[tipo]["matched"] += v["matched"]

            # Status accuracy
            status_aval += 1
            if resultado.get("overall_status") == entrada_gt.get("true_status"):
                status_ok += 1

            # Fill rate MAE
            try:
                fill_erros.append(abs(
                    float(resultado.get("shelf_fill_rate", 0))
                    - float(entrada_gt.get("true_fill_rate", 0))
                ))
            except (TypeError, ValueError):
                pass

            det.update({
                "status_gt": entrada_gt.get("true_status"),
                "fill_gt": entrada_gt.get("true_fill_rate"),
                "issues_gt": [i.get("type") for i in gt_issues],
                "gt_detetados": m["matched"],
                "gt_total": m["gt_total"],
                "falsos_positivos": m["fp"],
            })

        detalhes.append(det)

    # --- Cálculo das métricas ---
    def _pct(num, den):
        return round(num / den, 4) if den else None

    sev_correctas = sum(1 for g, p in sev_pares if g == p)

    por_tipo = {
        tipo: {
            "suporte_gt": v["gt"],
            "detetados": v["matched"],
            "recall": _pct(v["matched"], v["gt"]),
        }
        for tipo, v in sorted(por_tipo_acc.items())
    }

    return {
        "n_imagens": len(imagens),
        "n_parse_ok": n_parse_ok,
        "n_tentadas": n_tentadas,
        "n_com_ground_truth": sum(1 for d in detalhes if d.get("tem_gt")),
        "metricas": {
            "issue_detection_rate": _pct(matched_total, gt_total),   # recall
            "false_positive_rate":  _pct(fp_total, pred_total),
            "severity_accuracy":    _pct(sev_correctas, len(sev_pares)),
            "json_parse_rate":      _pct(n_parse_ok, n_tentadas),
            "hallucination_rate":   None,  # requer LLM-as-Judge (M3)
            # --- extras ---
            "status_accuracy":      _pct(status_ok, status_aval),
            "fill_rate_mae":        round(sum(fill_erros) / len(fill_erros), 4) if fill_erros else None,
        },
        "suporte": {
            "issues_gt_total": gt_total,
            "issues_pred_total": pred_total,
            "issues_casados": matched_total,
            "falsos_positivos": fp_total,
            "pares_severidade": len(sev_pares),
        },
        "por_tipo": por_tipo,
        "nota_hallucination": (
            "Hallucination Rate requer o LLM-as-Judge (M3): cada afirmação no "
            "campo 'description' tem de ser verificada contra a imagem por um "
            "juiz. Não é computável de forma puramente programática."
        ),
        "detalhes_por_imagem": detalhes,
        "predicoes_raw": predicoes_raw,
    }


# ==========================================
# 4. AVALIAÇÃO RAG (opcional, sem quota)
# ==========================================

def avaliar_rag(caminho_queries_gt: str | None) -> dict:
    """
    Recall@3 das duas estratégias de chunking. Usa só embeddings locais
    (sintetizar=False) — NÃO consome quota da API.

    queries_gt (JSON): [{"query": "...", "inspection_ids_relevantes": ["INS_..."]}]
    Se não fornecido, devolve um aviso (precisa de ground truth de queries).
    """
    try:
        from rag_memory import avaliar_recall_k, estatisticas_vectorstore
    except ImportError as e:
        return {"erro": f"rag_memory indisponível: {e}"}

    stats = estatisticas_vectorstore()
    if stats.get("colecao_hibrida", 0) == 0:
        return {"erro": "Vectorstore vazio — indexa o dataset antes de avaliar o RAG."}

    if not caminho_queries_gt or not Path(caminho_queries_gt).exists():
        return {
            "aviso": "Sem ground truth de queries (--rag-queries). "
                     "Recall@3 não computado.",
            "vectorstore": stats,
        }

    with open(caminho_queries_gt, "r", encoding="utf-8") as f:
        queries_gt = json.load(f)

    resultado = {"vectorstore": stats}
    for est in ("hybrid", "full"):
        r = avaliar_recall_k(queries_gt, k=3, estrategia=est)
        resultado[f"recall_at_3_{est}"] = r["recall_at_k"]
        resultado[f"detalhe_{est}"] = r
    return resultado


# ==========================================
# 5. AVALIAÇÃO RULE ENGINE (opcional, sem quota)
# ==========================================

def avaliar_rules() -> dict:
    """
    Rule Correctness sobre dados sintéticos — NÃO consome quota (só executa
    _avaliar_condicoes sobre regras já guardadas, sem chamar a API).

    Rule Parse Rate e Ambiguity Detection precisam de chamadas à API
    (conversão NL→JSON) e ficam para uma run com quota disponível.
    """
    try:
        from rule_engine import listar_regras, _avaliar_condicoes
    except ImportError as e:
        return {"erro": f"rule_engine indisponível: {e}"}

    regras = [r for r in listar_regras() if r.get("validation", {}).get("is_valid")]
    if not regras:
        return {"aviso": "Sem regras válidas guardadas — cria regras antes de avaliar."}

    # Inspeções sintéticas com resultado esperado conhecido
    casos = [
        {
            "nome": "rutura grave Z_EMPTY",
            "inspeccao": {
                "inspection_id": "EVAL_SYN_001", "zone_id": "Z_EMPTY",
                "overall_status": "critical", "shelf_fill_rate": 0.30,
                "issues": [{"type": "empty_shelf", "severity": "high", "location": "inferior"}],
                "timestamp": "2026-06-10T12:00:00Z",
            },
        },
        {
            "nome": "prateleira saudável Z_NORMAL",
            "inspeccao": {
                "inspection_id": "EVAL_SYN_002", "zone_id": "Z_NORMAL",
                "overall_status": "ok", "shelf_fill_rate": 0.95,
                "issues": [], "timestamp": "2026-06-10T12:00:00Z",
            },
        },
    ]

    resultados = []
    for caso in casos:
        disparos = {
            r["rule_id"]: bool(_avaliar_condicoes(r, caso["inspeccao"]))
            for r in regras
        }
        resultados.append({"caso": caso["nome"], "disparos": disparos})

    return {
        "n_regras_avaliadas": len(regras),
        "casos": resultados,
        "nota": ("Rule Parse Rate e Ambiguity Detection requerem chamadas à API "
                 "(conversão NL→JSON) — correr numa sessão com quota."),
    }


# ==========================================
# 6. CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Harness de avaliação do Retail Vision Intelligence System."
    )
    parser.add_argument("--images-dir", required=True,
                        help="Pasta com as imagens a avaliar (ex: test_images/).")
    parser.add_argument("--output", default="evaluation_report.json",
                        help="Ficheiro JSON de saída.")
    parser.add_argument("--ground-truth", default=None,
                        help="Ground truth das imagens (auto-detetado se omitido).")
    parser.add_argument("--strategy", default="cot",
                        choices=["zero_shot", "cot", "few_shot"],
                        help="Estratégia de prompting a avaliar (default: cot).")
    parser.add_argument("--rag-eval", action="store_true",
                        help="Inclui avaliação do RAG (Recall@3, sem quota).")
    parser.add_argument("--rag-queries", default=None,
                        help="JSON com ground truth de queries para Recall@3.")
    parser.add_argument("--rules-eval", action="store_true",
                        help="Inclui avaliação do Rule Engine (sem quota).")
    parser.add_argument("--judge", action="store_true",
                        help="Calcula a Hallucination Rate via LLM-as-Judge "
                             "(multimodal; usa o bucket do gemini-2.5-flash).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    images_dir = Path(args.images_dir)
    gt = _carregar_ground_truth(images_dir, args.ground_truth)

    print("=" * 64)
    print("   HARNESS DE AVALIAÇÃO — RETAIL VISION".center(64))
    print("=" * 64)

    relatorio = {
        "metadata": {
            "gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "images_dir": str(images_dir),
            "estrategia": args.strategy,
            "modelo": MODEL_NAME,
            "tem_ground_truth": bool(gt),
        },
        "analise_visual": avaliar_visual(images_dir, gt, args.strategy),
    }

    # --- Hallucination Rate via LLM-as-Judge (opcional, multimodal) ---
    if args.judge:
        print("\n[EVAL] A calcular Hallucination Rate (LLM-as-Judge multimodal)...")
        try:
            import llm_judge
            preds = list(relatorio["analise_visual"].get("predicoes_raw", {}).values())
            halluc = llm_judge.taxa_alucinacao(preds)
            relatorio["analise_visual"]["metricas"]["hallucination_rate"] = \
                halluc.get("hallucination_rate")
            relatorio["analise_visual"]["hallucination_detalhe"] = halluc
            print(f"[EVAL] {halluc.get('n_descricoes_avaliadas')} descrições avaliadas, "
                  f"{halluc.get('n_alucinadas')} alucinada(s).")
        except Exception as e:
            logger.warning("[EVAL] Falha no juiz de alucinação: %s", e)
            print(f"  [AVISO] Hallucination Rate não computada: {e}")

    if args.rag_eval:
        print("\n[EVAL] A avaliar RAG (Recall@3)...")
        relatorio["analise_rag"] = avaliar_rag(args.rag_queries)

    if args.rules_eval:
        print("\n[EVAL] A avaliar Rule Engine...")
        relatorio["analise_rules"] = avaliar_rules()

    # --- Persistência ---
    caminho_out = Path(args.output)
    with open(caminho_out, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=4, ensure_ascii=False)

    # --- Resumo na consola ---
    visual = relatorio["analise_visual"]
    metr = visual.get("metricas", {})
    print("\n" + "=" * 64)
    print("   RESUMO DAS MÉTRICAS VISUAIS".center(64))
    print("=" * 64)
    print(f"  Imagens avaliadas        : {visual.get('n_parse_ok')}/{visual.get('n_imagens')}")
    print(f"  Com ground truth         : {visual.get('n_com_ground_truth')}")
    print("  " + "-" * 50)

    def _fmt(v):
        return f"{v:.1%}" if isinstance(v, float) else "n/d"

    print(f"  Issue Detection Rate     : {_fmt(metr.get('issue_detection_rate'))}")
    print(f"  False Positive Rate      : {_fmt(metr.get('false_positive_rate'))}")
    print(f"  Severity Accuracy        : {_fmt(metr.get('severity_accuracy'))}")
    print(f"  JSON Parse Rate          : {_fmt(metr.get('json_parse_rate'))}")
    _hr = metr.get("hallucination_rate")
    print(f"  Hallucination Rate       : "
          f"{_fmt(_hr) if _hr is not None else 'requer --judge'}")
    print(f"  [extra] Status Accuracy  : {_fmt(metr.get('status_accuracy'))}")
    print(f"  [extra] Fill Rate MAE    : {metr.get('fill_rate_mae')}")
    print("  " + "-" * 50)
    print("  Recall por tipo de issue:")
    for tipo, v in visual.get("por_tipo", {}).items():
        print(f"    {tipo:14}: {_fmt(v.get('recall'))}  (suporte GT={v.get('suporte_gt')})")
    print("=" * 64)
    print(f"  Relatório completo: {caminho_out}")
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EVAL] Interrompido pelo utilizador.")
    except Exception as e:
        # Nunca expor stack trace ao utilizador (enunciado §8 / §11).
        logger.error("[EVAL] Erro inesperado: %s", e, exc_info=True)
        print(f"\n[ERRO] A avaliação falhou: {e}")
        sys.exit(1)
