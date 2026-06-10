"""
interface.py
------------
Componente 5 do Retail Vision Intelligence System — a CLI conversacional que
orquestra todo o sistema (enunciado §8). É o ponto de entrada principal e o
palco da demonstração ao vivo.

Princípios de design (não-negociáveis, ditados pelo enunciado):
  - NUNCA expor um stack trace ao utilizador (§8/§11). Todo o comando corre
    dentro de um try/except que devolve uma mensagem informativa.
  - Mantém ESTADO DE SESSÃO: inspeções feitas, regra pendente de clarificação.
  - Cada comando reutiliza as funções públicas já validadas dos componentes
    1–4; esta camada é só orquestração + parsing + apresentação.

EXECUÇÃO (a partir da RAIZ do projeto):
    python src/interface.py                 # modo interativo (REPL)
    python src/interface.py inspect all --images-dir data/images/normal
    python src/interface.py report --session today

COMANDOS (§8):
    inspect <zona> --image <ficheiro>
    inspect all --images-dir <pasta>
    add rule "<regra em linguagem natural>"
    list rules
    delete rule <RULE_ID>
    test rule <RULE_ID> --image <ficheiro>
    resolve "<respostas às ambiguidades>"      (clarifica a última regra pendente)
    history "<pergunta em linguagem natural>"
    compare <zonaA> <zonaB> --period "last 7 days"
    report --session today
    report --zone <zona> --period "last 14 days"
    help | exit
"""

import sys
import json
import shlex
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

# --- src/ no path para imports planos a partir da raiz ---
sys.path.insert(0, str(Path(__file__).parent))

from utils.console import configurar_consola
configurar_consola()  # consola UTF-8 segura — DEMO-CRÍTICO (tem de ser o 1.º)

import re

from shelf_inspector import inspecionar_prateleira, esta_em_cache
from utils.api_client import gemini_client, MODEL_NAME
from rule_engine import (
    criar_regra, responder_ambiguidade, listar_regras,
    eliminar_regra, testar_regra, executar_regras,
)

logger = logging.getLogger(__name__)

EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg")
DATASET_PADRAO = Path("cache/vision/dataset_final.json")
ESTRATEGIA_PADRAO = "cot"


# ==========================================
# IMPORTS PESADOS (lazy — só quando precisos)
# ==========================================
# rag_memory e report_generator puxam chromadb / sentence-transformers, que
# são lentos a carregar. Importamo-los só no primeiro comando que precise
# deles, para o arranque da REPL ser instantâneo.

def _rag():
    import rag_memory
    return rag_memory


def _report():
    import report_generator
    return report_generator


# ==========================================
# PARSING DE ARGUMENTOS
# ==========================================

def _parse_args(tokens: list) -> tuple[list, dict]:
    """
    Separa posicionais de flags '--chave valor'.
    Aceita '--images-dir' e '--images_dir' como equivalentes (normaliza '-'→'_').
    """
    posicionais, flags = [], {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            chave = t[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[chave] = tokens[i + 1]
                i += 2
            else:
                flags[chave] = True
                i += 1
        else:
            posicionais.append(t)
            i += 1
    return posicionais, flags


def _periodo_para_dias(periodo) -> int | None:
    """'last 7 days' → 7 ; 'today' → 0 ; outro → None (sem filtro)."""
    if not periodo or periodo is True:
        return None
    p = str(periodo).lower().strip()
    if p in ("today", "hoje"):
        return 0
    m = re.search(r"(\d+)\s*(day|dia|week|semana)", p)
    if m:
        n = int(m.group(1))
        return n * 7 if m.group(2).startswith(("week", "semana")) else n
    return None


# ==========================================
# CARREGAMENTO DE INSPEÇÕES (dataset + sessão)
# ==========================================

def _carregar_inspecoes_dataset() -> list:
    """Lê as inspeções do dataset_final.json do batch_processor."""
    if not DATASET_PADRAO.exists():
        return []
    try:
        with open(DATASET_PADRAO, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return [
            e["analise_ia"]
            for e in dataset.get("inspecoes", {}).values()
            if e.get("analise_ia") is not None
        ]
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("[INTERFACE] Erro a ler dataset: %s", e)
        return []


def _filtrar_por_periodo(inspeccoes: list, dias: int | None) -> list:
    """Filtra inspeções cujo timestamp está dentro dos últimos `dias`."""
    if dias is None:
        return inspeccoes
    limite = datetime.now(timezone.utc) - timedelta(days=max(dias, 1))
    resultado = []
    for insp in inspeccoes:
        ts = insp.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt >= limite:
                resultado.append(insp)
        except (ValueError, AttributeError):
            resultado.append(insp)  # sem timestamp válido → não exclui
    return resultado


def _listar_imagens(pasta: Path) -> list[Path]:
    if not pasta.exists():
        return []
    fich = [p for p in pasta.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSOES_VALIDAS]
    return sorted(fich, key=lambda p: p.name.lower())


# ==========================================
# SESSÃO INTERATIVA
# ==========================================

class Sessao:
    """Mantém o estado da sessão conversacional."""

    def __init__(self):
        self.inspecoes_sessao: list = []     # inspeções feitas nesta sessão
        self.regra_pendente: dict | None = None  # regra à espera de clarificação

    # ---------- apresentação de uma inspeção ----------
    def _mostrar_inspecao(self, r: dict, alertas: list):
        emoji = {"ok": "[OK]", "warning": "[!]", "critical": "[X]"}.get(
            r.get("overall_status"), "[?]")
        print(f"\n  {emoji} {r.get('inspection_id')}  |  zona {r.get('zone_id')}  |  "
              f"status: {r.get('overall_status')}  |  fill: {r.get('shelf_fill_rate', 0):.0%}")
        issues = r.get("issues", [])
        if issues:
            print(f"  Issues ({len(issues)}):")
            for it in issues:
                print(f"    - {it.get('type')} ({it.get('severity')}) "
                      f"@ {it.get('location', '?')}  [conf {it.get('confidence', 0):.0%}]")
        else:
            print("  Sem issues detetados.")
        reasoning = r.get("model_reasoning", "")
        if reasoning:
            print(f"  Raciocínio: {reasoning[:160]}{'...' if len(reasoning) > 160 else ''}")
        if alertas:
            print(f"\n  >>> {len(alertas)} alerta(s) de regra disparado(s):")
            for a in alertas:
                print(f"    [{a.get('alert_level', 'info').upper()}] "
                      f"{a.get('rule_id')}: {a.get('message')}")

    def _processar_inspecao(self, r: dict, indexar: bool = True):
        """Pós-processa uma inspeção: regras, indexação RAG, estado de sessão."""
        self.inspecoes_sessao.append(r)
        try:
            alertas = executar_regras(r)
        except Exception as e:
            logger.warning("[INTERFACE] Erro ao executar regras: %s", e)
            alertas = []
        if indexar:
            try:
                self._rag_indexar(r)
            except Exception as e:
                logger.warning("[INTERFACE] Indexação RAG falhou (não-fatal): %s", e)
        self._mostrar_inspecao(r, alertas)

    def _rag_indexar(self, r: dict):
        """
        Indexa no RAG com summary SINTÉTICO (sem chamada à API) — preserva a
        quota de visão durante a demo. O bulk-indexing do dataset usa o
        summary rico via LLM (rag_memory.indexar_a_partir_de_dataset).
        """
        self._rag = getattr(self, "_rag", None) or _rag()
        self._rag.indexar_inspeccao(r, estrategia="hybrid", gerar_summary_llm=False)

    # ---------- inspect ----------
    def cmd_inspect(self, tokens: list):
        pos, flags = _parse_args(tokens[1:])

        if pos and pos[0].lower() == "all":
            pasta = flags.get("images_dir")
            if not pasta or pasta is True:
                print("  Uso: inspect all --images-dir <pasta>")
                return
            imagens = _listar_imagens(Path(pasta))
            if not imagens:
                print(f"  Nenhuma imagem encontrada em '{pasta}'.")
                return
            print(f"  A inspecionar {len(imagens)} imagem(ns) de '{pasta}'...")
            for caminho in imagens:
                cstr = str(caminho).replace("\\", "/")
                if not esta_em_cache(cstr, ESTRATEGIA_PADRAO) and gemini_client.quota_esgotada:
                    print(f"  [QUOTA] saltada (sem cache): {caminho.name}")
                    continue
                r = inspecionar_prateleira(cstr, tipo_prompt=ESTRATEGIA_PADRAO)
                if r is None:
                    print(f"  [FALHA] não foi possível inspecionar: {caminho.name}")
                    continue
                self._processar_inspecao(r)
            return

        # inspect <zona> --image <ficheiro>
        zona = pos[0] if pos else None
        imagem = flags.get("image")
        if not imagem or imagem is True:
            print("  Uso: inspect <zona> --image <ficheiro>")
            return
        if not Path(imagem).exists():
            print(f"  Ficheiro não encontrado: '{imagem}'.")
            return
        cstr = str(imagem).replace("\\", "/")
        if not esta_em_cache(cstr, ESTRATEGIA_PADRAO) and gemini_client.quota_esgotada:
            print("  [QUOTA] Quota diária esgotada e imagem não está em cache. "
                  "Tenta de novo após o reset, ou usa uma imagem já inspecionada.")
            return
        r = inspecionar_prateleira(cstr, tipo_prompt=ESTRATEGIA_PADRAO, zone_id=zona)
        if r is None:
            print("  [FALHA] Não foi possível analisar a imagem "
                  "(quota esgotada ou resposta não parseável).")
            return
        self._processar_inspecao(r)

    # ---------- add rule / resolve ----------
    def cmd_add_rule(self, tokens: list):
        texto = " ".join(tokens[2:]).strip()
        if not texto:
            print('  Uso: add rule "<regra em linguagem natural>"')
            return
        r = criar_regra(texto)
        val = r.get("validation", {})
        if val.get("is_valid"):
            print(f"  [OK] Regra {r['rule_id']} criada e guardada.")
            print(f"       {r.get('description', '')}")
        else:
            self.regra_pendente = r
            print(f"  [?] Regra {r['rule_id']} tem ambiguidades — não foi guardada:")
            for a in val.get("ambiguities", []):
                print(f"      - {a}")
            print('  Responde com:  resolve "<as tuas respostas>"   (ou: cancel)')

    def cmd_resolve(self, tokens: list):
        if not self.regra_pendente:
            print("  Não há nenhuma regra pendente de clarificação.")
            return
        respostas = " ".join(tokens[1:]).strip()
        if not respostas:
            print('  Uso: resolve "<respostas às ambiguidades>"')
            return
        r = responder_ambiguidade(self.regra_pendente, respostas)
        if r.get("validation", {}).get("is_valid"):
            print(f"  [OK] Regra {r['rule_id']} clarificada e guardada.")
            print(f"       {r.get('description', '')}")
            self.regra_pendente = None
        else:
            self.regra_pendente = r
            print("  [?] Ainda restam ambiguidades:")
            for a in r.get("validation", {}).get("ambiguities", []):
                print(f"      - {a}")

    def cmd_cancel(self, tokens: list):
        if self.regra_pendente:
            print(f"  Regra pendente {self.regra_pendente.get('rule_id')} descartada.")
            self.regra_pendente = None
        else:
            print("  Nada para cancelar.")

    # ---------- list / delete / test rules ----------
    def cmd_list_rules(self, tokens: list):
        regras = listar_regras()
        if not regras:
            print("  Não há regras guardadas.")
            return
        print(f"  {len(regras)} regra(s):")
        for r in regras:
            estado = "OK" if r.get("validation", {}).get("is_valid") else "PENDENTE"
            print(f"    [{estado:8}] {r.get('rule_id')}: {r.get('description', '')[:70]}")

    def cmd_delete_rule(self, tokens: list):
        if len(tokens) < 3:
            print("  Uso: delete rule <RULE_ID>")
            return
        rid = tokens[2]
        if eliminar_regra(rid):
            print(f"  [OK] Regra {rid} eliminada.")
        else:
            print(f"  Regra {rid} não encontrada.")

    def cmd_test_rule(self, tokens: list):
        pos, flags = _parse_args(tokens[2:])
        if not pos:
            print('  Uso: test rule <RULE_ID> --image <ficheiro>')
            return
        rid = pos[0]
        imagem = flags.get("image")
        if not imagem or imagem is True:
            print('  Uso: test rule <RULE_ID> --image <ficheiro>')
            return
        if not Path(imagem).exists():
            print(f"  Ficheiro não encontrado: '{imagem}'.")
            return
        cstr = str(imagem).replace("\\", "/")
        r = inspecionar_prateleira(cstr, tipo_prompt=ESTRATEGIA_PADRAO)
        if r is None:
            print("  [FALHA] Não foi possível inspecionar a imagem para o teste.")
            return
        resultado = testar_regra(rid, r)
        if "erro" in resultado:
            print(f"  {resultado['erro']}")
            return
        disparo = resultado.get("dispararia")
        print(f"  Regra {rid} {'DISPARARIA' if disparo else 'NÃO dispararia'} "
              f"sobre {Path(imagem).name}.")
        if disparo and resultado.get("mensagem_que_geraria"):
            print(f"    Mensagem: {resultado['mensagem_que_geraria']}")
        if not disparo and resultado.get("ambiguidades"):
            print(f"    (Regra inválida: {resultado['ambiguidades']})")

    # ---------- history ----------
    def cmd_history(self, tokens: list):
        pergunta = " ".join(tokens[1:]).strip().strip('"')
        if not pergunta:
            print('  Uso: history "<pergunta>"')
            return
        print("  A consultar a memória (RAG)...")
        try:
            r = _rag().query_memoria(pergunta, k=5, sintetizar=True)
        except Exception as e:
            logger.warning("[INTERFACE] Erro no RAG: %s", e)
            print(f"  Não foi possível consultar a memória: {e}")
            return
        print(f"\n  {r.get('resposta', '(sem resposta)')}")
        ids = r.get("inspection_ids_citados", [])
        if ids:
            print(f"\n  Inspeções referenciadas: {', '.join(ids)}")

    # ---------- compare ----------
    def cmd_compare(self, tokens: list):
        pos, flags = _parse_args(tokens[1:])
        if len(pos) < 2:
            print('  Uso: compare <zonaA> <zonaB> --period "last 7 days"')
            return
        zona_a, zona_b = pos[0], pos[1]
        dias = _periodo_para_dias(flags.get("period"))

        todas = self.inspecoes_sessao + _carregar_inspecoes_dataset()
        todas = _filtrar_por_periodo(todas, dias)

        def _resumo(zona):
            insps = [i for i in todas if i.get("zone_id") == zona]
            if not insps:
                return None
            n = len(insps)
            fills = [i.get("shelf_fill_rate", 0) for i in insps]
            fill_medio = sum(fills) / n if n else 0
            criticos = sum(1 for i in insps if i.get("overall_status") == "critical")
            warnings = sum(1 for i in insps if i.get("overall_status") == "warning")
            n_issues = sum(len(i.get("issues", [])) for i in insps)
            return {"n": n, "fill": fill_medio, "crit": criticos,
                    "warn": warnings, "issues": n_issues}

        ra, rb = _resumo(zona_a), _resumo(zona_b)
        periodo_txt = f"últimos {dias} dias" if dias else "todo o histórico"
        print(f"\n  Comparação ({periodo_txt}):")
        print(f"  {'':14}{zona_a:>14}{zona_b:>14}")
        if ra is None and rb is None:
            print("  Sem dados para nenhuma das zonas.")
            return
        ra = ra or {"n": 0, "fill": 0, "crit": 0, "warn": 0, "issues": 0}
        rb = rb or {"n": 0, "fill": 0, "crit": 0, "warn": 0, "issues": 0}
        print(f"  {'inspeções':14}{ra['n']:>14}{rb['n']:>14}")
        print(f"  {'fill médio':14}{ra['fill']:>13.0%}{rb['fill']:>14.0%}")
        print(f"  {'críticos':14}{ra['crit']:>14}{rb['crit']:>14}")
        print(f"  {'warnings':14}{ra['warn']:>14}{rb['warn']:>14}")
        print(f"  {'total issues':14}{ra['issues']:>14}{rb['issues']:>14}")

    # ---------- report ----------
    def cmd_report(self, tokens: list):
        pos, flags = _parse_args(tokens[1:])
        dias = _periodo_para_dias(flags.get("period"))
        zona = flags.get("zone")

        # Fonte das inspeções: sessão se houver, senão o dataset processado.
        if self.inspecoes_sessao:
            inspeccoes = list(self.inspecoes_sessao)
            origem = "sessão atual"
        else:
            inspeccoes = _carregar_inspecoes_dataset()
            origem = "dataset processado"

        if zona and zona is not True:
            inspeccoes = [i for i in inspeccoes if i.get("zone_id") == zona]
        if dias is not None:
            inspeccoes = _filtrar_por_periodo(inspeccoes, dias)

        if not inspeccoes:
            print("  Não há inspeções para gerar o relatório "
                  f"(fonte: {origem}{', zona ' + zona if zona and zona is not True else ''}).")
            return

        print(f"  A gerar relatório ({len(inspeccoes)} inspeções, fonte: {origem})...")
        try:
            res = _report().gerar_relatorio(inspeccoes)
        except Exception as e:
            logger.warning("[INTERFACE] Erro a gerar relatório: %s", e)
            print(f"  Não foi possível gerar o relatório: {e}")
            return
        print(f"  [OK] Relatório gerado: {res.get('caminho_ficheiro')}")
        print(f"       Sumário: {res.get('n_palavras_sumario')} palavras | "
              f"{len(res.get('alertas', []))} alerta(s).")

    # ---------- ajuda / estado ----------
    def cmd_ajuda(self, tokens: list):
        print(_AJUDA)

    def cmd_estado(self, tokens: list):
        print(f"  Inspeções nesta sessão : {len(self.inspecoes_sessao)}")
        print(f"  Regras guardadas       : {len(listar_regras())}")
        print(f"  Regra pendente         : "
              f"{self.regra_pendente.get('rule_id') if self.regra_pendente else 'nenhuma'}")
        print(f"  Modelo                 : {MODEL_NAME} "
              f"({'quota esgotada' if gemini_client.quota_esgotada else 'quota disponível'})")


# ==========================================
# DISPATCH
# ==========================================

_AJUDA = """
Comandos disponíveis:
  inspect <zona> --image <ficheiro>           Inspeciona uma imagem
  inspect all --images-dir <pasta>            Inspeciona todas as imagens de uma pasta
  add rule "<regra>"                          Cria uma regra em linguagem natural
  resolve "<respostas>"                       Clarifica a última regra ambígua
  list rules                                  Lista as regras guardadas
  delete rule <RULE_ID>                       Elimina uma regra
  test rule <RULE_ID> --image <ficheiro>      Testa (dry-run) uma regra numa imagem
  history "<pergunta>"                        Consulta o histórico (RAG)
  compare <zonaA> <zonaB> --period "last 7 days"   Compara duas zonas
  report --session today                      Gera relatório da sessão
  report --zone <zona> --period "last 14 days"     Gera relatório filtrado
  status                                      Mostra o estado da sessão
  help                                        Mostra esta ajuda
  exit                                        Sai
"""


def _dispatch(sessao: Sessao, linha: str) -> bool:
    """
    Executa uma linha de comando. Devolve False se for para terminar.
    Todo o corpo está protegido — NUNCA propaga exceções ao utilizador.
    """
    linha = linha.strip()
    if not linha:
        return True

    try:
        tokens = shlex.split(linha)
    except ValueError:
        # aspas não fechadas, etc. — tenta split simples
        tokens = linha.split()
    if not tokens:
        return True

    cmd = tokens[0].lower()
    sub = tokens[1].lower() if len(tokens) > 1 else ""

    try:
        if cmd in ("exit", "quit", "sair"):
            return False
        elif cmd in ("help", "ajuda", "?"):
            sessao.cmd_ajuda(tokens)
        elif cmd in ("status", "estado"):
            sessao.cmd_estado(tokens)
        elif cmd == "inspect":
            sessao.cmd_inspect(tokens)
        elif cmd == "add" and sub == "rule":
            sessao.cmd_add_rule(tokens)
        elif cmd == "resolve":
            sessao.cmd_resolve(tokens)
        elif cmd == "cancel":
            sessao.cmd_cancel(tokens)
        elif cmd == "list" and sub == "rules":
            sessao.cmd_list_rules(tokens)
        elif cmd == "delete" and sub == "rule":
            sessao.cmd_delete_rule(tokens)
        elif cmd == "test" and sub == "rule":
            sessao.cmd_test_rule(tokens)
        elif cmd == "history":
            sessao.cmd_history(tokens)
        elif cmd == "compare":
            sessao.cmd_compare(tokens)
        elif cmd == "report":
            sessao.cmd_report(tokens)
        else:
            print(f"  Comando não reconhecido: '{linha}'. Escreve 'help' para a lista.")
    except Exception as e:
        # Rede de segurança final — o enunciado proíbe stack traces ao utilizador.
        logger.error("[INTERFACE] Erro a executar '%s': %s", linha, e, exc_info=True)
        print(f"  [ERRO] Não foi possível executar o comando: {e}")

    return True


def _banner():
    print("=" * 64)
    print("   RETAIL VISION INTELLIGENCE SYSTEM — Interface CLI".center(64))
    print("=" * 64)
    print(f"  Modelo: {MODEL_NAME}")
    print("  Escreve 'help' para a lista de comandos, 'exit' para sair.")
    print("=" * 64)


def main():
    sessao = Sessao()

    # Modo não-interativo: executa o comando passado em argv e sai.
    # (útil para scripting e testes: python src/interface.py inspect all --images-dir ...)
    if len(sys.argv) > 1:
        linha = " ".join(
            f'"{a}"' if " " in a else a for a in sys.argv[1:]
        )
        _dispatch(sessao, linha)
        return

    # Modo interativo (REPL).
    _banner()
    while True:
        try:
            linha = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nAté à próxima.")
            break
        if not _dispatch(sessao, linha):
            print("Até à próxima.")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Mesmo o arranque está protegido.
        logger.error("[INTERFACE] Erro fatal: %s", e, exc_info=True)
        print(f"[ERRO] Falha no arranque da interface: {e}")
        sys.exit(1)
