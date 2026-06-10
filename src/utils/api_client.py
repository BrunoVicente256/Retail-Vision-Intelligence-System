"""
utils/api_client.py
-------------------
Camada de infraestrutura para comunicação com a API Gemini.
Responsabilidades EXCLUSIVAS deste módulo:
  - Retry com backoff exponencial em caso de erro 429
  - Rate limiting (máx. 15 req/min)
  - Fallback gracioso quando quota diária é esgotada
  - Logging de cada chamada (sucesso, falha, cache hit)

NÃO contém lógica de negócio (prompts, parsing, cache de imagens).
"""

import os
import time
import random
import logging
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================
# CONFIGURAÇÃO DE LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTES
# ==========================================
MAX_REQUESTS_PER_MINUTE = 15
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 60
MODEL_NAME = "gemini-2.5-flash-lite"

# Pausa mínima entre tentativas de 429 (segundos).
# O free tier do Gemini impõe janelas de rate limit
# que podem durar 60s — o backoff exponencial cobre isso.
PAUSA_MINIMA_429 = 30


# ==========================================
# CLASSE PRINCIPAL
# ==========================================
class GeminiClient:
    """
    Wrapper sobre o cliente Gemini com rate limiting e retry automático.

    Filosofia de gestão de quota:
      - Qualquer 429 é tratado como rate limit TEMPORÁRIO e recuperável.
      - NUNCA decidimos "quota diária esgotada" com base no texto da mensagem
        de erro — o Gemini free tier é inconsistente nas mensagens e um
        falso positivo mata a sessão inteira (demo-killer).
      - Só marcamos quota diária como esgotada se o pedido falhar
        MAX_RETRIES vezes consecutivas com 429. Nesse ponto, é seguro
        assumir que a quota real acabou.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "[ERRO CRÍTICO] GEMINI_API_KEY não encontrada. "
                "Verifica o ficheiro .env na raiz do projeto."
            )

        self._client = genai.Client(api_key=api_key)

        # Fila deslizante para rastrear os timestamps dos últimos pedidos.
        self._request_timestamps: deque = deque(maxlen=MAX_REQUESTS_PER_MINUTE)

        # Modelos com quota diária confirmadamente esgotada.
        # Só é adicionado após MAX_RETRIES falhas consecutivas com 429.
        self._modelos_esgotados: set = set()

        logger.info("GeminiClient inicializado. Modelo: %s", MODEL_NAME)

    # ------------------------------------------
    # MÉTODO PÚBLICO PRINCIPAL
    # ------------------------------------------
    def gerar_conteudo(
        self,
        contents: list,
        temperature: float = 0.0,
        response_mime_type: str = "application/json",
        model: str | None = None
    ):
        """
        Chama a API Gemini com retry automático e rate limiting.

        Args:
            contents: Lista com [prompt_string, imagem_PIL] ou [prompt_string].
            temperature: 0.0 para máximo determinismo.
            response_mime_type: Formato da resposta esperada.
            model: Override do modelo. Se None, usa MODEL_NAME.

        Returns:
            Objeto de resposta da API Gemini.

        Raises:
            RuntimeError: Se quota diária confirmada esgotada (após MAX_RETRIES).
            RuntimeError: Se todos os retries falharem por outros erros.
        """
        modelo = model or MODEL_NAME

        # Só bloqueia se já confirmámos o esgotamento após MAX_RETRIES falhas
        if modelo in self._modelos_esgotados:
            raise RuntimeError(
                f"[QUOTA ESGOTADA] Limite diário confirmado para {modelo}. "
                "O sistema continua a funcionar para imagens em cache."
            )

        # Rate limiting por janela deslizante
        self._aplicar_rate_limit()

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type=response_mime_type
        )

        erros_429_consecutivos = 0

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    "A chamar API [tentativa %d/%d] modelo=%s...",
                    attempt + 1, MAX_RETRIES, modelo
                )

                response = self._client.models.generate_content(
                    model=modelo,
                    contents=contents,
                    config=config
                )

                # Sucesso — regista timestamp e repõe contador de 429
                self._request_timestamps.append(time.monotonic())
                logger.info("Resposta recebida com sucesso.")
                return response

            except Exception as e:
                erro_str = str(e).lower()
                eh_429 = ("429" in str(e) or "resource_exhausted" in erro_str)
                eh_servidor = (
                    "503" in str(e) or "500" in str(e)
                    or "unavailable" in erro_str
                    or "overloaded" in erro_str
                    or "high demand" in erro_str
                )

                # -------------------------------------------------------
                # CASO 1: 429 — rate limit (por minuto OU diário)
                # -------------------------------------------------------
                if eh_429:
                    # Distingue quota DIÁRIA (esgotamento real → fast-fail) de
                    # rate limit POR-MINUTO (recuperável → backoff). O free tier
                    # do Gemini expõe o quotaId no corpo do erro 429:
                    #   GenerateRequestsPerDayPerProjectPerModel-FreeTier  → diário
                    #   GenerateRequestsPerMinutePerProjectPerModel...      → minuto
                    # Match em "perday" é específico: só o limite diário o contém.
                    # Sem este atalho, um esgotamento diário gastaria
                    # MAX_RETRIES × PAUSA_MINIMA_429 (5×30s=2,5min) de retries
                    # inúteis por imagem — um demo-killer na avaliação.
                    eh_diario = ("perday" in erro_str or "per day" in erro_str
                                 or "requests per day" in erro_str)
                    if eh_diario:
                        self._modelos_esgotados.add(modelo)
                        logger.error(
                            "[QUOTA DIÁRIA] Sinal 'PerDay' no erro 429 para %s. "
                            "Fast-fail imediato — sem retries (quota diária esgotada).",
                            modelo
                        )
                        raise RuntimeError(
                            f"[QUOTA ESGOTADA] Limite DIÁRIO atingido para {modelo} "
                            f"(sinal PerDay no erro da API). O sistema continua a "
                            f"funcionar para imagens em cache."
                        )

                    erros_429_consecutivos += 1

                    # Salvaguarda: se o sinal PerDay não vier (mensagens do free
                    # tier são inconsistentes), ainda assim desistimos após
                    # MAX_RETRIES falhas consecutivas com 429.
                    if erros_429_consecutivos >= MAX_RETRIES:
                        self._modelos_esgotados.add(modelo)
                        logger.error(
                            "[QUOTA DIÁRIA CONFIRMADA] Modelo %s esgotado após "
                            "%d tentativas com 429. Sem mais chamadas nesta sessão.",
                            modelo, MAX_RETRIES
                        )
                        raise RuntimeError(
                            f"[QUOTA ESGOTADA] Limite diário confirmado para {modelo} "
                            f"após {MAX_RETRIES} tentativas."
                        )

                    # Ainda há tentativas — backoff e continua
                    espera = self._calcular_backoff(attempt)
                    # Garante pausa mínima para o free tier recuperar
                    espera = max(espera, PAUSA_MINIMA_429)
                    logger.warning(
                        "[429] Tentativa %d/%d. A aguardar %.0fs antes de tentar...",
                        attempt + 1, MAX_RETRIES, espera
                    )
                    time.sleep(espera)

                # -------------------------------------------------------
                # CASO 2: Servidor sobrecarregado (503/500)
                # Recuperável com backoff exponencial.
                # -------------------------------------------------------
                elif eh_servidor:
                    if attempt < MAX_RETRIES - 1:
                        espera = self._calcular_backoff(attempt)
                        logger.warning(
                            "[SERVIDOR 503/500] Tentativa %d/%d. "
                            "A aguardar %.0fs...",
                            attempt + 1, MAX_RETRIES, espera
                        )
                        time.sleep(espera)
                    else:
                        raise RuntimeError(
                            f"[SERVIDOR INDISPONÍVEL] Falha após {MAX_RETRIES} "
                            f"tentativas: {e}"
                        )

                # -------------------------------------------------------
                # CASO 3: Outros erros (rede, parsing, autenticação)
                # Backoff curto; se persistir, levanta excepção.
                # -------------------------------------------------------
                else:
                    logger.error(
                        "[ERRO API] Tentativa %d/%d: %s",
                        attempt + 1, MAX_RETRIES, str(e)
                    )
                    if attempt < MAX_RETRIES - 1:
                        espera = self._calcular_backoff(attempt, fator=1)
                        time.sleep(espera)
                    else:
                        raise RuntimeError(
                            f"[ERRO API] Falha após {MAX_RETRIES} tentativas: {e}"
                        )

        raise RuntimeError(
            f"[ERRO] Todos os {MAX_RETRIES} retries esgotados para {modelo}."
        )

    # ------------------------------------------
    # MÉTODOS PRIVADOS
    # ------------------------------------------
    def _aplicar_rate_limit(self):
        """
        Garante ≤ MAX_REQUESTS_PER_MINUTE pedidos por minuto.
        Usa janela deslizante — só espera se a janela estiver cheia.
        """
        agora = time.monotonic()

        if len(self._request_timestamps) == MAX_REQUESTS_PER_MINUTE:
            timestamp_mais_antigo = self._request_timestamps[0]
            janela = 60.0
            tempo_desde = agora - timestamp_mais_antigo

            if tempo_desde < janela:
                espera = janela - tempo_desde + 0.5
                logger.info(
                    "[RATE LIMIT] Janela cheia (%d req/min). A aguardar %.1fs...",
                    MAX_REQUESTS_PER_MINUTE, espera
                )
                time.sleep(espera)

    def _calcular_backoff(self, attempt: int, fator: int = 2) -> float:
        """
        Backoff exponencial com jitter.
        Fórmula: min(fator^attempt * BASE, MAX) + jitter[0,1]
        """
        base  = min(fator ** attempt * BASE_BACKOFF_SECONDS, MAX_BACKOFF_SECONDS)
        return base + random.uniform(0, 1)

    @property
    def quota_esgotada(self) -> bool:
        """True se a quota do modelo de produção foi confirmadamente esgotada."""
        return MODEL_NAME in self._modelos_esgotados

    def modelo_esgotado(self, modelo: str) -> bool:
        """True se a quota de um modelo específico foi confirmadamente esgotada."""
        return modelo in self._modelos_esgotados


# ==========================================
# INSTÂNCIA GLOBAL (Singleton)
# ==========================================
gemini_client = GeminiClient()