"""
utils/console.py
----------------
Torna o output de consola seguro em terminais cp1252 (Windows por defeito).

Sem isto, qualquer print com acentos ou símbolos (→, ═, ●, …) lança
UnicodeEncodeError e crasha o programa — exatamente o "stack trace exposto
ao utilizador" que o enunciado proíbe e que mataria a demonstração ao vivo.

Chamar configurar_consola() no topo de CADA ponto de entrada
(batch_processor.py, interface.py, evaluate.py).
"""

import sys


def configurar_consola() -> None:
    """Reconfigura stdout/stderr para UTF-8, tolerante a falhas de codificação."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # Stream não suporta reconfigure (ex: redirecionado por outro meio).
            pass
