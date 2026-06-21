"""Punto de entrada del Tetris instrumentado.

Uso:
    python main.py [--condition easy|hard|ramp]

Ejecuta una sesión completa con prompts para covariables de estado,
asignación de condición, bloque de juego y esfuerzo percibido.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Intentar forzar UTF-8 en la consola de Windows para evitar errores con acentos/flechas.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Asegurar que src está en el path cuando se ejecuta directamente.
sys.path.insert(0, str(Path(__file__).parent))

from src.session_manager import CONDITIONS, run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Tetris instrumentado")
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        help="Condición experimental (si no se indica, se sugiere por contrabalanceo)",
    )
    args = parser.parse_args()
    run_session(condition=args.condition)


if __name__ == "__main__":
    main()
