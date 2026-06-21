"""Gestión de sesiones experimentales.

Orquesta el inicio de sesión (covariables de estado), la asignación de
condición con contrabalanceo, y la recolección de esfuerzo percibido al final.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import SessionLogger, _iso_with_tz
from .tetris_core import (
    EASY_GRAVITY_CPS,
    HARD_GRAVITY_CPS,
    RAMP_DURATION_MS,
    RAMP_END_CPS,
    RAMP_START_CPS,
    get_ramp_curve_id,
    get_ramp_curve_params,
)
from .tetris_ui import TetrisUI


CONDITIONS = ["easy", "hard", "ramp"]
COUNTER_FILE = Path("data") / ".session_counter.json"


def _load_counter() -> Dict[str, int]:
    if COUNTER_FILE.exists():
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {c: 0 for c in CONDITIONS}


def _save_counter(counter: Dict[str, int]) -> None:
    COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        json.dump(counter, f, indent=2)


def suggest_condition() -> str:
    """Sugiere la condición menos usada hasta ahora."""
    counter = _load_counter()
    return min(CONDITIONS, key=lambda c: counter[c])


def _prompt_int(message: str, min_val: int, max_val: int) -> int:
    while True:
        raw = input(f"{message} [{min_val}-{max_val}]: ").strip()
        try:
            value = int(raw)
            if min_val <= value <= max_val:
                return value
        except ValueError:
            pass
        print(f"Por favor introduce un número entre {min_val} y {max_val}.")


def _prompt_float(message: str) -> float:
    while True:
        raw = input(f"{message}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Por favor introduce un número.")


def _prompt_text(message: str) -> str:
    return input(f"{message}: ").strip()


def prompt_covariates() -> Dict[str, Any]:
    print("\n--- Covariables de estado basal ---")
    return {
        "sleep_hours": _prompt_float("Horas de sueño"),
        "caffeine_mg": _prompt_int("Cafeína (mg)", 0, 2000),
        "minutes_since_last_meal": _prompt_int("Minutos desde la última comida", 0, 1440),
        "hydration_subjective_1_5": _prompt_int("Hidratación subjetiva (1-5)", 1, 5),
        "notes": _prompt_text("Notas (opcional)"),
    }


def prompt_effort() -> int:
    print("\n--- Fin de sesión ---")
    return _prompt_int("Esfuerzo percibido (1-10)", 1, 10)


def build_config(condition: str) -> Dict[str, Any]:
    return {
        "generator": "7bag",
        "preview_count": 1,
        "hold_enabled": False,
        "ramp_curve_id": get_ramp_curve_id(),
        "ramp_curve_params": get_ramp_curve_params(),
        "easy_gravity": EASY_GRAVITY_CPS,
        "hard_gravity": HARD_GRAVITY_CPS,
        "lock_delay_ms": 500,
        "das_ms": 170,
        "arr_ms": 30,
        "rotation_system": "SRS",
        "board_size": {"width": 10, "height": 20},
    }


def run_session(condition: Optional[str] = None) -> str:
    """Ejecuta una sesión completa y devuelve el session_id."""
    if condition is None:
        suggested = suggest_condition()
        raw = input(
            f"Condición sugerida (contrabalanceo): {suggested}. "
            f"Presiona Enter para aceptar o escribe {CONDITIONS}: "
        ).strip()
        condition = raw if raw in CONDITIONS else suggested

    counter = _load_counter()
    counter[condition] += 1
    _save_counter(counter)

    covariates = prompt_covariates()
    config = build_config(condition)

    logger = SessionLogger(
        condition=condition,
        config=config,
        state_covariates=covariates,
    )

    print(f"\nSesión iniciada: {logger.session_id}")
    print(f"Condición: {condition}")
    print("Controles: LEFT/RIGHT mover, DOWN soft drop, UP/Z rotar CW, X/Ctrl rotar CCW, SPACE hard drop")
    print("Juega partidas cortas. Cierra la ventana para terminar la sesión.")

    game_count = 0
    while True:
        game_count += 1
        game_id = f"game_{game_count:04d}"
        seed = int(time.time() * 1000) % (2**31)
        wall_clock_start = _iso_with_tz()

        from .tetris_core import TetrisCoreGame

        logger.start_game(
            game_id=game_id,
            seed=seed,
            condition=condition,
            wall_clock_start=wall_clock_start,
        )
        game = TetrisCoreGame(
            condition=condition,
            seed=seed,
            wall_clock_start=wall_clock_start,
        )
        ui = TetrisUI(game=game, logger=logger, game_id=game_id)
        ui.run()

        print(f"Partida {game_count} finalizada.")
        cont = input("¿Jugar otra partida? (s/n): ").strip().lower()
        if cont != "s":
            break

    effort = prompt_effort()
    logger.close_session(perceived_effort=effort)
    print(f"Sesión guardada en: {logger.session_dir}")
    return logger.session_id


if __name__ == "__main__":
    run_session()
