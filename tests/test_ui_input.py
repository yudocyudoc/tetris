"""Test de UI con input simulado.

Inicia pygame, envía eventos de teclado para mover/rotar/hard_drop varias piezas
y verifica que se generen actions.csv y pieces.csv correctamente.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame

from src.logger import SessionLogger
from src.tetris_core import TetrisCoreGame
from src.tetris_ui import TetrisUI


def main() -> None:
    session_id = "test_ui_input"
    session_dir = Path("data_test") / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    logger = SessionLogger(
        session_id=session_id,
        data_root="data_test",
        condition="easy",
        config={},
        state_covariates={},
    )
    game_id = "game_ui_input_0001"
    logger.start_game(
        game_id=game_id,
        seed=777,
        condition="easy",
        wall_clock_start="2026-01-01T00:00:00+00:00",
    )
    game = TetrisCoreGame(condition="easy", seed=777)
    ui = TetrisUI(game=game, logger=logger, game_id=game_id)

    # Secuencia de inputs: unos movimientos y hard drop repetidos.
    key_sequence = [
        pygame.K_LEFT,
        pygame.K_UP,
        pygame.K_SPACE,
    ]
    step_ms = 300

    start = time.time()
    running = True
    step_index = 0
    while running and time.time() - start < 10.0:
        t_ms = ui._t_ms()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulamos keydown/keyup cada step_ms.
        if t_ms >= step_index * step_ms and step_index < 30:
            key = key_sequence[step_index % len(key_sequence)]
            ui._handle_keydown(key, t_ms)
            ui._handle_keyup(key, t_ms)
            step_index += 1

        ui.game.update(t_ms)
        ui._update_das(t_ms)
        ui._check_and_log_lock(t_ms)
        ui._render()
        ui.clock.tick(60)

        if ui.game.game_over:
            running = False

    ui._end_game()
    logger.close_session(perceived_effort=3)
    print(f"UI input test OK. Datos en: {logger.session_dir}")

    from src.validate_session import validate_session
    validate_session(logger.session_dir)


if __name__ == "__main__":
    main()
