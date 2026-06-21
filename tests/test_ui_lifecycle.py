"""Test de ciclo de vida de la UI (sin interacción humana).

Inicia pygame, crea una partida, la deja correr unos segundos y cierra.
Sirve para detectar errores de importación/inicialización.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame

from src.logger import SessionLogger
from src.tetris_core import TetrisCoreGame
from src.tetris_ui import TetrisUI


def main() -> None:
    import shutil
    session_id = "test_ui_lifecycle"
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
    game_id = "game_ui_0001"
    logger.start_game(
        game_id=game_id,
        seed=999,
        condition="easy",
        wall_clock_start="2026-01-01T00:00:00+00:00",
    )
    game = TetrisCoreGame(condition="easy", seed=999)
    ui = TetrisUI(game=game, logger=logger, game_id=game_id)

    # Simulamos unos frames y cerramos.
    start = time.time()
    running = True
    while running and time.time() - start < 2.0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        ui.game.update(ui._t_ms())
        ui._check_and_log_lock(ui._t_ms())
        ui._render()
        ui.clock.tick(60)
    ui._end_game()
    logger.close_session(perceived_effort=1)
    print(f"UI test OK. Datos en: {logger.session_dir}")


if __name__ == "__main__":
    main()
