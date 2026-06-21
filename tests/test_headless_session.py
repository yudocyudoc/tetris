"""Test headless de una sesión completa.

Crea un juego, aplica acciones deterministas, usa el logger para volcar datos
y finalmente ejecuta validate_session.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import SessionLogger
from src.tetris_core import TetrisCoreGame, gravity_for_condition
from src.validate_session import validate_session


def simulate_game(logger: SessionLogger, game_id: str, condition: str, seed: int) -> None:
    wall_clock_start = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
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


    t = 0.0
    for _ in range(80):
        t += 100.0
        game.update(t)
        if game.game_over:
            break
        # Jugada sencilla: mover un poco y hard drop.
        if game.current_piece:
            piece_idx = game.current_piece.piece_idx
            game.move_left(t)
            logger.log_action(
                game_id=game_id,
                t_ms=t,
                piece_idx=piece_idx,
                action="move_left",
                x=game.current_piece.x,
                y=game.current_piece.y,
                rot=game.current_piece.rotation,
            )
        t += 50.0
        game.update(t)
        if game.current_piece:
            piece_idx = game.current_piece.piece_idx
            game.rotate_cw(t)
            logger.log_action(
                game_id=game_id,
                t_ms=t,
                piece_idx=piece_idx,
                action="rotate_cw",
                x=game.current_piece.x,
                y=game.current_piece.y,
                rot=game.current_piece.rotation,
            )
        t += 50.0
        game.update(t)
        if game.current_piece:
            piece_idx = game.current_piece.piece_idx
            game.hard_drop(t)
            logger.log_action(
                game_id=game_id,
                t_ms=t,
                piece_idx=piece_idx,
                action="hard_drop",
                x=game.current_piece.x,
                y=game.current_piece.y,
                rot=game.current_piece.rotation,
            )

        # Registrar locks ocurridos.
        locked = game.consume_last_locked()
        while locked:
            logger.log_piece(
                game_id=game_id,
                piece_idx=locked["piece_idx"],
                piece_type=locked["piece_type"],
                bag_idx=locked["bag_idx"],
                preview_at_spawn=game.preview_at_spawn(),
                t_spawn_ms=locked["t_spawn_ms"],
                t_lock_ms=locked["t_lock_ms"],
                n_inputs=locked["n_inputs"],
                final_x=locked["final_x"],
                final_y=locked["final_y"],
                final_rot=locked["final_rot"],
                gravity_at_spawn=gravity_for_condition(condition, locked["t_spawn_ms"]),
                lines_cleared_by_lock=locked["lines_cleared_by_lock"],
            )
            logger.log_snapshot(
                game_id=game_id,
                piece_idx=locked["piece_idx"],
                t_ms=locked["t_lock_ms"],
                board=game.get_board(),
            )
            locked = game.consume_last_locked()

    result = game.result()
    logger.set_piece_sequence(game_id=game_id, seed=seed, sequence=game.generated_sequence)
    logger.log_event(
        game_id=game_id,
        t_ms=game.current_t_ms,
        event="game_over",
        detail=result.game_over_reason,
    )
    logger.end_game(
        game_id=game_id,
        duration_ms=result.duration_ms,
        total_pieces=result.total_pieces,
        total_lines=result.total_lines,
        score=result.score,
        game_over_reason=result.game_over_reason,
    )


def main() -> None:
    import shutil
    session_id = "test_session_001"
    session_dir = Path("data_test") / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    logger = SessionLogger(
        session_id=session_id,
        data_root="data_test",
        condition="easy",
        config={"generator": "7bag", "preview_count": 1, "hold_enabled": False},
        state_covariates={
            "sleep_hours": 7.0,
            "caffeine_mg": 100,
            "minutes_since_last_meal": 120,
            "hydration_subjective_1_5": 3,
            "notes": "test headless",
        },
    )
    simulate_game(logger, "game_0001", "easy", seed=12345)
    simulate_game(logger, "game_0002", "ramp", seed=67890)
    logger.close_session(perceived_effort=5)

    print(f"Datos guardados en: {logger.session_dir}")
    validate_session(logger.session_dir)


if __name__ == "__main__":
    main()
