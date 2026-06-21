"""Interfaz pygame del Tetris instrumentado.

Captura input crudo (keydown/keyup), implementa DAS/ARR, renderiza el juego
y orquesta las llamadas al motor y al logger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

from .tetris_core import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    TetrisCoreGame,
    gravity_for_condition,
)
from .logger import SessionLogger

# ---------------------------------------------------------------------------
# Configuración de UI
# ---------------------------------------------------------------------------
CELL_SIZE = 30
BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50
PREVIEW_OFFSET_X = BOARD_OFFSET_X + BOARD_WIDTH * CELL_SIZE + 40
PREVIEW_OFFSET_Y = BOARD_OFFSET_Y

COLORS = {
    "I": (0, 255, 255),
    "O": (255, 255, 0),
    "T": (128, 0, 128),
    "S": (0, 255, 0),
    "Z": (255, 0, 0),
    "J": (0, 0, 255),
    "L": (255, 165, 0),
    "grid": (50, 50, 50),
    "bg": (0, 0, 0),
    "text": (255, 255, 255),
}

# Timing de auto-repeat (DAS/ARR). Fijos e idénticos entre condiciones.
DAS_MS = 170.0
ARR_MS = 30.0

# Controles por defecto.
KEY_BINDINGS = {
    pygame.K_LEFT: "move_left",
    pygame.K_RIGHT: "move_right",
    pygame.K_DOWN: "soft_drop",
    pygame.K_UP: "rotate_cw",
    pygame.K_z: "rotate_cw",
    pygame.K_x: "rotate_ccw",
    pygame.K_LCTRL: "rotate_ccw",
    pygame.K_SPACE: "hard_drop",
}


@dataclass
class KeyState:
    action: str
    is_down: bool = False
    down_t_ms: float = 0.0
    last_repeat_t_ms: float = 0.0
    down_piece_idx: int = -1


class TetrisUI:
    """Envoltura pygame sobre TetrisCoreGame."""

    def __init__(
        self,
        game: TetrisCoreGame,
        logger: SessionLogger,
        game_id: str,
    ):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (
                PREVIEW_OFFSET_X + 8 * CELL_SIZE,
                BOARD_OFFSET_Y + BOARD_HEIGHT * CELL_SIZE + 100,
            )
        )
        pygame.display.set_caption("Tetris instrumentado")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)

        self.game = game
        self.logger = logger
        self.game_id = game_id

        self.start_perf_ns = time.perf_counter_ns()
        self.piece_spawn_time_ms = 0.0
        self.current_piece_idx = 0

        # Estado del input crudo.
        self.key_states: Dict[int, KeyState] = {}
        for key, action in KEY_BINDINGS.items():
            self.key_states[key] = KeyState(action=action)

        # Buffers temporales para snapshots.
        self._pending_snapshots: List[Dict[str, Any]] = []

        # Inicializar primera pieza.
        self._on_new_piece()

    def _t_ms(self) -> float:
        return (time.perf_counter_ns() - self.start_perf_ns) / 1_000_000.0

    def _on_new_piece(self) -> None:
        state = self.game.get_current_piece_state()
        if state is None:
            return
        self.current_piece_idx = state["piece_idx"]
        self.piece_spawn_time_ms = self.game.current_t_ms

    def _log_raw_key(self, key: int, event: str, t_ms: float) -> None:
        self._log_raw_key_with_piece_idx(key, event, t_ms, self.current_piece_idx)

    def _log_raw_key_with_piece_idx(self, key: int, event: str, t_ms: float, piece_idx: int) -> None:
        action_name = self.key_states[key].action
        self.logger.log_action(
            game_id=self.game_id,
            t_ms=t_ms,
            piece_idx=piece_idx,
            action=f"key_{event}",
            x=-1,
            y=-1,
            rot=-1,
            raw_key=action_name,
            key_event=event,
        )

    def _log_game_action(self, action: str, state_after: Dict[str, Any], t_ms: float) -> None:
        self.logger.log_action(
            game_id=self.game_id,
            t_ms=t_ms,
            piece_idx=self.current_piece_idx,
            action=action,
            x=state_after["x"],
            y=state_after["y"],
            rot=state_after["rotation"],
        )

    def _handle_keydown(self, key: int, t_ms: float) -> None:
        if key not in self.key_states:
            return
        ks = self.key_states[key]
        ks.is_down = True
        ks.down_t_ms = t_ms
        ks.last_repeat_t_ms = t_ms
        ks.down_piece_idx = self.current_piece_idx
        self._log_raw_key(key, "down", t_ms)
        self._execute_action(ks.action, t_ms, is_first=True)

    def _handle_keyup(self, key: int, t_ms: float) -> None:
        if key not in self.key_states:
            return
        ks = self.key_states[key]
        ks.is_down = False
        # El keyup se asocia a la pieza que estaba activa cuando se presionó.
        self._log_raw_key_with_piece_idx(key, "up", t_ms, ks.down_piece_idx)

    def _execute_action(self, action: str, t_ms: float, is_first: bool = False) -> None:
        before = self.game.get_current_piece_state()
        if before is None or self.game.game_over:
            return

        executed = False
        if action == "move_left":
            executed = self.game.move_left(t_ms)
        elif action == "move_right":
            executed = self.game.move_right(t_ms)
        elif action == "soft_drop":
            executed = self.game.soft_drop(t_ms)
        elif action == "rotate_cw":
            executed = self.game.rotate_cw(t_ms)
        elif action == "rotate_ccw":
            executed = self.game.rotate_ccw(t_ms)
        elif action == "hard_drop":
            self.game.hard_drop(t_ms)
            executed = True

        after = self.game.get_current_piece_state()
        if after is None:
            return

        # Si es hard_drop o un movimiento/rotación efectivo, logueamos la acción de juego.
        if executed or action == "hard_drop":
            self._log_game_action(action, after, t_ms)

        self._check_and_log_lock(t_ms)

    def _check_and_log_lock(self, t_ms: float) -> None:
        """Registra la pieza que acaba de lockear (si hay) y el snapshot."""
        locked = self.game.consume_last_locked()
        if locked is None:
            return
        gravity = gravity_for_condition(self.game.condition, locked["t_spawn_ms"])
        # El preview visto al spawnear es la pieza que ahora es current (la siguiente).
        preview = self.game.preview_at_spawn()
        self.logger.log_piece(
            game_id=self.game_id,
            piece_idx=locked["piece_idx"],
            piece_type=locked["piece_type"],
            bag_idx=locked["bag_idx"],
            preview_at_spawn=preview,
            t_spawn_ms=locked["t_spawn_ms"],
            t_lock_ms=locked["t_lock_ms"],
            n_inputs=locked["n_inputs"],
            final_x=locked["final_x"],
            final_y=locked["final_y"],
            final_rot=locked["final_rot"],
            gravity_at_spawn=gravity,
            lines_cleared_by_lock=locked["lines_cleared_by_lock"],
        )
        state = self.game.get_current_piece_state()
        if state:
            self.logger.log_snapshot(
                game_id=self.game_id,
                piece_idx=state["piece_idx"],
                t_ms=locked["t_lock_ms"],
                board=self.game.get_board(),
            )
            self.current_piece_idx = state["piece_idx"]
            self.piece_spawn_time_ms = self.game.current_t_ms

    def _update_das(self, t_ms: float) -> None:
        for key, ks in self.key_states.items():
            if not ks.is_down:
                continue
            if ks.action in ("rotate_cw", "rotate_ccw", "hard_drop"):
                continue
            elapsed = t_ms - ks.down_t_ms
            if elapsed < DAS_MS:
                continue
            if t_ms - ks.last_repeat_t_ms >= ARR_MS:
                ks.last_repeat_t_ms = t_ms
                self._execute_action(ks.action, t_ms)

    def run(self) -> None:
        running = True
        while running:
            t_ms = self._t_ms()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key, t_ms)
                elif event.type == pygame.KEYUP:
                    self._handle_keyup(event.key, t_ms)

            self._update_das(t_ms)
            self.game.update(t_ms)
            self._check_and_log_lock(t_ms)

            self._render()
            self.clock.tick(60)

            if self.game.game_over:
                # Pequeña pausa para que el jugador vea el game over.
                time.sleep(1.0)
                running = False

        self._end_game()

    def _render(self) -> None:
        self.screen.fill(COLORS["bg"])

        # Dibujar tablero (líneas de cuadrícula).
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (BOARD_OFFSET_X + x * CELL_SIZE, BOARD_OFFSET_Y),
                (BOARD_OFFSET_X + x * CELL_SIZE, BOARD_OFFSET_Y + BOARD_HEIGHT * CELL_SIZE),
            )
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (BOARD_OFFSET_X, BOARD_OFFSET_Y + y * CELL_SIZE),
                (BOARD_OFFSET_X + BOARD_WIDTH * CELL_SIZE, BOARD_OFFSET_Y + y * CELL_SIZE),
            )

        # Celdas fijadas.
        board = self.game.get_board()
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_cell(x, y, COLORS[cell])

        # Pieza actual.
        state = self.game.get_current_piece_state()
        if state:
            matrix = self.game.current_piece.matrix()
            color = COLORS[state["piece_type"]]
            for r, row in enumerate(matrix):
                for c, val in enumerate(row):
                    if val:
                        self._draw_cell(int(state["x"] + c), int(state["y"] + r), color)

        # Preview.
        preview = self.game.preview
        if preview:
            label = self.font.render("Next:", True, COLORS["text"])
            self.screen.blit(label, (PREVIEW_OFFSET_X, PREVIEW_OFFSET_Y))
            self._draw_preview_piece(preview[0], PREVIEW_OFFSET_X, PREVIEW_OFFSET_Y + 30)

        # HUD.
        hud_y = BOARD_OFFSET_Y + BOARD_HEIGHT * CELL_SIZE + 20
        texts = [
            f"Lines: {self.game.lines_cleared}",
            f"Score: {self.game.score}",
            f"Pieces: {self.game.piece_idx_counter}",
            f"Time: {self.game.current_t_ms / 1000:.1f}s",
        ]
        for text in texts:
            surface = self.font.render(text, True, COLORS["text"])
            self.screen.blit(surface, (BOARD_OFFSET_X, hud_y))
            hud_y += 24

        pygame.display.flip()

    def _draw_cell(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            rect = pygame.Rect(
                BOARD_OFFSET_X + x * CELL_SIZE + 1,
                BOARD_OFFSET_Y + y * CELL_SIZE + 1,
                CELL_SIZE - 2,
                CELL_SIZE - 2,
            )
            pygame.draw.rect(self.screen, color, rect)

    def _draw_preview_piece(self, piece_type: str, px: int, py: int) -> None:
        from .tetris_core import PIECES
        matrix = PIECES[piece_type][0]
        color = COLORS[piece_type]
        for r, row in enumerate(matrix):
            for c, val in enumerate(row):
                if val:
                    rect = pygame.Rect(
                        px + c * CELL_SIZE + 1,
                        py + r * CELL_SIZE + 1,
                        CELL_SIZE - 2,
                        CELL_SIZE - 2,
                    )
                    pygame.draw.rect(self.screen, color, rect)

    def _end_game(self) -> None:
        # Loguear la última pieza si no se logueó.
        self._check_and_log_lock(self.game.current_t_ms)
        pygame.quit()
        result = self.game.result()
        self.logger.set_piece_sequence(
            game_id=self.game_id,
            seed=self.game.seed,
            sequence=self.game.generated_sequence,
        )
        self.logger.log_event(
            game_id=self.game_id,
            t_ms=self.game.current_t_ms,
            event="game_over",
            detail=result.game_over_reason,
        )
        self.logger.end_game(
            game_id=self.game_id,
            duration_ms=result.duration_ms,
            total_pieces=result.total_pieces,
            total_lines=result.total_lines,
            score=result.score,
            game_over_reason=result.game_over_reason,
        )
