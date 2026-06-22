"""Motor determinista de Tetris.

Implementa las reglas del juego, el generador 7-bag con semilla,
SRS con wall kicks estándar, gravedad exógena al tiempo, lock delay fijo
y scoring básico. No depende de pygame ni de I/O.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constantes del juego
# ---------------------------------------------------------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

LOCK_DELAY_MS = 500.0  # Fijo, sin reset infinito, idéntico entre condiciones.

# Piezas con sus matrices de rotación SRS.
# Cada matriz usa 1 para celdas ocupadas, 0 para vacías.
# Origen de rotación implícito en las matrices (tamaño 3x3 salvo I 4x4, O 2x2).
PIECES: Dict[str, List[List[List[int]]]] = {
    "I": [
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    ],
    "O": [
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
    ],
    "T": [
        [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
    ],
    "S": [
        [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ],
    "Z": [
        [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    "J": [
        [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 0]],
    ],
    "L": [
        [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 0]],
    ],
}

# Wall kicks SRS. Formato: (delta_x, delta_y).
# y positivo hacia abajo en este motor.
JLSTZ_WALL_KICKS: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}

I_WALL_KICKS: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}

# Scoring estándar por líneas limpiadas de una sola vez.
LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}

# Curvas de gravedad (celdas por segundo).
EASY_GRAVITY_CPS = 0.8
HARD_GRAVITY_CPS = 3.5
RAMP_START_CPS = 0.8
RAMP_END_CPS = 6.0
RAMP_DURATION_MS = 300_000.0


def get_ramp_curve_id() -> str:
    return "ramp_v1"


def get_ramp_curve_params() -> dict:
    return {
        "start_cps": RAMP_START_CPS,
        "end_cps": RAMP_END_CPS,
        "duration_ms": RAMP_DURATION_MS,
        "kind": "linear",
    }


def gravity_for_condition(condition: str, t_ms: float) -> float:
    """Devuelve celdas por segundo para la condición en el tiempo dado."""
    if condition == "easy":
        return EASY_GRAVITY_CPS
    if condition == "hard":
        return HARD_GRAVITY_CPS
    if condition == "ramp":
        frac = min(1.0, t_ms / RAMP_DURATION_MS)
        return RAMP_START_CPS + (RAMP_END_CPS - RAMP_START_CPS) * frac
    raise ValueError(f"Condición desconocida: {condition}")


# ---------------------------------------------------------------------------
# Generador 7-bag
# ---------------------------------------------------------------------------
class SevenBagGenerator:
    """Generador determinista 7-bag."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.bag: List[str] = []
        self.bag_counter = 0  # índice global de pieza dentro del flujo

    def _refill(self) -> None:
        self.bag = ["I", "O", "T", "S", "Z", "J", "L"]
        self.rng.shuffle(self.bag)

    def next(self) -> Tuple[str, int]:
        if not self.bag:
            self._refill()
        piece_type = self.bag.pop(0)
        bag_idx = self.bag_counter % 7
        self.bag_counter += 1
        return piece_type, bag_idx

    def preview_sequence(self, n: int) -> List[str]:
        """Devuelve las próximas n piezas sin avanzar el generador."""
        result = []
        temp_bag = list(self.bag)
        temp_rng_state = self.rng.getstate()
        while len(result) < n:
            if not temp_bag:
                temp_bag = ["I", "O", "T", "S", "Z", "J", "L"]
                # Para reproducir el shuffle necesitamos usar un rng temporal
                # con el mismo estado. Pero modificar temp_bag afecta al rng.
                # Esta función es solo informativa; para preview de 1 no se usa.
                # Implementación simplificada:
                temp_rng = random.Random(0)
                temp_rng.setstate(temp_rng_state)
                temp_rng.shuffle(temp_bag)
                temp_rng_state = temp_rng.getstate()
            result.append(temp_bag.pop(0))
        return result


# ---------------------------------------------------------------------------
# Estado del juego
# ---------------------------------------------------------------------------
@dataclass
class ActivePiece:
    piece_type: str
    x: int
    y: float  # posición vertical en celdas (puede ser fraccionaria durante caída)
    rotation: int  # 0, 1, 2, 3
    t_spawn_ms: float
    piece_idx: int
    bag_idx: int
    n_inputs: int = 0
    # El lock delay se inicia cuando la pieza toca superficie.
    lock_timer_ms: Optional[float] = None
    lowest_y: float = field(init=False)  # para detectar si bajó mientras tocaba
    # Acumulador de gravedad (celdas fraccionarias).
    gravity_accumulator: float = 0.0

    def __post_init__(self):
        self.lowest_y = self.y

    def matrix(self) -> List[List[int]]:
        return PIECES[self.piece_type][self.rotation]


@dataclass
class GameResult:
    duration_ms: float
    total_pieces: int
    total_lines: int
    score: int
    game_over_reason: str


class TetrisCoreGame:
    """Motor determinista de Tetris.

    El tiempo se maneja en milisegundos monotónicos desde el inicio de la partida.
    La gravedad depende exclusivamente de `condition` y `t_ms`.
    """

    def __init__(
        self,
        condition: str,
        seed: int,
        wall_clock_start: Optional[str] = None,
    ):
        self.condition = condition
        self.seed = seed
        self.wall_clock_start = wall_clock_start or time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime()
        )
        self.generator = SevenBagGenerator(seed)
        self.board: List[List[Optional[str]]] = [
            [None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)
        ]

        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.game_over_reason: Optional[str] = None
        self.start_t_ms: float = 0.0
        self.current_t_ms: float = 0.0
        self.piece_idx_counter = 0

        self.current_piece: Optional[ActivePiece] = None
        self.next_piece_type: Optional[str] = None
        self.preview: List[str] = []
        self.generated_sequence: List[str] = []
        self._last_locked: Optional[Dict[str, Any]] = None

        self._spawn_next_piece(0.0)

    # -----------------------------------------------------------------------
    # Helpers de colisión
    # -----------------------------------------------------------------------
    def _cells(self, piece: ActivePiece, x: Optional[int] = None, y: Optional[float] = None, rotation: Optional[int] = None) -> List[Tuple[int, int]]:
        """Devuelve las coordenadas (col, row) ocupadas por la pieza."""
        x = x if x is not None else piece.x
        y = y if y is not None else piece.y
        rot = rotation if rotation is not None else piece.rotation
        matrix = PIECES[piece.piece_type][rot]
        cells = []
        for r, row in enumerate(matrix):
            for c, val in enumerate(row):
                if val:
                    cells.append((int(x + c), int(y + r)))
        return cells

    def _valid(self, piece: ActivePiece, x: Optional[int] = None, y: Optional[float] = None, rotation: Optional[int] = None) -> bool:
        for cx, cy in self._cells(piece, x, y, rotation):
            if cx < 0 or cx >= BOARD_WIDTH or cy >= BOARD_HEIGHT:
                return False
            if cy >= 0 and self.board[cy][cx] is not None:
                return False
        return True

    def _is_on_surface(self, piece: ActivePiece) -> bool:
        """True si mover la pieza 1 celda hacia abajo colisiona."""
        return not self._valid(piece, y=piece.y + 1)

    # -----------------------------------------------------------------------
    # Spawn
    # -----------------------------------------------------------------------
    def _spawn_next_piece(self, t_ms: float) -> None:
        if self.next_piece_type is None:
            self.next_piece_type, _ = self.generator.next()

        piece_type = self.next_piece_type
        self.next_piece_type, bag_idx = self.generator.next()
        self.preview = [self.next_piece_type]

        # Posición de spawn estándar SRS: centrada arriba.
        matrix = PIECES[piece_type][0]
        width = len(matrix[0])
        spawn_x = (BOARD_WIDTH - width) // 2
        spawn_y = 0

        piece = ActivePiece(
            piece_type=piece_type,
            x=spawn_x,
            y=float(spawn_y),
            rotation=0,
            t_spawn_ms=t_ms,
            piece_idx=self.piece_idx_counter,
            bag_idx=bag_idx,
        )
        self.generated_sequence.append(piece_type)

        if not self._valid(piece):
            # Colisión al spawnear: game over (topout).
            self.game_over = True
            self.game_over_reason = "topout"
            self.current_piece = piece
            return

        self.current_piece = piece
        self.piece_idx_counter += 1
        return piece

    # -----------------------------------------------------------------------
    # Acciones del jugador
    # -----------------------------------------------------------------------
    def move_left(self, t_ms: float) -> bool:
        return self._try_move(-1, 0, t_ms)

    def move_right(self, t_ms: float) -> bool:
        return self._try_move(1, 0, t_ms)

    def soft_drop(self, t_ms: float) -> bool:
        return self._try_move(0, 1, t_ms)

    def hard_drop(self, t_ms: float) -> None:
        piece = self.current_piece
        if piece is None or self.game_over:
            return
        while self._valid(piece, y=piece.y + 1):
            piece.y += 1
        piece.n_inputs += 1
        self._lock_piece(t_ms)

    def rotate_cw(self, t_ms: float) -> bool:
        return self._try_rotation(+1, t_ms)

    def rotate_ccw(self, t_ms: float) -> bool:
        return self._try_rotation(-1, t_ms)

    def _try_move(self, dx: int, dy: int, t_ms: float) -> bool:
        piece = self.current_piece
        if piece is None or self.game_over:
            return False
        new_x = piece.x + dx
        new_y = piece.y + dy
        if self._valid(piece, x=new_x, y=new_y):
            piece.x = new_x
            piece.y = new_y
            piece.n_inputs += 1
            self._update_lock_state(piece, t_ms)
            return True
        return False

    def _try_rotation(self, direction: int, t_ms: float) -> bool:
        piece = self.current_piece
        if piece is None or self.game_over or piece.piece_type == "O":
            return False
        new_rotation = (piece.rotation + direction) % 4
        kicks = self._wall_kicks(piece.rotation, new_rotation, piece.piece_type)
        for dx, dy in kicks:
            if self._valid(piece, x=piece.x + dx, y=piece.y + dy, rotation=new_rotation):
                piece.x += dx
                piece.y += dy
                piece.rotation = new_rotation
                piece.n_inputs += 1
                self._update_lock_state(piece, t_ms)
                return True
        return False

    def _wall_kicks(self, old_rot: int, new_rot: int, piece_type: str) -> List[Tuple[int, int]]:
        table = I_WALL_KICKS if piece_type == "I" else JLSTZ_WALL_KICKS
        return table.get((old_rot, new_rot), [(0, 0)])

    # -----------------------------------------------------------------------
    # Lock delay
    # -----------------------------------------------------------------------
    def _update_lock_state(self, piece: ActivePiece, t_ms: float) -> None:
        """Actualiza el temporizador de lock.

        Lock delay fijo, sin reset infinito. Si la pieza toca superficie,
        se inicia (o continúa) el temporizador. Si la pieza baja mientras
        toca, el temporizador sigue corriendo; no se reinicia.
        """
        on_surface = self._is_on_surface(piece)
        if on_surface:
            if piece.lock_timer_ms is None:
                piece.lock_timer_ms = t_ms + LOCK_DELAY_MS
            # Actualizar lowest_y solo como referencia; no reinicia timer.
            piece.lowest_y = max(piece.lowest_y, piece.y)
        else:
            # Si deja de tocar superficie, cancelamos el timer.
            piece.lock_timer_ms = None

    def _lock_piece(self, t_ms: float) -> None:
        piece = self.current_piece
        if piece is None or self.game_over:
            return
        for cx, cy in self._cells(piece):
            if 0 <= cy < BOARD_HEIGHT and 0 <= cx < BOARD_WIDTH:
                self.board[cy][cx] = piece.piece_type
        cleared = self._clear_lines()
        self._award_score(cleared)
        self._last_locked = {
            "piece_idx": piece.piece_idx,
            "piece_type": piece.piece_type,
            "bag_idx": piece.bag_idx,
            "t_spawn_ms": piece.t_spawn_ms,
            "t_lock_ms": t_ms,
            "n_inputs": piece.n_inputs,
            "final_x": piece.x,
            "final_y": piece.y,
            "final_rot": piece.rotation,
            "lines_cleared_by_lock": cleared,
        }
        self._spawn_next_piece(t_ms)

    def _clear_lines(self) -> int:
        new_board = [row for row in self.board if any(cell is None for cell in row)]
        cleared = BOARD_HEIGHT - len(new_board)
        for _ in range(cleared):
            new_board.insert(0, [None for _ in range(BOARD_WIDTH)])
        self.board = new_board
        self.lines_cleared += cleared
        return cleared

    def _award_score(self, lines: int) -> None:
        if lines in LINE_SCORES:
            self.score += LINE_SCORES[lines]

    # -----------------------------------------------------------------------
    # Actualización por tiempo (gravedad)
    # -----------------------------------------------------------------------
    def update(self, t_ms: float) -> None:
        """Actualiza el estado del juego hasta el tiempo t_ms."""
        if self.game_over:
            return
        dt_ms = t_ms - self.current_t_ms
        self.current_t_ms = t_ms
        piece = self.current_piece
        if piece is None:
            return

        # Gravedad: celdas por segundo.
        gravity = gravity_for_condition(self.condition, t_ms)
        if gravity <= 0:
            return

        # Acumulamos celdas fraccionarias y bajamos enteras.
        piece.gravity_accumulator += gravity * (dt_ms / 1000.0)
        while piece.gravity_accumulator >= 1.0:
            if self._valid(piece, y=piece.y + 1):
                piece.y += 1
                piece.gravity_accumulator -= 1.0
            else:
                # Chocó con superficie: consumimos el acumulador para no repetir.
                piece.gravity_accumulator = 0.0
                break

        # Si está en superficie, gestionamos lock delay.
        if self._is_on_surface(piece):
            if piece.lock_timer_ms is None:
                piece.lock_timer_ms = t_ms + LOCK_DELAY_MS
            if t_ms >= piece.lock_timer_ms:
                self._lock_piece(t_ms)
        else:
            piece.lock_timer_ms = None

    # -----------------------------------------------------------------------
    # Consultas
    # -----------------------------------------------------------------------
    def get_current_piece_state(self) -> Optional[Dict]:
        piece = self.current_piece
        if piece is None:
            return None
        return {
            "piece_type": piece.piece_type,
            "x": piece.x,
            "y": piece.y,
            "rotation": piece.rotation,
            "piece_idx": piece.piece_idx,
        }

    def get_board(self) -> List[List[Optional[str]]]:
        # Devuelve una copia superficial.
        return [row[:] for row in self.board]

    def result(self) -> GameResult:
        return GameResult(
            duration_ms=self.current_t_ms,
            total_pieces=self.piece_idx_counter,
            total_lines=self.lines_cleared,
            score=self.score,
            game_over_reason=self.game_over_reason or "unknown",
        )

    def preview_at_spawn(self) -> str:
        """Devuelve el tipo de pieza en preview (único)."""
        return self.preview[0] if self.preview else ""

    def consume_last_locked(self) -> Optional[Dict[str, Any]]:
        """Devuelve los datos del último lock y los limpia."""
        data = self._last_locked
        self._last_locked = None
        return data


# ---------------------------------------------------------------------------
# Replay determinista
# ---------------------------------------------------------------------------
def replay_game(condition: str, seed: int, actions: List[Dict]) -> TetrisCoreGame:
    """Reproduce una partida a partir de seed y acciones con timestamps."""
    game = TetrisCoreGame(condition=condition, seed=seed)
    for action in sorted(actions, key=lambda a: a["t_ms"]):
        t_ms = action["t_ms"]
        game.update(t_ms)
        act = action["action"]
        if act == "move_left":
            game.move_left(t_ms)
        elif act == "move_right":
            game.move_right(t_ms)
        elif act == "rotate_cw":
            game.rotate_cw(t_ms)
        elif act == "rotate_ccw":
            game.rotate_ccw(t_ms)
        elif act == "soft_drop":
            game.soft_drop(t_ms)
        elif act == "hard_drop":
            game.hard_drop(t_ms)
    # Actualiza hasta el final de la última acción para que termine el juego si corresponde.
    if actions:
        game.update(actions[-1]["t_ms"] + 1000)
    return game
