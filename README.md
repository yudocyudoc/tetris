# Tetris instrumentado

Implementación en Python de Tetris para un piloto idiográfico N=1 orientado a registrar la señal fina de la conducta bajo un campo exógeno controlado.

## Requisitos

- Python 3.10+
- Dependencias: `pygame`, `pandas`, `pyarrow`

## Instalación

```bash
python -m venv .venv
# En Windows:
.venv\Scripts\python -m pip install -r requirements.txt
# En Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
.venv\Scripts\python main.py
```

El programa guiará al experimentador con prompts para:
1. Covariables de estado basal (sueño, cafeína, etc.).
2. Confirmación/anulación de la condición sugerida (`easy`, `hard`, `ramp`).
3. Bloque de juego: juega partidas cortas hasta game over; indica si quieres otra.
4. Esfuerzo percibido al finalizar.

Los datos se guardan en `data/<session_id>/`.

## Validación

```bash
.venv\Scripts\python -m src.validate_session data/<session_id>
```

## Resumen consolidado de todas las sesiones

```bash
.venv\Scripts\python -m src.summarize_sessions
```

Genera `data/sessions_summary.json` y `data/sessions_summary.csv` con un overview por sesión (covariables, esfuerzo, número de partidas/piezas/líneas, duración, y estadísticas de `decision_time_ms` y `n_inputs`). Útil para compartir con Kimi Desktop u otras herramientas.

## Controles

- `←` / `→`: mover izquierda / derecha
- `↓`: soft drop
- `↑` o `Z`: rotar CW
- `X` o `Ctrl`: rotar CCW
- `Espacio`: hard drop

## Tests

```bash
.venv\Scripts\python tests/test_headless_session.py
.venv\Scripts\python tests/test_ui_lifecycle.py
.venv\Scripts\python tests/test_ui_input.py
```

## Estructura del código

- `src/tetris_core.py`: motor determinista, reglas, generador 7-bag, SRS.
- `src/logger.py`: buffers en memoria y volcado por partida.
- `src/tetris_ui.py`: pygame, captura de input crudo, DAS/ARR.
- `src/session_manager.py`: orquesta de sesión y prompts.
- `src/validate_session.py`: validación de integridad de datos.
