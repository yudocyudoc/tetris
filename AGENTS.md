# AGENTS.md — Tetris instrumentado

## Propósito del proyecto
Implementar un Tetris en Python para un piloto idiográfico N=1. El objetivo es registrar la señal fina de la conducta (tiempo de decisión por pieza, número de inputs, secuencia de acciones) bajo un campo exógeno controlado (velocidad de caída por tiempo).

## Principios rector
- Todo detalle que toque `decision_time_ms`, `n_inputs`, la secuencia de piezas, el preview o la velocidad-por-tiempo es una decisión consciente, idéntica entre condiciones y documentada.
- El 99% de los demás detalles se asume estándar y solo se documenta cuál estándar se usa.

## Decisiones de diseño fijadas (afectan la medición)

### Lock delay
- **Valor:** fijo de 500 ms.
- **Comportamiento:** la pieza se fija automáticamente al agotarse el temporizador, sin reset infinito por mover/rotar. Si toca superficie y el jugador sigue moviendo, el contador sigue corriendo; no se reinicia.
- **Razón:** evitar que `decision_time_ms` se infle con jugueteo en condiciones lentas (`easy`).

### Auto-repeat (DAS/ARR) y logging de input
- **DAS:** 170 ms (delay antes de empezar a repetir).
- **ARR:** 30 ms (intervalo entre repeticiones).
- **Logging:** el stream primario es el input crudo: `keydown` y `keyup` con timestamp. A partir de ahí se derivan los movimientos de celda.
- **Razón:** `n_inputs` debe reflejar pulsaciones intencionales, no celdas desplazadas.

### Generador de piezas
- **7-bag** con semilla fija por partida.

### Preview
- **1 pieza**.
- **Hold deshabilitado** en el piloto.

### Velocidad (campo)
- Función exógena del tiempo transcurrido.
- Condiciones: `easy` (constante baja), `hard` (constante alta), `ramp` (curva fija creciente).
- No sube por líneas ni score.

## Estándares asumidos (no afectan la medición)

- **Tablero:** 10 columnas × 20 filas.
- **Sistema de rotación:** SRS (Super Rotation System) con wall kicks estándar.
- **Piezas:** I, O, T, S, Z, J, L con sus matrices de rotación SRS.
- **Scoring:** sistema estándar de puntos por líneas (single, double, triple, tetris).
- **Gravedad de línea:** las líneas completas se limpian inmediatamente al lock.
- **Game over:** topout cuando una pieza nueva colisiona al spawnear.
- **Controles:** teclas configurables; por defecto:
  - ← / → : mover izquierda/derecha
  - ↓ : soft drop
  - ↑ o Z : rotar CW
  - X o Ctrl : rotar CCW
  - Espacio : hard drop

## Qué es la señal y qué no

- **`decision_time_ms = t_lock − t_spawn` es contexto, no señal principal.** Está contaminado por la gravedad: en `hard` (3.5 celdas/s) una pieza que cae 20 filas tarda ~5.7 s, que se convierte en techo artificial de `decision_time`. No mide deliberación, mide física de caída.
- **Eje principal del σ:** `n_inputs` (número de acciones de juego sobre la pieza). Es una medida conductual genuina, poco atada a la gravedad.
- **Métricas derivadas limpias de `actions.csv`:**
  - `time_to_first_input_ms`: latencia desde spawn hasta primer acción de juego.
  - `active_time_ms`: tiempo entre primera y última acción de juego (manipulación real).
  - `hard_drop_ratio`: proporción de piezas terminadas con hard drop.

Estas métricas se calculan offline en `src/metrics.py` y se incluyen en el resumen consolidado.

## Estructura del código
- `src/tetris_core.py`: motor determinista, reglas, generador 7-bag.
- `src/logger.py`: buffers en memoria y volcado por partida.
- `src/tetris_ui.py`: pygame, input crudo, render.
- `src/session_manager.py`: orquesta de sesión, prompts, contrabalanceo.
- `src/validate_session.py`: validación de integridad de datos.
- `src/metrics.py`: métricas conductuales derivadas de los streams finos.
- `src/summarize_sessions.py`: consolidado por sesión.
- `main.py`: punto de entrada.

## Convenciones
- Timestamps intra-partida (`t_ms`) con `time.perf_counter_ns`, en ms desde el inicio de la partida.
- I/O a disco solo al final de cada partida.
- Cero integración con wearable en el código del juego.
