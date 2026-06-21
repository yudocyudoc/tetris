# Blueprint — Tetris instrumentado para validación del σ y observable de predicción/nostalgia

**Destinatario:** CC (implementación)
**Tipo de estudio:** idiográfico N=1, piloto. No es el estudio confirmatorio; su objetivo es validar el paradigma y el pipeline antes de escalar.
**Lenguaje:** Python.
**Principio rector:** registrar la señal **fina** del campo y de la conducta, evento a evento, nunca agregados. Reproducibilidad total vía semilla. El wearable NO se integra al juego.

---

## 0. Qué medimos y qué NO

**Observables objetivo (ambas conductuales, ninguna necesita el reloj):**

1. **σ conductual** — volatilidad de las decisiones del jugador bajo la contracción del campo. Es la réplica, en el espacio de decisiones de Tetris, del σ = log(movement_var) validado en PUBG. Operacionalización: dispersión, por ventana temporal, del *tiempo de decisión por pieza* y del *número de inputs por pieza*. Es el resultado robusto del piloto.

2. **Predicción / nostalgia (exploratoria, estilo Still informacional)** — cuánta información del pasado del campo (piezas ya vistas) predice las decisiones actuales frente a cuánta se retiene sin valor predictivo. Requiere la secuencia de piezas, el preview y todas las acciones. Es la mitad *informacional* de Still, no la termodinámica; no se mide costo.

**Fuera de alcance:** fisiología. El Garmin corre aparte y solo aporta covariables de **estado basal** (sueño, HRV de reposo nocturno) que se anexan después por timestamp. El código del juego no habla con el reloj.

---

## 1. Decisiones de diseño del juego (no negociables — son el resultado de descartar 10 datasets)

### 1.1. El campo = velocidad de caída como función EXÓGENA del tiempo
- La velocidad de gravedad sigue una **curva fija del tiempo transcurrido**, idéntica entre sesiones de la misma condición.
- **PROHIBIDO** que la velocidad suba por líneas completadas o por puntuación. Eso introduce endogeneidad (la habilidad del jugador aceleraría el campo) y contamina la separación campo↔respuesta. La velocidad depende solo del reloj de la partida.
- La curva debe estar **parametrizada y versionada** (un `ramp_curve_id` con sus parámetros) para que "Ramp" sea reproducible bit a bit.

### 1.2. El driving signal estocástico = secuencia de piezas
- Generador **7-bag** (cada "bolsa" contiene las 7 piezas en orden aleatorio). **No usar generador uniforme IID.**
  - Razón crítica para la observable 2: con 7-bag el pasado **sí** predice el futuro (las piezas no salidas de la bolsa están por venir), así que existe información predictiva genuina que el jugador puede o no explotar. Con uniforme IID el pasado no predice nada y la observable de predicción/nostalgia se vuelve degenerada (toda memoria sería no-predictiva por construcción).
- La **semilla** del RNG se fija por partida, se registra, y debe permitir regenerar la secuencia completa de forma determinista.
- Registrar además la secuencia explícita (no confiar solo en la semilla; redundancia deliberada).

### 1.3. Preview
- **Preview = 1 pieza** (se muestra solo la siguiente). Suficiente para jugabilidad; obliga a que la anticipación dependa de la estructura del bag (memoria) y no de ver muchas piezas adelante, que trivializaría la predicción.
- **Hold deshabilitado** en el piloto (simplifica el espacio de acciones y la definición de "decisión por pieza"). Si se habilita después, debe loguearse como acción.

### 1.4. Condiciones
| Condición | Velocidad |
|---|---|
| `easy` | constante baja |
| `hard` | constante alta |
| `ramp` | empieza baja, sube monótonamente por la curva fija hasta muy alta |

- **Sin dual-task, sin tareas secundarias, sin alarmas.** (Lección directa de Dybvik et al. 2025: el dual-task y las alarmas contaminan la atribución de la respuesta al campo.) Tetris puro.
- Las tres condiciones comparten todo lo demás (generador, preview, controles, render).

---

## 2. Esquema de datos — el corazón del blueprint

Un **directorio por sesión**: `data/<session_id>/`. Dentro, los archivos de abajo. Una sesión contiene varias **partidas** (games); cada partida tiene su `game_id` y su propia semilla.

**Reglas transversales:**
- Timestamps intra-partida (`t_ms`): reloj **monotónico** (`time.perf_counter_ns`), en milisegundos desde el inicio de la partida. Nunca usar `time.time()` para esto (puede saltar).
- Cada partida registra también un `wall_clock_start` (ISO 8601 con zona horaria) para poder anexar después datos del Garmin por timestamp.
- I/O: bufferizar eventos en memoria y volcar a disco **al final de cada partida**. Nunca escribir a disco por frame (el lag arruinaría los tiempos de decisión, que son la señal).

### 2.1. `session_meta.json` (uno por sesión)
```json
{
  "session_id": "uuid",
  "wall_clock_start": "2026-06-21T20:14:03-06:00",
  "condition": "ramp",
  "software_git_hash": "abc1234",
  "config": {
    "generator": "7bag",
    "preview_count": 1,
    "hold_enabled": false,
    "ramp_curve_id": "ramp_v1",
    "ramp_curve_params": { "...": "..." },
    "easy_gravity": 0.0,
    "hard_gravity": 0.0
  },
  "state_covariates": {
    "sleep_hours": 0.0,
    "caffeine_mg": 0,
    "minutes_since_last_meal": 0,
    "hydration_subjective_1_5": 3,
    "notes": ""
  },
  "perceived_effort_1_10": null
}
```
- `state_covariates` se piden por prompt al **inicio** de la sesión. `perceived_effort_1_10` se pide al **final**.
- `ramp_curve_params`, `easy_gravity`, `hard_gravity`: lo necesario para reconstruir el campo de cada condición sin ambigüedad.

### 2.2. `pieces.csv` (una fila por pieza spawneada) — clave para σ y para la observable
| Campo | Descripción |
|---|---|
| `game_id` | id de la partida |
| `piece_idx` | índice secuencial de la pieza dentro de la partida (0,1,2,…) |
| `piece_type` | I/O/T/S/Z/J/L |
| `bag_idx` | posición dentro de la bolsa 7-bag (0–6) |
| `preview_at_spawn` | tipo(s) de pieza visible(s) en el preview al spawnear |
| `t_spawn_ms` | timestamp de aparición |
| `t_lock_ms` | timestamp de fijación |
| `decision_time_ms` | `t_lock_ms − t_spawn_ms` ← **insumo directo del σ** |
| `n_inputs` | número de acciones del jugador para esta pieza ← **insumo directo del σ** |
| `final_x`, `final_y`, `final_rot` | posición/rotación de lock |
| `gravity_at_spawn` | velocidad vigente al spawnear (valor de la curva en t) |
| `lines_cleared_by_lock` | líneas eliminadas por esta fijación |

### 2.3. `actions.csv` (una fila por input del jugador) — para la observable de predicción/nostalgia
| Campo | Descripción |
|---|---|
| `game_id` | id de la partida |
| `t_ms` | timestamp monotónico |
| `piece_idx` | pieza sobre la que actúa |
| `action` | `move_left`/`move_right`/`rotate_cw`/`rotate_ccw`/`soft_drop`/`hard_drop` (`hold` si se habilita) |
| `x`, `y`, `rot` | estado de la pieza tras la acción |

### 2.4. `board_snapshots.parquet` (un snapshot por lock; opcionalmente cada N ms)
| Campo | Descripción |
|---|---|
| `game_id`, `piece_idx`, `t_ms` | referencia temporal |
| `board` | estado del tablero (matriz 10×20 serializada, o lista de celdas ocupadas, o hash + RLE) |
- Mínimo: un snapshot tras cada lock. Permite reconstruir el estado del campo en cada decisión.

### 2.5. `piece_sequence.json` (una por partida)
```json
{ "game_id": "...", "seed": 123456789, "sequence": ["I","O","T", "..."] }
```
- La secuencia completa generada. Redundante con la semilla a propósito.

### 2.6. `game_events.csv` (una fila por evento de partida)
| Campo | Descripción |
|---|---|
| `game_id`, `t_ms` | referencia |
| `event` | `game_start`/`line_clear`/`game_over` |
| `detail` | p.ej. nº de líneas, razón de game over (topout) |

### 2.7. `games_summary.csv` (una fila por partida) — agregados, **último** en prioridad
`game_id`, `seed`, `condition`, `wall_clock_start`, `duration_ms`, `total_pieces`, `total_lines`, `score`, `game_over_reason`.
- Los agregados se guardan para conveniencia, pero **no sustituyen** los streams finos. Si en algún momento se siente la tentación de loguear solo esto, es exactamente el error que cometieron los 10 datasets revisados.

---

## 3. Definiciones operacionales (para que CC entienda para qué existe cada campo y no "optimice" eliminándolo)

- **σ por ventana:** se segmenta la partida en ventanas temporales; en cada ventana se calcula la dispersión (p.ej. varianza o IQR) de `decision_time_ms` y de `n_inputs` sobre las piezas de esa ventana. La predicción del programa CEI es σ↑ a medida que `gravity_at_spawn` crece (condición `ramp`), y σ mayor en `hard` que en `easy`. → **Requiere** `decision_time_ms`, `n_inputs`, `gravity_at_spawn`, `t_spawn_ms` por pieza. No eliminar ninguno.
- **Predicción vs nostalgia:** se relaciona la información en el historial de piezas (`piece_sequence`, `bag_idx`) y el preview (`preview_at_spawn`) con las decisiones (`actions`, posiciones de lock en `pieces`). La parte del historial que predice las colocaciones futuras es "predictiva"; la retenida sin valor predictivo es "nostalgia". → **Requiere** la secuencia completa, el preview por spawn, y todas las acciones con su `piece_idx`. No agregar ni resumir.

CC no necesita implementar estos análisis; solo **garantizar que los datos los soportan**.

---

## 4. Protocolo de sesión (lo que el software orquesta)

1. **Inicio de sesión:** prompt que pide `state_covariates` (sueño, cafeína, minutos desde última comida, hidratación 1–5, notas). Guardar en `session_meta.json`.
2. **Condición:** el software lleva un conteo de sesiones por condición y sugiere la siguiente para mantener el **contrabalanceo**; el experimentador puede confirmar/anular. Registrar la condición efectiva.
3. **Bloque de juego:** varias partidas cortas hasta game over, repitiendo, con un objetivo de ~15–20 min de juego activo por sesión. **Sesiones cortas, muchas** (lección Dybvik: las sesiones largas confunden el efecto del campo con la fatiga / U invertida).
4. **Fin de sesión:** prompt de `perceived_effort_1_10`. Cerrar y volcar todo.

Meta del piloto: ~10–15 sesiones por condición a lo largo de 2–3 semanas, con condiciones intercaladas, antes de decidir si se escala.

---

## 5. Reproducibilidad y validación

- **Modo replay:** dado `seed` + `actions.csv`, el motor debe reproducir la partida de forma determinista (verifica que el campo es función limpia de semilla + curva, sin estado oculto).
- **`software_git_hash`** en cada `session_meta`.
- **Script de validación de integridad** (`validate_session.py`): por cada sesión comprueba que (a) existen todos los streams, (b) los `t_ms` son monotónicos crecientes dentro de cada partida, (c) `sequence` es consistente con `seed`, (d) cada `piece_idx` en `actions` existe en `pieces`, (e) no hay `decision_time_ms` negativos o nulos. Falla ruidosamente si algo no cuadra.

---

## 6. Stack sugerido (ligero, sin sobre-ingeniería)

- **Pygame** para el juego (humano jugando con teclado; suficiente y directo).
- Lógica de gravedad desacoplada del framerate: la velocidad de caída se deriva de la curva en función de `t_ms`, no del número de frames.
- Logging a buffers en memoria → volcado por partida (`pandas`/`pyarrow` para parquet, `csv`/`json` estándar para el resto).
- Un módulo `tetris_core` (reglas, generador 7-bag con semilla, motor determinista) separado de `tetris_ui` (pygame, input, render) y de `logger` (esquema de datos). Esto permite el modo replay headless reusando `tetris_core`.
- **Cero dependencias del wearable** en el código del juego.

---

## 7. Anti-patrones (qué NO hacer — cada uno es una lección pagada)

1. Acelerar por líneas o score (endogeneidad). La velocidad depende solo del tiempo.
2. Guardar solo agregados (score/líneas/tiempo). Guardar la señal fina por pieza y por acción.
3. Generador uniforme IID (degenera la observable de predicción). Usar 7-bag.
4. Dual-task, distractores, sonidos punitivos. Tetris puro.
5. I/O bloqueante por frame. Bufferizar y volcar por partida.
6. Sesiones largas (confunden campo con fatiga). Cortas y muchas.
7. Acoplar el reloj al juego. El wearable es covariable de estado, externa y posterior.
8. Estado oculto no reproducible en el motor (rompe el replay). Todo deriva de semilla + curva + acciones.


## 8. Anexo

Para casi todo se debe asumir los estándares, con dos excepciones que parecen estándar pero tocan directamente la señal y no debes dejar que CC las resuelva por defecto.

Lo que es estándar e irrelevante para la medición, adelante sin consultar: tablero 10×20 (ya está implícito en los snapshots), controles de teclado (flechas + rotación), SRS con wall kicks como sistema de rotación, scoring, gravedad de línea al limpiar. Nada de eso afecta el σ ni la observable de piezas. La única regla: documenta *cuál* estándar usaste (p.ej. "SRS"), para que el replay sea reproducible. Qué teclas exactas, da igual.

Las dos que NO son inocentes, porque contaminan `decision_time_ms` y `n_inputs` —los insumos directos del σ—:

El **lock delay**. Si la pieza puede descansar sobre la superficie y moverse indefinidamente antes de fijarse (lock delay con reset infinito, que es lo "moderno"), entonces `decision_time_ms` se infla con jugueteo que no es decisión, y peor, se infla más cuando hay más tiempo, justo en `easy` —metiéndote una correlación espuria con la condición—. Decisión limpia: lock delay corto y fijo, sin reset infinito (o directamente fijación al tocar superficie, estilo clásico). Lo que elijas, que sea idéntico entre las tres condiciones y quede documentado.

El **auto-repeat (DAS/ARR)**. Si mantener la flecha desliza la pieza varias celdas, un solo gesto intencional genera muchos movimientos, y `n_inputs` deja de medir esfuerzo/volatilidad para medir cuánto tiempo apretaste. La solución no es desactivarlo (jugarías peor), es **loguear el input crudo**: registra `keydown`/`keyup` con timestamp como el evento primario, y deriva aparte el movimiento de celda. Así `n_inputs` puede definirse sobre pulsaciones intencionales (lo que quieres para el σ) en vez de sobre celdas desplazadas. Es un refinamiento del §2.3: la fila de `actions` debería ser la pulsación, no el movimiento resultante —o ambos, en streams separados—.

La heurística general, para que resuelvas las próximas tú mismo sin volver a preguntar: **¿este detalle toca `decision_time_ms`, `n_inputs`, la secuencia de piezas, el preview, o la velocidad-por-tiempo?** Si no —y la mayoría no—, asume el estándar y documéntalo. Si sí, es una decisión consciente que hay que fijar y dejar idéntica entre condiciones. Lock delay y auto-repeat caen del lado "sí"; el 99% de lo demás cae del lado "asume y sigue".

Si quieres, integro estas dos al blueprint para que CC tenga la fuente única y no las redescubra a mano —son las dos cosas que el documento daba por obvias y no lo eran—.