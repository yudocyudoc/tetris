# Blueprint — Análisis exploratorio (fase piloto, N=1)

**Destinatario:** CC (implementación)
**Naturaleza:** EXPLORATORIO, no confirmatorio. Su única función es decidir si las métricas conductuales responden a la contracción del campo, con los datos que ya existen, antes de invertir en el sistema confirmatorio.
**Regla de oro:** este análisis no prueba nada; decide si vale la pena montar lo que sí probaría. No reportar p-valores como si fueran confirmatorios. No tomar decisiones de publicación con esto.

---

## 0. Qué decide este análisis

Una sola pregunta binaria: **¿las métricas conductuales limpias se mueven sistemáticamente cuando el campo se contrae?**

- Si **sí** (tendencia clara intra-ramp en la dirección predicha) → el paradigma funciona; pasar a Fase 2 (recolectar N, pre-registrar confirmatorio, diseñar el null, construir el sistema).
- Si **no** → reconsiderar la métrica, la calibración de velocidades, o el paradigma, **antes** de gastar en infraestructura.

---

## 1. Entradas

Por sesión: `actions.csv`, `pieces.csv`, `board_snapshots.parquet`, `game_events.csv`, `session_meta.json`.
Métrica por pieza (ya en `metrics.py`): `n_inputs`, `time_to_first_input_ms`, `active_time_ms`, más `gravity_at_spawn` (de `pieces.csv`).

Métrica primaria del piloto: **`time_to_first_input_ms`** — es la más limpia respecto a la física de caída (ocurre antes de que la gravedad domine). Secundaria: `n_inputs`. `active_time_ms` solo como apoyo (conserva contaminación residual de gravedad). `hard_drop_ratio` se ignora en este jugador (base ~0, sin poder discriminante).

---

## 2. Análisis 1 — EL central: trayectoria intra-ramp

Esto es el análogo directo del σ↑ de PUBG bajo contracción, y es la razón de ser del piloto.

1. Tomar la(s) sesión(es) `ramp`, a nivel **pieza**.
2. Ordenar/agrupar las piezas por `gravity_at_spawn` (la velocidad vigente al spawnear, que dentro de la rampa crece con el tiempo). Bins por velocidad (p.ej. 6–8 bins entre 0.8 y 6.0 cps) o por ventanas de tiempo.
3. Por bin: media y dispersión (IQR) de `time_to_first_input_ms` y `n_inputs`, y N de piezas en el bin.
4. **Predicción direccional, fijarla ANTES de mirar:** `time_to_first_input_ms` **decrece** al crecer `gravity_at_spawn` (más presión → reacción más rápida); `n_inputs` **decrece**.
5. Cuantificar la tendencia: regresión de la métrica sobre `gravity_at_spawn` a nivel pieza (o Spearman). Reportar pendiente + IC + N. Sin lenguaje confirmatorio.

### 2.1. Control obligatorio — separar campo de tiempo-disponible
El tiempo de vida de la pieza decrece con la gravedad, lo que **mecánicamente** limita `n_inputs` y `active_time`. Antes de atribuir cualquier caída al campo:
- Calcular, por pieza, el **tiempo de caída disponible** = tiempo que la pieza tardaría en tocar la pila solo por gravedad, dado `gravity_at_spawn` y la altura de la pila en ese momento (del `board_snapshot` previo).
- Para `n_inputs`: comprobar si la caída excede lo que el tiempo-disponible explica. Si `n_inputs` baja más abrupto que el techo físico de inputs (DAS/ARR permiten decenas de inputs en el tiempo disponible incluso a 6 cps), la caída es **conductual**. Si coincide con el techo físico, es **artefacto**.
- `time_to_first_input_ms` está casi libre de esto (ocurre temprano); por eso es la primaria.

---

## 3. Análisis 2 — comparación entre condiciones, controlada por estado

- Comparar **`easy` vs `ramp` de la misma noche/estado** (la comparación limpia). Medias de las métricas conductuales.
- Tratar `hard` **por separado** o excluirla: está confundida con madrugada + cafeína. No promediarla con las demás como si fuera solo "otra condición".
- N=1 sesión por condición: reportar como descriptivo, nunca como test confirmatorio.

---

## 4. Análisis 3 — el σ propiamente (volatilidad, no media)

Hasta ahora se miran medias; el σ de PUBG era **volatilidad**. Definir explícitamente:

- **σ-Tetris** := dispersión (varianza o IQR) de la métrica conductual elegida por **ventana temporal** (no por sesión completa).
- Métrica base del σ: decidir y documentar (candidata: `n_inputs` por ventana; alternativa `time_to_first_input_ms`).
- En `ramp`: ¿la volatilidad cambia **a lo largo** de la contracción? Reportar σ por bin de `gravity_at_spawn`.
- Reportar media **y** volatilidad por separado: pueden moverse en direcciones distintas (la media bajar mientras la volatilidad sube), y esa disociación es informativa.

---

## 5. Salidas

- Una tabla por análisis (bin × métrica × {media, IQR, N}).
- Gráficas: métrica vs `gravity_at_spawn` (Análisis 1), métrica por condición (Análisis 2), σ vs `gravity_at_spawn` (Análisis 3).
- Un `exploratory_report.md` con las tres tablas, las pendientes, y una línea de veredicto: ¿hay tendencia intra-ramp en la dirección predicha, sí/no?

---

## 6. Reglas de higiene

- Documentar la regla para piezas con `n_inputs ≤ 1` (donde `active_time` es indefinido): excluir o tratar como 0, explícito.
- Reportar siempre N por bin (con N pequeño, las medias por bin son ruidosas).
- No mezclar sesiones de distinto `software_git_hash` sin marcarlas.
- Marcar la sesión `hard` como `pre_decision_time_correction` / madrugada en cualquier figura.

---

## 7. Lo que queda EXPLÍCITAMENTE fuera (Fase 2, condicionada a que esto muestre señal)

- Modelos jerárquicos (múltiples sesiones/sujetos).
- **Diseño del null.** El σ de PUBG era fuerte por el null de partículas (humano vs null en el mismo campo). El equivalente en Tetris no existe aún y es no-trivial: ¿un agente que coloca piezas sin responder a la velocidad? ¿una política fija? Sin un null, "la métrica se mueve con la velocidad" puede ser mera mecánica de apuro, no agencia. **Este es el problema de diseño más importante de la Fase 2**, no del piloto.
- Pre-registro confirmatorio del análisis.
- Tests con pretensión confirmatoria, corrección por comparaciones múltiples, potencia.
- Dashboards o automatización.

No implementar nada de esta sección todavía. Si el Análisis 1 no muestra tendencia, nada de esto importa.

## Anexo
### conversación
Todavía no, y la razón es concreta, no un reflejo de prudencia: aún no has corrido ni una vez el análisis que decidiría qué vale la pena sistematizar. La trayectoria intra-ramp —el único análisis que te dice si hay señal— no la has mirado. Construir el sistema completo ahora es construir infraestructura para un fenómeno que no sabes si existe; si resulta que la métrica no responde dentro de la rampa, habrás automatizado el análisis de nada. Es el mismo orden que peleamos toda la conversación: valida el núcleo antes de montar el aparato.

Dicho eso, tu instinto tiene una mitad correcta que conviene rescatar. Pre-especificar el análisis antes de mirar los datos es buena práctica —es tu propio estándar de PUBG, el pre-registro contra el HARKing—. Pero eso es un *plan de análisis*, no un *sistema de software*, y la versión que toca ahora es exploratoria (decidir si hay señal con lo que ya tienes), no confirmatoria (esa llega con N suficiente y pre-registro formal). El sistema completo —modelos jerárquicos, null models, dashboards— pertenece a la fase confirmatoria, que requiere tres cosas que hoy no tienes: señal exploratoria confirmada, varias sesiones por condición en estado comparable, y el análisis pre-registrado. Montarlo antes es generalidad especulativa.

Lo que sí vale escribir ya es el blueprint del análisis *exploratorio* acotado: el que define exactamente qué computa el script que decide si el paradigma funciona, reproducible para CC, y honesto sobre lo que deja fuera. Eso te da rigor y reproducibilidad sin sobre-construir. Lo escribí.Lo que importa del blueprint, en orden de peso:

El Análisis 1 es el único que decide; los otros dos son contexto. Y dentro de él, el §2.1 es lo que separa un resultado de un artefacto: si `n_inputs` cae con la velocidad, tienes que demostrar que cae *más* de lo que el menor tiempo por pieza explica por sí solo, o no es conducta, es física otra vez. Por eso `time_to_first_input` es la primaria —es la que no necesita ese descargo—.

Y el punto que dejé para el final del documento a propósito, porque es el que va a determinar si todo esto vale como ciencia o solo como ingeniería: **el null.** Tu σ de PUBG era fuerte no porque subiera, sino porque las partículas en el mismo campo *no* lo reproducían. En Tetris todavía no tienes ese contraste, y sin él, "el jugador reacciona más rápido cuando la pieza cae más rápido" es una observación casi tautológica —apuro mecánico, no agencia—. Diseñar qué es un agente-null en Tetris (¿una política fija que coloca sin leer la velocidad? ¿un colocador heurístico insensible al reloj?) es el problema más difícil de la Fase 2, y es no-trivial. No lo toques en el piloto, pero tenlo presente: es lo que va a decidir si el σ-Tetris significa lo mismo que el σ-PUBG o solo se le parece.

Así que el camino queda: corre el Análisis 1 sobre la sesión `ramp` que ya tienes. Si `time_to_first_input` baja monótono con la velocidad y sobrevive al control del §2.1, el paradigma funciona y ahí sí se justifica todo lo demás —más sesiones en estado parejo, el null, el pre-registro, y entonces el sistema completo que preguntabas—. Si no baja, lo sabrás con el dato de una tarde en vez de después de construir la infraestructura. Ese es el ahorro de hacerlo en este orden.