# Blueprint — Tetris como laboratorio de interacción agente/campo

**Destinatario:** CC (implementación) y registro de diseño del programa.
**Estatus de fases:** Fase 0 y Fase 1 se especifican para implementar. Fase 2 queda **solo enunciada** (preguntas candidatas, sin diseño) porque depende de lo que Fase 1 establezca como medible. Diseñarla ahora sería especificar análisis para observables no validadas.
**Decisión de frontera (tomada):** ESTRECHA. Ver §0.1.
**Marco heredado:** este programa deja de tratar a Tetris como confirmatorio de PUBG. Tetris se estudia como objeto propio. Consecuencia directa: el null de PUBG (partículas) no se hereda; se construye un null propio de Tetris (§0.4), sin el cual las mediciones son descriptivas, no evidencia de agencia.

---

## FASE 0 — Definiciones (fundacional)

Nada en Fase 1 se implementa hasta que estas definiciones estén fijadas. El error que esta fase evita es medir antes de definir qué se mide.

### 0.1. Frontera campo / agente — ESTRECHA

- **Campo** ≔ secuencia de piezas (tipo + orden) + curva de gravedad. Exógeno puro: ni el orden de piezas ni la velocidad dependen de lo que hace el agente. Es lo único que cuenta como "campo".
- **Tablero** ≔ estado co-construido. Se **registra completo** (cada celda, en cada snapshot) pero **NO es campo**: es producto de las acciones del agente. Queda disponible como variable de estado/covariable, fuera de la definición de campo.
- **Agente** ≔ el jugador humano.

Razón de la elección: exogeneidad limpia (no hay endogeneidad como en el tablero-co-construido ni como en el campo estratégico de LoL), null construible, y toda "respuesta al campo" es inatacable. Costo aceptado: se deja fuera, por ahora, la interacción agente↔tablero (gestión del estado que el agente mismo produce). Puede ampliarse a frontera amplia en una fase posterior **solo si** Fase 1 lo justifica; no se asume desde el inicio.

### 0.2. Dimensiones del agente (no es una cosa; son varias, con operacionalizaciones distintas y probablemente no correlacionadas)

| Dimensión | Operacionalización candidata | Datos requeridos | ¿Medible hoy? |
|---|---|---|---|
| **Reacción / latencia motora** | `time_to_first_input_ms` por pieza | actions + pieces (ya existe) | Sí, con caveat físico (§1.4) |
| **Esfuerzo / actividad motora** | `n_inputs` por pieza | actions (ya existe) | Sí, con caveat de tiempo-disponible |
| **Reconocimiento / chunking perceptual** | latencia de identificación; ¿cae tras repeticiones del mismo tipo de pieza? (hipótesis de color, §0.3) | secuencia de tipos + color por tipo + first_input | Parcial; requiere test específico |
| **Anticipación / inferencia de piezas** | ¿la colocación de la pieza actual deja el tablero mejor dispuesto para la pieza que efectivamente viene que para una aleatoria? + ¿usa la estructura del 7-bag? | bag-state por spawn, board snapshots alineados, secuencia | No con el logging actual; requiere §1.5 |
| **Control / precisión** | ¿el agente ejecuta la colocación que pretendía? | **requiere un modelo de intención** | Problemático — ver nota |
| **Volatilidad (σ)** | dispersión por ventana (CV o residualizada, NUNCA absoluta) | métricas por pieza | Sí, pero el piloto la encontró **plana**; baja prioridad |

**Nota sobre Control/precisión:** es la dimensión más débil del menú. "Ejecutar lo que se pretendía" exige inferir la intención, que no está en los datos. Sin un modelo externo de colocación-objetivo (p.ej. un solver que define la "mejor" jugada y se mide desviación respecto a ella — pero entonces se mide desviación-respecto-al-solver, no intención real), no es operacionalizable de forma limpia. **Marcar como no medible en Fase 1** salvo que se acepte un proxy explícito y sus supuestos.

**Descartado:** "agrupación por colores" como *mecánica* (no existe objetivo de color en Tetris). Pero ver §0.3: el color sí entra como canal perceptual, distinto de una mecánica.

### 0.3. El color como canal perceptual (corrección incorporada)

En Tetris estándar (guideline) cada tipo de tetromino tiene color fijo (I=cian, O=amarillo, T=púrpura, S=verde, Z=rojo, J=azul, L=naranja). Por tanto el color es un **codificador redundante del tipo de pieza**, no decoración. Hipótesis a registrar (no a asumir): bajo presión de velocidad, el agente identifica por color más rápido que parseando la forma, y secuencias del mismo tipo consecutivo producen un atajo (chunking).

- **Requisito de implementación:** confirmar que el Tetris propio asigna **color fijo por tipo** (estándar). Si no, esta dimensión no aplica.
- **Test concreto (exploratorio):** ¿`time_to_first_input` o la latencia de identificación cae tras runs del mismo tipo de pieza en la secuencia? Es comprobable con los datos actuales + la secuencia de tipos.
- Estatus: hipótesis legítima sobre *cómo* el agente procesa el campo, candidata de Fase 1, no mecánica.

### 0.4. El null (transversal, no es una pregunta de fase — es el instrumento)

Sin un baseline, "el agente reacciona/anticipa/mejora" no tiene referencia: parte de lo que parece agencia es estructura del juego. Para la frontera estrecha el null es construible:

- **Agente-null** ≔ una política algorítmica que juega **sin modelar el campo**: no usa el preview ni la estructura del 7-bag; coloca por heurística fija simple (p.ej. minimizar huecos y altura agregada). Determinista o con ruido controlado.
- Función: para cada dimensión del agente, el null fija el nivel que se obtiene **por estructura del Tetris sola**, sin anticipación ni modelo. La señal humana es interesante solo en la medida en que **excede** al null.
- Caso crítico — anticipación: un colocador heurístico ciego *parecerá* anticipar, porque buenas colocaciones genéricas acomodan bien cualquier pieza siguiente. Si el humano no supera al null ciego en la medida de anticipación, lo que se medía era la estructura del juego, no agencia. **Por eso el null es parte de la definición de la observable de anticipación, no un add-on.**
- Implementación: el null corre sobre las **mismas secuencias de piezas (misma semilla)** y la misma curva de gravedad que el humano, para que el campo sea idéntico.

### 0.5. Aprendizaje como variable de primer orden (confound estructural)

La habilidad del agente cambia entre sesiones, correlacionada con el tiempo → confound monótono que contamina cualquier medida acumulada, y ataca especialmente a la anticipación (que *es* la habilidad que se aprende).

- **Pregunta empírica previa (barata, hacer primero):** ¿el agente mejora siquiera, a su nivel actual, en la escala temporal del estudio? Graficar score/líneas por sesión. Si la curva ya está cuasi-plana, el confound es menor y se simplifica todo. **No teorizar sobre aprendizaje antes de mirar esta curva.**
- "Habilidad" tampoco es una cosa: se puede mejorar en reacción, anticipación o control por separado. Registrar un proxy de habilidad por sesión (p.ej. líneas/min, score normalizado) como **eje**, no como nuisance.
- Decisión de régimen (define el diseño de recolección, elegir conscientemente):
  - **Meseta:** jugar hasta estabilizar habilidad, recién entonces recolectar como experto estable; sesiones = réplicas que se acumulan.
  - **Curva:** hacer del aprendizaje el objeto; medir cómo la anticipación crece con la práctica; sesiones = puntos de una curva, habilidad como eje.
  - **Prohibido:** mezclar sesiones de habilidad creciente y promediarlas como si midieran lo mismo. Produce un número sin significado (mismo error que mover dos variables a la vez).

---

## FASE 1 — ¿Qué es medible y existe sobre el null?

Objetivo: para cada dimensión definida en Fase 0 que sea medible hoy, establecer (a) si hay señal por encima del null, y (b) si cambia con la práctica. No es confirmatorio; es el cribado que decide qué pasa a Fase 2.

### 1.1. Estructura

Para cada dimensión medible {reacción, esfuerzo, reconocimiento/color, anticipación si §1.5 se implementa}:
1. Definición operacional congelada (antes de mirar datos).
2. Cómputo de la métrica para el humano.
3. Cómputo de la misma métrica para el agente-null sobre las mismas semillas.
4. Contraste humano vs null: ¿excede? ¿en qué régimen de gravedad?
5. ¿La métrica (y su exceso sobre el null) cambia con la habilidad por sesión?

### 1.2. Disciplina de recolección (lecciones del piloto, no negociables)

- **Una variable a la vez.** Estado fijo entre sesiones: misma franja horaria, cafeína fija (preferible 0), hidratación/sueño anotados. La condición es lo único que se mueve. (El esfuerzo 8→2 ininterpretable vino de mover rampa + cafeína juntas.)
- **Rampa sin meseta.** La curva debe seguir subiendo durante la mayor parte de las piezas (o la partida termina al saturar, o la rampa es bastante más lenta que la duración típica). Objetivo: ventanas repartidas a lo largo de la gravedad, no apiladas en el tope. (La rampa de 120 s saturaba a los 2 min y dejaba el 60% de ventanas en g=6.)
- **Excluir la primera pieza de cada partida** (su `time_to_first_input` incluye reacción al arranque). Categórico en el pipeline. Revisar efecto análogo tras game over/pausa.
- Marcar `software_git_hash` siempre; no mezclar hashes sin marcar.

### 1.3. σ — estatus rebajado

El piloto, con rampa larga y ventanas distribuidas, encontró σ **plano**: el CV se mueve por la media (fantasma media-varianza un nivel arriba), pero la σ **residualizada** no muestra tendencia en ninguna métrica. Conclusión del piloto: la firma de volatilidad de PUBG **no se replica** en Tetris. En Fase 1, σ se reporta solo en CV y residualizado, nunca absoluto, y con baja prioridad. No es la observable central de Tetris.

### 1.4. Caveat físico (bug corregido — no repetir)

`board_height()` devolvía la altura invertida; corregido, el control físico pasó de r≈0.016 ("hermético", falso) a r=0.351 (p<0.001), parcial controlando `stack_height` r=0.154 (p=0.030). Implicaciones:
- `time_to_first_input` **no es hermética**; tiene residuo físico real. Reportarla como "respuesta sobre el nivel, parcialmente mecánica", controlando `stack_height` por correlación parcial. No volver a llamarla limpia sin el control.
- Auditar si la altura invertida tocó otras métricas/filtros además del control físico.
- El Análisis-1 sobre medias (efecto de campo sobre el nivel) hay que **recalcularlo** con `board_height` corregido y reportarlo controlando `stack_height`. Es probable que sobreviva (la parcial sigue significativa) pero con tamaño menor.

### 1.5. Logging adicional requerido (ajustes al juego para habilitar anticipación)

El setup actual NO permite medir anticipación. Para habilitarla:
- **Bag-state por spawn:** qué piezas quedan en la bolsa 7-bag vigente en cada aparición. Es el futuro objetivamente predecible contra el cual se mide si el agente anticipa. (Hoy probablemente no se registra.)
- **Board snapshots completos y alineados** por pieza (estado antes y después de cada lock), verificando alineación con la secuencia.
- **Preview = 1** (decidido): suficiente para jugar, obliga a que la anticipación venga del modelo del bag y no de leer muchas piezas adelante. Subir el preview trivializaría la predicción.
- Color fijo por tipo confirmado (§0.3).

### 1.6. Advertencia de potencia (inferencia, no certeza)

Las medidas de anticipación son informacionales y, **es probable que**, sean mucho más hambrientas de datos que medias o desviaciones. Con ~100–470 piezas por sesión **puede no haber potencia** para estimar anticipación de forma estable. **Antes** de construir la maquinaria informacional completa, estimar la potencia: simular cuántas piezas/sesiones se necesitan para distinguir humano de null en la medida de anticipación con el efecto esperado. Si el volumen requerido es inviable, replantear la observable. No invertir en el análisis informacional sin este cálculo previo.

---

## FASE 2 — Solo enunciada (preguntas candidatas, SIN diseño)

Se listan para registrar dirección, no para implementar. Cada una se diseña solo si Fase 1 confirma que su observable base es medible y excede al null. Reformuladas como comparaciones, no como "¿existe el fenómeno?":

- **Anticipación / información predictiva:** ¿el agente coloca anticipando el futuro del campo (bag/preview) por encima del null ciego? (mitad **informacional** de Still — predicción vs nostalgia —, no la termodinámica, que es inalcanzable sin disipación). Probablemente la observable central de Tetris.
- **Estructura temporal:** ¿algún modelo dinámico (O/U u otro) describe mejor que un baseline más simple alguna serie del agente? Comparación de modelos, no descubrimiento. No asumir O/U.
- **Memoria / autocorrelación** en las series conductuales.
- **Chunking perceptual por color** (§0.3) como atajo de reconocimiento bajo presión.
- **Curva de aprendizaje** de cada dimensión (si §0.5 elige el régimen "curva"): cómo el agente construye su modelo del campo con la práctica.
- **Optimalidad:** ¿hay comportamiento óptimo del agente, o la interacción agente-campo es la que optimiza? — aterrizar en algo medible antes de tratarla; tal como está es semi-filosófica.

---

## Orden de ejecución sugerido

1. Mirar la curva de habilidad por sesión (§0.5) — barato, decide el régimen.
2. Implementar el agente-null (§0.4) y el logging adicional (§1.5).
3. Recalcular Análisis-1 con `board_height` corregido + control parcial (§1.4); auditar el alcance del bug.
4. Recolección disciplinada (§1.2) para el cribado de dimensiones (§1.1).
5. Cálculo de potencia para anticipación (§1.6) antes de construir su análisis.
6. Solo entonces, abrir Fase 2 sobre lo que Fase 1 dejó en pie.
