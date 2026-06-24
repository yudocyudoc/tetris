# Traspaso — Programa Tetris como laboratorio de interacción agente/campo

**Propósito:** continuar el trabajo en un chat nuevo sin perder estado. Documento denso a propósito.
**Convención:** [HECHO] = establecido/verificado en este trabajo; [INFERENCIA] = razonado, no seguro; [DECISIÓN] = elección tomada; [ABIERTO] = sin resolver.

---

## 1. De dónde viene

- CEI es el marco del programa. El dominio original fue **PUBG**, donde se validó el **σ**: la volatilidad conductual (log de varianza de movimiento por ventana) **sube** cuando el campo (la zona) se contrae, y un **null de partículas calibrado** NO reproduce ese σ↑. [HECHO] Resultado sólido, limpio sobre todo en la transición 1→2; las fases tardías están contaminadas por survivorship (collider).
- Exploración intermedia de **Still** (termodinámica de la predicción). Conclusión: solo la mitad **informacional** (predicción vs "nostalgia" = información del pasado que no predice el futuro) es alcanzable en conducta; la mitad **termodinámica** (nostalgia = disipación) es inalcanzable sin calorímetro. [HECHO/conceptual]
- Búsqueda extensa de datasets con campo exógeno + costo fisiológico + dinámica + N (PUBG, MIDUS, HRV-Baigutanova, SSAQS, ExamStress-PhysioNet, eSports-Smerdov/LoL, CEPAV, Tetris-Dybvik, JSTAGE, etc.). **Ninguno tiene los ejes a la vez.** [HECHO] Razón estructural: los esports tienen campo *endógeno* (estado de juego co-creado por jugadores); los de workload no registran el campo fino. [INFERENCIA fundada]

---

## 2. Por qué Tetris, y desvinculado de PUBG

- [DECISIÓN] Tetris se estudia como **laboratorio propio** de interacción agente/campo, **NO como confirmatorio de PUBG**. Riesgo declarado y a vigilar: no convertir la no-replicación del σ en un comodín que siempre da la razón.
- Tetris es [INFERENCIA fundada] el mejor sustrato accesible: campo **exógeno** (velocidad por tiempo), **estocástico** (secuencia de piezas), **contractivo** (velocidad creciente), y **totalmente instrumentable** (lo controlas entero, sin API ni CDN como PUBG).
- Precio de desvincular: se pierde el null heredado de PUBG (partículas). El null de Tetris hay que construirlo y es el problema central abierto (§7).

---

## 3. El Tetris instrumentado (construido y funcionando)

Decisiones de diseño [DECISIÓN], cada una es una lección:
- **Velocidad de gravedad por TIEMPO transcurrido**, no por líneas. (Por líneas = endógeno; la habilidad aceleraría el campo.)
- **Generador 7-bag**, NO uniforme IID. Razón: con 7-bag el pasado predice el futuro (estructura de bolsa), condición necesaria para que la observable de predicción/nostalgia no sea degenerada.
- **Preview = 1**. Suficiente para jugar; obliga a que la anticipación venga del modelo del bag, no de leer muchas piezas adelante. Subirlo trivializa la predicción.
- **Sin dual-task** (lección de Dybvik 2025: dual-task + alarmas contaminan la atribución al campo).
- Condiciones: `easy` / `hard` (constantes) / `ramp` (contracción).
- **Telemetría fina** por pieza y por acción + board snapshots + secuencia con semilla. NO agregados.
- **Lock delay y auto-repeat (DAS/ARR)** contaminan `decision_time` y `n_inputs`: registrar input crudo (keydown/keyup) y derivar el movimiento aparte. Lock delay corto, fijo, idéntico entre condiciones.
- El **wearable (Garmin)** queda FUERA del juego: covariable de estado basal (sueño, HRV de reposo), no integrado. [DECISIÓN]

---

## 4. Hallazgos del piloto (N=1, ~4 sesiones)

- [HECHO] **`decision_time_ms` está contaminado por la física de caída** (mide cuánto tarda la gravedad en bajar la pieza, no deliberación). NO usar como señal. Sustituido por métricas conductuales: `n_inputs`, `time_to_first_input_ms`, `active_time_ms`, `hard_drop_ratio`.
- [HECHO] **Efecto sobre el NIVEL**: `time_to_first_input` cae al subir la gravedad (medias por bin). Robusto.
- [HECHO — corrección importante] **Bug: `board_height()` devolvía la altura invertida.** El control físico que parecía "hermético" (r≈0.016) era falso. Corregido: r=0.351 (p<0.001); parcial controlando `stack_height` r=0.154 (p=0.030). Implicación: `time_to_first_input` **NO es hermética**, tiene residuo físico real. Pendiente: recalcular Análisis-1 con `board_height` corregido y reportar controlando `stack_height`; auditar si el bug tocó otras métricas.
- [HECHO] **σ (volatilidad): plano.** La σ absoluta cae pero es artefacto media-varianza (piso en 0). En CV el efecto es la media arrastrando; en σ **residualizada** no hay tendencia en ninguna métrica. [INFERENCIA fundada] **La firma σ↑ de PUBG NO se replica en Tetris.** Tetris responde a la contracción en el **nivel**, no en la volatilidad.
- [HECHO] Confound recurrente: mover dos variables a la vez (rampa+cafeína; hora de día). El esfuerzo percibido 8→2 entre sesiones quedó ininterpretable por esto.
- [HECHO] La rampa de 120 s **saturaba** (meseta a velocidad máxima los últimos minutos), apilando ~60% de ventanas en la velocidad tope. Alargada a 300 s para distribuir ventanas. Efecto colateral [INFERENCIA]: la rampa larga da mucho tiempo en el extremo lento, donde `time_to_first_input` podría reganar correlación con el tiempo disponible — recalcular el control físico por sesión.

---

## 5. Reorientación conceptual

- [DECISIÓN] **Frontera estrecha**: campo = secuencia de piezas + gravedad (exógeno puro). El **tablero** se registra completo pero NO es campo (es estado co-construido por el agente). Mantiene exogeneidad limpia y null construible. Ampliable a frontera amplia solo si Fase 1 lo justifica.
- [DECISIÓN] **Observable central candidata = anticipación** (¿el agente modela el campo? = predicción vs nostalgia, mitad informacional de Still), NO el σ.
- [DECISIÓN] **Centro de gravedad = caracterización** (cómo se comporta la agencia una vez presente), NO demarcación (dónde empieza).

---

## 6. Marco conceptual (estructura de préstamos, no identidad disciplinar)

Documento: `MARCO_conceptual_agente_campo.md`. Resumen:
- **Hogar metodológico:** sistemas dinámicos (Kelso, Thelen, Beer). El σ, O/U y series ya viven aquí.
- **Ontología del agente:** Still (loop percibir-procesar-actuar bajo observabilidad parcial, sin interior). Se toma la definición, NO la termodinámica ni la normatividad. [ABIERTO §9] entre Still y enactivismo.
- **Distinción modela-vs-reacciona:** inferencia activa / Friston, préstamo **acotado** (la distinción, no el free energy). Caveat: falsabilidad de Friston es debate abierto.
- **Relacionalidad de la agencia:** Gibson (la agencia vive en la relación agente-campo), sin su antirrepresentacionalismo.
- **Tensión declarada:** método dinamicista + observable representacional mínima (la anticipación postula modelo interno). Es una elección intermedia, no un punto neutro. [INFERENCIA] Beer como precedente.
- **Propiedad valiosa:** un resultado de Fase 1 puede resolver una decisión conceptual — si la anticipación no supera al null, no hay modelado que postular y el marco colapsa limpio a dinamicismo puro.
- *Verificar antes de citar:* referencia de Barandiaran, Di Paolo & Rohde sobre definición de agencia (~2009, *Adaptive Behavior*) está anotada **de memoria**, no verificada.

---

## 7. EL PROBLEMA CENTRAL ABIERTO: el null [ABIERTO — máxima prioridad]

- **No es software pendiente; es un problema conceptual sin resolver.** (Error previo: tratarlo como ingeniería trivial. Corregido.)
- PUBG tuvo null **natural**: la partícula es "materia en el campo sin agencia", un contrafactual físico observable. Tetris **no** lo tiene: jugar *es* decidir, no existe "jugar pasivo". Todo null en Tetris es un **agente que diseñas**, y cada diseño incorpora una teoría de "no anticipar".
- Tres familias:
  - **null-agente** (política heurística ciega): DESCARTADO por arbitrario — la heurística es injustificada, y un null ciego bueno enmascara la anticipación humana mientras uno malo la exagera.
  - **null-permutación / surrogate** [INFERENCIA: probablemente el camino]: no diseñas un agente; tomas los datos del humano y rompes la relación temporal con el futuro del campo. Pregunta: ¿la colocación de la pieza *t* se alinea más con la pieza *t+1* **real** que con una *t+1* **permutada**? Surrogate testing es técnica estándar y sólida. [HECHO] Pendientes reales: (a) una **métrica de alineación** colocación↔pieza-siguiente (carga supuestos sobre qué es "buen encaje"); (b) **permutar respetando la estructura del 7-bag** (no IID), o el null queda mal calibrado.
  - **null-ablación** (modelar al humano y quitarle el preview): probablemente circular, al final.
- [REORIENTACIÓN CLAVE] **El null es el PRIMER paso, no el último.** Determina qué datos se necesitan. No recolectar más sesiones ni añadir features hasta tenerlo decidido. Esto contradice el orden que se venía asumiendo.
- Observación de fondo: resolver el null de Tetris ES, en miniatura, el problema central de medir agencia por fuera — ¿cuál es el contrafactual de "sin agencia" cuando el sistema siempre actúa? PUBG fue la excepción afortunada (el movimiento tiene contrafactual pasivo); casi ningún dominio de agencia humana lo tiene.

---

## 8. EEG / MR — descartado por ahora [DECISIÓN]

- Premisa de origen falsa: "Friston usa EEG → EEG mide agencia". El EEG mide actividad cortical agregada, no agencia; en el marco, la agencia se define por la interacción conductual (por fuera).
- EEG de consumo (Muse, OpenBCI, Emotiv): pocos canales, electrodos secos, vulnerables a artefactos. Jugar bajo presión genera EMG facial / artefacto ocular **correlacionados con la dificultad** = confound en la dirección que engaña (haría creer que funcionó). Mismo patrón que el Garmin ya descartado.
- Meter EEG cambia el marco (de caracterizar interacción a buscar correlato neural del modelado) y reabre la tensión representacional que se dejó acotada.
- Legítimo **solo si** la anticipación conductual sobrevive en Fase 1 y entonces se quiere su sustrato — con equipo serio, no de consumo.

---

## 9. Decisiones abiertas

1. [ABIERTO — alta prioridad] **Diseño del null** (§7). Cuello de botella de todo el programa.
2. [ABIERTO] **Ontología del agente: Still o enactivismo.** Still = deflacionario, cómodo con el método, pero define "agencia" tan flaco que casi cualquier sistema acoplado califica (frontera permisiva). Enactivismo = exigente (individualidad, normatividad, que las cosas le importen al sistema), más fiel a agencia *humana*, pero importa normatividad fuerte que un humano-jugando-Tetris quizá no sostiene, y antirrepresentacionalismo que pelea con la anticipación. Define qué tan ancha es la palabra que nombra todo el programa.
3. [ABIERTO] **Régimen de aprendizaje: meseta vs curva.** El aprendizaje es confound monótono (correlacionado con el tiempo) que ataca especialmente la anticipación (que *es* la habilidad que se aprende). Decisión: jugar hasta meseta y recolectar como experto estable (sesiones = réplicas) vs hacer del aprendizaje el objeto (sesiones = puntos de una curva, habilidad como eje). Pregunta empírica previa barata: ¿mejora siquiera a su nivel en la escala del estudio? → mirar curva de score/líneas por sesión.

---

## 10. Disciplinas / anti-patrones (lecciones pagadas, no negociables)

- Una variable a la vez (estado fijo entre sesiones: hora, cafeína).
- Registrar señal fina por pieza/acción, nunca solo agregados.
- Velocidad por tiempo, no por líneas (exogeneidad).
- 7-bag, no IID (estructura predecible para la observable).
- Excluir la primera pieza de cada partida (incluye reacción al arranque); revisar tras game over/pausa.
- σ solo en CV/residualizado, nunca absoluto (artefacto media-varianza).
- Control físico por `stack_height` (correlación parcial), no global.
- Rampa sin meseta (ventanas distribuidas a lo largo de la gravedad).
- No añadir ejes fisiológicos que no cooperan (Garmin, EEG) antes de validar lo conductual. (SSAQS y Dybvik ya mostraron que el canal autonómico no discrimina / se desacopla.)
- Calcular **potencia** antes de construir el análisis de anticipación (informacional = hambriento de datos; con ~100–470 piezas/sesión puede no haber potencia). [INFERENCIA]
- El null es fundacional, no add-on: sin él, todo es descriptivo.

---

## 11. Documentos del programa (subir al chat nuevo si se necesitan)

- `BLUEPRINT_tetris_instrumentado.md` — spec del juego y esquema de datos.
- `BLUEPRINT_analisis_exploratorio.md` — análisis exploratorio del piloto.
- `BLUEPRINT_programa_tetris_agente_campo.md` — Fase 0 (definiciones) + Fase 1 (qué es medible sobre el null); Fase 2 solo enunciada.
- `MARCO_conceptual_agente_campo.md` — estructura de préstamos disciplinares.
- (Código: Tetris instrumentado + `metrics.py` + `exploratory_analysis.py`, en el repo de Enrique. `software_git_hash` debe registrarse siempre.)

---

## 12. Próximo paso concreto

**Resolver el null antes que nada** (§7), empezando por el null-permutación: definir la métrica de alineación colocación↔pieza-siguiente y el esquema de permutación que respeta el 7-bag. Es trabajo conceptual + prototipo, no recolección. Hasta tenerlo, no tiene sentido recolectar más sesiones ni añadir features, porque el diseño del null determina qué datos se necesitan.

En paralelo, barato: (a) mirar la curva de habilidad por sesión (decide régimen de aprendizaje §9.3); (b) recalcular Análisis-1 con `board_height` corregido + control parcial y auditar el alcance del bug (§4).
