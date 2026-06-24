# Blueprint — Medición del piso de confound (compuerta de Fase 1A)

**Destinatario:** CC (ejecución directa) y registro de diseño.
**Entorno:** Windows / PowerShell. Python. **No generar .docx.**
**Estatus:** este documento especifica **un solo experimento sintético** y su regla de decisión. No es recolección humana, no es cálculo de potencia, no es enumeración de estados críticos. Esos pasos quedan **bloqueados** hasta que este experimento entregue su número (§6).
**Convención de tags:** [HECHO] establecido en el hilo; [DECISIÓN] elección tomada; [INFERENCIA] razonado, no seguro; [ABIERTO] sin resolver; [PARÁMETRO] valor que Enrique fija, no es teoría.

---

## 0. Estado heredado (autocontenido)

- [DECISIÓN] La observable central es **anticipación-como-modelado-del-bag**, operacionalizada sobre la primera pieza oculta. Con preview=1, esa es **t+2** (t+1 es visible → leerla es reacción al preview, no modelado).
- [HECHO] La anticipación puntual de t+2 es **distribucional**: trackear el agotamiento de la bolsa afila la distribución sobre el conjunto restante `S_t`, aunque la identidad de t+2 siga siendo unpredecible.
- [DECISIÓN] El conocimiento estático ("el juego es 7-bag") no es agencia. El **acto vivo** (trackear qué queda en *esta* bolsa y condicionar la colocación en `S_t`) sí lo es.
- [DECISIÓN] Operacionalización limpia: ¿la colocación de `P_t` depende de `S_t` dado `(board_t, P_t, P_{t+1})`? Es un test de independencia condicional sobre datos reales.
- [DECISIÓN — evolución del diseño] El **instrumento inicial de Fase 1A fue well-building** (gestión de pozo de I). Se descartó como familia principal porque en esa familia el horizonte de decisión está fijado por la capacidad-hasta-topar: horizonte corto = H alto = cerca de muerte = **colisionador de survivorship**. La señal y el colisionador están colocalizados. El remedio correcto es cambiar el horizonte, no añadir features.
- [DECISIÓN] El instrumento corregido es **acomodación a t+2 en partidas naturales a H moderado**: el horizonte está fijado por preview=1, no por supervivencia, diseñando el colisionador fuera por construcción.

---

## 1. Objetivo único

Medir **β_piso**: el coeficiente de la dependencia colocación↔`S_t` que produce un **no-tracker** (que por construcción ignora el bag) cuando se le aplica el estimador de Fase 1A sobre datos sintéticos que sí contienen confound A (historia real de partida) pero **no** confound B (H moderado + censura desactivada o, si se activa, no afecta porque no hay selección por supervivencia).

Resultado entregable: un número (β_piso) con intervalo de confianza, una prueba de desacople censura on/off, y la rama de decisión que dispara (§6).

---

## 2. Qué se corrige respecto al prototipo previo (well-building)

| Aspecto | Prototipo well-building | Este blueprint |
|---|---|---|
| Horizonte de decisión | Capacidad-hasta-topar (`N(H)`) | Preview fijado (`t+2`) |
| Confound B (survivorship) | Presente y fatal a H alto | Ausente por diseño a H moderado |
| Familia de estados | Pozo de I artificial | Partidas naturales, H moderado |
| Estimador | Logit binario leave/close | Conditional logit sobre colocaciones |
| Predictor | `p_grad_excess = P(I llega) − p_stat(H)` | `p_grad_excess = P(t+2 favorable \| S_t, clase) − P(t+2 favorable \| clase)` |

La lección de la ronda anterior: el piso de well-building no era un defecto de featurización; era un **colisionador de survivorship**. Añadir features no lo habría cerrado. El remedio es re-anclar en un horizonte fijado por preview.

---

## 3. Especificación del simulador

### 3.1. Campo y generador

- Tablero **10 columnas × 20 filas**.
- Generador **7-bag** con semilla fija.
- **Preview = 1** (visible: pieza actual `P_t` + `P_{t+1}`).
- Sin gravedad en la dinámica de decisión (colocaciones instantáneas).

### 3.2. Escenario: partidas naturales a H moderado

- El agente base (bag-ciego) juega partidas completas desde tablero vacío usando una política de relleno `π_fill` determinista.
- Se loguean decisiones donde el stack height `H` está en `[H_min, H_max]` (default `[4, 8]`). H moderado evita survivorship y mantiene tableros featurizables.
- Para cada decisión logueada se construye el **conjunto de consideración**: las `k` mejores colocaciones de `P_t` según `π_fill` base (default `k=5`).
- Cada colocación produce un `board_resultante`.

### 3.3. Clase de favorabilidad rica (local)

Para cada `board_resultante`, se define una **clase** que resume su compatibilidad con los 7 tipos de pieza posibles en `t+2`:
- Un tipo "encaja limpio" si existe al menos una colocación legal en el `board_resultante` que no aumente el número de huecos.
- `clase = (count, compatible_pieces)`, donde `count ∈ {0,...,7}`.

La clase es **determinista dado `(board_t, acción)`**; no depende de `S_t`. Es una featurización local del tablero resultante (no una función de valor global).

### 3.4. Distribución de t+2

Dado `S_t` (piezas no vistas de la bolsa actual) y `P_{t+1}` (preview):
- Si `P_{t+1}` no es la última pieza del bag: `t+2` es uniforme sobre `S_t \ {P_{t+1}}`.
- Si `P_{t+1}` es la última del bag: `t+2` proviene del siguiente bag → uniforme sobre las 7 piezas.

### 3.5. Predictor graduado

Definimos "favorable" como "`t+2` pertenece al conjunto de tipos compatibles con el `board_resultante`".

- `p_stat_clase = P(t+2 favorable | clase) = count / 7` (marginal 7-bag).
- `p_tracker_clase = P(t+2 favorable | S_t, clase) = Σ_{x ∈ compatible} P(t+2 = x | S_t, P_{t+1})`.
- `p_grad_excess = p_tracker_clase − p_stat_clase`.

`p_grad_excess` es el regresor del estimador de Fase 1A. El no-tracker, por construcción, no usa `S_t`; por tanto cualquier correlación entre su elección y `p_grad_excess` es piso.

### 3.6. Censura (diagnóstico de desacople)

El simulador permite `--no_censorship`. Cuando está activada, si el agente base no puede colocar una pieza, el tablero se reinicia y la secuencia continúa. A H moderado, la predicción es que **censura on/off no debe mover β_piso**: si lo hace, el argumento de desacople falla.

### 3.7. Telemetría

Una fila por alternativa por decisión por agente, en formato largo para conditional logit:
`game_id, decision_id, agent_type, chosen, piece, next_piece, S_t, H, alternative_id, base_val, compatible_count, p_stat_clase, p_tracker_clase, p_grad_excess, res_* (features del board_resultante)`.

---

## 4. Los dos agentes sintéticos

**Principio rector:** tracker y no-tracker comparten `π_fill` base y el mismo ruido; solo difieren en la fuente de creencia sobre `t+2`.

### 4.1. Agente base / no-tracker

- Elige dentro del conjunto de consideración maximizando una utilidad logística:
  `U_j = −board_value_weight · base_val_j + tau · p_stat_clase_j + ε_j`.
- `ε_j` i.i.d. tipo I (softmax).
- No usa `S_t`.

### 4.2. Tracker (control positivo)

- Idéntico al no-tracker salvo que usa `p_tracker_clase_j` en lugar de `p_stat_clase_j`:
  `U_j = −board_value_weight · base_val_j + tau · p_tracker_clase_j + ε_j`.
- Debe producir señal sobre `p_grad_excess`.

---

## 5. Estimador y medición del piso

### 5.1. Modelo: conditional logit

Datos en formato largo: cada alternativa del conjunto de consideración es una fila; `chosen ∈ {0,1}`. El modelo es:

`P(alternativa j es elegida | decisión i) = exp(x_{ij}'β) / Σ_{j'} exp(x_{ij'}'β)`.

### 5.2. Features

- **Predictor:** `p_grad_excess`.
- **Control de geometría resultante (bruto):** `p_stat_clase`.
- **Control oráculo (L2):** `base_val`, `p_stat_clase`, y features del `board_resultante`: `res_n_holes`, `res_bumpiness`, `res_full_h_max`, `res_full_h_std`, `res_full_h_var`, perfil completo de alturas por columna, y huecos por columna. La penalización L2 estabiliza la inversión con muchas features colineales.

### 5.3. Modelos a correr

Sobre cada agente:
1. **Bruto:** `chosen ~ p_grad_excess + p_stat_clase`.
2. **Oráculo L2:** `chosen ~ p_grad_excess + p_stat_clase + controles_ricos`.

El **piso** es `β(p_grad_excess)` del no-tracker en el modelo oráculo. Si es ≈0, el confound A es absorbido por la featurización del tablero resultante.

---

## 6. Regla de decisión sobre el piso

Sea `Δ_tracker` el β del tracker **imperfecto** (`tracker_prob < 1`) en el modelo bruto. [INFERENCIA] El efecto humano esperado es probablemente una fracción de `Δ_tracker`. **No usar** `tracker_prob=1.0` como ancla (ver §7.1).

- **Rama 1 — piso limpio.** Si el IC de `β_piso_oraculo` incluye 0 (o `|β_piso| < ε`): el estimador es robusto a confound A en este sustrato. → Se gana el derecho a calcular potencia contra cero.
- **Rama 2 — piso moderado.** Si `β_piso` es positivo/negativo pero pequeño en valor relativo a `Δ_tracker` (p. ej. `|β_piso| < 0.10 · |Δ_tracker|`): el humano debe exceder el piso. → La potencia se recalcula contra el piso.
- **Rama 3 — piso fatal.** Si `|β_piso|` se aproxima o supera `|Δ_tracker|`: el test no separa tracking de confound. → Rediseño.

### 6.1. Diagnóstico de desacople

Correr el prototipo con y sin `--no_censorship`. Predicción: `β_piso_oraculo` no debe moverse. Si se mueve, confound B está vivo y el argumento de desacople es incorrecto.

---

## 7. Resultados del prototipo

### 7.1. Advertencia sobre el tracker perfecto

El primer tracker sintético (`tracker_prob=1.0`) decidía con `p_tracker_clase` y el estimador usaba `p_grad_excess = p_tracker_clase − p_stat_clase`. La correlación intra-decisión entre `p_grad_excess` y `p_tracker_clase` fue ~0.89, y el conditional logit recuperó β≈10. Ese número no es una señal empírica de "efecto detectable"; es una **tautología generación-estimación**: el predictor del estimador es casi la misma cantidad que generó la conducta del tracker. **No debe usarse como ancla de potencia humana.**

Para obtener una ancla realista se introduce `tracker_prob < 1.0`: el tracker usa `S_t` solo en una fracción de las decisiones; en el resto usa la creencia estacionaria. Esto simula un tracking imperfecto, que es el caso humano.

### 7.2. Distribución de stack height en juego natural

Bajo `π_fill` base (100 partidas, 500 piezas o game over):
- Mediana H = 8.
- 50% de las decisiones: H ≤ 8.
- 75% de las decisiones: H ≤ 14.
- 90% de las decisiones: H ≤ 18.

Por tanto, un test válido debe cubrir H = 4–15 (donde vive la mayoría del juego), no solo H = 4–8.

### 7.3. Resultados finales (H=4–15, k=5, tau=10, tracker_prob=0.5, n_games=300)

| Modelo | Agente | β sobre `p_grad_excess` | IC 95% | p-valor | Interpretación |
|---|---|---|---|---|---|
| Bruto | no-tracker | −0.13 | (−0.33, 0.08) | 0.23 | — |
| **Oráculo L2** | **no-tracker** | **−0.05** | **—** | **0.64** | **Piso limpio global** |
| Bruto | tracker imperfecto | 2.84 | (2.57, 3.11) | 3.0e-95 | Señal realista |
| Oráculo L2 | tracker imperfecto | 3.24 | — | 4.8e-108 | Señal robusta |

**Piso por bins de H (no-tracker oráculo L2, con IC 95%):**

| H | n_decisiones | β | IC 95% | p |
|---|---|---|---|---|
| 4–6 | 8,001 | −0.00 | (−0.35, 0.34) | 0.99 |
| 7–8 | 3,161 | −0.14 | (−0.66, 0.37) | 0.58 |
| 9–10 | 2,421 | −0.24 | (−0.80, 0.33) | 0.41 |
| 11–12 | 1,971 | −0.11 | (−0.74, 0.53) | 0.75 |
| 13–15 | 2,453 | 0.24 | (−0.36, 0.84) | 0.43 |

El piso es consistente con cero en todos los bins. El β=−1.43 observado en H=9–10 con n=50 fue fluctuación muestral; con n=300 converge a −0.24 y su IC incluye cero ampliamente.

**Señal por bins de H (tracker imperfecto, bruto):**

| H | β | IC 95% | p |
|---|---|---|---|
| 4–6 | 2.95 | (2.53, 3.37) | <1e-3 |
| 7–8 | 2.88 | (2.24, 3.53) | <1e-3 |
| 9–10 | 3.07 | (2.34, 3.80) | <1e-3 |
| 11–12 | 2.21 | (1.50, 2.93) | <1e-3 |
| 13–15 | 2.90 | (2.18, 3.61) | <1e-3 |

**Test de desacople (n_games=50, H=4–8):**
- Censura on: no-tracker oráculo β = −0.15, p = 0.74.
- Censura off: no-tracker oráculo β = 0.08, p = 0.83.
- **Censura no mueve el piso.** Confirma que no hay colisionador de survivorship a H moderado.

**Rama disparada: 1 — piso limpio.** El piso es absorbido por la featurización del `board_resultante` en el rango de H realista.

---

## 8. Salidas requeridas

En `analysis/confound_floor/out_t2/`:
1. `resultados_piso_k{k}.json` — β bruto/oráculo por agente, p-valores, parámetros, diagnóstico de varianza intra-decisión, `rama_disparada`.
2. `fig_desacople_censura.png` — comparación no-tracker oráculo censura on/off.
3. `fig_piso_por_H.png` — piso vs señal por bins de H.
4. `decisions_log_k{k}.parquet` — telemetría cruda.

En `analysis/confound_floor/out_power_t2/` (posterior a §11):
5. `bin_results.json` — β por bin, IC del piso, varianza intra-decisión, ratios empíricos.
6. `power_curves.json` — curvas N(tracker_prob) por bin.
7. `fig_power_curve_sessions.png` — figura con umbrales de factibilidad.
8. `answer.txt` — respuesta explícita a la pregunta de factibilidad.

En `analysis/confound_floor/out_fast_calibrate_v2/`:
9. `fast_calibration_v2.json` — forma calibrada `β_señal(p) = β(0.5)·(p/0.5)^b` por bin.
10. `fig_fast_calibration_v2.png` — curvas calibradas vs lineales.

---

## 9. Estructura de archivos y comandos

Scripts: `analysis/confound_floor/confound_floor_t2.py` y `analysis/confound_floor/power_curve_t2.py`.

```powershell
cd analysis\confound_floor

# Corrida principal (tracker imperfecto, H realista)
.venv\Scripts\python confound_floor_t2.py `
  --n_games 50 --max_pieces 100 --tau 10 --H_max 15 --tracker_prob 0.5 --out out_t2

# Diagnóstico de desacople
.venv\Scripts\python confound_floor_t2.py `
  --n_games 50 --max_pieces 100 --tau 10 --H_max 15 --tracker_prob 0.5 `
  --no_censorship --out out_t2_nocensor

# Curva de potencia (contra resultados de la corrida principal)
.venv\Scripts\python power_curve_t2.py `
  --results_dir out_t2_H15_n300_tp05_nocensor --out out_power_t2

# Calibracion rapida de la forma de beta_señal(p) (proxy; no reemplaza simulaciones)
.venv\Scripts\python fast_calibrate_signal_v2.py `
  --log out_t2_H15_n300_tp05_nocensor/decisions_log_k5_sample20.parquet `
  --real_results out_power_t2/bin_results.json --out out_fast_calibrate_v2

# Recalcular p_min con curva calibrada
.venv\Scripts\python recalc_power_calibrated.py `
  --bin_results out_power_t2/bin_results.json `
  --calibration out_fast_calibrate_v2/fast_calibration_v2.json
```

---

## 10. Supuestos a vigilar

1. **Rango de H validado.** El piso se midió en H=4–15 con n=300, cubriendo ≥75% de la distribución de stack height bajo `π_fill` base. El β=−1.43 observado en H=9–10 con n=50 fue fluctuación muestral; con n=300 converge a −0.24 (IC incluye cero). No extrapolar a H>15 sin nueva medición.
2. **Función de favorabilidad local:** el criterio "no aumenta huecos" es una elección de diseño; otras definiciones de "limpia" cambiarían la clase pero no la lógica del test, siempre que sean locales y no dependan de `S_t`.
3. **`k` del conjunto de consideración:** default `k=5`. Si el piso se moviera con `k`, la selección del choice set estaría introduciendo geometría no controlada; en los datos actuales no ocurrió.
4. **Varianza intra-decisión de `p_stat_clase`:** ~6% de las decisiones tienen `p_stat_clase` constante entre alternativas; no impide la identificación pero limita la eficiencia del control en esas decisiones.
5. **Tracker imperfecto como ancla de potencia humana.** `tracker_prob=0.5` es una primera aproximación; el valor humano real debe calibrarse con datos piloto. El punto crítico es que `tracker_prob=1.0` no es una ancla válida porque el predictor del estimador reproduce casi exactamente la cantidad que generó la conducta del tracker (tautología generación-estimación, β≈10).

---

## 11. Potencia: curva N vs `tracker_prob` y umbral de factibilidad

### 11.1. Metodología

El piso entra como **intervalo**, no como punto. La separación efectiva en cada bin es:

`δ(H, p) = β_señal(H, p) − CI_high_piso(H)`

usando el extremo superior del IC del no-tracker oráculo (el piso más desfavorable porque reduce la señal). La señal se extrapola linealmente desde el ancla `tracker_prob=0.5`:

`β_señal(H, p) = p · β_tracker_bruto(H, 0.5) / 0.5`.

La potencia se calcula para conditional logit con información por decisión:

`I_dec = ((k−1)/k) · σ²_intra(p_grad_excess)`,

y se ajusta por la fracción de decisiones con varianza intra nula (`frac_zero_std`), que no aportan a la identificación. El N se traduce a sesiones humanas usando el ratio empírico `decisiones/partida` y un rango de `piezas/sesión` realista (100–470).

### 11.2. Supuestos críticos (explícitos)

1. **Extrapolación lineal de la señal.** La curva base asume que `β_señal` escala linealmente con `tracker_prob`. Se intentó validar con simulaciones en `p=0.25, 0.75`, pero el entorno actual hace que una corrida de n=50 tome más de una hora (la corrida n=300 histórica parece haber corrido en condiciones distintas). Se recurrió a una calibración rápida basada en el log de `p=0.5`, que estima la forma `β(p) = β(0.5)·(p/0.5)^b`.
2. **`tracker_prob` es el efecto a medir.** No es un parámetro de nuisance que se calibra aparte. La curva dice "para un humano que trackea fracción p, necesito N(p) decisiones", pero p no se conocerá hasta datos pilotos.
3. **Piezas por partida.** Con `--no_censorship` cada partida del simulador corre exactamente `max_pieces=500` piezas. Las sesiones humanas reales pueden diferir.
4. **Decisiones forzadas no aportan.** Solo cuentan las decisiones con `k≥2` alternativas viables y `p_grad_excess` no degenerado.

### 11.3. Resultados

| H | n_dec | β_señal(0.5) bruto | piso oráculo CI_high | p_min campaña (10–15 ses, 100–470 p/ses) |
|---|---|---|---|---|
| 4–6 | 8,001 | 2.95 | 0.34 | **0.70 – >1** |
| 7–8 | 3,161 | 2.88 | 0.37 | >1 – >1 |
| 9–10 | 2,421 | 3.07 | 0.33 | >1 – >1 |
| 11–12 | 1,971 | 2.21 | 0.53 | >1 – >1 |
| 13–15 | 2,453 | 2.89 | 0.84 | >1 – >1 |

`p_min` = `tracker_prob` mínimo detectable al 80% de potencia. `>1` significa que ni siquiera un tracker perfecto (`tracker_prob=1.0`) sería detectable en ese escenario bajo los supuestos actuales.

### 11.4. Validación de la forma de `β_señal(p)`

Se intentó correr simulaciones en `p=0.10, 0.25, 0.75`, pero cada una excede el tiempo disponible en este entorno (n=50 no terminó en 1 hora). Como proxy se usó una **calibración rápida** sobre el log de `p=0.5`: se simulan elecciones con utilidades mezcladas `-bw·base_val + τ·(p_stat + p·p_grad_excess)`, se ajusta `β(p) = β(0.5)·(p/0.5)^b`, y se normaliza al `β(0.5)` real.

Formas estimadas (5 replicaciones):

| H | forma `b` | interpretación |
|---|---|---|
| 4–6 | 1.21 | ligeramente convexa |
| 7–8 | 0.94 | casi lineal |
| 9–10 | 0.89 | cóncava |
| 11–12 | 0.91 | cóncava |
| 13–15 | 0.91 | cóncava |

La forma es ruidosa y depende del sampleo de los εpsilon Gumbel, así que estos números son indicativos, no definitivos. Lo relevante es el efecto sobre `p_min`:

| H | p_min lineal | p_min calibrado | cambio |
|---|---|---|---|
| 4–6 | 0.70 – >1 | 0.66 – >1 | marginal |
| 7–8 | >1 – >1 | >1 – >1 | ninguno |
| 9–10 | >1 – >1 | >1 – >1 | ninguno |
| 11–12 | >1 – >1 | >1 – >1 | ninguno |
| 13–15 | >1 – >1 | >1 – >1 | ninguno |

**Conclusión de la calibración:** corregir la no-linealidad no mueve el veredicto. El cuello de botella es el **N efectivo por sesión**, no la extrapolación lineal.

### 11.5. Respuesta a la pregunta de factibilidad

**Pregunta:** ¿Cuál es el `tracker_prob` mínimo detectable con el N que una campaña humana realista entrega, y está por debajo del tracking humano plausible?

**Respuesta:** En el mejor bin y escenario de campaña (H=4–6, 15 sesiones de 470 piezas), `p_min ≈ 0.66–0.70`. En todos los demás bins, `p_min > 1` (no detectable ni con tracking perfecto).

**Implicación:** Si el tracking humano real es menor que ~0.66–0.70 —lo cual es plausible dado que el preview ya proporciona la pieza siguiente y el residuo de modelado del bag suele ser pequeño—, el **estudio conductual puro no tendrá potencia** para detectar anticipación via modelado del bag en Fase 1A. La información de bag-modeling por unidad de tiempo de juego es intrínsecamente escasa en este sustrato (preview=1 limita el horizonte a t+2; Şimşek deja pocas decisiones informativas; el 7-bag degenera rápido).

### 11.6. Franqueza metodológica y siguiente palanca

- `tracker_prob` **es** la cantidad que el estudio existe para medir; no se puede fijar a priori para dimensionar sin circularidad.
- La curva no dice si el efecto existe; dice bajo qué valor del efecto un diseño dado es factible.
- La extrapolación lineal y el supuesto de 500 piezas/partida deben revisarse con datos piloto humanos antes de una decisión final.
- Antes de saltar al **probe exógeno** (que es un cambio de pregunta: deja de medir tracking espontáneo y mide uso-de-información-manipulada), la palabra correcta sobre el denominador es **densificar decisiones informativas por sesión**: concentrar el juego en H=4–6 (donde vive la mayoría de las decisiones), evaluar sesiones más largas, o diseñar un régimen que produzca más estados con ≥2 alternativas viables y bolsa no degenerada.

---

## Resumen operativo (una frase)

Simular partidas 7-bag naturales en el rango de H realista, evaluar la colocación de `P_t` contra la distribución de `t+2`, y medir cuánto β residual produce un no-tracker que no usa `S_t`: ese número es el piso de confound A. Validar con tracker imperfecto (`tracker_prob<1`) como ancla realista de potencia humana, con censura on/off como prueba de desacople, y con una curva N(`tracker_prob`) que incluya el piso como intervalo, N efectivo en decisiones útiles y un umbral de factibilidad de campaña humana.
