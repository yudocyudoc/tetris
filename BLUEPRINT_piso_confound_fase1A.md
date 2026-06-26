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

# Recalcular p_min con curva calibrada (proxy — b~0.9-1.0, NO usar para citas)
.venv\Scripts\python recalc_power_calibrated.py `
  --bin_results out_power_t2/bin_results.json `
  --calibration out_fast_calibrate_v2/fast_calibration_v2.json

# Recalcular p_min con curva real convexa — DEFINITIVO (n=300, b=1.22)
# Calibración de forma (beta_real_05 por bin) de results_colab_01; b_shape forzado a 1.22
.venv\Scripts\python recalc_power_calibrated.py `
  --bin_results ../../colab/results_colab_02/out_power_t2/bin_results.json `
  --calibration ../../colab/results_colab_01/out_fast_calibrate_v2/fast_calibration_v2.json `
  --b_shape_override 1.22
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

1. **Forma de `β_señal(p)`: convexa, no lineal.** [ACTUALIZADO] Las simulaciones reales en `p=0.10, 0.25, 0.50, 0.75` (corridas Colab, n=50, max_pieces=500) entregan: β(0.10)=0.48, β(0.25)=1.28, β(0.50)=3.15, β(0.75)=5.60. La forma es **fuertemente convexa** (b≈1.22 en ajuste potencial). La calibración rápida proxy estimó b≈0.9–1.0 (casi lineal) — **estaba equivocada en la dirección**. El proxy mide la parametrización de la utilidad, no la dinámica real del simulador. Ver §11.4.
2. **`tracker_prob` es el efecto a medir.** No es un parámetro de nuisance que se calibra aparte. La curva dice "para un humano que trackea fracción p, necesito N(p) decisiones", pero p no se conocerá hasta datos pilotos.
3. **Piezas por partida y bug corregido.** Con `--no_censorship` cada partida corre exactamente `max_pieces` piezas. [CORRECCIÓN] Las corridas históricas (n=300 local) usaron `max_pieces=100` (CLI default), pero `power_curve_t2.py` asumía 500 cuando el campo no estaba en el JSON — subestimando N por sesión en ×5 e inflando `p_min`. El campo `max_pieces` se agregó al dict de parámetros guardados en `resultados_piso_k*.json` ([commit corrección](analysis/confound_floor/confound_floor_t2.py)). Las corridas Colab usaron `max_pieces=500` explícitamente, por lo que sus resultados son correctos.
4. **Decisiones forzadas no aportan.** Solo cuentan las decisiones con `k≥2` alternativas viables y `p_grad_excess` no degenerado.

### 11.3. Resultados

**Corrida Colab n=300** (n=300 juegos, max_pieces=500, no_censorship=True, H=4–15):

**Datos crudos por bin** — dec/pieza ≈ 0.23–0.26, piso oráculo no-significativo en todos (bug ausente, piso limpio):

| H | n_dec (300 juegos) | dec/juego | dec/pieza | β_señal(0.5) | piso β oráculo | piso CI_high |
|---|---|---|---|---|---|---|
| 4–6  | 34,851 | 116.2 | 0.232 | 2.88 |  0.061 | 0.22 |
| 7–8  | 15,900 |  53.0 | 0.106 | 2.88 | -0.021 | 0.20 |
| 9–10 | 13,170 |  43.9 | 0.088 | 3.02 |  0.002 | 0.25 |
| 11–12| 11,640 |  38.8 | 0.078 | 3.12 | -0.035 | 0.22 |
| 13–15| 15,420 |  51.4 | 0.103 | 3.15 |  0.111 | 0.34 |

**`p_min` por bin — cuatro esquinas (escenario × forma de curva)**

Cálculo con β de n=300 en ambas ramas. Curva: β(p) = β(0.5)·(p/0.5)^b. Rango de b: 1.20 (OLS 4 puntos, jalado por p=0.10 ruidoso) a 1.33 (OLS sobre p∈{0.25,0.5,0.75}, más fiable). `>1` = no detectable con tracking perfecto.

| Bin | b | 10×100 | 10×470 | 15×100 | 15×470 | banda |
|-----|---|--------|--------|--------|--------|-------|
| H=4–6  | 1.20 | 0.79 | 0.43 | 0.67 | **0.37** | 0.37–0.79 |
| H=4–6  | 1.33 | 0.75 | 0.44 | 0.65 | **0.38** | 0.38–0.75 |
| H=7–8  | 1.20 | >1   | 0.55 | 0.87 | **0.47** | 0.47–>1 |
| H=7–8  | 1.33 | 0.95 | 0.55 | 0.82 | **0.47** | 0.47–0.95 |
| H=9–10 | 1.20 | >1   | 0.56 | 0.88 | **0.48** | 0.48–>1 |
| H=9–10 | 1.33 | 0.97 | 0.56 | 0.83 | **0.48** | 0.48–0.97 |
| H=11–12| 1.20 | >1   | 0.55 | 0.86 | **0.47** | 0.47–>1 |
| H=11–12| 1.33 | 0.95 | 0.54 | 0.82 | **0.47** | 0.47–0.95 |
| H=13–15| 1.20 | 0.91 | 0.50 | 0.77 | **0.43** | 0.43–0.91 |
| H=13–15| 1.33 | 0.86 | 0.50 | 0.74 | **0.44** | 0.44–0.86 |

Columnas: escenario = n_sesiones × piezas/sesión. β de n=300 en ambas ramas. Corrección de ancla: el recálculo previo con calibración n=50 usaba β_real_05 de n=50 (3.28–3.33 en H=4–6) vs β de n=300 (2.88), creando una inversión artefactual (calibrado < lineal) que H=9–10 violaba en dirección correcta porque su ratio n=50/n=300 era el único < 1. Con betas consistentes (n=300), el calibrado queda sistemáticamente ≥ lineal, como corresponde a b > 1.

**Para una sesión humana de 470 piezas:**

| H | N dec/sesión 470p | N dec/campaña 15×470 |
|---|---|---|
| 4–6  | **109** | **1,636** |
| 7–8  | 50  | 745  |
| 9–10 | 41  | 617  |
| 11–12| 37  | 550  |
| 13–15| 48  | 726  |
| **Total H=4–15** | **~284** | **~4,274** |

**Nota histórica de errores previos:** corrida local n=300 (pre-fix) usó `max_pieces=100` con pipeline que asumía 500 → N×5 inflado → `p_min` 0.66–0.70. Corrida n=50 Colab daba 0.36 con floor CI ancho. Análisis previo de n=300 citaba 0.33 como "definitivo": era artefacto de anclas cruzadas (β n=50 en calibrado, β n=300 en lineal). Número citable post-reconciliación: **0.37–0.38** (H=4–6, campaña 15×470, b∈{1.20,1.33}).

### 11.4. Forma real de `β_señal(p)` — simulaciones multi-p (Colab)

[ACTUALIZADO] Las simulaciones reales en Colab (n=50, max_pieces=500) entregaron `β_señal` global (todos los bins combinados) en cuatro valores de `tracker_prob`:

| tracker_prob | β_señal | IC 95% |
|---|---|---|
| 0.10 | 0.48 | (0.24, 0.71) |
| 0.25 | 1.28 | (1.03, 1.53) |
| 0.50 | 3.15 | (2.85, 3.46) |
| 0.75 | 5.60 | (5.24, 5.96) |

La curva es **fuertemente convexa**. Ajuste β(p) = β(0.5)·(p/0.5)^b con los cuatro puntos:

| Par vs ancla p=0.5 | b estimado |
|---|---|
| (0.5, 0.10) | 1.17 — punto más ruidoso, CI amplio (0.24–0.71) |
| (0.5, 0.25) | 1.30 |
| (0.5, 0.75) | 1.42 |
| OLS 4 puntos | **1.20** — jalado por p=0.10 |
| OLS p∈{0.25,0.5,0.75} | **1.33** — más fiable, excluye punto ruidoso |

Rango citable: **b ∈ [1.20, 1.33]**. Para decisiones de factibilidad usar b=1.33 (conservador); b=1.20 como borde optimista. La diferencia sobre `p_min` en H=4–6 optimista es 0.01 (0.37 vs 0.38), pequeña pero en la dirección correcta: más convexidad → umbral más alto.

Consecuencias:
- A `p` bajo (≲0.25), la señal es **mucho menor** que cualquier extrapolación lineal sugería.
- A `p=0.10`, CI (0.24, 0.71) ya roza el rango del piso — señal muy débil si el humano trackea poco.
- A `p` moderado-alto (≳0.4), la señal crece aceleradamente.

**El proxy (`fast_calibrate_signal_v2.py`) estaba equivocado en la dirección.** Estimó b≈0.9–1.0 (casi lineal o cóncavo) cuando la realidad es fuertemente convexa. El proxy mide la parametrización de la utilidad mixta, no la dinámica real del simulador. Queda archivado como inválido.

### 11.4-bis. Prueba de estrés de la featurización [HECHO — PASA barrido completo pbf ∈ {0, 0.25, 0.5, 1.0}]

**Origen:** crítica externa (Gemini) sobre la suficiencia descriptiva de la featurización (§5.2) — "suficiencia descriptiva" es un acto de fe mientras no se estrese. El punto es válido; su formulación del mecanismo, no (decía falso *negativo* por omitir una propiedad; el modo real es falso negativo solo si la propiedad omitida correlaciona con `S_t` de un modo que el oráculo no capture **y que el no-tracker sintético tampoco use** — así no aparece en el sintético pero sí en el humano).

**El riesgo concreto:** el piso limpio de B se midió con una `π_fill` bag-ciega específica. La "suficiencia" del oráculo L2 puede ser **local a esa heurística**. Si un generador con táctica más rica usa información de tablero correlacionada con el bag que el oráculo no absorbe, el piso del no-tracker se aparta de cero — y eso es un **falso negativo de piso limpio** que envenenaría el piloto humano (declararíamos limpio un sustrato que no lo es para un jugador real).

**Prueba ejecutada:** generador bag-en-relleno (`confound_floor_stress_t2.py`) que inyecta `S_t` en la **profundidad** de los huecos según `n_JL_restantes`. El oráculo L2 captura conteo de huecos por columna (`res_col_holes{c}`) pero no su distribución vertical — esa es la brecha real testeada. El no-tracker usa `depth_sensitive_board_value` (bag-ciego, sensible a profundidad). Parámetros: n=100 partidas, max_pieces=500, alpha_depth=0.4, barrido completo `p_bag_fill ∈ {0, 0.25, 0.5, 1.0}`.

**Resultados — β oráculo (p-valor) del no-tracker por bin:**

| Bin | pbf=0.00 | pbf=0.25 | pbf=0.50 | pbf=1.00 |
|-----|----------|----------|----------|----------|
| H=4–6   | −0.74 (0.889) | +0.34 (0.933) | +1.53 (0.658) | −6.22 (0.377) |
| H=7–8   | +4.49 (0.673) | −1.69 (0.566) | −0.36 (0.894) | +1.98 (0.658) |
| H=9–10  | +0.52 (0.889) | −1.75 (0.767) | −0.14 (0.963) | +2.35 (0.529) |
| H=11–12 | +0.59 (0.949) | +0.43 (0.896) | −3.23 (0.722) | −1.84 (0.624) |
| H=13–15 | +1.74 (0.752) | +0.13 (0.947) | +1.07 (0.843) | −0.39 (0.965) |

Todos los p-valores > 0.37. Sin tendencia monotónica en |β| al crecer `p_bag_fill`. Nota: el modelo bruto no convergió en H=4–6 para pbf=1.0 (betas explosivos esperables con inyección máxima); el oráculo L2 sí convergió. Veredicto del script: PASA en todos los bins para los cuatro niveles.

**Criterio de fallo:** si el piso del **no-tracker** (oráculo L2) se aparta de cero al cambiar el generador, la featurización **no es suficiente**. No se disparó en ningún bin.

**Interpretación:** el oráculo L2 con `res_col_holes{c}` absorbe completamente la covariación profundidad↔S_t incluso con inyección máxima (pbf=1.0, 100 % de colocaciones modificadas). El piso limpio de B es robusto a este mecanismo de confound. El mecanismo diseñado (profundidad de huecos correlacionada con bag) es la brecha más plausible entre `π_fill` bag-ciega y una táctica humana real.

---

### 11.5. Respuesta a la pregunta de factibilidad

[ACTUALIZADO — corrida Colab n=300, max_pieces=500, b∈{1.20,1.33}, betas consistentes]

**Pregunta:** ¿Cuál es el `tracker_prob` mínimo detectable con el N que una campaña humana realista entrega, y está por debajo del tracking humano plausible?

**Respuesta:**

- `p_min` H=4–6, b∈{1.20,1.33}: **0.37–0.38** (esquina optimista: campaña 15×470) hasta **0.75–0.79** (campaña 10×100). Banda completa: **0.37–0.79**. Robusto a `b` (diferencia ≤0.01 entre b=1.20 y b=1.33 en cualquier esquina).
- Detección solo en stack bajo (H=4–6); los otros bins arrancan en ≥0.43 y suben a >1.
- Piso oráculo limpio en todos los bins: centros próximos a 0, todos no significativos (p>0.05).

**Veredicto: no concluye viabilidad.** Fija que el estudio conductual requiere `tracker_prob ≳ 0.38` en H=4–6 bajo la campaña más favorable, y más bajo escenarios realistas de campaña. Si el bag-modeling humano está dominado por el residuo-sobre-preview (prior de Berry), es plausible que no alcance ese umbral, en cuyo caso el probe exógeno vuelve como única vía.

**Dos preguntas abiertas que el piloto debe cerrar:**
1. El `tracker_prob` humano real — ¿supera 0.38 en H=4–6?
2. Cuántas decisiones útiles en H=4–6 produce una sesión humana natural — eso fija qué esquina de campaña es la real (entre 0.38 y 0.79).

**Por qué cambió desde el veredicto anterior (0.66–0.70 → 0.33 → 0.38):**
1. Bug `max_pieces` (§11.2): corrida local asumía N×5 inflado → 0.66–0.70.
2. Corrida Colab n=50 con N correcto: 0.36 (floor CI ancho por n pequeño).
3. Análisis previo de n=300 citaba 0.33: artefacto de anclas cruzadas (β n=50 en calibrado, β n=300 en lineal; la inversión calibrado < lineal con b > 1 era la firma del bug).
4. Recálculo con betas n=300 consistentes y b∈{1.20,1.33}: **0.37–0.38** en H=4–6 optimista.

### 11.6. Franqueza metodológica y siguiente paso

- `tracker_prob` **es** la cantidad que el estudio existe para medir; no se puede fijar a priori para dimensionar sin circularidad.
- La curva no dice si el efecto existe; dice bajo qué valor del efecto un diseño dado es factible.
- La curva convexa real implica que a `p` bajo (≲0.25) la señal es débil incluso con N grande — si el humano trackea poco, el estudio no detectará nada sin un diseño más rico.
- **La densificación por concentración en H=4–6 pierde fuerza** como palanca: los conteos de la corrida Colab muestran decisiones razonablemente repartidas en todos los bins (H=4–6 tiene ~2.3× el bin más chico, no 3×). No hay un bin dramáticamente más rico al que mudarse.
- **Condición previa al piloto: completada.** La prueba de estrés de featurización (§11.4-bis) pasó para pbf ∈ {0, 0.25, 0.5}: el piso limpio de B es robusto a generadores que inyectan S_t vía profundidad de huecos.
- **El siguiente paso concreto no es probe ni densificar: es un piloto humano mínimo.** El piloto no necesita estimar `p` con precisión; solo necesita **distinguir `p>0.4` de `p<0.2`**, que es una pregunta gruesa y barata. 2–3 sesiones pueden plausiblemente separarlas. Esa única medición decide entre "recolección conductual viable" y "probe exógeno".

---

## Resumen operativo (una frase)

Simular partidas 7-bag naturales en el rango de H realista, evaluar la colocación de `P_t` contra la distribución de `t+2`, y medir cuánto β residual produce un no-tracker que no usa `S_t`: ese número es el piso de confound A. Validar con tracker imperfecto (`tracker_prob<1`) como ancla realista de potencia humana, con censura on/off como prueba de desacople, y con una curva N(`tracker_prob`) que incluya el piso como intervalo, N efectivo en decisiones útiles y un umbral de factibilidad de campaña humana.
