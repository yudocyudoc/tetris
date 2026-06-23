# Blueprint — Medición del piso de confound (compuerta de Fase 1A)

**Destinatario:** CC (ejecución directa) y registro de diseño.
**Entorno:** Windows / PowerShell. Python. **No generar .docx.**
**Estatus:** este documento especifica **un solo experimento sintético** y su regla de decisión. No es recolección humana, no es cálculo de potencia, no es enumeración de estados críticos. Esos pasos quedan **bloqueados** hasta que este experimento entregue su número (§6).
**Convención de tags:** [HECHO] establecido en el hilo; [DECISIÓN] elección tomada; [INFERENCIA] razonado, no seguro; [ABIERTO] sin resolver; [PARÁMETRO] valor que Enrique fija, no es teoría.

---

## 0. Estado heredado (autocontenido — qué quedó cerrado antes de este blueprint)

Para que CC no tenga que reconstruir el hilo:

- [DECISIÓN] La observable central es **anticipación-como-modelado-del-bag**, operacionalizada sobre la primera pieza **oculta**. Con preview=1, esa es **t+2** (t+1 es visible → leerla es reacción al preview, no modelado).
- [HECHO] **La anticipación puntual de t+2 es degenerada** bajo 7-bag + preview=1: el orden del tramo no visto es información que el sustrato no contiene. Permutar t+2 dentro del conjunto restante no cambia su distribución. La única anticipación posible es **distribucional**: trackear el agotamiento de la bolsa afila la distribución sobre el conjunto restante `S_t`, aunque la identidad de t+2 siga siendo unpredecible.
- [DECISIÓN] El **conocimiento estático** ("el juego es 7-bag") no es agencia. El **acto vivo** (trackear qué queda en *esta* bolsa y condicionar la colocación en `S_t`) sí lo es. El test debe aislar lo segundo.
- [DECISIÓN] Operacionalización limpia, **no circular**: ¿la colocación depende de `S_t` dado (board, P_t, P_{t+1})? Es un test de independencia condicional / inclusión de variable sobre datos reales, **no** un contrafactual generativo de agente (por eso no hereda la circularidad del null-ablación).
- [DECISIÓN] Instrumento de Fase 1A = **test estructural en la familia well-building** (gestión de pozo de 4 para la pieza I).
- [HECHO, del prototipo previo] El ground truth es **graduado, no binario**. El predictor correcto no es "I ∈ S_t" sino la probabilidad graduada `P(I llega antes de topar | S_t, H)` bajo orden uniforme del tramo no visto. → el test es **regresión**, no comparación de tasas.
- [DECISIÓN — corrección de skill-leakage] El ground truth de 1A se calcula con `H` y geometría del pozo **sin gravedad**. La gravedad modula el *logro* del óptimo (skill), no *qué es* óptimo; entra en 1B (Programa C), no aquí.
- [DECISIÓN] Optimalidad = comparación de **dos brazos**: *dejar* el pozo (apostar a la I) vs *cerrarlo* (seguridad). El umbral sale del **margen de utilidad** entre brazos, no del nivel de uno solo.
- [HECHO] La prueba de separación sintética **pasó su versión débil**: la regresión recupera la dependencia en `S_t` que se construyó en el tracker y no alucina una donde se quitó (no-tracker, β≈0).

### Lo que ese β≈0 **no** demostró (la razón de este blueprint)

El no-tracker del prototipo previo condicionaba en exactamente el mismo conjunto que la regresión controla → la independencia condicional era **por construcción**, no ganada. En datos reales el condicionamiento perceptual del humano es desconocido y casi con certeza ≠ al vector de features de la regresión. Hay **dos confounds** que el sintético previo no podía ver porque [INFERENCIA, a confirmar en §2.1] generó estados *muestreando* `(H, m, I∈S)` en vez de simular partidas reales:

- **Confound A — endogeneidad de `I∈S_t` respecto al tablero.** `I∈S_t` y `board_t` comparten causa común (la secuencia de piezas): puerta trasera `I∈S ← secuencia → board`. `I∉S_t` implica que la I ya salió en esta bolsa (probablemente se usó, probablemente el tablero cambió). Condicionar en una representación **incompleta** del tablero deja la puerta abierta → un agente que **no trackea** exhibe β≠0 por residuo de historia no capturado. Es el problema de **suficiencia de la featurización** mordiendo sobre un objetivo medible.
- **Confound B — survivorship en la zona de máxima señal (H alto).** A H alto solo se observan estados donde el jugador **sobrevivió**. "Dejar pozo a H alto sin que llegue la I → topa → no se observa." La censura correlaciona con `I∈S` (si I∈S, dejar sobrevive más → se observa más): **colisionador** que infla β bajo el null sin tracking alguno. Es el mismo collider que contaminó las fases tardías de PUBG, y cae exactamente en H=12–17, la zona objetivo.

Ambos empujan β>0 bajo el null de no-tracking en datos reales. Por tanto **el β humano no compite contra 0; compite contra un piso de confound desconocido y plausiblemente positivo**. Medir ese piso es la compuerta.

---

## 1. Objetivo único

Medir **β_piso**: el coeficiente de la dependencia colocación↔`S_t` que produce un **no-tracker** (que por construcción ignora el bag) cuando se le aplica el estimador de Fase 1A sobre datos sintéticos que **sí contienen** los dos confounds (historia real de partida + censura por topar).

Y **descomponerlo** en su parte A (insuficiencia de featurización) y su parte B (survivorship), porque el remedio difiere.

Resultado entregable: un número (β_piso) con intervalo de confianza, su descomposición A/B, y la rama de decisión que dispara (§6).

---

## 2. Qué se corrige respecto al prototipo previo

### 2.1. Verificación previa (CC debe reportar antes de codificar lo nuevo)

Inspeccionar el script del prototipo de separación previo y reportar **cómo generó los estados**: ¿muestreó `(H, m, I∈S)` de forma independiente, o simuló partidas 7-bag con historia? Esto confirma o refuta la sospecha de que los confounds estaban ausentes por construcción. Es un dato, no un reproche; condiciona la interpretación de que "el estimador funciona".

### 2.2. Las tres diferencias estructurales

| Aspecto | Prototipo previo | Este blueprint |
|---|---|---|
| Generación de estados | [INFERENCIA] muestreo de `(H,m,I∈S)` | **Partidas 7-bag reales simuladas**, con historia de piezas |
| Muerte / censura | Ausente | **Presente** (topar elimina trayectorias) |
| No-tracker | Condiciona en el mismo conjunto que controla la regresión | Condiciona en el **tablero completo verdadero**, con **más** información de tablero de la que la regresión puede controlar |

La tercera es la clave: si el no-tracker no tiene más información de tablero que la regresión, el confound A no puede manifestarse y el test vuelve a ser tautológico.

---

## 3. Especificación del simulador

### 3.1. Campo y generador

- Tablero **10 columnas × 20 filas** (visibles). [PARÁMETRO si la geometría real difiere.]
- Generador **7-bag** con semilla fija. Registrar `seed`. La misma semilla alimenta tracker y no-tracker (campo idéntico).
- **Preview = 1** (visible: pieza actual + 1).
- **Sin gravedad en la dinámica de decisión.** El simulador coloca piezas instantáneamente; el tiempo no entra. (La gravedad es 1B.)

### 3.2. Escenario well-building (el dominio del test)

Estado bien-formado: **9 columnas construidas a altura H**, **1 columna-pozo** vacía de profundidad ≥4 (lista para un I vertical que limpia 4 líneas). [PARÁMETRO: posición de la columna-pozo; default = borde, columna 0 o 9.]

En cada spawn dentro de este escenario, el agente decide entre **dos brazos**:
- **Dejar:** colocar la pieza actual (no-I) en las 9 columnas, preservando el pozo, apostando a que la I llegue.
- **Cerrar:** colocar la pieza actual tapando/rellenando hacia el pozo, renunciando al tetris pero estabilizando.

### 3.2.1. π_fill — política de relleno (nuisance compartida, NO la política bajo test)

Separar dos políticas que están acopladas pero son distintas:

- **π_fill (relleno):** dónde colocar una pieza no-I en las 9 columnas. **Fija, determinista, compartida idéntica por tracker y no-tracker, bag-ciega.** Condiciona en el **vector completo de alturas de columna + mapa de huecos** (no en agregados). Define `N(H)` (§3.3). Es nuisance: debe ser la misma para ambos agentes o la diferencia de `N` contamina el contraste, y debe ser bag-ciega o la ventaja del tracker se filtra al relleno en vez de quedar localizada en la decisión bajo test.
- **Decisión deja/cerra (bajo test):** dónde tracker y no-tracker difieren, vía `p_grad` vs `p_stat` (§4). Es lo único que el experimento interroga.

Que π_fill condicione en el **perfil completo** (no en BCTS agregado) es lo que abre el canal del confound A: el micro-patrón de superficie lleva traza de las piezas recientes → correlaciona con `S_t` → y la regresión, que solo controla agregados BCTS, no puede bloquearlo del todo. Si π_fill usara solo `agg_height`/`bumpiness` (ambos en BCTS), el canal se cierra y el piso saldría ≈0 trivialmente.

### 3.3. `N(H)` — horizonte de supervivencia, **sin gravedad** [PARÁMETRO de forma, invariante de contenido]

`N(H)` = número de piezas no-I que caben en las 9 columnas antes de que estas topen (fila 20), bajo la **política de relleno π_fill** (§3.2.1).

- **Invariante no negociable:** `N` es función de `H` y la geometría del pozo **únicamente**. **No** contiene término de gravedad. Si CC introduce gravedad en `N`, está reintroduciendo el skill-leakage que este diseño corrige.
- **`N(H)` se computa empíricamente, no por fórmula cerrada.** Simular π_fill desde altura `H` con piezas no-I del bag hasta topar; contar piezas colocadas. **Requisito de consistencia (crítico):** la política que define `N(H)` y la política con que el agente realmente rellena al elegir "dejar" **deben ser la misma π_fill**. Si difieren, `p_grad` se calcula sobre un horizonte que el agente no realiza y el ground truth queda descalibrado.
- `N(H)` bajo π_fill es **estocástico** (la secuencia de relleno viene del bag). Usar `E[N | H]` como escalar es aproximación de primer orden aceptable para esta corrida; el acoplamiento secuencia-de-relleno ↔ llegada-de-I (mismo bag) queda como supuesto a revisar (§10.6), no enterrado.
- CC debe **reportar la curva `E[N|H]`** y su dispersión.

### 3.4. Modelo de llegada de la I (predictor graduado) [HECHO — del prototipo]

Bajo orden uniforme del tramo no visto, con `m = |S_t|` (piezas no vistas en la bolsa actual):

- Si **I ∈ S_t**: `P(I dentro de N piezas) = min(N / m, 1)`
- Si **I ∉ S_t**: `P(I dentro de N piezas) = clamp( (N − m) / 7, 0, 1 )`

Este es el **predictor graduado** `p_grad := P(I llega antes de topar | S_t, H)`. Es continuo y es el regresor correcto (§5), no su binarización `I∈S`.

### 3.5. Censura (confound B, obligatoria)

El simulador corre **trayectorias secuenciales**, no decisiones one-shot. Reglas:
- Una decisión "dejar" a H alto sin que la I llegue dentro de `N` → el stack topa → **game over**; los estados posteriores de esa trayectoria **no se loguean**.
- Solo se loguean decisiones en estados **alcanzados con vida** (survived-to-here).
- Esto reproduce el colisionador: el muestreo a H alto queda enriquecido por `I∈S` entre quienes eligieron "dejar".

### 3.6. Telemetría por decisión (una fila por punto de decisión observado)

Columnas mínimas:
`game_id, piece_idx, H, m, S_t (conjunto), I_in_S (bool), p_grad, action (deja=1/cerra=0), board_full (estado completo verdadero), survived_next (bool), agent_type (tracker/no_tracker), seed, software_git_hash`

---

## 4. Los dos agentes sintéticos

**Principio rector:** tracker y no-tracker son **la misma política en todo** (mismo π_fill, mismo brazo de utilidad, mismo ruido `τ`), y difieren **solo** en la fuente de creencia sobre la llegada de la I. Cualquier otra diferencia rompe la atribución: un β distinto podría venir de esa otra diferencia, no del tracking.

### 4.1. Tracker (control positivo)

Decide por el **margen de utilidad de los dos brazos**, condicionando en `S_t` verdadero:
- `U_dejar = p_grad · V_tetris − (1 − p_grad) · C_topar`
- `U_cerrar = V_estable`
- `p_grad = p_grad(S_t, H)` (§3.4): depende de `S_t` vía `m` e `I∈S`.
- Acción = `dejar` si `U_dejar > U_cerrar`, con ruido logit (temperatura `τ`) para que la frontera no sea un escalón. [PARÁMETRO: `V_tetris, C_topar, V_estable, τ`.]

El tracker **debe** producir señal (control positivo). Si no, hay bug.

### 4.2. No-tracker (el que mide el piso — el agente crítico)

Idéntico al tracker salvo un cambio, y un requisito:

- **Cambio único:** sustituir `p_grad(S_t, H)` por `p_stat(H) = min(N/7, 1)` — creencia **estacionaria** del 7-bag, **independiente de `S_t`** (la I tratada como uniforme en la ventana de 7, sin agotamiento de *esta* bolsa). Todo lo demás (π_fill, utilidades, `τ`) es **el mismo** que el tracker.
  - **Por qué `min(N/7,1)` y no `1−(6/7)^N` — no es elección libre.** `min(N/7,1)` modela un agente que **sabe que el juego es 7-bag** (la I aparece una vez cada 7) pero **no trackea esta bolsa**. `1−(6/7)^N` modela un agente que trata las piezas como iid uniforme y **ni siquiera sabe que es 7-bag**. El segundo es un null **mal especificado**: en §0 se decidió que el conocimiento estático "es 7-bag" **no es agencia** y el null no debe acreditarlo. Usar `1−(6/7)^N` le quita al no-tracker ese saber estático, así que la diferencia tracker−no-tracker mezclaría *tracking vivo* con *saber que existe la estructura de bolsa* — y contaría lo segundo como anticipación, sobreestimando todo. El null correcto es el que ya posee el saber estático y solo carece del tracking vivo: **`min(N/7,1)`**.
- **Requisito para que el test pueda fallar:** el no-tracker (vía π_fill) condiciona en el **vector completo de alturas + huecos** (`board_full`), que es **más rico** que el resumen BCTS que la regresión controlará (§5). Ese gap es el canal del confound A. Si el no-tracker decidiera como función solo de `H` (que está en BCTS), no habría fuga y el piso saldría ≈0 por construcción — repitiendo el error del prototipo previo. **Esto NO es "BCTS sin bag features": esa opción cierra el canal y vuelve el test tautológico.**
- El no-tracker **no depende de `S_t`** en absoluto. Por tanto cualquier `β ≠ 0` que la regresión le estime sobre `p_grad`/`I∈S` es **piso de confound** (historia + survivorship), no tracking. Ese es el número que se mide.

---

## 5. Estimador y medición del piso

### 5.1. Features de control (deliberadamente incompletas — BCTS)

Vector de features del tablero que la regresión controla (de Şimşek / BCTS):
`agg_height, n_holes, bumpiness, well_depth, landing_height, row_transitions, col_transitions`

Son un resumen **lossy** de `board_full`. La distancia entre este vector y `board_full` es exactamente lo que el confound A explota.

### 5.2. Modelos de regresión (logística; respuesta = `action`)

Correr **sobre datos del no-tracker** (y replicar en tracker como contraste):

1. **Piso bruto:** `action ~ p_grad + BCTS_features`. → `β(p_grad)` = **β_piso bruto** (contiene A + B).
2. **Control oráculo (aísla A):** `action ~ p_grad + f(board_full)` con control rico del tablero verdadero. Si `β(p_grad)` cae a ≈0 aquí, se confirma que el residuo del modelo (1) era **insuficiencia de featurización** (confound A). El gap `β(1) − β(2)` cuantifica A.
3. **Sin survivorship (aísla B):** repetir el modelo (1) pero (a) restringido a estados de bajo H no censurados, o (b) con ponderación por inverso de probabilidad de supervivencia. La diferencia con (1) cuantifica B.

Usar `p_grad` continuo como regresor (no `I∈S` binario): la binarización tira la estructura graduada del landscape y pierde potencia. Reportar también la versión binaria solo para comparabilidad con el prototipo previo.

### 5.3. Descomposición entregable

```
β_piso_bruto      = β del modelo (1)        [A + B juntos]
contrib_A         = β(1) − β(2)             [insuficiencia de featurización]
contrib_B         = β(1) − β(3)             [survivorship]
```
(La descomposición no es exactamente aditiva si A y B interactúan; reportar las tres cantidades y señalar si `contrib_A + contrib_B` se aparta de `β(1)`, lo que indicaría interacción.)

---

## 6. Regla de decisión sobre el piso

Sea `Δ_humano_esp` el efecto humano esperado. [INFERENCIA, prior heredado: el sintético del tracker dio Δ≈0.26; en humanos, probablemente la mitad, ~0.13, por ruido motor/fatiga/skill imperfecto. Es prior, no medición — tratar como parámetro, no como dato.]

- **Rama 1 — piso limpio.** Si el IC de `β_piso_bruto` incluye 0 (o `|β_piso| < ε`, `ε` pequeño [PARÁMETRO, p.ej. 0.02]): el estimador es robusto a los confounds. → Se **gana el derecho** a calcular potencia contra cero. Registrar explícitamente que la crítica de confound se desinfla en este sustrato.
- **Rama 2 — piso moderado.** Si `β_piso` es positivo pero claramente `< Δ_humano_esp` (p.ej. 0.03–0.08): el humano debe **exceder el piso**, no el cero. → La potencia se recalcula contra el piso. La descomposición A/B dice qué atacar (más features ⇒ baja A; restringir a estados backdoor-cerrados ⇒ baja B).
- **Rama 3 — piso fatal.** Si `β_piso` se aproxima o supera `Δ_humano_esp`: el test estructural **en su forma actual no separa tracking de confound**. → Rediseño antes de cualquier recolección: (a) restringir el test a estados donde la puerta trasera esté **cerrada por construcción** (p.ej. controlar la historia de la bolsa explícitamente, o usar solo transiciones donde `I∈S` no covaría con el perfil de tablero observado); y/o (b) añadir features hasta que el piso baje — que es **el criterio de suficiencia descriptiva aplicado a un objetivo medible** (no "¿rankea bien?", sino "¿el piso cae a 0?").

**Bloqueo:** el cálculo de potencia (§9) no se ejecuta hasta que este resultado seleccione Rama 1 o 2. Si selecciona Rama 3, el siguiente paso es rediseño, no potencia.

---

## 7. Salidas requeridas

En `.\out\`:
1. `resultados_piso.json` — `β_piso_bruto` (+IC), `contrib_A`, `contrib_B`, `rama_disparada`, `N(H)` usada, todos los [PARÁMETRO], `seed`, `software_git_hash`, y el reporte de §2.1 (cómo generó estados el prototipo previo).
2. `fig_piso_por_H.png` — `β_piso` (o señal `P(deja|I∈S)−P(deja|I∉S)`) del **no-tracker** por stack height, junto al del tracker en el mismo eje, para ver si el piso se concentra en H alto (firma de survivorship).
3. `fig_descomposicion.png` — barras de `β(1)`, `β(2)`, `β(3)` lado a lado.
4. `tabla_regresiones.csv` — coeficientes, t/z, p, R², N efectivo por modelo y por agente.
5. `decisions_log.parquet` (o CSV) — telemetría cruda (§3.6) para auditoría.

Cada figura con título que indique agente y qué confound aísla. Reportar **N de filas** efectivamente logueadas por agente (la censura reduce N — dato relevante para la potencia posterior).

---

## 8. Estructura de archivos y comandos (PowerShell / Windows)

Ubicación sugerida en el repo de Enrique: `analysis\confound_floor\`.

```powershell
# desde la raíz del repo
cd analysis\confound_floor

# entorno aislado
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# dependencias
pip install numpy scipy pandas statsmodels matplotlib pyarrow

# registrar hash del código (no negociable del programa)
git rev-parse HEAD | Out-File -Encoding utf8 .\out\git_hash.txt

# ejecutar
python confound_floor.py --seed 42 --n_games 5000 --tau 0.5 --out .\out
```

Script único `confound_floor.py` con CLI: `--seed`, `--n_games`, `--tau`, `--out`, y los [PARÁMETRO] de utilidad (`--v_tetris`, `--c_topar`, `--v_estable`). `N(H)` se computa internamente bajo π_fill (no hay flag `--eta`). Determinista dado `seed`.

Si `Activate.ps1` falla por execution policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 9. Qué NO hacer en este blueprint (bloqueado hasta §6 → Rama 1/2)

- **No** calcular potencia. (Depende del piso; potencia contra null mal especificado no es informativa.)
- **No** recolectar sesiones humanas ni añadir features al juego.
- **No** enumerar los 10–20 estados críticos en detalle.
- **No** reescribir el `MARCO`/`BLUEPRINT` general con la secuencia 1A/1B/1C/2 todavía (es consolidación correcta pero prematura hasta saber qué forma tiene 1A tras el piso).

Cuando §6 dé Rama 1 o 2, el siguiente blueprint será **potencia con `N_eff` compuesto** = `N_total × P(t+2 no determinada por bolsa) × P(decisión sub-determinada por Şimşek)`, estratificada por zona de señal (H=12–17, m=3–5), asumiendo `Δ_humano_esp` pequeño, y midiendo contra el **piso**, no contra cero.

---

## 10. Supuestos a vigilar (CC debe exponerlos, no enterrarlos)

1. **`N(H)` empírico** (§3.3): se computa por simulación de π_fill, no por fórmula cerrada. Reportar `E[N|H]` y su dispersión. El invariante (sin gravedad) se mantiene; la forma de la curva es resultado de π_fill, no un supuesto impuesto.
2. **Utilidades de los brazos** (`V_tetris, C_topar, V_estable`): definen el umbral del tracker. Son [PARÁMETRO]; el resultado del piso **no debería** depender fuertemente de ellos (el piso lo produce el *no-tracker*, que no usa el margen para mirar `S_t`). Verificar esa insensibilidad como chequeo de sanidad.
3. **Independencia ruido↔futuro** (supuesto de calibración del null): el ruido de ejecución/decisión debe ser ⊥ a las piezas futuras. Si el ruido del no-tracker covaría con el futuro, contamina el piso. Mantener el ruido del no-tracker explícitamente independiente de `S_t`.
4. **`Δ_humano_esp`** (§6): es prior heredado (~0.13), no medición. La rama de decisión depende de él; declararlo como parámetro y, si se vuelve decisivo, marcar que la conclusión es condicional a ese prior.
5. **Geometría de un solo pozo / un solo tipo de estado:** este test cubre la familia well-building/pozo-de-I. Es la más visible, no la única. Generalizar a otras familias es trabajo posterior; no sobre-interpretar un piso bajo en pozos como "no hay confound en ninguna familia".
6. **`N(H)` escalar vs estocástico:** `N(H)` bajo π_fill es una distribución (las piezas de relleno vienen del bag), y el modelo `p_grad = min(N/m,1)` la colapsa a `E[N|H]`. Además, la secuencia de relleno y la llegada de la I comparten el mismo bag → están acopladas, y `p_grad` las trata como separables. Es aproximación de primer orden; si el piso resulta sensible a esta simplificación (chequeable variando cómo se resume `N`), hay que modelar `N` distribucional antes de la potencia.

---

## Resumen operativo (una frase)

Simular partidas 7-bag reales con censura, hacer que un **no-tracker con tablero completo** juegue el escenario de pozo, correr el estimador de Fase 1A controlando solo features BCTS lossy, y **medir el β residual** que ese no-tracker produce: ese número es el piso contra el cual —no contra cero— deberá competir cualquier humano, y su descomposición A/B dice si el remedio es más features o restringir estados.
