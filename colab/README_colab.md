# Pipeline de Colab para Tetris — Fase 1A: piso y potencia

Este directorio contiene un notebook de Google Colab para replicar el análisis de
potencia de Fase 1A en un entorno con más recursos que la máquina local.

Repo: `https://github.com/yudocyudoc/tetris.git`

## Antes de usar

1. Abre `tetris_confound_colab.ipynb` en Colab.
2. La celda de configuración ya apunta al repo.
3. Si usas Colab Pro, activa un runtime de mayor capacidad (Runtime → Change runtime type → CPU de alto rendimiento). El código es CPU-bound, pero un runtime más potente acelera las simulaciones.

## Qué hace el notebook

1. Clona el repo en `/content/tetris`.
2. Instala las dependencias necesarias.
3. Corre `confound_floor_t2.py` para `tracker_prob ∈ {0.10, 0.25, 0.50, 0.75}` con `n_games=100` y `max_pieces=500`.
4. Corre `power_curve_t2.py` sobre la corrida `tracker_prob=0.5`.
5. Corre `fast_calibrate_signal_v2.py` como proxy de la forma de `β_señal(p)`.
6. Recalcula `p_min` con la curva calibrada.
7. Empaqueta los resultados en un zip descargable.

## Tiempo estimado

- `tracker_prob=0.5, n=100`: ~5–15 min en Colab Pro (depende del runtime).
- Cuatro valores de `tracker_prob`: ~20–60 min.
- Análisis de potencia y calibración: ~5 min.

Si el tiempo es crítico, reduce `N_GAMES` a 50 en la celda de configuración.

## Salidas

En `/content/tetris/analysis/confound_floor/`:
- `out_t2_H15_n100_tp{010,025,050,075}_nocensor/`: simulaciones por tracker_prob.
- `out_power_t2/`: curvas de potencia lineales.
- `out_fast_calibrate_v2/`: curvas calibradas.
- `results_colab.zip`: paquete con todos los JSON y figuras para descargar.

## Nota sobre el repo

El notebook asume que el repo contiene los scripts en
`analysis/confound_floor/`:
- `confound_floor_t2.py`
- `power_curve_t2.py`
- `fast_calibrate_signal_v2.py`
- `recalc_power_calibrated.py`

Si cambias la estructura de archivos, actualiza las rutas en el notebook.
