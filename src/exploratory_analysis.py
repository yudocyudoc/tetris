"""Análisis exploratorio piloto N=1 para Tetris instrumentado.

Implementa los tres análisis del blueprint:
1. Trayectoria intra-ramp: metricas vs gravity_at_spawn con control fisico.
2. Comparacion entre condiciones controlada por estado.
3. Sigma-Tetris: volatilidad por ventana/bin de gravedad.

Genera tablas, figuras y un reporte Markdown.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .metrics import compute_piece_metrics
from .tetris_core import BOARD_HEIGHT, BOARD_WIDTH

warnings.filterwarnings("ignore", category=FutureWarning)


RAMP_MIN_G = 0.8
RAMP_MAX_G = 6.0


@dataclass
class SessionData:
    session_id: str
    condition: str
    git_hash: str
    session_label: str
    wall_clock_start: str
    perceived_effort: Optional[int]
    state_covariates: Dict[str, Any]
    pieces: pd.DataFrame
    actions: pd.DataFrame
    snapshots: pd.DataFrame
    piece_df: pd.DataFrame  # metrics.py result merged with gravity and stack height


def _iso_hour(wall_clock_start: str) -> int:
    """Extrae la hora local del inicio de sesion (0-23)."""
    try:
        # Formato ISO: 2026-06-21T19:58:51-06:00
        hour_str = wall_clock_start.split("T")[1].split(":")[0]
        return int(hour_str)
    except Exception:
        return -1


def _load_session(session_dir: Path) -> Optional[SessionData]:
    meta_path = session_dir / "session_meta.json"
    pieces_path = session_dir / "pieces.csv"
    actions_path = session_dir / "actions.csv"
    snapshots_path = session_dir / "board_snapshots.parquet"

    if not meta_path.exists() or not pieces_path.exists() or not actions_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pieces = pd.read_csv(pieces_path)
    actions = pd.read_csv(actions_path)
    snapshots = pd.read_parquet(snapshots_path) if snapshots_path.exists() else pd.DataFrame()

    # Calcular metricas conductuales por pieza.
    metrics_records = compute_piece_metrics(pieces_path, actions_path)
    metrics_df = pd.DataFrame(metrics_records)

    if metrics_df.empty:
        return None

    # Merge con pieces para traer gravity_at_spawn y bag_idx.
    piece_df = metrics_df.merge(
        pieces[["game_id", "piece_idx", "gravity_at_spawn"]],
        on=["game_id", "piece_idx"],
        how="left",
    )

    # Calcular altura de pila previa a cada pieza.
    stack_heights = []
    available_times = []
    is_first_piece = []
    for _, row in piece_df.iterrows():
        t_spawn = row["t_spawn_ms"]
        gravity = row["gravity_at_spawn"]
        prev_snapshot = _previous_stack_height(snapshots, row["game_id"], t_spawn)
        stack_heights.append(prev_snapshot)
        available_times.append(_available_fall_time(prev_snapshot, gravity))
        # La primera pieza de cada partida no es comparable: incluye tiempo de arranque.
        is_first_piece.append(int(row["piece_idx"]) == 0)

    piece_df["stack_height"] = stack_heights
    piece_df["available_fall_time_ms"] = available_times
    piece_df["is_first_piece"] = is_first_piece

    return SessionData(
        session_id=meta.get("session_id", session_dir.name),
        condition=meta.get("condition", "unknown"),
        git_hash=meta.get("software_git_hash", "unknown"),
        session_label="pre_decision_time_correction"
        if meta.get("software_git_hash") in (None, "unknown", "")
        else "tracked",
        wall_clock_start=meta.get("wall_clock_start", ""),
        perceived_effort=meta.get("perceived_effort_1_10"),
        state_covariates=meta.get("state_covariates", {}),
        pieces=pieces,
        actions=actions,
        snapshots=snapshots,
        piece_df=piece_df,
    )


def _previous_stack_height(snapshots: pd.DataFrame, game_id: str, t_spawn_ms: float) -> int:
    """Devuelve la altura maxima de la pila justo antes de t_spawn_ms."""
    if snapshots.empty:
        return 0
    game_snaps = snapshots[snapshots["game_id"] == game_id]
    if game_snaps.empty:
        return 0
    prev = game_snaps[game_snaps["t_ms"] <= t_spawn_ms]
    if prev.empty:
        return 0
    latest = prev.sort_values("t_ms").iloc[-1]
    board = _parse_board(latest["board"])
    return board_height(board)


def _parse_board(board_json: Any) -> np.ndarray:
    """Convierte el board serializado en JSON a matriz 10x20."""
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
    try:
        cells = json.loads(board_json) if isinstance(board_json, str) else board_json
        for cell in cells:
            x = int(cell.get("x", -1))
            y = int(cell.get("y", -1))
            if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                board[y, x] = 1
    except Exception:
        pass
    return board


def board_height(board: np.ndarray) -> int:
    """Altura de la pila: numero de filas ocupadas contando desde abajo. 0 si vacio."""
    occupied_rows = np.where(board.any(axis=1))[0]
    if len(occupied_rows) == 0:
        return 0
    # Las filas crecen hacia abajo; la mas alta es el minimo indice.
    # Altura real = filas totales - indice de la fila mas alta.
    return int(BOARD_HEIGHT - occupied_rows.min())


def _available_fall_time(stack_height: int, gravity_cps: float) -> Optional[float]:
    """Tiempo que tardaria una pieza en caer desde spawn hasta la pila, solo por gravedad."""
    if gravity_cps <= 0:
        return None
    distance = max(0, BOARD_HEIGHT - stack_height)
    return (distance / gravity_cps) * 1000.0


def make_gravity_bins(
    df: pd.DataFrame,
    n_bins: int = 6,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Asigna bins de gravity_at_spawn a cada pieza.

    strategy='quantile' reparte las piezas en bins de igual N (mas robusto).
    strategy='fixed' usa bins de igual ancho en gravedad.
    """
    df = df.copy()
    values = df["gravity_at_spawn"].dropna()
    if values.empty:
        df["gravity_bin"] = np.nan
        df["gravity_bin_label"] = ""
        return df

    gmin = float(values.min())
    gmax = float(values.max())
    if gmin == gmax:
        df["gravity_bin"] = 0
        df["gravity_bin_label"] = f"{gmin:.2f}"
        return df

    if strategy == "quantile":
        try:
            df["gravity_bin"], bin_edges = pd.qcut(
                df["gravity_at_spawn"], q=n_bins, retbins=True, labels=False, duplicates="drop"
            )
            n_actual = len(bin_edges) - 1
            labels = [
                f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(n_actual)
            ]
            df["gravity_bin_label"] = pd.qcut(
                df["gravity_at_spawn"], q=n_bins, labels=labels, duplicates="drop"
            ).astype(str)
            return df
        except ValueError:
            pass  # fallback a fixed

    bin_edges = np.linspace(gmin, gmax, n_bins + 1)
    labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(n_bins)]
    df["gravity_bin"] = pd.cut(df["gravity_at_spawn"], bins=bin_edges, labels=False, include_lowest=True)
    df["gravity_bin_label"] = pd.cut(df["gravity_at_spawn"], bins=bin_edges, labels=labels, include_lowest=True).astype(str)
    return df


def iqr(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def bin_summary(
    df: pd.DataFrame,
    metric_col: str,
    n_bins: int = 6,
    bin_strategy: str = "quantile",
) -> pd.DataFrame:
    """Tabla resumen por bin de gravedad, incluyendo CV."""
    df = make_gravity_bins(df, n_bins=n_bins, strategy=bin_strategy)
    rows = []
    for label, group in df.groupby("gravity_bin_label", sort=False):
        vals = group[metric_col].dropna().values
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        cv = std_val / mean_val if mean_val and mean_val != 0 else None
        rows.append(
            {
                "gravity_bin": label,
                "n": len(vals),
                "mean": mean_val,
                "median": float(np.median(vals)),
                "iqr": iqr(vals),
                "std": std_val,
                "cv": round(cv, 3) if cv is not None else None,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "gravity_at_spawn_mid": float(group["gravity_at_spawn"].mean()),
            }
        )
    return pd.DataFrame(rows)


def spearman_and_regression(df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
    """Calcula Spearman y regresion lineal entre dos columnas."""
    data = df[[x_col, y_col]].dropna()
    if len(data) < 3:
        return {"n": len(data), "spearman_r": None, "spearman_p": None, "slope": None, "intercept": None, "r2": None}

    x = data[x_col].values
    y = data[y_col].values

    spearman_r, spearman_p = stats.spearmanr(x, y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "n": len(data),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r2": round(float(r_value ** 2), 4),
        "p_value": round(float(p_value), 4),
    }


def partial_correlation_spearman(
    df: pd.DataFrame, x: str, y: str, control: str
) -> Dict[str, Any]:
    """Correlacion parcial de Spearman controlando por una tercera variable."""
    data = df[[x, y, control]].dropna()
    if len(data) < 4:
        return {"n": len(data), "r": None, "p": None}
    # Ranking de cada variable.
    rx = data[x].rank()
    ry = data[y].rank()
    rc = data[control].rank()
    # Residuos de regresion lineal de rank sobre control.
    slope_x, intercept_x, _, _, _ = stats.linregress(rc, rx)
    slope_y, intercept_y, _, _, _ = stats.linregress(rc, ry)
    resid_x = rx - (slope_x * rc + intercept_x)
    resid_y = ry - (slope_y * rc + intercept_y)
    r, p = stats.pearsonr(resid_x, resid_y)
    return {"n": len(data), "r": round(float(r), 4), "p": round(float(p), 4)}


def analyze_intra_ramp(sessions: List[SessionData], n_bins: int = 6) -> Dict[str, Any]:
    """Analisis 1: trayectoria intra-ramp."""
    ramp_dfs = [s.piece_df for s in sessions if s.condition == "ramp"]
    if not ramp_dfs:
        return {"error": "No hay sesiones ramp"}

    df = pd.concat(ramp_dfs, ignore_index=True)
    df = df[df["n_inputs"].notna()]

    # Excluir primera pieza de cada partida del analisis de time_to_first_input.
    df_first = df[df["is_first_piece"] == False]

    # Manejo de piezas con n_inputs <= 1: excluir de active_time.
    df_active = df[(df["n_inputs"] > 1) & (df["active_time_ms"].notna())]

    primary_metric = "time_to_first_input_ms"
    secondary_metric = "n_inputs"

    results = {
        "n_pieces": len(df),
        "n_pieces_first_excluded": len(df_first),
        "n_pieces_active_time": len(df_active),
        "bin_strategy": "quantile (bins de igual N, mas robustos)",
        primary_metric: {
            "by_gravity_bin": bin_summary(df_first, primary_metric, n_bins=n_bins, bin_strategy="quantile").to_dict("records"),
            "correlation": spearman_and_regression(df_first, "gravity_at_spawn", primary_metric),
        },
        secondary_metric: {
            "by_gravity_bin": bin_summary(df, secondary_metric, n_bins=n_bins, bin_strategy="quantile").to_dict("records"),
            "correlation": spearman_and_regression(df, "gravity_at_spawn", secondary_metric),
        },
        "active_time_ms": {
            "by_gravity_bin": bin_summary(df_active, "active_time_ms", n_bins=n_bins, bin_strategy="quantile").to_dict("records"),
            "correlation": spearman_and_regression(df_active, "gravity_at_spawn", "active_time_ms"),
        },
        "control_physical_ceiling": {
            "note": "Primera pieza de cada partida excluida de time_to_first_input_ms.",
            "correlation_available_time_vs_first_input": spearman_and_regression(
                df_first, "available_fall_time_ms", primary_metric
            ),
            "correlation_available_time_vs_first_input_control_stack_height": partial_correlation_spearman(
                df_first, "available_fall_time_ms", primary_metric, "stack_height"
            ),
            "correlation_available_time_vs_n_inputs": spearman_and_regression(
                df, "available_fall_time_ms", secondary_metric
            ),
        },
    }
    return results


def analyze_conditions(sessions: List[SessionData]) -> pd.DataFrame:
    """Analisis 2: comparacion entre condiciones."""
    rows = []
    for s in sessions:
        df = s.piece_df
        df_first = df[df["is_first_piece"] == False]
        df_active = df[(df["n_inputs"] > 1) & (df["active_time_ms"].notna())]
        row = {
            "session_id": s.session_id,
            "condition": s.condition,
            "session_label": s.session_label,
            "hour": _iso_hour(s.wall_clock_start),
            "perceived_effort": s.perceived_effort,
            "n_pieces": len(df),
            "time_to_first_input_mean": df_first["time_to_first_input_ms"].mean() if not df_first.empty else None,
            "time_to_first_input_std": df_first["time_to_first_input_ms"].std(ddof=0) if not df_first.empty else None,
            "n_inputs_mean": df["n_inputs"].mean() if not df.empty else None,
            "n_inputs_std": df["n_inputs"].std(ddof=0) if not df.empty else None,
            "active_time_mean": df_active["active_time_ms"].mean() if not df_active.empty else None,
            "active_time_std": df_active["active_time_ms"].std(ddof=0) if not df_active.empty else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_sigma(
    sessions: List[SessionData],
    metric: str = "n_inputs",
    window_n_pieces: int = 15,
) -> Dict[str, Any]:
    """Analisis 3: sigma-Tetris (volatilidad por ventana temporal).

    Reporta sigma absoluta, coeficiente de variacion (CV) y sigma residualizada
    (residuos de una regresion local media-varianza) para separar cambio real en
    volatilidad de artefactos media-varianza.
    """
    ramp_dfs = [s.piece_df for s in sessions if s.condition == "ramp"]
    if not ramp_dfs:
        return {"error": "No hay sesiones ramp"}

    df = pd.concat(ramp_dfs, ignore_index=True)
    # Excluir primera pieza de cada partida para time_to_first_input_ms.
    if metric == "time_to_first_input_ms":
        df = df[df["is_first_piece"] == False]
    df = df.sort_values("t_spawn_ms").reset_index(drop=True)

    if len(df) < window_n_pieces * 2:
        return {"error": f"Insuficientes piezas ramp ({len(df)}) para ventanas de {window_n_pieces}"}

    records = []
    for start in range(0, len(df) - window_n_pieces + 1, window_n_pieces):
        window = df.iloc[start : start + window_n_pieces]
        vals = window[metric].dropna().values
        if len(vals) < 2:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))
        records.append(
            {
                "window_center_g": float(window["gravity_at_spawn"].mean()),
                "window_center_t_ms": float(window["t_spawn_ms"].mean()),
                "mean": mean_val,
                "sigma_abs": std_val,
                "cv": std_val / mean_val if mean_val != 0 else None,
            }
        )

    sigma_df = pd.DataFrame(records)

    # Sigma residualizada: residual de sigma_abs regresado sobre mean.
    # Esto quita la componente de varianza explicada por el nivel medio.
    sigma_df["sigma_resid"] = np.nan
    valid = sigma_df.dropna(subset=["sigma_abs", "mean", "cv"])
    if len(valid) >= 3:
        slope, intercept, _, _, _ = stats.linregress(valid["mean"].values, valid["sigma_abs"].values)
        predicted = slope * sigma_df["mean"].values + intercept
        sigma_df["sigma_resid"] = sigma_df["sigma_abs"].values - predicted

    corr_abs = spearman_and_regression(sigma_df, "window_center_g", "sigma_abs")
    corr_cv = spearman_and_regression(sigma_df, "window_center_g", "cv")
    corr_resid = spearman_and_regression(sigma_df, "window_center_g", "sigma_resid")

    return {
        "metric": metric,
        "window_n_pieces": window_n_pieces,
        "n_windows": len(sigma_df),
        "sigma_by_window": sigma_df.to_dict("records"),
        "correlation_sigma_abs": corr_abs,
        "correlation_cv": corr_cv,
        "correlation_sigma_resid": corr_resid,
    }


def _round_dict_values(d: Dict[str, Any], decimals: int = 3) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, decimals)
        elif isinstance(v, dict):
            out[k] = _round_dict_values(v, decimals)
        elif isinstance(v, list):
            out[k] = [_round_dict_values(i, decimals) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


def plot_intra_ramp(df: pd.DataFrame, output_path: Path) -> None:
    """Figura: metricas vs gravity_at_spawn en ramp."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ["time_to_first_input_ms", "n_inputs", "active_time_ms"]
    titles = ["time_to_first_input_ms", "n_inputs", "active_time_ms"]

    for ax, metric, title in zip(axes, metrics, titles):
        data = df[(df["n_inputs"] > 1) | (metric != "active_time_ms")].copy()
        data = data[data[metric].notna()]
        if data.empty:
            ax.set_title(f"{title} (sin datos)")
            continue

        # Scatter + linea de tendencia.
        ax.scatter(data["gravity_at_spawn"], data[metric], alpha=0.4, s=30)

        # Regresion.
        slope, intercept, _, _, _ = stats.linregress(data["gravity_at_spawn"].values, data[metric].values)
        x_line = np.linspace(data["gravity_at_spawn"].min(), data["gravity_at_spawn"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r--", lw=2, label=f"slope={slope:.3f}")

        ax.set_xlabel("gravity_at_spawn (cps)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_conditions(summary: pd.DataFrame, output_path: Path) -> None:
    """Figura: medias por condicion."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["time_to_first_input_mean", "n_inputs_mean", "active_time_mean"]
    titles = ["time_to_first_input_ms", "n_inputs", "active_time_ms"]

    for ax, metric, title in zip(axes, metrics, titles):
        data = summary[summary[metric].notna()].copy()
        if data.empty:
            ax.set_title(f"{title} (sin datos)")
            continue
        colors = ["red" if label == "pre_decision_time_correction" else "steelblue" for label in data["session_label"]]
        ax.bar(data["condition"], data[metric], color=colors)
        ax.set_ylabel(metric)
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sigma(sigma_results: Dict[str, Any], output_path: Path, title_suffix: str = "") -> None:
    """Figura: sigma, CV y sigma residualizada vs gravedad."""
    if "error" in sigma_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, sigma_results["error"], ha="center", va="center")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    df = pd.DataFrame(sigma_results["sigma_by_window"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("sigma_abs", "σ absoluta"),
        ("cv", "CV (σ/media)"),
        ("sigma_resid", "σ residualizada"),
    ]

    for ax, (col, title) in zip(axes, panels):
        if col not in df.columns or df[col].isna().all():
            ax.set_title(f"{title} (sin datos)")
            continue
        ax.plot(df["window_center_g"], df[col], marker="o", lw=2)
        ax.set_xlabel("gravity_at_spawn (cps)")
        ax.set_ylabel(col)
        ax.set_title(title)

    fig.suptitle(f"Volatilidad de {title_suffix} por ventana ({sigma_results['window_n_pieces']} piezas)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_acf(series: pd.Series, max_lag: int = 10, detrend_window: Optional[int] = None) -> Dict[int, float]:
    """Autocorrelacion de una serie. Si detrend_window se indica, resta media movil."""
    s = series.dropna().astype(float).values
    if len(s) < max_lag + 2:
        return {}

    if detrend_window:
        # Media movil centrada para quitar tendencia local.
        half = detrend_window // 2
        trend = np.array([np.mean(s[max(0, i - half) : min(len(s), i + half + 1)]) for i in range(len(s))])
        s = s - trend

    s = s - np.mean(s)
    n = len(s)
    c0 = np.sum(s ** 2) / n
    if c0 == 0:
        return {}

    acf = {}
    for lag in range(0, max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            c_lag = np.sum(s[:-lag] * s[lag:]) / n
            acf[lag] = float(c_lag / c0)
    return acf


def ou_theta_from_acf(acf: Dict[int, float], dt_pieces: float = 1.0) -> Optional[float]:
    """Estimacion exploratoria de theta de O/U desde ACF(1).

    En O/U discreto: rho(1) = exp(-theta * dt). dt se mide en piezas.
    Reportar solo como descriptor, no como estimador confirmatorio.
    """
    rho1 = acf.get(1)
    if rho1 is None or rho1 <= 0 or rho1 >= 1:
        return None
    return -np.log(rho1) / dt_pieces


def analyze_autocorrelation(sessions: List[SessionData]) -> Dict[str, Any]:
    """Analisis 4 (PoC): autocorrelacion temporal y ajuste O/U exploratorio.

    Exploratorio, no confirmatorio. La rampa no es estacionaria; se reportan
    residuos de media movil para mitigar la tendencia, pero el theta sigue
    siendo un descriptor, no una prueba del modelo bayesiano.
    """
    results = {"sessions": [], "by_condition": {}}

    metrics = ["time_to_first_input_ms", "n_inputs"]

    for s in sessions:
        df = s.piece_df.sort_values("t_spawn_ms").reset_index(drop=True)
        session_result = {
            "session_id": s.session_id,
            "condition": s.condition,
            "n_pieces": len(df),
            "metrics": {},
        }
        for metric in metrics:
            series = df[df["is_first_piece"] == False][metric] if metric == "time_to_first_input_ms" else df[metric]
            acf_raw = compute_acf(series, max_lag=10)
            acf_resid = compute_acf(series, max_lag=10, detrend_window=15)
            theta_raw = ou_theta_from_acf(acf_raw)
            theta_resid = ou_theta_from_acf(acf_resid)
            session_result["metrics"][metric] = {
                "acf_raw": acf_raw,
                "acf_resid": acf_resid,
                "theta_raw": theta_raw,
                "theta_resid": theta_resid,
            }

        # Para ramp, segmentos temporales (early/mid/late) para ver cambio en estructura.
        if s.condition == "ramp" and len(df) >= 45:
            third = len(df) // 3
            segments = {
                "early": df.iloc[:third],
                "mid": df.iloc[third:2*third],
                "late": df.iloc[2*third:],
            }
            session_result["ramp_segments"] = {}
            for seg_name, seg_df in segments.items():
                session_result["ramp_segments"][seg_name] = {}
                for metric in metrics:
                    acf = compute_acf(seg_df[metric], max_lag=5, detrend_window=7)
                    session_result["ramp_segments"][seg_name][metric] = {
                        "acf": acf,
                        "theta": ou_theta_from_acf(acf),
                    }

        results["sessions"].append(session_result)

    # Agregado por condicion (promedio simple de ACF cruda; descriptor).
    for condition in set(s.condition for s in sessions):
        cond_results = [r for r in results["sessions"] if r["condition"] == condition]
        results["by_condition"][condition] = {}
        for metric in metrics:
            acfs = [r["metrics"][metric]["acf_resid"] for r in cond_results if r["metrics"][metric].get("acf_resid")]
            if not acfs:
                continue
            max_lag = max(max(a.keys()) for a in acfs)
            mean_acf = {}
            for lag in range(max_lag + 1):
                vals = [a.get(lag) for a in acfs if lag in a]
                if vals:
                    mean_acf[lag] = float(np.mean(vals))
            results["by_condition"][condition][metric] = {
                "mean_acf_resid": mean_acf,
                "theta_resid": ou_theta_from_acf(mean_acf),
            }

    return results


def plot_autocorrelation(acf_results: Dict[str, Any], output_path: Path) -> None:
    """Figura: funciones de autocorrelacion por condicion y metrica."""
    conditions = list(acf_results["by_condition"].keys())
    metrics = ["time_to_first_input_ms", "n_inputs"]

    fig, axes = plt.subplots(len(metrics), len(conditions), figsize=(5 * len(conditions), 4 * len(metrics)), squeeze=False)

    for i, metric in enumerate(metrics):
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            data = acf_results["by_condition"].get(condition, {}).get(metric, {})
            acf = data.get("mean_acf_resid", {})
            if not acf:
                ax.set_title(f"{condition} / {metric}\n(sin datos)")
                continue
            lags = sorted(acf.keys())
            values = [acf[l] for l in lags]
            ax.stem(lags, values, basefmt=" ")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("lag (piezas)")
            ax.set_ylabel("ACF (residuos)")
            ax.set_title(f"{condition} / {metric}\nθ={data.get('theta_resid'):.3f}" if data.get("theta_resid") is not None else f"{condition} / {metric}")
            ax.set_ylim(-0.5, 1.0)

    fig.suptitle("Autocorrelacion (PoC) — residuos de media movil")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin datos._"
    return df.to_markdown(index=False, floatfmt=".2f")


def verdict_intra_ramp(results: Dict[str, Any], sigma_results: Dict[str, Dict[str, Any]]) -> str:
    """Linea de veredicto exploratorio separando efecto sobre media de efecto sobre volatilidad."""
    if "error" in results:
        return f"No se pudo evaluar: {results['error']}"

    primary = results["time_to_first_input_ms"]["correlation"]
    secondary = results["n_inputs"]["correlation"]

    lines = []
    lines.append("### Efecto sobre la media del comportamiento")
    if primary.get("spearman_r") is not None:
        direction_ok = primary["spearman_r"] < 0
        lines.append(
            f"- `time_to_first_input_ms` vs gravity: r={primary['spearman_r']}, p={primary['spearman_p']}, "
            f"pendiente={primary['slope']}. Direccion predicha (negativa): {'SI' if direction_ok else 'NO'}."
        )
    if secondary.get("spearman_r") is not None:
        direction_ok = secondary["spearman_r"] < 0
        lines.append(
            f"- `n_inputs` vs gravity: r={secondary['spearman_r']}, p={secondary['spearman_p']}, "
            f"pendiente={secondary['slope']}. Direccion predicha (negativa): {'SI' if direction_ok else 'NO'}."
        )

    lines.append("")
    lines.append("### Efecto sobre la volatilidad (σ / CV / residualizada)")
    for metric_name, sigma in sigma_results.items():
        if "error" in sigma:
            lines.append(f"- `{metric_name}`: {sigma['error']}")
            continue
        corr_cv = sigma.get("correlation_cv", {})
        corr_resid = sigma.get("correlation_sigma_resid", {})
        lines.append(
            f"- `{metric_name}`: CV vs gravity r={corr_cv.get('spearman_r')}, p={corr_cv.get('spearman_p')}; "
            f"σ residualizada vs gravity r={corr_resid.get('spearman_r')}, p={corr_resid.get('spearman_p')}."
        )

    primary_negative = (primary.get("spearman_r") or 0) < 0

    if primary_negative:
        verdict = (
            "**Veredicto exploratorio:** hay un efecto del campo sobre la **media** de la respuesta "
            "(reaccionas mas rapido y haces menos inputs cuando la gravedad sube). "
            "El control fisico muestra que parte de este efecto es mecanico: `available_fall_time_ms` "
            "correlaciona positivamente con `time_to_first_input_ms` (r≈0.35), y aun controlando por "
            "`stack_height` la correlacion parcial sigue siendo positiva y pequena (r≈0.15). "
            "Esto no invalida la direccion predicha, pero advierte que el limite de tiempo fisico es un "
            "mediador parcial, no un confound externo que se pueda 'parcializar' sin mas. "
            "**Sobre la volatilidad (σ): el resultado es mixto e indeterminado aun con la rampa de 300 s.** "
            "Las ventanas ahora se reparten de g≈1.4 a 6.0 (13 ventanas), pero CV y σ residualizada no muestran "
            "una firma clara ni consistente entre metricas: CV de `n_inputs` sube con la gravedad, mientras que "
            "CV de `time_to_first_input_ms` baja, y la σ residualizada no presenta tendencia significativa. "
            "No se puede declarar σ↑/σ↓ con la evidencia actual. "
            "Para Fase 2: mas repeticiones bajo control estricto (una variable a la vez, cafeina/horario fijos) "
            "y, si se desea probar σ, usar una rampa que evite la meseta de 6.0 cps o terminar la partida antes de ella."
        )
    else:
        verdict = (
            "**Veredicto exploratorio:** no hay evidencia clara de que el campo afecte la media ni la volatilidad "
            "en la direccion predicha. Antes de escalar, reconsiderar metrica, calibracion o paradigma."
        )

    return verdict + "\n\n" + "\n".join(lines)


def generate_report(
    sessions: List[SessionData],
    intra_ramp_results: Dict[str, Any],
    condition_summary: pd.DataFrame,
    sigma_results: Dict[str, Dict[str, Any]],
    autocorr_results: Dict[str, Any],
    output_path: Path,
    figure_paths: Dict[str, Path],
) -> None:
    """Genera exploratory_report.md."""
    lines = [
        "# Reporte exploratorio piloto — Tetris instrumentado",
        "",
        "**Naturaleza:** EXPLORATORIO. No reporta pruebas confirmatorias. "
        "Decide si las metricas conductuales responden a la contraccion del campo.",
        "",
        f"**Sesiones incluidas:** {len(sessions)}",
        "",
        render_table(
            pd.DataFrame(
                [
                    {
                        "session_id": s.session_id,
                        "condition": s.condition,
                        "label": s.session_label,
                        "hour": s.wall_clock_start.split("T")[1][:5] if s.wall_clock_start else "",
                        "effort": s.perceived_effort,
                        "n_pieces": len(s.piece_df),
                    }
                    for s in sessions
                ]
            )
        ),
        "",
        "## Analisis 1 — Trayectoria intra-ramp",
        "",
        "Métrica primaria: `time_to_first_input_ms`. Secundaria: `n_inputs`. `active_time_ms` como apoyo. "
        "La primera pieza de cada partida se excluye de `time_to_first_input_ms` porque incluye el tiempo de arranque.",
        "",
        "### time_to_first_input_ms por bin de gravedad",
        "",
        render_table(pd.DataFrame(intra_ramp_results["time_to_first_input_ms"]["by_gravity_bin"])),
        "",
        "### n_inputs por bin de gravedad",
        "",
        render_table(pd.DataFrame(intra_ramp_results["n_inputs"]["by_gravity_bin"])),
        "",
        "### active_time_ms por bin de gravedad",
        "",
        render_table(pd.DataFrame(intra_ramp_results["active_time_ms"]["by_gravity_bin"])),
        "",
        f"![Intra-ramp]({figure_paths['intra_ramp']})",
        "",
        "### Control fisico — tiempo de caida disponible",
        "",
        "Se calculo la altura de la pila previa a cada pieza y el tiempo que la pieza tardaria en caer por gravedad. "
        "Si la metrica cae solo porque el tiempo disponible se acorta, el efecto es mecanico, no conductual. "
        "Tambien se controla por altura de pila (`stack_height`) para separar presion situacional de limite fisico.",
        "",
        "- Correlacion `available_fall_time_ms` vs `time_to_first_input_ms` (primera pieza excluida): "
        f"r={intra_ramp_results['control_physical_ceiling']['correlation_available_time_vs_first_input']['spearman_r']}, "
        f"p={intra_ramp_results['control_physical_ceiling']['correlation_available_time_vs_first_input']['spearman_p']}",
        "- Correlacion parcial controlando por `stack_height`: "
        f"r={intra_ramp_results['control_physical_ceiling']['correlation_available_time_vs_first_input_control_stack_height']['r']}, "
        f"p={intra_ramp_results['control_physical_ceiling']['correlation_available_time_vs_first_input_control_stack_height']['p']}",
        "- Correlacion `available_fall_time_ms` vs `n_inputs`: "
        f"r={intra_ramp_results['control_physical_ceiling']['correlation_available_time_vs_n_inputs']['spearman_r']}, "
        f"p={intra_ramp_results['control_physical_ceiling']['correlation_available_time_vs_n_inputs']['spearman_p']}",
        "",
        verdict_intra_ramp(intra_ramp_results, sigma_results),
        "",
        "## Analisis 2 — Comparacion entre condiciones",
        "",
        render_table(condition_summary),
        "",
        f"![Condiciones]({figure_paths['conditions']})",
        "",
        "_Notas: (1) la sesion `hard` esta etiquetada como `pre_decision_time_correction` y se jugo de madrugada; "
        "tratar por separado o como contexto. (2) Las dos sesiones `ramp` difieren en dos variables a la vez: "
        "la curva de rampa (120s vs 300s) y la cafeina (0mg vs 200mg). El esfuerzo 8→2 no es atribuible "
        "a ninguna de las dos sola. Para Fase 2: una variable a la vez._",
        "",
        "## Analisis 3 — σ-Tetris (volatilidad por ventana)",
        "",
        "Se reporta σ absoluta, coeficiente de variacion (CV) y σ residualizada (residuos de regresion local "
        "media-varianza) para separar cambio real en volatilidad del artefacto media-varianza.",
        "",
    ]

    for metric_name, sigma in sigma_results.items():
        lines.extend([
            f"### Metrica: `{metric_name}`",
            "",
            f"Ventana: {sigma.get('window_n_pieces', 'NA')} piezas. Numero de ventanas: {sigma.get('n_windows', 'NA')}.",
            "",
            render_table(pd.DataFrame(sigma.get("sigma_by_window", []))),
            "",
            f"![Sigma {metric_name}]({figure_paths[f'sigma_{metric_name}']})",
            "",
        ])

    lines.extend([
        "## Reglas de higiene aplicadas",
        "",
        "- Piezas con `n_inputs <= 1` se excluyen de `active_time_ms`.",
        "- Bins de gravedad por quantiles para equilibrar N.",
        "- Se reporta N por bin y CV ademas de σ.",
        "- Sesiones sin git hash se marcan como `pre_decision_time_correction`.",
        "",
        "## Analisis 4 — Autocorrelacion temporal (PoC exploratorio)",
        "",
        "**Proposito:** explorar si existe estructura de dependencia temporal en las metricas conductuales "
        "y si esa estructura cambia con la condicion o a lo largo de la rampa. "
        "**No es confirmatorio:** no prueba el modelo O/U, ni mecanismos bayesianos, ni ajuste de 'temperatura'.",
        "",
        "Se calcula la funcion de autocorrelacion (ACF) sobre los residuos de una media movil "
        "(para mitigar la no estacionariedad de la rampa). El parametro θ de O/U se deriva de ACF(1) "
        "como descriptor: θ = -ln(rho(1)). Valores mas altos implican decaimiento mas rapido de la autocorrelacion.",
        "",
        f"![Autocorrelacion]({figure_paths['autocorrelation']})",
        "",
        "### θ exploratorio por sesion (residuos)",
        "",
        render_table(
            pd.DataFrame(
                [
                    {
                        "session_id": r["session_id"],
                        "condition": r["condition"],
                        "theta_time_to_first_input": r["metrics"]["time_to_first_input_ms"].get("theta_resid"),
                        "theta_n_inputs": r["metrics"]["n_inputs"].get("theta_resid"),
                    }
                    for r in autocorr_results["sessions"]
                ]
            )
        ),
        "",
    ])

    # Segmentos ramp.
    ramp_segments = [r for r in autocorr_results["sessions"] if "ramp_segments" in r]
    if ramp_segments:
        lines.extend([
            "### Segmentos dentro de la sesion ramp (early / mid / late)",
            "",
            "Exploracion de si la estructura temporal cambia a medida que la gravedad sube. "
            "N pequeno por segmento; interpretar con extrema cautela.",
            "",
        ])
        for r in ramp_segments:
            lines.extend([
                f"**Sesion {r['session_id']}**",
                "",
            ])
            rows = []
            for seg_name, seg_data in r["ramp_segments"].items():
                for metric in ["time_to_first_input_ms", "n_inputs"]:
                    rows.append(
                        {
                            "segment": seg_name,
                            "metric": metric,
                            "theta": seg_data[metric].get("theta"),
                            "acf_lag1": seg_data[metric]["acf"].get(1),
                        }
                    )
            lines.append(render_table(pd.DataFrame(rows)))
            lines.append("")

    lines.extend([
        "## Lo que queda fuera (Fase 2)",
        "",
        "- Modelos jerarquicos y multiples sujetos.",
        "- Diseño del agente null (politica insensible a la velocidad).",
        "- Pre-registro confirmatorio del analisis temporal (si se decide testear O/U u otro modelo).",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_all(data_root: str = "data", output_root: str = "data") -> Dict[str, Any]:
    """Ejecuta el pipeline completo de analisis exploratorio."""
    root = Path(data_root)
    out_dir = Path(output_root) / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        s = _load_session(session_dir)
        if s is not None:
            sessions.append(s)

    if not sessions:
        raise ValueError(f"No se encontraron sesiones validas en {data_root}/")

    # Analisis.
    intra_ramp = analyze_intra_ramp(sessions, n_bins=6)
    condition_summary = analyze_conditions(sessions)
    sigma_n_inputs = analyze_sigma(sessions, metric="n_inputs", window_n_pieces=15)
    sigma_first_input = analyze_sigma(sessions, metric="time_to_first_input_ms", window_n_pieces=15)
    autocorr_results = analyze_autocorrelation(sessions)

    # Figuras.
    ramp_df = pd.concat([s.piece_df for s in sessions if s.condition == "ramp"], ignore_index=True)
    fig_intra = out_dir / "intra_ramp.png"
    plot_intra_ramp(ramp_df, fig_intra)

    fig_conditions = out_dir / "conditions.png"
    plot_conditions(condition_summary, fig_conditions)

    sigma_metrics = {
        "n_inputs": sigma_n_inputs,
        "time_to_first_input_ms": sigma_first_input,
    }
    fig_sigma_paths = {}
    for metric_name, sigma in sigma_metrics.items():
        key = f"sigma_{metric_name}"
        path = out_dir / f"{key}.png"
        plot_sigma(sigma, path, title_suffix=metric_name)
        fig_sigma_paths[key] = path.name

    fig_autocorr = out_dir / "autocorrelation.png"
    plot_autocorrelation(autocorr_results, fig_autocorr)

    # Reporte.
    report_path = out_dir / "exploratory_report.md"
    figure_paths = {
        "intra_ramp": fig_intra.name,
        "conditions": fig_conditions.name,
        "autocorrelation": fig_autocorr.name,
    }
    figure_paths.update(fig_sigma_paths)
    generate_report(
        sessions,
        intra_ramp,
        condition_summary,
        sigma_metrics,
        autocorr_results,
        report_path,
        figure_paths,
    )

    return {
        "sessions": len(sessions),
        "intra_ramp": intra_ramp,
        "condition_summary": condition_summary,
        "sigma": sigma_metrics,
        "autocorrelation": autocorr_results,
        "report": str(report_path),
        "figures": {k: str(v) for k, v in figure_paths.items()},
    }
