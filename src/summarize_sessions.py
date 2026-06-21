"""Consolidado de resumen por sesión.

Lee todas las sesiones en `data/` y genera un resumen agregado por sesión
en JSON y CSV. Útil para compartir un overview con otros agentes o herramientas.

Uso:
    python -m src.summarize_sessions
    python -m src.summarize_sessions --output data/sessions_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import summarize_piece_metrics


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_session(session_dir: Path) -> Optional[Dict[str, Any]]:
    """Devuelve un diccionario con el resumen de una sesión."""
    meta_path = session_dir / "session_meta.json"
    pieces_path = session_dir / "pieces.csv"
    actions_path = session_dir / "actions.csv"
    summary_path = session_dir / "games_summary.csv"

    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Métricas de pieces.csv
    decision_times: List[float] = []
    n_inputs_list: List[float] = []
    piece_count = 0
    line_clears: List[int] = []

    if pieces_path.exists():
        with open(pieces_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = _safe_float(row.get("decision_time_ms"))
                if dt is not None:
                    decision_times.append(dt)
                ni = _safe_float(row.get("n_inputs"))
                if ni is not None:
                    n_inputs_list.append(ni)
                lc = _safe_float(row.get("lines_cleared_by_lock"))
                if lc is not None:
                    line_clears.append(int(lc))
                piece_count += 1

    # Métricas de games_summary.csv
    games = []
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8", newline="") as f:
            games = list(csv.DictReader(f))

    total_games = len(games)
    total_duration_ms = sum(
        _safe_float(g["duration_ms"]) or 0.0 for g in games
    )
    total_lines = sum(_safe_float(g["total_lines"]) or 0.0 for g in games)
    total_score = sum(_safe_float(g["score"]) or 0.0 for g in games)

    behavioral = summarize_piece_metrics(pieces_path, actions_path)

    return {
        "session_id": meta.get("session_id"),
        "wall_clock_start": meta.get("wall_clock_start"),
        "condition": meta.get("condition"),
        "software_git_hash": meta.get("software_git_hash"),
        "state_covariates": meta.get("state_covariates", {}),
        "perceived_effort_1_10": meta.get("perceived_effort_1_10"),
        "config": meta.get("config", {}),
        "total_games": total_games,
        "total_pieces": piece_count,
        "total_lines": int(total_lines),
        "total_score": int(total_score),
        "total_duration_ms": round(total_duration_ms, 2),
        "total_duration_min": round(total_duration_ms / 60000.0, 2),
        "note": "decision_time_ms crudo está contaminado por la gravedad; usar métricas conductuales para el σ",
        "behavioral_metrics": behavioral,
        "line_clears_distribution": {
            "0": line_clears.count(0),
            "1": line_clears.count(1),
            "2": line_clears.count(2),
            "3": line_clears.count(3),
            "4": line_clears.count(4),
        },
    }


def summarize_all(data_root: str = "data") -> List[Dict[str, Any]]:
    root = Path(data_root)
    if not root.exists():
        return []

    summaries = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        summary = summarize_session(session_dir)
        if summary is not None:
            summaries.append(summary)
    return summaries


def write_summaries(summaries: List[Dict[str, Any]], output_root: str = "data") -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / "sessions_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    csv_path = root / "sessions_summary.csv"
    if summaries:
        # Aplanamos covariables y algunas métricas para CSV.
        flat_rows = []
        for s in summaries:
            row = {
                "session_id": s["session_id"],
                "wall_clock_start": s["wall_clock_start"],
                "condition": s["condition"],
                "software_git_hash": s["software_git_hash"],
                "perceived_effort_1_10": s["perceived_effort_1_10"],
                "total_games": s["total_games"],
                "total_pieces": s["total_pieces"],
                "total_lines": s["total_lines"],
                "total_score": s["total_score"],
                "total_duration_min": s["total_duration_min"],
                "n_inputs_mean": s["behavioral_metrics"]["n_inputs"]["mean"],
                "n_inputs_std": s["behavioral_metrics"]["n_inputs"]["std"],
                "first_input_mean_ms": s["behavioral_metrics"]["time_to_first_input_ms"]["mean"],
                "active_time_mean_ms": s["behavioral_metrics"]["active_time_ms"]["mean"],
                "hard_drop_ratio": s["behavioral_metrics"]["hard_drop_ratio"],
            }
            cov = s.get("state_covariates", {})
            for key, value in cov.items():
                row[f"cov_{key}"] = value
            flat_rows.append(row)

        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    print(f"Consolidado guardado en:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Resumen consolidado de sesiones")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Directorio raíz donde están las sesiones",
    )
    parser.add_argument(
        "--output-root",
        default="data",
        help="Directorio donde guardar el consolidado",
    )
    args = parser.parse_args()

    summaries = summarize_all(args.data_root)
    if not summaries:
        print(f"No se encontraron sesiones en {args.data_root}/")
        sys.exit(0)

    write_summaries(summaries, args.output_root)
    print(f"\nTotal de sesiones resumidas: {len(summaries)}")


if __name__ == "__main__":
    main()
