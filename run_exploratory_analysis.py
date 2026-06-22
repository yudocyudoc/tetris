"""Punto de entrada para el analisis exploratorio.

Uso:
    python run_exploratory_analysis.py
    python run_exploratory_analysis.py --data-root data --output-root data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.exploratory_analysis import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis exploratorio piloto Tetris")
    parser.add_argument("--data-root", default="data", help="Directorio raiz con sesiones")
    parser.add_argument("--output-root", default="data", help="Directorio de salida")
    args = parser.parse_args()

    results = run_all(args.data_root, args.output_root)
    print(f"Reporte generado: {results['report']}")
    print(f"Sesiones analizadas: {results['sessions']}")


if __name__ == "__main__":
    main()
