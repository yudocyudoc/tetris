"""
recalc_power_calibrated.py — Recalcula p_min usando la curva beta_señal(p) calibrada.

Lee bin_results.json (del analisis de potencia lineal) y fast_calibration_v2.json
(con la forma calibrada), y emite una tabla comparativa: lineal vs calibrada.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_curve_t2 import (
    PIECES_PER_SESSION_RANGE,
    SESSIONS_PER_CAMPAIGN_RANGE,
    conditional_logit_info_per_dec,
    effective_separation,
    min_detectable_tracker_prob,
)


def beta_calibrated(p: float, beta_05: float, b_shape: float) -> float:
    return beta_05 * ((p / 0.5) ** b_shape)


def recalc_p_min_for_bin(
    br: Dict,
    cal: Dict,
    n_dec_available: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Optional[float]:
    """Resuelve p tal que beta_calibrated(p) - CI_high_piso = delta_min."""
    beta_05 = cal["beta_real_05"]
    b_shape = cal["b_shape"]
    ci_high_floor = br["no_tracker_oracle"]["ci_high"]
    std = br["intra_decision"]["p_grad_excess"]["mean_intra_std"]
    frac_zero = br["intra_decision"]["p_grad_excess"]["frac_zero_std"]
    k = 5

    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_beta = st.norm.ppf(power)
    info_per_dec = conditional_logit_info_per_dec(std, k)
    n_effective = n_dec_available * (1 - frac_zero)
    if n_effective <= 0 or info_per_dec <= 0:
        return None
    delta_min = np.sqrt((z_alpha + z_beta) ** 2 / (n_effective * info_per_dec))

    # beta(p) - ci_high = delta_min  =>  beta_05 * (p/0.5)^b = delta_min + ci_high
    target = delta_min + ci_high_floor
    if target <= 0 or beta_05 <= 0:
        return None
    p_min = 0.5 * ((target / beta_05) ** (1.0 / b_shape))
    return float(p_min)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_results", type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument(
        "--b_shape_override", type=float, default=None,
        help="Fuerza b_shape para todos los bins (overrides fast_calibration_v2.json). "
             "Usar con la curva real: b=1.22 (ajustado en β(0.1,0.25,0.5,0.75) de corridas Colab). "
             "El proxy fast_calibrate_signal_v2 estimó b≈0.9–1.0, que estaba equivocado en dirección."
    )
    args = parser.parse_args()

    with open(args.bin_results, encoding="utf-8") as f:
        bin_results = json.load(f)
    with open(args.calibration, encoding="utf-8") as f:
        calibration = json.load(f)

    if args.b_shape_override is not None:
        print(f"NOTA: usando b_shape_override={args.b_shape_override:.3f} para todos los bins.")
        print("      (proxy fast_calibrate estimo b~0.9-1.0; real multi-p Colab b~1.22)")
    print("p_min por bin y escenario: LINEAL vs CALIBRADO")
    print("=" * 80)
    print(f"{'Bin':<8} {'Escenario':<20} {'p_min lineal':<15} {'p_min calibrado':<15} {'forma b':<10}")
    print("-" * 80)

    for label, br in bin_results.items():
        if "error" in br or label not in calibration:
            continue
        cal = calibration[label]
        b_shape = args.b_shape_override if args.b_shape_override is not None else cal["b_shape"]
        # Parcheamos el cal temporalmente para que recalc_p_min_for_bin use el b correcto
        cal_effective = dict(cal, b_shape=b_shape)
        beta_05 = br["tracker_bruto"]["beta"]

        # Escenarios: campaña (min y max)
        scenarios = []
        for n_sessions in SESSIONS_PER_CAMPAIGN_RANGE:
            for pieces_per_session in PIECES_PER_SESSION_RANGE:
                total_pieces = n_sessions * pieces_per_session
                n_dec = total_pieces / br["pieces_per_game"] * br["decisions_per_game"]
                # lineal
                p_min_lin = min_detectable_tracker_prob(
                    n_dec, beta_05, 0.5,
                    br["no_tracker_oracle"]["ci_high"],
                    br["intra_decision"]["p_grad_excess"]["mean_intra_std"],
                    5, args.alpha, args.power,
                    br["intra_decision"]["p_grad_excess"]["frac_zero_std"],
                )
                # calibrado
                p_min_cal = recalc_p_min_for_bin(br, cal_effective, n_dec, args.alpha, args.power)
                scenarios.append((n_sessions, pieces_per_session, p_min_lin, p_min_cal))

        p_min_lin_range = (min(s[2] for s in scenarios if s[2] is not None),
                           max(s[2] for s in scenarios if s[2] is not None))
        p_min_cal_range = (min(s[3] for s in scenarios if s[3] is not None),
                           max(s[3] for s in scenarios if s[3] is not None))

        def fmt_range(lo, hi):
            lo_str = f"{lo:.2f}" if lo < 1 else ">1"
            hi_str = f"{hi:.2f}" if hi < 1 else ">1"
            return f"{lo_str}-{hi_str}"

        b_label = f"{b_shape:.2f}" + ("*" if args.b_shape_override is not None else "")
        print(f"{label:<8} {'campaña':<20} {fmt_range(*p_min_lin_range):<15} {fmt_range(*p_min_cal_range):<15} {b_label:<10}")

    print("\nNota: p_min calibrado usa beta(p) = beta(0.5) * (p/0.5)^b_shape.")
    print("b<1 indica curva concava (mas señal a p bajo de lo que predice lineal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
