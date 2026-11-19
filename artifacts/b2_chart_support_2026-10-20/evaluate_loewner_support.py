"""Fit on sample A, validate on sample B: the Loewner-interval chart support (review B2).

    PYTHONPATH=src:scripts python artifacts/b2_chart_support_2026-10-20/evaluate_loewner_support.py \
        FIT.json VALIDATION.json OUTPUT.json

Support per cover (conventional atom count N, reference G0(N) at 20 A^3/atom):

* lower bound, from physics rather than a fit:
  l(N) = s_min^2 / lambda_min(G0(N)) with s_min = 0.75 A (the declared
  shortest-vector support), so that sigma_min(L) >= s_min exactly;
* upper bound, fitted on sample A only, by the rule fixed before seeing
  sample B: M = ceil(2 (max lambda_max(log W) + 0.25)) / 2, u = e^M.

A record is inside if every eigenvalue of log W = log(G0^-1/2 G G0^-1/2) lies
in (ln l(N), M).  Records present in both samples are excluded from the
validation score.  The production ball ||log W||_F < 2 is reported alongside
for comparison.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import b2_chart_support_study as b2
from crystalatlas.full_catalogue import build_full_catalogue
from crystalatlas.lattice import spd_matrix_log

S_MIN = 0.75
FAMILIES = [name for name, _, _ in b2.FAMILIES]


def spectra(study: dict, settings: dict) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    for attempt in study["attempts"]:
        if "metric" not in attempt:
            continue
        chart = b2.chart_for(settings[attempt["setting"]], attempt["conventional_atoms"])
        whitened = chart.inverse_sqrt_reference @ np.asarray(attempt["metric"]) @ (
            chart.inverse_sqrt_reference
        )
        eigenvalues = np.linalg.eigvalsh(spd_matrix_log((whitened + whitened.T) / 2.0))
        rows.append(
            {
                "material_id": attempt["material_id"],
                "family": attempt["family"],
                "log_lower": float(eigenvalues[0]),
                "log_upper": float(eigenvalues[-1]),
                "log_floor": float(
                    math.log(S_MIN**2 / np.linalg.eigvalsh(chart.reference_metric)[0])
                ),
                "ball_norm": attempt["coefficient_norm"],
            }
        )
    return rows


def coverage(rows: list[dict], upper: float) -> dict:  # type: ignore[type-arg]
    result = {}
    for family in FAMILIES + ["all"]:
        members = rows if family == "all" else [r for r in rows if r["family"] == family]
        inside = [r["log_lower"] > r["log_floor"] and r["log_upper"] < upper for r in members]
        result[family] = {
            "n": len(members),
            "loewner_inside": int(sum(inside)),
            "loewner_rate": round(sum(inside) / len(members), 4) if members else None,
            "ball_inside": int(sum(r["ball_norm"] < 2.0 for r in members)),
            "min_floor_slack": round(min(r["log_lower"] - r["log_floor"] for r in members), 3)
            if members
            else None,
            "max_log_upper": round(max(r["log_upper"] for r in members), 3) if members else None,
        }
    return result


def main() -> None:
    fit_path, validation_path, output_path = map(Path, sys.argv[1:4])
    catalogue = build_full_catalogue()
    settings = {setting.key: setting for setting in catalogue.settings.values()}
    fit = spectra(json.loads(fit_path.read_text()), settings)
    validation = spectra(json.loads(validation_path.read_text()), settings)
    fitted_ids = {row["material_id"] for row in fit}
    held_out = [row for row in validation if row["material_id"] not in fitted_ids]
    upper = math.ceil(2.0 * (max(row["log_upper"] for row in fit) + 0.25)) / 2.0
    report = {
        "rule": "M = ceil(2 (max_A lambda_max(log W) + 0.25)) / 2; l(N) = s_min^2/lambda_min(G0(N))",
        "s_min_angstrom": S_MIN,
        "fitted_M": upper,
        "fit_records": len(fit),
        "validation_records": len(validation),
        "validation_overlap_excluded": len(validation) - len(held_out),
        "fit_coverage": coverage(fit, upper),
        "held_out_coverage": coverage(held_out, upper),
    }
    output_path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("fitted_M", "validation_overlap_excluded")}))
    for name in ("fit_coverage", "held_out_coverage"):
        print(name)
        for family, value in report[name].items():
            print(f"  {family:12s} {value}")


if __name__ == "__main__":
    main()
