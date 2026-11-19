"""Fit on sample A, validate on sample B: the Hencky volumetric window of the
uniform control law under the Loewner chart (review B2 / m3).

    PYTHONPATH=src:scripts python artifacts/b2_chart_support_2026-10-20/evaluate_volumetric_window.py \
        FIT.json VALIDATION.json OUTPUT.json

With W = G0^-1/2 G G0^-1/2 and G0(N) the setting's reference shape at
20 A^3 per conventional atom, the Hencky volumetric strain is
e_v = (1/2) tr log W = ln(V / V0).  Window, by the rule of the Loewner upper
bound, fixed before reading sample B:

    hi = ceil(2 (max_A e_v + 0.25)) / 2,   lo = -ceil(2 (max_A (-e_v) + 0.25)) / 2.

Records present in both samples are excluded from the validation score.  The
script also reports, per space-group type, the fraction of the uniform
Loewner-support law (and of the legacy ball law) that falls outside [lo, hi].
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import b2_chart_support_study as b2
import numpy as np

from crystalatlas.full_catalogue import build_full_catalogue
from crystalatlas.lattice import (
    LoewnerSPDChart,
    loewner_lower_bound,
    scale_metric_to_volume,
    spd_matrix_log,
)

FAMILIES = [name for name, _, _ in b2.FAMILIES]
PROBE_TYPES = (2, 12, 62, 139, 166, 194, 225)
PROBE_DRAWS = 4000


def strains(study: dict, settings: dict) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    for attempt in study["attempts"]:
        if "metric" not in attempt:
            continue
        chart = b2.chart_for(settings[attempt["setting"]], attempt["conventional_atoms"])
        whitened = chart.inverse_sqrt_reference @ np.asarray(attempt["metric"]) @ (
            chart.inverse_sqrt_reference
        )
        log_w = spd_matrix_log((whitened + whitened.T) / 2.0)
        rows.append(
            {
                "material_id": attempt["material_id"],
                "family": attempt["family"],
                "volumetric": float(np.trace(log_w) / 2.0),
            }
        )
    return rows


def coverage(rows: list[dict], lo: float, hi: float) -> dict:  # type: ignore[type-arg]
    result = {}
    for family in FAMILIES + ["all"]:
        members = rows if family == "all" else [r for r in rows if r["family"] == family]
        inside = [lo < r["volumetric"] < hi for r in members]
        result[family] = {
            "n": len(members),
            "inside": int(sum(inside)),
            "min_volumetric": round(min(r["volumetric"] for r in members), 3) if members else None,
            "max_volumetric": round(max(r["volumetric"] for r in members), 3) if members else None,
        }
    return result


def law_mass_outside(catalogue, lo: float, hi: float) -> dict:  # type: ignore[no-untyped-def,type-arg]
    rng = np.random.default_rng(20261020)
    first_setting = {}
    for setting in catalogue.settings.values():
        first_setting.setdefault(setting.international_number, setting)
    report = {}
    for number in PROBE_TYPES:
        setting = first_setting[number]
        rotations = [np.asarray(r, dtype=np.float64) for r in setting.point_group_rotations]
        reference = scale_metric_to_volume(np.asarray(setting.reference_metric), 20.0 * 8)
        chart = LoewnerSPDChart(
            rotations,
            reference_metric=reference,
            lower=loewner_lower_bound(reference, 0.75),
            upper=math.exp(4.5),
        )
        basis = np.stack(chart.basis)
        traces = np.array(
            [
                np.trace(np.einsum("i,ijk->jk", chart.uniform_coefficients(rng), basis)) / 2.0
                for _ in range(PROBE_DRAWS)
            ]
        )
        report[str(number)] = {
            "dimension": chart.dimension,
            "loewner_uniform_fraction_outside_window": round(
                float(np.mean((traces <= lo) | (traces >= hi))), 4
            ),
            "loewner_uniform_fraction_V_over_V0_above_10": round(
                float(np.mean(traces > math.log(10.0))), 4
            ),
        }
    return report


def main() -> None:
    fit_path, validation_path, output_path = map(Path, sys.argv[1:4])
    catalogue = build_full_catalogue()
    settings = {setting.key: setting for setting in catalogue.settings.values()}
    fit = strains(json.loads(fit_path.read_text()), settings)
    hi = math.ceil(2.0 * (max(r["volumetric"] for r in fit) + 0.25)) / 2.0
    lo = -math.ceil(2.0 * (max(-r["volumetric"] for r in fit) + 0.25)) / 2.0
    print(json.dumps({"fitted_lo": lo, "fitted_hi": hi, "fit_records": len(fit)}), flush=True)
    validation = strains(json.loads(validation_path.read_text()), settings)
    fitted_ids = {row["material_id"] for row in fit}
    held_out = [row for row in validation if row["material_id"] not in fitted_ids]
    report = {
        "rule": "hi = ceil(2 (max_A e_v + 0.25))/2; lo = -ceil(2 (max_A (-e_v) + 0.25))/2; "
        "e_v = tr(log W)/2 = ln(V/V0)",
        "fitted_lo": lo,
        "fitted_hi": hi,
        "fit_records": len(fit),
        "validation_records": len(validation),
        "validation_overlap_excluded": len(validation) - len(held_out),
        "fit_coverage": coverage(fit, lo, hi),
        "held_out_coverage": coverage(held_out, lo, hi),
        "uniform_law_probe_N8": law_mass_outside(catalogue, lo, hi),
    }
    output_path.write_text(json.dumps(report, indent=1) + "\n")
    for name in ("fit_coverage", "held_out_coverage", "uniform_law_probe_N8"):
        print(name)
        for key, value in report[name].items():
            print(f"  {key:12s} {value}")


if __name__ == "__main__":
    main()
