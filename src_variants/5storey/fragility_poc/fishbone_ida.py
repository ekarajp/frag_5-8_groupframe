"""IDA capacity and fragility utilities for Fishbone-v2 results."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

FISHBONE_IDA_SCHEMA_VERSION = (
    "ifb3-ida-v2-success-physical-or-cp-endpoint-upper-log-interpolation"
)


def _valid_lower(result: dict[str, Any], threshold: float) -> bool:
    return (
        str(result.get("status")) == "success"
        and float(result.get("max_midr", math.nan)) < threshold
    )


def _valid_upper(result: dict[str, Any], threshold: float) -> bool:
    status = str(result.get("status"))
    return (
        status in {"success", "dynamic_instability", "cp_endpoint_reached"}
        and float(result.get("max_midr", math.nan)) >= threshold
        and not bool(result.get("numerical_failure", False))
    )


def estimate_limit_state_capacities(
    results: Iterable[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    bracket_tolerance: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Estimate IO/LS/CP only from a genuine response bracket.

    A generic numerical failure is never treated as a capacity crossing.
    A directly observed physical instability or an intentional CP first-
    passage endpoint is a valid upper point.  Neither is inferred from a
    generic numerical failure.
    """

    if not 0.0 < bracket_tolerance < 1.0:
        raise ValueError("Fishbone IDA bracket tolerance must lie in (0, 1)")
    rows = sorted(
        (
            dict(row)
            for row in results
            if math.isfinite(float(row.get("target_im_g", math.nan)))
            and float(row["target_im_g"]) > 0.0
        ),
        key=lambda row: float(row["target_im_g"]),
    )
    capacities: dict[str, dict[str, Any]] = {}
    for name, threshold_value in thresholds.items():
        threshold = float(threshold_value)
        if threshold <= 0.0:
            raise ValueError("Fishbone limit-state drifts must be positive")
        lower = None
        upper = None
        for row in rows:
            if _valid_lower(row, threshold):
                lower = row
            if (
                lower is not None
                and float(row["target_im_g"])
                > float(lower["target_im_g"])
                and _valid_upper(row, threshold)
            ):
                upper = row
                break
        if lower is None or upper is None:
            capacities[name] = {
                "status": "unbracketed",
                "threshold_midr": threshold,
                "capacity_im_g": None,
                "certified": False,
            }
            continue
        lower_im = float(lower["target_im_g"])
        upper_im = float(upper["target_im_g"])
        lower_response = max(float(lower["max_midr"]), 1.0e-12)
        upper_response = max(float(upper["max_midr"]), threshold)
        if math.isclose(
            lower_response, upper_response, rel_tol=0.0, abs_tol=1.0e-14
        ):
            capacity = math.sqrt(lower_im * upper_im)
        else:
            fraction = (
                math.log(threshold) - math.log(lower_response)
            ) / (
                math.log(upper_response) - math.log(lower_response)
            )
            fraction = min(max(fraction, 0.0), 1.0)
            capacity = math.exp(
                math.log(lower_im)
                + fraction * (math.log(upper_im) - math.log(lower_im))
            )
        relative_width = upper_im / lower_im - 1.0
        capacities[name] = {
            "status": "bracketed",
            "threshold_midr": threshold,
            "capacity_im_g": float(capacity),
            "lower_im_g": lower_im,
            "upper_im_g": upper_im,
            "lower_midr": lower_response,
            "upper_midr": upper_response,
            "relative_bracket_width": relative_width,
            "certified": bool(relative_width <= bracket_tolerance),
            "upper_status": str(upper["status"]),
        }
    return capacities


def fit_lognormal_fragility(
    capacities_im_g: Iterable[float],
) -> dict[str, float | int]:
    values = np.asarray(tuple(capacities_im_g), dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("Fishbone fragility capacities must be positive")
    logarithms = np.log(values)
    return {
        "sample_count": int(values.size),
        "theta_g": float(np.exp(np.mean(logarithms))),
        "beta": float(
            np.std(logarithms, ddof=1) if values.size > 1 else 0.0
        ),
    }
