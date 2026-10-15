"""Bidirectional NLTHA for the independent Fishbone-v2 reduced model.

This module intentionally has its own schema and cache namespace.  It does
not alter or reuse an MDOF-v1 result as a Fishbone result.  The solver accepts
the two components of a physical ground-motion pair simultaneously and
returns floor/story histories in the same engineering units used by the
Full-3D pipeline.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G_STD
from .fishbone import (
    FishboneArtifact,
    FishboneModelInfo,
    build_model,
    modal_properties,
    run_gravity,
)
from .ground_motion import load_acceleration, pair_sa_geomean_g
from .io_utils import atomic_write_json, read_json, stable_hash
from .structural import _ops

FISHBONE_NLTHA_SCHEMA_VERSION = (
    "bidirectional-ifb3-nltha-v2-modal-damping-parity-plain-mp"
)


def _rayleigh_coefficients(
    omega_low: float,
    omega_high: float,
    damping_ratio: float,
) -> tuple[float, float]:
    if not (
        math.isfinite(omega_low)
        and math.isfinite(omega_high)
        and 0.0 < omega_low <= omega_high
    ):
        raise ValueError("Fishbone Rayleigh frequencies are invalid")
    if not 0.0 <= damping_ratio < 1.0:
        raise ValueError("Fishbone damping ratio must lie in [0, 1)")
    if damping_ratio == 0.0:
        return 0.0, 0.0
    alpha_m = (
        2.0
        * damping_ratio
        * omega_low
        * omega_high
        / (omega_low + omega_high)
    )
    beta_k_comm = 2.0 * damping_ratio / (omega_low + omega_high)
    return float(alpha_m), float(beta_k_comm)


def _apply_modal_damping(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    damping_ratio: float,
    damping_model: str,
) -> dict[str, Any]:
    modal = modal_properties(artifact, info)
    normalized_model = str(damping_model).lower()
    ops = _ops()
    if normalized_model == "modal":
        # Match the Full-3D production model exactly: assign the same modal
        # damping ratio to every extracted mode after eigen analysis.  This
        # avoids the response bias that a degrading-stiffness Rayleigh model
        # can introduce near collapse.
        ops.modalDamping(
            *[float(damping_ratio) for _ in modal.periods_s]
        )
        return {
            "damping_ratio": float(damping_ratio),
            "method": "modal",
            "mode_count": float(len(modal.periods_s)),
            "lower_period_s": float(max(modal.periods_s)),
            "upper_period_s": float(min(modal.periods_s)),
            "alpha_m": 0.0,
            "beta_k_comm": 0.0,
        }
    if normalized_model != "rayleigh_committed":
        raise ValueError(
            "Fishbone damping model must be modal or rayleigh_committed"
        )
    periods = list(modal.directional_periods_x_s)
    periods.extend(modal.directional_periods_y_s)
    periods = sorted(
        {
            round(float(value), 12)
            for value in periods
            if math.isfinite(float(value)) and float(value) > 0.0
        },
        reverse=True,
    )
    if not periods:
        raise RuntimeError("Fishbone has no modes for damping calibration")
    low_period = periods[0]
    high_period = periods[min(2, len(periods) - 1)]
    omega_low = 2.0 * math.pi / low_period
    omega_high = 2.0 * math.pi / high_period
    if omega_high < omega_low:
        omega_low, omega_high = omega_high, omega_low
    alpha_m, beta_k_comm = _rayleigh_coefficients(
        omega_low, omega_high, damping_ratio
    )
    ops.rayleigh(alpha_m, 0.0, 0.0, beta_k_comm)
    return {
        "damping_ratio": float(damping_ratio),
        "method": "rayleigh_committed",
        "mode_count": float(len(modal.periods_s)),
        "lower_period_s": float(low_period),
        "upper_period_s": float(high_period),
        "alpha_m": alpha_m,
        "beta_k_comm": beta_k_comm,
    }


def _set_algorithm(algorithm: str) -> None:
    ops = _ops()
    if algorithm == "NewtonLineSearch":
        ops.algorithm("NewtonLineSearch", 0.8)
    elif algorithm == "KrylovNewton":
        ops.algorithm("KrylovNewton")
    elif algorithm == "ModifiedNewton":
        ops.algorithm("ModifiedNewton", "-initial")
    else:
        ops.algorithm("Newton")


def _configure_transient(
    *,
    algorithm: str = "Newton",
    test_type: str = "EnergyIncr",
    tolerance: float = 1.0e-7,
    maximum_iterations: int = 40,
) -> None:
    ops = _ops()
    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test(test_type, tolerance, maximum_iterations, 0)
    _set_algorithm(algorithm)
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")


def _story_hinge_rotations(
    info: FishboneModelInfo,
    story_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return column-X, column-Y, beam-X and beam-Y maxima by story."""

    ops = _ops()
    column_x = np.zeros(story_count, dtype=float)
    column_y = np.zeros(story_count, dtype=float)
    beam_x = np.zeros(story_count, dtype=float)
    beam_y = np.zeros(story_count, dtype=float)
    for index, tag in enumerate(info.column_hinge_elements):
        values = np.asarray(
            ops.eleResponse(tag, "deformation"), dtype=float
        ).reshape(-1)
        if values.size >= 2:
            story = (index % (2 * story_count)) // 2
            column_y[story] = max(column_y[story], abs(float(values[0])))
            column_x[story] = max(column_x[story], abs(float(values[1])))
    for index, tag in enumerate(info.beam_hinge_elements):
        values = np.asarray(
            ops.eleResponse(tag, "deformation"), dtype=float
        ).reshape(-1)
        if not values.size:
            continue
        local_index = index % (4 * story_count)
        story = local_index // 4
        arm = local_index % 4
        if arm < 2:
            beam_x[story] = max(beam_x[story], abs(float(values[0])))
        else:
            beam_y[story] = max(beam_y[story], abs(float(values[0])))
    return column_x, column_y, beam_x, beam_y


def _state(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    time_s: float,
    ground_x_m_s2: float,
    ground_y_m_s2: float,
) -> dict[str, np.ndarray | float]:
    ops = _ops()
    displacement_x = np.asarray(
        [ops.nodeDisp(node, 1) for node in info.floor_nodes], dtype=float
    )
    displacement_y = np.asarray(
        [ops.nodeDisp(node, 2) for node in info.floor_nodes], dtype=float
    )
    deformation_x = np.diff(
        np.concatenate(([0.0], displacement_x))
    )
    deformation_y = np.diff(
        np.concatenate(([0.0], displacement_y))
    )
    heights = np.asarray(artifact.story_heights_m, dtype=float)
    relative_acceleration_x = np.asarray(
        [ops.nodeAccel(node, 1) for node in info.floor_nodes], dtype=float
    )
    relative_acceleration_y = np.asarray(
        [ops.nodeAccel(node, 2) for node in info.floor_nodes], dtype=float
    )
    masses_x = np.asarray(
        artifact.floor_masses_x_kn_s2_m, dtype=float
    )
    masses_y = np.asarray(
        artifact.floor_masses_y_kn_s2_m, dtype=float
    )
    absolute_acceleration_x = relative_acceleration_x + ground_x_m_s2
    absolute_acceleration_y = relative_acceleration_y + ground_y_m_s2
    column_x, column_y, beam_x, beam_y = _story_hinge_rotations(
        info, artifact.story_count
    )
    return {
        "time_s": float(time_s),
        "floor_displacement_x_m": displacement_x,
        "floor_displacement_y_m": displacement_y,
        "story_deformation_x_m": deformation_x,
        "story_deformation_y_m": deformation_y,
        "story_drift_ratio_x": deformation_x / heights,
        "story_drift_ratio_y": deformation_y / heights,
        "floor_absolute_acceleration_x_m_s2": absolute_acceleration_x,
        "floor_absolute_acceleration_y_m_s2": absolute_acceleration_y,
        "base_shear_x_kn": float(
            np.sum(masses_x * absolute_acceleration_x)
        ),
        "base_shear_y_kn": float(
            np.sum(masses_y * absolute_acceleration_y)
        ),
        "column_hinge_rotation_x_rad": column_x,
        "column_hinge_rotation_y_rad": column_y,
        "beam_hinge_rotation_x_rad": beam_x,
        "beam_hinge_rotation_y_rad": beam_y,
    }


def _append_state(
    histories: dict[str, list[Any]],
    state: dict[str, np.ndarray | float],
) -> None:
    for key, value in state.items():
        histories.setdefault(key, []).append(value)


def _finalize_histories(
    histories: dict[str, list[Any]],
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(values, dtype=np.float64)
        for key, values in histories.items()
    }


def _maximum_hinge_utilization(
    artifact: FishboneArtifact,
    arrays: dict[str, np.ndarray],
) -> tuple[float, str | None]:
    maximum = 0.0
    label: str | None = None
    for direction in ("x", "y"):
        column = arrays[f"column_hinge_rotation_{direction}_rad"]
        beam = arrays[f"beam_hinge_rotation_{direction}_rad"]
        for index, story in enumerate(artifact.stories):
            column_hinges = (
                story.column_bottom_hinge_x,
                story.column_top_hinge_x,
            ) if direction == "x" else (
                story.column_bottom_hinge_y,
                story.column_top_hinge_y,
            )
            column_limit = min(
                min(
                    hinge.ultimate_rotation_pos_rad,
                    hinge.ultimate_rotation_neg_rad,
                )
                for hinge in column_hinges
            )
            beam_hinge = (
                story.beam_hinge_x
                if direction == "x"
                else story.beam_hinge_y
            )
            beam_limit = min(
                beam_hinge.ultimate_rotation_pos_rad,
                beam_hinge.ultimate_rotation_neg_rad,
            )
            column_ratio = float(
                np.max(column[:, index]) / max(column_limit, 1.0e-12)
            )
            beam_ratio = float(
                np.max(beam[:, index]) / max(beam_limit, 1.0e-12)
            )
            for ratio, kind in (
                (column_ratio, "column"),
                (beam_ratio, "beam"),
            ):
                if ratio > maximum:
                    maximum = ratio
                    label = (
                        f"{kind}_hinge_ultimate_{direction}_story_{index + 1}"
                    )
    return maximum, label


def run_bidirectional_nltha(
    artifact: FishboneArtifact,
    *,
    acceleration_x_g: np.ndarray,
    acceleration_y_g: np.ndarray,
    record_dt_s: float,
    scale_factor: float = 1.0,
    damping_ratio: float = 0.05,
    damping_model: str = "modal",
    dynamic_instability_drift: float = 0.10,
    maximum_runtime_s: float = 300.0,
    minimum_substep_factor: int = 16,
) -> dict[str, Any]:
    """Run simultaneous X-Y NLTHA and return summary plus full histories."""

    artifact.validate()
    x_g = np.asarray(acceleration_x_g, dtype=float).reshape(-1)
    y_g = np.asarray(acceleration_y_g, dtype=float).reshape(-1)
    if x_g.size < 2 or y_g.shape != x_g.shape:
        raise ValueError("Fishbone X/Y acceleration arrays must match")
    if (
        np.any(~np.isfinite(x_g))
        or np.any(~np.isfinite(y_g))
        or not math.isfinite(record_dt_s)
        or record_dt_s <= 0.0
        or not math.isfinite(scale_factor)
        or scale_factor <= 0.0
    ):
        raise ValueError("Fishbone NLTHA input is invalid")
    if dynamic_instability_drift <= 0.0 or maximum_runtime_s <= 0.0:
        raise ValueError("Fishbone NLTHA guards must be positive")
    if minimum_substep_factor < 1:
        raise ValueError("Fishbone minimum substep factor is invalid")

    started = time.perf_counter()
    info = build_model(artifact)
    run_gravity()
    damping = _apply_modal_damping(
        artifact, info, damping_ratio, damping_model
    )
    ops = _ops()
    acceleration_x = x_g * G_STD * float(scale_factor)
    acceleration_y = y_g * G_STD * float(scale_factor)
    ops.timeSeries(
        "Path", 201, "-dt", record_dt_s, "-values", *acceleration_x
    )
    ops.timeSeries(
        "Path", 202, "-dt", record_dt_s, "-values", *acceleration_y
    )
    ops.pattern("UniformExcitation", 201, 1, "-accel", 201)
    ops.pattern("UniformExcitation", 202, 2, "-accel", 202)
    _configure_transient()

    time_grid = np.arange(x_g.size, dtype=float) * record_dt_s
    duration = float(time_grid[-1])
    histories: dict[str, list[Any]] = {}
    _append_state(
        histories,
        _state(
            artifact,
            info,
            0.0,
            float(acceleration_x[0]),
            float(acceleration_y[0]),
        ),
    )
    recovery_log: list[dict[str, Any]] = []
    numerical_failure = False
    physical_collapse = False
    collapse_basis: list[str] = []
    tolerance_time = max(1.0e-12, record_dt_s * 1.0e-8)
    recovery_plan = (
        ("Newton", "EnergyIncr", 1.0e-7, 40),
        ("NewtonLineSearch", "EnergyIncr", 1.0e-7, 60),
        ("KrylovNewton", "EnergyIncr", 1.0e-7, 80),
        ("ModifiedNewton", "EnergyIncr", 1.0e-6, 120),
        ("KrylovNewton", "NormDispIncr", 1.0e-6, 120),
    )
    while float(ops.getTime()) < duration - tolerance_time:
        if time.perf_counter() - started > maximum_runtime_s:
            numerical_failure = True
            collapse_basis.append("runtime_guard")
            break
        committed_time = float(ops.getTime())
        target_step = min(record_dt_s, duration - committed_time)
        solved = False
        accepted_dt = target_step
        for substep_factor in (
            factor
            for factor in (1, 2, 4, 8, 16, 32, 64)
            if factor <= max(1, minimum_substep_factor)
        ):
            trial_dt = target_step / substep_factor
            for (
                algorithm,
                test_type,
                tolerance,
                iterations,
            ) in recovery_plan:
                _configure_transient(
                    algorithm=algorithm,
                    test_type=test_type,
                    tolerance=tolerance,
                    maximum_iterations=iterations,
                )
                code = int(ops.analyze(1, trial_dt))
                if (
                    code != 0
                    or substep_factor > 1
                    or algorithm != "Newton"
                ):
                    recovery_log.append(
                        {
                            "time_s": committed_time,
                            "substep_factor": int(substep_factor),
                            "algorithm": algorithm,
                            "test_type": test_type,
                            "tolerance": tolerance,
                            "maximum_iterations": iterations,
                            "code": code,
                        }
                    )
                if code == 0:
                    accepted_dt = trial_dt
                    solved = True
                    break
                # Discard failed trial vectors before changing algorithm.
                _configure_transient()
            if solved:
                break
        if not solved:
            numerical_failure = True
            collapse_basis.append("solver_nonconvergence")
            break

        current_time = float(ops.getTime())
        ground_x = float(np.interp(current_time, time_grid, acceleration_x))
        ground_y = float(np.interp(current_time, time_grid, acceleration_y))
        state = _state(
            artifact, info, current_time, ground_x, ground_y
        )
        _append_state(histories, state)
        drift_x = np.asarray(state["story_drift_ratio_x"], dtype=float)
        drift_y = np.asarray(state["story_drift_ratio_y"], dtype=float)
        current_midr = max(
            float(np.max(np.abs(drift_x))),
            float(np.max(np.abs(drift_y))),
        )
        if current_midr >= dynamic_instability_drift:
            physical_collapse = True
            collapse_basis.append("dynamic_instability_drift")
            break
        # A reduced step is accepted physically.  The next loop attempts the
        # normal record step again; this keeps recovery local in time.
        _ = accepted_dt

    arrays = _finalize_histories(histories)
    absolute_x = np.abs(arrays["story_drift_ratio_x"])
    absolute_y = np.abs(arrays["story_drift_ratio_y"])
    peak_x_index = np.unravel_index(
        int(np.argmax(absolute_x)), absolute_x.shape
    )
    peak_y_index = np.unravel_index(
        int(np.argmax(absolute_y)), absolute_y.shape
    )
    max_x = float(absolute_x[peak_x_index])
    max_y = float(absolute_y[peak_y_index])
    if max_x >= max_y:
        maximum_midr = max_x
        critical_direction = "X"
        critical_index = peak_x_index
    else:
        maximum_midr = max_y
        critical_direction = "Y"
        critical_index = peak_y_index
    hinge_utilization, hinge_basis = _maximum_hinge_utilization(
        artifact, arrays
    )
    if hinge_utilization >= 1.0:
        physical_collapse = True
        if hinge_basis and hinge_basis not in collapse_basis:
            collapse_basis.append(hinge_basis)
    completed_fraction = min(
        max(float(arrays["time_s"][-1]) / max(duration, 1.0e-12), 0.0),
        1.0,
    )
    if physical_collapse:
        status = "dynamic_instability"
    elif numerical_failure or completed_fraction < 1.0 - 1.0e-8:
        status = "numerical_failure"
    else:
        status = "success"
    recovery_failures = sum(
        int(row["code"] != 0) for row in recovery_log
    )
    return {
        "schema_version": FISHBONE_NLTHA_SCHEMA_VERSION,
        "status": status,
        "analysis_engine": "fishbone_v2",
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "calibration_stage": artifact.calibration_stage,
        "certified": bool(artifact.certified),
        "maximum_midr": maximum_midr,
        "max_midr": maximum_midr,
        "maximum_midr_x": max_x,
        "maximum_midr_y": max_y,
        "critical_direction": critical_direction,
        "critical_story": int(critical_index[1] + 1),
        "critical_time_s": float(arrays["time_s"][critical_index[0]]),
        "maximum_roof_displacement_x_m": float(
            np.max(np.abs(arrays["floor_displacement_x_m"][:, -1]))
        ),
        "maximum_roof_displacement_y_m": float(
            np.max(np.abs(arrays["floor_displacement_y_m"][:, -1]))
        ),
        "maximum_base_shear_x_kn": float(
            np.max(np.abs(arrays["base_shear_x_kn"]))
        ),
        "maximum_base_shear_y_kn": float(
            np.max(np.abs(arrays["base_shear_y_kn"]))
        ),
        "maximum_hinge_utilization": hinge_utilization,
        "collapse_basis": tuple(collapse_basis),
        "physical_collapse": physical_collapse,
        "numerical_failure": numerical_failure,
        "completed_fraction": completed_fraction,
        "termination_time_s": float(arrays["time_s"][-1]),
        "record_duration_s": duration,
        "recovery_attempt_count": len(recovery_log),
        "recovery_failure_count": recovery_failures,
        "damping": damping,
        "runtime_s_solver": float(time.perf_counter() - started),
        "_history_arrays": arrays,
        "_recovery_log": recovery_log,
    }


def run_pair_nltha(
    artifact: FishboneArtifact,
    pair: dict[str, Any],
    target_im_g: float,
    *,
    damping_ratio: float = 0.05,
    damping_model: str = "modal",
    dynamic_instability_drift: float = 0.10,
    maximum_runtime_s: float = 300.0,
) -> dict[str, Any]:
    """Scale one physical X-Y pair by geometric-mean Sa(T1)."""

    time_x, acceleration_x_g = load_acceleration(
        pair["component_x_path"], pair["units"]
    )
    time_y, acceleration_y_g = load_acceleration(
        pair["component_y_path"], pair["units"]
    )
    if not np.allclose(time_x, time_y, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: X/Y time vectors differ")
    signature_data = pair_case_signature(
        artifact,
        pair,
        target_im_g,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=dynamic_instability_drift,
    )
    reference_period = float(signature_data["reference_period_s"])
    unscaled_sa = float(signature_data["unscaled_sa_g"])
    scale_factor = float(signature_data["scale_factor"])
    result = run_bidirectional_nltha(
        artifact,
        acceleration_x_g=acceleration_x_g,
        acceleration_y_g=acceleration_y_g,
        record_dt_s=float(pair["dt_s"]),
        scale_factor=scale_factor,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=dynamic_instability_drift,
        maximum_runtime_s=maximum_runtime_s,
    )
    result.update(
        {
            "pair_id": str(pair["pair_id"]),
            "target_im_g": float(target_im_g),
            "reference_period_s": reference_period,
            "unscaled_sa_g": unscaled_sa,
            "scale_factor": scale_factor,
            "achieved_im_g": unscaled_sa * scale_factor,
            "case_signature": str(signature_data["case_signature"]),
        }
    )
    return result


def pair_case_signature(
    artifact: FishboneArtifact,
    pair: dict[str, Any],
    target_im_g: float,
    *,
    damping_ratio: float = 0.05,
    damping_model: str = "modal",
    dynamic_instability_drift: float = 0.10,
) -> dict[str, float | str]:
    """Return immutable scaling and signature data before an analysis."""

    period_x = float(artifact.target_modal_x.periods_s[0])
    period_y = float(artifact.target_modal_y.periods_s[0])
    reference_period = math.sqrt(period_x * period_y)
    unscaled_sa = float(pair_sa_geomean_g(pair, reference_period))
    if (
        not math.isfinite(target_im_g)
        or target_im_g <= 0.0
        or not math.isfinite(unscaled_sa)
        or unscaled_sa <= 0.0
    ):
        raise ValueError("Fishbone target or unscaled Sa is invalid")
    scale_factor = float(target_im_g / unscaled_sa)
    signature = stable_hash(
        {
            "schema_version": FISHBONE_NLTHA_SCHEMA_VERSION,
            "artifact_hash": artifact.artifact_hash,
            "source_model_hash": artifact.source_model_hash,
            "pair_id": str(pair["pair_id"]),
            "component_x_path": str(pair["component_x_path"]),
            "component_y_path": str(pair["component_y_path"]),
            "sha256_x": str(pair.get("sha256_x") or ""),
            "sha256_y": str(pair.get("sha256_y") or ""),
            "dt_s": float(pair["dt_s"]),
            "npts": int(pair.get("npts") or 0),
            "target_im_g": float(target_im_g),
            "scale_factor": scale_factor,
            "damping_ratio": float(damping_ratio),
            "damping_model": str(damping_model),
            "dynamic_instability_drift": float(
                dynamic_instability_drift
            ),
        }
    )
    return {
        "reference_period_s": reference_period,
        "unscaled_sa_g": unscaled_sa,
        "scale_factor": scale_factor,
        "case_signature": signature,
    }


def history_path(
    run_dir: str | Path,
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> Path:
    token = f"{target_im_g:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        Path(run_dir)
        / "fishbone_v2"
        / "histories"
        / building_id
        / pair_id
        / f"im_{token}.npz"
    )


def result_path(
    run_dir: str | Path,
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> Path:
    return history_path(
        run_dir, building_id, pair_id, target_im_g
    ).with_suffix(".json")


def save_pair_result(
    result: dict[str, Any],
    run_dir: str | Path,
) -> tuple[Path, Path]:
    """Atomically save one Fishbone case without touching Full-3D caches."""

    building_id = str(result["building_id"])
    pair_id = str(result["pair_id"])
    target_im_g = float(result["target_im_g"])
    history = history_path(
        run_dir, building_id, pair_id, target_im_g
    )
    metadata = result_path(
        run_dir, building_id, pair_id, target_im_g
    )
    arrays = result.get("_history_arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Fishbone result lacks history arrays")
    atomic_write_history(history, arrays)
    payload = {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }
    payload["history_path"] = str(history.absolute())
    atomic_write_json(metadata, payload)
    return metadata, history


def load_cached_pair_result(
    run_dir: str | Path,
    building_id: str,
    pair_id: str,
    target_im_g: float,
    *,
    expected_signature: str | None = None,
) -> dict[str, Any] | None:
    metadata = result_path(
        run_dir, building_id, pair_id, target_im_g
    )
    history = history_path(
        run_dir, building_id, pair_id, target_im_g
    )
    if not metadata.is_file() or not history.is_file():
        return None
    payload = read_json(metadata)
    if payload.get("schema_version") != FISHBONE_NLTHA_SCHEMA_VERSION:
        return None
    if (
        expected_signature is not None
        and payload.get("case_signature") != expected_signature
    ):
        return None
    if str(payload.get("history_path")) != str(history.absolute()):
        return None
    return payload


def atomic_write_history(
    path: str | Path,
    arrays: dict[str, np.ndarray],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.npz",
        dir=target.parent,
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary_name,
            **{
                key: np.asarray(value, dtype=np.float64)
                for key, value in arrays.items()
            },
        )
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
