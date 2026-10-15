"""Quasi-static cyclic evidence for Fishbone-v2 hinge calibration."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .fishbone import FishboneArtifact, build_model, run_gravity
from .fishbone_spo import (
    _configure_static,
    _hinge_story_peak_rotations,
    _set_static_strategy,
    modal_pattern,
)
from .structural import _ops

FISHBONE_CYCLIC_SCHEMA_VERSION = (
    "ifb3-cyclic-quasistatic-v1-multiple-pattern-committed-retry"
)
FISHBONE_CYCLIC_COMPARISON_SCHEMA_VERSION = (
    "ifb3-full3d-cyclic-common-completed-targets-v1"
)


def cyclic_pattern(
    artifact: FishboneArtifact,
    direction: str,
    pattern_name: str,
) -> np.ndarray:
    normalized = pattern_name.lower()
    if normalized == "first_mode":
        pattern, _control = modal_pattern(artifact, direction, 1)
        return pattern
    if normalized == "uniform":
        values = np.ones(artifact.story_count, dtype=float)
        values /= float(np.sum(values))
        return values
    raise ValueError("Fishbone cyclic pattern must be first_mode or uniform")


def protocol_targets(
    total_height_m: float,
    roof_drift_protocol: tuple[float, ...],
    cycles_per_level: int,
) -> tuple[float, ...]:
    if total_height_m <= 0.0 or cycles_per_level < 1:
        raise ValueError("Fishbone cyclic protocol controls are invalid")
    targets: list[float] = []
    for ratio in roof_drift_protocol:
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("Fishbone cyclic drift amplitudes must be positive")
        amplitude = float(ratio) * total_height_m
        for _cycle in range(cycles_per_level):
            targets.extend((amplitude, -amplitude))
    targets.append(0.0)
    return tuple(targets)


def _path_energy(force: np.ndarray, deformation: np.ndarray) -> float:
    force_values = np.asarray(force, dtype=float).reshape(-1)
    deformation_values = np.asarray(deformation, dtype=float).reshape(-1)
    if force_values.size != deformation_values.size:
        raise ValueError("Cyclic force/deformation history lengths differ")
    if force_values.size < 2:
        return 0.0
    increments = np.diff(deformation_values)
    average_force = 0.5 * (
        force_values[1:] + force_values[:-1]
    )
    return abs(float(np.sum(average_force * increments)))


def _relative_error(observed: float, reference: float) -> float:
    scale = abs(float(reference))
    if scale <= 1.0e-12:
        return math.nan
    return float((float(observed) - float(reference)) / scale)


def _profile_nrmse(
    observed: np.ndarray,
    reference: np.ndarray,
) -> float:
    left = np.asarray(observed, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Cyclic comparison profiles have different shapes")
    scale = float(np.sqrt(np.mean(right**2)))
    if scale <= 1.0e-12:
        return math.nan
    return float(np.sqrt(np.mean((left - right) ** 2)) / scale)


def _completed_target_index(
    target_index: np.ndarray,
    status: str,
) -> int:
    values = np.asarray(target_index, dtype=int).reshape(-1)
    if values.size == 0:
        return 0
    last = int(np.max(values))
    if status in {"protocol_complete", "complete"}:
        return last
    # A terminated history can contain a partially traversed final target.
    # Excluding it prevents a short failed branch from appearing to match a
    # complete Full 3D reversal.
    return max(last - 1, 0)


def _history_until_target(
    arrays: dict[str, np.ndarray],
    target: int,
) -> dict[str, np.ndarray]:
    indices = np.asarray(
        arrays["protocol_target_index"], dtype=int
    ).reshape(-1)
    mask = indices <= int(target)
    return {
        key: np.asarray(value)[mask]
        for key, value in arrays.items()
        if np.asarray(value).shape
        and np.asarray(value).shape[0] == indices.size
    }


def _resampled_base_loop(
    arrays: dict[str, np.ndarray],
    target: int,
    *,
    points_per_target: int = 51,
) -> np.ndarray:
    target_index = np.asarray(
        arrays["protocol_target_index"], dtype=int
    ).reshape(-1)
    if "base_shear_kn" in arrays:
        base = np.asarray(arrays["base_shear_kn"], dtype=float)
    else:
        base = np.asarray(
            arrays["story_shear_kn"], dtype=float
        )[:, 0]
    rows: list[np.ndarray] = []
    sample = np.linspace(0.0, 1.0, int(points_per_target))
    for index in range(1, int(target) + 1):
        values = base[target_index == index]
        if values.size == 0:
            continue
        progress = np.linspace(0.0, 1.0, values.size)
        rows.append(np.interp(sample, progress, values))
    return np.concatenate(rows) if rows else np.asarray([], dtype=float)


def compare_cyclic_histories(
    full3d_arrays: dict[str, np.ndarray],
    fishbone_arrays: dict[str, np.ndarray],
    story_heights_m: tuple[float, ...] | np.ndarray,
    *,
    full3d_status: str = "protocol_complete",
    fishbone_status: str = "protocol_complete",
) -> dict[str, Any]:
    """Compare only common *completed* cyclic reversals.

    The function deliberately excludes a partially traversed last target
    after numerical termination.  This makes energy, strength-retention and
    drift-localisation errors usable as calibration evidence without
    rewarding an early Fishbone failure.
    """

    heights = np.asarray(story_heights_m, dtype=float).reshape(-1)
    if heights.size == 0 or np.any(heights <= 0.0):
        raise ValueError("Cyclic comparison story heights are invalid")
    full_completed = _completed_target_index(
        full3d_arrays["protocol_target_index"], full3d_status
    )
    fishbone_completed = _completed_target_index(
        fishbone_arrays["protocol_target_index"], fishbone_status
    )
    common = min(full_completed, fishbone_completed)
    if common < 1:
        raise ValueError("No common completed cyclic target is available")
    full = _history_until_target(full3d_arrays, common)
    reduced = _history_until_target(fishbone_arrays, common)
    full_deformation = np.asarray(
        full["story_deformation_m"], dtype=float
    )
    reduced_deformation = np.asarray(
        reduced["story_deformation_m"], dtype=float
    )
    full_shear = np.asarray(full["story_shear_kn"], dtype=float)
    reduced_shear = np.asarray(reduced["story_shear_kn"], dtype=float)
    expected = (heights.size,)
    if (
        full_deformation.ndim != 2
        or reduced_deformation.ndim != 2
        or full_shear.shape != full_deformation.shape
        or reduced_shear.shape != reduced_deformation.shape
        or full_deformation.shape[1:] != expected
        or reduced_deformation.shape[1:] != expected
    ):
        raise ValueError("Cyclic history story-array shapes are inconsistent")

    full_idr_peak = np.max(
        np.abs(full_deformation / heights[None, :]), axis=0
    )
    reduced_idr_peak = np.max(
        np.abs(reduced_deformation / heights[None, :]), axis=0
    )
    full_shear_peak = np.max(np.abs(full_shear), axis=0)
    reduced_shear_peak = np.max(np.abs(reduced_shear), axis=0)
    full_energy = np.asarray(
        [
            _path_energy(full_shear[:, story], full_deformation[:, story])
            for story in range(heights.size)
        ],
        dtype=float,
    )
    reduced_energy = np.asarray(
        [
            _path_energy(
                reduced_shear[:, story],
                reduced_deformation[:, story],
            )
            for story in range(heights.size)
        ],
        dtype=float,
    )
    drift_errors = np.asarray(
        [
            _relative_error(reduced_idr_peak[index], full_idr_peak[index])
            for index in range(heights.size)
        ],
        dtype=float,
    )
    strength_errors = np.asarray(
        [
            _relative_error(
                reduced_shear_peak[index], full_shear_peak[index]
            )
            for index in range(heights.size)
        ],
        dtype=float,
    )
    energy_errors = np.asarray(
        [
            _relative_error(reduced_energy[index], full_energy[index])
            for index in range(heights.size)
        ],
        dtype=float,
    )
    full_loop = _resampled_base_loop(full3d_arrays, common)
    reduced_loop = _resampled_base_loop(fishbone_arrays, common)
    loop_nrmse = _profile_nrmse(reduced_loop, full_loop)

    def finite_summary(values: np.ndarray) -> dict[str, float]:
        finite = np.abs(values[np.isfinite(values)])
        return {
            "mean_abs": float(np.mean(finite)) if finite.size else math.nan,
            "median_abs": (
                float(np.median(finite)) if finite.size else math.nan
            ),
            "maximum_abs": (
                float(np.max(finite)) if finite.size else math.nan
            ),
        }

    return {
        "schema_version": FISHBONE_CYCLIC_COMPARISON_SCHEMA_VERSION,
        "full3d_completed_target_index": full_completed,
        "fishbone_completed_target_index": fishbone_completed,
        "common_completed_target_index": common,
        "fishbone_target_coverage": float(
            fishbone_completed / max(full_completed, 1)
        ),
        "full3d_peak_idr_by_story": full_idr_peak,
        "fishbone_peak_idr_by_story": reduced_idr_peak,
        "peak_idr_signed_relative_error_by_story": drift_errors,
        "peak_idr_error_summary": finite_summary(drift_errors),
        "drift_profile_nrmse": _profile_nrmse(
            reduced_idr_peak, full_idr_peak
        ),
        "full3d_peak_story_shear_kn": full_shear_peak,
        "fishbone_peak_story_shear_kn": reduced_shear_peak,
        "peak_story_shear_signed_relative_error": strength_errors,
        "peak_story_shear_error_summary": finite_summary(strength_errors),
        "full3d_story_energy_kn_m": full_energy,
        "fishbone_story_energy_kn_m": reduced_energy,
        "story_energy_signed_relative_error": energy_errors,
        "story_energy_error_summary": finite_summary(energy_errors),
        "energy_profile_nrmse": _profile_nrmse(
            reduced_energy, full_energy
        ),
        "base_shear_loop_nrmse": loop_nrmse,
        "full3d_critical_story": int(np.argmax(full_idr_peak)) + 1,
        "fishbone_critical_story": int(np.argmax(reduced_idr_peak)) + 1,
        "critical_story_match": bool(
            int(np.argmax(full_idr_peak))
            == int(np.argmax(reduced_idr_peak))
        ),
    }


def _signed_step(
    control_candidates: tuple[tuple[int, float], ...],
    dof: int,
    requested_increment: float,
    minimum_increment: float,
) -> tuple[bool, float, int, int]:
    ops = _ops()
    magnitude = abs(float(requested_increment))
    sign = math.copysign(1.0, float(requested_increment))
    level = 0
    strategies = (
        ("Newton", "NormDispIncr", 1.0e-7, 80),
        ("NewtonLineSearch", "NormDispIncr", 1.0e-7, 100),
        ("KrylovNewton", "EnergyIncr", 1.0e-7, 100),
        ("ModifiedNewton", "EnergyIncr", 1.0e-6, 140),
    )
    while magnitude + 1.0e-15 >= minimum_increment:
        for control_node, control_scale in control_candidates:
            candidate_increment = sign * magnitude * control_scale
            if abs(candidate_increment) < minimum_increment:
                continue
            for algorithm, test_type, tolerance, iterations in strategies:
                _set_static_strategy(
                    algorithm,
                    test_type=test_type,
                    tolerance=tolerance,
                    maximum_iterations=iterations,
                )
                ops.integrator(
                    "DisplacementControl",
                    control_node,
                    dof,
                    candidate_increment,
                )
                if int(ops.analyze(1)) == 0:
                    return (
                        True,
                        candidate_increment,
                        level,
                        control_node,
                    )
                # Recreate the Analysis object so no failed trial vector is
                # reused by the next algorithm or control floor.
                _configure_static()
        magnitude *= 0.5
        level += 1
    return False, sign * magnitude, level, control_candidates[0][0]


def run_cyclic_history(
    artifact: FishboneArtifact,
    *,
    direction: str,
    pattern_name: str,
    roof_drift_protocol: tuple[float, ...] = (
        0.0025,
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.040,
        0.060,
    ),
    cycles_per_level: int = 2,
    displacement_increment_m: float = 0.001,
    minimum_displacement_increment_m: float = 1.0e-6,
    maximum_steps_per_target: int = 10000,
    maximum_runtime_s: float = 120.0,
) -> dict[str, Any]:
    """Run one signed cyclic protocol without using ground-motion evidence."""

    artifact.validate()
    normalized_direction = direction.upper()
    if normalized_direction not in {"X", "Y"}:
        raise ValueError("Fishbone cyclic direction must be X or Y")
    if not (
        displacement_increment_m > 0.0
        and 0.0
        < minimum_displacement_increment_m
        <= displacement_increment_m
        and maximum_steps_per_target > 0
        and maximum_runtime_s > 0.0
    ):
        raise ValueError("Fishbone cyclic solver controls are invalid")

    pattern = cyclic_pattern(
        artifact, normalized_direction, pattern_name
    )
    info = build_model(artifact)
    run_gravity()
    ops = _ops()
    dof = 1 if normalized_direction == "X" else 2
    ops.timeSeries("Linear", 951)
    ops.pattern("Plain", 951, 951)
    for node, value in zip(info.floor_nodes, pattern):
        vector = [0.0] * 6
        vector[dof - 1] = float(value)
        ops.load(node, *vector)
    _configure_static()
    control_node = info.floor_nodes[-1]
    first_shape = np.asarray(
        (
            artifact.target_modal_x
            if normalized_direction == "X"
            else artifact.target_modal_y
        ).mode_shapes[0],
        dtype=float,
    )
    roof_shape = max(abs(float(first_shape[-1])), 1.0e-12)
    ranked_controls = [artifact.story_count - 1]
    ranked_controls.extend(
        int(index)
        for index in np.argsort(-np.abs(first_shape))
        if int(index) != artifact.story_count - 1
        and abs(float(first_shape[index])) > 1.0e-8
    )
    control_candidates = tuple(
        (
            info.floor_nodes[index],
            max(abs(float(first_shape[index])) / roof_shape, 0.05),
        )
        for index in ranked_controls
    )
    targets = protocol_targets(
        artifact.floor_elevations_m[-1],
        tuple(float(value) for value in roof_drift_protocol),
        int(cycles_per_level),
    )
    histories: dict[str, list[Any]] = {
        "analysis_step": [0.0],
        "protocol_target_index": [0.0],
        "load_factor": [0.0],
        "floor_displacement_m": [
            np.zeros(artifact.story_count, dtype=float)
        ],
        "story_deformation_m": [
            np.zeros(artifact.story_count, dtype=float)
        ],
        "story_drift_ratio": [
            np.zeros(artifact.story_count, dtype=float)
        ],
        "story_shear_kn": [
            np.zeros(artifact.story_count, dtype=float)
        ],
        "roof_displacement_m": [0.0],
        "column_hinge_rotation_rad": [
            np.zeros(artifact.story_count, dtype=float)
        ],
        "beam_hinge_rotation_rad": [
            np.zeros(artifact.story_count, dtype=float)
        ],
    }
    started = time.perf_counter()
    maximum_recovery_level = 0
    accepted_steps = 0
    control_floors_used = {artifact.story_count}
    status = "protocol_complete"
    for target_index, target in enumerate(targets, start=1):
        for _local_step in range(maximum_steps_per_target):
            if time.perf_counter() - started > maximum_runtime_s:
                status = "runtime_guard"
                break
            current = float(ops.nodeDisp(control_node, dof))
            remaining = float(target) - current
            if abs(remaining) <= minimum_displacement_increment_m:
                break
            requested = math.copysign(
                min(abs(remaining), displacement_increment_m),
                remaining,
            )
            solved, _accepted, level, accepted_control = _signed_step(
                control_candidates,
                dof,
                requested,
                minimum_displacement_increment_m,
            )
            control_floors_used.add(accepted_control - 100)
            maximum_recovery_level = max(maximum_recovery_level, level)
            if not solved:
                status = "numerical_failure"
                break
            accepted_steps += 1
            floor = np.asarray(
                [ops.nodeDisp(node, dof) for node in info.floor_nodes],
                dtype=float,
            )
            deformation = np.diff(np.concatenate(([0.0], floor)))
            drift = deformation / np.asarray(
                artifact.story_heights_m, dtype=float
            )
            factor = float(ops.getLoadFactor(951))
            story_shear = factor * np.cumsum(pattern[::-1])[::-1]
            column, beam = _hinge_story_peak_rotations(
                info.column_hinge_elements,
                info.beam_hinge_elements,
                artifact.story_count,
            )
            histories["analysis_step"].append(float(accepted_steps))
            histories["protocol_target_index"].append(float(target_index))
            histories["load_factor"].append(factor)
            histories["floor_displacement_m"].append(floor)
            histories["story_deformation_m"].append(deformation)
            histories["story_drift_ratio"].append(drift)
            histories["story_shear_kn"].append(story_shear)
            histories["roof_displacement_m"].append(float(floor[-1]))
            histories["column_hinge_rotation_rad"].append(column)
            histories["beam_hinge_rotation_rad"].append(beam)
        else:
            status = "maximum_steps_per_target"
        if status != "protocol_complete":
            break

    arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in histories.items()
    }
    story_energy = np.asarray(
        [
            abs(
                float(
                    np.trapz(
                        arrays["story_shear_kn"][:, index],
                        arrays["story_deformation_m"][:, index],
                    )
                )
            )
            for index in range(artifact.story_count)
        ],
        dtype=float,
    )
    return {
        "schema_version": FISHBONE_CYCLIC_SCHEMA_VERSION,
        "status": status,
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "calibration_stage": artifact.calibration_stage,
        "direction": normalized_direction,
        "pattern_name": pattern_name,
        "step_count": accepted_steps,
        "maximum_recovery_level": maximum_recovery_level,
        "control_floors_used": tuple(sorted(control_floors_used)),
        "maximum_story_drift_ratio": float(
            np.max(np.abs(arrays["story_drift_ratio"]))
        ),
        "story_dissipated_energy_kn_m": story_energy,
        "runtime_s": float(time.perf_counter() - started),
        "_history_arrays": arrays,
    }
