"""Exact-output batched solver orchestration for grouped Fishbone v5.

The structural domain and integration settings are unchanged from v4.  Node
recorders retain every committed response state while OpenSees advances a
small block of steps in one C++ call.  Failed blocks fall back to the same
local algorithms and substep factors used by v4.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G_STD
from .fishbone import FishboneArtifact
from .fishbone_v4_dynamic import (
    _apply_damping,
    _combine_directional_results,
    _configure_transient,
    _state,
)
from .fishbone_v4_grouped_frame import (
    build_grouped_frame_model,
    run_grouped_frame_gravity,
)
from .ground_motion import load_acceleration, pair_sa_geomean_g
from .io_utils import stable_hash
from .structural import _ops


FISHBONE_V5_BATCHED_SCHEMA_VERSION = (
    "bidirectional-fishbone-v5-grouped-2n-batched-recorder-v1"
)
SOLVER_POLICY = "batched-10-recorder-reconfigure-on-failure-v1"
DEFAULT_CHUNK_STEPS = 10


def pair_case_signature(
    artifact: FishboneArtifact,
    pair: dict[str, Any],
    target_im_g: float,
    *,
    candidate_definition_hash: str,
    damping_ratio: float,
    damping_model: str,
    dynamic_instability_drift: float,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
) -> dict[str, float | str]:
    period_x = float(artifact.target_modal_x.periods_s[0])
    period_y = float(artifact.target_modal_y.periods_s[0])
    reference_period = math.sqrt(period_x * period_y)
    unscaled_sa = float(pair_sa_geomean_g(pair, reference_period))
    scale_factor = float(target_im_g / unscaled_sa)
    signature = stable_hash(
        {
            "schema_version": FISHBONE_V5_BATCHED_SCHEMA_VERSION,
            "solver_policy": SOLVER_POLICY,
            "chunk_steps": int(chunk_steps),
            "output_sampling": "every-committed-step-opensees-node-recorder",
            "artifact_hash": artifact.artifact_hash,
            "source_model_hash": artifact.source_model_hash,
            "candidate_definition_hash": candidate_definition_hash,
            "pair_id": str(pair["pair_id"]),
            "sha256_x": str(pair.get("sha256_x") or ""),
            "sha256_y": str(pair.get("sha256_y") or ""),
            "target_im_g": float(target_im_g),
            "scale_factor": scale_factor,
            "damping_ratio": float(damping_ratio),
            "damping_model": str(damping_model),
            "dynamic_instability_drift": float(dynamic_instability_drift),
            "coupling_policy": "2n-block-diagonal-independent-direction",
        }
    )
    return {
        "reference_period_s": reference_period,
        "unscaled_sa_g": unscaled_sa,
        "scale_factor": scale_factor,
        "case_signature": signature,
    }


def _read_recorder(path: Path, column_count: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, column_count), dtype=float)
    values = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != column_count:
        raise RuntimeError(
            f"Recorder {path} has {values.shape[1]} columns; expected {column_count}"
        )
    return values


def _run_direction_batched(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    config: dict[str, Any],
    direction: str,
    acceleration_g: np.ndarray,
    *,
    record_dt_s: float,
    scale_factor: float,
    damping_ratio: float,
    damping_model: str,
    dynamic_instability_drift: float,
    maximum_runtime_s: float,
    minimum_substep_factor: int,
    chunk_steps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized = direction.upper()
    dof = 1 if normalized == "X" else 2
    info, audit = build_grouped_frame_model(
        artifact,
        building,
        config,
        normalized,
        preserve_all_frame_lines=False,
        include_transverse_torsion=False,
    )
    gravity_error = run_grouped_frame_gravity(info, audit)
    damping = _apply_damping(
        artifact, info, normalized, damping_ratio, damping_model
    )
    acceleration = np.asarray(acceleration_g, dtype=float) * G_STD * scale_factor
    ops = _ops()
    series_tag = 201 if normalized == "X" else 202
    pattern_tag = 301 if normalized == "X" else 302
    ops.timeSeries(
        "Path", series_tag, "-dt", record_dt_s, "-values", *acceleration
    )
    ops.pattern("UniformExcitation", pattern_tag, dof, "-accel", series_tag)
    _configure_transient()

    time_grid = np.arange(acceleration.size, dtype=float) * record_dt_s
    duration = float(time_grid[-1])
    initial = _state(
        artifact, info, normalized, 0.0, float(acceleration[0])
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
    allowed_factors = tuple(
        factor
        for factor in (1, 2, 4, 8, 16, 32, 64)
        if factor <= max(1, minimum_substep_factor)
    )

    node_count = len(info.floor_nodes)
    with tempfile.TemporaryDirectory(prefix="fishbone_v5_recorder_") as temporary:
        displacement_path = Path(temporary) / "displacement.out"
        acceleration_path = Path(temporary) / "acceleration.out"
        ops.recorder(
            "Node",
            "-file",
            str(displacement_path),
            "-precision",
            16,
            "-time",
            "-node",
            *info.floor_nodes,
            "-dof",
            dof,
            "disp",
        )
        ops.recorder(
            "Node",
            "-file",
            str(acceleration_path),
            "-precision",
            16,
            "-time",
            "-node",
            *info.floor_nodes,
            "-dof",
            dof,
            "accel",
        )
        try:
            while float(ops.getTime()) < duration - tolerance_time:
                if time.perf_counter() - started > maximum_runtime_s:
                    numerical_failure = True
                    collapse_basis.append("runtime_guard")
                    break
                committed_time = float(ops.getTime())
                remaining = duration - committed_time
                steps = min(
                    max(1, int(chunk_steps)),
                    max(int(math.floor(remaining / record_dt_s + 1.0e-9)), 1),
                )
                step_dt = min(record_dt_s, remaining / steps)
                code = int(ops.analyze(steps, step_dt))
                if code == 0:
                    continue

                recovery_log.append(
                    {
                        "time_s": float(ops.getTime()),
                        "requested_chunk_steps": int(steps),
                        "algorithm": "Newton",
                        "test_type": "EnergyIncr",
                        "tolerance": 1.0e-7,
                        "maximum_iterations": 40,
                        "code": code,
                        "stage": "batched_primary",
                    }
                )
                state = _state(
                    artifact,
                    info,
                    normalized,
                    float(ops.getTime()),
                    float(np.interp(float(ops.getTime()), time_grid, acceleration)),
                )
                if float(np.max(np.abs(state["story_drift_ratio"]))) >= (
                    dynamic_instability_drift
                ):
                    physical_collapse = True
                    collapse_basis.append("dynamic_instability_drift")
                    break

                solved = False
                for substep_factor in allowed_factors:
                    target_step = min(
                        record_dt_s, duration - float(ops.getTime())
                    )
                    trial_dt = target_step / substep_factor
                    for algorithm, test_type, tolerance, iterations in recovery_plan:
                        if time.perf_counter() - started > maximum_runtime_s:
                            numerical_failure = True
                            collapse_basis.append("runtime_guard_inside_recovery")
                            break
                        _configure_transient(
                            algorithm=algorithm,
                            test_type=test_type,
                            tolerance=tolerance,
                            maximum_iterations=iterations,
                        )
                        local_code = int(ops.analyze(1, trial_dt))
                        recovery_log.append(
                            {
                                "time_s": float(ops.getTime()),
                                "substep_factor": int(substep_factor),
                                "algorithm": algorithm,
                                "test_type": test_type,
                                "tolerance": tolerance,
                                "maximum_iterations": iterations,
                                "code": local_code,
                                "stage": "local_recovery",
                            }
                        )
                        if local_code == 0:
                            solved = True
                            break
                    if solved or numerical_failure:
                        break
                if numerical_failure:
                    break
                if not solved:
                    numerical_failure = True
                    collapse_basis.append("solver_nonconvergence")
                    break
                _configure_transient()
        finally:
            try:
                ops.remove("recorders")
            except Exception:
                pass
        displacement_rows = _read_recorder(displacement_path, node_count + 1)
        acceleration_rows = _read_recorder(acceleration_path, node_count + 1)

    if displacement_rows.shape != acceleration_rows.shape or not np.allclose(
        displacement_rows[:, 0],
        acceleration_rows[:, 0],
        rtol=0.0,
        atol=record_dt_s * 1.0e-5,
    ):
        raise RuntimeError("Fishbone displacement and acceleration recorders differ")
    recorded_time = displacement_rows[:, 0]
    displacement = displacement_rows[:, 1:]
    relative_acceleration = acceleration_rows[:, 1:]
    time_s = np.concatenate(([float(initial["time_s"])], recorded_time))
    floor_displacement = np.vstack(
        (np.asarray(initial["floor_displacement_m"], dtype=float), displacement)
    )
    ground = np.interp(recorded_time, time_grid, acceleration)
    absolute_acceleration = np.vstack(
        (
            np.asarray(initial["floor_absolute_acceleration_m_s2"], dtype=float),
            relative_acceleration + ground[:, None],
        )
    )
    story_deformation = np.diff(
        np.column_stack((np.zeros(floor_displacement.shape[0]), floor_displacement)),
        axis=1,
    )
    heights = np.asarray(artifact.story_heights_m, dtype=float)
    story_drift_ratio = story_deformation / heights[None, :]
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if normalized == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    base_shear = np.sum(absolute_acceleration * masses[None, :], axis=1)

    collapse_indices = np.flatnonzero(
        np.max(np.abs(story_drift_ratio), axis=1) >= dynamic_instability_drift
    )
    if collapse_indices.size:
        stop = int(collapse_indices[0]) + 1
        time_s = time_s[:stop]
        floor_displacement = floor_displacement[:stop]
        story_deformation = story_deformation[:stop]
        story_drift_ratio = story_drift_ratio[:stop]
        absolute_acceleration = absolute_acceleration[:stop]
        base_shear = base_shear[:stop]
        physical_collapse = True
        numerical_failure = False
        if "dynamic_instability_drift" not in collapse_basis:
            collapse_basis.append("dynamic_instability_drift")

    arrays = {
        "time_s": np.asarray(time_s, dtype=np.float64),
        "floor_displacement_m": np.asarray(floor_displacement, dtype=np.float64),
        "story_deformation_m": np.asarray(story_deformation, dtype=np.float64),
        "story_drift_ratio": np.asarray(story_drift_ratio, dtype=np.float64),
        "floor_absolute_acceleration_m_s2": np.asarray(
            absolute_acceleration, dtype=np.float64
        ),
        "base_shear_kn": np.asarray(base_shear, dtype=np.float64),
    }
    completed_fraction = min(
        max(float(arrays["time_s"][-1]) / max(duration, 1.0e-12), 0.0), 1.0
    )
    if physical_collapse:
        status = "dynamic_instability"
    elif numerical_failure or completed_fraction < 1.0 - 1.0e-8:
        status = "numerical_failure"
    else:
        status = "success"
    return {
        "direction": normalized,
        "status": status,
        "physical_collapse": physical_collapse,
        "numerical_failure": numerical_failure,
        "collapse_basis": collapse_basis,
        "completed_fraction": completed_fraction,
        "record_duration_s": duration,
        "termination_time_s": float(arrays["time_s"][-1]),
        "runtime_s": float(time.perf_counter() - started),
        "gravity_equilibrium_relative_error": float(gravity_error),
        "damping": damping,
        "audit": {
            "physical_frame_line_count": audit.physical_frame_line_count,
            "reduced_frame_group_count": audit.reduced_frame_group_count,
            "reduced_column_element_count": audit.reduced_column_element_count,
            "reduced_beam_element_count": audit.reduced_beam_element_count,
        },
        "recovery_attempt_count": len(recovery_log),
        "recovery_failure_count": sum(
            int(row["code"] != 0) for row in recovery_log
        ),
        "solver_policy": SOLVER_POLICY,
        "chunk_steps": int(chunk_steps),
        "arrays": arrays,
        "recovery_log": recovery_log,
    }


def run_pair_nltha(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    candidate_definition_hash: str,
    maximum_runtime_s_per_direction: float = 900.0,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
) -> dict[str, Any]:
    time_x, acceleration_x_g = load_acceleration(
        pair["component_x_path"], pair["units"]
    )
    time_y, acceleration_y_g = load_acceleration(
        pair["component_y_path"], pair["units"]
    )
    if not np.allclose(time_x, time_y, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: X/Y time vectors differ")
    damping_ratio = float(config["model"]["damping_ratio"])
    damping_model = str(config["model"]["damping_model"])
    instability = float(config["model"]["dynamic_instability_drift"])
    signature_data = pair_case_signature(
        artifact,
        pair,
        target_im_g,
        candidate_definition_hash=candidate_definition_hash,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=instability,
        chunk_steps=chunk_steps,
    )
    started = time.perf_counter()
    directional: dict[str, dict[str, Any]] = {}
    for direction, acceleration in (("X", acceleration_x_g), ("Y", acceleration_y_g)):
        directional[direction] = _run_direction_batched(
            artifact,
            building,
            config,
            direction,
            acceleration,
            record_dt_s=float(pair["dt_s"]),
            scale_factor=float(signature_data["scale_factor"]),
            damping_ratio=damping_ratio,
            damping_model=damping_model,
            dynamic_instability_drift=instability,
            maximum_runtime_s=maximum_runtime_s_per_direction,
            minimum_substep_factor=64,
            chunk_steps=chunk_steps,
        )
    arrays, summary = _combine_directional_results(
        artifact, directional["X"], directional["Y"]
    )
    return {
        "schema_version": FISHBONE_V5_BATCHED_SCHEMA_VERSION,
        "analysis_engine": "fishbone_v5_grouped_2n_batched_recorder",
        "solver_policy": SOLVER_POLICY,
        "chunk_steps": int(chunk_steps),
        "output_sampling": "every-committed-step-opensees-node-recorder",
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "source_model_hash": artifact.source_model_hash,
        "candidate_definition_hash": candidate_definition_hash,
        "pair_id": str(pair["pair_id"]),
        "target_im_g": float(target_im_g),
        "reference_period_s": float(signature_data["reference_period_s"]),
        "unscaled_sa_g": float(signature_data["unscaled_sa_g"]),
        "scale_factor": float(signature_data["scale_factor"]),
        "achieved_im_g": float(target_im_g),
        "case_signature": str(signature_data["case_signature"]),
        **summary,
        "directional_results": {
            direction: {
                key: value
                for key, value in result.items()
                if key not in {"arrays", "recovery_log"}
            }
            for direction, result in directional.items()
        },
        "runtime_s_solver": float(time.perf_counter() - started),
        "coupling_policy": "2n-block-diagonal-independent-direction",
        "limitations": [
            "no cross-direction axial-history coupling",
            "no torsional degree of freedom",
            "no bidirectional member interaction surface",
        ],
        "physics_changed_from_v4": False,
        "_history_arrays": arrays,
        "_recovery_logs": {
            direction: result["recovery_log"]
            for direction, result in directional.items()
        },
    }
