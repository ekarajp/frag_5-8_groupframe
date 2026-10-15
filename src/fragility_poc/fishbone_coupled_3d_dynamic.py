"""Development-only coupled four-spine Fishbone feasibility engine.

The first prototype deliberately targets one-bay symmetric archetypes.  It
retains the four physical corner columns and their source 3D fibre sections,
so axial force, P-Mx-My response, hinge history and P-Delta are shared during
one simultaneous X/Y analysis.  Diaphragm torsional rotation is suppressed.

For a one-bay frame the parallel-beam grouping used by v5C cannot be combined
with four independently loaded corner columns without inventing a load-sharing
rule.  This feasibility engine therefore retains the four perimeter beams.  It
is a proof of the coupled-column mechanism and its attainable runtime, not yet
a production reduced model.  A later beam condensation is authorized only if
this mechanism passes both the accuracy and speed gates preregistered by the
pilot definition.
"""

from __future__ import annotations

import copy
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G_STD
from .fishbone import FishboneArtifact
from .ground_motion import load_acceleration, pair_sa_geomean_g
from .ida import _configure_transient, _set_transient_solution_strategy
from .io_utils import stable_hash
from .structural import (
    _ops,
    apply_damping,
    build_model,
    modal_properties,
    run_gravity,
)


SCHEMA_VERSION = "fishbone-coupled-3d-column-four-spine-feasibility-v2"
SOLVER_POLICY = "simultaneous-xy-batched-recorder-cp-first-passage-v2"
DEFAULT_CHUNK_STEPS = 10


def pair_case_signature(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    *,
    candidate_definition_hash: str,
    damping_ratio: float,
    damping_model: str,
    cp_threshold: float,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
) -> dict[str, float | str]:
    """Return the immutable case signature for the feasibility prototype."""

    reference_period = float(building["t1_s"])
    unscaled_sa = float(pair_sa_geomean_g(pair, reference_period))
    scale_factor = float(target_im_g / unscaled_sa)
    signature = stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "solver_policy": SOLVER_POLICY,
            "candidate_definition_hash": candidate_definition_hash,
            "artifact_hash": artifact.artifact_hash,
            "source_model_hash": artifact.source_model_hash,
            "building_model_hash": str(building["model_hash"]),
            "pair_id": str(pair["pair_id"]),
            "sha256_x": str(pair.get("sha256_x") or ""),
            "sha256_y": str(pair.get("sha256_y") or ""),
            "target_im_g": float(target_im_g),
            "scale_factor": scale_factor,
            "damping_ratio": float(damping_ratio),
            "damping_model": str(damping_model),
            "cp_threshold": float(cp_threshold),
            "chunk_steps": int(chunk_steps),
            "torsional_rotation": "absent-translation-only-diaphragm",
            "column_policy": "four-physical-corner-3d-fibre-spines",
            "beam_policy": "four-source-perimeter-beams-one-bay",
            "coupling_policy": "shared-axial-p-mx-my-hinge-pdelta",
        }
    )
    return {
        "reference_period_s": reference_period,
        "unscaled_sa_g": unscaled_sa,
        "scale_factor": scale_factor,
        "case_signature": signature,
    }


def _row_maximum_midr(
    row: list[float], story_heights_m: np.ndarray
) -> float:
    displacement = np.asarray(row[1:], dtype=float).reshape(-1, 2)
    deformation = np.diff(
        np.vstack((np.zeros((1, 2), dtype=float), displacement)), axis=0
    )
    return float(np.max(np.abs(deformation / story_heights_m[:, None])))


def _current_maximum_midr(
    master_nodes: tuple[int, ...], story_heights_m: np.ndarray
) -> float:
    ops = _ops()
    displacement = np.asarray(
        [[ops.nodeDisp(node, 1), ops.nodeDisp(node, 2)] for node in master_nodes],
        dtype=float,
    )
    deformation = np.diff(
        np.vstack((np.zeros((1, 2), dtype=float), displacement)), axis=0
    )
    return float(np.max(np.abs(deformation / story_heights_m[:, None])))


def _read_recorder(path: Path, columns: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, columns), dtype=float)
    values = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != columns:
        raise RuntimeError(
            f"Recorder {path} has {values.shape[1]} columns; expected {columns}"
        )
    return values


def run_pair_nltha(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    candidate_definition_hash: str,
    maximum_runtime_s: float = 3600.0,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
) -> dict[str, Any]:
    """Run one simultaneous-X/Y four-corner-column feasibility case."""

    artifact.validate()
    if int(building["number_of_bays"]) != 1:
        raise ValueError("The coupled four-spine feasibility engine is one-bay only")
    if str(building["building_id"]) != artifact.building_id:
        raise ValueError("Building and artifact IDs differ")
    if str(building["model_hash"]) != artifact.source_model_hash:
        raise ValueError("Artifact source model is stale")
    if chunk_steps < 1:
        raise ValueError("chunk_steps must be positive")

    time_x, acceleration_x_g = load_acceleration(
        pair["component_x_path"], pair["units"]
    )
    time_y, acceleration_y_g = load_acceleration(
        pair["component_y_path"], pair["units"]
    )
    if not np.allclose(time_x, time_y, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: X/Y time vectors differ")
    record_dt = float(pair["dt_s"])
    observed_dt = float(np.median(np.diff(time_x)))
    if not math.isclose(observed_dt, record_dt, rel_tol=1.0e-6, abs_tol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: catalog and record dt differ")

    damping_ratio = float(config["model"]["damping_ratio"])
    damping_model = str(config["model"]["damping_model"])
    cp_threshold = 0.04
    signature = pair_case_signature(
        artifact,
        building,
        pair,
        target_im_g,
        candidate_definition_hash=candidate_definition_hash,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        cp_threshold=cp_threshold,
        chunk_steps=chunk_steps,
    )

    candidate_config = copy.deepcopy(config)
    candidate_config["model"]["lock_diaphragm_torsion"] = False
    candidate_config["model"]["diaphragm_constraint_mode"] = (
        "translation-only-no-torsion"
    )
    candidate_config["model"]["dynamic_instability_drift"] = cp_threshold
    candidate_config["ida"]["component_response_recording"]["enabled"] = False

    started = time.perf_counter()
    ops = _ops()
    info = build_model(building, candidate_config)
    if info.columns_per_storey != 4 or info.beams_per_storey != 4:
        raise AssertionError("One-bay coupled prototype must retain 4 columns and 4 beams")
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"gravity equilibrium error {gravity_error:.3%} exceeds 1%"
        )
    # Locking RZ removes the five torsional floor modes.  Request only the
    # two translational mode families; asking the source Full3D default of 12
    # returns numerically infinite eigenvalues and corrupts mass orthogonality.
    modal_mode_count = min(
        int(candidate_config["model"].get("modal_mode_count", 12)),
        2 * len(info.master_nodes),
    )
    modal = modal_properties(
        info,
        mode_count=modal_mode_count,
        solver=str(candidate_config["model"].get("eigen_solver", "fullGenLapack")),
        minimum_cumulative_mass_ratio=float(
            candidate_config["model"]["minimum_cumulative_translational_mass_ratio"]
        ),
    )
    apply_damping(modal, damping_ratio, method=damping_model)

    scale_factor = float(signature["scale_factor"])
    acceleration_x = np.asarray(acceleration_x_g, dtype=float) * G_STD * scale_factor
    acceleration_y = np.asarray(acceleration_y_g, dtype=float) * G_STD * scale_factor
    ops.timeSeries("Path", 101, "-dt", record_dt, "-values", *acceleration_x)
    ops.timeSeries("Path", 102, "-dt", record_dt, "-values", *acceleration_y)
    ops.pattern("UniformExcitation", 101, 1, "-accel", 101)
    ops.pattern("UniformExcitation", 102, 2, "-accel", 102)
    _configure_transient(
        "Newton", test_type="EnergyIncr", tolerance=1.0e-7, max_iterations=30
    )

    duration = float(time_x[-1] - time_x[0])
    end_time = float(ops.getTime()) + duration
    tolerance_time = max(1.0e-10, record_dt * 1.0e-8)
    heights = np.diff(np.concatenate(([0.0], np.asarray(info.floor_elevations_m))))
    node_count = len(info.master_nodes)
    recovery_plan = (
        (2, "Newton", "EnergyIncr", 1.0e-7, 30),
        (4, "Newton", "EnergyIncr", 1.0e-7, 30),
        (4, "NewtonLineSearch", "EnergyIncr", 1.0e-7, 60),
        (4, "ModifiedNewton", "EnergyIncr", 1.0e-6, 120),
        (4, "KrylovNewton", "NormDispIncr", 1.0e-6, 120),
    )
    recovery_log: list[dict[str, Any]] = []
    numerical_failure = False
    cp_endpoint_reached = False

    with tempfile.TemporaryDirectory(prefix="fishbone_coupled3d_") as temporary:
        displacement_path = Path(temporary) / "displacement.out"
        acceleration_path = Path(temporary) / "acceleration.out"
        common_recorder = (
            "Node",
            "-precision",
            16,
            "-time",
            "-node",
            *info.master_nodes,
            "-dof",
            1,
            2,
        )
        ops.recorder(
            common_recorder[0],
            "-file",
            str(displacement_path),
            *common_recorder[1:],
            "disp",
        )
        ops.recorder(
            common_recorder[0],
            "-file",
            str(acceleration_path),
            *common_recorder[1:],
            "accel",
        )
        while float(ops.getTime()) < end_time - tolerance_time:
            if time.perf_counter() - started > maximum_runtime_s:
                numerical_failure = True
                recovery_log.append({"stage": "runtime_guard", "time_s": float(ops.getTime())})
                break
            remaining = end_time - float(ops.getTime())
            steps = min(
                int(chunk_steps),
                max(int(math.floor(remaining / record_dt + 1.0e-9)), 1),
            )
            step_dt = min(record_dt, remaining / steps)
            _set_transient_solution_strategy(
                "Newton", test_type="EnergyIncr", tolerance=1.0e-7, max_iterations=30
            )
            code = int(ops.analyze(steps, step_dt))
            if _current_maximum_midr(info.master_nodes, heights) >= cp_threshold:
                cp_endpoint_reached = True
                break
            if code == 0:
                continue

            recovered = False
            for factor, algorithm, test_type, tolerance, iterations in recovery_plan:
                if time.perf_counter() - started > maximum_runtime_s:
                    break
                local_dt = min(record_dt / factor, end_time - float(ops.getTime()))
                _set_transient_solution_strategy(
                    algorithm,
                    test_type=test_type,
                    tolerance=tolerance,
                    max_iterations=iterations,
                )
                local_code = int(ops.analyze(1, local_dt))
                recovery_log.append(
                    {
                        "stage": "local_recovery",
                        "time_s": float(ops.getTime()),
                        "substep_factor": factor,
                        "algorithm": algorithm,
                        "test_type": test_type,
                        "tolerance": tolerance,
                        "maximum_iterations": iterations,
                        "code": local_code,
                    }
                )
                if _current_maximum_midr(info.master_nodes, heights) >= cp_threshold:
                    cp_endpoint_reached = True
                    recovered = True
                    break
                if local_code == 0:
                    recovered = True
                    break
            if cp_endpoint_reached:
                break
            if not recovered:
                numerical_failure = True
                break

        ops.remove("recorders")
        displacement_rows = _read_recorder(
            displacement_path, 1 + 2 * node_count
        )
        acceleration_rows = _read_recorder(
            acceleration_path, 1 + 2 * node_count
        )

    if displacement_rows.size == 0 or acceleration_rows.size == 0:
        raise RuntimeError("Coupled prototype recorders returned no response")
    row_count = min(displacement_rows.shape[0], acceleration_rows.shape[0])
    displacement_rows = displacement_rows[:row_count]
    acceleration_rows = acceleration_rows[:row_count]
    if not np.allclose(
        displacement_rows[:, 0], acceleration_rows[:, 0], rtol=0.0, atol=record_dt * 1.0e-5
    ):
        raise RuntimeError("Coupled displacement and acceleration times differ")

    recorded_time = displacement_rows[:, 0]
    displacement = displacement_rows[:, 1:].reshape(row_count, node_count, 2)
    deformation = np.diff(
        np.concatenate((np.zeros((row_count, 1, 2)), displacement), axis=1), axis=1
    )
    drift = deformation / heights[None, :, None]
    combined_abs = np.abs(drift)
    crossings = np.flatnonzero(np.max(combined_abs, axis=(1, 2)) >= cp_threshold)
    if crossings.size:
        stop = int(crossings[0]) + 1
        recorded_time = recorded_time[:stop]
        displacement = displacement[:stop]
        deformation = deformation[:stop]
        drift = drift[:stop]
        acceleration_rows = acceleration_rows[:stop]
        cp_endpoint_reached = True

    ground_x = np.interp(recorded_time, time_x, acceleration_x)
    ground_y = np.interp(recorded_time, time_y, acceleration_y)
    relative_acceleration = acceleration_rows[:, 1:].reshape(-1, node_count, 2)
    absolute_acceleration = relative_acceleration + np.column_stack(
        (ground_x, ground_y)
    )[:, None, :]
    masses_x = np.asarray(info.floor_masses_x_kn_s2_m, dtype=float)
    masses_y = np.asarray(info.floor_masses_y_kn_s2_m, dtype=float)
    base_x = np.sum(absolute_acceleration[:, :, 0] * masses_x[None, :], axis=1)
    base_y = np.sum(absolute_acceleration[:, :, 1] * masses_y[None, :], axis=1)

    absolute = np.abs(drift)
    critical = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    max_midr = float(absolute[critical])
    completed_fraction = min(float(recorded_time[-1]) / max(duration, 1.0e-12), 1.0)
    if cp_endpoint_reached:
        status = "cp_endpoint_reached"
        numerical_failure = False
    elif numerical_failure or completed_fraction < 1.0 - 1.0e-8:
        status = "numerical_failure"
    else:
        status = "success"

    arrays = {
        "time_s": np.asarray(recorded_time, dtype=np.float64),
        "floor_displacement_x_m": np.asarray(displacement[:, :, 0], dtype=np.float64),
        "floor_displacement_y_m": np.asarray(displacement[:, :, 1], dtype=np.float64),
        "story_deformation_x_m": np.asarray(deformation[:, :, 0], dtype=np.float64),
        "story_deformation_y_m": np.asarray(deformation[:, :, 1], dtype=np.float64),
        "story_drift_ratio_x": np.asarray(drift[:, :, 0], dtype=np.float64),
        "story_drift_ratio_y": np.asarray(drift[:, :, 1], dtype=np.float64),
        "floor_absolute_acceleration_x_m_s2": np.asarray(
            absolute_acceleration[:, :, 0], dtype=np.float64
        ),
        "floor_absolute_acceleration_y_m_s2": np.asarray(
            absolute_acceleration[:, :, 1], dtype=np.float64
        ),
        "base_shear_x_kn": np.asarray(base_x, dtype=np.float64),
        "base_shear_y_kn": np.asarray(base_y, dtype=np.float64),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_engine": "coupled_3d_column_four_spine_feasibility",
        "solver_policy": SOLVER_POLICY,
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "source_model_hash": artifact.source_model_hash,
        "candidate_definition_hash": candidate_definition_hash,
        "pair_id": str(pair["pair_id"]),
        "target_im_g": float(target_im_g),
        "reference_period_s": float(signature["reference_period_s"]),
        "unscaled_sa_g": float(signature["unscaled_sa_g"]),
        "scale_factor": scale_factor,
        "achieved_im_g": float(target_im_g),
        "case_signature": str(signature["case_signature"]),
        "status": status,
        "cp_endpoint_reached": cp_endpoint_reached,
        "physical_collapse": False,
        "numerical_failure": numerical_failure,
        "max_midr": max_midr,
        "maximum_midr": max_midr,
        "maximum_midr_x": float(np.max(np.abs(drift[:, :, 0]))),
        "maximum_midr_y": float(np.max(np.abs(drift[:, :, 1]))),
        "critical_direction": "X" if critical[2] == 0 else "Y",
        "critical_story": int(critical[1] + 1),
        "critical_time_s": float(recorded_time[critical[0]]),
        "maximum_roof_displacement_x_m": float(np.max(np.abs(displacement[:, -1, 0]))),
        "maximum_roof_displacement_y_m": float(np.max(np.abs(displacement[:, -1, 1]))),
        "maximum_base_shear_x_kn": float(np.max(np.abs(base_x))),
        "maximum_base_shear_y_kn": float(np.max(np.abs(base_y))),
        "completed_fraction": completed_fraction,
        "termination_time_s": float(recorded_time[-1]),
        "record_duration_s": duration,
        "runtime_s_solver": float(time.perf_counter() - started),
        "gravity_equilibrium_relative_error": float(gravity_error),
        "modal_period_x_s": float(modal.t_x_s),
        "modal_period_y_s": float(modal.t_y_s),
        "recovery_attempt_count": len(recovery_log),
        "coupling_policy": "shared-axial-p-mx-my-hinge-damage-pdelta",
        "torsional_rotation": "absent-translation-only-diaphragm",
        "topology_audit": {
            "one_bay_only": True,
            "corner_column_spine_count": 4,
            "column_elements": int(info.columns_per_storey * len(info.master_nodes)),
            "perimeter_beam_elements": int(info.beams_per_storey * len(info.master_nodes)),
            "source_full3d_member_topology_retained": True,
            "grouped_beam_condensation_applied": False,
            "reason": (
                "one-bay four-corner axial redistribution requires the four perimeter "
                "beam load paths; this pilot measures whether coupling is worth a later "
                "beam-condensation derivation"
            ),
        },
        "limitations": [
            "development-only one-bay feasibility prototype",
            "beam topology is not yet reduced relative to one-bay Full3D",
            "diaphragm torsional rotation suppressed",
            "response after first CP passage is not evaluated",
        ],
        "_history_arrays": arrays,
        "_recovery_logs": {"simultaneous_xy": recovery_log},
    }
