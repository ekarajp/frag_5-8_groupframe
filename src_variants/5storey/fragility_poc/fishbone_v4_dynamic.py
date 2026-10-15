"""Bidirectional NLTHA for the frozen grouped-frame Fishbone v4 candidate.

The current candidate is a block-diagonal 2N model: the X and Y grouped frame
systems receive the two components of the same physical record pair with one
geometric-mean Sa scale factor.  Because the frozen topology has no X--Y
coupling, solving the two blocks sequentially is mathematically equivalent to
assembling them in one OpenSees domain.  Directional P--M interaction and
P-Delta are retained inside each grouped frame; cross-direction axial-history
coupling is explicitly outside this schema.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G_STD
from .fishbone import FishboneArtifact, FishboneModelInfo
from .fishbone_dynamic import atomic_write_history
from .fishbone_v4_grouped_frame import (
    GroupedFrameAudit,
    build_grouped_frame_model,
    grouped_frame_directional_modes,
    run_grouped_frame_gravity,
)
from .ground_motion import load_acceleration, pair_sa_geomean_g
from .io_utils import atomic_write_json, read_json, stable_hash
from .structural import _ops


FISHBONE_V4_NLTHA_SCHEMA_VERSION = (
    "bidirectional-fishbone-v4-grouped-2n-block-diagonal-v1"
)


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


def _rayleigh_coefficients(
    omega_low: float, omega_high: float, damping_ratio: float
) -> tuple[float, float]:
    alpha_m = (
        2.0
        * damping_ratio
        * omega_low
        * omega_high
        / (omega_low + omega_high)
    )
    beta_k_comm = 2.0 * damping_ratio / (omega_low + omega_high)
    return float(alpha_m), float(beta_k_comm)


def _apply_damping(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    direction: str,
    damping_ratio: float,
    damping_model: str,
) -> dict[str, Any]:
    modal = grouped_frame_directional_modes(
        artifact, info, direction, requested_modes=artifact.story_count
    )
    rows = list(modal["modes"])
    normalized = str(damping_model).lower()
    ops = _ops()
    if normalized == "modal":
        ops.modalDamping(*([float(damping_ratio)] * artifact.story_count))
        return {
            "method": "modal",
            "damping_ratio": float(damping_ratio),
            "mode_count": artifact.story_count,
            "reported_directional_modes": rows,
        }
    if normalized != "rayleigh_committed":
        raise ValueError("Grouped Fishbone damping must be modal or rayleigh_committed")
    periods = [float(row["period_s"]) for row in rows]
    low_period = max(periods)
    high_period = min(periods)
    omega_low = 2.0 * math.pi / low_period
    omega_high = 2.0 * math.pi / high_period
    alpha_m, beta_k_comm = _rayleigh_coefficients(
        omega_low, omega_high, damping_ratio
    )
    ops.rayleigh(alpha_m, 0.0, 0.0, beta_k_comm)
    return {
        "method": "rayleigh_committed",
        "damping_ratio": float(damping_ratio),
        "alpha_m": alpha_m,
        "beta_k_comm": beta_k_comm,
        "reported_directional_modes": rows,
    }


def _state(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    direction: str,
    time_s: float,
    ground_acceleration_m_s2: float,
) -> dict[str, Any]:
    normalized = direction.upper()
    dof = 1 if normalized == "X" else 2
    displacement = np.asarray(
        [_ops().nodeDisp(node, dof) for node in info.floor_nodes], dtype=float
    )
    relative_acceleration = np.asarray(
        [_ops().nodeAccel(node, dof) for node in info.floor_nodes], dtype=float
    )
    absolute_acceleration = relative_acceleration + ground_acceleration_m_s2
    deformation = np.diff(np.concatenate(([0.0], displacement)))
    heights = np.asarray(artifact.story_heights_m, dtype=float)
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if normalized == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    return {
        "time_s": float(time_s),
        "floor_displacement_m": displacement,
        "story_deformation_m": deformation,
        "story_drift_ratio": deformation / heights,
        "floor_absolute_acceleration_m_s2": absolute_acceleration,
        "base_shear_kn": float(np.sum(masses * absolute_acceleration)),
    }


def _append(histories: dict[str, list[Any]], state: dict[str, Any]) -> None:
    for key, value in state.items():
        histories.setdefault(key, []).append(value)


def _run_direction(
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
    persistent_primary_analysis: bool = False,
    model_builder: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized = direction.upper()
    dof = 1 if normalized == "X" else 2
    supplemental_audit = None
    if model_builder is None:
        info, audit = build_grouped_frame_model(
            artifact,
            building,
            config,
            normalized,
            preserve_all_frame_lines=False,
            include_transverse_torsion=False,
        )
    else:
        built = model_builder(artifact, building, config, normalized)
        if len(built) != 3:
            raise ValueError("An alternate grouped-frame builder must return three values")
        info, audit, supplemental_audit = built
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
    histories: dict[str, list[Any]] = {}
    _append(
        histories,
        _state(artifact, info, normalized, 0.0, float(acceleration[0])),
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
        for factor in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if factor <= max(1, minimum_substep_factor)
    )
    while float(ops.getTime()) < duration - tolerance_time:
        if time.perf_counter() - started > maximum_runtime_s:
            numerical_failure = True
            collapse_basis.append("runtime_guard")
            break
        committed_time = float(ops.getTime())
        target_step = min(record_dt_s, duration - committed_time)
        solved = False
        recovery_used = False
        if persistent_primary_analysis:
            code = int(ops.analyze(1, target_step))
            if code == 0:
                solved = True
            else:
                recovery_used = True
                recovery_log.append(
                    {
                        "time_s": committed_time,
                        "substep_factor": 1,
                        "algorithm": "Newton",
                        "test_type": "EnergyIncr",
                        "tolerance": 1.0e-7,
                        "maximum_iterations": 40,
                        "code": code,
                        "stage": "persistent_primary",
                    }
                )
        if not solved:
            for substep_factor in allowed_factors:
                trial_dt = target_step / substep_factor
                for algorithm, test_type, tolerance, iterations in recovery_plan:
                    if (
                        persistent_primary_analysis
                        and substep_factor == 1
                        and algorithm == "Newton"
                        and test_type == "EnergyIncr"
                        and math.isclose(tolerance, 1.0e-7)
                        and iterations == 40
                    ):
                        continue
                    if time.perf_counter() - started > maximum_runtime_s:
                        numerical_failure = True
                        collapse_basis.append("runtime_guard_inside_recovery")
                        break
                    recovery_used = True
                    _configure_transient(
                        algorithm=algorithm,
                        test_type=test_type,
                        tolerance=tolerance,
                        maximum_iterations=iterations,
                    )
                    code = int(ops.analyze(1, trial_dt))
                    if code != 0 or substep_factor > 1 or algorithm != "Newton":
                        recovery_log.append(
                            {
                                "time_s": committed_time,
                                "substep_factor": int(substep_factor),
                                "algorithm": algorithm,
                                "test_type": test_type,
                                "tolerance": tolerance,
                                "maximum_iterations": iterations,
                                "code": code,
                                "stage": "local_recovery",
                            }
                        )
                    if code == 0:
                        solved = True
                        break
                if solved or numerical_failure:
                    break
        if persistent_primary_analysis and solved and recovery_used:
            # Restore the inexpensive primary configuration once after a
            # recovery.  The v4 loop rebuilt it at every successful time step.
            _configure_transient()
        if numerical_failure:
            break
        if not solved:
            numerical_failure = True
            collapse_basis.append("solver_nonconvergence")
            break
        current_time = float(ops.getTime())
        ground = float(np.interp(current_time, time_grid, acceleration))
        state = _state(artifact, info, normalized, current_time, ground)
        _append(histories, state)
        current_midr = float(
            np.max(np.abs(np.asarray(state["story_drift_ratio"], dtype=float)))
        )
        if current_midr >= dynamic_instability_drift:
            physical_collapse = True
            collapse_basis.append("dynamic_instability_drift")
            break

    arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in histories.items()
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
    audit_payload = {
        "physical_frame_line_count": audit.physical_frame_line_count,
        "reduced_frame_group_count": audit.reduced_frame_group_count,
        "reduced_column_element_count": audit.reduced_column_element_count,
        "reduced_beam_element_count": audit.reduced_beam_element_count,
    }
    if supplemental_audit is not None:
        audit_payload["alternate_member_formulation"] = {
            key: value for key, value in vars(supplemental_audit).items()
        }
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
        "audit": audit_payload,
        "recovery_attempt_count": len(recovery_log),
        "recovery_failure_count": sum(
            int(row["code"] != 0) for row in recovery_log
        ),
        "solver_policy": (
            "persistent-primary-reconfigure-on-failure-v1"
            if persistent_primary_analysis
            else "v4-reconfigure-every-step"
        ),
        "arrays": arrays,
        "recovery_log": recovery_log,
    }


def _interpolate_matrix(
    target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        return np.interp(target_time, source_time, matrix)
    return np.column_stack(
        [np.interp(target_time, source_time, matrix[:, index]) for index in range(matrix.shape[1])]
    )


def _combine_directional_results(
    artifact: FishboneArtifact,
    result_x: dict[str, Any],
    result_y: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays_x = result_x["arrays"]
    arrays_y = result_y["arrays"]
    common_end = min(
        float(arrays_x["time_s"][-1]), float(arrays_y["time_s"][-1])
    )
    time = np.unique(
        np.concatenate(
            (
                arrays_x["time_s"][arrays_x["time_s"] <= common_end + 1.0e-12],
                arrays_y["time_s"][arrays_y["time_s"] <= common_end + 1.0e-12],
                np.asarray([common_end]),
            )
        )
    )
    combined: dict[str, np.ndarray] = {"time_s": time}
    mappings = {
        "floor_displacement": "floor_displacement_m",
        "story_deformation": "story_deformation_m",
        "story_drift_ratio": "story_drift_ratio",
        "floor_absolute_acceleration": "floor_absolute_acceleration_m_s2",
        "base_shear": "base_shear_kn",
    }
    for suffix, source_key in mappings.items():
        combined[f"{suffix}_x" + ("" if suffix == "base_shear" else "_m" if suffix in {"floor_displacement", "story_deformation"} else "_m_s2" if suffix == "floor_absolute_acceleration" else "")] = _interpolate_matrix(
            time, arrays_x["time_s"], arrays_x[source_key]
        )
        combined[f"{suffix}_y" + ("" if suffix == "base_shear" else "_m" if suffix in {"floor_displacement", "story_deformation"} else "_m_s2" if suffix == "floor_absolute_acceleration" else "")] = _interpolate_matrix(
            time, arrays_y["time_s"], arrays_y[source_key]
        )
    # Normalize names to the historical validation contract.
    combined["floor_displacement_x_m"] = combined.pop("floor_displacement_x_m")
    combined["floor_displacement_y_m"] = combined.pop("floor_displacement_y_m")
    combined["story_deformation_x_m"] = combined.pop("story_deformation_x_m")
    combined["story_deformation_y_m"] = combined.pop("story_deformation_y_m")
    combined["story_drift_ratio_x"] = combined.pop("story_drift_ratio_x")
    combined["story_drift_ratio_y"] = combined.pop("story_drift_ratio_y")
    combined["floor_absolute_acceleration_x_m_s2"] = combined.pop(
        "floor_absolute_acceleration_x_m_s2"
    )
    combined["floor_absolute_acceleration_y_m_s2"] = combined.pop(
        "floor_absolute_acceleration_y_m_s2"
    )
    combined["base_shear_x_kn"] = combined.pop("base_shear_x")
    combined["base_shear_y_kn"] = combined.pop("base_shear_y")

    absolute_x = np.abs(combined["story_drift_ratio_x"])
    absolute_y = np.abs(combined["story_drift_ratio_y"])
    index_x = np.unravel_index(int(np.argmax(absolute_x)), absolute_x.shape)
    index_y = np.unravel_index(int(np.argmax(absolute_y)), absolute_y.shape)
    max_x = float(absolute_x[index_x])
    max_y = float(absolute_y[index_y])
    if max_x >= max_y:
        critical_direction, critical_index, maximum = "X", index_x, max_x
    else:
        critical_direction, critical_index, maximum = "Y", index_y, max_y
    physical = bool(result_x["physical_collapse"] or result_y["physical_collapse"])
    numerical = bool(result_x["numerical_failure"] or result_y["numerical_failure"])
    if physical:
        status = "dynamic_instability"
    elif numerical:
        status = "numerical_failure"
    else:
        status = "success"
    summary = {
        "status": status,
        "physical_collapse": physical,
        "numerical_failure": numerical,
        "maximum_midr": maximum,
        "max_midr": maximum,
        "maximum_midr_x": max_x,
        "maximum_midr_y": max_y,
        "critical_direction": critical_direction,
        "critical_story": int(critical_index[1] + 1),
        "critical_time_s": float(time[critical_index[0]]),
        "maximum_roof_displacement_x_m": float(
            np.max(np.abs(combined["floor_displacement_x_m"][:, -1]))
        ),
        "maximum_roof_displacement_y_m": float(
            np.max(np.abs(combined["floor_displacement_y_m"][:, -1]))
        ),
        "maximum_base_shear_x_kn": float(
            np.max(np.abs(combined["base_shear_x_kn"]))
        ),
        "maximum_base_shear_y_kn": float(
            np.max(np.abs(combined["base_shear_y_kn"]))
        ),
        "completed_fraction": min(
            float(result_x["completed_fraction"]),
            float(result_y["completed_fraction"]),
        ),
        "termination_time_s": common_end,
        "record_duration_s": min(
            float(result_x["record_duration_s"]),
            float(result_y["record_duration_s"]),
        ),
        "collapse_basis": [
            *[f"X:{value}" for value in result_x["collapse_basis"]],
            *[f"Y:{value}" for value in result_y["collapse_basis"]],
        ],
    }
    return combined, summary


def pair_case_signature(
    artifact: FishboneArtifact,
    pair: dict[str, Any],
    target_im_g: float,
    *,
    certification_hash: str,
    damping_ratio: float,
    damping_model: str,
    dynamic_instability_drift: float,
) -> dict[str, float | str]:
    period_x = float(artifact.target_modal_x.periods_s[0])
    period_y = float(artifact.target_modal_y.periods_s[0])
    reference_period = math.sqrt(period_x * period_y)
    unscaled_sa = float(pair_sa_geomean_g(pair, reference_period))
    scale_factor = float(target_im_g / unscaled_sa)
    signature = stable_hash(
        {
            "schema_version": FISHBONE_V4_NLTHA_SCHEMA_VERSION,
            "artifact_hash": artifact.artifact_hash,
            "source_model_hash": artifact.source_model_hash,
            "certification_hash": certification_hash,
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


def run_pair_nltha(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    certification_hash: str,
    maximum_runtime_s_per_direction: float = 900.0,
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
        certification_hash=certification_hash,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=instability,
    )
    started = time.perf_counter()
    directional: dict[str, dict[str, Any]] = {}
    for direction, acceleration in (("X", acceleration_x_g), ("Y", acceleration_y_g)):
        directional[direction] = _run_direction(
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
        )
    arrays, summary = _combine_directional_results(
        artifact, directional["X"], directional["Y"]
    )
    return {
        "schema_version": FISHBONE_V4_NLTHA_SCHEMA_VERSION,
        "analysis_engine": "fishbone_v4_grouped_2n_block_diagonal",
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "source_model_hash": artifact.source_model_hash,
        "certification_hash": certification_hash,
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
        "_history_arrays": arrays,
        "_recovery_logs": {
            direction: result["recovery_log"]
            for direction, result in directional.items()
        },
    }


def case_paths(
    run_dir: str | Path,
    freeze_hash: str,
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> tuple[Path, Path]:
    token = f"{target_im_g:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    root = (
        Path(run_dir)
        / "fishbone_v4"
        / "dynamic"
        / freeze_hash
        / building_id
        / pair_id
        / f"im_{token}"
    )
    return root / "case.json", root / "case.npz"


def load_cached_case(
    metadata_path: Path, history_path: Path, expected_signature: str
) -> dict[str, Any] | None:
    if not metadata_path.is_file() or not history_path.is_file():
        return None
    payload = read_json(metadata_path)
    if payload.get("case_signature") != expected_signature:
        return None
    payload["history_path"] = str(history_path.absolute())
    return payload


def save_case(
    result: dict[str, Any], metadata_path: Path, history_path: Path
) -> dict[str, Any]:
    arrays = result["_history_arrays"]
    atomic_write_history(history_path, arrays)
    payload = {key: value for key, value in result.items() if not key.startswith("_")}
    payload["history_path"] = str(history_path.absolute())
    atomic_write_json(metadata_path, payload)
    return payload
