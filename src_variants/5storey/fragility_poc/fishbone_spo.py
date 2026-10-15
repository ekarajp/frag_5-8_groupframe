"""Nonlinear static analysis for Fishbone-v2.

The runner keeps physical post-peak termination distinct from numerical
failure and returns floor/story histories compatible with the Full 3D
calibration curves.  It is diagnostic until monotonic/cyclic calibration and
blind certification are complete.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from .fishbone import FishboneArtifact, build_model, run_gravity
from .structural import _ops

FISHBONE_SPO_SCHEMA_VERSION = "ifb3-pattern-spo-v2-plain-mp"
FISHBONE_LOAD_CONTROL_SCHEMA_VERSION = (
    "ifb3-pattern-spo-v3-limited-modal-load-control-development"
)
FISHBONE_MODE1_TERMINATION_SCHEMA_VERSION = (
    "ifb3-mode1-termination-v1-localized-drift-inner-runtime-guard"
)


@dataclass(frozen=True)
class FishboneSPOResult:
    status: str
    direction: str
    pattern_name: str
    control_node: int
    control_dof: int
    runtime_s: float
    numerical_failure: bool
    physical_postpeak_termination: bool
    arrays: dict[str, np.ndarray]
    summary: dict[str, Any]


def modal_pattern(
    artifact: FishboneArtifact,
    direction: str,
    mode_number: int,
) -> tuple[np.ndarray, int]:
    """Return normalized M*phi pattern and stable modal control floor."""

    normalized = direction.upper()
    targets = (
        artifact.target_modal_x
        if normalized == "X"
        else artifact.target_modal_y
        if normalized == "Y"
        else None
    )
    if targets is None:
        raise ValueError("Fishbone modal SPO direction must be X or Y")
    if not 1 <= int(mode_number) <= min(3, len(targets.mode_shapes)):
        raise ValueError("Fishbone modal SPO mode must be 1, 2 or 3")
    shape = np.asarray(
        targets.mode_shapes[int(mode_number) - 1], dtype=float
    )
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if normalized == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    pattern = masses * shape
    scale = float(np.sum(np.abs(pattern)))
    if scale <= 0.0:
        raise RuntimeError("Fishbone modal SPO pattern is zero")
    pattern /= scale
    control_floor = int(np.argmax(np.abs(shape))) + 1
    return pattern, control_floor


def run_modal_spo(
    artifact: FishboneArtifact,
    *,
    direction: str,
    mode_number: int,
    forced_control_floor: int | None = None,
    pattern_sign: float = 1.0,
    adaptive_control_switching: bool = False,
    control_strategy: str = "displacement",
    load_control_modal_increment_m: float = 0.00025,
    **controls: Any,
) -> FishboneSPOResult:
    """Run one of the first three directional modal pushovers."""

    pattern, control_floor = modal_pattern(
        artifact, direction, mode_number
    )
    sign = float(pattern_sign)
    if sign not in {-1.0, 1.0}:
        raise ValueError("Modal SPO pattern sign must be +1 or -1")
    pattern *= sign
    targets = (
        artifact.target_modal_x
        if direction.upper() == "X"
        else artifact.target_modal_y
    )
    shape = np.asarray(
        targets.mode_shapes[int(mode_number) - 1], dtype=float
    )
    strategy = str(control_strategy).lower()
    if strategy not in {"displacement", "load"}:
        raise ValueError("Modal SPO control strategy must be displacement or load")
    if strategy == "load":
        if forced_control_floor is not None or adaptive_control_switching:
            raise ValueError(
                "Load-controlled modal SPO does not use a control floor"
            )
        modal_increment = float(load_control_modal_increment_m)
        if modal_increment <= 0.0:
            raise ValueError("Load-control modal increment must be positive")
        masses = np.asarray(
            artifact.floor_masses_x_kn_s2_m
            if direction.upper() == "X"
            else artifact.floor_masses_y_kn_s2_m,
            dtype=float,
        )
        period = float(targets.periods_s[int(mode_number) - 1])
        modal_tangent = float(np.sum(np.abs(masses * shape))) * (
            2.0 * math.pi / period
        ) ** 2
        result = run_pattern_spo(
            artifact,
            direction=direction,
            pattern=pattern,
            pattern_name=f"mode_{int(mode_number)}",
            control_strategy="load",
            load_increment_kn=modal_increment * modal_tangent,
            minimum_load_increment_kn=min(1.0e-6, modal_increment)
            * modal_tangent,
            **controls,
        )
        summary = dict(result.summary)
        summary.update(
            {
                "requested_modal_control_floor": control_floor,
                "accepted_control_floor": None,
                "forced_control_floor": None,
                "pattern_sign": sign,
                "adaptive_control_switching": False,
                "control_strategy": "load",
                "load_control_modal_increment_m": modal_increment,
                "target_modal_tangent_kn_m": modal_tangent,
            }
        )
        return replace(result, summary=summary)
    ranked = [
        int(index) + 1
        for index in np.argsort(-np.abs(shape))
        if abs(float(shape[index])) > 1.0e-8
    ]
    if forced_control_floor is not None:
        forced = int(forced_control_floor)
        if not 1 <= forced <= artifact.story_count:
            raise ValueError("Forced modal control floor is outside the model")
        candidate_floors = [forced]
    else:
        candidate_floors = [control_floor]
        candidate_floors.extend(
            floor for floor in ranked if floor != control_floor
        )
    if adaptive_control_switching and forced_control_floor is None:
        effective_shape = sign * shape
        scale = max(float(np.max(np.abs(effective_shape))), 1.0e-12)
        result = run_pattern_spo(
            artifact,
            direction=direction,
            pattern=pattern,
            pattern_name=f"mode_{int(mode_number)}",
            control_floor=control_floor,
            alternate_control_floors=tuple(candidate_floors[1:]),
            control_floor_signs={
                floor: math.copysign(
                    1.0, float(effective_shape[floor - 1])
                )
                for floor in candidate_floors
            },
            control_floor_increment_scales={
                floor: abs(float(effective_shape[floor - 1])) / scale
                for floor in candidate_floors
            },
            **controls,
        )
        summary = dict(result.summary)
        summary.update(
            {
                "requested_modal_control_floor": control_floor,
                "accepted_control_floor": summary.get(
                    "final_control_floor", control_floor
                ),
                "forced_control_floor": None,
                "pattern_sign": sign,
                "adaptive_control_switching": True,
            }
        )
        return replace(result, summary=summary)
    attempts: list[FishboneSPOResult] = []
    for candidate in candidate_floors:
        result = run_pattern_spo(
            artifact,
            direction=direction,
            pattern=pattern,
            pattern_name=f"mode_{int(mode_number)}",
            control_floor=candidate,
            **controls,
        )
        attempts.append(result)
        if not result.numerical_failure:
            summary = dict(result.summary)
            summary.update(
                {
                    "requested_modal_control_floor": control_floor,
                    "accepted_control_floor": candidate,
                    "forced_control_floor": forced_control_floor,
                    "pattern_sign": sign,
                    "adaptive_control_switching": False,
                    "control_floors_attempted": tuple(
                        candidate_floors[index]
                        for index in range(len(attempts))
                    ),
                }
            )
            return replace(result, summary=summary)
    best_index = max(
        range(len(attempts)),
        key=lambda index: (
            attempts[index].summary["peak_generalized_force_kn"],
            attempts[index].summary["step_count"],
        ),
    )
    best = attempts[best_index]
    summary = dict(best.summary)
    summary.update(
        {
            "requested_modal_control_floor": control_floor,
            "accepted_control_floor": candidate_floors[best_index],
            "forced_control_floor": forced_control_floor,
            "pattern_sign": sign,
            "adaptive_control_switching": False,
            "control_floors_attempted": tuple(candidate_floors),
            "all_modal_controls_numerically_failed": True,
        }
    )
    return replace(best, summary=summary)


def run_six_modal_spo_curves(
    artifact: FishboneArtifact,
    **controls: Any,
) -> tuple[FishboneSPOResult, ...]:
    """Return separate Mode 1-3 curves in X and Y for ML provenance."""

    results: list[FishboneSPOResult] = []
    for direction in ("X", "Y"):
        targets = (
            artifact.target_modal_x
            if direction == "X"
            else artifact.target_modal_y
        )
        for mode_number in range(1, min(3, len(targets.mode_shapes)) + 1):
            results.append(
                run_modal_spo(
                    artifact,
                    direction=direction,
                    mode_number=mode_number,
                    **controls,
                )
            )
    return tuple(results)


def _set_static_strategy(
    algorithm: str,
    *,
    test_type: str = "NormDispIncr",
    tolerance: float,
    maximum_iterations: int,
) -> None:
    ops = _ops()
    ops.test(test_type, tolerance, maximum_iterations, 0)
    if algorithm == "NewtonLineSearch":
        ops.algorithm("NewtonLineSearch", 0.8)
    elif algorithm == "KrylovNewton":
        ops.algorithm("KrylovNewton")
    elif algorithm == "ModifiedNewton":
        ops.algorithm("ModifiedNewton", "-initial")
    else:
        ops.algorithm("Newton")


def _configure_static(constraint_handler: str = "Plain") -> None:
    ops = _ops()
    ops.wipeAnalysis()
    if constraint_handler not in {"Plain", "Transformation"}:
        raise ValueError("Unsupported static constraint handler")
    ops.constraints(constraint_handler)
    ops.numberer("RCM")
    ops.system("BandGeneral")
    _set_static_strategy(
        "Newton", tolerance=1.0e-7, maximum_iterations=60
    )
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def _hinge_story_peak_rotations(
    column_hinges: tuple[int, ...],
    beam_hinges: tuple[int, ...],
    story_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ops = _ops()
    if len(column_hinges) % (2 * story_count) != 0:
        raise ValueError("Fishbone column hinge topology is inconsistent")
    if len(beam_hinges) % (4 * story_count) != 0:
        raise ValueError("Fishbone beam hinge topology is inconsistent")
    column_peak = np.zeros(story_count, dtype=float)
    beam_peak = np.zeros(story_count, dtype=float)
    for index, tag in enumerate(column_hinges):
        values = np.asarray(
            ops.eleResponse(tag, "deformation"), dtype=float
        ).reshape(-1)
        if values.size:
            story = (index % (2 * story_count)) // 2
            column_peak[story] = max(
                column_peak[story], float(np.max(np.abs(values)))
            )
    for index, tag in enumerate(beam_hinges):
        values = np.asarray(
            ops.eleResponse(tag, "deformation"), dtype=float
        ).reshape(-1)
        if values.size:
            story = (index % (4 * story_count)) // 4
            beam_peak[story] = max(
                beam_peak[story], float(np.max(np.abs(values)))
            )
    return column_peak, beam_peak


def run_pattern_spo(
    artifact: FishboneArtifact,
    *,
    direction: str,
    pattern: np.ndarray,
    pattern_name: str,
    control_floor: int | None = None,
    displacement_increment_m: float = 0.001,
    minimum_displacement_increment_m: float = 1.0e-5,
    maximum_roof_drift: float = 0.10,
    maximum_story_drift: float = 0.20,
    maximum_steps: int = 20000,
    maximum_runtime_s: float = 300.0,
    postpeak_force_ratio: float = 0.20,
    postpeak_minimum_roof_drift: float = 0.02,
    postpeak_minimum_story_drift: float = 0.02,
    model_builder: Callable[[FishboneArtifact, str], Any] | None = None,
    gravity_runner: Callable[[Any], Any] | None = None,
    constraint_handler: str = "Plain",
    alternate_control_floors: tuple[int, ...] = (),
    control_floor_signs: dict[int, float] | None = None,
    control_floor_increment_scales: dict[int, float] | None = None,
    control_strategy: str = "displacement",
    load_increment_kn: float | None = None,
    minimum_load_increment_kn: float | None = None,
) -> FishboneSPOResult:
    """Run one signed modal/code-pattern pushover."""

    artifact.validate()
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Fishbone SPO direction must be X or Y")
    loads = np.asarray(pattern, dtype=float).reshape(-1)
    if loads.shape != (artifact.story_count,):
        raise ValueError("Fishbone SPO pattern length is inconsistent")
    if np.any(~np.isfinite(loads)) or float(np.max(np.abs(loads))) <= 0.0:
        raise ValueError("Fishbone SPO pattern is invalid")
    if displacement_increment_m <= 0.0:
        raise ValueError("Fishbone SPO increment must be positive")
    if not 0.0 < minimum_displacement_increment_m <= displacement_increment_m:
        raise ValueError("Fishbone minimum SPO increment is invalid")
    if not 0.0 < postpeak_force_ratio < 1.0:
        raise ValueError("Fishbone post-peak force ratio must lie in (0, 1)")
    if postpeak_minimum_roof_drift <= 0.0:
        raise ValueError("Fishbone post-peak roof drift must be positive")
    if postpeak_minimum_story_drift <= 0.0:
        raise ValueError("Fishbone post-peak story drift must be positive")
    if control_floor is not None and not (
        1 <= int(control_floor) <= artifact.story_count
    ):
        raise ValueError("Fishbone SPO control floor is outside the model")
    strategy = str(control_strategy).lower()
    if strategy not in {"displacement", "load"}:
        raise ValueError("Fishbone SPO control strategy must be displacement or load")
    if strategy == "load":
        if load_increment_kn is None or float(load_increment_kn) <= 0.0:
            raise ValueError("Fishbone load-control increment must be positive")
        if minimum_load_increment_kn is None:
            minimum_load_increment_kn = float(load_increment_kn) / 250.0
        if not (
            0.0 < float(minimum_load_increment_kn) <= float(load_increment_kn)
        ):
            raise ValueError("Fishbone minimum load-control increment is invalid")

    info = (
        build_model(artifact)
        if model_builder is None
        else model_builder(artifact, normalized)
    )
    if gravity_runner is None:
        run_gravity()
    else:
        gravity_runner(info)
    ops = _ops()
    dof = 1 if normalized == "X" else 2
    pattern_tag = 901
    series_tag = 901
    ops.timeSeries("Linear", series_tag)
    ops.pattern("Plain", pattern_tag, series_tag)
    for node, value in zip(info.floor_nodes, loads):
        vector = [0.0] * 6
        vector[dof - 1] = float(value)
        ops.load(node, *vector)
    _configure_static(constraint_handler)
    control_index = (
        artifact.story_count - 1
        if control_floor is None
        else int(control_floor) - 1
    )
    control_node = info.floor_nodes[control_index]
    # A user-facing roof-control request retains the traditional positive
    # roof displacement convention.  When a higher-mode control floor is
    # selected explicitly, follow that modal ordinate so the initial
    # generalized load factor is positive and the strongest modal floor does
    # not become a near-zero control DOF.
    control_sign = (
        1.0
        if control_floor is None
        else math.copysign(1.0, float(loads[control_index]))
    )
    primary_floor = control_index + 1
    candidate_floors = [primary_floor]
    candidate_floors.extend(
        int(floor)
        for floor in alternate_control_floors
        if int(floor) != primary_floor
    )
    if any(not 1 <= floor <= artifact.story_count for floor in candidate_floors):
        raise ValueError("Alternate SPO control floor is outside the model")
    floor_signs = {
        floor: (
            math.copysign(1.0, float(loads[floor - 1]))
            if control_floor_signs is None
            else math.copysign(1.0, float(control_floor_signs[floor]))
        )
        for floor in candidate_floors
    }
    floor_scales = {
        floor: (
            1.0
            if control_floor_increment_scales is None
            else max(
                abs(float(control_floor_increment_scales[floor])), 1.0e-4
            )
        )
        for floor in candidate_floors
    }
    active_control_floor = primary_floor
    control_sign = floor_signs[active_control_floor]
    control_floors_used = (
        {active_control_floor} if strategy == "displacement" else set()
    )

    time_values = [0.0]
    load_factors = [0.0]
    floor_displacement = [
        np.zeros(artifact.story_count, dtype=float)
    ]
    story_deformation = [
        np.zeros(artifact.story_count, dtype=float)
    ]
    story_drift_ratio = [
        np.zeros(artifact.story_count, dtype=float)
    ]
    story_shear = [np.zeros(artifact.story_count, dtype=float)]
    roof_displacement = [0.0]
    base_shear = [0.0]
    column_hinge_rotation = [
        np.zeros(artifact.story_count, dtype=float)
    ]
    beam_hinge_rotation = [
        np.zeros(artifact.story_count, dtype=float)
    ]
    control_floor_history = [
        active_control_floor if strategy == "displacement" else 0
    ]

    started = time.perf_counter()
    current_increment = float(displacement_increment_m)
    peak_force = 0.0
    numerical_failure = False
    physical_postpeak = False
    status = "maximum_steps"
    strategies = (
        ("Newton", "NormDispIncr", 1.0e-7, 60, 1.0),
        ("NewtonLineSearch", "NormDispIncr", 1.0e-7, 80, 1.0),
        ("KrylovNewton", "EnergyIncr", 1.0e-7, 80, 0.5),
        ("ModifiedNewton", "EnergyIncr", 1.0e-6, 120, 0.25),
    )
    consecutive_nominal_steps = 0
    current_load_increment = (
        float(load_increment_kn) if load_increment_kn is not None else 0.0
    )
    runtime_exceeded = False

    def beyond_runtime() -> bool:
        return time.perf_counter() - started > maximum_runtime_s

    for _step in range(1, int(maximum_steps) + 1):
        if beyond_runtime():
            status = "runtime_guard"
            numerical_failure = True
            break
        solved = False
        if strategy == "load":
            nominal_load = float(load_increment_kn)
            minimum_load = float(minimum_load_increment_kn)
            trial_load = current_load_increment
            while trial_load + 1.0e-15 >= minimum_load:
                if beyond_runtime():
                    runtime_exceeded = True
                    break
                for (
                    algorithm,
                    test_type,
                    tolerance,
                    maximum_iterations,
                    factor,
                ) in strategies:
                    if beyond_runtime():
                        runtime_exceeded = True
                        break
                    increment = max(minimum_load, trial_load * factor)
                    _set_static_strategy(
                        algorithm,
                        test_type=test_type,
                        tolerance=tolerance,
                        maximum_iterations=maximum_iterations,
                    )
                    ops.integrator("LoadControl", increment)
                    if ops.analyze(1) == 0:
                        if math.isclose(
                            increment,
                            nominal_load,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        ):
                            consecutive_nominal_steps += 1
                        else:
                            consecutive_nominal_steps = 0
                        current_load_increment = (
                            nominal_load
                            if consecutive_nominal_steps >= 2
                            else min(
                                nominal_load,
                                max(increment, minimum_load) * 2.0,
                            )
                        )
                        solved = True
                        break
                    _configure_static(constraint_handler)
                if solved:
                    break
                if runtime_exceeded:
                    break
                trial_load *= 0.5
        else:
            active_scale = floor_scales[active_control_floor]
            control_order = [active_control_floor]
            control_order.extend(
                floor
                for floor in candidate_floors
                if floor != active_control_floor
            )
            for candidate_floor in control_order:
                if beyond_runtime():
                    runtime_exceeded = True
                    break
                candidate_scale = floor_scales[candidate_floor]
                candidate_nominal = displacement_increment_m * candidate_scale
                candidate_minimum = (
                    minimum_displacement_increment_m * candidate_scale
                )
                trial_increment = (
                    current_increment * candidate_scale / active_scale
                )
                candidate_node = info.floor_nodes[candidate_floor - 1]
                candidate_sign = floor_signs[candidate_floor]
                while trial_increment + 1.0e-15 >= candidate_minimum:
                    if beyond_runtime():
                        runtime_exceeded = True
                        break
                    for (
                        algorithm,
                        test_type,
                        tolerance,
                        maximum_iterations,
                        factor,
                    ) in strategies:
                        if beyond_runtime():
                            runtime_exceeded = True
                            break
                        increment = max(
                            candidate_minimum,
                            trial_increment * factor,
                        )
                        _set_static_strategy(
                            algorithm,
                            test_type=test_type,
                            tolerance=tolerance,
                            maximum_iterations=maximum_iterations,
                        )
                        ops.integrator(
                            "DisplacementControl",
                            candidate_node,
                            dof,
                            candidate_sign * increment,
                        )
                        if ops.analyze(1) == 0:
                            if math.isclose(
                                increment,
                                candidate_nominal,
                                rel_tol=0.0,
                                abs_tol=1.0e-15,
                            ):
                                consecutive_nominal_steps += 1
                            else:
                                consecutive_nominal_steps = 0
                            if consecutive_nominal_steps >= 2:
                                current_increment = candidate_nominal
                            else:
                                current_increment = min(
                                    candidate_nominal,
                                    max(increment, candidate_minimum) * 2.0,
                                )
                            active_control_floor = candidate_floor
                            control_node = candidate_node
                            control_sign = candidate_sign
                            control_floors_used.add(candidate_floor)
                            solved = True
                            break
                        _configure_static(constraint_handler)
                    if solved:
                        break
                    if runtime_exceeded:
                        break
                    trial_increment *= 0.5
                if solved:
                    break
                if runtime_exceeded:
                    break
        if runtime_exceeded:
            status = "runtime_guard"
            numerical_failure = True
            break
        if not solved:
            status = "numerical_failure"
            numerical_failure = True
            break

        displacement = np.asarray(
            [
                float(ops.nodeDisp(node, dof))
                for node in info.floor_nodes
            ],
            dtype=float,
        )
        deformation = np.diff(np.concatenate(([0.0], displacement)))
        drift = deformation / np.asarray(
            artifact.story_heights_m, dtype=float
        )
        factor = float(ops.getLoadFactor(pattern_tag))
        force = abs(factor)
        # Plain equalDOF constraints place some support reactions in the
        # constraint equations rather than on the retained base node.  Static
        # equilibrium gives the physical base shear exactly from the applied
        # pattern, including signed higher-mode forces.
        equilibrium_base_shear = factor * float(np.sum(loads))
        column_story_rotation, beam_story_rotation = (
            _hinge_story_peak_rotations(
                info.column_hinge_elements,
                info.beam_hinge_elements,
                artifact.story_count,
            )
        )
        applied_story_shear = factor * np.cumsum(loads[::-1])[::-1]
        time_values.append(float(ops.getTime()))
        load_factors.append(factor)
        floor_displacement.append(displacement)
        story_deformation.append(deformation)
        story_drift_ratio.append(drift)
        story_shear.append(applied_story_shear)
        roof_displacement.append(float(displacement[-1]))
        base_shear.append(abs(equilibrium_base_shear))
        column_hinge_rotation.append(column_story_rotation)
        beam_hinge_rotation.append(beam_story_rotation)
        control_floor_history.append(
            active_control_floor if strategy == "displacement" else 0
        )
        peak_force = max(peak_force, force)

        roof_drift = abs(float(displacement[-1])) / max(
            artifact.floor_elevations_m[-1], 1.0e-12
        )
        max_story = float(np.max(np.abs(drift)))
        if max_story >= maximum_story_drift:
            status = "physical_story_drift_limit"
            physical_postpeak = True
            break
        if (
            peak_force > 0.0
            and force <= postpeak_force_ratio * peak_force
            and (
                roof_drift >= postpeak_minimum_roof_drift
                or max_story >= postpeak_minimum_story_drift
            )
        ):
            status = "physical_postpeak_termination"
            physical_postpeak = True
            break
        if roof_drift >= maximum_roof_drift:
            status = "target_roof_drift_reached"
            break

    arrays = {
        "analysis_time_s": np.asarray(time_values, dtype=float),
        "load_factor": np.asarray(load_factors, dtype=float),
        "floor_displacement_m": np.asarray(
            floor_displacement, dtype=float
        ),
        "story_deformation_m": np.asarray(
            story_deformation, dtype=float
        ),
        "story_drift_ratio": np.asarray(
            story_drift_ratio, dtype=float
        ),
        "story_shear_kn": np.asarray(story_shear, dtype=float),
        "roof_displacement_m": np.asarray(
            roof_displacement, dtype=float
        ),
        "base_shear_kn": np.asarray(base_shear, dtype=float),
        "column_hinge_peak_rotation_rad": np.asarray(
            column_hinge_rotation, dtype=float
        ),
        "beam_hinge_peak_rotation_rad": np.asarray(
            beam_hinge_rotation, dtype=float
        ),
        "control_floor": np.asarray(control_floor_history, dtype=int),
        "pattern": loads,
    }
    absolute_load = np.abs(arrays["load_factor"])
    peak_index = int(np.argmax(absolute_load)) if absolute_load.size else 0
    summary = {
        "schema_version": (
            FISHBONE_LOAD_CONTROL_SCHEMA_VERSION
            if strategy == "load"
            else FISHBONE_SPO_SCHEMA_VERSION
        ),
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "calibration_stage": artifact.calibration_stage,
        "step_count": int(arrays["load_factor"].size - 1),
        "peak_generalized_force_kn": float(
            absolute_load[peak_index] if absolute_load.size else 0.0
        ),
        "peak_roof_displacement_m": float(
            arrays["roof_displacement_m"][peak_index]
            if arrays["roof_displacement_m"].size
            else 0.0
        ),
        "maximum_story_drift_ratio": float(
            np.max(np.abs(arrays["story_drift_ratio"]))
            if arrays["story_drift_ratio"].size
            else 0.0
        ),
        "maximum_column_hinge_rotation_rad": float(
            np.max(arrays["column_hinge_peak_rotation_rad"])
            if arrays["column_hinge_peak_rotation_rad"].size
            else 0.0
        ),
        "maximum_beam_hinge_rotation_rad": float(
            np.max(arrays["beam_hinge_peak_rotation_rad"])
            if arrays["beam_hinge_peak_rotation_rad"].size
            else 0.0
        ),
        "control_floors_used": tuple(sorted(control_floors_used)),
        "final_control_floor": (
            int(active_control_floor) if strategy == "displacement" else None
        ),
        "control_strategy": strategy,
        "nominal_load_increment_kn": (
            float(load_increment_kn) if strategy == "load" else None
        ),
        "postpeak_minimum_story_drift": float(
            postpeak_minimum_story_drift
        ),
        "termination_schema": FISHBONE_MODE1_TERMINATION_SCHEMA_VERSION,
    }
    return FishboneSPOResult(
        status=status,
        direction=normalized,
        pattern_name=str(pattern_name),
        control_node=control_node,
        control_dof=dof,
        runtime_s=float(time.perf_counter() - started),
        numerical_failure=numerical_failure,
        physical_postpeak_termination=physical_postpeak,
        arrays=arrays,
        summary=summary,
    )
