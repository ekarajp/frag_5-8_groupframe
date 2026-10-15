"""Story-imposed-displacement evidence for Fishbone-v2 condensation.

Two boundary conditions are supported:

``single_floor``
    The selected floor moves while every other floor is restrained.  This is
    the imposed-displacement test in Jamsek and Dolsek (2021), Equation (6).

``story_isolation``
    Every floor above the selected story is tied to the control floor while
    floors below it are restrained.  Only the selected story deforms, giving
    a direct story shear--story deformation relation for calibration.

No ground-motion record is used in either test.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from .fishbone import (
    FishboneArtifact,
    FishboneModelInfo,
    build_model as build_fishbone_model,
    run_gravity as run_fishbone_gravity,
)
from .structural import (
    ModelInfo,
    _ops,
    build_model as build_full3d_model,
    run_gravity as run_full3d_gravity,
)

FISHBONE_IMPOSED_DISPLACEMENT_SCHEMA_VERSION = (
    "ifb3-story-imposed-displacement-eq6-and-isolation-v3-full3d-prescribed-sp"
)
FISHBONE_IMPOSED_COMPARISON_SCHEMA_VERSION = (
    "ifb3-story-imposed-common-completed-cycles-v1"
)


def imposed_displacement_targets(
    story_height_m: float,
    drift_protocol: tuple[float, ...],
    cycles_per_level: int,
) -> tuple[float, ...]:
    if story_height_m <= 0.0 or cycles_per_level < 1:
        raise ValueError("Imposed-displacement protocol controls are invalid")
    targets: list[float] = []
    for drift in drift_protocol:
        if not math.isfinite(drift) or drift <= 0.0:
            raise ValueError("Imposed story drifts must be positive")
        amplitude = float(drift) * story_height_m
        for _cycle in range(int(cycles_per_level)):
            targets.extend((amplitude, -amplitude))
    targets.append(0.0)
    return tuple(targets)


def _completed_target(
    arrays: dict[str, np.ndarray],
    status: str,
) -> int:
    values = np.asarray(
        arrays["protocol_target_index"], dtype=int
    ).reshape(-1)
    last = int(np.max(values)) if values.size else 0
    return last if status == "protocol_complete" else max(last - 1, 0)


def _path_energy(force: np.ndarray, deformation: np.ndarray) -> float:
    if force.size < 2:
        return 0.0
    return abs(
        float(
            np.sum(
                0.5 * (force[1:] + force[:-1])
                * np.diff(deformation)
            )
        )
    )


def _resample_protocol(
    arrays: dict[str, np.ndarray],
    story_index: int,
    target: int,
    *,
    points_per_target: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(
        arrays["protocol_target_index"], dtype=int
    ).reshape(-1)
    deformation = np.asarray(
        arrays["story_deformation_m"], dtype=float
    )[:, story_index]
    shear = np.asarray(
        arrays["story_shear_kn"], dtype=float
    )[:, story_index]
    sample = np.linspace(0.0, 1.0, int(points_per_target))
    deformation_rows: list[np.ndarray] = []
    shear_rows: list[np.ndarray] = []
    for index in range(1, target + 1):
        mask = indices == index
        if not np.any(mask):
            continue
        progress = np.linspace(0.0, 1.0, int(np.sum(mask)))
        deformation_rows.append(np.interp(sample, progress, deformation[mask]))
        shear_rows.append(np.interp(sample, progress, shear[mask]))
    if not deformation_rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.concatenate(deformation_rows), np.concatenate(shear_rows)


def compare_imposed_histories(
    full3d_arrays: dict[str, np.ndarray],
    fishbone_arrays: dict[str, np.ndarray],
    *,
    selected_story: int,
    full3d_status: str = "protocol_complete",
    fishbone_status: str = "protocol_complete",
) -> dict[str, Any]:
    """Compare the selected story on common fully traversed targets."""

    story_index = int(selected_story) - 1
    if story_index < 0:
        raise ValueError("Selected imposed-displacement story is invalid")
    full_completed = _completed_target(full3d_arrays, full3d_status)
    fish_completed = _completed_target(fishbone_arrays, fishbone_status)
    common = min(full_completed, fish_completed)
    if common < 1:
        raise ValueError("No common completed imposed-displacement target")
    full_deformation, full_shear = _resample_protocol(
        full3d_arrays, story_index, common
    )
    fish_deformation, fish_shear = _resample_protocol(
        fishbone_arrays, story_index, common
    )
    if (
        full_deformation.size == 0
        or full_deformation.shape != fish_deformation.shape
        or full_shear.shape != fish_shear.shape
    ):
        raise ValueError("Imposed-displacement histories cannot be aligned")
    peak_full = float(np.max(np.abs(full_shear)))
    peak_fish = float(np.max(np.abs(fish_shear)))
    energy_full = _path_energy(full_shear, full_deformation)
    energy_fish = _path_energy(fish_shear, fish_deformation)
    force_scale = max(peak_full, 1.0e-12)
    loop_nrmse = float(
        np.sqrt(np.mean((fish_shear - full_shear) ** 2)) / force_scale
    )

    def relative(observed: float, reference: float) -> float:
        return float(
            (observed - reference) / max(abs(reference), 1.0e-12)
        )

    # The first target is kept intentionally small; its secant is a robust
    # initial-stiffness proxy even when the force history has a few repeated
    # zero points.
    first_indices = np.asarray(
        full3d_arrays["protocol_target_index"], dtype=int
    ) == 1
    fish_first_indices = np.asarray(
        fishbone_arrays["protocol_target_index"], dtype=int
    ) == 1
    full_first_delta = np.asarray(
        full3d_arrays["story_deformation_m"], dtype=float
    )[first_indices, story_index]
    full_first_shear = np.asarray(
        full3d_arrays["story_shear_kn"], dtype=float
    )[first_indices, story_index]
    fish_first_delta = np.asarray(
        fishbone_arrays["story_deformation_m"], dtype=float
    )[fish_first_indices, story_index]
    fish_first_shear = np.asarray(
        fishbone_arrays["story_shear_kn"], dtype=float
    )[fish_first_indices, story_index]
    full_initial = float(
        full_first_shear[-1] / full_first_delta[-1]
    )
    fish_initial = float(
        fish_first_shear[-1] / fish_first_delta[-1]
    )
    return {
        "schema_version": FISHBONE_IMPOSED_COMPARISON_SCHEMA_VERSION,
        "selected_story": int(selected_story),
        "full3d_completed_target_index": full_completed,
        "fishbone_completed_target_index": fish_completed,
        "common_completed_target_index": common,
        "fishbone_target_coverage": float(
            fish_completed / max(full_completed, 1)
        ),
        "full3d_initial_secant_stiffness_kn_m": full_initial,
        "fishbone_initial_secant_stiffness_kn_m": fish_initial,
        "initial_stiffness_signed_relative_error": relative(
            fish_initial, full_initial
        ),
        "full3d_peak_abs_story_shear_kn": peak_full,
        "fishbone_peak_abs_story_shear_kn": peak_fish,
        "peak_story_shear_signed_relative_error": relative(
            peak_fish, peak_full
        ),
        "full3d_hysteretic_energy_kn_m": energy_full,
        "fishbone_hysteretic_energy_kn_m": energy_fish,
        "energy_signed_relative_error": relative(
            energy_fish, energy_full
        ),
        "story_shear_loop_nrmse": loop_nrmse,
    }


def _add_boundary_conditions(
    floor_nodes: tuple[int, ...],
    *,
    selected_story: int,
    dof: int,
    boundary_condition: str,
    restrain_torsion: bool,
    prescribed_control: bool,
) -> int:
    ops = _ops()
    if not 1 <= selected_story <= len(floor_nodes):
        raise ValueError("Selected imposed-displacement story is invalid")
    if boundary_condition not in {"single_floor", "story_isolation"}:
        raise ValueError("Unsupported imposed-displacement boundary condition")
    control = floor_nodes[selected_story - 1]
    other_dof = 2 if dof == 1 else 1
    # A Full 3D rigid-diaphragm master must be moved through a prescribed
    # displacement.  DisplacementControl can move the retained master without
    # propagating that motion to its constrained physical floor nodes under
    # OpenSees' Transformation handler.  A linearly scaled SP constraint
    # preserves the diaphragm kinematics.  Fishbone keeps DisplacementControl
    # with the Plain handler because its retained-node topology only supports
    # homogeneous SP constraints.
    ops.timeSeries("Linear" if prescribed_control else "Constant", 971)
    ops.pattern("Plain", 971, 971)
    for node in floor_nodes:
        # Calibration is intentionally uniaxial and torsion-free because the
        # current Fishbone certification scope excludes torsion.
        ops.sp(node, other_dof, 0.0)
        if restrain_torsion:
            ops.sp(node, 6, 0.0)
    if boundary_condition == "single_floor":
        for node in floor_nodes:
            if node != control:
                ops.sp(node, dof, 0.0)
    else:
        for index, node in enumerate(floor_nodes, start=1):
            if index < selected_story:
                ops.sp(node, dof, 0.0)
            elif index > selected_story:
                ops.equalDOF(control, node, dof)
    if prescribed_control:
        ops.sp(control, dof, 1.0)
    else:
        load = [0.0] * 6
        load[dof - 1] = 1.0
        ops.load(control, *load)
    return control


def _configure_static(constraint_handler: str) -> None:
    ops = _ops()
    ops.wipeAnalysis()
    ops.constraints(constraint_handler)
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-7, 80, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.0)
    ops.analysis("Static")


def _signed_step(
    control_node: int,
    dof: int,
    requested_increment: float,
    minimum_increment: float,
    constraint_handler: str,
    prescribed_control: bool,
    recovery_policy: str = "legacy_v1",
) -> tuple[bool, float, int]:
    ops = _ops()
    attempt = float(requested_increment)
    level = 0
    if recovery_policy == "legacy_v1":
        strategies = (
            ("Newton", "NormDispIncr", 1.0e-7, 80),
            ("NewtonLineSearch", "NormDispIncr", 1.0e-7, 100),
            ("KrylovNewton", "EnergyIncr", 1.0e-7, 100),
            ("ModifiedNewton", "EnergyIncr", 1.0e-6, 140),
        )
    elif recovery_policy == "robust_v2":
        # Numerical-only recovery: no material, geometry, loading, damping or
        # convergence-tolerance acceptance criterion is changed.  Additional
        # equilibrium algorithms are tried before the increment is bisected.
        strategies = (
            ("Newton", "NormDispIncr", 1.0e-8, 160),
            ("NewtonLineSearch", "NormDispIncr", 1.0e-8, 200),
            ("KrylovNewton", "EnergyIncr", 1.0e-8, 200),
            ("Broyden", "NormUnbalance", 1.0e-7, 240),
            ("SecantNewton", "EnergyIncr", 1.0e-8, 240),
            ("ModifiedNewton", "EnergyIncr", 1.0e-7, 260),
        )
    elif recovery_policy == "safe_dbe_v3":
        # Candidate-v5F numerical recovery for displacement-based elements.
        # Development diagnostics demonstrated stable gravity, positive
        # eigenvalues and Newton convergence through the nominal 1 mm step,
        # while robust_v2 entered Broyden/SecantNewton and crashed OpenSees.
        # Keep the physical model and convergence thresholds unchanged, omit
        # those two unsafe quasi-Newton paths, and bisect after conservative
        # Newton-family attempts fail.
        strategies = (
            ("Newton", "NormDispIncr", 1.0e-8, 160),
            ("NewtonLineSearch", "NormDispIncr", 1.0e-8, 200),
            ("KrylovNewton", "EnergyIncr", 1.0e-8, 200),
        )
    else:
        raise ValueError(f"Unsupported imposed-history recovery policy: {recovery_policy}")
    while abs(attempt) + 1.0e-15 >= minimum_increment:
        for algorithm, test_type, tolerance, iterations in strategies:
            ops.test(test_type, tolerance, iterations, 0)
            if algorithm == "NewtonLineSearch":
                ops.algorithm("NewtonLineSearch", 0.8)
            elif algorithm == "KrylovNewton":
                ops.algorithm("KrylovNewton")
            elif algorithm == "ModifiedNewton":
                ops.algorithm("ModifiedNewton", "-initial")
            elif algorithm == "Broyden":
                ops.algorithm("Broyden", 8)
            elif algorithm == "SecantNewton":
                ops.algorithm("SecantNewton")
            else:
                ops.algorithm("Newton")
            if prescribed_control:
                # The reference SP value is 1.0 m, so a load-factor increment
                # is numerically the requested displacement increment in m.
                ops.integrator("LoadControl", attempt)
            else:
                ops.integrator(
                    "DisplacementControl",
                    control_node,
                    dof,
                    attempt,
                )
            if int(ops.analyze(1)) == 0:
                return True, attempt, level
            # Do not reuse failed trial vectors with another algorithm.
            _configure_static(constraint_handler)
        attempt *= 0.5
        level += 1
    return False, attempt, level


def _full3d_story_shear(
    info: ModelInfo,
    dof: int,
    _load_factor: float,
) -> np.ndarray:
    ops = _ops()
    return np.asarray(
        [
            -sum(
                float(ops.eleForce(tag)[dof - 1])
                for tag in tags
            )
            for tags in info.column_element_tags_by_storey
        ],
        dtype=float,
    )


def _fishbone_story_shear(
    info: FishboneModelInfo,
    dof: int,
    _load_factor: float,
) -> np.ndarray:
    ops = _ops()
    forces = np.asarray(
        [
            -float(ops.eleForce(tag)[dof - 1])
            for tag in info.column_elements
        ],
        dtype=float,
    )
    story_count = len(info.story_heights_m)
    if forces.size % story_count:
        raise ValueError("Fishbone column topology is inconsistent")
    return forces.reshape(-1, story_count).sum(axis=0)


def _grouped_frame_story_shear(
    info: FishboneModelInfo,
    dof: int,
    _load_factor: float,
) -> np.ndarray:
    """Aggregate grouped-frame column forces in story-major tag order."""

    ops = _ops()
    story_count = len(info.story_heights_m)
    forces = np.asarray(
        [-float(ops.eleForce(tag)[dof - 1]) for tag in info.column_elements],
        dtype=float,
    )
    if forces.size % story_count:
        raise ValueError("Grouped-frame column topology is inconsistent")
    return forces.reshape(story_count, -1).sum(axis=1)


def _run_imposed_history(
    *,
    model_kind: str,
    building_id: str,
    floor_nodes: tuple[int, ...],
    story_heights_m: tuple[float, ...],
    selected_story: int,
    direction: str,
    boundary_condition: str,
    story_shear: Callable[[int, float], np.ndarray],
    constraint_handler: str,
    restrain_torsion: bool,
    prescribed_control: bool,
    drift_protocol: tuple[float, ...],
    cycles_per_level: int,
    displacement_increment_m: float,
    minimum_displacement_increment_m: float,
    maximum_steps_per_target: int,
    maximum_runtime_s: float,
    recovery_policy: str = "legacy_v1",
    terminate_at_absolute_drift_ratio: float | None = None,
) -> dict[str, Any]:
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Imposed-displacement direction must be X or Y")
    if not (
        displacement_increment_m > 0.0
        and 0.0 < minimum_displacement_increment_m
        <= displacement_increment_m
        and maximum_steps_per_target > 0
        and maximum_runtime_s > 0.0
    ):
        raise ValueError("Imposed-displacement solver controls are invalid")
    if (
        terminate_at_absolute_drift_ratio is not None
        and (
            not math.isfinite(terminate_at_absolute_drift_ratio)
            or terminate_at_absolute_drift_ratio <= 0.0
        )
    ):
        raise ValueError("Imposed-displacement endpoint drift is invalid")
    dof = 1 if normalized == "X" else 2
    control = _add_boundary_conditions(
        floor_nodes,
        selected_story=selected_story,
        dof=dof,
        boundary_condition=boundary_condition,
        restrain_torsion=restrain_torsion,
        prescribed_control=prescribed_control,
    )
    _configure_static(constraint_handler)
    targets = imposed_displacement_targets(
        story_heights_m[selected_story - 1],
        drift_protocol,
        cycles_per_level,
    )
    histories: dict[str, list[Any]] = {
        "analysis_step": [0.0],
        "protocol_target_index": [0.0],
        "load_factor": [0.0],
        "floor_displacement_m": [
            np.zeros(len(floor_nodes), dtype=float)
        ],
        "story_deformation_m": [
            np.zeros(len(floor_nodes), dtype=float)
        ],
        "story_drift_ratio": [
            np.zeros(len(floor_nodes), dtype=float)
        ],
        "story_shear_kn": [
            np.zeros(len(floor_nodes), dtype=float)
        ],
        "control_displacement_m": [0.0],
    }
    ops = _ops()
    started = time.perf_counter()
    status = "protocol_complete"
    accepted_steps = 0
    maximum_recovery_level = 0
    for target_index, target in enumerate(targets, start=1):
        for _local_step in range(int(maximum_steps_per_target)):
            if time.perf_counter() - started > maximum_runtime_s:
                status = "runtime_guard"
                break
            current = float(ops.nodeDisp(control, dof))
            remaining = float(target) - current
            if abs(remaining) <= minimum_displacement_increment_m:
                break
            requested = math.copysign(
                min(abs(remaining), displacement_increment_m),
                remaining,
            )
            solved, _accepted, level = _signed_step(
                control,
                dof,
                requested,
                minimum_displacement_increment_m,
                constraint_handler,
                prescribed_control,
                recovery_policy,
            )
            maximum_recovery_level = max(maximum_recovery_level, level)
            if not solved:
                status = "numerical_failure"
                break
            accepted_steps += 1
            floor = np.asarray(
                [ops.nodeDisp(node, dof) for node in floor_nodes],
                dtype=float,
            )
            deformation = np.diff(np.concatenate(([0.0], floor)))
            factor = float(ops.getLoadFactor(971))
            histories["analysis_step"].append(float(accepted_steps))
            histories["protocol_target_index"].append(float(target_index))
            histories["load_factor"].append(factor)
            histories["floor_displacement_m"].append(floor)
            histories["story_deformation_m"].append(deformation)
            histories["story_drift_ratio"].append(
                deformation / np.asarray(story_heights_m, dtype=float)
            )
            histories["story_shear_kn"].append(story_shear(dof, factor))
            histories["control_displacement_m"].append(
                float(floor[selected_story - 1])
            )
            if (
                terminate_at_absolute_drift_ratio is not None
                and abs(
                    float(histories["story_drift_ratio"][-1][selected_story - 1])
                )
                + 1.0e-12
                >= terminate_at_absolute_drift_ratio
            ):
                status = "drift_endpoint_reached"
                break
        else:
            status = "maximum_steps_per_target"
        if status != "protocol_complete":
            break
    arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in histories.items()
    }
    return {
        "schema_version": FISHBONE_IMPOSED_DISPLACEMENT_SCHEMA_VERSION,
        "model_kind": model_kind,
        "building_id": building_id,
        "direction": normalized,
        "selected_story": int(selected_story),
        "boundary_condition": boundary_condition,
        "status": status,
        "step_count": accepted_steps,
        "completed_target_index": int(
            arrays["protocol_target_index"][-1]
        ),
        "maximum_recovery_level": maximum_recovery_level,
        "solver_recovery_policy": recovery_policy,
        "maximum_selected_story_drift_ratio": float(
            np.max(
                np.abs(
                    arrays["story_drift_ratio"][
                        :, selected_story - 1
                    ]
                )
            )
        ),
        "terminate_at_absolute_drift_ratio": (
            float(terminate_at_absolute_drift_ratio)
            if terminate_at_absolute_drift_ratio is not None
            else None
        ),
        "scientific_endpoint_reached": bool(
            terminate_at_absolute_drift_ratio is not None
            and float(
                np.max(
                    np.abs(
                        arrays["story_drift_ratio"][:, selected_story - 1]
                    )
                )
            )
            + 1.0e-12
            >= terminate_at_absolute_drift_ratio
        ),
        "runtime_s": float(time.perf_counter() - started),
        "_history_arrays": arrays,
    }


def run_fishbone_imposed_history(
    artifact: FishboneArtifact,
    *,
    selected_story: int,
    direction: str,
    boundary_condition: str = "story_isolation",
    drift_protocol: tuple[float, ...] = (
        0.0025,
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.040,
    ),
    cycles_per_level: int = 1,
    displacement_increment_m: float = 0.001,
    minimum_displacement_increment_m: float = 1.0e-6,
    maximum_steps_per_target: int = 10000,
    maximum_runtime_s: float = 120.0,
) -> dict[str, Any]:
    artifact.validate()
    info = build_fishbone_model(artifact)
    run_fishbone_gravity()
    return _run_imposed_history(
        model_kind="Fishbone-v2",
        building_id=artifact.building_id,
        floor_nodes=info.floor_nodes,
        story_heights_m=artifact.story_heights_m,
        selected_story=selected_story,
        direction=direction,
        boundary_condition=boundary_condition,
        story_shear=lambda dof, factor: _fishbone_story_shear(
            info, dof, factor
        ),
        constraint_handler="Plain",
        restrain_torsion=False,
        prescribed_control=False,
        drift_protocol=drift_protocol,
        cycles_per_level=cycles_per_level,
        displacement_increment_m=displacement_increment_m,
        minimum_displacement_increment_m=minimum_displacement_increment_m,
        maximum_steps_per_target=maximum_steps_per_target,
        maximum_runtime_s=maximum_runtime_s,
    )


def run_grouped_frame_imposed_history(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    *,
    selected_story: int,
    direction: str,
    boundary_condition: str = "story_isolation",
    drift_protocol: tuple[float, ...] = (
        0.0025,
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.040,
    ),
    cycles_per_level: int = 1,
    displacement_increment_m: float = 0.001,
    minimum_displacement_increment_m: float = 1.0e-6,
    maximum_steps_per_target: int = 10000,
    maximum_runtime_s: float = 900.0,
    recovery_policy: str = "legacy_v1",
    terminate_at_absolute_drift_ratio: float | None = None,
) -> dict[str, Any]:
    """Run a prescribed story test on an already-gravitated grouped frame.

    The grouped model has diaphragm masters with physical frame nodes attached
    through equalDOF constraints, so it follows the Full3D prescribed-SP route
    rather than the simple spring-model DisplacementControl route.
    """

    artifact.validate()
    if tuple(info.story_heights_m) != tuple(artifact.story_heights_m):
        raise ValueError("Grouped-frame and artifact story geometry differ")
    return _run_imposed_history(
        model_kind="Fishbone-v5-grouped-frame",
        building_id=artifact.building_id,
        floor_nodes=info.floor_nodes,
        story_heights_m=artifact.story_heights_m,
        selected_story=selected_story,
        direction=direction,
        boundary_condition=boundary_condition,
        story_shear=lambda dof, factor: _grouped_frame_story_shear(
            info, dof, factor
        ),
        constraint_handler="Transformation",
        restrain_torsion=False,
        prescribed_control=True,
        drift_protocol=drift_protocol,
        cycles_per_level=cycles_per_level,
        displacement_increment_m=displacement_increment_m,
        minimum_displacement_increment_m=minimum_displacement_increment_m,
        maximum_steps_per_target=maximum_steps_per_target,
        maximum_runtime_s=maximum_runtime_s,
        recovery_policy=recovery_policy,
        terminate_at_absolute_drift_ratio=terminate_at_absolute_drift_ratio,
    )


def run_full3d_imposed_history(
    building: dict[str, Any],
    config: dict[str, Any],
    *,
    selected_story: int,
    direction: str,
    boundary_condition: str = "story_isolation",
    drift_protocol: tuple[float, ...] = (
        0.0025,
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.040,
    ),
    cycles_per_level: int = 1,
    displacement_increment_m: float = 0.002,
    minimum_displacement_increment_m: float = 0.00003125,
    maximum_steps_per_target: int = 10000,
    maximum_runtime_s: float = 3600.0,
) -> dict[str, Any]:
    """Run one isolated Full 3D quasi-static evidence case."""

    info = build_full3d_model(building, config)
    gravity_error = run_full3d_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"{building['building_id']}: gravity error exceeds 1%"
        )
    heights = tuple(
        float(info.floor_elevations_m[index])
        - (
            float(info.floor_elevations_m[index - 1])
            if index
            else 0.0
        )
        for index in range(len(info.floor_elevations_m))
    )
    return _run_imposed_history(
        model_kind="Full3D",
        building_id=str(building["building_id"]),
        floor_nodes=info.master_nodes,
        story_heights_m=heights,
        selected_story=selected_story,
        direction=direction,
        boundary_condition=boundary_condition,
        story_shear=lambda dof, factor: _full3d_story_shear(
            info, dof, factor
        ),
        constraint_handler="Transformation",
        restrain_torsion=True,
        prescribed_control=True,
        drift_protocol=drift_protocol,
        cycles_per_level=cycles_per_level,
        displacement_increment_m=displacement_increment_m,
        minimum_displacement_increment_m=minimum_displacement_increment_m,
        maximum_steps_per_target=maximum_steps_per_target,
        maximum_runtime_s=maximum_runtime_s,
    )
