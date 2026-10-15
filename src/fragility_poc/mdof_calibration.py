"""Full-3D-to-MDOF elastic and nonlinear calibration workflow."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize

from .db import connect, initialize, transaction, upsert_many
from .io_utils import read_json, stable_hash
from .mdof import (
    DirectionalReduction,
    MDOFArtifact,
    StorySpringParameters,
    artifact_path,
    mdof_config,
    modal_assurance_criterion,
    save_artifact,
    shear_stiffness_matrix,
)
from .structural import (
    ModalResult,
    ModelInfo,
    _ops,
    build_model as build_full3d_model,
    modal_properties,
    run_gravity,
)

MDOF_CALIBRATION_SCHEMA_VERSION = (
    "full3d-multiple-pattern-story-condensation-v6-first-three-neutral-cracking"
)


def calibration_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(mdof_config(config).get("calibration", {}))
    settings.setdefault("elastic_probe_roof_drift", 1.0e-5)
    settings.setdefault("nonlinear_maximum_roof_drift", 0.10)
    settings.setdefault(
        "nonlinear_displacement_increment_m",
        float(config["model"]["spo_displacement_step_m"]),
    )
    settings.setdefault(
        "nonlinear_minimum_displacement_increment_m",
        float(config["model"].get("spo_min_displacement_step_m", 1.0e-5)),
    )
    settings.setdefault("maximum_static_steps", 12000)
    settings.setdefault("maximum_static_runtime_s", 900.0)
    settings.setdefault("higher_mode_minimum_mass_ratio", 0.05)
    settings.setdefault("nonlinear_drift_bin_count", 24)
    settings.setdefault("postpeak_minimum_force_drop_ratio", 0.10)
    settings.setdefault("postpeak_minimum_drift_increment", 0.002)
    settings.setdefault("censored_minimum_capping_drift", 0.01)
    settings.setdefault("censored_minimum_residual_drift", 0.06)
    settings.setdefault("censored_minimum_ultimate_drift", 0.08)
    settings.setdefault("censored_capping_deformation_factor", 1.15)
    settings.setdefault("censored_capping_strength_factor", 1.05)
    settings.setdefault("censored_residual_strength_ratio", 0.20)
    settings.setdefault("higher_mode_backbone_compatibility_nrmse", 0.05)
    settings.setdefault("higher_mode_backbone_minimum_weight_factor", 0.10)
    settings.setdefault("higher_mode_minimum_compatibility_drift", 0.001)
    settings.setdefault("cracking_fit_minimum_improvement_ratio", 0.05)
    settings.setdefault("cracking_maximum_drift", 0.01)
    settings.setdefault("cracking_maximum_residual_force_fraction", 0.90)
    settings.setdefault("cracking_regularization_enabled", False)
    settings.setdefault("first_story_cracking_deformation_factor", 1.0)
    settings.setdefault("upper_story_cracking_regularization_start", 0.50)
    settings.setdefault("upper_story_cracking_branch_decay", 0.0)
    settings.setdefault(
        "pattern_weights",
        {
            "first_mode": 0.40,
            "uniform": 0.20,
            "triangular": 0.20,
            "higher_modes_total": 0.20,
        },
    )
    settings.setdefault(
        "elastic_objective_weights",
        {
            "drift_profiles": 0.45,
            "first_mode": 0.35,
            "higher_modes": 0.15,
            "regularization": 0.05,
        },
    )
    settings.setdefault(
        "cyclic_defaults",
        {
            # A large-energy neutral prior is used only until the locked
            # development-building cyclic regression replaces it.  The former
            # value of 1.0 imposed strong deterioration before any Full 3D
            # cyclic evidence existed.
            "lambda_s": 20.0,
            "lambda_c": 20.0,
            "lambda_a": 20.0,
            "lambda_k": 20.0,
            "c_s": 1.0,
            "c_c": 1.0,
            "c_a": 1.0,
            "c_k": 1.0,
        },
    )
    settings.setdefault("raw_curve_subdir", "mdof/calibration_curves")
    return settings


def calibration_signature(
    building: dict[str, Any],
    config: dict[str, Any],
) -> str:
    # Modal-SPO continuation/guard controls do not change masses, elastic
    # fitting, story backbones, or deterioration calibration.  Excluding them
    # prevents a reporting-solver setting from invalidating every calibrated
    # artifact and all cached NLTHA evidence.
    artifact_mdof_config = {
        key: value
        for key, value in mdof_config(config).items()
        if key
        not in {
            "modal_spo_maximum_steps",
            "modal_spo_maximum_runtime_s",
            "modal_spo_near_zero_tangent_ratio",
        }
    }
    return stable_hash(
        {
            "schema_version": MDOF_CALIBRATION_SCHEMA_VERSION,
            "building_id": building["building_id"],
            "model_hash": building.get("model_hash"),
            "model_config": config["model"],
            "mdof_config": artifact_mdof_config,
        }
    )


def _load_buildings(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    initialize(config["database_path"])
    parameters: list[Any] = []
    where = "valid=1 AND selected=1"
    if building_ids:
        marks = ",".join("?" for _ in building_ids)
        where += f" AND building_id IN ({marks})"
        parameters.extend(building_ids)
    query = (
        f"SELECT * FROM building_catalog WHERE {where} ORDER BY queue_rank"
    )
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    with connect(config["database_path"]) as connection:
        return [
            dict(row) for row in connection.execute(query, parameters)
        ]


def _directional_modal_targets(
    modal: ModalResult,
    direction: str,
    story_count: int,
) -> dict[str, Any]:
    normalized = direction.upper()
    if normalized == "X":
        shapes = modal.normalized_mode_shapes_x
        ratios = modal.x_effective_mass_ratios
        participation = modal.x_participation_factors
    elif normalized == "Y":
        shapes = modal.normalized_mode_shapes_y
        ratios = modal.y_effective_mass_ratios
        participation = modal.y_participation_factors
    else:
        raise ValueError("Calibration direction must be X or Y")
    if not shapes or not ratios:
        raise RuntimeError("Full 3D modal result lacks directional mode data")
    candidates = [
        index
        for index, (shape, ratio) in enumerate(zip(shapes, ratios))
        if float(ratio) > 1.0e-6
        and float(np.max(np.abs(shape))) > 1.0e-8
    ]
    # Eigenvalues are returned from lowest frequency upward, so the period
    # tuple is already ordered from the first mode toward higher modes.
    candidates.sort(key=lambda index: -float(modal.periods_s[index]))
    selected: list[int] = []
    for index in candidates:
        period = float(modal.periods_s[index])
        if any(
            math.isclose(
                period,
                float(modal.periods_s[prior]),
                rel_tol=1.0e-7,
                abs_tol=1.0e-9,
            )
            for prior in selected
        ):
            # Repeated X/Y eigenspaces contain one directional target, not two
            # same-frequency modes in the same reduced direction.
            continue
        selected.append(index)
        if len(selected) >= min(3, story_count):
            break
    if len(selected) < min(3, story_count):
        for index in candidates:
            if index not in selected:
                selected.append(index)
            if len(selected) >= min(3, story_count):
                break
    if not selected:
        raise RuntimeError(f"No translational {normalized} modes were found")
    return {
        "global_mode_indices": tuple(index + 1 for index in selected),
        "periods_s": tuple(float(modal.periods_s[index]) for index in selected),
        "mode_shapes": tuple(
            tuple(float(value) for value in shapes[index])
            for index in selected
        ),
        "effective_mass_ratios": tuple(
            float(ratios[index]) for index in selected
        ),
        "participation_factors": tuple(
            float(participation[index]) for index in selected
        ),
    }


def _calibration_patterns(
    targets: dict[str, Any],
    masses: np.ndarray,
    elevations: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    patterns: dict[str, np.ndarray] = {}

    def normalized(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if float(np.sum(vector)) < 0.0:
            vector = -vector
        scale = float(np.sum(np.abs(vector)))
        if scale <= 0.0:
            raise ValueError("A calibration load pattern is zero")
        return vector / scale

    first_shape = np.asarray(targets["mode_shapes"][0], dtype=float)
    patterns["first_mode"] = normalized(masses * first_shape)
    patterns["uniform"] = normalized(masses)
    patterns["triangular"] = normalized(masses * elevations)
    for index in range(1, len(targets["mode_shapes"])):
        # Preserve the first three translational modes even when a higher
        # mode has small effective mass.  The mass ratio is retained as an ML
        # feature and is a better measure of importance than silently
        # deleting a requested modal SPO from some building archetypes.
        pattern_name = f"mode_{index + 1}"
        patterns[pattern_name] = normalized(
            masses
            * np.asarray(targets["mode_shapes"][index], dtype=float)
        )
    return patterns


def _story_displacements(
    info: ModelInfo,
    dof: int,
) -> tuple[np.ndarray, np.ndarray]:
    ops = _ops()
    floor = np.asarray(
        [float(ops.nodeDisp(node, dof)) for node in info.master_nodes],
        dtype=float,
    )
    delta = np.diff(np.concatenate(([0.0], floor)))
    return floor, delta


def _story_cut_shears(
    info: ModelInfo,
    dof: int,
    *,
    load_factor: float,
    absolute: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Return story cut shears and equilibrium errors from column end forces."""
    ops = _ops()
    raw = []
    for tags in info.column_element_tags_by_storey:
        force = sum(
            float(ops.eleForce(tag)[dof - 1])
            for tag in tags
        )
        raw.append(force)
    magnitude = np.abs(np.asarray(raw, dtype=float))
    story = (
        magnitude
        if absolute
        else math.copysign(1.0, load_factor or 1.0) * magnitude
    )
    ops.reactions()
    base_reaction = abs(
        sum(float(ops.nodeReaction(tag, dof)) for tag in info.base_nodes)
    )
    cut_base = float(story[0])
    equilibrium_error = abs(cut_base - base_reaction) / max(
        base_reaction, 1.0
    )
    signed_base = math.copysign(base_reaction, load_factor or 1.0)
    return story, signed_base, float(equilibrium_error)


def _configure_static() -> None:
    ops = _ops()
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-7, 60, 0)
    ops.algorithm("Newton")
    # OpenSees creates a default integrator if analysis() is called without
    # one.  Declare a zero load step explicitly so the calibration log stays
    # warning-free; displacement control replaces it before every real step.
    ops.integrator("LoadControl", 0.0)
    ops.analysis("Static")


def _analyze_displacement_step(
    control_node: int,
    dof: int,
    requested_increment: float,
    minimum_increment: float,
) -> tuple[int, float, int]:
    ops = _ops()
    attempt = requested_increment
    level = 0
    while abs(attempt) + 1.0e-15 >= minimum_increment:
        ops.integrator("DisplacementControl", control_node, dof, attempt)
        for algorithm in ("Newton", "NewtonLineSearch", "KrylovNewton"):
            if algorithm == "NewtonLineSearch":
                ops.algorithm("NewtonLineSearch", 0.8)
            elif algorithm == "KrylovNewton":
                ops.algorithm("KrylovNewton")
            else:
                ops.algorithm("Newton")
            code = int(ops.analyze(1))
            if code == 0:
                ops.algorithm("Newton")
                return 0, attempt, level
        attempt *= 0.5
        level += 1
    return -1, attempt, level


def _run_full3d_static_pattern(
    building: dict[str, Any],
    config: dict[str, Any],
    *,
    direction: str,
    pattern_name: str,
    pattern: np.ndarray,
    nonlinear: bool,
) -> dict[str, Any]:
    settings = calibration_config(config)
    info = build_full3d_model(building, config)
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"{building['building_id']}: gravity error exceeds 1%"
        )
    normalized_direction = direction.upper()
    dof = 1 if normalized_direction == "X" else 2
    ops = _ops()
    series_tag = 60
    pattern_tag = 60
    ops.timeSeries("Linear", series_tag)
    ops.pattern("Plain", pattern_tag, series_tag)
    for node, force in zip(info.master_nodes, pattern):
        load = [0.0] * 6
        load[dof - 1] = float(force)
        ops.load(node, *load)
    _configure_static()
    control_node = info.master_nodes[-1]
    height = float(info.floor_elevations_m[-1])
    if nonlinear:
        requested_increment = float(
            settings["nonlinear_displacement_increment_m"]
        )
        minimum_increment = float(
            settings["nonlinear_minimum_displacement_increment_m"]
        )
        roof_limit = float(settings["nonlinear_maximum_roof_drift"]) * height
    else:
        requested_increment = (
            float(settings["elastic_probe_roof_drift"]) * height
        )
        minimum_increment = requested_increment / 64.0
        roof_limit = requested_increment
    floor_rows = [np.zeros(len(info.master_nodes), dtype=float)]
    delta_rows = [np.zeros(len(info.master_nodes), dtype=float)]
    shear_rows = [np.zeros(len(info.master_nodes), dtype=float)]
    roof_rows = [0.0]
    base_rows = [0.0]
    load_factors = [0.0]
    equilibrium_errors = [0.0]
    reduction_levels: list[int] = []
    started = time.perf_counter()
    termination = "target_reached"
    maximum_steps = int(settings["maximum_static_steps"])
    for _step in range(maximum_steps):
        code, accepted_increment, reduction_level = _analyze_displacement_step(
            control_node,
            dof,
            requested_increment,
            minimum_increment,
        )
        reduction_levels.append(reduction_level)
        if code != 0:
            termination = "solver_exhaustion"
            break
        floor, delta = _story_displacements(info, dof)
        factor = float(ops.getLoadFactor(pattern_tag))
        shear, base, equilibrium_error = _story_cut_shears(
            info, dof, load_factor=factor
        )
        floor_rows.append(floor)
        delta_rows.append(delta)
        shear_rows.append(shear)
        roof_rows.append(abs(float(floor[-1])))
        base_rows.append(abs(float(base)))
        load_factors.append(factor)
        equilibrium_errors.append(equilibrium_error)
        if not nonlinear or roof_rows[-1] >= roof_limit:
            break
        peak = max(base_rows)
        if (
            len(base_rows) > 10
            and peak > 0.0
            and base_rows[-1] <= 0.20 * peak
            and base_rows[-2] > base_rows[-1]
        ):
            termination = "postpeak_20pct_endpoint"
            break
        if (
            time.perf_counter() - started
            > float(settings["maximum_static_runtime_s"])
        ):
            termination = "runtime_guard"
            break
        # Preserve the accepted reduced increment after a difficult state.
        requested_increment = min(requested_increment, accepted_increment)
    else:
        termination = "step_guard"
    return {
        "direction": normalized_direction,
        "pattern_name": pattern_name,
        "pattern": np.asarray(pattern, dtype=float),
        "nonlinear": bool(nonlinear),
        "floor_displacement_m": np.asarray(floor_rows, dtype=float),
        "story_deformation_m": np.asarray(delta_rows, dtype=float),
        "story_shear_kn": np.asarray(shear_rows, dtype=float),
        "roof_displacement_m": np.asarray(roof_rows, dtype=float),
        "base_shear_kn": np.asarray(base_rows, dtype=float),
        "load_factor": np.asarray(load_factors, dtype=float),
        "equilibrium_error": np.asarray(equilibrium_errors, dtype=float),
        "maximum_equilibrium_error": float(max(equilibrium_errors)),
        "maximum_step_reduction_level": max(reduction_levels, default=0),
        "termination_reason": termination,
        "runtime_s": time.perf_counter() - started,
    }


def _reduced_modes(
    material_stiffness: np.ndarray,
    geometric_stiffness: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    net = np.asarray(material_stiffness) + np.asarray(geometric_stiffness)
    if np.any(net <= 0.0):
        raise ValueError("Reduced model has non-positive net story stiffness")
    stiffness = shear_stiffness_matrix(net)
    mass_root_inverse = np.diag(1.0 / np.sqrt(masses))
    transformed = mass_root_inverse @ stiffness @ mass_root_inverse
    eigenvalues, eigenvectors = eigh(transformed)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Reduced model has a non-positive eigenvalue")
    floor_shapes = mass_root_inverse @ eigenvectors
    periods = 2.0 * math.pi / np.sqrt(eigenvalues)
    order = np.argsort(periods)[::-1]
    periods = periods[order]
    floor_shapes = floor_shapes[:, order].T
    for index, shape in enumerate(floor_shapes):
        scale = float(np.max(np.abs(shape)))
        if scale > 0.0:
            shape /= scale
        if float(shape @ masses) < 0.0:
            shape *= -1.0
        floor_shapes[index] = shape
    return periods, floor_shapes


def _fit_elastic_stiffness(
    probes: dict[str, dict[str, Any]],
    patterns: dict[str, np.ndarray],
    targets: dict[str, Any],
    masses: np.ndarray,
    story_heights: np.ndarray,
    gravity_above: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    geometric = -gravity_above / story_heights
    estimates: list[np.ndarray] = []
    for curve in probes.values():
        delta = np.asarray(curve["story_deformation_m"][-1], dtype=float)
        shear = np.asarray(curve["story_shear_kn"][-1], dtype=float)
        material_shear = shear - geometric * np.abs(delta)
        estimate = material_shear / np.maximum(np.abs(delta), 1.0e-12)
        if np.all(np.isfinite(estimate)) and np.all(estimate > 0.0):
            estimates.append(estimate)
    if not estimates:
        raise RuntimeError("Elastic probes did not produce positive stiffness")
    objective_weights = settings["elastic_objective_weights"]
    target_periods = np.asarray(targets["periods_s"], dtype=float)
    target_shapes = np.asarray(targets["mode_shapes"], dtype=float)
    # A shear-building stiffness that exactly satisfies the target first
    # translational eigenpair is the physically compatible starting point.
    # Starting from independent V/Delta medians can bias every story upward
    # when signed higher-mode probes contain local frame deformation, which
    # sacrifices the drift-critical first mode before optimization begins.
    first_shape = target_shapes[0].copy()
    if float(first_shape @ masses) < 0.0:
        first_shape *= -1.0
    omega_first = 2.0 * math.pi / float(target_periods[0])
    modal_story_shear = np.cumsum(
        (omega_first**2 * masses * first_shape)[::-1]
    )[::-1]
    modal_story_deformation = np.diff(
        np.concatenate(([0.0], first_shape))
    )
    modal_net = modal_story_shear / modal_story_deformation
    modal_material = modal_net - geometric
    probe_initial = np.median(np.vstack(estimates), axis=0)
    initial = np.where(
        np.isfinite(modal_material) & (modal_material > 0.0),
        modal_material,
        probe_initial,
    )
    significant = min(len(target_periods), len(masses), 3)
    residual_size = (
        len(probes) * len(masses)
        + 2
        + 2 * max(significant - 1, 0)
        + len(masses)
    )

    def residual(log_stiffness: np.ndarray) -> np.ndarray:
        material = np.exp(log_stiffness)
        net = material + geometric
        if np.any(net <= 0.0):
            # scipy requires a fixed residual vector length throughout the
            # optimization, including trial points with unstable net story
            # stiffness.
            return np.full(residual_size, 1.0e3)
        stiffness_matrix = shear_stiffness_matrix(net)
        rows: list[float] = []
        drift_scale = math.sqrt(
            float(objective_weights["drift_profiles"])
            / max(len(probes), 1)
        )
        for name, curve in probes.items():
            factor = float(curve["load_factor"][-1])
            force = np.asarray(patterns[name], dtype=float) * factor
            predicted_floor = np.linalg.solve(stiffness_matrix, force)
            predicted_delta = np.diff(
                np.concatenate(([0.0], predicted_floor))
            )
            target_delta = np.asarray(
                curve["story_deformation_m"][-1], dtype=float
            )
            scale = max(float(np.linalg.norm(target_delta)), 1.0e-12)
            rows.extend(
                (
                    drift_scale * (predicted_delta - target_delta) / scale
                ).tolist()
            )
        periods, shapes = _reduced_modes(
            material, geometric, masses
        )
        first_weight = math.sqrt(float(objective_weights["first_mode"]))
        rows.append(
            first_weight
            * (float(periods[0]) - float(target_periods[0]))
            / float(target_periods[0])
        )
        rows.append(
            first_weight
            * math.sqrt(
                max(
                    1.0
                    - modal_assurance_criterion(shapes[0], target_shapes[0]),
                    0.0,
                )
            )
        )
        higher_weight = math.sqrt(
            float(objective_weights["higher_modes"])
            / max(significant - 1, 1)
        )
        for mode in range(1, significant):
            rows.append(
                higher_weight
                * (float(periods[mode]) - float(target_periods[mode]))
                / float(target_periods[mode])
            )
            rows.append(
                higher_weight
                * math.sqrt(
                    max(
                        1.0
                        - modal_assurance_criterion(
                            shapes[mode], target_shapes[mode]
                        ),
                        0.0,
                    )
                )
            )
        regularization = math.sqrt(
            float(objective_weights["regularization"])
            / max(len(material), 1)
        )
        rows.extend(
            (
                regularization
                * (log_stiffness - np.log(initial))
            ).tolist()
        )
        return np.asarray(rows, dtype=float)

    solution = least_squares(
        residual,
        np.log(initial),
        bounds=(np.log(initial * 0.20), np.log(initial * 5.0)),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=1000,
    )
    core_names = tuple(
        name
        for name in ("first_mode", "uniform", "triangular")
        if name in probes
    )

    def primary_residual(
        log_stiffness: np.ndarray,
        higher_mode_multiplier: float = 1.0,
    ) -> np.ndarray:
        material = np.exp(log_stiffness)
        net = material + geometric
        size = (
            len(core_names) * len(material)
            + 2
            + 2 * max(significant - 1, 0)
            + len(material)
        )
        if np.any(net <= 0.0):
            return np.full(size, 1.0e3)
        stiffness_matrix = shear_stiffness_matrix(net)
        rows: list[float] = []
        core_weight = math.sqrt(
            float(objective_weights["drift_profiles"])
            / max(len(core_names), 1)
        )
        for name in core_names:
            curve = probes[name]
            factor = float(curve["load_factor"][-1])
            predicted = np.linalg.solve(
                stiffness_matrix, patterns[name] * factor
            )
            predicted_delta = np.diff(
                np.concatenate(([0.0], predicted))
            )
            target_delta = np.asarray(
                curve["story_deformation_m"][-1], dtype=float
            )
            rows.extend(
                (
                    core_weight
                    * (predicted_delta - target_delta)
                    / max(float(np.linalg.norm(target_delta)), 1.0e-12)
                ).tolist()
            )
        periods, shapes = _reduced_modes(material, geometric, masses)
        first_weight = math.sqrt(
            float(objective_weights["first_mode"])
        )
        rows.extend(
            (
                first_weight
                * (periods[0] - target_periods[0])
                / target_periods[0],
                first_weight
                * math.sqrt(
                    max(
                        1.0
                        - modal_assurance_criterion(
                            shapes[0], target_shapes[0]
                        ),
                        0.0,
                    )
                ),
            )
        )
        higher_weight = math.sqrt(
            float(objective_weights["higher_modes"])
            / max(significant - 1, 1)
        ) * math.sqrt(higher_mode_multiplier)
        for mode in range(1, significant):
            rows.extend(
                (
                    higher_weight
                    * (periods[mode] - target_periods[mode])
                    / target_periods[mode],
                    higher_weight
                    * math.sqrt(
                        max(
                            1.0
                            - modal_assurance_criterion(
                                shapes[mode], target_shapes[mode]
                            ),
                            0.0,
                        )
                    ),
                )
            )
        regularization = math.sqrt(
            float(objective_weights["regularization"])
            / max(len(material), 1)
        )
        rows.extend(
            (
                regularization
                * (log_stiffness - np.log(initial))
            ).tolist()
        )
        return np.asarray(rows, dtype=float)

    primary_solutions = []
    starting_point = np.log(initial)
    for multiplier in (0.50, 1.0, 2.0, 4.0, 8.0):
        candidate = least_squares(
            lambda values, current=multiplier: primary_residual(
                values, current
            ),
            starting_point,
            bounds=(np.log(initial * 0.20), np.log(initial * 5.0)),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=1000,
        )
        primary_solutions.append((multiplier, candidate))
        starting_point = candidate.x

    def solution_metrics(
        log_stiffness: np.ndarray,
    ) -> tuple[float, float, float]:
        material = np.exp(log_stiffness)
        stiffness_matrix = shear_stiffness_matrix(material + geometric)
        errors = []
        for name in core_names:
            curve = probes[name]
            predicted = np.linalg.solve(
                stiffness_matrix,
                patterns[name] * float(curve["load_factor"][-1]),
            )
            predicted_delta = np.diff(
                np.concatenate(([0.0], predicted))
            )
            target_delta = np.asarray(
                curve["story_deformation_m"][-1], dtype=float
            )
            errors.append(
                float(
                    np.linalg.norm(predicted_delta - target_delta)
                    / max(np.linalg.norm(target_delta), 1.0e-12)
                )
            )
        periods, shapes = _reduced_modes(material, geometric, masses)
        return (
            float(np.mean(errors)),
            abs(float(periods[0] - target_periods[0]))
            / float(target_periods[0]),
            modal_assurance_criterion(shapes[0], target_shapes[0]),
        )

    feasible_candidates = []
    for multiplier, candidate in primary_solutions:
        core_error, first_period_error, first_mac = solution_metrics(
            candidate.x
        )
        material = np.exp(candidate.x)
        candidate_periods, candidate_shapes = _reduced_modes(
            material, geometric, masses
        )
        modal_period_errors = [
            abs(
                float(candidate_periods[mode])
                - float(target_periods[mode])
            )
            / float(target_periods[mode])
            for mode in range(significant)
        ]
        modal_shape_errors = [
            math.sqrt(
                max(
                    1.0
                    - modal_assurance_criterion(
                        candidate_shapes[mode], target_shapes[mode]
                    ),
                    0.0,
                )
            )
            for mode in range(significant)
        ]
        if (
            core_error <= 0.05
            and first_period_error <= 0.02
            and first_mac >= 0.98
        ):
            # The first-mode/drift constraints are hard.  Within that feasible
            # set, minimize the worst error among all first three
            # translational modes so mode 2 or 3 cannot be silently ignored.
            three_mode_score = max(modal_period_errors) + 0.25 * max(
                modal_shape_errors
            )
            feasible_candidates.append(
                (
                    three_mode_score,
                    core_error,
                    -multiplier,
                    candidate,
                )
            )
    if feasible_candidates:
        *_, selected = min(feasible_candidates, key=lambda row: row[:3])
        fit_selection = "three_mode_pareto_constrained"
    else:
        selected = solution
        fit_selection = "weighted_plan_objective"
    fitted = np.exp(selected.x)
    preliminary_periods, _preliminary_shapes = _reduced_modes(
        fitted, geometric, masses
    )
    first_period_scale = (
        float(preliminary_periods[0]) / float(target_periods[0])
    ) ** 2
    # Scale the net shear stiffness to match the dominant period exactly.
    # This preserves all reduced mode shapes and improved the paired dynamic
    # drift/MIDR comparison.  T2/T3 incompatibility remains explicit in the
    # stored metrics instead of being hidden.
    fitted = (fitted + geometric) * first_period_scale - geometric
    if np.any(fitted <= 0.0):
        raise RuntimeError("Exact-T1 scaling produced invalid story stiffness")
    fit_selection += "+exact_t1_scaling"
    periods, shapes = _reduced_modes(fitted, geometric, masses)
    drift_errors = []
    stiffness_matrix = shear_stiffness_matrix(fitted + geometric)
    for name, curve in probes.items():
        factor = float(curve["load_factor"][-1])
        predicted = np.linalg.solve(
            stiffness_matrix, patterns[name] * factor
        )
        predicted_delta = np.diff(np.concatenate(([0.0], predicted)))
        target_delta = np.asarray(
            curve["story_deformation_m"][-1], dtype=float
        )
        drift_errors.append(
            float(
                np.linalg.norm(predicted_delta - target_delta)
                / max(np.linalg.norm(target_delta), 1.0e-12)
            )
        )
    core_errors = [
        error
        for name, error in zip(probes, drift_errors)
        if name in core_names
    ]
    higher_errors = [
        error
        for name, error in zip(probes, drift_errors)
        if name not in core_names
    ]
    metrics = {
        "elastic_objective": float(
            np.mean(residual(np.log(fitted)) ** 2)
        ),
        "elastic_drift_nrmse": float(np.mean(drift_errors)),
        "elastic_core_drift_nrmse": float(np.mean(core_errors)),
        "elastic_higher_mode_drift_nrmse": (
            float(np.mean(higher_errors)) if higher_errors else 0.0
        ),
        "elastic_fit_selection": fit_selection,
        "first_mode_mac": modal_assurance_criterion(
            shapes[0], target_shapes[0]
        ),
        "first_period_error": abs(periods[0] - target_periods[0])
        / target_periods[0],
        "reduced_periods_s": tuple(
            float(value) for value in periods[:significant]
        ),
        "mode_period_errors": tuple(
            float(abs(periods[mode] - target_periods[mode]))
            / float(target_periods[mode])
            for mode in range(significant)
        ),
        "mode_macs": tuple(
            modal_assurance_criterion(shapes[mode], target_shapes[mode])
            for mode in range(significant)
        ),
    }
    return fitted, metrics


def _backbone_force(
    deformation: np.ndarray,
    *,
    stiffness: float,
    yield_force: float,
    capping_ratio: float,
    capping_deformation: float,
    residual_ratio: float,
    residual_deformation: float,
    ultimate_deformation: float,
) -> np.ndarray:
    x = np.abs(np.asarray(deformation, dtype=float))
    yield_deformation = yield_force / stiffness
    capping_force = yield_force * capping_ratio
    residual_force = capping_force * residual_ratio
    hardening = (capping_force - yield_force) / max(
        capping_deformation - yield_deformation, 1.0e-12
    )
    post_capping = (residual_force - capping_force) / max(
        residual_deformation - capping_deformation, 1.0e-12
    )
    return np.piecewise(
        x,
        [
            x <= yield_deformation,
            (x > yield_deformation) & (x <= capping_deformation),
            (x > capping_deformation) & (x <= residual_deformation),
            (x > residual_deformation) & (x <= ultimate_deformation),
            x > ultimate_deformation,
        ],
        [
            lambda value: stiffness * value,
            lambda value: yield_force
            + hardening * (value - yield_deformation),
            lambda value: capping_force
            + post_capping * (value - capping_deformation),
            residual_force,
            0.0,
        ],
    )


def _backbone_force_with_cracking(
    deformation: np.ndarray,
    *,
    stiffness: float,
    post_cracking_stiffness: float,
    cracking_deformation: float,
    yield_force: float,
    capping_force: float,
    capping_deformation: float,
    residual_force: float,
    residual_deformation: float,
    ultimate_deformation: float,
) -> np.ndarray:
    x = np.abs(np.asarray(deformation, dtype=float))
    branch_force = (
        stiffness - post_cracking_stiffness
    ) * cracking_deformation
    core_yield = yield_force - branch_force
    yield_deformation = core_yield / post_cracking_stiffness
    hardening = (capping_force - yield_force) / max(
        capping_deformation - yield_deformation, 1.0e-12
    )
    post_capping = (residual_force - capping_force) / max(
        residual_deformation - capping_deformation, 1.0e-12
    )
    return np.piecewise(
        x,
        [
            x <= cracking_deformation,
            (x > cracking_deformation) & (x <= yield_deformation),
            (x > yield_deformation) & (x <= capping_deformation),
            (x > capping_deformation) & (x <= residual_deformation),
            (x > residual_deformation) & (x <= ultimate_deformation),
            x > ultimate_deformation,
        ],
        [
            lambda value: stiffness * value,
            lambda value: branch_force
            + post_cracking_stiffness * value,
            lambda value: yield_force
            + hardening * (value - yield_deformation),
            lambda value: capping_force
            + post_capping * (value - capping_deformation),
            residual_force,
            0.0,
        ],
    )


def _fit_cracking_branch(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    stiffness: float,
    height: float,
    yield_force: float,
    capping_force: float,
    capping_deformation: float,
    residual_force: float,
    residual_deformation: float,
    ultimate_deformation: float,
    settings: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Fit a pre-yield cracking knee while holding capacity points fixed."""
    peak = max(float(np.max(y)), 1.0e-12)
    uncracked = _backbone_force(
        x,
        stiffness=stiffness,
        yield_force=yield_force,
        capping_ratio=capping_force / yield_force,
        capping_deformation=capping_deformation,
        residual_ratio=residual_force / capping_force,
        residual_deformation=residual_deformation,
        ultimate_deformation=ultimate_deformation,
    )

    def weighted_nrmse(prediction: np.ndarray) -> float:
        return float(
            np.sqrt(
                np.sum(weights * (prediction - y) ** 2)
                / max(np.sum(weights), 1.0e-12)
            )
            / peak
        )

    baseline_error = weighted_nrmse(uncracked)
    maximum_cracking = min(
        float(settings["cracking_maximum_drift"]) * height,
        0.80 * yield_force / stiffness,
        0.80 * capping_deformation,
    )
    if maximum_cracking <= 1.0e-8:
        return 0.0, 0.0, yield_force / stiffness, baseline_error

    maximum_branch_force = (
        float(settings["cracking_maximum_residual_force_fraction"])
        * residual_force
    )

    def objective(values: np.ndarray) -> float:
        fraction, cracking_deformation = values
        post_cracking = stiffness * fraction
        branch_force = (
            stiffness - post_cracking
        ) * cracking_deformation
        core_yield = yield_force - branch_force
        yield_deformation = core_yield / max(post_cracking, 1.0e-12)
        penalties = 0.0
        if branch_force >= maximum_branch_force:
            penalties += 1.0e3 * (
                (branch_force - maximum_branch_force)
                / max(residual_force, 1.0)
            ) ** 2
        for violation in (
            cracking_deformation * 1.01 - yield_deformation,
            yield_deformation * 1.01 - capping_deformation,
        ):
            if violation > 0.0:
                penalties += 1.0e3 * (
                    violation / max(capping_deformation, 1.0e-8)
                ) ** 2
        prediction = _backbone_force_with_cracking(
            x,
            stiffness=stiffness,
            post_cracking_stiffness=post_cracking,
            cracking_deformation=cracking_deformation,
            yield_force=yield_force,
            capping_force=capping_force,
            capping_deformation=capping_deformation,
            residual_force=residual_force,
            residual_deformation=residual_deformation,
            ultimate_deformation=ultimate_deformation,
        )
        return weighted_nrmse(prediction) + penalties

    solutions = []
    for fraction in (0.25, 0.50, 0.75, 0.90):
        initial = np.asarray(
            [fraction, 0.50 * maximum_cracking], dtype=float
        )
        candidate = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=((0.05, 0.95), (1.0e-8, maximum_cracking)),
            options={"maxiter": 1000, "ftol": 1.0e-12},
        )
        if np.all(np.isfinite(candidate.x)):
            solutions.append(candidate)
    if not solutions:
        return 0.0, 0.0, yield_force / stiffness, baseline_error
    solution = min(solutions, key=lambda candidate: float(candidate.fun))
    fraction, cracking_deformation = map(float, solution.x)
    post_cracking = stiffness * fraction
    branch_force = (
        stiffness - post_cracking
    ) * cracking_deformation
    yield_deformation = (yield_force - branch_force) / post_cracking
    improvement = (
        baseline_error - float(solution.fun)
    ) / max(baseline_error, 1.0e-12)
    if (
        improvement
        < float(settings["cracking_fit_minimum_improvement_ratio"])
        or branch_force <= 0.01 * yield_force
        or branch_force >= maximum_branch_force
        or not (
            cracking_deformation
            < yield_deformation
            < capping_deformation
        )
    ):
        return 0.0, 0.0, yield_force / stiffness, baseline_error
    return (
        float(cracking_deformation),
        float(post_cracking),
        float(yield_deformation),
        float(solution.fun),
    )


def _regularize_cracking_by_height(
    *,
    story: int,
    story_count: int,
    stiffness: float,
    yield_force: float,
    cracking_deformation: float,
    cracking_branch_force: float,
    settings: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Optionally regularize a local cracking fit before story-material use.

    Roof-controlled static patterns can make the fitted cracking branch too
    dominant near the top of a shear MDOF model.  The empirical height taper is
    disabled by default because it was identified on only one five-story smoke
    case.  It may be enabled only for an explicitly versioned study after
    cross-building development/holdout validation.
    """
    if cracking_deformation <= 0.0 or cracking_branch_force <= 0.0:
        return 0.0, 0.0, yield_force / stiffness, 0.0
    if (
        cracking_branch_force >= yield_force
        or cracking_branch_force
        >= stiffness * cracking_deformation
    ):
        return 0.0, 0.0, yield_force / stiffness, 0.0
    post_cracking = stiffness - (
        cracking_branch_force / cracking_deformation
    )
    yield_deformation = (
        yield_force - cracking_branch_force
    ) / post_cracking
    original = (
        float(cracking_deformation),
        float(post_cracking),
        float(yield_deformation),
        float(cracking_branch_force),
    )
    if not bool(settings.get("cracking_regularization_enabled", False)):
        return original
    adjusted_deformation = float(cracking_deformation)
    if story == 0:
        adjusted_deformation *= float(
            settings["first_story_cracking_deformation_factor"]
        )
    normalized_height = (
        float(story) / float(story_count - 1)
        if story_count > 1
        else 0.0
    )
    start = float(settings["upper_story_cracking_regularization_start"])
    decay = float(settings["upper_story_cracking_branch_decay"])
    branch_scale = (
        2.0 ** (-decay * (normalized_height - start))
        if normalized_height > start
        else 1.0
    )
    adjusted_branch_force = float(cracking_branch_force) * branch_scale
    post_cracking = stiffness - (
        adjusted_branch_force / adjusted_deformation
    )
    yield_deformation = (
        yield_force - adjusted_branch_force
    ) / post_cracking
    if not (
        0.0
        < adjusted_deformation
        < yield_deformation
        and 0.0 < post_cracking < stiffness
    ):
        return original
    return (
        adjusted_deformation,
        float(post_cracking),
        float(yield_deformation),
        adjusted_branch_force,
    )


def _monotonic_story_envelope(
    deformation: np.ndarray,
    force: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only new local-deformation maxima from a global pushover.

    A story can unload while another story localizes.  Those unloading points
    are valid system response but are not points on that story's monotonic
    backbone and must not be interpreted as local post-capping degradation.
    """
    deformation = np.abs(np.asarray(deformation, dtype=float))
    force = np.asarray(force, dtype=float)
    keep = (
        np.isfinite(deformation)
        & np.isfinite(force)
        & (deformation > 1.0e-10)
        & (force >= 0.0)
    )
    deformation = deformation[keep]
    force = force[keep]
    if deformation.size == 0:
        return deformation, force
    retained = []
    running_maximum = -math.inf
    tolerance = max(float(np.max(deformation)) * 1.0e-8, 1.0e-12)
    for index, value in enumerate(deformation):
        if float(value) > running_maximum + tolerance:
            retained.append(index)
            running_maximum = float(value)
    indices = np.asarray(retained, dtype=int)
    return deformation[indices], force[indices]


def _adaptive_pattern_weights(
    envelopes: dict[str, tuple[np.ndarray, np.ndarray]],
    nominal: dict[str, float],
    *,
    compatibility_nrmse: float,
    minimum_weight_factor: float,
    minimum_overlap_deformation: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Downweight higher-mode curves that one local shear spring cannot fit.

    Higher-mode Full 3D response can contain frame/chord deformation and a
    modal node inside a story.  In that case story shear is not a unique
    function of the local interstory deformation, so forcing that curve into
    one zero-length spring creates a false story capacity.  The curve remains
    available for modal SPO/ML features, while its backbone weight is reduced
    according to its incompatibility with the three core lateral patterns.
    """
    core_names = tuple(
        name
        for name in ("first_mode", "uniform", "triangular")
        if name in envelopes
    )
    higher_names = tuple(
        name for name in envelopes if name.startswith("mode_")
    )
    if not core_names:
        raise RuntimeError("Nonlinear backbone has no core load patterns")
    weights = {
        name: float(nominal.get(name, 0.0)) for name in core_names
    }
    higher_total = float(nominal["higher_modes_total"])
    nominal_higher = higher_total / max(len(higher_names), 1)
    errors: dict[str, float] = {}
    rejected_weight = 0.0
    for name in higher_names:
        deformation, force = envelopes[name]
        overlap_upper = min(
            float(np.max(deformation)),
            max(float(np.max(envelopes[core][0])) for core in core_names),
        )
        if overlap_upper < minimum_overlap_deformation:
            error = 1.0
        else:
            lower_candidates = [
                float(np.min(deformation)),
                *[
                    float(np.min(envelopes[core][0]))
                    for core in core_names
                ],
            ]
            overlap_lower = max(
                max(lower_candidates),
                0.05 * overlap_upper,
            )
            if overlap_lower >= overlap_upper:
                error = 1.0
            else:
                query = np.linspace(overlap_lower, overlap_upper, 64)
                core_forces = []
                for core in core_names:
                    core_x, core_y = envelopes[core]
                    row = np.full(query.size, np.nan, dtype=float)
                    available = query <= float(np.max(core_x))
                    row[available] = np.interp(
                        query[available], core_x, core_y
                    )
                    core_forces.append(row)
                reference = np.nanmedian(
                    np.asarray(core_forces, dtype=float), axis=0
                )
                comparison = np.interp(query, deformation, force)
                available = np.isfinite(reference)
                scale = max(
                    float(np.max(reference[available])), 1.0e-12
                )
                error = float(
                    np.sqrt(
                        np.mean(
                            (
                                comparison[available]
                                - reference[available]
                            )
                            ** 2
                        )
                    )
                    / scale
                )
        errors[name] = error
        factor = float(
            np.clip(
                compatibility_nrmse / max(error, 1.0e-12),
                minimum_weight_factor,
                1.0,
            )
        )
        weights[name] = nominal_higher * factor
        rejected_weight += nominal_higher * (1.0 - factor)
    core_total = sum(weights[name] for name in core_names)
    if rejected_weight > 0.0 and core_total > 0.0:
        for name in core_names:
            weights[name] += rejected_weight * weights[name] / core_total
    total = sum(weights.values())
    if total <= 0.0:
        raise RuntimeError("Nonlinear backbone pattern weights are zero")
    return (
        {name: value / total for name, value in weights.items()},
        errors,
    )


def _fit_story_spring(
    story: int,
    story_count: int,
    stiffness: float,
    height: float,
    gravity_above: float,
    curves: dict[str, dict[str, Any]],
    pattern_weights: dict[str, float],
    cyclic: dict[str, float],
    drift_bin_count: int,
    settings: dict[str, Any],
) -> StorySpringParameters:
    geometric = -gravity_above / height
    x_rows = []
    y_rows = []
    weight_rows = []
    core_x_rows = []
    core_y_rows = []
    core_weight_rows = []
    envelopes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    postpeak_confirmed: dict[str, bool] = {}
    for name, curve in curves.items():
        raw_x = np.abs(
            np.asarray(curve["story_deformation_m"], dtype=float)[:, story]
        )
        total = np.abs(
            np.asarray(curve["story_shear_kn"], dtype=float)[:, story]
        )
        raw_material = total - geometric * raw_x
        x, material = _monotonic_story_envelope(
            raw_x, raw_material
        )
        if x.size == 0:
            continue
        envelopes[name] = (x, material)
        peak_index = int(np.argmax(material))
        peak_force = float(material[peak_index])
        postpeak_drop = (
            (peak_force - float(material[-1])) / peak_force
            if peak_force > 0.0
            else 0.0
        )
        postpeak_drift_increment = (
            float(x[-1] - x[peak_index]) / height
        )
        postpeak_confirmed[name] = bool(
            x.size - peak_index - 1 >= 3
            and postpeak_drop
            >= float(settings["postpeak_minimum_force_drop_ratio"])
            and postpeak_drift_increment
            >= float(settings["postpeak_minimum_drift_increment"])
        )
    effective_weights, compatibility_errors = _adaptive_pattern_weights(
        envelopes,
        pattern_weights,
        compatibility_nrmse=float(
            settings["higher_mode_backbone_compatibility_nrmse"]
        ),
        minimum_weight_factor=float(
            settings["higher_mode_backbone_minimum_weight_factor"]
        ),
        minimum_overlap_deformation=(
            float(settings["higher_mode_minimum_compatibility_drift"])
            * height
        ),
    )
    postpeak_pattern_count = sum(
        int(confirmed)
        for name, confirmed in postpeak_confirmed.items()
        if not name.startswith("mode_")
        or effective_weights.get(name, 0.0)
        >= (
            0.5
            * float(pattern_weights["higher_modes_total"])
            / max(len(compatibility_errors), 1)
        )
    )
    for name, (x, material) in envelopes.items():
        pattern_weight = effective_weights[name]
        x_rows.append(x)
        y_rows.append(material)
        # A displacement-controlled pushover typically contributes many more
        # points at small drift than near capping.  Give every occupied drift
        # bin equal total weight within a load pattern so dense elastic samples
        # cannot dominate the story-backbone fit.
        bin_count = max(2, int(drift_bin_count))
        maximum_x = float(np.max(x))
        if maximum_x <= 1.0e-12:
            point_weights = np.full(
                x.size, pattern_weight / x.size, dtype=float
            )
        else:
            edges = np.linspace(0.0, maximum_x, bin_count + 1)
            bin_indices = np.clip(
                np.searchsorted(edges, x, side="right") - 1,
                0,
                bin_count - 1,
            )
            occupied = np.unique(bin_indices)
            point_weights = np.zeros(x.size, dtype=float)
            for bin_index in occupied:
                in_bin = bin_indices == bin_index
                point_weights[in_bin] = (
                    pattern_weight
                    / float(occupied.size)
                    / float(np.count_nonzero(in_bin))
                )
        weight_rows.append(point_weights)
        if name in {"first_mode", "uniform", "triangular"}:
            core_x_rows.append(x)
            core_y_rows.append(material)
            core_weight_rows.append(
                point_weights / max(pattern_weight, 1.0e-12)
            )
    if not x_rows:
        raise RuntimeError(f"Story {story + 1}: no nonlinear calibration data")
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    weights = np.concatenate(weight_rows)
    peak = float(np.max(y))
    maximum_deformation = float(np.max(x))
    if peak <= 0.0 or maximum_deformation <= 0.0:
        raise RuntimeError(f"Story {story + 1}: invalid nonlinear envelope")
    peak_index = int(np.argmax(y))
    peak_deformation = float(x[peak_index])
    initial_yield = min(0.70 * peak, stiffness * 0.60 * peak_deformation)
    initial_yield = max(initial_yield, 0.10 * peak)
    initial_dy = initial_yield / stiffness
    initial_dc = max(peak_deformation, initial_dy * 1.20)
    last_force = float(np.median(y[x >= np.quantile(x, 0.90)]))
    initial_residual_ratio = float(
        np.clip(last_force / peak, 0.05, 0.50)
    )
    initial_dr = max(initial_dc * 1.20, 0.85 * maximum_deformation)
    initial_du = max(maximum_deformation * 1.05, initial_dr * 1.05)
    initial = np.asarray(
        [
            initial_yield,
            max(peak / initial_yield, 1.0),
            initial_dc,
            initial_residual_ratio,
            initial_dr,
            initial_du,
        ],
        dtype=float,
    )
    bounds = (
        (0.05 * peak, 1.0, 1.0e-8, 0.02, 1.0e-8, 1.0e-8),
        (
            peak,
            1.60,
            maximum_deformation * 1.50,
            0.70,
            maximum_deformation * 2.0,
            maximum_deformation * 3.0,
        ),
    )

    def objective(values: np.ndarray) -> float:
        fy, ratio, dc, residual_ratio, dr, du = values
        dy = fy / stiffness
        penalties = 0.0
        scale = max(maximum_deformation, 1.0e-8)
        for violation in (
            dy * 1.01 - dc,
            dc * 1.01 - dr,
            dr * 1.01 - du,
        ):
            if violation > 0.0:
                penalties += 1.0e4 * (violation / scale) ** 2
        predicted = _backbone_force(
            x,
            stiffness=stiffness,
            yield_force=fy,
            capping_ratio=ratio,
            capping_deformation=dc,
            residual_ratio=residual_ratio,
            residual_deformation=dr,
            ultimate_deformation=du,
        )
        normalized_error = (predicted - y) / peak
        robust = np.sqrt(1.0 + (normalized_error / 0.05) ** 2) - 1.0
        return float(np.sum(weights * robust) + penalties)

    solutions = []
    for yield_fraction in (0.30, 0.50, 0.70, 0.85):
        candidate_initial = initial.copy()
        candidate_initial[0] = float(
            np.clip(
                yield_fraction * peak,
                bounds[0][0],
                bounds[1][0],
            )
        )
        candidate_initial[1] = float(
            np.clip(
                peak / candidate_initial[0],
                bounds[0][1],
                bounds[1][1],
            )
        )
        candidate = minimize(
            objective,
            candidate_initial,
            method="L-BFGS-B",
            bounds=list(zip(*bounds)),
            options={"maxiter": 2000, "ftol": 1.0e-12},
        )
        if np.all(np.isfinite(candidate.x)) and math.isfinite(
            float(candidate.fun)
        ):
            solutions.append(candidate)
    if not solutions:
        raise RuntimeError(
            f"Story {story + 1}: nonlinear backbone optimization failed"
        )
    solution = min(solutions, key=lambda candidate: float(candidate.fun))
    fy, capping_ratio, dc, residual_ratio, dr, du = solution.x
    dy = fy / stiffness
    backbone_censored = postpeak_pattern_count == 0
    if backbone_censored:
        # The observed curve is a lower bound: the story remained on an
        # increasing local envelope while another story controlled the global
        # mechanism.  Extrapolate a declared, validation-gated capacity rather
        # than treating the last demanded deformation as failure.
        fy = max(
            float(fy),
            peak
            * float(settings["censored_capping_strength_factor"])
            / 1.60,
        )
        dy = fy / stiffness
        dc = max(
            maximum_deformation
            * float(settings["censored_capping_deformation_factor"]),
            float(settings["censored_minimum_capping_drift"]) * height,
            dy * 1.20,
        )
        capping_force = max(
            peak * float(settings["censored_capping_strength_factor"]),
            fy * 1.05,
        )
        capping_force = min(capping_force, fy * 1.60)
        capping_ratio = capping_force / fy
        residual_ratio = float(
            settings["censored_residual_strength_ratio"]
        )
        residual_force = min(
            capping_force * residual_ratio, 0.95 * fy
        )
        residual_ratio = residual_force / capping_force
        dr = max(
            float(settings["censored_minimum_residual_drift"]) * height,
            dc + 0.02 * height,
        )
        du = max(
            float(settings["censored_minimum_ultimate_drift"]) * height,
            dr + 0.02 * height,
        )
    elif not (dy < dc < dr < du):
        # A constrained, physically valid fallback is preferable to silently
        # emitting an unusable IMK material from a locally flat objective.
        dc = max(initial_dc, dy * 1.20)
        dr = max(initial_dr, dc * 1.20)
        du = max(initial_du, dr * 1.05)
    if not backbone_censored:
        capping_force = fy * capping_ratio
        residual_force = capping_force * residual_ratio
    if residual_force >= fy:
        residual_force = 0.95 * fy
        residual_ratio = residual_force / capping_force
    post_capping_slope = (residual_force - capping_force) / max(
        dr - dc, 1.0e-12
    )
    (
        cracking_deformation,
        post_cracking_stiffness,
        dy,
        _cracking_fit_error,
    ) = _fit_cracking_branch(
        np.concatenate(core_x_rows),
        np.concatenate(core_y_rows),
        np.concatenate(core_weight_rows),
        stiffness=stiffness,
        height=height,
        yield_force=fy,
        capping_force=capping_force,
        capping_deformation=dc,
        residual_force=residual_force,
        residual_deformation=dr,
        ultimate_deformation=du,
        settings=settings,
    )
    cracking_branch_force = (
        (stiffness - post_cracking_stiffness) * cracking_deformation
        if cracking_deformation > 0.0
        else 0.0
    )
    (
        cracking_deformation,
        post_cracking_stiffness,
        dy,
        cracking_branch_force,
    ) = _regularize_cracking_by_height(
        story=story,
        story_count=story_count,
        stiffness=stiffness,
        yield_force=fy,
        cracking_deformation=cracking_deformation,
        cracking_branch_force=cracking_branch_force,
        settings=settings,
    )
    if cracking_deformation > 0.0:
        predicted = _backbone_force_with_cracking(
            x,
            stiffness=stiffness,
            post_cracking_stiffness=post_cracking_stiffness,
            cracking_deformation=cracking_deformation,
            yield_force=fy,
            capping_force=capping_force,
            capping_deformation=dc,
            residual_force=residual_force,
            residual_deformation=dr,
            ultimate_deformation=du,
        )
    else:
        predicted = _backbone_force(
            x,
            stiffness=stiffness,
            yield_force=fy,
            capping_ratio=capping_ratio,
            capping_deformation=dc,
            residual_ratio=residual_ratio,
            residual_deformation=dr,
            ultimate_deformation=du,
        )
    fit_nrmse = float(
        np.sqrt(np.sum(weights * (predicted - y) ** 2) / np.sum(weights))
        / peak
    )
    pattern_errors = []
    for name, curve in curves.items():
        raw_deformation = np.abs(
            np.asarray(curve["story_deformation_m"], dtype=float)[:, story]
        )
        total = np.abs(
            np.asarray(curve["story_shear_kn"], dtype=float)[:, story]
        )
        deformation, material = _monotonic_story_envelope(
            raw_deformation,
            total - geometric * raw_deformation,
        )
        if deformation.size == 0:
            continue
        if cracking_deformation > 0.0:
            prediction = _backbone_force_with_cracking(
                deformation,
                stiffness=stiffness,
                post_cracking_stiffness=post_cracking_stiffness,
                cracking_deformation=cracking_deformation,
                yield_force=fy,
                capping_force=capping_force,
                capping_deformation=dc,
                residual_force=residual_force,
                residual_deformation=dr,
                ultimate_deformation=du,
            )
        else:
            prediction = _backbone_force(
                deformation,
                stiffness=stiffness,
                yield_force=fy,
                capping_ratio=capping_ratio,
                capping_deformation=dc,
                residual_ratio=residual_ratio,
                residual_deformation=dr,
                ultimate_deformation=du,
            )
        pattern_errors.append(
            float(
                np.sqrt(np.mean((prediction - material) ** 2))
                / max(np.max(material), 1.0e-12)
            )
        )
    return StorySpringParameters(
        story=story + 1,
        height_m=float(height),
        initial_stiffness_kn_m=float(stiffness),
        yield_force_kn=float(fy),
        yield_deformation_m=float(dy),
        capping_force_kn=float(capping_force),
        capping_deformation_m=float(dc),
        post_capping_slope_kn_m=float(post_capping_slope),
        residual_force_kn=float(residual_force),
        ultimate_deformation_m=float(du),
        lambda_s=float(cyclic["lambda_s"]),
        lambda_c=float(cyclic["lambda_c"]),
        lambda_a=float(cyclic["lambda_a"]),
        lambda_k=float(cyclic["lambda_k"]),
        c_s=float(cyclic["c_s"]),
        c_c=float(cyclic["c_c"]),
        c_a=float(cyclic["c_a"]),
        c_k=float(cyclic["c_k"]),
        gravity_load_above_kn=float(gravity_above),
        geometric_stiffness_kn_m=float(geometric),
        fit_nrmse=fit_nrmse,
        pattern_dispersion=float(np.std(pattern_errors)),
        backbone_censored=backbone_censored,
        postpeak_pattern_count=postpeak_pattern_count,
        observed_max_deformation_m=maximum_deformation,
        higher_mode_compatibility_nrmse=(
            float(np.mean(tuple(compatibility_errors.values())))
            if compatibility_errors
            else 0.0
        ),
        effective_higher_mode_weight=float(
            sum(
                value
                for name, value in effective_weights.items()
                if name.startswith("mode_")
            )
        ),
        cracking_deformation_m=cracking_deformation,
        post_cracking_stiffness_kn_m=post_cracking_stiffness,
        cracking_branch_force_kn=cracking_branch_force,
    )


def _raw_curve_path(
    config: dict[str, Any],
    building_id: str,
    direction: str,
) -> Path:
    return (
        Path(config["run_dir"])
        / str(calibration_config(config)["raw_curve_subdir"])
        / building_id
        / f"{direction.lower()}_curves.npz"
    )


def _save_raw_curves(
    path: Path,
    probes: dict[str, dict[str, Any]],
    nonlinear: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for family, curves in (("elastic", probes), ("nonlinear", nonlinear)):
        for name, curve in curves.items():
            for key in (
                "pattern",
                "floor_displacement_m",
                "story_deformation_m",
                "story_shear_kn",
                "roof_displacement_m",
                "base_shear_kn",
                "load_factor",
                "equilibrium_error",
            ):
                arrays[f"{family}__{name}__{key}"] = np.asarray(
                    curve[key], dtype=np.float64
                )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _calibrate_direction(
    building: dict[str, Any],
    config: dict[str, Any],
    *,
    info: ModelInfo,
    modal: ModalResult,
    direction: str,
) -> DirectionalReduction:
    settings = calibration_config(config)
    story_count = len(info.master_nodes)
    targets = _directional_modal_targets(modal, direction, story_count)
    masses = np.asarray(
        info.floor_masses_x_kn_s2_m
        if direction == "X"
        else info.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    elevations = np.asarray(info.floor_elevations_m, dtype=float)
    heights = np.diff(np.concatenate(([0.0], elevations)))
    gravity_floor = np.asarray(info.gravity_floor_loads_kn, dtype=float)
    gravity_above = np.cumsum(gravity_floor[::-1])[::-1]
    patterns = _calibration_patterns(
        targets, masses, elevations, settings
    )
    probes = {
        name: _run_full3d_static_pattern(
            building,
            config,
            direction=direction,
            pattern_name=name,
            pattern=pattern,
            nonlinear=False,
        )
        for name, pattern in patterns.items()
    }
    stiffness, elastic_metrics = _fit_elastic_stiffness(
        probes,
        patterns,
        targets,
        masses,
        heights,
        gravity_above,
        settings,
    )
    nonlinear = {
        name: _run_full3d_static_pattern(
            building,
            config,
            direction=direction,
            pattern_name=name,
            pattern=pattern,
            nonlinear=True,
        )
        for name, pattern in patterns.items()
    }
    cyclic = dict(settings["cyclic_defaults"])
    springs = tuple(
        _fit_story_spring(
            story,
            story_count,
            float(stiffness[story]),
            float(heights[story]),
            float(gravity_above[story]),
            nonlinear,
            settings["pattern_weights"],
            cyclic,
            int(settings["nonlinear_drift_bin_count"]),
            settings,
        )
        for story in range(story_count)
    )
    raw_path = _raw_curve_path(
        config, str(building["building_id"]), direction
    )
    _save_raw_curves(raw_path, probes, nonlinear)
    nonlinear_nrmse = float(
        np.median([spring.fit_nrmse for spring in springs])
    )
    return DirectionalReduction(
        direction=direction,
        target_periods_s=tuple(targets["periods_s"]),
        target_mode_shapes=tuple(targets["mode_shapes"]),
        target_effective_mass_ratios=tuple(
            targets["effective_mass_ratios"]
        ),
        target_participation_factors=tuple(
            targets["participation_factors"]
        ),
        story_stiffnesses_kn_m=tuple(float(value) for value in stiffness),
        springs=springs,
        calibration_patterns=tuple(patterns),
        elastic_objective=elastic_metrics["elastic_objective"],
        elastic_drift_nrmse=elastic_metrics["elastic_drift_nrmse"],
        first_mode_mac=elastic_metrics["first_mode_mac"],
        first_period_error=elastic_metrics["first_period_error"],
        nonlinear_backbone_nrmse=nonlinear_nrmse,
        elastic_core_drift_nrmse=elastic_metrics[
            "elastic_core_drift_nrmse"
        ],
        elastic_higher_mode_drift_nrmse=elastic_metrics[
            "elastic_higher_mode_drift_nrmse"
        ],
        elastic_fit_selection=elastic_metrics["elastic_fit_selection"],
        reduced_periods_s=elastic_metrics["reduced_periods_s"],
        mode_period_errors=elastic_metrics["mode_period_errors"],
        mode_macs=elastic_metrics["mode_macs"],
        raw_curve_path=str(raw_path.absolute()),
    )


def build_one_artifact(
    building: dict[str, Any],
    config: dict[str, Any],
) -> MDOFArtifact:
    """Calibrate and persist one uncoupled bidirectional 2N artifact."""
    calibration_started = time.perf_counter()
    settings = calibration_config(config)
    if bool(mdof_config(config)["shear_failure_enabled"]):
        raise ValueError("MDOF v1 does not permit shear failure")
    info = build_full3d_model(building, config)
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"{building['building_id']}: gravity equilibrium exceeds 1%"
        )
    modal = modal_properties(
        info,
        mode_count=int(config["model"].get("modal_mode_count", 12)),
        solver=str(config["model"].get("eigen_solver", "fullGenLapack")),
        minimum_cumulative_mass_ratio=float(
            config["model"]["minimum_cumulative_translational_mass_ratio"]
        ),
    )
    direction_x = _calibrate_direction(
        building,
        config,
        info=info,
        modal=modal,
        direction="X",
    )
    direction_y = _calibrate_direction(
        building,
        config,
        info=info,
        modal=modal,
        direction="Y",
    )
    elevations = tuple(float(value) for value in info.floor_elevations_m)
    heights = tuple(
        float(value)
        for value in np.diff(np.concatenate(([0.0], elevations)))
    )
    flags: list[str] = ["cyclic-calibration-pending"]
    cracking_regularization_enabled = bool(
        settings.get("cracking_regularization_enabled", False)
    )
    flags.append(
        (
            "empirical-cracking-height-regularization-enabled"
            if cracking_regularization_enabled
            else "empirical-cracking-height-regularization-disabled"
        )
    )
    if modal.period_error > 0.01:
        flags.append("xy-period-mismatch")
    for reduction, masses in (
        (direction_x, np.asarray(info.floor_masses_x_kn_s2_m, dtype=float)),
        (direction_y, np.asarray(info.floor_masses_y_kn_s2_m, dtype=float)),
    ):
        geometric = np.asarray(
            [
                spring.geometric_stiffness_kn_m
                for spring in reduction.springs
            ],
            dtype=float,
        )
        periods, _shapes = _reduced_modes(
            np.asarray(reduction.story_stiffnesses_kn_m, dtype=float),
            geometric,
            masses,
        )
        significant_incompatibility = any(
            abs(
                float(periods[mode])
                - float(reduction.target_periods_s[mode])
            )
            / float(reduction.target_periods_s[mode])
            > 0.05
            for mode in range(
                1,
                min(
                    len(periods),
                    len(reduction.target_periods_s),
                    len(reduction.target_effective_mass_ratios),
                    3,
                ),
            )
        )
        if significant_incompatibility:
            flags.append(
                f"three-mode-period-incompatibility-"
                f"{reduction.direction.lower()}"
            )
    artifact = MDOFArtifact(
        building_id=str(building["building_id"]),
        source_model_hash=str(building.get("model_hash", "")),
        source_signature=calibration_signature(building, config),
        story_count=len(elevations),
        floor_elevations_m=elevations,
        story_heights_m=heights,
        floor_masses_x_kn_s2_m=tuple(
            float(value) for value in info.floor_masses_x_kn_s2_m
        ),
        floor_masses_y_kn_s2_m=tuple(
            float(value) for value in info.floor_masses_y_kn_s2_m
        ),
        gravity_floor_loads_kn=tuple(
            float(value) for value in info.gravity_floor_loads_kn
        ),
        direction_x=direction_x,
        direction_y=direction_y,
        applicability_flags=tuple(flags),
        calibration_metrics={
            "gravity_equilibrium_error": gravity_error,
            "full3d_xy_period_error": modal.period_error,
            "cyclic_calibration_status": "pending",
            "cracking_height_regularization_enabled": (
                cracking_regularization_enabled
            ),
            "cracking_height_regularization_status": (
                "experimental_requires_cross-building-validation"
                if cracking_regularization_enabled
                else "disabled-to-prevent-single-building-overfit"
            ),
            "torsion_supported": False,
            "xy_strength_interaction_supported": False,
            "calibration_runtime_s": (
                time.perf_counter() - calibration_started
            ),
        },
        certified=False,
    ).with_hash()
    saved_path = save_artifact(artifact, config)
    with transaction(config["database_path"]) as connection:
        upsert_many(
            connection,
            "mdof_artifacts",
            [
                {
                    "building_id": artifact.building_id,
                    "artifact_path": str(saved_path.absolute()),
                    "artifact_hash": artifact.artifact_hash,
                    "source_signature": artifact.source_signature,
                    "schema_version": artifact.schema_version,
                    "story_count": artifact.story_count,
                    "certified": int(artifact.certified),
                    "certification_path": artifact.certification_path,
                    "applicability_flags_json": json.dumps(
                        artifact.applicability_flags, sort_keys=True
                    ),
                    "calibration_metrics_json": json.dumps(
                        artifact.calibration_metrics, sort_keys=True
                    ),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            ],
            ("building_id",),
        )
    return artifact


def build_artifacts(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    buildings = _load_buildings(
        config, building_ids=building_ids, limit=limit
    )
    results = []
    failures = []
    skipped = 0
    for building in buildings:
        building_id = str(building["building_id"])
        path = artifact_path(config, building_id)
        expected = calibration_signature(building, config)
        if resume and path.is_file():
            try:
                existing = MDOFArtifact.from_dict(read_json(path))
                existing.validate()
                if existing.source_signature == expected:
                    skipped += 1
                    results.append(existing.to_dict())
                    continue
            except Exception:
                pass
        try:
            results.append(build_one_artifact(building, config).to_dict())
        except Exception as exc:
            failures.append(
                {
                    "building_id": building_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return {
        "building_count": len(buildings),
        "completed_count": len(results) - skipped,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "failures": failures,
        "artifacts": results,
    }
