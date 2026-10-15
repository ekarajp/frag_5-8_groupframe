"""Targeted Full3D diagnostics for an axial-interaction-aware Fishbone v4.

The production Full3D and Fishbone-v3 models are deliberately left unchanged.
This module reruns only selected *development* modal-static cases and records
member-level states that were not present in the original reusable modal-SPO
cache.  Its artifacts therefore have their own schema and analysis signature.

Units are kN, m, seconds and radians.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import column_pm_capacity_at_axial
from .constants import KSC_TO_KN_M2
from .mdof_calibration import (
    _analyze_displacement_step,
    _configure_static,
    _story_cut_shears,
    _story_displacements,
    calibration_config,
)
from .structural import (
    ModelInfo,
    _maximum_longitudinal_bar_strain,
    _mechanism_longitudinal_bar_positions,
    _ops,
    build_model,
    run_gravity,
)


FISHBONE_V4_COMPONENT_DIAGNOSTIC_SCHEMA_VERSION = (
    "fishbone-v4-full3d-modal-component-diagnostic-v2-signed-axial-pm-range"
)


def _column_grid_indices(info: ModelInfo) -> tuple[tuple[int, int], ...]:
    grid_size = int(info.number_of_bays) + 1
    expected = grid_size**2
    if info.columns_per_storey != expected:
        raise ValueError("Full3D column ordering is inconsistent with its grid")
    return tuple(
        (index // grid_size, index % grid_size) for index in range(expected)
    )


def _member_end_yield_ratios(
    element_tags_by_storey: tuple[tuple[int, ...], ...],
    bar_positions: tuple[tuple[float, float], ...],
    yield_strain: float,
) -> np.ndarray:
    """Return [story, member, I/J] maximum reinforcing-strain ratios."""

    ops = _ops()
    rows: list[list[list[float]]] = []
    for tags in element_tags_by_storey:
        story_rows: list[list[float]] = []
        for tag in tags:
            point_count = len(ops.eleResponse(tag, "integrationPoints"))
            if point_count < 2:
                raise ValueError(f"Element {tag} has fewer than two sections")
            end_rows = []
            for section_number in (1, point_count):
                deformation = tuple(
                    float(value)
                    for value in ops.eleResponse(
                        tag, "section", section_number, "deformation"
                    )
                )
                end_rows.append(
                    _maximum_longitudinal_bar_strain(
                        deformation, bar_positions
                    )
                    / yield_strain
                )
            story_rows.append(end_rows)
        rows.append(story_rows)
    return np.asarray(rows, dtype=float)


def _column_state(
    info: ModelInfo,
    *,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return axial compression, directional end moments and curvatures."""

    ops = _ops()
    normalized = direction.upper()
    # For vertical members created with vecxz=(1,0,0), global-X sway bends
    # about local y and global-Y sway bends about local z.
    moment_index = 4 if normalized == "X" else 5
    curvature_index = 2 if normalized == "X" else 1
    axial_rows: list[list[float]] = []
    moment_rows: list[list[list[float]]] = []
    curvature_rows: list[list[list[float]]] = []
    for tags in info.column_element_tags_by_storey:
        story_axial: list[float] = []
        story_moment: list[list[float]] = []
        story_curvature: list[list[float]] = []
        for tag in tags:
            local = tuple(float(value) for value in ops.eleResponse(tag, "localForce"))
            if len(local) != 12:
                raise ValueError(f"Column {tag} local force vector is incomplete")
            # OpenSees local end forces have opposite signs at the two ends.
            # Positive local-I axial force is compression for this model.
            # Retain the sign so an uplift/tension state cannot be mistaken
            # for additional compression by an absolute-value envelope.
            story_axial.append(0.5 * (local[0] - local[6]))
            story_moment.append(
                [abs(local[moment_index]), abs(local[6 + moment_index])]
            )
            point_count = len(ops.eleResponse(tag, "integrationPoints"))
            end_curvatures = []
            for section_number in (1, point_count):
                deformation = tuple(
                    float(value)
                    for value in ops.eleResponse(
                        tag, "section", section_number, "deformation"
                    )
                )
                if len(deformation) <= curvature_index:
                    raise ValueError(
                        f"Column {tag} section deformation is incomplete"
                    )
                end_curvatures.append(float(deformation[curvature_index]))
            story_curvature.append(end_curvatures)
        axial_rows.append(story_axial)
        moment_rows.append(story_moment)
        curvature_rows.append(story_curvature)
    return (
        np.asarray(axial_rows, dtype=float),
        np.asarray(moment_rows, dtype=float),
        np.asarray(curvature_rows, dtype=float),
    )


def _pm_capacity(
    building: dict[str, Any],
    config: dict[str, Any],
    axial_compression_kn: float,
) -> float:
    metadata_raw = building.get("design_metadata_json", "{}")
    metadata = (
        json.loads(metadata_raw or "{}")
        if isinstance(metadata_raw, str)
        else dict(metadata_raw)
    )
    detailing = config["model"]["material_detailing"]
    result = column_pm_capacity_at_axial(
        width_m=float(building["column_b_m"]),
        depth_m=float(building["column_h_m"]),
        bar_count=int(building["column_bar_count"]),
        bar_diameter_m=float(building["column_bar_diameter_m"]),
        clear_cover_m=float(detailing["clear_cover_m"]),
        hoop_diameter_m=float(
            metadata.get(
                "column_hoop_diameter_m",
                detailing["column_hoop_diameter_m"],
            )
        ),
        fc_kn_m2=float(building["fc_ksc"]) * KSC_TO_KN_M2,
        fy_kn_m2=float(
            building.get(
                "steel_fy_ksc", config["building"]["steel_fy_ksc"]
            )
        )
        * KSC_TO_KN_M2,
        steel_elastic_modulus_mpa=float(
            config["model"]["steel_hysteretic"]["elastic_modulus_mpa"]
        ),
        target_axial_kn=max(float(axial_compression_kn), 0.0),
        strength_mode="probable",
        probable_steel_strength_factor=float(
            config["building"]["capacity_design_shear"].get(
                "probable_steel_strength_factor", 1.25
            )
        ),
    )
    return float(result["moment_capacity_knm"])


def summarize_component_diagnostic(
    arrays: dict[str, np.ndarray],
    *,
    building: dict[str, Any],
    config: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    """Condense member histories into causal, model-design diagnostics."""

    axial = np.asarray(arrays["column_axial_compression_kn"], dtype=float)
    moments = np.asarray(arrays["column_directional_end_moment_knm"], dtype=float)
    column_yield = np.asarray(arrays["column_end_yield_strain_ratio"], dtype=float)
    beam_yield = np.asarray(arrays["active_beam_end_yield_strain_ratio"], dtype=float)
    drift = np.asarray(arrays["story_drift_ratio"], dtype=float)
    base = np.asarray(arrays["base_shear_kn"], dtype=float)
    gravity = np.maximum(axial[0], 1.0e-9)
    amplification = axial / gravity.reshape((1,) + gravity.shape)
    axial_change = (axial - gravity) / gravity.reshape((1,) + gravity.shape)
    peak_index = int(np.argmax(np.abs(base)))
    terminal_index = int(base.size - 1)
    maximum_midr_index = int(np.argmax(np.max(np.abs(drift), axis=1)))

    grid = _column_grid_indices_from_count(
        int(building.get("number_of_bays", 1)), axial.shape[2]
    )
    axis_coordinate = 0 if direction.upper() == "X" else 1
    rocking_coordinate = np.asarray(
        [indices[axis_coordinate] for indices in grid], dtype=float
    )
    centered = rocking_coordinate - float(np.mean(rocking_coordinate))
    rocking_correlations = []
    for step in range(axial.shape[0]):
        change = (axial[step] - gravity).reshape(-1)
        coordinate = np.tile(centered, axial.shape[1])
        if np.std(change) > 1.0e-12 and np.std(coordinate) > 0.0:
            rocking_correlations.append(
                abs(float(np.corrcoef(change, coordinate)[0, 1]))
            )

    gravity_capacity = np.empty_like(gravity)
    minimum_compression_capacity = np.empty_like(gravity)
    maximum_compression_capacity = np.empty_like(gravity)
    for story in range(gravity.shape[0]):
        for column in range(gravity.shape[1]):
            gravity_capacity[story, column] = _pm_capacity(
                building, config, float(gravity[story, column])
            )
            minimum_compression_capacity[story, column] = _pm_capacity(
                building, config, float(np.min(axial[:, story, column]))
            )
            maximum_compression_capacity[story, column] = _pm_capacity(
                building, config, float(np.max(axial[:, story, column]))
            )
    pm_ratios = np.stack(
        (minimum_compression_capacity, maximum_compression_capacity), axis=0
    ) / np.maximum(
        gravity_capacity.reshape((1,) + gravity_capacity.shape), 1.0e-12
    )

    def state(index: int) -> dict[str, Any]:
        return {
            "step_index": int(index),
            "base_shear_kn": float(base[index]),
            "maximum_midr": float(np.max(np.abs(drift[index]))),
            "critical_story": int(np.argmax(np.abs(drift[index])) + 1),
            "column_yielded_end_fraction": float(
                np.mean(column_yield[index] >= 1.0)
            ),
            "active_beam_yielded_end_fraction": float(
                np.mean(beam_yield[index] >= 1.0)
            ),
            "maximum_column_moment_knm": float(np.max(moments[index])),
            "maximum_axial_amplification": float(np.max(amplification[index])),
            "minimum_axial_amplification": float(np.min(amplification[index])),
        }

    story_rows = []
    for story in range(axial.shape[1]):
        story_rows.append(
            {
                "story": story + 1,
                "maximum_compression_amplification": float(
                    np.max(amplification[:, story])
                ),
                "maximum_unloading_fraction": float(
                    np.max(-axial_change[:, story])
                ),
                "maximum_abs_axial_change_fraction": float(
                    np.max(np.abs(axial_change[:, story]))
                ),
                "minimum_probable_pm_capacity_ratio_to_gravity": float(
                    np.min(pm_ratios[:, story])
                ),
                "maximum_probable_pm_capacity_ratio_to_gravity": float(
                    np.max(pm_ratios[:, story])
                ),
                "maximum_abs_probable_pm_capacity_change_fraction": float(
                    np.max(np.abs(pm_ratios[:, story] - 1.0))
                ),
                "maximum_column_yielded_end_fraction": float(
                    np.max(np.mean(column_yield[:, story] >= 1.0, axis=(1, 2)))
                ),
                "maximum_active_beam_yielded_end_fraction": float(
                    np.max(np.mean(beam_yield[:, story] >= 1.0, axis=(1, 2)))
                ),
            }
        )

    maximum_axial_change = float(np.max(np.abs(axial_change)))
    minimum_pm_ratio = float(np.min(pm_ratios))
    maximum_pm_ratio = float(np.max(pm_ratios))
    maximum_abs_pm_change = float(np.max(np.abs(pm_ratios - 1.0)))
    return {
        "maximum_axial_compression_amplification": float(
            np.max(amplification)
        ),
        "maximum_axial_unloading_fraction": float(np.max(-axial_change)),
        "maximum_abs_axial_change_fraction": maximum_axial_change,
        "minimum_probable_pm_capacity_ratio_to_gravity": minimum_pm_ratio,
        "maximum_probable_pm_capacity_ratio_to_gravity": maximum_pm_ratio,
        "maximum_abs_probable_pm_capacity_change_fraction": (
            maximum_abs_pm_change
        ),
        "maximum_abs_rocking_axial_correlation": float(
            max(rocking_correlations, default=0.0)
        ),
        "axial_interaction_materially_active": bool(
            maximum_axial_change >= 0.25 or maximum_abs_pm_change >= 0.10
        ),
        "fishbone_v3_limitation": (
            "column hinge backbones are fixed at gravity axial load; dynamic "
            "axial redistribution and its P-M capacity change are absent"
        ),
        "peak_base_shear_state": state(peak_index),
        "maximum_midr_state": state(maximum_midr_index),
        "terminal_state": state(terminal_index),
        "stories": story_rows,
    }


def _column_grid_indices_from_count(
    number_of_bays: int, column_count: int
) -> tuple[tuple[int, int], ...]:
    grid_size = int(number_of_bays) + 1
    if grid_size**2 != int(column_count):
        raise ValueError("Column count does not match the square plan grid")
    return tuple(
        (index // grid_size, index % grid_size)
        for index in range(column_count)
    )


def run_full3d_modal_component_diagnostic(
    building: dict[str, Any],
    config: dict[str, Any],
    *,
    direction: str,
    mode_number: int,
    modal_pattern: np.ndarray,
    reference_roof_limit_m: float,
    maximum_runtime_s: float | None = None,
) -> dict[str, Any]:
    """Rerun one cached modal pattern while retaining member response states."""

    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Diagnostic direction must be X or Y")
    if int(mode_number) not in {1, 2, 3}:
        raise ValueError("Diagnostic mode must be 1, 2 or 3")
    pattern = np.asarray(modal_pattern, dtype=float).reshape(-1)
    if pattern.size != int(building["stories"]):
        raise ValueError("Modal pattern length differs from the story count")
    if not np.all(np.isfinite(pattern)) or np.sum(np.abs(pattern)) <= 0.0:
        raise ValueError("Modal pattern is invalid")
    if reference_roof_limit_m <= 0.0:
        raise ValueError("Reference roof limit must be positive")

    settings = calibration_config(config)
    runtime_limit = float(
        maximum_runtime_s
        if maximum_runtime_s is not None
        else settings["maximum_static_runtime_s"]
    )
    started = time.perf_counter()
    info = build_model(building, config)
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError("Full3D diagnostic gravity error exceeds 1%")

    beam_positions, column_positions, layout_audit = (
        _mechanism_longitudinal_bar_positions(building, config)
    )
    fy_kn_m2 = float(
        building.get("steel_fy_ksc", config["building"]["steel_fy_ksc"])
    ) * KSC_TO_KN_M2
    es_kn_m2 = float(
        config["model"]["steel_hysteretic"]["elastic_modulus_mpa"]
    ) * 1000.0
    yield_strain = fy_kn_m2 / es_kn_m2
    dof = 1 if normalized == "X" else 2
    ops = _ops()
    ops.timeSeries("Linear", 60)
    ops.pattern("Plain", 60, 60)
    for node, force in zip(info.master_nodes, pattern):
        load = [0.0] * 6
        load[dof - 1] = float(force)
        ops.load(node, *load)
    _configure_static()
    control_node = info.master_nodes[-1]
    requested_increment = float(settings["nonlinear_displacement_increment_m"])
    minimum_increment = float(
        settings["nonlinear_minimum_displacement_increment_m"]
    )
    maximum_steps = int(settings["maximum_static_steps"])
    active_count = info.beams_per_storey // 2
    active_beam_tags = tuple(
        tuple(
            tags[:active_count] if normalized == "X" else tags[active_count:]
        )
        for tags in info.beam_element_tags_by_storey
    )

    histories: dict[str, list[np.ndarray | float]] = {
        "floor_displacement_m": [],
        "story_deformation_m": [],
        "story_drift_ratio": [],
        "story_shear_kn": [],
        "base_shear_kn": [],
        "load_factor": [],
        "column_axial_compression_kn": [],
        "column_directional_end_moment_knm": [],
        "column_directional_end_curvature_1_m": [],
        "column_end_yield_strain_ratio": [],
        "active_beam_end_yield_strain_ratio": [],
    }

    heights = np.diff(
        np.concatenate(([0.0], np.asarray(info.floor_elevations_m, dtype=float)))
    )

    def record_state() -> None:
        floor, deformation = _story_displacements(info, dof)
        factor = float(ops.getLoadFactor(60))
        shear, base, _equilibrium = _story_cut_shears(
            info, dof, load_factor=factor
        )
        axial, moments, curvatures = _column_state(info, direction=normalized)
        histories["floor_displacement_m"].append(floor)
        histories["story_deformation_m"].append(deformation)
        histories["story_drift_ratio"].append(deformation / heights)
        histories["story_shear_kn"].append(shear)
        histories["base_shear_kn"].append(float(base))
        histories["load_factor"].append(factor)
        histories["column_axial_compression_kn"].append(axial)
        histories["column_directional_end_moment_knm"].append(moments)
        histories["column_directional_end_curvature_1_m"].append(curvatures)
        histories["column_end_yield_strain_ratio"].append(
            _member_end_yield_ratios(
                info.column_element_tags_by_storey,
                column_positions,
                yield_strain,
            )
        )
        histories["active_beam_end_yield_strain_ratio"].append(
            _member_end_yield_ratios(
                active_beam_tags,
                beam_positions,
                yield_strain,
            )
        )

    record_state()
    termination = "reference_roof_limit_reached"
    reduction_levels: list[int] = []
    accepted_steps = 0
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
        accepted_steps += 1
        record_state()
        roof = abs(float(histories["floor_displacement_m"][-1][-1]))
        if roof >= reference_roof_limit_m - 0.5 * minimum_increment:
            break
        base_values = np.abs(np.asarray(histories["base_shear_kn"], dtype=float))
        if (
            base_values.size > 10
            and float(np.max(base_values)) > 0.0
            and base_values[-1] <= 0.20 * float(np.max(base_values))
            and base_values[-2] > base_values[-1]
        ):
            termination = "postpeak_20pct_endpoint"
            break
        if time.perf_counter() - started >= runtime_limit:
            termination = "runtime_guard"
            break
        requested_increment = min(requested_increment, accepted_increment)
    else:
        termination = "step_guard"

    arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in histories.items()
    }
    summary = summarize_component_diagnostic(
        arrays,
        building=building,
        config=config,
        direction=normalized,
    )
    return {
        "schema_version": FISHBONE_V4_COMPONENT_DIAGNOSTIC_SCHEMA_VERSION,
        "building_id": str(building["building_id"]),
        "source_model_hash": str(building.get("model_hash", "")),
        "direction": normalized,
        "mode_number": int(mode_number),
        "modal_pattern": pattern.tolist(),
        "reference_roof_limit_m": float(reference_roof_limit_m),
        "gravity_error": float(gravity_error),
        "termination_reason": termination,
        "accepted_steps": accepted_steps,
        "maximum_step_reduction_level": max(reduction_levels, default=0),
        "runtime_s": float(time.perf_counter() - started),
        "yield_strain": float(yield_strain),
        "layout_audit": layout_audit,
        "column_grid_indices": [list(value) for value in _column_grid_indices(info)],
        "summary": summary,
        "_history_arrays": arrays,
    }
