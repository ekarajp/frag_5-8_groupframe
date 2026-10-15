"""Hierarchical calibration for the independent Fishbone-v2 model.

Elastic and monotonic stages are intentionally separate.  Both use Full 3D
quasi-static evidence only; no ground-motion response enters the objective.
The returned artifact remains uncertified until cyclic and blind holdout
certification pass.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .fishbone import (
    DirectionalModalTargets,
    FishboneArtifact,
    FishboneModalResult,
    HingeBackbone,
    build_model,
    modal_properties,
    run_gravity,
)
from .fishbone_spo import FishboneSPOResult, run_pattern_spo

FISHBONE_ELASTIC_CALIBRATION_SCHEMA_VERSION = (
    "ifb3-elastic-modal-1to3-bounded-regularised-v1"
)
FISHBONE_MONOTONIC_CALIBRATION_SCHEMA_VERSION = (
    "ifb3-monotonic-five-pattern-three-height-control-v1"
)


@dataclass(frozen=True)
class NonlinearCalibrationParameters:
    """Small, height-normalised parameter set valid for any story count."""

    strength_low: float = 1.0
    strength_mid: float = 1.0
    strength_high: float = 1.0
    pre_capping_rotation_scale: float = 1.0
    post_capping_rotation_scale: float = 1.0
    column_to_beam_strength_ratio: float = 1.0
    capping_to_yield_scale: float = 1.0

    def validate(self) -> None:
        values = (
            self.strength_low,
            self.strength_mid,
            self.strength_high,
            self.pre_capping_rotation_scale,
            self.post_capping_rotation_scale,
            self.column_to_beam_strength_ratio,
            self.capping_to_yield_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Fishbone nonlinear calibration is invalid")


@dataclass(frozen=True)
class StaticStoryCalibrationRecipe:
    """Population-level correction identified from development buildings.

    Only three height-normalised stiffness and strength controls are used,
    irrespective of the number of stories.  Rotation controls are shared by
    the whole population.  This deliberately prevents one free parameter per
    building/story and makes blind-holdout prediction possible.
    """

    stiffness_low: float = 1.0
    stiffness_mid: float = 1.0
    stiffness_high: float = 1.0
    nonlinear: NonlinearCalibrationParameters = (
        NonlinearCalibrationParameters()
    )

    def validate(self) -> None:
        stiffness = (
            self.stiffness_low,
            self.stiffness_mid,
            self.stiffness_high,
        )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in stiffness
        ):
            raise ValueError("Fishbone static stiffness calibration is invalid")
        self.nonlinear.validate()


def apply_static_story_recipe(
    artifact: FishboneArtifact,
    direction: str,
    recipe: StaticStoryCalibrationRecipe,
) -> FishboneArtifact:
    """Apply one generic development-population recipe to an artifact."""

    artifact.validate()
    recipe.validate()
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Fishbone direction must be X or Y")
    profile = _height_profile(
        artifact.story_count,
        recipe.stiffness_low,
        recipe.stiffness_mid,
        recipe.stiffness_high,
    )
    stories = []
    for index, story in enumerate(artifact.stories):
        scale = float(profile[index])
        if normalized == "X":
            stories.append(
                replace(
                    story,
                    column_stiffness_modifier_x=(
                        story.column_stiffness_modifier_x * scale
                    ),
                    beam_stiffness_modifier_x=(
                        story.beam_stiffness_modifier_x * scale
                    ),
                )
            )
        else:
            stories.append(
                replace(
                    story,
                    column_stiffness_modifier_y=(
                        story.column_stiffness_modifier_y * scale
                    ),
                    beam_stiffness_modifier_y=(
                        story.beam_stiffness_modifier_y * scale
                    ),
                )
            )
    stiffness_corrected = replace(
        artifact, stories=tuple(stories), artifact_hash=""
    )
    return apply_nonlinear_parameters(
        stiffness_corrected,
        normalized,
        recipe.nonlinear,
    )


def _height_profile(
    story_count: int,
    low: float,
    middle: float,
    high: float,
) -> np.ndarray:
    if story_count < 1:
        raise ValueError("Fishbone height profile requires at least one story")
    if story_count == 1:
        return np.asarray((middle,), dtype=float)
    z = np.linspace(0.0, 1.0, story_count)
    return np.exp(
        np.interp(
            z,
            (0.0, 0.5, 1.0),
            np.log((low, middle, high)),
        )
    )


def _scale_hinge(
    hinge: HingeBackbone,
    *,
    strength_scale: float,
    pre_scale: float,
    post_scale: float,
    capping_to_yield_scale: float,
    source: str,
) -> HingeBackbone:
    pre_pos = hinge.pre_capping_rotation_pos_rad * pre_scale
    pre_neg = hinge.pre_capping_rotation_neg_rad * pre_scale
    post_pos = hinge.post_capping_rotation_pos_rad * post_scale
    post_neg = hinge.post_capping_rotation_neg_rad * post_scale
    ultimate_pos = max(
        hinge.ultimate_rotation_pos_rad * max(pre_scale, post_scale),
        1.05 * (pre_pos + post_pos),
    )
    ultimate_neg = max(
        hinge.ultimate_rotation_neg_rad * max(pre_scale, post_scale),
        1.05 * (pre_neg + post_neg),
    )
    return replace(
        hinge,
        yield_moment_pos_knm=(
            hinge.yield_moment_pos_knm * strength_scale
        ),
        yield_moment_neg_knm=(
            hinge.yield_moment_neg_knm * strength_scale
        ),
        capping_to_yield_pos=(
            hinge.capping_to_yield_pos * capping_to_yield_scale
        ),
        capping_to_yield_neg=(
            hinge.capping_to_yield_neg * capping_to_yield_scale
        ),
        pre_capping_rotation_pos_rad=pre_pos,
        pre_capping_rotation_neg_rad=pre_neg,
        post_capping_rotation_pos_rad=post_pos,
        post_capping_rotation_neg_rad=post_neg,
        ultimate_rotation_pos_rad=ultimate_pos,
        ultimate_rotation_neg_rad=ultimate_neg,
        source=source,
    )


def apply_nonlinear_parameters(
    artifact: FishboneArtifact,
    direction: str,
    parameters: NonlinearCalibrationParameters,
) -> FishboneArtifact:
    """Apply a smooth three-control height profile to member hinges.

    The number of fitted controls does not increase with the number of
    stories.  This is deliberate: it prevents a five-storey calibration from
    becoming five unrelated story knobs and makes the same equation usable
    for taller and shorter buildings.
    """

    artifact.validate()
    parameters.validate()
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Fishbone direction must be X or Y")
    profile = _height_profile(
        artifact.story_count,
        parameters.strength_low,
        parameters.strength_mid,
        parameters.strength_high,
    )
    allocation = math.sqrt(
        parameters.column_to_beam_strength_ratio
    )
    stories = []
    for index, story in enumerate(artifact.stories):
        common = float(profile[index])
        column_scale = common * allocation
        beam_scale = common / allocation
        source = (
            f"{FISHBONE_MONOTONIC_CALIBRATION_SCHEMA_VERSION}:"
            f"{normalized}"
        )
        if normalized == "X":
            stories.append(
                replace(
                    story,
                    column_bottom_hinge_x=_scale_hinge(
                        story.column_bottom_hinge_x,
                        strength_scale=column_scale,
                        pre_scale=parameters.pre_capping_rotation_scale,
                        post_scale=parameters.post_capping_rotation_scale,
                        capping_to_yield_scale=parameters.capping_to_yield_scale,
                        source=source,
                    ),
                    column_top_hinge_x=_scale_hinge(
                        story.column_top_hinge_x,
                        strength_scale=column_scale,
                        pre_scale=parameters.pre_capping_rotation_scale,
                        post_scale=parameters.post_capping_rotation_scale,
                        capping_to_yield_scale=parameters.capping_to_yield_scale,
                        source=source,
                    ),
                    beam_hinge_x=_scale_hinge(
                        story.beam_hinge_x,
                        strength_scale=beam_scale,
                        pre_scale=parameters.pre_capping_rotation_scale,
                        post_scale=parameters.post_capping_rotation_scale,
                        capping_to_yield_scale=parameters.capping_to_yield_scale,
                        source=source,
                    ),
                )
            )
        else:
            stories.append(
                replace(
                    story,
                    column_bottom_hinge_y=_scale_hinge(
                        story.column_bottom_hinge_y,
                        strength_scale=column_scale,
                        pre_scale=parameters.pre_capping_rotation_scale,
                        post_scale=parameters.post_capping_rotation_scale,
                        capping_to_yield_scale=parameters.capping_to_yield_scale,
                        source=source,
                    ),
                    column_top_hinge_y=_scale_hinge(
                        story.column_top_hinge_y,
                        strength_scale=column_scale,
                        pre_scale=parameters.pre_capping_rotation_scale,
                        post_scale=parameters.post_capping_rotation_scale,
                        capping_to_yield_scale=parameters.capping_to_yield_scale,
                        source=source,
                    ),
                    beam_hinge_y=_scale_hinge(
                        story.beam_hinge_y,
                        strength_scale=beam_scale,
                        pre_scale=parameters.pre_capping_rotation_scale,
                        post_scale=parameters.post_capping_rotation_scale,
                        capping_to_yield_scale=parameters.capping_to_yield_scale,
                        source=source,
                    ),
                )
            )
    result = replace(
        artifact, stories=tuple(stories), artifact_hash=""
    ).with_hash()
    result.validate()
    return result


def load_full3d_monotonic_evidence(
    path: Path | str,
) -> dict[str, dict[str, np.ndarray]]:
    """Load the five cached Full 3D nonlinear static curves."""

    names = ("first_mode", "uniform", "triangular", "mode_2", "mode_3")
    evidence: dict[str, dict[str, np.ndarray]] = {}
    with np.load(Path(path)) as values:
        for name in names:
            prefix = f"nonlinear__{name}__"
            required = (
                "pattern",
                "floor_displacement_m",
                "story_deformation_m",
                "story_shear_kn",
                "roof_displacement_m",
                "base_shear_kn",
                "load_factor",
            )
            if all(f"{prefix}{key}" in values.files for key in required):
                evidence[name] = {
                    key: np.asarray(values[f"{prefix}{key}"], dtype=float)
                    for key in required
                }
    if "first_mode" not in evidence:
        raise ValueError("Full 3D first-mode nonlinear evidence is missing")
    return evidence


def _curve_values(
    arrays: dict[str, np.ndarray],
    control_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacement = np.abs(
        np.asarray(arrays["floor_displacement_m"], dtype=float)[
            :, control_index
        ]
    )
    force = np.abs(np.asarray(arrays["load_factor"], dtype=float))
    drift = np.abs(
        np.asarray(arrays["story_deformation_m"], dtype=float)
    )
    order = np.argsort(displacement, kind="stable")
    displacement = displacement[order]
    force = force[order]
    drift = drift[order]
    unique, indices = np.unique(displacement, return_index=True)
    return unique, force[indices], drift[indices]


def compare_monotonic_curve(
    full3d: dict[str, np.ndarray],
    fishbone: FishboneSPOResult,
    story_heights_m: tuple[float, ...] | np.ndarray,
) -> dict[str, Any]:
    """Compare strength, deformation and drift distribution for one SPO."""

    heights = np.asarray(story_heights_m, dtype=float)
    pattern = np.asarray(full3d["pattern"], dtype=float)
    control = int(np.argmax(np.abs(pattern)))
    target_x, target_y, target_deformation = _curve_values(full3d, control)
    reduced_x, reduced_y, reduced_deformation = _curve_values(
        fishbone.arrays, control
    )
    target_peak_index = int(np.argmax(target_y))
    reduced_peak_index = int(np.argmax(reduced_y))
    target_peak = float(target_y[target_peak_index])
    reduced_peak = float(reduced_y[reduced_peak_index])
    target_peak_x = float(target_x[target_peak_index])
    reduced_peak_x = float(reduced_x[reduced_peak_index])
    common_end = min(float(target_x[-1]), float(reduced_x[-1]))
    sample = np.linspace(0.0, max(common_end, 1.0e-12), 101)
    target_sample = np.interp(sample, target_x, target_y)
    reduced_sample = np.interp(sample, reduced_x, reduced_y)
    force_scale = max(target_peak, 1.0e-12)
    curve_nrmse = float(
        np.sqrt(np.mean((reduced_sample - target_sample) ** 2))
        / force_scale
    )
    target_idr = target_deformation[target_peak_index] / heights
    reduced_idr = reduced_deformation[reduced_peak_index] / heights
    drift_scale = max(
        float(np.sqrt(np.mean(target_idr**2))), 1.0e-12
    )
    drift_nrmse = float(
        np.sqrt(np.mean((reduced_idr - target_idr) ** 2))
        / drift_scale
    )
    target_midr = float(np.max(target_idr))
    reduced_midr = float(np.max(reduced_idr))

    def signed_relative(observed: float, reference: float) -> float:
        return float(
            (observed - reference) / max(abs(reference), 1.0e-12)
        )

    return {
        "pattern_name": fishbone.pattern_name,
        "peak_force_signed_relative_error": signed_relative(
            reduced_peak, target_peak
        ),
        "peak_control_displacement_signed_relative_error": signed_relative(
            reduced_peak_x, target_peak_x
        ),
        "peak_midr_signed_relative_error": signed_relative(
            reduced_midr, target_midr
        ),
        "peak_drift_profile_nrmse": drift_nrmse,
        "curve_force_nrmse": curve_nrmse,
        "endpoint_coverage": float(
            min(float(reduced_x[-1]) / max(float(target_x[-1]), 1.0e-12), 1.0)
        ),
        "full3d_critical_story": int(np.argmax(target_idr)) + 1,
        "fishbone_critical_story": int(np.argmax(reduced_idr)) + 1,
        "critical_story_match": bool(
            int(np.argmax(target_idr)) == int(np.argmax(reduced_idr))
        ),
        "fishbone_status": fishbone.status,
    }


def monotonic_evidence_score(
    metrics: dict[str, dict[str, Any]],
) -> float:
    """Drift-first scalar used by the bounded low-dimensional optimiser."""

    weights = {
        "first_mode": 1.00,
        "mode_2": 0.90,
        "mode_3": 0.90,
        "uniform": 0.70,
        "triangular": 0.70,
    }
    total = 0.0
    total_weight = 0.0
    for name, row in metrics.items():
        weight = weights.get(name, 0.5)
        score = (
            0.34 * min(float(row["peak_drift_profile_nrmse"]), 5.0)
            + 0.20
            * min(
                abs(float(row["peak_midr_signed_relative_error"])), 5.0
            )
            + 0.16
            * min(
                abs(
                    float(
                        row[
                            "peak_control_displacement_signed_relative_error"
                        ]
                    )
                ),
                5.0,
            )
            + 0.14
            * min(
                abs(float(row["peak_force_signed_relative_error"])), 5.0
            )
            + 0.10 * min(float(row["curve_force_nrmse"]), 5.0)
            + 0.04 * (0.0 if row["critical_story_match"] else 1.0)
            + 0.02 * (1.0 - float(row["endpoint_coverage"]))
        )
        if row["fishbone_status"] in {
            "numerical_failure",
            "runtime_guard",
        }:
            score += 1.0
        total += weight * score
        total_weight += weight
    return float(total / max(total_weight, 1.0e-12))


def _fit_population_height_profile(
    normalized_heights: np.ndarray,
    ratios: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    """Robustly fit three log-linear height controls to positive ratios."""

    z = np.asarray(normalized_heights, dtype=float)
    observed = np.asarray(ratios, dtype=float)
    valid = (
        np.isfinite(z)
        & np.isfinite(observed)
        & (z >= 0.0)
        & (z <= 1.0)
        & (observed > 0.0)
    )
    z = z[valid]
    observed = np.clip(observed[valid], lower, upper)
    if observed.size < 3:
        raise ValueError("At least three valid story ratios are required")

    def interpolate(log_controls: np.ndarray) -> np.ndarray:
        return np.interp(z, (0.0, 0.5, 1.0), log_controls)

    def residual(log_controls: np.ndarray) -> np.ndarray:
        data = interpolate(log_controls) - np.log(observed)
        # A weak smoothness prior suppresses a five-storey saw-tooth fit.
        smoothness = 0.08 * np.diff(log_controls)
        return np.concatenate((data, smoothness))

    initial = np.full(3, float(np.median(np.log(observed))))
    solution = least_squares(
        residual,
        initial,
        bounds=(math.log(lower), math.log(upper)),
        loss="soft_l1",
        f_scale=0.12,
        max_nfev=200,
    )
    return tuple(float(value) for value in np.exp(solution.x))


def identify_static_story_recipe(
    comparisons: list[dict[str, Any]],
    *,
    story_counts: dict[str, int],
    development_buildings: set[str],
) -> tuple[StaticStoryCalibrationRecipe, dict[str, Any]]:
    """Identify a generic recipe from schema-v3 development evidence only.

    The function fails closed if a holdout row is supplied.  Peak/capping
    rotations are inferred from the selected-story shear peak.  Post-capping
    length is inferred only over the common completed drift range, so a
    failed terminal half-cycle cannot masquerade as physical deterioration.
    """

    z_values: list[float] = []
    stiffness_ratios: list[float] = []
    strength_ratios: list[float] = []
    pre_ratios: list[float] = []
    post_ratios: list[float] = []
    used_buildings: set[str] = set()
    for row in comparisons:
        building_id = str(row["building_id"])
        if building_id not in development_buildings:
            raise ValueError(
                f"Blind holdout evidence entered calibration: {building_id}"
            )
        count = int(story_counts[building_id])
        story = int(row["selected_story"])
        if not 1 <= story <= count:
            raise ValueError("Static evidence story index is inconsistent")
        full_stiffness = float(row["full3d_initial_secant_stiffness_kn_m"])
        fish_stiffness = float(row["fishbone_initial_secant_stiffness_kn_m"])
        full_strength = float(row["full3d_peak_abs_story_shear_kn"])
        fish_strength = float(row["fishbone_peak_abs_story_shear_kn"])
        if min(
            full_stiffness,
            fish_stiffness,
            full_strength,
            fish_strength,
        ) <= 0.0:
            continue
        z_values.append(0.5 if count == 1 else (story - 1) / (count - 1))
        stiffness_ratios.append(full_stiffness / fish_stiffness)
        strength_ratios.append(full_strength / fish_strength)

        with np.load(Path(row["full3d_history_path"])) as values:
            full_drift = np.abs(
                np.asarray(values["story_drift_ratio"], dtype=float)[:, story - 1]
            )
            full_shear = np.abs(
                np.asarray(values["story_shear_kn"], dtype=float)[:, story - 1]
            )
        with np.load(Path(row["fishbone_history_path"])) as values:
            fish_drift = np.abs(
                np.asarray(values["story_drift_ratio"], dtype=float)[:, story - 1]
            )
            fish_shear = np.abs(
                np.asarray(values["story_shear_kn"], dtype=float)[:, story - 1]
            )
        if full_shear.size and fish_shear.size:
            full_peak_drift = float(full_drift[int(np.nanargmax(full_shear))])
            fish_peak_drift = float(fish_drift[int(np.nanargmax(fish_shear))])
            common_limit = min(float(np.nanmax(full_drift)), float(np.nanmax(fish_drift)))
            if (
                full_peak_drift > 1.0e-6
                and fish_peak_drift > 1.0e-6
                and full_peak_drift < 0.98 * common_limit
                and fish_peak_drift < 0.98 * common_limit
            ):
                pre_ratios.append(full_peak_drift / fish_peak_drift)
                full_post = common_limit - full_peak_drift
                fish_post = common_limit - fish_peak_drift
                if min(full_post, fish_post) > 1.0e-6:
                    post_ratios.append(full_post / fish_post)
        used_buildings.add(building_id)

    z = np.asarray(z_values, dtype=float)
    diagnostic_stiffness = _fit_population_height_profile(
        z,
        np.asarray(stiffness_ratios, dtype=float),
        lower=0.40,
        upper=1.60,
    )
    strength = _fit_population_height_profile(
        z,
        np.asarray(strength_ratios, dtype=float),
        lower=0.60,
        upper=1.60,
    )
    pre_scale = float(
        np.clip(np.median(np.asarray(pre_ratios, dtype=float)), 0.25, 2.0)
    )
    post_scale = float(
        np.clip(np.median(np.asarray(post_ratios, dtype=float)), 0.50, 2.5)
    )
    recipe = StaticStoryCalibrationRecipe(
        # Modal Modes 1--3 measure the true near-origin tangent stiffness.
        # The first story-isolation secant already includes early cracking and
        # yielding, so treating it as elastic EI caused an unacceptable modal
        # period shift.  Preserve the modal-calibrated EI and use the observed
        # secant profile only as a diagnostic/nonlinear target.
        stiffness_low=1.0,
        stiffness_mid=1.0,
        stiffness_high=1.0,
        nonlinear=NonlinearCalibrationParameters(
            strength_low=strength[0],
            strength_mid=strength[1],
            strength_high=strength[2],
            pre_capping_rotation_scale=pre_scale,
            post_capping_rotation_scale=post_scale,
            column_to_beam_strength_ratio=1.0,
        ),
    )
    recipe.validate()
    metrics = {
        "schema_version": "ifb3-development-story-population-fit-v1",
        "uses_ground_motion_evidence": False,
        "development_buildings": sorted(used_buildings),
        "comparison_count": len(z_values),
        "stiffness_ratio_median": float(np.median(stiffness_ratios)),
        "diagnostic_story_secant_height_profile": {
            "low": diagnostic_stiffness[0],
            "mid": diagnostic_stiffness[1],
            "high": diagnostic_stiffness[2],
            "applied_to_elastic_ei": False,
            "reason": "preserve_full3d_modal_modes_1_to_3",
        },
        "strength_ratio_median": float(np.median(strength_ratios)),
        "pre_capping_ratio_median": pre_scale,
        "post_capping_ratio_median": post_scale,
    }
    return recipe, metrics


def evaluate_monotonic_parameters(
    artifact: FishboneArtifact,
    direction: str,
    evidence: dict[str, dict[str, np.ndarray]],
    parameters: NonlinearCalibrationParameters,
    *,
    displacement_increment_m: float = 0.002,
    maximum_runtime_s_per_curve: float = 30.0,
) -> tuple[FishboneArtifact, dict[str, Any]]:
    """Evaluate all available patterns without any NLTHA/IDA observations."""

    candidate = apply_nonlinear_parameters(
        artifact, direction, parameters
    )
    metrics: dict[str, dict[str, Any]] = {}
    for name, full3d in evidence.items():
        pattern = np.asarray(full3d["pattern"], dtype=float)
        result = run_pattern_spo(
            candidate,
            direction=direction,
            pattern=pattern,
            pattern_name=name,
            control_floor=int(np.argmax(np.abs(pattern))) + 1,
            displacement_increment_m=displacement_increment_m,
            minimum_displacement_increment_m=min(
                displacement_increment_m / 32.0, 1.0e-5
            ),
            maximum_runtime_s=maximum_runtime_s_per_curve,
        )
        metrics[name] = compare_monotonic_curve(
            full3d, result, artifact.story_heights_m
        )
    return candidate, {
        "schema_version": FISHBONE_MONOTONIC_CALIBRATION_SCHEMA_VERSION,
        "direction": direction.upper(),
        "parameters": {
            name: float(value)
            for name, value in parameters.__dict__.items()
        },
        "patterns": metrics,
        "drift_first_score": monotonic_evidence_score(metrics),
        "uses_ground_motion_evidence": False,
    }


def _replace_modifiers(
    artifact: FishboneArtifact,
    direction: str,
    log_parameters: np.ndarray,
) -> FishboneArtifact:
    normalized = direction.upper()
    count = artifact.story_count
    values = np.exp(np.asarray(log_parameters, dtype=float))
    if values.shape != (2 * count,):
        raise ValueError("Fishbone elastic parameter length is inconsistent")
    column = values[:count]
    beam = values[count:]
    stories = []
    for index, story in enumerate(artifact.stories):
        if normalized == "X":
            stories.append(
                replace(
                    story,
                    column_stiffness_modifier_x=float(column[index]),
                    beam_stiffness_modifier_x=float(beam[index]),
                )
            )
        elif normalized == "Y":
            stories.append(
                replace(
                    story,
                    column_stiffness_modifier_y=float(column[index]),
                    beam_stiffness_modifier_y=float(beam[index]),
                )
            )
        else:
            raise ValueError("Fishbone direction must be X or Y")
    return replace(artifact, stories=tuple(stories), artifact_hash="")


def _copy_x_modifiers_to_y(
    artifact: FishboneArtifact,
) -> FishboneArtifact:
    return replace(
        artifact,
        stories=tuple(
            replace(
                story,
                column_stiffness_modifier_y=(
                    story.column_stiffness_modifier_x
                ),
                beam_stiffness_modifier_y=story.beam_stiffness_modifier_x,
            )
            for story in artifact.stories
        ),
        artifact_hash="",
    )


def _directional_result(
    modal: FishboneModalResult,
    direction: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if direction.upper() == "X":
        return (
            np.asarray(modal.directional_periods_x_s, dtype=float),
            np.asarray(modal.normalized_mode_shapes_x, dtype=float),
            np.asarray(modal.effective_mass_ratios_x, dtype=float),
        )
    if direction.upper() == "Y":
        return (
            np.asarray(modal.directional_periods_y_s, dtype=float),
            np.asarray(modal.normalized_mode_shapes_y, dtype=float),
            np.asarray(modal.effective_mass_ratios_y, dtype=float),
        )
    raise ValueError("Fishbone direction must be X or Y")


def _mac(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float((a @ a) * (b @ b))
    if denominator <= 0.0:
        return 0.0
    return float((a @ b) ** 2 / denominator)


def _align_shape(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    values = np.asarray(predicted, dtype=float)
    reference = np.asarray(target, dtype=float)
    if float(values @ reference) < 0.0:
        values = -values
    return values


def _evaluate_modal(
    artifact: FishboneArtifact,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = build_model(artifact)
    run_gravity()
    modal = modal_properties(artifact, info)
    return _directional_result(modal, direction)


def _modal_metrics(
    predicted_periods: np.ndarray,
    predicted_shapes: np.ndarray,
    targets: DirectionalModalTargets,
) -> dict[str, Any]:
    target_periods = np.asarray(targets.periods_s, dtype=float)
    target_shapes = np.asarray(targets.mode_shapes, dtype=float)
    count = min(
        target_periods.size,
        predicted_periods.size,
        target_shapes.shape[0] if target_shapes.ndim == 2 else 0,
        predicted_shapes.shape[0] if predicted_shapes.ndim == 2 else 0,
    )
    period_errors = [
        float(
            abs(predicted_periods[index] - target_periods[index])
            / target_periods[index]
        )
        for index in range(count)
    ]
    macs = [
        _mac(
            _align_shape(predicted_shapes[index], target_shapes[index]),
            target_shapes[index],
        )
        for index in range(count)
    ]
    return {
        "period_errors": tuple(period_errors),
        "mode_macs": tuple(float(value) for value in macs),
        "maximum_period_error": max(period_errors, default=math.nan),
        "minimum_mac": min(macs, default=math.nan),
    }


def _targets_are_xy_symmetric(artifact: FishboneArtifact) -> bool:
    x = artifact.target_modal_x
    y = artifact.target_modal_y
    if len(x.periods_s) != len(y.periods_s):
        return False
    if not x.periods_s:
        return False
    period_close = np.allclose(
        np.asarray(x.periods_s),
        np.asarray(y.periods_s),
        rtol=2.0e-4,
        atol=1.0e-7,
    )
    shape_close = np.allclose(
        np.abs(np.asarray(x.mode_shapes)),
        np.abs(np.asarray(y.mode_shapes)),
        rtol=2.0e-3,
        atol=2.0e-4,
    )
    mass_close = np.allclose(
        np.asarray(artifact.floor_masses_x_kn_s2_m),
        np.asarray(artifact.floor_masses_y_kn_s2_m),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    return bool(period_close and shape_close and mass_close)


def _calibrate_direction(
    artifact: FishboneArtifact,
    direction: str,
    *,
    maximum_evaluations: int,
) -> tuple[FishboneArtifact, dict[str, Any]]:
    targets = (
        artifact.target_modal_x
        if direction.upper() == "X"
        else artifact.target_modal_y
    )
    target_periods = np.asarray(targets.periods_s, dtype=float)
    target_shapes = np.asarray(targets.mode_shapes, dtype=float)
    target_mass = np.asarray(targets.effective_mass_ratios, dtype=float)
    if target_periods.size == 0:
        raise RuntimeError(f"{direction}: no Full 3D modal targets")
    count = artifact.story_count
    initial = np.zeros(2 * count, dtype=float)
    lower = np.full(2 * count, math.log(0.10), dtype=float)
    upper = np.full(2 * count, math.log(3.00), dtype=float)
    period_weights = np.asarray((4.0, 2.5, 1.5), dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = _replace_modifiers(artifact, direction, parameters)
        predicted_periods, predicted_shapes, _ratios = _evaluate_modal(
            candidate, direction
        )
        modes = min(
            target_periods.size,
            predicted_periods.size,
            predicted_shapes.shape[0],
            target_shapes.shape[0],
            3,
        )
        values: list[float] = []
        for mode in range(modes):
            values.append(
                period_weights[mode]
                * (
                    predicted_periods[mode] - target_periods[mode]
                )
                / target_periods[mode]
            )
            aligned = _align_shape(
                predicted_shapes[mode], target_shapes[mode]
            )
            mass_weight = math.sqrt(
                max(
                    float(target_mass[mode])
                    if mode < target_mass.size
                    else 0.05,
                    0.02,
                )
            )
            values.extend(
                (
                    0.9
                    * mass_weight
                    * (aligned - target_shapes[mode])
                ).tolist()
            )
        column = parameters[:count]
        beam = parameters[count:]
        # Weak priors prevent non-identifiable beam/column combinations and
        # highly oscillatory storey modifiers from fitting a single building.
        values.extend((0.025 * parameters).tolist())
        if count > 1:
            values.extend((0.10 * np.diff(column)).tolist())
            values.extend((0.10 * np.diff(beam)).tolist())
        values.extend((0.035 * (column - beam)).tolist())
        return np.asarray(values, dtype=float)

    initial_periods, initial_shapes, _initial_mass = _evaluate_modal(
        artifact, direction
    )
    initial_metrics = _modal_metrics(
        initial_periods, initial_shapes, targets
    )
    started = time.perf_counter()
    solution = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        method="trf",
        max_nfev=int(maximum_evaluations),
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
        verbose=0,
    )
    calibrated = _replace_modifiers(
        artifact, direction, solution.x
    )
    final_periods, final_shapes, _final_mass = _evaluate_modal(
        calibrated, direction
    )
    final_metrics = _modal_metrics(final_periods, final_shapes, targets)
    return calibrated, {
        "schema_version": FISHBONE_ELASTIC_CALIBRATION_SCHEMA_VERSION,
        "direction": direction.upper(),
        "success": bool(solution.success),
        "status": int(solution.status),
        "message": str(solution.message),
        "evaluation_count": int(solution.nfev),
        "runtime_s": float(time.perf_counter() - started),
        "cost": float(solution.cost),
        "optimality": float(solution.optimality),
        "initial": initial_metrics,
        "final": final_metrics,
        "column_modifiers": tuple(
            float(value) for value in np.exp(solution.x[:count])
        ),
        "beam_modifiers": tuple(
            float(value) for value in np.exp(solution.x[count:])
        ),
    }


def calibrate_elastic_modal(
    artifact: FishboneArtifact,
    *,
    maximum_evaluations: int = 80,
) -> FishboneArtifact:
    """Fit member stiffness to Full 3D Modes 1–3 without NLTHA evidence."""

    artifact.validate()
    if artifact.certified:
        raise ValueError("A certified Fishbone artifact cannot be recalibrated")
    calibrated, metrics_x = _calibrate_direction(
        replace(artifact, artifact_hash=""),
        "X",
        maximum_evaluations=maximum_evaluations,
    )
    if _targets_are_xy_symmetric(artifact):
        calibrated = _copy_x_modifiers_to_y(calibrated)
        periods_y, shapes_y, _mass_y = _evaluate_modal(calibrated, "Y")
        metrics_y = {
            "schema_version": FISHBONE_ELASTIC_CALIBRATION_SCHEMA_VERSION,
            "direction": "Y",
            "selection": "copied_from_x_due_locked_plan_symmetry",
            "final": _modal_metrics(
                periods_y, shapes_y, artifact.target_modal_y
            ),
            "column_modifiers": tuple(
                story.column_stiffness_modifier_y
                for story in calibrated.stories
            ),
            "beam_modifiers": tuple(
                story.beam_stiffness_modifier_y
                for story in calibrated.stories
            ),
        }
        selection = "xy-symmetry-tied"
    else:
        calibrated, metrics_y = _calibrate_direction(
            calibrated,
            "Y",
            maximum_evaluations=maximum_evaluations,
        )
        selection = "independent-x-y"
    flags = tuple(
        flag
        for flag in artifact.applicability_flags
        if flag != "elastic-calibration-required"
    )
    metrics = dict(artifact.calibration_metrics)
    metrics["elastic_calibration"] = {
        "schema_version": FISHBONE_ELASTIC_CALIBRATION_SCHEMA_VERSION,
        "selection": selection,
        "x": metrics_x,
        "y": metrics_y,
    }
    result = replace(
        calibrated,
        calibration_stage="elastic_modal_calibrated",
        applicability_flags=flags,
        calibration_metrics=metrics,
        artifact_hash="",
    ).with_hash()
    result.validate()
    return result
