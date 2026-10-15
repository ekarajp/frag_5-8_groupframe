"""Development-only residual calibration for the Fishbone-v3 candidate.

The v2 artifact and all Full 3D caches remain immutable.  This module applies
bounded, population-level *residual* corrections to a v2 artifact.  Story
isolation identifies the combined story response; it cannot uniquely identify
beam and column contributions.  Consequently the first stage constrains beam
and column deformation/deterioration corrections to be equal.  A later modal
SPO stage may release the beam/column strength allocation using Modes 1--3.

No NLTHA, IDA, or blind-holdout observation is accepted by the identifier.
P-Delta is deliberately not calibrated: the auxiliary-column stiffness stays
equal to ``-gravity_load_above / story_height``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .fishbone import FishboneArtifact, HingeBackbone


FISHBONE_V3_STORY_CALIBRATION_SCHEMA_VERSION = (
    "ifb3-v3-development-story-residual-mechanism-v3-mode-specific-cyclic"
)


@dataclass(frozen=True)
class MechanismStoryCalibrationV3:
    """Bounded residual controls whose count is independent of story count.

    Beam/column fields are explicit so a subsequent Mode 1--3 SPO stage can
    identify mechanism allocation.  The story-isolation identifier below
    keeps corresponding beam/column controls equal because aggregate story
    loops alone do not provide a unique decomposition.
    """

    strength_low: float = 1.0
    strength_mid: float = 1.0
    strength_high: float = 1.0
    yield_strength_fraction: float = 1.0
    column_to_beam_strength_ratio: float = 1.0
    column_pre_capping_scale: float = 1.0
    beam_pre_capping_scale: float = 1.0
    column_post_capping_scale: float = 1.0
    beam_post_capping_scale: float = 1.0
    column_residual_strength_scale: float = 1.0
    beam_residual_strength_scale: float = 1.0
    column_deterioration_capacity_scale: float = 1.0
    beam_deterioration_capacity_scale: float = 1.0
    accelerated_reloading_capacity_scale: float = 1.0
    unloading_stiffness_capacity_scale: float = 1.0
    ultimate_rotation_scale: float = 1.0
    capping_to_yield_scale: float = 1.0
    pdelta_scale: float = 1.0

    def validate(self) -> None:
        values = tuple(
            float(value)
            for name, value in self.__dict__.items()
            if name != "yield_strength_fraction"
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Fishbone-v3 residual calibration is invalid")
        if not 0.0 <= self.yield_strength_fraction <= 1.0:
            raise ValueError(
                "Fishbone-v3 yield-strength fraction must lie in [0, 1]"
            )
        if not math.isclose(self.pdelta_scale, 1.0, abs_tol=1.0e-12):
            raise ValueError("Fishbone-v3 P-Delta must remain physics-fixed")


def _height_profile(
    story_count: int,
    low: float,
    middle: float,
    high: float,
) -> np.ndarray:
    if story_count < 1:
        raise ValueError("Fishbone-v3 requires at least one story")
    if story_count == 1:
        return np.asarray((middle,), dtype=float)
    return np.exp(
        np.interp(
            np.linspace(0.0, 1.0, story_count),
            (0.0, 0.5, 1.0),
            np.log((low, middle, high)),
        )
    )


def _fit_height_profile(
    z: np.ndarray,
    ratios: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    valid = (
        np.isfinite(z)
        & np.isfinite(ratios)
        & (z >= 0.0)
        & (z <= 1.0)
        & (ratios > 0.0)
    )
    z = np.asarray(z[valid], dtype=float)
    observed = np.clip(np.asarray(ratios[valid], dtype=float), lower, upper)
    if observed.size < 3:
        raise ValueError("At least three valid development ratios are required")

    def residual(log_controls: np.ndarray) -> np.ndarray:
        data = (
            np.interp(z, (0.0, 0.5, 1.0), log_controls)
            - np.log(observed)
        )
        # Smoothness and shrinkage priors prevent a five-story-specific fit.
        smoothness = 0.10 * np.diff(log_controls)
        shrinkage = 0.04 * log_controls
        return np.concatenate((data, smoothness, shrinkage))

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


def _scale_hinge(
    hinge: HingeBackbone,
    *,
    strength_scale: float,
    pre_scale: float,
    post_scale: float,
    residual_scale: float,
    deterioration_scale: float,
    accelerated_reloading_scale: float,
    unloading_stiffness_scale: float,
    ultimate_rotation_scale: float,
    capping_scale: float,
    source: str,
) -> HingeBackbone:
    pre_pos = hinge.pre_capping_rotation_pos_rad * pre_scale
    pre_neg = hinge.pre_capping_rotation_neg_rad * pre_scale
    post_pos = hinge.post_capping_rotation_pos_rad * post_scale
    post_neg = hinge.post_capping_rotation_neg_rad * post_scale
    ultimate_pos = max(
        hinge.ultimate_rotation_pos_rad
        * max(pre_scale, post_scale)
        * ultimate_rotation_scale,
        1.05 * (pre_pos + post_pos),
    )
    ultimate_neg = max(
        hinge.ultimate_rotation_neg_rad
        * max(pre_scale, post_scale)
        * ultimate_rotation_scale,
        1.05 * (pre_neg + post_neg),
    )
    return replace(
        hinge,
        yield_moment_pos_knm=hinge.yield_moment_pos_knm * strength_scale,
        yield_moment_neg_knm=hinge.yield_moment_neg_knm * strength_scale,
        pre_capping_rotation_pos_rad=pre_pos,
        pre_capping_rotation_neg_rad=pre_neg,
        post_capping_rotation_pos_rad=post_pos,
        post_capping_rotation_neg_rad=post_neg,
        ultimate_rotation_pos_rad=ultimate_pos,
        ultimate_rotation_neg_rad=ultimate_neg,
        capping_to_yield_pos=max(
            1.0, hinge.capping_to_yield_pos * capping_scale
        ),
        capping_to_yield_neg=max(
            1.0, hinge.capping_to_yield_neg * capping_scale
        ),
        residual_to_yield_pos=float(
            np.clip(hinge.residual_to_yield_pos * residual_scale, 0.02, 0.95)
        ),
        residual_to_yield_neg=float(
            np.clip(hinge.residual_to_yield_neg * residual_scale, 0.02, 0.95)
        ),
        lambda_s=hinge.lambda_s * deterioration_scale,
        lambda_c=hinge.lambda_c * deterioration_scale,
        lambda_a=hinge.lambda_a * accelerated_reloading_scale,
        lambda_k=hinge.lambda_k * unloading_stiffness_scale,
        source=source,
    )


def apply_mechanism_story_recipe_v3(
    artifact: FishboneArtifact,
    direction: str,
    recipe: MechanismStoryCalibrationV3,
) -> FishboneArtifact:
    """Return an inactive v3 candidate without modifying v2 or P-Delta."""

    artifact.validate()
    recipe.validate()
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Fishbone-v3 direction must be X or Y")
    strength = _height_profile(
        artifact.story_count,
        recipe.strength_low,
        recipe.strength_mid,
        recipe.strength_high,
    )
    allocation = math.sqrt(recipe.column_to_beam_strength_ratio)
    source = f"{FISHBONE_V3_STORY_CALIBRATION_SCHEMA_VERSION}:{normalized}"
    stories = []
    for index, story in enumerate(artifact.stories):
        common = float(strength[index])
        yield_common = common ** recipe.yield_strength_fraction
        cap_common = common ** (1.0 - recipe.yield_strength_fraction)
        column_kwargs = {
            "strength_scale": yield_common * allocation,
            "pre_scale": recipe.column_pre_capping_scale,
            "post_scale": recipe.column_post_capping_scale,
            "residual_scale": recipe.column_residual_strength_scale,
            "deterioration_scale": (
                recipe.column_deterioration_capacity_scale
            ),
            "accelerated_reloading_scale": (
                recipe.accelerated_reloading_capacity_scale
            ),
            "unloading_stiffness_scale": (
                recipe.unloading_stiffness_capacity_scale
            ),
            "ultimate_rotation_scale": recipe.ultimate_rotation_scale,
            "capping_scale": cap_common * recipe.capping_to_yield_scale,
            "source": source,
        }
        beam_kwargs = {
            "strength_scale": yield_common / allocation,
            "pre_scale": recipe.beam_pre_capping_scale,
            "post_scale": recipe.beam_post_capping_scale,
            "residual_scale": recipe.beam_residual_strength_scale,
            "deterioration_scale": recipe.beam_deterioration_capacity_scale,
            "accelerated_reloading_scale": (
                recipe.accelerated_reloading_capacity_scale
            ),
            "unloading_stiffness_scale": (
                recipe.unloading_stiffness_capacity_scale
            ),
            "ultimate_rotation_scale": recipe.ultimate_rotation_scale,
            "capping_scale": cap_common * recipe.capping_to_yield_scale,
            "source": source,
        }
        if normalized == "X":
            updated = replace(
                story,
                column_bottom_hinge_x=_scale_hinge(
                    story.column_bottom_hinge_x, **column_kwargs
                ),
                column_top_hinge_x=_scale_hinge(
                    story.column_top_hinge_x, **column_kwargs
                ),
                beam_hinge_x=_scale_hinge(story.beam_hinge_x, **beam_kwargs),
            )
        else:
            updated = replace(
                story,
                column_bottom_hinge_y=_scale_hinge(
                    story.column_bottom_hinge_y, **column_kwargs
                ),
                column_top_hinge_y=_scale_hinge(
                    story.column_top_hinge_y, **column_kwargs
                ),
                beam_hinge_y=_scale_hinge(story.beam_hinge_y, **beam_kwargs),
            )
        # This assertion guards against accidental empirical P-Delta tuning.
        if updated.geometric_stiffness_kn_m != story.geometric_stiffness_kn_m:
            raise AssertionError("Fishbone-v3 changed physics-based P-Delta")
        stories.append(updated)
    result = replace(
        artifact,
        stories=tuple(stories),
        certified=False,
        certification_path=None,
        artifact_hash="",
    ).with_hash()
    result.validate()
    return result


def _history_features(
    row: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    story_index = int(row["selected_story"]) - 1
    common = int(row["common_completed_target_index"])
    with np.load(Path(row[f"{prefix}_history_path"])) as values:
        target = np.asarray(values["protocol_target_index"], dtype=int)
        drift = np.asarray(values["story_drift_ratio"], dtype=float)[
            :, story_index
        ]
        shear = np.asarray(values["story_shear_kn"], dtype=float)[:, story_index]
    mask = (target >= 1) & (target <= common)
    target = target[mask]
    drift = drift[mask]
    shear = shear[mask]
    if not shear.size:
        raise ValueError("Empty story-isolation history")
    peak_index = int(np.nanargmax(np.abs(shear)))
    peak_strength = float(np.abs(shear[peak_index]))
    peak_drift = float(np.abs(drift[peak_index]))

    # The last target is the zero-return branch.  Residual strength is taken
    # at the largest completed nonzero amplitude, averaging both signs.
    nonzero_targets = [
        value
        for value in np.unique(target)
        if np.nanmax(np.abs(drift[target == value])) > 1.0e-5
    ]
    max_amplitude = max(
        float(np.nanmax(np.abs(drift[target == value])))
        for value in nonzero_targets
    )
    terminal_shears: list[float] = []
    for value in nonzero_targets:
        local = target == value
        local_drift = np.abs(drift[local])
        if float(np.nanmax(local_drift)) >= 0.98 * max_amplitude:
            local_shear = np.abs(shear[local])
            terminal_shears.append(float(local_shear[int(np.argmax(local_drift))]))
    residual_ratio = float(np.median(terminal_shears)) / max(
        peak_strength, 1.0e-12
    )
    energy = abs(float(np.trapz(shear, drift)))
    drift_range = max(float(np.nanmax(drift) - np.nanmin(drift)), 1.0e-12)
    normalized_energy = energy / max(peak_strength * drift_range, 1.0e-12)
    return {
        "peak_drift": peak_drift,
        "peak_strength": peak_strength,
        "max_drift": float(np.nanmax(np.abs(drift))),
        "residual_ratio": residual_ratio,
        "normalized_energy": normalized_energy,
    }


def identify_mechanism_story_recipe_v3(
    comparisons: list[dict[str, Any]],
    *,
    story_counts: dict[str, int],
    development_buildings: set[str],
) -> tuple[MechanismStoryCalibrationV3, dict[str, Any]]:
    """Fit bounded residual controls from development static evidence only."""

    z_values: list[float] = []
    strength_ratios: list[float] = []
    pre_ratios: list[float] = []
    post_ratios: list[float] = []
    residual_ratios: list[float] = []
    deterioration_ratios: list[float] = []
    used_buildings: set[str] = set()
    for row in comparisons:
        building_id = str(row["building_id"])
        if building_id not in development_buildings:
            raise ValueError(
                f"Blind holdout evidence entered v3 calibration: {building_id}"
            )
        count = int(story_counts[building_id])
        story = int(row["selected_story"])
        if not 1 <= story <= count:
            raise ValueError("Fishbone-v3 story index is inconsistent")
        full_strength = float(row["full3d_peak_abs_story_shear_kn"])
        fish_strength = float(row["fishbone_peak_abs_story_shear_kn"])
        if min(full_strength, fish_strength) <= 0.0:
            continue
        z_values.append(0.5 if count == 1 else (story - 1) / (count - 1))
        strength_ratios.append(full_strength / fish_strength)
        full = _history_features(row, "full3d")
        fish = _history_features(row, "fishbone")
        if min(full["peak_drift"], fish["peak_drift"]) > 1.0e-6:
            pre_ratios.append(full["peak_drift"] / fish["peak_drift"])
            full_post = full["max_drift"] - full["peak_drift"]
            fish_post = fish["max_drift"] - fish["peak_drift"]
            if min(full_post, fish_post) > 1.0e-6:
                post_ratios.append(full_post / fish_post)
        if min(full["residual_ratio"], fish["residual_ratio"]) > 1.0e-6:
            residual_ratios.append(
                full["residual_ratio"] / fish["residual_ratio"]
            )
        if min(full["normalized_energy"], fish["normalized_energy"]) > 1.0e-6:
            deterioration_ratios.append(
                full["normalized_energy"] / fish["normalized_energy"]
            )
        used_buildings.add(building_id)

    strength = _fit_height_profile(
        np.asarray(z_values, dtype=float),
        np.asarray(strength_ratios, dtype=float),
        lower=0.70,
        upper=1.70,
    )

    def bounded_median(values: list[float], low: float, high: float) -> float:
        if not values:
            return 1.0
        return float(np.clip(np.median(np.asarray(values, dtype=float)), low, high))

    pre = bounded_median(pre_ratios, 0.60, 1.60)
    post = bounded_median(post_ratios, 0.60, 1.80)
    residual = bounded_median(residual_ratios, 0.60, 1.50)
    deterioration = bounded_median(deterioration_ratios, 0.50, 1.80)
    recipe = MechanismStoryCalibrationV3(
        strength_low=strength[0],
        strength_mid=strength[1],
        strength_high=strength[2],
        # Preserve the v2 yield point and low-drift secant.  The missing peak
        # capacity is assigned to the aggregate capping branch, representing
        # progressive yielding of many source members rather than a later
        # first yield of the equivalent member.
        yield_strength_fraction=0.0,
        # Equality constraints are mandatory at this stage: aggregate story
        # loops cannot uniquely separate the two mechanisms.
        column_pre_capping_scale=pre,
        beam_pre_capping_scale=pre,
        column_post_capping_scale=post,
        beam_post_capping_scale=post,
        column_residual_strength_scale=residual,
        beam_residual_strength_scale=residual,
        # Preserve strength/post-capping deterioration capacity.  The energy
        # residual is a loop-shape observation, so it is assigned to the two
        # stiffness-deterioration modes instead of weakening the envelope.
        column_deterioration_capacity_scale=1.0,
        beam_deterioration_capacity_scale=1.0,
        accelerated_reloading_capacity_scale=deterioration,
        unloading_stiffness_capacity_scale=deterioration,
        pdelta_scale=1.0,
    )
    recipe.validate()
    metrics = {
        "schema_version": FISHBONE_V3_STORY_CALIBRATION_SCHEMA_VERSION,
        "uses_ground_motion_evidence": False,
        "uses_ida_evidence": False,
        "uses_holdout_evidence": False,
        "development_buildings": sorted(used_buildings),
        "comparison_count": len(z_values),
        "effective_free_parameter_count": 7,
        "story_specific_parameter_count": 0,
        "building_specific_parameter_count": 0,
        "strength_ratio_median": float(np.median(strength_ratios)),
        "peak_drift_ratio_median": pre,
        "post_peak_range_ratio_median": post,
        "residual_strength_ratio_median": residual,
        "normalized_energy_ratio_median": deterioration,
        "cyclic_allocation_policy": (
            "normalized loop-energy residual controls lambda_A and lambda_K; "
            "lambda_S and lambda_C remain physics-initialized"
        ),
        "beam_column_policy": (
            "equal_static_corrections; allocation remains locked until "
            "development modal SPO Modes 1-3"
        ),
        "strength_allocation_policy": (
            "preserve_v2_yield_moment; residual peak-strength correction "
            "enters the aggregate capping branch"
        ),
        "pdelta_policy": "physics_fixed_minus_gravity_above_over_height",
    }
    return recipe, metrics
