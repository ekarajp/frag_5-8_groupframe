"""Bidirectional Improved Fishbone reduced model (Fishbone v2).

The module is intentionally independent of :mod:`fragility_poc.mdof` so the
active MDOF-v1 validation campaign cannot be invalidated by Fishbone-v2
development.  It implements the physics-based IFB3 initialisation only.
Elastic, monotonic and cyclic calibration are separate mandatory stages
before an artifact can be certified for production IDA.

Units are kN, m, s and radians.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import column_pm_capacity_at_axial
from .constants import KSC_TO_KN_M2
from .io_utils import atomic_write_json, read_json, stable_hash
from .structural import ModalResult, ModelInfo, _ops

FISHBONE_ARTIFACT_SCHEMA_VERSION = (
    "bidirectional-ifb3-artifact-v3-joint-equilibrium-story-evidence"
)
FISHBONE_MODEL_SCHEMA_VERSION = (
    "bidirectional-ifb3-opensees-v3-plain-mp-member-hinges-common-pdelta"
)


@dataclass(frozen=True)
class HingeBackbone:
    """Deteriorating peak-oriented moment-rotation hinge.

    Positive and negative moments are stored as positive magnitudes.  The
    OpenSees IMK material receives independent capacities for both branches.
    """

    initial_stiffness_knm_rad: float
    yield_moment_pos_knm: float
    yield_moment_neg_knm: float
    pre_capping_rotation_pos_rad: float
    pre_capping_rotation_neg_rad: float
    post_capping_rotation_pos_rad: float
    post_capping_rotation_neg_rad: float
    ultimate_rotation_pos_rad: float
    ultimate_rotation_neg_rad: float
    capping_to_yield_pos: float
    capping_to_yield_neg: float
    residual_to_yield_pos: float
    residual_to_yield_neg: float
    lambda_s: float
    lambda_c: float
    lambda_a: float
    lambda_k: float
    c_s: float = 1.0
    c_c: float = 1.0
    c_a: float = 1.0
    c_k: float = 1.0
    deterioration_pos: float = 1.0
    deterioration_neg: float = 1.0
    source: str = "physics_initialisation"

    def validate(self) -> None:
        positive = {
            "initial_stiffness_knm_rad": self.initial_stiffness_knm_rad,
            "yield_moment_pos_knm": self.yield_moment_pos_knm,
            "yield_moment_neg_knm": self.yield_moment_neg_knm,
            "pre_capping_rotation_pos_rad": (
                self.pre_capping_rotation_pos_rad
            ),
            "pre_capping_rotation_neg_rad": (
                self.pre_capping_rotation_neg_rad
            ),
            "post_capping_rotation_pos_rad": (
                self.post_capping_rotation_pos_rad
            ),
            "post_capping_rotation_neg_rad": (
                self.post_capping_rotation_neg_rad
            ),
            "ultimate_rotation_pos_rad": self.ultimate_rotation_pos_rad,
            "ultimate_rotation_neg_rad": self.ultimate_rotation_neg_rad,
            "capping_to_yield_pos": self.capping_to_yield_pos,
            "capping_to_yield_neg": self.capping_to_yield_neg,
            "residual_to_yield_pos": self.residual_to_yield_pos,
            "residual_to_yield_neg": self.residual_to_yield_neg,
            "lambda_s": self.lambda_s,
            "lambda_c": self.lambda_c,
            "lambda_a": self.lambda_a,
            "lambda_k": self.lambda_k,
            "c_s": self.c_s,
            "c_c": self.c_c,
            "c_a": self.c_a,
            "c_k": self.c_k,
            "deterioration_pos": self.deterioration_pos,
            "deterioration_neg": self.deterioration_neg,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "Invalid Fishbone hinge values: " + ", ".join(invalid)
            )
        if self.capping_to_yield_pos < 1.0:
            raise ValueError("Positive capping moment must not precede yield")
        if self.capping_to_yield_neg < 1.0:
            raise ValueError("Negative capping moment must not precede yield")
        if not 0.0 < self.residual_to_yield_pos < 1.0:
            raise ValueError("Positive residual ratio must lie in (0, 1)")
        if not 0.0 < self.residual_to_yield_neg < 1.0:
            raise ValueError("Negative residual ratio must lie in (0, 1)")
        if (
            self.ultimate_rotation_pos_rad
            <= self.pre_capping_rotation_pos_rad
            + self.post_capping_rotation_pos_rad
        ):
            raise ValueError(
                "Positive ultimate rotation must exceed capping rotations"
            )
        if (
            self.ultimate_rotation_neg_rad
            <= self.pre_capping_rotation_neg_rad
            + self.post_capping_rotation_neg_rad
        ):
            raise ValueError(
                "Negative ultimate rotation must exceed capping rotations"
            )


@dataclass(frozen=True)
class DirectionalModalTargets:
    direction: str
    global_mode_indices: tuple[int, ...]
    periods_s: tuple[float, ...]
    mode_shapes: tuple[tuple[float, ...], ...]
    effective_mass_ratios: tuple[float, ...]
    participation_factors: tuple[float, ...]

    def validate(self, story_count: int) -> None:
        if self.direction not in {"X", "Y"}:
            raise ValueError("Modal direction must be X or Y")
        vectors = (
            self.global_mode_indices,
            self.periods_s,
            self.mode_shapes,
            self.effective_mass_ratios,
            self.participation_factors,
        )
        if any(len(values) != len(self.periods_s) for values in vectors):
            raise ValueError("Directional modal target lengths differ")
        if len(self.periods_s) > min(3, story_count):
            raise ValueError("Fishbone v2 stores at most three target modes")
        if any(len(shape) != story_count for shape in self.mode_shapes):
            raise ValueError("Directional mode-shape length is inconsistent")


@dataclass(frozen=True)
class EquivalentStory:
    """One storey's IFB3-equivalent member properties."""

    story: int
    height_m: float
    column_count: int
    beam_count_per_direction: int
    column_area_m2: float
    column_inertia_x_m4: float
    column_inertia_y_m4: float
    column_torsion_m4: float
    beam_area_x_m2: float
    beam_area_y_m2: float
    beam_inertia_x_m4: float
    beam_inertia_y_m4: float
    beam_torsion_x_m4: float
    beam_torsion_y_m4: float
    beam_half_length_x_m: float
    beam_half_length_y_m: float
    column_stiffness_modifier_x: float
    column_stiffness_modifier_y: float
    beam_stiffness_modifier_x: float
    beam_stiffness_modifier_y: float
    beam_uniform_gravity_x_kn_m: float
    beam_uniform_gravity_y_kn_m: float
    gravity_load_above_kn: float
    geometric_stiffness_kn_m: float
    column_bottom_hinge_x: HingeBackbone
    column_bottom_hinge_y: HingeBackbone
    column_top_hinge_x: HingeBackbone
    column_top_hinge_y: HingeBackbone
    beam_hinge_x: HingeBackbone
    beam_hinge_y: HingeBackbone

    def validate(self) -> None:
        if self.story < 1:
            raise ValueError("Fishbone story number must be positive")
        positive = (
            self.height_m,
            self.column_area_m2,
            self.column_inertia_x_m4,
            self.column_inertia_y_m4,
            self.column_torsion_m4,
            self.beam_area_x_m2,
            self.beam_area_y_m2,
            self.beam_inertia_x_m4,
            self.beam_inertia_y_m4,
            self.beam_torsion_x_m4,
            self.beam_torsion_y_m4,
            self.beam_half_length_x_m,
            self.beam_half_length_y_m,
            self.column_stiffness_modifier_x,
            self.column_stiffness_modifier_y,
            self.beam_stiffness_modifier_x,
            self.beam_stiffness_modifier_y,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(f"Story {self.story}: invalid equivalent geometry")
        if self.column_count < 1 or self.beam_count_per_direction < 1:
            raise ValueError(f"Story {self.story}: invalid member counts")
        if self.beam_uniform_gravity_x_kn_m < 0.0:
            raise ValueError(f"Story {self.story}: negative beam gravity load")
        if self.beam_uniform_gravity_y_kn_m < 0.0:
            raise ValueError(f"Story {self.story}: negative beam gravity load")
        if self.gravity_load_above_kn <= 0.0:
            raise ValueError(f"Story {self.story}: gravity load is missing")
        expected = -self.gravity_load_above_kn / self.height_m
        if not math.isclose(
            self.geometric_stiffness_kn_m,
            expected,
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        ):
            raise ValueError(
                f"Story {self.story}: common P-Delta stiffness is inconsistent"
            )
        for hinge in (
            self.column_bottom_hinge_x,
            self.column_bottom_hinge_y,
            self.column_top_hinge_x,
            self.column_top_hinge_y,
            self.beam_hinge_x,
            self.beam_hinge_y,
        ):
            hinge.validate()


@dataclass(frozen=True)
class FishboneArtifact:
    building_id: str
    source_model_hash: str
    source_signature: str
    story_count: int
    number_of_bays: int
    plan_length_x_m: float
    plan_length_y_m: float
    floor_elevations_m: tuple[float, ...]
    story_heights_m: tuple[float, ...]
    floor_masses_x_kn_s2_m: tuple[float, ...]
    floor_masses_y_kn_s2_m: tuple[float, ...]
    floor_polar_masses_kn_m_s2: tuple[float, ...]
    gravity_floor_loads_kn: tuple[float, ...]
    elastic_modulus_kn_m2: float
    shear_modulus_kn_m2: float
    stories: tuple[EquivalentStory, ...]
    target_modal_x: DirectionalModalTargets
    target_modal_y: DirectionalModalTargets
    plan_symmetry_required: bool = True
    shear_failure_enabled: bool = False
    torsion_certified: bool = False
    xy_strength_interaction_supported: bool = False
    calibration_stage: str = "physics_initialisation"
    applicability_flags: tuple[str, ...] = ()
    calibration_metrics: dict[str, Any] = field(default_factory=dict)
    certified: bool = False
    certification_path: str | None = None
    schema_version: str = FISHBONE_ARTIFACT_SCHEMA_VERSION
    artifact_hash: str = ""

    def validate(self) -> None:
        if self.schema_version != FISHBONE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Fishbone artifact schema: {self.schema_version}"
            )
        if self.story_count < 1:
            raise ValueError("Fishbone artifact requires at least one storey")
        if self.number_of_bays < 1:
            raise ValueError("Fishbone artifact requires at least one bay")
        vectors = (
            self.floor_elevations_m,
            self.story_heights_m,
            self.floor_masses_x_kn_s2_m,
            self.floor_masses_y_kn_s2_m,
            self.floor_polar_masses_kn_m_s2,
            self.gravity_floor_loads_kn,
            self.stories,
        )
        if any(len(values) != self.story_count for values in vectors):
            raise ValueError("Fishbone story vectors have inconsistent length")
        if any(
            not math.isfinite(value) or value <= 0.0
            for values in (
                self.floor_masses_x_kn_s2_m,
                self.floor_masses_y_kn_s2_m,
                self.floor_polar_masses_kn_m_s2,
                self.gravity_floor_loads_kn,
            )
            for value in values
        ):
            raise ValueError("Fishbone mass/gravity vectors are invalid")
        if not all(
            self.floor_elevations_m[index]
            > (self.floor_elevations_m[index - 1] if index else 0.0)
            for index in range(self.story_count)
        ):
            raise ValueError("Fishbone floor elevations must increase")
        self.target_modal_x.validate(self.story_count)
        self.target_modal_y.validate(self.story_count)
        for index, story in enumerate(self.stories, 1):
            story.validate()
            if story.story != index:
                raise ValueError("Fishbone stories are not sequential")
        if self.certified and self.calibration_stage != "certified":
            raise ValueError("Certified artifact has incomplete calibration")
        if self.artifact_hash:
            expected = artifact_payload_hash(self)
            if self.artifact_hash != expected:
                raise ValueError("Fishbone artifact hash is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_hash(self) -> "FishboneArtifact":
        return replace(self, artifact_hash=artifact_payload_hash(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FishboneArtifact":
        def hinge(values: dict[str, Any]) -> HingeBackbone:
            return HingeBackbone(**values)

        story_rows: list[EquivalentStory] = []
        hinge_keys = {
            "column_bottom_hinge_x",
            "column_bottom_hinge_y",
            "column_top_hinge_x",
            "column_top_hinge_y",
            "beam_hinge_x",
            "beam_hinge_y",
        }
        for raw in payload["stories"]:
            row = dict(raw)
            for key in hinge_keys:
                row[key] = hinge(dict(row[key]))
            story_rows.append(EquivalentStory(**row))

        def targets(raw: dict[str, Any]) -> DirectionalModalTargets:
            return DirectionalModalTargets(
                direction=str(raw["direction"]),
                global_mode_indices=tuple(
                    int(value) for value in raw["global_mode_indices"]
                ),
                periods_s=tuple(float(value) for value in raw["periods_s"]),
                mode_shapes=tuple(
                    tuple(float(value) for value in shape)
                    for shape in raw["mode_shapes"]
                ),
                effective_mass_ratios=tuple(
                    float(value) for value in raw["effective_mass_ratios"]
                ),
                participation_factors=tuple(
                    float(value) for value in raw["participation_factors"]
                ),
            )

        return cls(
            building_id=str(payload["building_id"]),
            source_model_hash=str(payload["source_model_hash"]),
            source_signature=str(payload["source_signature"]),
            story_count=int(payload["story_count"]),
            number_of_bays=int(payload["number_of_bays"]),
            plan_length_x_m=float(payload["plan_length_x_m"]),
            plan_length_y_m=float(payload["plan_length_y_m"]),
            floor_elevations_m=tuple(
                float(value) for value in payload["floor_elevations_m"]
            ),
            story_heights_m=tuple(
                float(value) for value in payload["story_heights_m"]
            ),
            floor_masses_x_kn_s2_m=tuple(
                float(value)
                for value in payload["floor_masses_x_kn_s2_m"]
            ),
            floor_masses_y_kn_s2_m=tuple(
                float(value)
                for value in payload["floor_masses_y_kn_s2_m"]
            ),
            floor_polar_masses_kn_m_s2=tuple(
                float(value)
                for value in payload["floor_polar_masses_kn_m_s2"]
            ),
            gravity_floor_loads_kn=tuple(
                float(value) for value in payload["gravity_floor_loads_kn"]
            ),
            elastic_modulus_kn_m2=float(payload["elastic_modulus_kn_m2"]),
            shear_modulus_kn_m2=float(payload["shear_modulus_kn_m2"]),
            stories=tuple(story_rows),
            target_modal_x=targets(dict(payload["target_modal_x"])),
            target_modal_y=targets(dict(payload["target_modal_y"])),
            plan_symmetry_required=bool(
                payload.get("plan_symmetry_required", True)
            ),
            shear_failure_enabled=bool(
                payload.get("shear_failure_enabled", False)
            ),
            torsion_certified=bool(payload.get("torsion_certified", False)),
            xy_strength_interaction_supported=bool(
                payload.get("xy_strength_interaction_supported", False)
            ),
            calibration_stage=str(
                payload.get("calibration_stage", "physics_initialisation")
            ),
            applicability_flags=tuple(
                str(value) for value in payload.get("applicability_flags", ())
            ),
            calibration_metrics=dict(
                payload.get("calibration_metrics", {})
            ),
            certified=bool(payload.get("certified", False)),
            certification_path=payload.get("certification_path"),
            schema_version=str(
                payload.get(
                    "schema_version", FISHBONE_ARTIFACT_SCHEMA_VERSION
                )
            ),
            artifact_hash=str(payload.get("artifact_hash", "")),
        )


@dataclass(frozen=True)
class FishboneModelInfo:
    base_node: int
    pdelta_base_node: int
    floor_nodes: tuple[int, ...]
    column_elements: tuple[int, ...]
    beam_elements_x: tuple[int, ...]
    beam_elements_y: tuple[int, ...]
    column_hinge_elements: tuple[int, ...]
    beam_hinge_elements: tuple[int, ...]
    pdelta_elements: tuple[int, ...]
    floor_elevations_m: tuple[float, ...]
    story_heights_m: tuple[float, ...]
    gravity_applied_kn: float
    axial_chord_elements: tuple[int, ...] = ()


@dataclass(frozen=True)
class FishboneModalResult:
    periods_s: tuple[float, ...]
    global_mode_indices_x: tuple[int, ...]
    global_mode_indices_y: tuple[int, ...]
    directional_periods_x_s: tuple[float, ...]
    directional_periods_y_s: tuple[float, ...]
    normalized_mode_shapes_x: tuple[tuple[float, ...], ...]
    normalized_mode_shapes_y: tuple[tuple[float, ...], ...]
    effective_mass_ratios_x: tuple[float, ...]
    effective_mass_ratios_y: tuple[float, ...]
    participation_factors_x: tuple[float, ...]
    participation_factors_y: tuple[float, ...]


def fishbone_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("fishbone", {}))
    settings.setdefault("artifact_subdir", "fishbone_v2/artifacts")
    settings.setdefault("history_subdir", "fishbone_v2/histories")
    settings.setdefault("damping_ratio", config["model"]["damping_ratio"])
    settings.setdefault("nominal_flexural_phi", 0.90)
    settings.setdefault("hinge_stiffness_factor", 10.0)
    settings.setdefault("initial_capping_to_yield", 1.05)
    settings.setdefault("initial_residual_to_yield", 0.20)
    settings.setdefault("initial_pre_capping_rotation_rad", 0.020)
    settings.setdefault("initial_post_capping_rotation_rad", 0.040)
    settings.setdefault("initial_ultimate_rotation_rad", 0.100)
    settings.setdefault("initial_cyclic_lambda", 20.0)
    return settings


def artifact_path(config: dict[str, Any], building_id: str) -> Path:
    return (
        Path(config["run_dir"])
        / str(fishbone_config(config)["artifact_subdir"])
        / f"{building_id}.json"
    )


def artifact_payload_hash(artifact: FishboneArtifact) -> str:
    payload = artifact.to_dict()
    payload["artifact_hash"] = ""
    payload["certified"] = False
    payload["certification_path"] = None
    return stable_hash(payload)


def save_artifact(
    artifact: FishboneArtifact,
    config: dict[str, Any],
) -> Path:
    finalized = artifact.with_hash()
    finalized.validate()
    path = artifact_path(config, finalized.building_id)
    atomic_write_json(path, finalized.to_dict())
    return path


def load_artifact(
    config: dict[str, Any],
    building_id: str,
    *,
    require_certified: bool = False,
) -> FishboneArtifact:
    path = artifact_path(config, building_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"{building_id}: Fishbone-v2 artifact is missing"
        )
    artifact = FishboneArtifact.from_dict(read_json(path))
    artifact.validate()
    if require_certified and not artifact.certified:
        raise RuntimeError(
            f"{building_id}: Fishbone-v2 artifact is not certified"
        )
    return artifact


def _directional_targets(
    modal: ModalResult,
    direction: str,
    story_count: int,
) -> DirectionalModalTargets:
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
        raise ValueError("Fishbone modal direction must be X or Y")
    if not shapes or not ratios or not participation:
        raise ValueError("Full 3D modal result lacks directional mode data")
    candidates = [
        index
        for index, (shape, ratio) in enumerate(zip(shapes, ratios))
        if float(ratio) > 1.0e-6
        and len(shape) == story_count
        and float(np.max(np.abs(np.asarray(shape, dtype=float)))) > 1.0e-8
    ]
    candidates.sort(key=lambda index: -float(modal.periods_s[index]))
    selected: list[int] = []
    for index in candidates:
        if any(
            math.isclose(
                float(modal.periods_s[index]),
                float(modal.periods_s[prior]),
                rel_tol=1.0e-8,
                abs_tol=1.0e-10,
            )
            and float(ratios[index]) < float(ratios[prior])
            for prior in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= min(3, story_count):
            break
    return DirectionalModalTargets(
        direction=normalized,
        global_mode_indices=tuple(index + 1 for index in selected),
        periods_s=tuple(float(modal.periods_s[index]) for index in selected),
        mode_shapes=tuple(
            tuple(float(value) for value in shapes[index])
            for index in selected
        ),
        effective_mass_ratios=tuple(
            float(ratios[index]) for index in selected
        ),
        participation_factors=tuple(
            float(participation[index]) for index in selected
        ),
    )


def _initial_hinge(
    *,
    elastic_modulus_kn_m2: float,
    inertia_m4: float,
    member_length_m: float,
    yield_moment_knm: float,
    settings: dict[str, Any],
    source: str,
) -> HingeBackbone:
    factor = float(settings["hinge_stiffness_factor"])
    stiffness = factor * 6.0 * elastic_modulus_kn_m2 * inertia_m4
    stiffness /= member_length_m
    pre = float(settings["initial_pre_capping_rotation_rad"])
    post = float(settings["initial_post_capping_rotation_rad"])
    ultimate = float(settings["initial_ultimate_rotation_rad"])
    cap = float(settings["initial_capping_to_yield"])
    residual = float(settings["initial_residual_to_yield"])
    cyclic = float(settings["initial_cyclic_lambda"])
    return HingeBackbone(
        initial_stiffness_knm_rad=stiffness,
        yield_moment_pos_knm=yield_moment_knm,
        yield_moment_neg_knm=yield_moment_knm,
        pre_capping_rotation_pos_rad=pre,
        pre_capping_rotation_neg_rad=pre,
        post_capping_rotation_pos_rad=post,
        post_capping_rotation_neg_rad=post,
        ultimate_rotation_pos_rad=ultimate,
        ultimate_rotation_neg_rad=ultimate,
        capping_to_yield_pos=cap,
        capping_to_yield_neg=cap,
        residual_to_yield_pos=residual,
        residual_to_yield_neg=residual,
        lambda_s=cyclic,
        lambda_c=cyclic,
        lambda_a=cyclic,
        lambda_k=cyclic,
        source=source,
    )


def full3d_gravity_column_end_axials_kn(
    info: ModelInfo,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Read gravity axial force at both ends of every Full 3D column.

    Each inner pair is ``(bottom, top)`` compression magnitude.  Querying the
    built source model preserves edge/interior tributary-load differences
    that an average gravity load per column would erase.
    """

    ops = _ops()
    rows: list[tuple[tuple[float, float], ...]] = []
    for tags in info.column_element_tags_by_storey:
        story: list[tuple[float, float]] = []
        for tag in tags:
            force = tuple(float(value) for value in ops.eleForce(tag))
            if len(force) < 9:
                raise RuntimeError(
                    f"Column {tag}: incomplete Full 3D end-force vector"
                )
            story.append((abs(force[2]), abs(force[8])))
        rows.append(tuple(story))
    return tuple(rows)


def _column_nominal_moment_at_axial_knm(
    building: dict[str, Any],
    config: dict[str, Any],
    axial_kn: float,
) -> float:
    metadata = {}
    raw_metadata = building.get("design_metadata_json")
    if raw_metadata:
        try:
            import json

            metadata = json.loads(str(raw_metadata))
        except (TypeError, ValueError):
            metadata = {}
    hoop_diameter = float(
        metadata.get(
            "column_hoop_diameter_m",
            building.get("column_hoop_diameter_m", 0.012),
        )
    )
    result = column_pm_capacity_at_axial(
        width_m=float(building["column_b_m"]),
        depth_m=float(building["column_h_m"]),
        bar_count=int(building["column_bar_count"]),
        bar_diameter_m=float(building["column_bar_diameter_m"]),
        clear_cover_m=float(
            config["model"]["material_detailing"]["clear_cover_m"]
        ),
        hoop_diameter_m=hoop_diameter,
        fc_kn_m2=float(building["fc_ksc"]) * KSC_TO_KN_M2,
        fy_kn_m2=float(
            building.get(
                "steel_fy_ksc",
                config["building"]["steel_fy_ksc"],
            )
        )
        * KSC_TO_KN_M2,
        steel_elastic_modulus_mpa=float(
            config["model"]["steel_hysteretic"][
                "elastic_modulus_mpa"
            ]
        ),
        target_axial_kn=max(float(axial_kn), 0.0),
        strength_mode="nominal",
    )
    if not bool(result["valid"]):
        raise RuntimeError(
            "Column gravity axial force is outside the nominal P-M surface"
        )
    return float(result["moment_capacity_knm"])


def _story_column_hinge_moments_knm(
    building: dict[str, Any],
    config: dict[str, Any],
    column_end_axials_kn: tuple[
        tuple[tuple[float, float], ...], ...
    ],
) -> tuple[tuple[float, float], ...]:
    """Apply IFB Equation (2) using each column's gravity axial force."""

    rows: list[tuple[float, float]] = []
    for story in column_end_axials_kn:
        bottom = sum(
            _column_nominal_moment_at_axial_knm(
                building, config, pair[0]
            )
            for pair in story
        )
        top = sum(
            _column_nominal_moment_at_axial_knm(
                building, config, pair[1]
            )
            for pair in story
        )
        rows.append((float(bottom), float(top)))
    return tuple(rows)


def _joint_equilibrium_story_evidence(
    *,
    number_of_bays: int,
    direction: str,
    column_below_top_moments_knm: tuple[float, ...],
    column_above_bottom_moments_knm: tuple[float, ...] | None,
    beam_moment_pos_knm: float,
    beam_moment_neg_knm: float,
) -> dict[str, Any]:
    """Condense joint capacities without erasing the local mechanism.

    The original IFB procedure limits the contribution of the beam ends at
    each source-frame joint by the moment resistance of the adjacent
    columns.  Summing first and applying one global reduction afterwards is
    not equivalent: it lets strong joints incorrectly subsidise weak joints.

    The returned equivalent moment is for *one* Fishbone arm.  A physical
    beam has two ends while the Fishbone has two arms, so one half of the
    sum of all locally limited beam-end moments is assigned to each arm.
    The weak/strong group counts implement the IFB3 within-storey
    regularity check.  Mixed groups require the two-substructure topology
    before production certification.
    """

    bays = int(number_of_bays)
    normalized = direction.upper()
    if bays < 1:
        raise ValueError("Joint-equilibrium reduction requires at least one bay")
    if normalized not in {"X", "Y"}:
        raise ValueError("Joint-equilibrium direction must be X or Y")
    grid = bays + 1
    expected = grid**2
    if len(column_below_top_moments_knm) != expected:
        raise ValueError("Below-storey column capacity topology is inconsistent")
    if (
        column_above_bottom_moments_knm is not None
        and len(column_above_bottom_moments_knm) != expected
    ):
        raise ValueError("Above-storey column capacity topology is inconsistent")
    if beam_moment_pos_knm <= 0.0 or beam_moment_neg_knm <= 0.0:
        raise ValueError("Beam characteristic moments must be positive")

    joint_rows: list[dict[str, Any]] = []
    for ix in range(grid):
        for iy in range(grid):
            index = ix * grid + iy
            column_moment = float(column_below_top_moments_knm[index])
            if column_above_bottom_moments_knm is not None:
                column_moment += float(
                    column_above_bottom_moments_knm[index]
                )
            coordinate = ix if normalized == "X" else iy
            adjacent_beams = int(coordinate > 0) + int(coordinate < bays)
            beam_pos = adjacent_beams * float(beam_moment_pos_knm)
            beam_neg = adjacent_beams * float(beam_moment_neg_knm)
            reference_beam = max(0.5 * (beam_pos + beam_neg), 1.0e-12)
            ratio = column_moment / reference_beam
            group = "column_governed" if ratio < 1.0 else "beam_governed"
            joint_rows.append(
                {
                    "ix": ix,
                    "iy": iy,
                    "adjacent_beam_count": adjacent_beams,
                    "column_moment_knm": column_moment,
                    "beam_moment_pos_knm": beam_pos,
                    "beam_moment_neg_knm": beam_neg,
                    "column_to_beam_ratio": ratio,
                    "mechanism_group": group,
                    "limited_beam_moment_pos_knm": min(
                        beam_pos, column_moment
                    ),
                    "limited_beam_moment_neg_knm": min(
                        beam_neg, column_moment
                    ),
                }
            )

    ratios = tuple(
        float(row["column_to_beam_ratio"]) for row in joint_rows
    )
    weak_count = sum(
        row["mechanism_group"] == "column_governed" for row in joint_rows
    )
    strong_count = len(joint_rows) - weak_count
    full_pos = 0.5 * sum(
        float(row["beam_moment_pos_knm"]) for row in joint_rows
    )
    full_neg = 0.5 * sum(
        float(row["beam_moment_neg_knm"]) for row in joint_rows
    )
    limited_pos = 0.5 * sum(
        float(row["limited_beam_moment_pos_knm"]) for row in joint_rows
    )
    limited_neg = 0.5 * sum(
        float(row["limited_beam_moment_neg_knm"]) for row in joint_rows
    )
    return {
        "direction": normalized,
        "joint_count": len(joint_rows),
        "column_governed_joint_count": weak_count,
        "beam_governed_joint_count": strong_count,
        "mixed_mechanism_groups": bool(weak_count and strong_count),
        "column_to_beam_ratio_min": min(ratios),
        "column_to_beam_ratio_median": float(np.median(ratios)),
        "column_to_beam_ratio_max": max(ratios),
        "unreduced_beam_hinge_moment_pos_knm": full_pos,
        "unreduced_beam_hinge_moment_neg_knm": full_neg,
        "equivalent_beam_hinge_moment_pos_knm": limited_pos,
        "equivalent_beam_hinge_moment_neg_knm": limited_neg,
        "beam_strength_retained_pos": limited_pos / max(full_pos, 1.0e-12),
        "beam_strength_retained_neg": limited_neg / max(full_neg, 1.0e-12),
        "joint_rows": tuple(joint_rows),
    }


def _story_joint_mechanism_evidence(
    building: dict[str, Any],
    config: dict[str, Any],
    column_end_axials_kn: tuple[
        tuple[tuple[float, float], ...], ...
    ],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Apply IFB joint equilibrium at every floor and direction."""

    stories = len(column_end_axials_kn)
    bays = int(building["number_of_bays"])
    expected_columns = (bays + 1) ** 2
    if stories < 1 or any(
        len(story) != expected_columns for story in column_end_axials_kn
    ):
        raise ValueError("Column axial-force topology is inconsistent")

    raw_metadata = building.get("design_metadata_json")
    metadata: dict[str, Any] = {}
    if raw_metadata:
        try:
            import json

            metadata = json.loads(str(raw_metadata))
        except (TypeError, ValueError):
            metadata = {}
    phi = float(fishbone_config(config)["nominal_flexural_phi"])
    fallback = float(building["beam_phi_mn_knm"]) / phi
    beam_pos = float(
        metadata.get("beam_nominal_mn_positive_knm", fallback)
    )
    beam_neg = float(
        metadata.get("beam_nominal_mn_negative_knm", fallback)
    )

    bottom_rows: list[tuple[float, ...]] = []
    top_rows: list[tuple[float, ...]] = []
    for story in column_end_axials_kn:
        bottom_rows.append(
            tuple(
                _column_nominal_moment_at_axial_knm(
                    building, config, pair[0]
                )
                for pair in story
            )
        )
        top_rows.append(
            tuple(
                _column_nominal_moment_at_axial_knm(
                    building, config, pair[1]
                )
                for pair in story
            )
        )

    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for direction in ("X", "Y"):
        floors: list[dict[str, Any]] = []
        for floor in range(stories):
            floors.append(
                _joint_equilibrium_story_evidence(
                    number_of_bays=bays,
                    direction=direction,
                    column_below_top_moments_knm=top_rows[floor],
                    column_above_bottom_moments_knm=(
                        bottom_rows[floor + 1]
                        if floor + 1 < stories
                        else None
                    ),
                    beam_moment_pos_knm=beam_pos,
                    beam_moment_neg_knm=beam_neg,
                )
            )
        output[direction] = tuple(floors)
    return output


def _equivalent_beam_gravity_load_kn_m(
    building: dict[str, Any],
    config: dict[str, Any],
) -> float:
    """Match aggregate fixed-end gravity moment at an IFB half beam.

    Each IFB arm represents all same-side beam ends in one horizontal
    direction.  For an equal-bay frame, the fixed-pinned half-beam line load
    is selected so its central end moment ``w L_half^2 / 8`` equals the sum
    of source-frame fixed-end moments ``sum(w_k L^2 / 12)``.
    """

    bays = int(building["number_of_bays"])
    bay = float(building["bay_width_m"])
    beam_width = float(building["beam_b_m"])
    beam_depth = float(building["beam_h_m"])
    model = config["model"]
    area_load = float(building["dead_load_kn_m2"]) + float(
        model["live_load_mass_fraction"]
    ) * float(building["live_load_kn_m2"])
    self_weight = (
        beam_width
        * beam_depth
        * float(model["concrete_density_kn_m3"])
    )
    sum_line_loads = 0.0
    for grid_line in range(bays + 1):
        adjacent_panels = 1 if grid_line in {0, bays} else 2
        tributary = adjacent_panels * bay / 4.0
        line_load = area_load * tributary + self_weight
        sum_line_loads += bays * line_load
    aggregate_end_moment = sum_line_loads * bay**2 / 12.0
    half = bay / 2.0
    return 8.0 * aggregate_end_moment / half**2


def create_physics_artifact(
    building: dict[str, Any],
    config: dict[str, Any],
    full3d_info: ModelInfo,
    full3d_modal: ModalResult,
    *,
    gravity_column_end_axials_kn: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    | None = None,
) -> FishboneArtifact:
    """Create the uncalibrated, research-equation IFB3 starting artifact."""

    stories = int(building["stories"])
    bays = int(building["number_of_bays"])
    if stories != len(full3d_info.floor_elevations_m):
        raise ValueError("Full 3D model and building storey counts differ")
    if not full3d_info.floor_masses_x_kn_s2_m:
        raise ValueError("Full 3D floor masses are missing")
    if not full3d_info.gravity_floor_loads_kn:
        raise ValueError("Full 3D gravity-floor loads are missing")
    settings = fishbone_config(config)
    bay = float(building["bay_width_m"])
    height = float(building["story_height_m"])
    plan_x = bays * bay
    plan_y = bays * bay
    column_count = (bays + 1) ** 2
    beam_count_direction = bays * (bays + 1)
    column_b = float(building["column_b_m"])
    column_h = float(building["column_h_m"])
    beam_b = float(building["beam_b_m"])
    beam_h = float(building["beam_h_m"])
    column_area = column_count * column_b * column_h
    column_ix = column_count * column_b * column_h**3 / 12.0
    column_iy = column_count * column_h * column_b**3 / 12.0
    column_j = column_count * (
        column_b * column_h**3 + column_h * column_b**3
    ) / 12.0
    beam_area = beam_count_direction * beam_b * beam_h
    beam_i = beam_count_direction * beam_b * beam_h**3 / 12.0
    beam_j = beam_count_direction * (
        beam_b * beam_h**3 + beam_h * beam_b**3
    ) / 12.0
    fc = float(building["fc_ksc"]) * KSC_TO_KN_M2
    peak_strain = abs(float(config["model"]["hognestad"]["peak_strain"]))
    elastic_modulus = 2.0 * fc / peak_strain
    shear_modulus = elastic_modulus / (2.0 * (1.0 + 0.20))
    phi = float(settings["nominal_flexural_phi"])
    if not 0.0 < phi <= 1.0:
        raise ValueError("Fishbone nominal flexural phi must lie in (0, 1]")
    beam_yield = (
        beam_count_direction
        * float(building["beam_phi_mn_knm"])
        / phi
    )
    column_yield = (
        column_count * float(building["column_phi_mn_knm"]) / phi
    )
    beam_hinge = _initial_hinge(
        elastic_modulus_kn_m2=elastic_modulus,
        inertia_m4=beam_i,
        member_length_m=bay / 2.0,
        yield_moment_knm=beam_yield,
        settings=settings,
        source="R2_sum_moment_physics_initialisation",
    )
    column_hinge_x = _initial_hinge(
        elastic_modulus_kn_m2=elastic_modulus,
        inertia_m4=column_ix,
        member_length_m=height,
        yield_moment_knm=column_yield,
        settings=settings,
        source="R2_sum_moment_physics_initialisation",
    )
    column_hinge_y = _initial_hinge(
        elastic_modulus_kn_m2=elastic_modulus,
        inertia_m4=column_iy,
        member_length_m=height,
        yield_moment_knm=column_yield,
        settings=settings,
        source="R2_sum_moment_physics_initialisation",
    )
    equivalent_gravity = _equivalent_beam_gravity_load_kn_m(
        building, config
    )
    gravity_floors = tuple(
        float(value) for value in full3d_info.gravity_floor_loads_kn
    )
    if gravity_column_end_axials_kn is not None:
        if (
            len(gravity_column_end_axials_kn) != stories
            or any(
                len(row) != column_count
                for row in gravity_column_end_axials_kn
            )
        ):
            raise ValueError(
                "Full 3D gravity column axial-force topology is inconsistent"
            )
        column_hinge_moments = _story_column_hinge_moments_knm(
            building,
            config,
            gravity_column_end_axials_kn,
        )
        column_strength_source = (
            "R2_Eq2_sum_nominal_PM_at_full3d_gravity_axial"
        )
        joint_evidence = _story_joint_mechanism_evidence(
            building,
            config,
            gravity_column_end_axials_kn,
        )
        beam_strength_source = (
            "R1_IFB_joint_by_joint_maximum_moment_equilibrium"
        )
    else:
        column_hinge_moments = tuple(
            (column_yield, column_yield) for _ in range(stories)
        )
        column_strength_source = (
            "average_axial_fallback_requires_story_imposed_calibration"
        )
        joint_evidence = {
            direction: tuple(
                {
                    "direction": direction,
                    "mixed_mechanism_groups": False,
                    "equivalent_beam_hinge_moment_pos_knm": beam_yield,
                    "equivalent_beam_hinge_moment_neg_knm": beam_yield,
                    "beam_strength_retained_pos": 1.0,
                    "beam_strength_retained_neg": 1.0,
                    "source": "no_gravity_axial_fallback",
                }
                for _ in range(stories)
            )
            for direction in ("X", "Y")
        }
        beam_strength_source = (
            "sum_nominal_beam_moments_fallback_requires_joint_check"
        )
    mixed_joint_mechanism = any(
        bool(row["mixed_mechanism_groups"])
        for direction in ("X", "Y")
        for row in joint_evidence[direction]
    )
    story_rows: list[EquivalentStory] = []
    for index in range(stories):
        gravity_above = float(sum(gravity_floors[index:]))
        bottom_hinge_x = replace(
            column_hinge_x,
            yield_moment_pos_knm=column_hinge_moments[index][0],
            yield_moment_neg_knm=column_hinge_moments[index][0],
            source=column_strength_source,
        )
        top_hinge_x = replace(
            column_hinge_x,
            yield_moment_pos_knm=column_hinge_moments[index][1],
            yield_moment_neg_knm=column_hinge_moments[index][1],
            source=column_strength_source,
        )
        bottom_hinge_y = replace(
            column_hinge_y,
            yield_moment_pos_knm=column_hinge_moments[index][0],
            yield_moment_neg_knm=column_hinge_moments[index][0],
            source=column_strength_source,
        )
        top_hinge_y = replace(
            column_hinge_y,
            yield_moment_pos_knm=column_hinge_moments[index][1],
            yield_moment_neg_knm=column_hinge_moments[index][1],
            source=column_strength_source,
        )
        beam_hinge_x = replace(
            beam_hinge,
            yield_moment_pos_knm=float(
                joint_evidence["X"][index][
                    "equivalent_beam_hinge_moment_pos_knm"
                ]
            ),
            yield_moment_neg_knm=float(
                joint_evidence["X"][index][
                    "equivalent_beam_hinge_moment_neg_knm"
                ]
            ),
            source=beam_strength_source,
        )
        beam_hinge_y = replace(
            beam_hinge,
            yield_moment_pos_knm=float(
                joint_evidence["Y"][index][
                    "equivalent_beam_hinge_moment_pos_knm"
                ]
            ),
            yield_moment_neg_knm=float(
                joint_evidence["Y"][index][
                    "equivalent_beam_hinge_moment_neg_knm"
                ]
            ),
            source=beam_strength_source,
        )
        story_rows.append(
            EquivalentStory(
                story=index + 1,
                height_m=height,
                column_count=column_count,
                beam_count_per_direction=beam_count_direction,
                column_area_m2=column_area,
                column_inertia_x_m4=column_ix,
                column_inertia_y_m4=column_iy,
                column_torsion_m4=max(column_j, 1.0e-12),
                beam_area_x_m2=beam_area,
                beam_area_y_m2=beam_area,
                beam_inertia_x_m4=beam_i,
                beam_inertia_y_m4=beam_i,
                beam_torsion_x_m4=max(beam_j, 1.0e-12),
                beam_torsion_y_m4=max(beam_j, 1.0e-12),
                beam_half_length_x_m=bay / 2.0,
                beam_half_length_y_m=bay / 2.0,
                column_stiffness_modifier_x=1.0,
                column_stiffness_modifier_y=1.0,
                beam_stiffness_modifier_x=1.0,
                beam_stiffness_modifier_y=1.0,
                beam_uniform_gravity_x_kn_m=equivalent_gravity,
                beam_uniform_gravity_y_kn_m=equivalent_gravity,
                gravity_load_above_kn=gravity_above,
                geometric_stiffness_kn_m=-gravity_above / height,
                column_bottom_hinge_x=bottom_hinge_x,
                column_bottom_hinge_y=bottom_hinge_y,
                column_top_hinge_x=top_hinge_x,
                column_top_hinge_y=top_hinge_y,
                beam_hinge_x=beam_hinge_x,
                beam_hinge_y=beam_hinge_y,
            )
        )
    masses_x = tuple(
        float(value) for value in full3d_info.floor_masses_x_kn_s2_m
    )
    masses_y = tuple(
        float(value) for value in full3d_info.floor_masses_y_kn_s2_m
    )
    polar = tuple(
        0.5
        * (masses_x[index] + masses_y[index])
        * (plan_x**2 + plan_y**2)
        / 12.0
        for index in range(stories)
    )
    payload = {
        "schema_version": FISHBONE_ARTIFACT_SCHEMA_VERSION,
        "building_id": str(building["building_id"]),
        "model_hash": str(building.get("model_hash", "")),
        "model_config": config["model"],
        "fishbone_config": settings,
        "column_hinge_moments_knm": column_hinge_moments,
        "column_strength_source": column_strength_source,
        "beam_strength_source": beam_strength_source,
        "joint_mechanism_evidence": joint_evidence,
    }
    artifact = FishboneArtifact(
        building_id=str(building["building_id"]),
        source_model_hash=str(building.get("model_hash", "")),
        source_signature=stable_hash(payload),
        story_count=stories,
        number_of_bays=bays,
        plan_length_x_m=plan_x,
        plan_length_y_m=plan_y,
        floor_elevations_m=tuple(
            float(value) for value in full3d_info.floor_elevations_m
        ),
        story_heights_m=tuple(height for _ in range(stories)),
        floor_masses_x_kn_s2_m=masses_x,
        floor_masses_y_kn_s2_m=masses_y,
        floor_polar_masses_kn_m_s2=polar,
        gravity_floor_loads_kn=gravity_floors,
        elastic_modulus_kn_m2=elastic_modulus,
        shear_modulus_kn_m2=shear_modulus,
        stories=tuple(story_rows),
        target_modal_x=_directional_targets(
            full3d_modal, "X", stories
        ),
        target_modal_y=_directional_targets(
            full3d_modal, "Y", stories
        ),
        applicability_flags=tuple(
            (
                "physics-initialisation-only",
                "elastic-calibration-required",
                "monotonic-calibration-required",
                "cyclic-calibration-required",
                "predominantly-plan-symmetric-only",
                "torsion-not-certified",
                "xy-strength-interaction-not-supported",
                "shear-failure-excluded-by-research-scope",
            )
            + (
                (
                    "within-story-mixed-mechanism-requires-dual-substructure"
                )
                if mixed_joint_mechanism
                else (
                    "single-hinge-set-supported-by-joint-classification"
                ),
            )
        ),
        calibration_metrics={
            "reference_ids": (
                "R1",
                "R2",
                "R3",
                "R6",
                "R8",
                "R9",
                "R10",
            ),
            "column_count": column_count,
            "beam_count_per_direction": beam_count_direction,
            "equivalent_beam_gravity_kn_m": equivalent_gravity,
            "column_strength_source": column_strength_source,
            "column_hinge_moments_knm": column_hinge_moments,
            "beam_strength_source": beam_strength_source,
            "joint_mechanism_evidence": joint_evidence,
            "mixed_joint_mechanism_requires_dual_substructure": (
                mixed_joint_mechanism
            ),
            "column_gravity_axial_force_range_kn": (
                (
                    min(
                        value
                        for story in gravity_column_end_axials_kn
                        for pair in story
                        for value in pair
                    ),
                    max(
                        value
                        for story in gravity_column_end_axials_kn
                        for pair in story
                        for value in pair
                    ),
                )
                if gravity_column_end_axials_kn is not None
                else None
            ),
        },
    ).with_hash()
    artifact.validate()
    return artifact


def _define_hinge(
    material_tag: int,
    hinge: HingeBackbone,
    options: dict[str, Any] | None = None,
    *,
    story_index: int = 0,
    story_count: int = 1,
    mechanism: str = "member",
    direction: str | None = None,
) -> None:
    hinge.validate()
    settings = dict(options or {})
    material_model = str(
        settings.get("material_model", "IMKPeakOriented")
    )
    ops = _ops()
    if material_model in {
        "IMKPeakOrientedParallelElasticEnvelope",
        "IMKPeakOrientedDistributedParallelElasticEnvelope",
    }:
        if story_count < 1 or not 0 <= story_index < story_count:
            raise ValueError("Parallel-envelope story index is invalid")
        z = 0.5 if story_count == 1 else story_index / (story_count - 1)
        controls = (
            float(settings["strength_low"]),
            float(settings["strength_mid"]),
            float(settings["strength_high"]),
        )
        profile_scale = float(
            math.exp(
                np.interp(
                    z,
                    (0.0, 0.5, 1.0),
                    np.log(controls),
                )
            )
        )
        if mechanism in {"column", "beam"}:
            default_factor = float(
                settings.get(f"{mechanism}_envelope_factor", 1.0)
            )
            factor_controls = tuple(
                float(
                    settings.get(
                        f"{mechanism}_envelope_{level}_factor",
                        default_factor,
                    )
                )
                for level in ("low", "mid", "high")
            )
            # The correction factor may be zero, so interpolate it directly
            # rather than logarithmically.  Only three population-level
            # controls are used for each mechanism; no building- or
            # story-specific fit is introduced.
            mechanism_factor = float(
                np.interp(z, (0.0, 0.5, 1.0), factor_controls)
            )
            if direction is not None:
                story_factors = settings.get(
                    f"{mechanism}_envelope_story_factors_"
                    f"{direction.lower()}"
                )
                if story_factors is not None:
                    if len(story_factors) != story_count:
                        raise ValueError(
                            "Story envelope factor length is inconsistent"
                        )
                    mechanism_factor = float(story_factors[story_index])
        else:
            factor_controls = (1.0, 1.0, 1.0)
            mechanism_factor = 1.0
        if not all(0.0 <= value <= 3.0 for value in factor_controls):
            raise ValueError("Parallel envelope mechanism factor is invalid")
        strength_scale = 1.0 + mechanism_factor * (profile_scale - 1.0)
        default_pre_scale = float(
            settings.get("envelope_pre_capping_scale", 1.0)
        )
        pre_controls = tuple(
            float(
                settings.get(
                    f"envelope_pre_capping_{level}_scale",
                    default_pre_scale,
                )
            )
            for level in ("low", "mid", "high")
        )
        # A smooth three-control profile captures systematic lower, middle
        # and upper-storey capping differences without adding one parameter
        # per storey or per building.
        pre_scale = float(
            math.exp(
                np.interp(
                    z,
                    (0.0, 0.5, 1.0),
                    np.log(pre_controls),
                )
            )
        )
        if direction is not None:
            story_pre = settings.get(
                f"envelope_pre_capping_story_scales_{direction.lower()}"
            )
            if story_pre is not None:
                if len(story_pre) != story_count:
                    raise ValueError(
                        "Story pre-capping scale length is inconsistent"
                    )
                pre_scale = float(story_pre[story_index])
        residual_fraction = float(
            settings.get("envelope_residual_fraction", 0.0)
        )
        if (
            min(controls) < 1.0
            or min(pre_controls) <= 0.0
            or not 0.0 <= residual_fraction <= 1.0
        ):
            raise ValueError("Parallel elastic envelope controls are invalid")
        base_tag = 1_000_000 + 2 * material_tag
        elastic_tag = base_tag + 1
        _define_hinge(
            base_tag,
            hinge,
            {
                **settings,
                "material_model": (
                    "IMKPeakOrientedDistributed"
                    if material_model
                    == "IMKPeakOrientedDistributedParallelElasticEnvelope"
                    else "IMKPeakOriented"
                ),
            },
            story_index=story_index,
            story_count=story_count,
            mechanism=mechanism,
            direction=direction,
        )
        yield_pos = (
            hinge.yield_moment_pos_knm
            / hinge.initial_stiffness_knm_rad
        )
        yield_neg = (
            hinge.yield_moment_neg_knm
            / hinge.initial_stiffness_knm_rad
        )
        cap_pos = min(
            yield_pos + pre_scale * hinge.pre_capping_rotation_pos_rad,
            0.90 * hinge.ultimate_rotation_pos_rad,
        )
        cap_neg = min(
            yield_neg + pre_scale * hinge.pre_capping_rotation_neg_rad,
            0.90 * hinge.ultimate_rotation_neg_rad,
        )
        extra_pos = (
            (strength_scale - 1.0)
            * hinge.capping_to_yield_pos
            * hinge.yield_moment_pos_knm
        )
        extra_neg = (
            (strength_scale - 1.0)
            * hinge.capping_to_yield_neg
            * hinge.yield_moment_neg_knm
        )
        strains = (
            -hinge.ultimate_rotation_neg_rad,
            -cap_neg,
            -yield_neg,
            0.0,
            yield_pos,
            cap_pos,
            hinge.ultimate_rotation_pos_rad,
        )
        stresses = (
            -residual_fraction * extra_neg,
            -extra_neg,
            0.0,
            0.0,
            0.0,
            extra_pos,
            residual_fraction * extra_pos,
        )
        ops.uniaxialMaterial(
            "ElasticMultiLinear",
            elastic_tag,
            0.0,
            "-strain",
            *strains,
            "-stress",
            *stresses,
        )
        ops.uniaxialMaterial(
            "Parallel", material_tag, base_tag, elastic_tag
        )
        return
    if material_model == "IMKPeakOrientedDistributed":
        dispersion = float(settings.get("distributed_yield_dispersion", 0.20))
        if not 0.0 < dispersion < 0.60:
            raise ValueError("Distributed-yield dispersion is invalid")
        # Three parallel components approximate progressive yielding of the
        # source members that were condensed into one equivalent hinge.  The
        # component stiffnesses sum exactly to the original stiffness and
        # the weighted yield moments sum exactly to the original moment, so
        # the elastic modal model and aggregate nominal strength are
        # preserved.  Only the sequence of yielding/capping is distributed.
        weights = (0.25, 0.50, 0.25)
        yield_factors = (1.0 - dispersion, 1.0, 1.0 + dispersion)
        component_tags = []
        for component, (weight, yield_factor) in enumerate(
            zip(weights, yield_factors, strict=True), 1
        ):
            component_tag = 10_000_000 + 4 * material_tag + component
            component_tags.append(component_tag)
            component_hinge = replace(
                hinge,
                initial_stiffness_knm_rad=(
                    weight * hinge.initial_stiffness_knm_rad
                ),
                yield_moment_pos_knm=(
                    weight * yield_factor * hinge.yield_moment_pos_knm
                ),
                yield_moment_neg_knm=(
                    weight * yield_factor * hinge.yield_moment_neg_knm
                ),
                source=(
                    f"{hinge.source}:distributed-yield-component-"
                    f"{component}-of-3"
                ),
            )
            _define_hinge(
                component_tag,
                component_hinge,
                {"material_model": "IMKPeakOriented"},
            )
        ops.uniaxialMaterial("Parallel", material_tag, *component_tags)
        return
    if material_model == "IMKPinching":
        kappa_f = float(settings.get("kappa_f", 0.50))
        kappa_d = float(settings.get("kappa_d", 0.50))
        if not 0.0 < kappa_f < 1.0 or not 0.0 < kappa_d < 1.0:
            raise ValueError("Fishbone IMKPinching controls are invalid")
        ops.uniaxialMaterial(
            "IMKPinching",
            material_tag,
            hinge.initial_stiffness_knm_rad,
            hinge.pre_capping_rotation_pos_rad,
            hinge.post_capping_rotation_pos_rad,
            hinge.ultimate_rotation_pos_rad,
            hinge.yield_moment_pos_knm,
            hinge.capping_to_yield_pos,
            hinge.residual_to_yield_pos,
            hinge.pre_capping_rotation_neg_rad,
            hinge.post_capping_rotation_neg_rad,
            hinge.ultimate_rotation_neg_rad,
            hinge.yield_moment_neg_knm,
            hinge.capping_to_yield_neg,
            hinge.residual_to_yield_neg,
            hinge.lambda_s,
            hinge.lambda_c,
            hinge.lambda_a,
            hinge.lambda_k,
            hinge.c_s,
            hinge.c_c,
            hinge.c_a,
            hinge.c_k,
            hinge.deterioration_pos,
            hinge.deterioration_neg,
            kappa_f,
            kappa_d,
        )
        return
    if material_model == "Hysteretic":
        pinch_x = float(settings.get("pinch_x", 1.0))
        pinch_y = float(settings.get("pinch_y", 1.0))
        damage_1 = float(settings.get("damage_1", 0.0))
        damage_2 = float(settings.get("damage_2", 0.0))
        beta = float(settings.get("beta", 0.0))
        if not (
            0.0 <= pinch_x <= 1.0
            and 0.0 <= pinch_y <= 1.0
            and damage_1 >= 0.0
            and damage_2 >= 0.0
            and beta >= 0.0
        ):
            raise ValueError("Fishbone Hysteretic options are invalid")
        yield_rotation_pos = (
            hinge.yield_moment_pos_knm
            / hinge.initial_stiffness_knm_rad
        )
        yield_rotation_neg = (
            hinge.yield_moment_neg_knm
            / hinge.initial_stiffness_knm_rad
        )
        capping_rotation_pos = (
            yield_rotation_pos + hinge.pre_capping_rotation_pos_rad
        )
        capping_rotation_neg = (
            yield_rotation_neg + hinge.pre_capping_rotation_neg_rad
        )
        implementation = str(
            settings.get("implementation_version", "hysteretic-envelope-v1")
        )
        if implementation == "hysteretic-envelope-v2-residual-at-ultimate":
            # Hysteretic has only three envelope points.  Putting point 3 at
            # residual onset would extrapolate the descending slope beyond
            # that point and erase residual strength.  Place it at ultimate
            # deformation so a ductile residual branch is retained.
            residual_rotation_pos = hinge.ultimate_rotation_pos_rad
            residual_rotation_neg = hinge.ultimate_rotation_neg_rad
        elif implementation == "hysteretic-envelope-v1":
            residual_rotation_pos = min(
                capping_rotation_pos + hinge.post_capping_rotation_pos_rad,
                hinge.ultimate_rotation_pos_rad,
            )
            residual_rotation_neg = min(
                capping_rotation_neg + hinge.post_capping_rotation_neg_rad,
                hinge.ultimate_rotation_neg_rad,
            )
        else:
            raise ValueError(
                f"Unsupported Fishbone Hysteretic revision: {implementation}"
            )
        ops.uniaxialMaterial(
            "Hysteretic",
            material_tag,
            hinge.yield_moment_pos_knm,
            yield_rotation_pos,
            hinge.capping_to_yield_pos * hinge.yield_moment_pos_knm,
            capping_rotation_pos,
            hinge.residual_to_yield_pos * hinge.yield_moment_pos_knm,
            residual_rotation_pos,
            -hinge.yield_moment_neg_knm,
            -yield_rotation_neg,
            -hinge.capping_to_yield_neg * hinge.yield_moment_neg_knm,
            -capping_rotation_neg,
            -hinge.residual_to_yield_neg * hinge.yield_moment_neg_knm,
            -residual_rotation_neg,
            pinch_x,
            pinch_y,
            damage_1,
            damage_2,
            beta,
        )
        return
    if material_model != "IMKPeakOriented":
        raise ValueError(
            f"Unsupported Fishbone hinge material: {material_model}"
        )
    ops.uniaxialMaterial(
        "IMKPeakOriented",
        material_tag,
        hinge.initial_stiffness_knm_rad,
        hinge.pre_capping_rotation_pos_rad,
        hinge.post_capping_rotation_pos_rad,
        hinge.ultimate_rotation_pos_rad,
        hinge.yield_moment_pos_knm,
        hinge.capping_to_yield_pos,
        hinge.residual_to_yield_pos,
        hinge.pre_capping_rotation_neg_rad,
        hinge.post_capping_rotation_neg_rad,
        hinge.ultimate_rotation_neg_rad,
        hinge.yield_moment_neg_knm,
        hinge.capping_to_yield_neg,
        hinge.residual_to_yield_neg,
        hinge.lambda_s,
        hinge.lambda_c,
        hinge.lambda_a,
        hinge.lambda_k,
        hinge.c_s,
        hinge.c_c,
        hinge.c_a,
        hinge.c_k,
        hinge.deterioration_pos,
        hinge.deterioration_neg,
    )


def _fractional_hinge(
    hinge: HingeBackbone,
    *,
    stiffness_fraction: float,
    strength_fraction_pos: float,
    strength_fraction_neg: float,
    source_suffix: str,
) -> HingeBackbone:
    if min(
        stiffness_fraction,
        strength_fraction_pos,
        strength_fraction_neg,
    ) <= 0.0:
        raise ValueError("Dual-substructure hinge fractions must be positive")
    result = replace(
        hinge,
        initial_stiffness_knm_rad=(
            hinge.initial_stiffness_knm_rad * stiffness_fraction
        ),
        yield_moment_pos_knm=(
            hinge.yield_moment_pos_knm * strength_fraction_pos
        ),
        yield_moment_neg_knm=(
            hinge.yield_moment_neg_knm * strength_fraction_neg
        ),
        source=f"{hinge.source}:{source_suffix}",
    )
    result.validate()
    return result


def _dual_substructure_fractions(
    artifact: FishboneArtifact,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Split source joints into two deterministic capacity quantiles."""

    evidence = artifact.calibration_metrics.get(
        "joint_mechanism_evidence", {}
    )
    direction_rows = list(evidence.get(direction, ()))
    rows = (
        list(direction_rows[story_index].get("joint_rows", ()))
        if story_index < len(direction_rows)
        else []
    )
    if len(rows) < 2:
        return (
            {
                "column_stiffness": 0.5,
                "column_strength": 0.5,
                "beam_stiffness": 0.5,
                "beam_strength_pos": 0.5,
                "beam_strength_neg": 0.5,
            },
            {
                "column_stiffness": 0.5,
                "column_strength": 0.5,
                "beam_stiffness": 0.5,
                "beam_strength_pos": 0.5,
                "beam_strength_neg": 0.5,
            },
        )
    column_governed = [
        row for row in rows if row["mechanism_group"] == "column_governed"
    ]
    beam_governed = [
        row for row in rows if row["mechanism_group"] == "beam_governed"
    ]
    if column_governed and beam_governed:
        groups = (column_governed, beam_governed)
    else:
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["column_to_beam_ratio"]),
                int(row["ix"]),
                int(row["iy"]),
            ),
        )
        split = len(ordered) // 2
        groups = (ordered[:split], ordered[split:])

    total_columns = float(len(rows))
    total_column_strength = sum(
        float(row["column_moment_knm"]) for row in rows
    )
    total_beam_count = sum(int(row["adjacent_beam_count"]) for row in rows)
    total_beam_pos = sum(
        float(row["limited_beam_moment_pos_knm"]) for row in rows
    )
    total_beam_neg = sum(
        float(row["limited_beam_moment_neg_knm"]) for row in rows
    )
    output = []
    for group in groups:
        output.append(
            {
                "column_stiffness": len(group) / total_columns,
                "column_strength": sum(
                    float(row["column_moment_knm"]) for row in group
                )
                / max(total_column_strength, 1.0e-12),
                "beam_stiffness": sum(
                    int(row["adjacent_beam_count"]) for row in group
                )
                / max(total_beam_count, 1),
                "beam_strength_pos": sum(
                    float(row["limited_beam_moment_pos_knm"])
                    for row in group
                )
                / max(total_beam_pos, 1.0e-12),
                "beam_strength_neg": sum(
                    float(row["limited_beam_moment_neg_knm"])
                    for row in group
                )
                / max(total_beam_neg, 1.0e-12),
            }
        )
    for name in output[0]:
        if not math.isclose(
            output[0][name] + output[1][name], 1.0, abs_tol=1.0e-10
        ):
            raise AssertionError("Dual-substructure fractions do not close")
    return output[0], output[1]


def _dual_height_stable_substructure_fractions(
    artifact: FishboneArtifact,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Split joints into two weak/strong groups fixed over the full height.

    The original dual topology reclassifies joints independently at every
    storey.  A reduced spine can consequently change physical identity with
    height, which is especially harmful to higher-mode mechanism migration.
    This variant ranks each source joint coordinate by its geometric-mean
    column-to-beam ratio over all storeys, splits the coordinates once, and
    then evaluates the actual stiffness and strength fractions at each storey
    using that fixed membership.
    """

    evidence = artifact.calibration_metrics.get(
        "joint_mechanism_evidence", {}
    )
    direction_rows = list(evidence.get(direction, ()))
    all_story_rows = [
        list(item.get("joint_rows", ())) for item in direction_rows
    ]
    rows = (
        all_story_rows[story_index]
        if story_index < len(all_story_rows)
        else []
    )
    if len(rows) < 2 or not all_story_rows:
        return _dual_substructure_fractions(
            artifact, direction, story_index
        )

    score_rows: dict[tuple[int, int], list[float]] = {}
    for story_rows in all_story_rows:
        coordinates = {
            (int(row["ix"]), int(row["iy"])) for row in story_rows
        }
        if score_rows and coordinates != set(score_rows):
            raise ValueError(
                "Height-stable dual joint coordinates vary between storeys"
            )
        for row in story_rows:
            coordinate = (int(row["ix"]), int(row["iy"]))
            ratio = float(row["column_to_beam_ratio"])
            if ratio <= 0.0 or not math.isfinite(ratio):
                raise ValueError(
                    "Height-stable dual capacity ratio must be positive"
                )
            score_rows.setdefault(coordinate, []).append(math.log(ratio))
    ordered_coordinates = sorted(
        score_rows,
        key=lambda coordinate: (
            float(np.mean(score_rows[coordinate])),
            coordinate,
        ),
    )
    split = len(ordered_coordinates) // 2
    coordinate_groups = (
        set(ordered_coordinates[:split]),
        set(ordered_coordinates[split:]),
    )
    if any(not group for group in coordinate_groups):
        return _dual_substructure_fractions(
            artifact, direction, story_index
        )
    row_coordinates = {
        (int(row["ix"]), int(row["iy"])) for row in rows
    }
    if row_coordinates != set(ordered_coordinates):
        raise ValueError(
            "Height-stable dual storey is missing a source joint coordinate"
        )
    groups = tuple(
        [
            row
            for row in rows
            if (int(row["ix"]), int(row["iy"])) in coordinates
        ]
        for coordinates in coordinate_groups
    )
    total_columns = float(len(rows))
    total_column_strength = sum(
        float(row["column_moment_knm"]) for row in rows
    )
    total_beam_count = sum(int(row["adjacent_beam_count"]) for row in rows)
    total_beam_pos = sum(
        float(row["limited_beam_moment_pos_knm"]) for row in rows
    )
    total_beam_neg = sum(
        float(row["limited_beam_moment_neg_knm"]) for row in rows
    )
    output = tuple(
        {
            "column_stiffness": len(group) / total_columns,
            "column_strength": sum(
                float(row["column_moment_knm"]) for row in group
            )
            / max(total_column_strength, 1.0e-12),
            "beam_stiffness": sum(
                int(row["adjacent_beam_count"]) for row in group
            )
            / max(total_beam_count, 1),
            "beam_strength_pos": sum(
                float(row["limited_beam_moment_pos_knm"])
                for row in group
            )
            / max(total_beam_pos, 1.0e-12),
            "beam_strength_neg": sum(
                float(row["limited_beam_moment_neg_knm"])
                for row in group
            )
            / max(total_beam_neg, 1.0e-12),
        }
        for group in groups
    )
    for name in output[0]:
        if not math.isclose(
            output[0][name] + output[1][name],
            1.0,
            abs_tol=1.0e-10,
        ):
            raise AssertionError(
                "Height-stable dual-substructure fractions do not close"
            )
    return output[0], output[1]


def _dual_edge_chord_substructure_fractions(
    artifact: FishboneArtifact,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Condense source joints into two physical overturning-edge groups.

    Edge coordinates map completely to their corresponding chord.  Interior
    coordinates are shared by linear barycentric weights, so symmetric odd
    grids do not assign the centre line arbitrarily and each source joint's
    stiffness/strength is conserved exactly.  Optional positive/negative P-M
    modulation redistributes strength between the two chords without changing
    aggregate strength in either loading sign.
    """

    evidence = artifact.calibration_metrics.get(
        "joint_mechanism_evidence", {}
    )
    direction_rows = list(evidence.get(direction, ()))
    rows = (
        list(direction_rows[story_index].get("joint_rows", ()))
        if story_index < len(direction_rows)
        else []
    )
    if len(rows) < 2:
        return _dual_substructure_fractions(
            artifact, direction, story_index
        )
    coordinate_key = "ix" if direction == "X" else "iy"
    coordinates = np.asarray(
        [float(row[coordinate_key]) for row in rows], dtype=float
    )
    lower = float(np.min(coordinates))
    upper = float(np.max(coordinates))
    if upper <= lower:
        return _dual_substructure_fractions(
            artifact, direction, story_index
        )
    positive_weights = (coordinates - lower) / (upper - lower)
    weights = (1.0 - positive_weights, positive_weights)
    total_column_strength = sum(
        float(row["column_moment_knm"]) for row in rows
    )
    total_beam_count = sum(int(row["adjacent_beam_count"]) for row in rows)
    total_beam_pos = sum(
        float(row["limited_beam_moment_pos_knm"]) for row in rows
    )
    total_beam_neg = sum(
        float(row["limited_beam_moment_neg_knm"]) for row in rows
    )
    output = []
    for group_weights in weights:
        output.append(
            {
                "column_stiffness": float(np.sum(group_weights))
                / len(rows),
                "column_strength": sum(
                    float(weight) * float(row["column_moment_knm"])
                    for weight, row in zip(group_weights, rows, strict=True)
                )
                / max(total_column_strength, 1.0e-12),
                "beam_stiffness": sum(
                    float(weight) * int(row["adjacent_beam_count"])
                    for weight, row in zip(group_weights, rows, strict=True)
                )
                / max(total_beam_count, 1),
                "beam_strength_pos": sum(
                    float(weight)
                    * float(row["limited_beam_moment_pos_knm"])
                    for weight, row in zip(group_weights, rows, strict=True)
                )
                / max(total_beam_pos, 1.0e-12),
                "beam_strength_neg": sum(
                    float(weight)
                    * float(row["limited_beam_moment_neg_knm"])
                    for weight, row in zip(group_weights, rows, strict=True)
                )
                / max(total_beam_neg, 1.0e-12),
            }
        )
    modulation = float(
        artifact.calibration_metrics.get("fishbone_v3_hysteresis", {}).get(
            "axial_pm_capacity_modulation", 0.0
        )
    )
    if not 0.0 <= modulation <= 0.40:
        raise ValueError("Axial P-M capacity modulation is invalid")
    base_strength = [float(item["column_strength"]) for item in output]
    positive = (
        base_strength[0] * (1.0 - modulation),
        base_strength[1] * (1.0 + modulation),
    )
    negative = (
        base_strength[0] * (1.0 + modulation),
        base_strength[1] * (1.0 - modulation),
    )
    positive_scale = max(sum(positive), 1.0e-12)
    negative_scale = max(sum(negative), 1.0e-12)
    for index, item in enumerate(output):
        item["column_strength_pos"] = positive[index] / positive_scale
        item["column_strength_neg"] = negative[index] / negative_scale
    for name in (
        "column_stiffness",
        "column_strength",
        "column_strength_pos",
        "column_strength_neg",
        "beam_stiffness",
        "beam_strength_pos",
        "beam_strength_neg",
    ):
        if not math.isclose(
            output[0][name] + output[1][name],
            1.0,
            abs_tol=1.0e-10,
        ):
            raise AssertionError("Dual edge-chord fractions do not close")
    return output[0], output[1]


def _joint_ensemble_substructure_fractions(
    artifact: FishboneArtifact,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], ...]:
    """Preserve each source-joint line as one continuous reduced spine."""

    evidence = artifact.calibration_metrics.get(
        "joint_mechanism_evidence", {}
    )
    direction_rows = list(evidence.get(direction, ()))
    rows = (
        list(direction_rows[story_index].get("joint_rows", ()))
        if story_index < len(direction_rows)
        else []
    )
    if not rows:
        return (
            {
                "column_stiffness": 1.0,
                "column_strength": 1.0,
                "beam_stiffness": 1.0,
                "beam_strength_pos": 1.0,
                "beam_strength_neg": 1.0,
            },
        )
    ordered = sorted(
        rows,
        key=lambda row: (int(row["ix"]), int(row["iy"])),
    )
    total_columns = float(len(ordered))
    total_column_strength = sum(
        float(row["column_moment_knm"]) for row in ordered
    )
    total_beam_count = sum(
        int(row["adjacent_beam_count"]) for row in ordered
    )
    total_beam_pos = sum(
        float(row["limited_beam_moment_pos_knm"]) for row in ordered
    )
    total_beam_neg = sum(
        float(row["limited_beam_moment_neg_knm"]) for row in ordered
    )
    output = tuple(
        {
            "column_stiffness": 1.0 / total_columns,
            "column_strength": float(row["column_moment_knm"])
            / max(total_column_strength, 1.0e-12),
            "beam_stiffness": int(row["adjacent_beam_count"])
            / max(total_beam_count, 1),
            "beam_strength_pos": float(
                row["limited_beam_moment_pos_knm"]
            )
            / max(total_beam_pos, 1.0e-12),
            "beam_strength_neg": float(
                row["limited_beam_moment_neg_knm"]
            )
            / max(total_beam_neg, 1.0e-12),
        }
        for row in ordered
    )
    for name in output[0]:
        if not math.isclose(
            sum(item[name] for item in output), 1.0, abs_tol=1.0e-10
        ):
            raise AssertionError("Joint-ensemble fractions do not close")
    return output


def _triple_quantile_substructure_fractions(
    artifact: FishboneArtifact,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], ...]:
    """Use three capacity quantiles whose joint membership is height-stable."""

    evidence = artifact.calibration_metrics.get(
        "joint_mechanism_evidence", {}
    )
    direction_rows = list(evidence.get(direction, ()))
    all_story_rows = [
        list(item.get("joint_rows", ())) for item in direction_rows
    ]
    rows = (
        all_story_rows[story_index]
        if story_index < len(all_story_rows)
        else []
    )
    if len(rows) < 3:
        return _joint_ensemble_substructure_fractions(
            artifact, direction, story_index
        )
    score_rows: dict[tuple[int, int], list[float]] = {}
    for story_rows in all_story_rows:
        for row in story_rows:
            coordinate = (int(row["ix"]), int(row["iy"]))
            score_rows.setdefault(coordinate, []).append(
                float(row["column_to_beam_ratio"])
            )
    ordered_coordinates = sorted(
        score_rows,
        key=lambda coordinate: (
            float(np.mean(score_rows[coordinate])),
            coordinate,
        ),
    )
    boundaries = np.linspace(0, len(ordered_coordinates), 4, dtype=int)
    coordinate_groups = tuple(
        set(ordered_coordinates[boundaries[index] : boundaries[index + 1]])
        for index in range(3)
    )
    groups = tuple(
        [
            row
            for row in rows
            if (int(row["ix"]), int(row["iy"])) in coordinates
        ]
        for coordinates in coordinate_groups
    )
    if any(not group for group in groups):
        raise ValueError("Triple-quantile joint group is empty")
    total_columns = float(len(rows))
    total_column_strength = sum(
        float(row["column_moment_knm"]) for row in rows
    )
    total_beam_count = sum(int(row["adjacent_beam_count"]) for row in rows)
    total_beam_pos = sum(
        float(row["limited_beam_moment_pos_knm"]) for row in rows
    )
    total_beam_neg = sum(
        float(row["limited_beam_moment_neg_knm"]) for row in rows
    )
    output = tuple(
        {
            "column_stiffness": len(group) / total_columns,
            "column_strength": sum(
                float(row["column_moment_knm"]) for row in group
            )
            / max(total_column_strength, 1.0e-12),
            "beam_stiffness": sum(
                int(row["adjacent_beam_count"]) for row in group
            )
            / max(total_beam_count, 1),
            "beam_strength_pos": sum(
                float(row["limited_beam_moment_pos_knm"])
                for row in group
            )
            / max(total_beam_pos, 1.0e-12),
            "beam_strength_neg": sum(
                float(row["limited_beam_moment_neg_knm"])
                for row in group
            )
            / max(total_beam_neg, 1.0e-12),
        }
        for group in groups
    )
    for name in output[0]:
        if not math.isclose(
            sum(item[name] for item in output), 1.0, abs_tol=1.0e-10
        ):
            raise AssertionError("Triple-quantile fractions do not close")
    return output


def _frame_line_substructure_fractions(
    artifact: FishboneArtifact,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], ...]:
    """Aggregate source joints by their physical planar frame line."""

    evidence = artifact.calibration_metrics.get(
        "joint_mechanism_evidence", {}
    )
    direction_rows = list(evidence.get(direction, ()))
    rows = (
        list(direction_rows[story_index].get("joint_rows", ()))
        if story_index < len(direction_rows)
        else []
    )
    if not rows:
        return _joint_ensemble_substructure_fractions(
            artifact, direction, story_index
        )
    # An X frame extends along X and is indexed by its transverse Y line;
    # conversely a Y frame is indexed by its transverse X line.
    transverse_key = "iy" if direction == "X" else "ix"
    line_ids = sorted({int(row[transverse_key]) for row in rows})
    groups = tuple(
        [row for row in rows if int(row[transverse_key]) == line]
        for line in line_ids
    )
    total_columns = float(len(rows))
    total_column_strength = sum(
        float(row["column_moment_knm"]) for row in rows
    )
    total_beam_count = sum(int(row["adjacent_beam_count"]) for row in rows)
    total_beam_pos = sum(
        float(row["limited_beam_moment_pos_knm"]) for row in rows
    )
    total_beam_neg = sum(
        float(row["limited_beam_moment_neg_knm"]) for row in rows
    )
    output = tuple(
        {
            "column_stiffness": len(group) / total_columns,
            "column_strength": sum(
                float(row["column_moment_knm"]) for row in group
            )
            / max(total_column_strength, 1.0e-12),
            "beam_stiffness": sum(
                int(row["adjacent_beam_count"]) for row in group
            )
            / max(total_beam_count, 1),
            "beam_strength_pos": sum(
                float(row["limited_beam_moment_pos_knm"])
                for row in group
            )
            / max(total_beam_pos, 1.0e-12),
            "beam_strength_neg": sum(
                float(row["limited_beam_moment_neg_knm"])
                for row in group
            )
            / max(total_beam_neg, 1.0e-12),
        }
        for group in groups
    )
    for name in output[0]:
        if not math.isclose(
            sum(item[name] for item in output), 1.0, abs_tol=1.0e-10
        ):
            raise AssertionError("Frame-line fractions do not close")
    return output


def _substructure_fractions(
    artifact: FishboneArtifact,
    topology_model: str,
    direction: str,
    story_index: int,
) -> tuple[dict[str, float], ...]:
    if topology_model == "dual_quantile_substructure":
        return _dual_substructure_fractions(
            artifact, direction, story_index
        )
    if topology_model == "dual_height_stable_substructure":
        return _dual_height_stable_substructure_fractions(
            artifact, direction, story_index
        )
    if topology_model == "dual_edge_chord_substructure":
        return _dual_edge_chord_substructure_fractions(
            artifact, direction, story_index
        )
    if topology_model == "joint_ensemble_substructure":
        return _joint_ensemble_substructure_fractions(
            artifact, direction, story_index
        )
    if topology_model == "triple_quantile_substructure":
        return _triple_quantile_substructure_fractions(
            artifact, direction, story_index
        )
    if topology_model == "frame_line_substructure":
        return _frame_line_substructure_fractions(
            artifact, direction, story_index
        )
    raise ValueError(f"Unsupported substructure topology: {topology_model}")


def _substructure_count(
    artifact: FishboneArtifact,
    topology_model: str,
) -> int:
    counts = {
        len(_substructure_fractions(artifact, topology_model, direction, story))
        for direction in ("X", "Y")
        for story in range(artifact.story_count)
    }
    if len(counts) != 1:
        raise ValueError(
            "Source joint count must remain constant through the building"
        )
    return counts.pop()


def _build_substructure_model(
    artifact: FishboneArtifact,
    topology_model: str,
) -> FishboneModelInfo:
    """Build coincident IFB spines with independent joint rotations."""

    ops = _ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)
    ops.geomTransf("Linear", 3, 0.0, 0.0, 1.0)

    master_base = 100
    ops.node(master_base, 0.0, 0.0, 0.0)
    ops.fix(master_base, 1, 1, 1, 1, 1, 1)
    floor_nodes = []
    for floor, elevation in enumerate(artifact.floor_elevations_m, 1):
        tag = 100 + floor
        floor_nodes.append(tag)
        ops.node(tag, 0.0, 0.0, float(elevation))
        # Mass masters carry translation only.  Each coincident spine owns
        # independent RX/RY joint rotations.
        ops.fix(tag, 0, 0, 1, 1, 1, 1)
        ops.mass(
            tag,
            artifact.floor_masses_x_kn_s2_m[floor - 1],
            artifact.floor_masses_y_kn_s2_m[floor - 1],
            1.0e-12,
            1.0e-12,
            1.0e-12,
            1.0e-12,
        )

    column_elements: list[int] = []
    beam_elements_x_by_story: list[list[int]] = [
        [] for _ in artifact.stories
    ]
    beam_elements_y_by_story: list[list[int]] = [
        [] for _ in artifact.stories
    ]
    column_hinges: list[int] = []
    beam_hinges: list[int] = []
    spine_joints_by_spine: list[tuple[int, ...]] = []
    hinge_options = dict(
        artifact.calibration_metrics.get("fishbone_v3_hysteresis", {})
    )

    # Fishbone-v4 common rocking coordinate.  Unlike the rejected direct
    # axial-chord prototype below, this coordinate is not tied to each local
    # spine rotation.  Beam-arm compatibility couples the spine joints to one
    # floor rocking rotation, and the condensed axial chords act only between
    # consecutive common rotations.
    common_axial_ratio = float(
        hinge_options.get("common_axial_chord_rocking_ratio", 0.0)
    )
    common_separation_ratio = float(
        hinge_options.get("common_axial_chord_separation_ratio", 1.0)
    )
    if not 0.0 <= common_axial_ratio <= 3.0:
        raise ValueError("Common axial-chord rocking ratio is invalid")
    if not 0.10 <= common_separation_ratio <= 1.50:
        raise ValueError("Common axial-chord separation ratio is invalid")
    common_rocking_nodes: list[int] = []
    common_axial_chord_elements: list[int] = []
    if common_axial_ratio > 0.0:
        rocking_base = 19_000_000
        ops.node(rocking_base, 0.0, 0.0, 0.0)
        ops.fix(rocking_base, 1, 1, 1, 1, 1, 1)
        prior_rocking = rocking_base
        for floor, story in enumerate(artifact.stories, 1):
            rocking = rocking_base + floor
            common_rocking_nodes.append(rocking)
            ops.node(rocking, 0.0, 0.0, 0.0)
            ops.fix(rocking, 1, 1, 1, 0, 0, 1)
            stiffness_x = (
                common_axial_ratio
                * artifact.elastic_modulus_kn_m2
                * story.column_count
                * story.column_area_m2
                * (common_separation_ratio * artifact.plan_length_x_m) ** 2
                / (4.0 * story.height_m)
            )
            stiffness_y = (
                common_axial_ratio
                * artifact.elastic_modulus_kn_m2
                * story.column_count
                * story.column_area_m2
                * (common_separation_ratio * artifact.plan_length_y_m) ** 2
                / (4.0 * story.height_m)
            )
            material_x = 19_100_000 + floor * 10 + 1
            material_y = material_x + 1
            ops.uniaxialMaterial("Elastic", material_x, stiffness_x)
            ops.uniaxialMaterial("Elastic", material_y, stiffness_y)
            element = 19_500_000 + floor
            ops.element(
                "zeroLength",
                element,
                prior_rocking,
                rocking,
                "-mat",
                material_y,
                material_x,
                "-dir",
                4,
                5,
            )
            common_axial_chord_elements.append(element)
            prior_rocking = rocking

    spine_count = _substructure_count(artifact, topology_model)
    for spine in range(spine_count):
        offset = 200_000 * (spine + 1)
        spine_base = offset + 100
        ops.node(spine_base, 0.0, 0.0, 0.0)
        ops.fix(spine_base, 1, 1, 1, 1, 1, 1)
        spine_joints = []
        for floor, elevation in enumerate(artifact.floor_elevations_m, 1):
            joint = offset + 100 + floor
            spine_joints.append(joint)
            ops.node(joint, 0.0, 0.0, float(elevation))
            ops.fix(joint, 0, 0, 1, 0, 0, 1)
            ops.equalDOF(floor_nodes[floor - 1], joint, 1, 2)
        spine_joints_by_spine.append(tuple(spine_joints))

        for floor, story in enumerate(artifact.stories, 1):
            x_fraction = _substructure_fractions(
                artifact, topology_model, "X", floor - 1
            )[spine]
            y_fraction = _substructure_fractions(
                artifact, topology_model, "Y", floor - 1
            )[spine]
            lower_joint = spine_base if floor == 1 else spine_joints[floor - 2]
            upper_joint = spine_joints[floor - 1]
            lower_master = master_base if floor == 1 else floor_nodes[floor - 2]
            upper_master = floor_nodes[floor - 1]
            z_lower = 0.0 if floor == 1 else artifact.floor_elevations_m[floor - 2]
            z_upper = artifact.floor_elevations_m[floor - 1]
            lower_member = offset + 10000 + floor * 10 + 1
            upper_member = offset + 10000 + floor * 10 + 2
            for node, master, z in (
                (lower_member, lower_master, z_lower),
                (upper_member, upper_master, z_upper),
            ):
                ops.node(node, 0.0, 0.0, float(z))
                ops.fix(node, 0, 0, 1, 0, 0, 1)
                ops.equalDOF(master, node, 1, 2)

            lower_mat_x = offset + 20000 + floor * 20 + 1
            lower_mat_y = offset + 20000 + floor * 20 + 2
            upper_mat_x = offset + 20000 + floor * 20 + 3
            upper_mat_y = offset + 20000 + floor * 20 + 4
            for tag, hinge, fraction, direction in (
                (lower_mat_x, story.column_bottom_hinge_x, x_fraction, "X"),
                (lower_mat_y, story.column_bottom_hinge_y, y_fraction, "Y"),
                (upper_mat_x, story.column_top_hinge_x, x_fraction, "X"),
                (upper_mat_y, story.column_top_hinge_y, y_fraction, "Y"),
            ):
                _define_hinge(
                    tag,
                    _fractional_hinge(
                        hinge,
                        stiffness_fraction=fraction["column_stiffness"],
                        strength_fraction_pos=fraction.get(
                            "column_strength_pos",
                            fraction["column_strength"],
                        ),
                        strength_fraction_neg=fraction.get(
                            "column_strength_neg",
                            fraction["column_strength"],
                        ),
                        source_suffix=(
                            f"{topology_model}-spine-{spine + 1}-{direction}"
                        ),
                    ),
                    hinge_options,
                    story_index=floor - 1,
                    story_count=artifact.story_count,
                    mechanism="column",
                    direction=direction,
                )
            lower_hinge = offset + 30000 + floor * 20 + 1
            upper_hinge = offset + 30000 + floor * 20 + 2
            ops.element(
                "zeroLength", lower_hinge, lower_joint, lower_member,
                "-mat", lower_mat_y, lower_mat_x, "-dir", 4, 5,
            )
            ops.element(
                "zeroLength", upper_hinge, upper_member, upper_joint,
                "-mat", upper_mat_y, upper_mat_x, "-dir", 4, 5,
            )
            column_hinges.extend((lower_hinge, upper_hinge))
            column_element = offset + 40000 + floor
            column_elements.append(column_element)
            area_fraction = 0.5 * (
                x_fraction["column_stiffness"]
                + y_fraction["column_stiffness"]
            )
            ops.element(
                "elasticBeamColumn",
                column_element,
                lower_member,
                upper_member,
                story.column_area_m2 * area_fraction,
                artifact.elastic_modulus_kn_m2,
                artifact.shear_modulus_kn_m2,
                story.column_torsion_m4 * area_fraction,
                story.column_inertia_x_m4
                * story.column_stiffness_modifier_x
                * y_fraction["column_stiffness"],
                story.column_inertia_y_m4
                * story.column_stiffness_modifier_y
                * x_fraction["column_stiffness"],
                1,
            )

            for arm, (axis, sign) in enumerate(
                (("X", -1.0), ("X", 1.0), ("Y", -1.0), ("Y", 1.0)), 1
            ):
                fraction = x_fraction if axis == "X" else y_fraction
                central = offset + 50000 + floor * 10 + arm
                tip = offset + 60000 + floor * 10 + arm
                length = (
                    story.beam_half_length_x_m
                    if axis == "X"
                    else story.beam_half_length_y_m
                )
                coordinates = (
                    (0.0, 0.0, z_upper),
                    (
                        (sign * length, 0.0, z_upper)
                        if axis == "X"
                        else (0.0, sign * length, z_upper)
                    ),
                )
                for node, coordinate in zip((central, tip), coordinates, strict=True):
                    ops.node(node, *coordinate)
                    if node == central:
                        ops.fix(
                            node,
                            0,
                            0,
                            1,
                            int(axis == "X"),
                            int(axis == "Y"),
                            1,
                        )
                    elif axis == "X":
                        ops.fix(node, 0, 0, 1, 1, 0, 1)
                    else:
                        ops.fix(node, 0, 0, 1, 0, 1, 1)
                    ops.equalDOF(upper_master, node, 1, 2)
                    if (
                        common_axial_ratio > 0.0
                        and node == tip
                    ):
                        ops.equalDOF(
                            common_rocking_nodes[floor - 1],
                            node,
                            5 if axis == "X" else 4,
                        )
                source_hinge = (
                    story.beam_hinge_x if axis == "X" else story.beam_hinge_y
                )
                hinge_material = offset + 70000 + floor * 20 + arm
                _define_hinge(
                    hinge_material,
                    _fractional_hinge(
                        source_hinge,
                        stiffness_fraction=fraction["beam_stiffness"],
                        strength_fraction_pos=fraction["beam_strength_pos"],
                        strength_fraction_neg=fraction["beam_strength_neg"],
                        source_suffix=(
                            f"{topology_model}-spine-{spine + 1}-{axis}"
                        ),
                    ),
                    hinge_options,
                    story_index=floor - 1,
                    story_count=artifact.story_count,
                    mechanism="beam",
                    direction=axis,
                )
                hinge_element = offset + 80000 + floor * 20 + arm
                ops.element(
                    "zeroLength", hinge_element, upper_joint, central,
                    "-mat", hinge_material, "-dir", 5 if axis == "X" else 4,
                )
                beam_element = offset + 90000 + floor * 20 + arm
                inertia = (
                    story.beam_inertia_x_m4
                    if axis == "X"
                    else story.beam_inertia_y_m4
                )
                modifier = (
                    story.beam_stiffness_modifier_x
                    if axis == "X"
                    else story.beam_stiffness_modifier_y
                )
                area = (
                    story.beam_area_x_m2
                    if axis == "X"
                    else story.beam_area_y_m2
                )
                torsion = (
                    story.beam_torsion_x_m4
                    if axis == "X"
                    else story.beam_torsion_y_m4
                )
                ops.element(
                    "elasticBeamColumn", beam_element, central, tip,
                    area * fraction["beam_stiffness"],
                    artifact.elastic_modulus_kn_m2,
                    artifact.shear_modulus_kn_m2,
                    torsion * fraction["beam_stiffness"],
                    inertia * modifier * fraction["beam_stiffness"],
                    max(inertia * fraction["beam_stiffness"] * 1.0e-6, 1.0e-12),
                    2 if axis == "X" else 3,
                )
                if axis == "X":
                    beam_elements_x_by_story[floor - 1].append(beam_element)
                else:
                    beam_elements_y_by_story[floor - 1].append(beam_element)
                beam_hinges.append(hinge_element)

    coupling_ratio = float(
        hinge_options.get("inter_spine_rotation_coupling_ratio", 0.0)
    )
    if not 0.0 <= coupling_ratio <= 3.0:
        raise ValueError("Inter-spine rotation coupling ratio is invalid")
    if coupling_ratio > 0.0 and spine_count > 1:
        # The coincident reduced spines share diaphragm translations but own
        # independent joint rotations.  Source transverse framing supplies a
        # finite compatibility moment between those rotations; represent it
        # with one population-level elastic ratio rather than a fitted spring
        # at every storey or building.  The reference stiffness is the local
        # aggregate joint rotational stiffness from the column ends and two
        # Fishbone beam arms in the corresponding direction.
        for spine in range(1, spine_count):
            for floor, story in enumerate(artifact.stories):
                above = (
                    artifact.stories[floor + 1]
                    if floor + 1 < artifact.story_count
                    else None
                )
                reference_x = (
                    story.column_top_hinge_x.initial_stiffness_knm_rad
                    + 2.0 * story.beam_hinge_x.initial_stiffness_knm_rad
                    + (
                        above.column_bottom_hinge_x.initial_stiffness_knm_rad
                        if above is not None
                        else 0.0
                    )
                )
                reference_y = (
                    story.column_top_hinge_y.initial_stiffness_knm_rad
                    + 2.0 * story.beam_hinge_y.initial_stiffness_knm_rad
                    + (
                        above.column_bottom_hinge_y.initial_stiffness_knm_rad
                        if above is not None
                        else 0.0
                    )
                )
                mat_x = 9_000_000 + spine * 10_000 + floor * 10 + 1
                mat_y = mat_x + 1
                ops.uniaxialMaterial(
                    "Elastic", mat_x, coupling_ratio * reference_x
                )
                ops.uniaxialMaterial(
                    "Elastic", mat_y, coupling_ratio * reference_y
                )
                element = 9_100_000 + spine * 10_000 + floor
                ops.element(
                    "zeroLength",
                    element,
                    spine_joints_by_spine[0][floor],
                    spine_joints_by_spine[spine][floor],
                    "-mat",
                    mat_y,
                    mat_x,
                    "-dir",
                    4,
                    5,
                )

    # Fishbone-v4 diagnostic option: the earlier coincident-spine topology
    # fixes vertical translation at every reduced joint, so column-line axial
    # extension and the resulting overturning couple are identically zero.
    # Statically condense two separated elastic axial chords into a rotational
    # spring between consecutive floors.  For total chord area A and spacing
    # L, the exact small-deformation pair stiffness is EA*L^2/(4h).  Auxiliary
    # coincident nodes use identity equalDOF constraints, retaining the Plain
    # handler that is required by the established Fishbone topology.
    # The option defaults to zero and therefore leaves every saved v3 artifact
    # and cache bit-for-bit unchanged.
    axial_chord_ratio = float(
        hinge_options.get("axial_chord_rocking_ratio", 0.0)
    )
    separation_ratio = float(
        hinge_options.get("axial_chord_separation_ratio", 1.0)
    )
    if not 0.0 <= axial_chord_ratio <= 3.0:
        raise ValueError("Axial-chord rocking ratio is invalid")
    if not 0.10 <= separation_ratio <= 1.50:
        raise ValueError("Axial-chord separation ratio is invalid")
    axial_chord_elements: list[int] = []
    if axial_chord_ratio > 0.0:
        for spine, spine_joints in enumerate(spine_joints_by_spine):
            auxiliary_base = 20_000_000 + spine * 100_000
            ops.node(auxiliary_base, 0.0, 0.0, 0.0)
            ops.fix(auxiliary_base, 1, 1, 1, 1, 1, 1)
            prior = auxiliary_base
            for floor, (story, joint) in enumerate(
                zip(artifact.stories, spine_joints, strict=True), 1
            ):
                auxiliary = auxiliary_base + floor
                ops.node(auxiliary, 0.0, 0.0, 0.0)
                ops.fix(auxiliary, 1, 1, 1, 0, 0, 1)
                ops.equalDOF(joint, auxiliary, 4, 5)
                x_fraction = _substructure_fractions(
                    artifact, topology_model, "X", floor - 1
                )[spine]
                y_fraction = _substructure_fractions(
                    artifact, topology_model, "Y", floor - 1
                )[spine]
                separation_x = (
                    separation_ratio * artifact.plan_length_x_m
                )
                separation_y = (
                    separation_ratio * artifact.plan_length_y_m
                )
                stiffness_x = (
                    axial_chord_ratio
                    * artifact.elastic_modulus_kn_m2
                    * story.column_count
                    * story.column_area_m2
                    * x_fraction["column_stiffness"]
                    * separation_x**2
                    / (4.0 * story.height_m)
                )
                stiffness_y = (
                    axial_chord_ratio
                    * artifact.elastic_modulus_kn_m2
                    * story.column_count
                    * story.column_area_m2
                    * y_fraction["column_stiffness"]
                    * separation_y**2
                    / (4.0 * story.height_m)
                )
                material_x = 21_000_000 + spine * 100_000 + floor * 10 + 1
                material_y = material_x + 1
                ops.uniaxialMaterial("Elastic", material_x, stiffness_x)
                ops.uniaxialMaterial("Elastic", material_y, stiffness_y)
                element = 21_500_000 + spine * 100_000 + floor
                ops.element(
                    "zeroLength",
                    element,
                    prior,
                    auxiliary,
                    "-mat",
                    material_y,
                    material_x,
                    "-dir",
                    4,
                    5,
                )
                axial_chord_elements.append(element)
                prior = auxiliary

    # Keep the one common bidirectional P-Delta potential used by the
    # single-spine model; gravity is not duplicated by the two flexural spines.
    pdelta_elements = []
    pdelta_base = 110000
    ops.node(pdelta_base, 0.0, 0.0, 0.0)
    ops.fix(pdelta_base, 1, 1, 1, 1, 1, 1)
    prior = pdelta_base
    for floor, story in enumerate(artifact.stories, 1):
        node = pdelta_base + floor
        ops.node(node, 0.0, 0.0, 0.0)
        ops.fix(node, 0, 0, 1, 1, 1, 1)
        ops.equalDOF(floor_nodes[floor - 1], node, 1, 2)
        mat_x = 120000 + floor * 2
        mat_y = mat_x + 1
        ops.uniaxialMaterial("Elastic", mat_x, story.geometric_stiffness_kn_m)
        ops.uniaxialMaterial("Elastic", mat_y, story.geometric_stiffness_kn_m)
        element = 130000 + floor
        ops.element(
            "zeroLength", element, prior, node,
            "-mat", mat_x, mat_y, "-dir", 1, 2,
        )
        pdelta_elements.append(element)
        prior = node

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for floor, story in enumerate(artifact.stories):
        x_fraction_rows = _substructure_fractions(
            artifact, topology_model, "X", floor
        )
        y_fraction_rows = _substructure_fractions(
            artifact, topology_model, "Y", floor
        )
        for spine in range(spine_count):
            tags = beam_elements_x_by_story[floor][
                2 * spine : 2 * (spine + 1)
            ]
            for tag in tags:
                ops.eleLoad(
                    "-ele", tag, "-type", "-beamUniform", 0.0,
                    -story.beam_uniform_gravity_x_kn_m
                    * x_fraction_rows[spine]["beam_stiffness"],
                    0.0,
                )
        for spine in range(spine_count):
            tags = beam_elements_y_by_story[floor][
                2 * spine : 2 * (spine + 1)
            ]
            for tag in tags:
                ops.eleLoad(
                    "-ele", tag, "-type", "-beamUniform", 0.0,
                    -story.beam_uniform_gravity_y_kn_m
                    * y_fraction_rows[spine]["beam_stiffness"],
                    0.0,
                )
    gravity_applied = sum(
        2.0
        * (
            story.beam_uniform_gravity_x_kn_m * story.beam_half_length_x_m
            + story.beam_uniform_gravity_y_kn_m * story.beam_half_length_y_m
        )
        for story in artifact.stories
    )
    return FishboneModelInfo(
        base_node=master_base,
        pdelta_base_node=pdelta_base,
        floor_nodes=tuple(floor_nodes),
        column_elements=tuple(column_elements),
        beam_elements_x=tuple(tag for rows in beam_elements_x_by_story for tag in rows),
        beam_elements_y=tuple(tag for rows in beam_elements_y_by_story for tag in rows),
        column_hinge_elements=tuple(column_hinges),
        beam_hinge_elements=tuple(beam_hinges),
        pdelta_elements=tuple(pdelta_elements),
        floor_elevations_m=artifact.floor_elevations_m,
        story_heights_m=artifact.story_heights_m,
        gravity_applied_kn=float(gravity_applied),
        axial_chord_elements=tuple(
            common_axial_chord_elements + axial_chord_elements
        ),
    )


def build_model(
    artifact: FishboneArtifact,
) -> FishboneModelInfo:
    """Build the 3D, simultaneous-X/Y IFB3-inspired OpenSees model."""

    artifact.validate()
    hinge_options = dict(
        artifact.calibration_metrics.get("fishbone_v3_hysteresis", {})
    )
    topology_model = str(
        hinge_options.get("topology_model", "single_spine")
    )
    if topology_model in {
        "dual_quantile_substructure",
        "dual_height_stable_substructure",
        "dual_edge_chord_substructure",
        "triple_quantile_substructure",
        "frame_line_substructure",
        "joint_ensemble_substructure",
    }:
        return _build_substructure_model(artifact, topology_model)
    if topology_model != "single_spine":
        raise ValueError(f"Unsupported Fishbone topology: {topology_model}")
    ops = _ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    # Local member axes: vertical column, X beam and Y beam.
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)
    ops.geomTransf("Linear", 3, 0.0, 0.0, 1.0)

    base_joint = 100
    ops.node(base_joint, 0.0, 0.0, 0.0)
    ops.fix(base_joint, 1, 1, 1, 1, 1, 1)
    floor_nodes: list[int] = []
    for floor, elevation in enumerate(artifact.floor_elevations_m, 1):
        tag = 100 + floor
        floor_nodes.append(tag)
        ops.node(tag, 0.0, 0.0, float(elevation))
        # The current certification scope is plan-symmetric translation.
        # UZ and floor torsion are restrained; RX/RY remain as member-joint
        # rotation DOFs. Polar mass is retained in the artifact for IFB3-T.
        ops.fix(tag, 0, 0, 1, 0, 0, 1)
        ops.mass(
            tag,
            artifact.floor_masses_x_kn_s2_m[floor - 1],
            artifact.floor_masses_y_kn_s2_m[floor - 1],
            1.0e-12,
            1.0e-12,
            1.0e-12,
            1.0e-12,
        )

    column_elements: list[int] = []
    beam_elements_x: list[int] = []
    beam_elements_y: list[int] = []
    column_hinges: list[int] = []
    beam_hinges: list[int] = []
    pdelta_elements: list[int] = []

    for floor, story in enumerate(artifact.stories, 1):
        lower_joint = base_joint if floor == 1 else floor_nodes[floor - 2]
        upper_joint = floor_nodes[floor - 1]
        z_lower = 0.0 if floor == 1 else artifact.floor_elevations_m[floor - 2]
        z_upper = artifact.floor_elevations_m[floor - 1]
        lower_member = 10000 + floor * 10 + 1
        upper_member = 10000 + floor * 10 + 2
        ops.node(lower_member, 0.0, 0.0, float(z_lower))
        ops.node(upper_member, 0.0, 0.0, float(z_upper))
        ops.equalDOF(lower_joint, lower_member, 1, 2, 3, 6)
        ops.equalDOF(upper_joint, upper_member, 1, 2, 3, 6)

        lower_mat_x = 20000 + floor * 20 + 1
        lower_mat_y = 20000 + floor * 20 + 2
        upper_mat_x = 20000 + floor * 20 + 3
        upper_mat_y = 20000 + floor * 20 + 4
        # Rotation about global Y controls X response and vice versa.
        _define_hinge(
            lower_mat_x,
            story.column_bottom_hinge_x,
            hinge_options,
            story_index=floor - 1,
            story_count=artifact.story_count,
            mechanism="column",
            direction="X",
        )
        _define_hinge(
            lower_mat_y,
            story.column_bottom_hinge_y,
            hinge_options,
            story_index=floor - 1,
            story_count=artifact.story_count,
            mechanism="column",
            direction="Y",
        )
        _define_hinge(
            upper_mat_x,
            story.column_top_hinge_x,
            hinge_options,
            story_index=floor - 1,
            story_count=artifact.story_count,
            mechanism="column",
            direction="X",
        )
        _define_hinge(
            upper_mat_y,
            story.column_top_hinge_y,
            hinge_options,
            story_index=floor - 1,
            story_count=artifact.story_count,
            mechanism="column",
            direction="Y",
        )
        lower_hinge = 30000 + floor * 20 + 1
        upper_hinge = 30000 + floor * 20 + 2
        ops.element(
            "zeroLength",
            lower_hinge,
            lower_joint,
            lower_member,
            "-mat",
            lower_mat_y,
            lower_mat_x,
            "-dir",
            4,
            5,
        )
        ops.element(
            "zeroLength",
            upper_hinge,
            upper_member,
            upper_joint,
            "-mat",
            upper_mat_y,
            upper_mat_x,
            "-dir",
            4,
            5,
        )
        column_hinges.extend((lower_hinge, upper_hinge))
        column_element = 40000 + floor
        column_elements.append(column_element)
        ops.element(
            "elasticBeamColumn",
            column_element,
            lower_member,
            upper_member,
            story.column_area_m2,
            artifact.elastic_modulus_kn_m2,
            artifact.shear_modulus_kn_m2,
            story.column_torsion_m4,
            story.column_inertia_x_m4
            * story.column_stiffness_modifier_x,
            story.column_inertia_y_m4
            * story.column_stiffness_modifier_y,
            1,
        )

        for arm, (axis, sign) in enumerate(
            (("X", -1.0), ("X", 1.0), ("Y", -1.0), ("Y", 1.0)),
            1,
        ):
            central = 50000 + floor * 10 + arm
            tip = 60000 + floor * 10 + arm
            if axis == "X":
                length = story.beam_half_length_x_m
                ops.node(central, 0.0, 0.0, float(z_upper))
                ops.node(tip, sign * length, 0.0, float(z_upper))
                ops.equalDOF(upper_joint, central, 1, 2, 3, 4, 6)
                ops.fix(tip, 0, 0, 1, 1, 0, 1)
                ops.equalDOF(upper_joint, tip, 1, 2)
                hinge_material = 70000 + floor * 20 + arm
                _define_hinge(
                    hinge_material,
                    story.beam_hinge_x,
                    hinge_options,
                    story_index=floor - 1,
                    story_count=artifact.story_count,
                    mechanism="beam",
                    direction="X",
                )
                hinge_element = 80000 + floor * 20 + arm
                ops.element(
                    "zeroLength",
                    hinge_element,
                    upper_joint,
                    central,
                    "-mat",
                    hinge_material,
                    "-dir",
                    5,
                )
                beam_element = 90000 + floor * 20 + arm
                ops.element(
                    "elasticBeamColumn",
                    beam_element,
                    central,
                    tip,
                    story.beam_area_x_m2,
                    artifact.elastic_modulus_kn_m2,
                    artifact.shear_modulus_kn_m2,
                    story.beam_torsion_x_m4,
                    story.beam_inertia_x_m4
                    * story.beam_stiffness_modifier_x,
                    max(story.beam_inertia_x_m4 * 1.0e-6, 1.0e-12),
                    2,
                )
                beam_elements_x.append(beam_element)
            else:
                length = story.beam_half_length_y_m
                ops.node(central, 0.0, 0.0, float(z_upper))
                ops.node(tip, 0.0, sign * length, float(z_upper))
                ops.equalDOF(upper_joint, central, 1, 2, 3, 5, 6)
                ops.fix(tip, 0, 0, 1, 0, 1, 1)
                ops.equalDOF(upper_joint, tip, 1, 2)
                hinge_material = 70000 + floor * 20 + arm
                _define_hinge(
                    hinge_material,
                    story.beam_hinge_y,
                    hinge_options,
                    story_index=floor - 1,
                    story_count=artifact.story_count,
                    mechanism="beam",
                    direction="Y",
                )
                hinge_element = 80000 + floor * 20 + arm
                ops.element(
                    "zeroLength",
                    hinge_element,
                    upper_joint,
                    central,
                    "-mat",
                    hinge_material,
                    "-dir",
                    4,
                )
                beam_element = 90000 + floor * 20 + arm
                ops.element(
                    "elasticBeamColumn",
                    beam_element,
                    central,
                    tip,
                    story.beam_area_y_m2,
                    artifact.elastic_modulus_kn_m2,
                    artifact.shear_modulus_kn_m2,
                    story.beam_torsion_y_m4,
                    story.beam_inertia_y_m4
                    * story.beam_stiffness_modifier_y,
                    max(story.beam_inertia_y_m4 * 1.0e-6, 1.0e-12),
                    3,
                )
                beam_elements_y.append(beam_element)
            beam_hinges.append(hinge_element)

    # Common bidirectional P-Delta potential.  The coincident auxiliary chain
    # is mathematically equivalent to a pin-ended leaning column at first
    # order and avoids applying the building gravity load independently in X
    # and Y.
    pdelta_base = 110000
    ops.node(pdelta_base, 0.0, 0.0, 0.0)
    ops.fix(pdelta_base, 1, 1, 1, 1, 1, 1)
    prior = pdelta_base
    for floor, story in enumerate(artifact.stories, 1):
        node = pdelta_base + floor
        ops.node(node, 0.0, 0.0, 0.0)
        ops.fix(node, 0, 0, 1, 1, 1, 1)
        ops.equalDOF(floor_nodes[floor - 1], node, 1, 2)
        mat_x = 120000 + floor * 2
        mat_y = mat_x + 1
        ops.uniaxialMaterial(
            "Elastic", mat_x, story.geometric_stiffness_kn_m
        )
        ops.uniaxialMaterial(
            "Elastic", mat_y, story.geometric_stiffness_kn_m
        )
        element = 130000 + floor
        ops.element(
            "zeroLength",
            element,
            prior,
            node,
            "-mat",
            mat_x,
            mat_y,
            "-dir",
            1,
            2,
        )
        pdelta_elements.append(element)
        prior = node

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for floor, story in enumerate(artifact.stories, 1):
        x_tags = beam_elements_x[2 * (floor - 1) : 2 * floor]
        y_tags = beam_elements_y[2 * (floor - 1) : 2 * floor]
        for tag in x_tags:
            ops.eleLoad(
                "-ele",
                tag,
                "-type",
                "-beamUniform",
                0.0,
                -story.beam_uniform_gravity_x_kn_m,
                0.0,
            )
        for tag in y_tags:
            ops.eleLoad(
                "-ele",
                tag,
                "-type",
                "-beamUniform",
                0.0,
                -story.beam_uniform_gravity_y_kn_m,
                0.0,
            )
    gravity_applied = sum(
        2.0
        * (
            story.beam_uniform_gravity_x_kn_m
            * story.beam_half_length_x_m
            + story.beam_uniform_gravity_y_kn_m
            * story.beam_half_length_y_m
        )
        for story in artifact.stories
    )
    return FishboneModelInfo(
        base_node=base_joint,
        pdelta_base_node=pdelta_base,
        floor_nodes=tuple(floor_nodes),
        column_elements=tuple(column_elements),
        beam_elements_x=tuple(beam_elements_x),
        beam_elements_y=tuple(beam_elements_y),
        column_hinge_elements=tuple(column_hinges),
        beam_hinge_elements=tuple(beam_hinges),
        pdelta_elements=tuple(pdelta_elements),
        floor_elevations_m=artifact.floor_elevations_m,
        story_heights_m=artifact.story_heights_m,
        gravity_applied_kn=float(gravity_applied),
    )


def run_gravity() -> None:
    """Apply equivalent beam gravity moments before lateral analysis."""

    ops = _ops()
    ops.wipeAnalysis()
    # The Fishbone topology uses several equalDOF constraints whose retained
    # floor node is shared by the column, four beam arms and the common
    # P-Delta chain.  OpenSees' Transformation handler does not support this
    # retained-node topology reliably: it can leave the member-end
    # translations at zero and create a spurious lateral mechanism.  Plain is
    # exact for these identity equalDOF constraints and preserves the intended
    # member compatibility.
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-9, 50, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("Fishbone equivalent gravity analysis did not converge")
    ops.loadConst("-time", 0.0)


def modal_properties(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    *,
    mode_count: int | None = None,
) -> FishboneModalResult:
    """Return the first three translational modes in each direction."""

    ops = _ops()
    requested = int(mode_count or 2 * artifact.story_count)
    requested = max(2, requested)
    eigenvalues = np.asarray(
        ops.eigen("-fullGenLapack", requested), dtype=float
    )
    finite = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0.0)]
    if finite.size < 2:
        raise RuntimeError("Fishbone model lacks positive translational modes")
    periods = 2.0 * math.pi / np.sqrt(finite)
    masses_x = np.asarray(artifact.floor_masses_x_kn_s2_m, dtype=float)
    masses_y = np.asarray(artifact.floor_masses_y_kn_s2_m, dtype=float)
    total_x = float(np.sum(masses_x))
    total_y = float(np.sum(masses_y))
    rows: list[dict[str, Any]] = []
    for mode in range(1, finite.size + 1):
        phi_x = np.asarray(
            [
                float(ops.nodeEigenvector(node, mode, 1))
                for node in info.floor_nodes
            ],
            dtype=float,
        )
        phi_y = np.asarray(
            [
                float(ops.nodeEigenvector(node, mode, 2))
                for node in info.floor_nodes
            ],
            dtype=float,
        )
        modal_mass = float(
            phi_x @ (masses_x * phi_x)
            + phi_y @ (masses_y * phi_y)
        )
        gamma_x = (
            float(phi_x @ masses_x) / modal_mass
            if modal_mass > 0.0
            else 0.0
        )
        gamma_y = (
            float(phi_y @ masses_y) / modal_mass
            if modal_mass > 0.0
            else 0.0
        )
        ratio_x = (
            float(phi_x @ masses_x) ** 2 / (modal_mass * total_x)
            if modal_mass > 0.0 and total_x > 0.0
            else 0.0
        )
        ratio_y = (
            float(phi_y @ masses_y) ** 2 / (modal_mass * total_y)
            if modal_mass > 0.0 and total_y > 0.0
            else 0.0
        )
        rows.append(
            {
                "mode": mode,
                "period": float(periods[mode - 1]),
                "phi_x": phi_x,
                "phi_y": phi_y,
                "gamma_x": gamma_x,
                "gamma_y": gamma_y,
                "ratio_x": ratio_x,
                "ratio_y": ratio_y,
            }
        )

    def select(direction: str) -> list[dict[str, Any]]:
        ratio_key = f"ratio_{direction.lower()}"
        candidates = [
            row for row in rows if float(row[ratio_key]) > 1.0e-6
        ]
        candidates.sort(key=lambda row: -float(row["period"]))
        return candidates[: min(3, artifact.story_count)]

    x_rows = select("X")
    y_rows = select("Y")

    def normalized(row: dict[str, Any], direction: str) -> tuple[float, ...]:
        vector = np.asarray(row[f"phi_{direction.lower()}"], dtype=float)
        scale = float(np.max(np.abs(vector)))
        if scale <= 0.0:
            return tuple(0.0 for _ in vector)
        vector = vector / scale
        if vector[-1] < 0.0:
            vector *= -1.0
        return tuple(float(value) for value in vector)

    return FishboneModalResult(
        periods_s=tuple(float(value) for value in periods),
        global_mode_indices_x=tuple(int(row["mode"]) for row in x_rows),
        global_mode_indices_y=tuple(int(row["mode"]) for row in y_rows),
        directional_periods_x_s=tuple(
            float(row["period"]) for row in x_rows
        ),
        directional_periods_y_s=tuple(
            float(row["period"]) for row in y_rows
        ),
        normalized_mode_shapes_x=tuple(
            normalized(row, "X") for row in x_rows
        ),
        normalized_mode_shapes_y=tuple(
            normalized(row, "Y") for row in y_rows
        ),
        effective_mass_ratios_x=tuple(
            float(row["ratio_x"]) for row in x_rows
        ),
        effective_mass_ratios_y=tuple(
            float(row["ratio_y"]) for row in y_rows
        ),
        participation_factors_x=tuple(
            float(row["gamma_x"]) for row in x_rows
        ),
        participation_factors_y=tuple(
            float(row["gamma_y"]) for row in y_rows
        ),
    )
