"""Bidirectional nonlinear lumped-mass MDOF model and result artifacts.

The reduced model intentionally has only two translational degrees of freedom
per floor.  Directional story springs are uncoupled in v1.  The same gravity
load contributes a negative geometric stiffness in both horizontal
directions, which is the two-dimensional gradient of the common P-Delta
potential and therefore does not double-count gravity.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G_STD
from .ground_motion import load_acceleration
from .io_utils import atomic_write_json, read_json, stable_hash
from .structural import _ops

MDOF_ARTIFACT_SCHEMA_VERSION = "nonlinear-bidirectional-mdof-artifact-v1"
MDOF_ANALYSIS_SCHEMA_VERSION = (
    "nonlinear-bidirectional-mdof-nltha-v7-nonlinear-elastic-cracking"
)
MDOF_CERTIFICATION_SCHEMA_VERSION = "strict-mdof-validation-v1"
MDOF_SPO_FEATURE_SCHEMA_VERSION = (
    "three-mode-spo-features-v4-full3d-calibration-source-36"
)


@dataclass(frozen=True)
class StorySpringParameters:
    """One direction's aggregate material and geometric story response."""

    story: int
    height_m: float
    initial_stiffness_kn_m: float
    yield_force_kn: float
    yield_deformation_m: float
    capping_force_kn: float
    capping_deformation_m: float
    post_capping_slope_kn_m: float
    residual_force_kn: float
    ultimate_deformation_m: float
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
    gravity_load_above_kn: float = 0.0
    geometric_stiffness_kn_m: float = 0.0
    fit_nrmse: float = math.nan
    pattern_dispersion: float = math.nan
    backbone_censored: bool = False
    postpeak_pattern_count: int = 0
    observed_max_deformation_m: float = 0.0
    higher_mode_compatibility_nrmse: float = 0.0
    effective_higher_mode_weight: float = 0.0
    cracking_deformation_m: float = 0.0
    post_cracking_stiffness_kn_m: float = 0.0
    cracking_branch_force_kn: float = 0.0

    def validate(self) -> None:
        positive = {
            "height_m": self.height_m,
            "initial_stiffness_kn_m": self.initial_stiffness_kn_m,
            "yield_force_kn": self.yield_force_kn,
            "yield_deformation_m": self.yield_deformation_m,
            "capping_force_kn": self.capping_force_kn,
            "capping_deformation_m": self.capping_deformation_m,
            "residual_force_kn": self.residual_force_kn,
            "ultimate_deformation_m": self.ultimate_deformation_m,
            "lambda_s": self.lambda_s,
            "lambda_c": self.lambda_c,
            "lambda_a": self.lambda_a,
            "lambda_k": self.lambda_k,
            "c_s": self.c_s,
            "c_c": self.c_c,
            "c_a": self.c_a,
            "c_k": self.c_k,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                f"Story {self.story}: non-positive spring parameters: "
                + ", ".join(invalid)
            )
        if self.capping_deformation_m <= self.yield_deformation_m:
            raise ValueError(
                f"Story {self.story}: capping deformation must exceed yield"
            )
        if self.ultimate_deformation_m <= self.capping_deformation_m:
            raise ValueError(
                f"Story {self.story}: ultimate deformation must exceed capping"
            )
        if self.capping_force_kn < self.yield_force_kn:
            raise ValueError(
                f"Story {self.story}: capping force cannot be below yield"
            )
        if self.residual_force_kn >= self.capping_force_kn:
            raise ValueError(
                f"Story {self.story}: residual force must be below capping"
            )
        if self.residual_force_kn >= self.yield_force_kn:
            raise ValueError(
                f"Story {self.story}: IMK residual force must be below yield"
            )
        if self.geometric_stiffness_kn_m > 1.0e-12:
            raise ValueError(
                f"Story {self.story}: P-Delta stiffness must be non-positive"
            )
        if (
            self.initial_stiffness_kn_m + self.geometric_stiffness_kn_m
            <= 0.0
        ):
            raise ValueError(
                f"Story {self.story}: initial tangent is P-Delta unstable"
            )
        if self.postpeak_pattern_count < 0:
            raise ValueError(
                f"Story {self.story}: postpeak pattern count is negative"
            )
        if not math.isfinite(self.observed_max_deformation_m) or (
            self.observed_max_deformation_m < 0.0
        ):
            raise ValueError(
                f"Story {self.story}: observed deformation is invalid"
            )
        if not math.isfinite(self.higher_mode_compatibility_nrmse) or (
            self.higher_mode_compatibility_nrmse < 0.0
        ):
            raise ValueError(
                f"Story {self.story}: higher-mode compatibility is invalid"
            )
        if not 0.0 <= self.effective_higher_mode_weight <= 1.0:
            raise ValueError(
                f"Story {self.story}: effective higher-mode weight is invalid"
            )
        cracking_values = (
            self.cracking_deformation_m,
            self.post_cracking_stiffness_kn_m,
            self.cracking_branch_force_kn,
        )
        if any(not math.isfinite(float(value)) for value in cracking_values):
            raise ValueError(
                f"Story {self.story}: cracking parameters are invalid"
            )
        cracking_enabled = self.cracking_deformation_m > 0.0
        if cracking_enabled:
            if not (
                0.0
                < self.post_cracking_stiffness_kn_m
                < self.initial_stiffness_kn_m
            ):
                raise ValueError(
                    f"Story {self.story}: post-cracking stiffness is invalid"
                )
            if not (
                0.0
                < self.cracking_deformation_m
                < self.yield_deformation_m
                < self.capping_deformation_m
            ):
                raise ValueError(
                    f"Story {self.story}: cracking/yield/capping "
                    "deformations must increase"
                )
            expected = (
                self.initial_stiffness_kn_m
                - self.post_cracking_stiffness_kn_m
            ) * self.cracking_deformation_m
            if not math.isclose(
                self.cracking_branch_force_kn,
                expected,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    f"Story {self.story}: cracking branch force is "
                    "inconsistent"
                )
            if self.cracking_branch_force_kn >= self.residual_force_kn:
                raise ValueError(
                    f"Story {self.story}: cracking branch exceeds residual "
                    "strength"
                )
        elif any(abs(float(value)) > 1.0e-12 for value in cracking_values):
            raise ValueError(
                f"Story {self.story}: partial cracking parameters are invalid"
            )


@dataclass(frozen=True)
class DirectionalReduction:
    """Calibrated targets and story springs for one principal direction."""

    direction: str
    target_periods_s: tuple[float, ...]
    target_mode_shapes: tuple[tuple[float, ...], ...]
    target_effective_mass_ratios: tuple[float, ...]
    target_participation_factors: tuple[float, ...]
    story_stiffnesses_kn_m: tuple[float, ...]
    springs: tuple[StorySpringParameters, ...]
    calibration_patterns: tuple[str, ...]
    elastic_objective: float
    elastic_drift_nrmse: float
    first_mode_mac: float
    first_period_error: float
    nonlinear_backbone_nrmse: float
    elastic_core_drift_nrmse: float = 0.0
    elastic_higher_mode_drift_nrmse: float = 0.0
    elastic_fit_selection: str = "weighted_plan_objective"
    reduced_periods_s: tuple[float, ...] = ()
    mode_period_errors: tuple[float, ...] = ()
    mode_macs: tuple[float, ...] = ()
    raw_curve_path: str | None = None

    def validate(self, story_count: int) -> None:
        if self.direction not in {"X", "Y"}:
            raise ValueError("Directional reduction must be X or Y")
        if len(self.story_stiffnesses_kn_m) != story_count:
            raise ValueError(
                f"{self.direction}: story-stiffness count is inconsistent"
            )
        if len(self.springs) != story_count:
            raise ValueError(f"{self.direction}: story-spring count is inconsistent")
        if not self.target_periods_s or not self.target_mode_shapes:
            raise ValueError(f"{self.direction}: modal targets are missing")
        for spring in self.springs:
            spring.validate()


@dataclass(frozen=True)
class MDOFArtifact:
    """Content-addressed reduced-model definition for one designed building."""

    building_id: str
    source_model_hash: str
    source_signature: str
    story_count: int
    floor_elevations_m: tuple[float, ...]
    story_heights_m: tuple[float, ...]
    floor_masses_x_kn_s2_m: tuple[float, ...]
    floor_masses_y_kn_s2_m: tuple[float, ...]
    gravity_floor_loads_kn: tuple[float, ...]
    direction_x: DirectionalReduction
    direction_y: DirectionalReduction
    shear_failure_enabled: bool = False
    torsion_supported: bool = False
    xy_strength_interaction_supported: bool = False
    applicability_flags: tuple[str, ...] = ()
    calibration_metrics: dict[str, Any] = field(default_factory=dict)
    certified: bool = False
    certification_path: str | None = None
    schema_version: str = MDOF_ARTIFACT_SCHEMA_VERSION
    artifact_hash: str = ""

    def validate(self) -> None:
        if self.schema_version != MDOF_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MDOF artifact schema: {self.schema_version}"
            )
        if self.story_count < 1:
            raise ValueError("MDOF artifact must contain at least one story")
        vectors = (
            self.floor_elevations_m,
            self.story_heights_m,
            self.floor_masses_x_kn_s2_m,
            self.floor_masses_y_kn_s2_m,
            self.gravity_floor_loads_kn,
        )
        if any(len(values) != self.story_count for values in vectors):
            raise ValueError("MDOF floor vectors do not match the story count")
        if any(value <= 0.0 for value in self.story_heights_m):
            raise ValueError("MDOF story heights must be positive")
        if any(value <= 0.0 for value in self.floor_masses_x_kn_s2_m):
            raise ValueError("Every X floor mass must be positive")
        if any(value <= 0.0 for value in self.floor_masses_y_kn_s2_m):
            raise ValueError("Every Y floor mass must be positive")
        if any(value <= 0.0 for value in self.gravity_floor_loads_kn):
            raise ValueError("Every gravity floor load must be positive")
        if self.shear_failure_enabled:
            raise ValueError("Shear failure is unsupported by the v1 MDOF model")
        self.direction_x.validate(self.story_count)
        self.direction_y.validate(self.story_count)
        expected_hash = artifact_payload_hash(self)
        if self.artifact_hash and self.artifact_hash != expected_hash:
            raise ValueError("MDOF artifact hash does not match its contents")

    def with_hash(self) -> "MDOFArtifact":
        payload = asdict(self)
        normalized = dict(payload)
        normalized["artifact_hash"] = ""
        # Certification is an external QA decision about an immutable
        # structural artifact.  It must not change the model-content hash or
        # invalidate the NLTHA histories that supplied the certificate.
        normalized["certified"] = False
        normalized["certification_path"] = None
        return MDOFArtifact.from_dict(
            {**payload, "artifact_hash": stable_hash(normalized)}
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "MDOFArtifact":
        def direction(value: dict[str, Any]) -> DirectionalReduction:
            springs = tuple(
                StorySpringParameters(**spring) for spring in value["springs"]
            )
            return DirectionalReduction(
                direction=str(value["direction"]),
                target_periods_s=tuple(value["target_periods_s"]),
                target_mode_shapes=tuple(
                    tuple(row) for row in value["target_mode_shapes"]
                ),
                target_effective_mass_ratios=tuple(
                    value["target_effective_mass_ratios"]
                ),
                target_participation_factors=tuple(
                    value["target_participation_factors"]
                ),
                story_stiffnesses_kn_m=tuple(
                    value["story_stiffnesses_kn_m"]
                ),
                springs=springs,
                calibration_patterns=tuple(value["calibration_patterns"]),
                elastic_objective=float(value["elastic_objective"]),
                elastic_drift_nrmse=float(value["elastic_drift_nrmse"]),
                first_mode_mac=float(value["first_mode_mac"]),
                first_period_error=float(value["first_period_error"]),
                nonlinear_backbone_nrmse=float(
                    value["nonlinear_backbone_nrmse"]
                ),
                elastic_core_drift_nrmse=float(
                    value.get("elastic_core_drift_nrmse", 0.0)
                ),
                elastic_higher_mode_drift_nrmse=float(
                    value.get("elastic_higher_mode_drift_nrmse", 0.0)
                ),
                elastic_fit_selection=str(
                    value.get(
                        "elastic_fit_selection",
                        "weighted_plan_objective",
                    )
                ),
                reduced_periods_s=tuple(
                    value.get("reduced_periods_s", ())
                ),
                mode_period_errors=tuple(
                    value.get("mode_period_errors", ())
                ),
                mode_macs=tuple(value.get("mode_macs", ())),
                raw_curve_path=value.get("raw_curve_path"),
            )

        artifact = MDOFArtifact(
            building_id=str(payload["building_id"]),
            source_model_hash=str(payload["source_model_hash"]),
            source_signature=str(payload["source_signature"]),
            story_count=int(payload["story_count"]),
            floor_elevations_m=tuple(payload["floor_elevations_m"]),
            story_heights_m=tuple(payload["story_heights_m"]),
            floor_masses_x_kn_s2_m=tuple(
                payload["floor_masses_x_kn_s2_m"]
            ),
            floor_masses_y_kn_s2_m=tuple(
                payload["floor_masses_y_kn_s2_m"]
            ),
            gravity_floor_loads_kn=tuple(payload["gravity_floor_loads_kn"]),
            direction_x=direction(payload["direction_x"]),
            direction_y=direction(payload["direction_y"]),
            shear_failure_enabled=bool(
                payload.get("shear_failure_enabled", False)
            ),
            torsion_supported=bool(payload.get("torsion_supported", False)),
            xy_strength_interaction_supported=bool(
                payload.get("xy_strength_interaction_supported", False)
            ),
            applicability_flags=tuple(
                payload.get("applicability_flags", ())
            ),
            calibration_metrics=dict(
                payload.get("calibration_metrics", {})
            ),
            certified=bool(payload.get("certified", False)),
            certification_path=payload.get("certification_path"),
            schema_version=str(
                payload.get("schema_version", MDOF_ARTIFACT_SCHEMA_VERSION)
            ),
            artifact_hash=str(payload.get("artifact_hash", "")),
        )
        return artifact


@dataclass(frozen=True)
class MDOFModelInfo:
    master_nodes: tuple[int, ...]
    story_elements: tuple[int, ...]
    floor_elevations_m: tuple[float, ...]
    story_heights_m: tuple[float, ...]
    modal_period_x_s: float
    modal_period_y_s: float
    modal_periods_s: tuple[float, ...]
    x_mode: int
    y_mode: int


@dataclass(frozen=True)
class MDOFRunResult:
    status: str
    max_midr: float
    critical_story: int | None
    critical_direction: str | None
    critical_time_s: float | None
    history_path: str | None
    summary: dict[str, Any]


def mdof_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return MDOF settings with backward-compatible defaults."""
    settings = dict(config.get("mdof", {}))
    settings.setdefault("artifact_subdir", "mdof/artifacts")
    settings.setdefault("history_subdir", "mdof/histories")
    settings.setdefault("story_material", "IMKPeakOriented")
    settings.setdefault("require_certification_for_ida", True)
    settings.setdefault("allow_uncertified_diagnostic", False)
    settings.setdefault("shear_failure_enabled", False)
    settings.setdefault("damping_ratio", config["model"]["damping_ratio"])
    return settings


def analysis_engine(config: dict[str, Any]) -> str:
    engine = str(config.get("ida", {}).get("analysis_engine", "full3d")).lower()
    if engine not in {"full3d", "mdof"}:
        raise ValueError("ida.analysis_engine must be full3d or mdof")
    return engine


def artifact_path(config: dict[str, Any], building_id: str) -> Path:
    return (
        Path(config["run_dir"])
        / str(mdof_config(config)["artifact_subdir"])
        / f"{building_id}.json"
    )


def history_path(
    config: dict[str, Any],
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> Path:
    token = f"{target_im_g:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        Path(config["run_dir"])
        / str(mdof_config(config)["history_subdir"])
        / building_id
        / pair_id
        / f"im_{token}.npz"
    )


def artifact_payload_hash(artifact: MDOFArtifact) -> str:
    payload = artifact.to_dict()
    payload["artifact_hash"] = ""
    payload["certified"] = False
    payload["certification_path"] = None
    return stable_hash(payload)


def save_artifact(
    artifact: MDOFArtifact,
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
) -> MDOFArtifact:
    path = artifact_path(config, building_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"{building_id}: MDOF artifact is missing; run build-mdof first"
        )
    artifact = MDOFArtifact.from_dict(read_json(path))
    artifact.validate()
    if require_certified and not artifact.certified:
        raise RuntimeError(
            f"{building_id}: MDOF artifact has not passed strict validation"
        )
    return artifact


def shear_stiffness_matrix(story_stiffnesses: np.ndarray) -> np.ndarray:
    """Return the tridiagonal floor stiffness matrix of a shear building."""
    stiffness = np.asarray(story_stiffnesses, dtype=float)
    if stiffness.ndim != 1 or stiffness.size < 1:
        raise ValueError("At least one story stiffness is required")
    if np.any(~np.isfinite(stiffness)) or np.any(stiffness <= 0.0):
        raise ValueError("Story stiffnesses must be finite and positive")
    count = stiffness.size
    matrix = np.zeros((count, count), dtype=float)
    for story, value in enumerate(stiffness):
        matrix[story, story] += value
        if story > 0:
            matrix[story - 1, story - 1] += value
            matrix[story - 1, story] -= value
            matrix[story, story - 1] -= value
    return matrix


def modal_assurance_criterion(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.shape != b.shape or not a.size:
        raise ValueError("MAC vectors must have the same nonzero size")
    denominator = float((a @ a) * (b @ b))
    if denominator <= 0.0:
        return 0.0
    return float((a @ b) ** 2 / denominator)


def _imk_material(
    material_tag: int,
    spring: StorySpringParameters,
) -> None:
    spring.validate()
    ops = _ops()
    cracking_enabled = spring.cracking_deformation_m > 0.0
    branch_force = (
        spring.cracking_branch_force_kn if cracking_enabled else 0.0
    )
    core_stiffness = (
        spring.post_cracking_stiffness_kn_m
        if cracking_enabled
        else spring.initial_stiffness_kn_m
    )
    core_yield = spring.yield_force_kn - branch_force
    core_capping = spring.capping_force_kn - branch_force
    core_residual = spring.residual_force_kn - branch_force
    if min(core_yield, core_capping, core_residual) <= 0.0:
        raise ValueError(
            f"Story {spring.story}: cracking leaves non-positive IMK force"
        )
    pre_capping = max(
        spring.capping_deformation_m - spring.yield_deformation_m,
        spring.yield_deformation_m * 1.0e-6,
    )
    if spring.post_capping_slope_kn_m < -1.0e-12:
        post_capping = (
            spring.capping_force_kn - spring.residual_force_kn
        ) / abs(spring.post_capping_slope_kn_m)
    else:
        post_capping = (
            spring.ultimate_deformation_m
            - spring.capping_deformation_m
    )
    post_capping = max(post_capping, pre_capping * 1.0e-6)
    core_tag = material_tag if not cracking_enabled else material_tag * 10 + 1
    energy_scale = spring.yield_force_kn / core_yield
    ops.uniaxialMaterial(
        "IMKPeakOriented",
        core_tag,
        core_stiffness,
        pre_capping,
        post_capping,
        spring.ultimate_deformation_m,
        core_yield,
        core_capping / core_yield,
        core_residual / core_yield,
        pre_capping,
        post_capping,
        spring.ultimate_deformation_m,
        core_yield,
        core_capping / core_yield,
        core_residual / core_yield,
        spring.lambda_s * energy_scale,
        spring.lambda_c * energy_scale,
        spring.lambda_a * energy_scale,
        spring.lambda_k * energy_scale,
        spring.c_s,
        spring.c_c,
        spring.c_a,
        spring.c_k,
        spring.deterioration_pos,
        spring.deterioration_neg,
    )
    if cracking_enabled:
        cracking_tag = material_tag * 10 + 2
        bounded_cracking_tag = material_tag * 10 + 3
        crack_force = spring.cracking_branch_force_kn
        crack_deformation = spring.cracking_deformation_m
        ultimate_deformation = spring.ultimate_deformation_m
        ops.uniaxialMaterial(
            "ElasticMultiLinear",
            cracking_tag,
            "-strain",
            -ultimate_deformation,
            -crack_deformation,
            0.0,
            crack_deformation,
            ultimate_deformation,
            "-stress",
            -crack_force,
            -crack_force,
            0.0,
            crack_force,
            crack_force,
        )
        ops.uniaxialMaterial(
            "MinMax",
            bounded_cracking_tag,
            cracking_tag,
            "-min",
            -spring.ultimate_deformation_m,
            "-max",
            spring.ultimate_deformation_m,
        )
        ops.uniaxialMaterial(
            "Parallel",
            material_tag,
            core_tag,
            bounded_cracking_tag,
        )


def _directional_modal_identification(
    nodes: tuple[int, ...],
    masses_x: tuple[float, ...],
    masses_y: tuple[float, ...],
    periods: np.ndarray,
) -> tuple[int, int]:
    ops = _ops()
    score_x = []
    score_y = []
    mx = np.asarray(masses_x, dtype=float)
    my = np.asarray(masses_y, dtype=float)
    for mode in range(1, len(periods) + 1):
        phi_x = np.asarray(
            [ops.nodeEigenvector(node, mode, 1) for node in nodes],
            dtype=float,
        )
        phi_y = np.asarray(
            [ops.nodeEigenvector(node, mode, 2) for node in nodes],
            dtype=float,
        )
        denominator = float(phi_x @ (mx * phi_x) + phi_y @ (my * phi_y))
        if denominator <= 0.0:
            score_x.append(0.0)
            score_y.append(0.0)
            continue
        score_x.append(float((phi_x @ mx) ** 2 / denominator))
        score_y.append(float((phi_y @ my) ** 2 / denominator))
    candidates = [
        (
            score_x[x] + score_y[y],
            -max(x, y),
            x,
            y,
        )
        for x in range(len(periods))
        for y in range(len(periods))
        if x != y
    ]
    if not candidates:
        raise RuntimeError("MDOF model did not produce distinct X/Y modes")
    *_, x_index, y_index = max(candidates)
    return int(x_index + 1), int(y_index + 1)


def build_model(
    artifact: MDOFArtifact,
    *,
    damping_ratio: float,
) -> MDOFModelInfo:
    """Construct the exact 2N-DOF OpenSees reduced model."""
    artifact.validate()
    if not 0.0 <= damping_ratio < 1.0:
        raise ValueError("MDOF damping ratio must lie in [0, 1)")
    ops = _ops()
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    base_node = 1
    ops.node(base_node, 0.0, 0.0)
    ops.fix(base_node, 1, 1)
    floor_nodes: list[int] = []
    story_elements: list[int] = []
    for floor in range(artifact.story_count):
        node = floor + 2
        floor_nodes.append(node)
        # Coincident nodes make the element deformation exactly u_i-u_{i-1};
        # physical story heights remain explicit artifact data for IDR/PDelta.
        ops.node(node, 0.0, 0.0)
        ops.mass(
            node,
            artifact.floor_masses_x_kn_s2_m[floor],
            artifact.floor_masses_y_kn_s2_m[floor],
        )
        sx = artifact.direction_x.springs[floor]
        sy = artifact.direction_y.springs[floor]
        material_x = 1000 + floor * 10 + 1
        material_y = 1000 + floor * 10 + 2
        geometric_x = 1000 + floor * 10 + 3
        geometric_y = 1000 + floor * 10 + 4
        parallel_x = 1000 + floor * 10 + 5
        parallel_y = 1000 + floor * 10 + 6
        _imk_material(material_x, sx)
        _imk_material(material_y, sy)
        ops.uniaxialMaterial(
            "Elastic", geometric_x, sx.geometric_stiffness_kn_m
        )
        ops.uniaxialMaterial(
            "Elastic", geometric_y, sy.geometric_stiffness_kn_m
        )
        ops.uniaxialMaterial(
            "Parallel", parallel_x, material_x, geometric_x
        )
        ops.uniaxialMaterial(
            "Parallel", parallel_y, material_y, geometric_y
        )
        element = 2000 + floor
        story_elements.append(element)
        lower = base_node if floor == 0 else floor_nodes[floor - 1]
        ops.element(
            "zeroLength",
            element,
            lower,
            node,
            "-mat",
            parallel_x,
            parallel_y,
            "-dir",
            1,
            2,
        )
    mode_count = 2 * artifact.story_count
    eigenvalues = np.asarray(
        ops.eigen("-fullGenLapack", mode_count), dtype=float
    )
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise RuntimeError("MDOF model has a non-positive elastic eigenvalue")
    periods = 2.0 * math.pi / np.sqrt(eigenvalues)
    x_mode, y_mode = _directional_modal_identification(
        tuple(floor_nodes),
        artifact.floor_masses_x_kn_s2_m,
        artifact.floor_masses_y_kn_s2_m,
        periods,
    )
    if damping_ratio > 0.0:
        ops.modalDamping(*[float(damping_ratio) for _ in periods])
    return MDOFModelInfo(
        master_nodes=tuple(floor_nodes),
        story_elements=tuple(story_elements),
        floor_elevations_m=artifact.floor_elevations_m,
        story_heights_m=artifact.story_heights_m,
        modal_period_x_s=float(periods[x_mode - 1]),
        modal_period_y_s=float(periods[y_mode - 1]),
        modal_periods_s=tuple(float(value) for value in periods),
        x_mode=x_mode,
        y_mode=y_mode,
    )


def _set_transient_strategy(
    algorithm: str,
    *,
    test_type: str,
    tolerance: float,
    max_iterations: int,
) -> None:
    ops = _ops()
    ops.test(test_type, tolerance, max_iterations, 0)
    if algorithm == "NewtonLineSearch":
        ops.algorithm("NewtonLineSearch", 0.8)
    elif algorithm == "ModifiedNewton":
        ops.algorithm("ModifiedNewton", "-initial")
    elif algorithm == "KrylovNewton":
        ops.algorithm("KrylovNewton")
    else:
        ops.algorithm("Newton")


def _configure_transient() -> None:
    ops = _ops()
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    _set_transient_strategy(
        "Newton",
        test_type="EnergyIncr",
        tolerance=1.0e-7,
        max_iterations=30,
    )
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")


def _load_recorder(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, 0), dtype=float)
    return np.loadtxt(path, dtype=float, ndmin=2)


def _parse_element_forces(
    recorder_data: np.ndarray,
    story_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total story forces from concatenated zeroLength force output."""
    if recorder_data.size == 0:
        return (
            np.empty((0, story_count), dtype=float),
            np.empty((0, story_count), dtype=float),
        )
    values = recorder_data[:, 1:]
    if values.shape[1] == 4 * story_count:
        reshaped = values.reshape(values.shape[0], story_count, 4)
        # End-j resisting forces have the same sign as the story restoring
        # force for a positive upper-floor displacement.
        return reshaped[:, :, 2], reshaped[:, :, 3]
    if values.shape[1] == 2 * story_count:
        reshaped = values.reshape(values.shape[0], story_count, 2)
        return reshaped[:, :, 0], reshaped[:, :, 1]
    raise RuntimeError(
        "Unexpected MDOF story-force recorder layout: "
        f"{values.shape[1]} columns for {story_count} stories"
    )


def _spring_backbone_tangent(
    spring: StorySpringParameters,
    absolute_deformation: np.ndarray | float,
) -> np.ndarray:
    absolute = np.asarray(absolute_deformation, dtype=float)
    residual_deformation = _spring_residual_deformation(spring)
    hardening = (
        spring.capping_force_kn - spring.yield_force_kn
    ) / max(
        spring.capping_deformation_m - spring.yield_deformation_m,
        1.0e-12,
    )
    return np.select(
        (
            absolute <= spring.cracking_deformation_m,
            absolute <= spring.yield_deformation_m,
            absolute <= spring.capping_deformation_m,
            absolute <= residual_deformation,
            absolute <= spring.ultimate_deformation_m,
        ),
        (
            spring.initial_stiffness_kn_m,
            (
                spring.post_cracking_stiffness_kn_m
                if spring.cracking_deformation_m > 0.0
                else spring.initial_stiffness_kn_m
            ),
            hardening,
            spring.post_capping_slope_kn_m,
            0.0,
        ),
        default=0.0,
    ).astype(float)


def _spring_residual_deformation(
    spring: StorySpringParameters,
) -> float:
    return float(
        spring.capping_deformation_m
        + (spring.capping_force_kn - spring.residual_force_kn)
        / max(abs(spring.post_capping_slope_kn_m), 1.0e-12)
    )


def _observed_cyclic_degradation(
    deformation: np.ndarray,
    force: np.ndarray,
    spring: StorySpringParameters,
) -> dict[str, float | int | None]:
    """Report degradation exercised by a response, without re-fitting IMK."""
    deformation = np.asarray(deformation, dtype=float)
    force = np.asarray(force, dtype=float)
    if deformation.size < 4:
        return {
            "inelastic_reversal_count": 0,
            "observed_strength_retention_ratio": None,
            "observed_unloading_stiffness_retention_ratio": None,
        }
    increment_sign = np.sign(np.diff(deformation))
    for index in range(1, increment_sign.size):
        if increment_sign[index] == 0.0:
            increment_sign[index] = increment_sign[index - 1]
    reversal_indices = (
        np.flatnonzero(increment_sign[1:] * increment_sign[:-1] < 0.0) + 1
    )
    reversal_indices = reversal_indices[
        np.abs(deformation[reversal_indices])
        >= spring.yield_deformation_m
    ]
    reversal_strengths = np.abs(force[reversal_indices])
    strength_retention: float | None = None
    if reversal_strengths.size >= 2 and np.max(reversal_strengths) > 0.0:
        strength_retention = float(
            reversal_strengths[-1] / np.max(reversal_strengths)
        )
    unloading_stiffnesses: list[float] = []
    for reversal in reversal_indices:
        following = min(int(reversal) + 2, deformation.size - 1)
        deformation_change = float(
            deformation[following] - deformation[int(reversal)]
        )
        if abs(deformation_change) <= 1.0e-12:
            continue
        unloading_stiffnesses.append(
            abs(
                float(force[following] - force[int(reversal)])
                / deformation_change
            )
        )
    unloading_retention: float | None = None
    if (
        len(unloading_stiffnesses) >= 2
        and unloading_stiffnesses[0] > 0.0
    ):
        unloading_retention = float(
            unloading_stiffnesses[-1] / unloading_stiffnesses[0]
        )
    return {
        "inelastic_reversal_count": int(reversal_indices.size),
        "observed_strength_retention_ratio": strength_retention,
        "observed_unloading_stiffness_retention_ratio": (
            unloading_retention
        ),
    }


def _physical_endpoint_basis(
    artifact: MDOFArtifact,
    info: MDOFModelInfo,
    *,
    drift_limit: float,
) -> list[str]:
    """Classify a committed state before a singular tangent is advanced."""
    ops = _ops()
    displacement_x = np.asarray(
        [ops.nodeDisp(node, 1) for node in info.master_nodes], dtype=float
    )
    displacement_y = np.asarray(
        [ops.nodeDisp(node, 2) for node in info.master_nodes], dtype=float
    )
    delta_x = np.diff(np.concatenate(([0.0], displacement_x)))
    delta_y = np.diff(np.concatenate(([0.0], displacement_y)))
    heights = np.asarray(artifact.story_heights_m, dtype=float)
    basis = []
    if max(
        float(np.max(np.abs(delta_x / heights))),
        float(np.max(np.abs(delta_y / heights))),
    ) >= drift_limit:
        basis.append("drift_limit")
    if any(
        abs(delta) >= spring.ultimate_deformation_m
        for reduction, delta_values in (
            (artifact.direction_x, delta_x),
            (artifact.direction_y, delta_y),
        )
        for spring, delta in zip(reduction.springs, delta_values)
    ):
        basis.append("story_spring_ultimate")
    for label, reduction, delta_values in (
        ("x", artifact.direction_x, delta_x),
        ("y", artifact.direction_y, delta_y),
    ):
        net_tangent = np.asarray(
            [
                float(_spring_backbone_tangent(spring, abs(delta)))
                + spring.geometric_stiffness_kn_m
                for spring, delta in zip(reduction.springs, delta_values)
            ],
            dtype=float,
        )
        minimum_eigenvalue = float(
            np.min(
                np.linalg.eigvalsh(
                    shear_stiffness_matrix_allow_nonpositive(net_tangent)
                )
            )
        )
        residual_branch_reached = any(
            abs(float(delta)) >= _spring_residual_deformation(spring)
            for spring, delta in zip(reduction.springs, delta_values)
        )
        if minimum_eigenvalue <= 0.0 and residual_branch_reached:
            basis.append(f"nonpositive_tangent_with_pdelta_{label}")
    return basis


def _response_summary(
    artifact: MDOFArtifact,
    time_s: np.ndarray,
    displacement_x_m: np.ndarray,
    displacement_y_m: np.ndarray,
    story_force_x_kn: np.ndarray,
    story_force_y_kn: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = artifact.story_count
    base = np.zeros((time_s.size, 1), dtype=float)
    delta_x = np.diff(
        np.concatenate((base, displacement_x_m), axis=1), axis=1
    )
    delta_y = np.diff(
        np.concatenate((base, displacement_y_m), axis=1), axis=1
    )
    heights = np.asarray(artifact.story_heights_m, dtype=float)
    idr_x = delta_x / heights.reshape(1, count)
    idr_y = delta_y / heights.reshape(1, count)
    stacked = np.stack((np.abs(idr_x), np.abs(idr_y)), axis=2)
    flat_index = int(np.argmax(stacked)) if stacked.size else 0
    time_index, story_index, direction_index = np.unravel_index(
        flat_index, stacked.shape
    )
    maximum = float(stacked[time_index, story_index, direction_index])
    material_force_x = np.empty_like(story_force_x_kn)
    material_force_y = np.empty_like(story_force_y_kn)
    stability_index_x = np.zeros_like(story_force_x_kn)
    stability_index_y = np.zeros_like(story_force_y_kn)
    backbone_tangent_x = np.zeros_like(story_force_x_kn)
    backbone_tangent_y = np.zeros_like(story_force_y_kn)
    state_rows: list[dict[str, Any]] = []
    peak_profile_x = np.max(np.abs(idr_x), axis=0)
    peak_profile_y = np.max(np.abs(idr_y), axis=0)
    for story in range(count):
        sx = artifact.direction_x.springs[story]
        sy = artifact.direction_y.springs[story]
        material_force_x[:, story] = (
            story_force_x_kn[:, story]
            - sx.geometric_stiffness_kn_m * delta_x[:, story]
        )
        material_force_y[:, story] = (
            story_force_y_kn[:, story]
            - sy.geometric_stiffness_kn_m * delta_y[:, story]
        )
        row: dict[str, Any] = {"story": story + 1}
        for label, delta, force, spring in (
            ("X", delta_x[:, story], material_force_x[:, story], sx),
            ("Y", delta_y[:, story], material_force_y[:, story], sy),
        ):
            absolute = np.abs(delta)
            tangent = _spring_backbone_tangent(spring, absolute)
            stability_index = (
                spring.gravity_load_above_kn * absolute
                / np.maximum(
                    np.abs(force) * spring.height_m,
                    np.finfo(float).eps,
                )
            )
            # The index is undefined at the undeformed zero-force origin.
            stability_index[
                (absolute <= 1.0e-14) & (np.abs(force) <= 1.0e-10)
            ] = 0.0
            if label == "X":
                backbone_tangent_x[:, story] = tangent
                stability_index_x[:, story] = stability_index
            else:
                backbone_tangent_y[:, story] = tangent
                stability_index_y[:, story] = stability_index

            def first_time(threshold: float) -> float | None:
                indices = np.flatnonzero(absolute >= threshold)
                return (
                    float(time_s[int(indices[0])])
                    if indices.size
                    else None
                )

            residual_indices = np.flatnonzero(
                (absolute >= spring.capping_deformation_m)
                & (np.abs(force) <= spring.residual_force_kn * 1.05)
            )
            energy = float(
                np.sum(
                    0.5
                    * (force[1:] + force[:-1])
                    * np.diff(delta)
                )
            )
            observed_degradation = _observed_cyclic_degradation(
                delta, force, spring
            )
            row[label] = {
                "peak_deformation_m": float(np.max(absolute)),
                "peak_material_force_kn": float(np.max(np.abs(force))),
                "first_yield_time_s": first_time(
                    spring.yield_deformation_m
                ),
                "first_capping_time_s": first_time(
                    spring.capping_deformation_m
                ),
                "first_residual_strength_time_s": (
                    float(time_s[int(residual_indices[0])])
                    if residual_indices.size
                    else None
                ),
                "first_ultimate_time_s": first_time(
                    spring.ultimate_deformation_m
                ),
                "dissipated_energy_kn_m": abs(energy),
                "maximum_story_stability_index": float(
                    np.max(stability_index)
                ),
                **observed_degradation,
            }
        state_rows.append(row)
    minimum_tangent_eigenvalue_x = np.empty(time_s.size, dtype=float)
    minimum_tangent_eigenvalue_y = np.empty(time_s.size, dtype=float)
    geometric_x = np.asarray(
        [
            spring.geometric_stiffness_kn_m
            for spring in artifact.direction_x.springs
        ],
        dtype=float,
    )
    geometric_y = np.asarray(
        [
            spring.geometric_stiffness_kn_m
            for spring in artifact.direction_y.springs
        ],
        dtype=float,
    )
    for index in range(time_s.size):
        minimum_tangent_eigenvalue_x[index] = float(
            np.min(
                np.linalg.eigvalsh(
                    shear_stiffness_matrix_allow_nonpositive(
                        backbone_tangent_x[index] + geometric_x
                    )
                )
            )
        )
        minimum_tangent_eigenvalue_y[index] = float(
            np.min(
                np.linalg.eigvalsh(
                    shear_stiffness_matrix_allow_nonpositive(
                        backbone_tangent_y[index] + geometric_y
                    )
                )
            )
        )
    residual_limits_x = np.asarray(
        [
            _spring_residual_deformation(spring)
            for spring in artifact.direction_x.springs
        ],
        dtype=float,
    )
    residual_limits_y = np.asarray(
        [
            _spring_residual_deformation(spring)
            for spring in artifact.direction_y.springs
        ],
        dtype=float,
    )
    residual_branch_instability_x = bool(
        np.any(
            (minimum_tangent_eigenvalue_x <= 0.0)
            & np.any(
                np.abs(delta_x) >= residual_limits_x.reshape(1, -1),
                axis=1,
            )
        )
    )
    residual_branch_instability_y = bool(
        np.any(
            (minimum_tangent_eigenvalue_y <= 0.0)
            & np.any(
                np.abs(delta_y) >= residual_limits_y.reshape(1, -1),
                axis=1,
            )
        )
    )
    base_shear_x = story_force_x_kn[:, 0]
    base_shear_y = story_force_y_kn[:, 0]
    residual_x = idr_x[-1] if idr_x.size else np.zeros(count)
    residual_y = idr_y[-1] if idr_y.size else np.zeros(count)
    summary = {
        "max_midr": maximum,
        "critical_story": int(story_index + 1),
        "critical_direction": "X" if direction_index == 0 else "Y",
        "critical_time_s": float(time_s[time_index]),
        "peak_midr_profile_x": peak_profile_x.tolist(),
        "peak_midr_profile_y": peak_profile_y.tolist(),
        "maximum_residual_midr": float(
            max(np.max(np.abs(residual_x)), np.max(np.abs(residual_y)))
        ),
        "peak_roof_displacement_x_m": float(
            np.max(np.abs(displacement_x_m[:, -1]))
        ),
        "peak_roof_displacement_y_m": float(
            np.max(np.abs(displacement_y_m[:, -1]))
        ),
        "peak_base_shear_x_kn": float(np.max(np.abs(base_shear_x))),
        "peak_base_shear_y_kn": float(np.max(np.abs(base_shear_y))),
        "minimum_tangent_eigenvalue_x_kn_m": float(
            np.min(minimum_tangent_eigenvalue_x)
        ),
        "minimum_tangent_eigenvalue_y_kn_m": float(
            np.min(minimum_tangent_eigenvalue_y)
        ),
        "maximum_story_stability_index": float(
            max(np.max(stability_index_x), np.max(stability_index_y))
        ),
        "residual_branch_tangent_instability_x": (
            residual_branch_instability_x
        ),
        "residual_branch_tangent_instability_y": (
            residual_branch_instability_y
        ),
        "story_states": state_rows,
    }
    arrays = {
        "time_s": time_s,
        "u_x_m": displacement_x_m,
        "u_y_m": displacement_y_m,
        "delta_x_m": delta_x,
        "delta_y_m": delta_y,
        "idr_x": idr_x,
        "idr_y": idr_y,
        "story_force_x_kn": story_force_x_kn,
        "story_force_y_kn": story_force_y_kn,
        "material_force_x_kn": material_force_x,
        "material_force_y_kn": material_force_y,
        "backbone_tangent_x_kn_m": backbone_tangent_x,
        "backbone_tangent_y_kn_m": backbone_tangent_y,
        "story_stability_index_x": stability_index_x,
        "story_stability_index_y": stability_index_y,
        "minimum_tangent_eigenvalue_x_kn_m": minimum_tangent_eigenvalue_x,
        "minimum_tangent_eigenvalue_y_kn_m": minimum_tangent_eigenvalue_y,
        "roof_displacement_x_m": displacement_x_m[:, -1],
        "roof_displacement_y_m": displacement_y_m[:, -1],
        "base_shear_x_kn": base_shear_x,
        "base_shear_y_kn": base_shear_y,
    }
    return summary, arrays


def shear_stiffness_matrix_allow_nonpositive(
    story_stiffnesses: np.ndarray,
) -> np.ndarray:
    """Assemble a tangent matrix, including zero/negative story tangents."""
    stiffness = np.asarray(story_stiffnesses, dtype=float)
    if (
        stiffness.ndim != 1
        or stiffness.size < 1
        or np.any(~np.isfinite(stiffness))
    ):
        raise ValueError("Tangent story stiffnesses must be a finite vector")
    count = stiffness.size
    matrix = np.zeros((count, count), dtype=float)
    for story, value in enumerate(stiffness):
        matrix[story, story] += value
        if story > 0:
            matrix[story - 1, story - 1] += value
            matrix[story - 1, story] -= value
            matrix[story, story - 1] -= value
    return matrix


def run_nltha_local_recovery(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    scale_factor: float,
) -> dict[str, Any]:
    """Run one paired-motion MDOF NLTHA with local failure recovery."""
    settings = mdof_config(config)
    require_certificate = bool(
        settings["require_certification_for_ida"]
        and not settings["allow_uncertified_diagnostic"]
    )
    artifact = load_artifact(
        config,
        str(building["building_id"]),
        require_certified=require_certificate,
    )
    started = time.perf_counter()
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
    if not math.isclose(
        observed_dt, record_dt, rel_tol=1.0e-6, abs_tol=1.0e-10
    ):
        raise ValueError(
            f"{pair['pair_id']}: catalog dt differs from component dt"
        )
    info = build_model(
        artifact,
        damping_ratio=float(settings["damping_ratio"]),
    )
    ops = _ops()
    acceleration_x = acceleration_x_g * G_STD * float(scale_factor)
    acceleration_y = acceleration_y_g * G_STD * float(scale_factor)
    ops.timeSeries("Path", 101, "-dt", record_dt, "-values", *acceleration_x)
    ops.timeSeries("Path", 102, "-dt", record_dt, "-values", *acceleration_y)
    ops.pattern("UniformExcitation", 101, 1, "-accel", 101)
    ops.pattern("UniformExcitation", 102, 2, "-accel", 102)
    _configure_transient()
    # OpenSees recorder file names are not Unicode-safe on every Windows build.
    # Allow an audit configuration to place only these disposable recorder files
    # in an ASCII-only directory; all scientific results still use run_dir.
    recorder_temp_dir_raw = settings.get("recorder_temp_dir")
    recorder_temp_dir = (
        Path(str(recorder_temp_dir_raw))
        if recorder_temp_dir_raw not in {None, ""}
        else None
    )
    if recorder_temp_dir is not None:
        recorder_temp_dir.mkdir(parents=True, exist_ok=True)
    descriptor_node, node_name = tempfile.mkstemp(
        prefix="mdof_node_", suffix=".txt", dir=recorder_temp_dir
    )
    descriptor_force, force_name = tempfile.mkstemp(
        prefix="mdof_force_", suffix=".txt", dir=recorder_temp_dir
    )
    os.close(descriptor_node)
    os.close(descriptor_force)
    node_path = Path(node_name)
    force_path = Path(force_name)
    recovery_plan = (
        (2, "Newton", "EnergyIncr", 1.0e-7, 30),
        (4, "Newton", "EnergyIncr", 1.0e-7, 30),
        (4, "NewtonLineSearch", "EnergyIncr", 1.0e-7, 30),
        (4, "ModifiedNewton", "EnergyIncr", 1.0e-7, 30),
        (4, "KrylovNewton", "EnergyIncr", 1.0e-7, 30),
        (4, "ModifiedNewton", "EnergyIncr", 1.0e-6, 100),
        (4, "KrylovNewton", "NormDispIncr", 1.0e-6, 100),
    )
    attempt_log: list[dict[str, Any]] = []
    maximum_recovery = 0
    terminal_code = 0
    total_duration = float(time_x[-1] - time_x[0])
    end_time = float(ops.getTime()) + total_duration
    tolerance_time = max(1.0e-10, record_dt * 1.0e-8)
    # The reduced system is cheap enough to inspect every committed state.
    # This prevents stepping beyond a physical tangent/ultimate endpoint and
    # then mislabelling the ensuing singular solve as generic nonconvergence.
    chunk_steps = min(
        int(config["ida"].get("local_recovery_chunk_steps", 200)), 10
    )
    physical_endpoint: list[str] = []
    try:
        ops.recorder(
            "Node",
            "-file",
            str(node_path),
            "-precision",
            16,
            "-time",
            "-node",
            *info.master_nodes,
            "-dof",
            1,
            2,
            "disp",
        )
        ops.recorder(
            "Element",
            "-file",
            str(force_path),
            "-precision",
            16,
            "-time",
            "-ele",
            *info.story_elements,
            "force",
        )
        while float(ops.getTime()) < end_time - tolerance_time:
            remaining = end_time - float(ops.getTime())
            steps = min(
                chunk_steps,
                max(int(math.floor(remaining / record_dt + 1.0e-9)), 1),
            )
            dt = min(record_dt, remaining / steps)
            _set_transient_strategy(
                "Newton",
                test_type="EnergyIncr",
                tolerance=1.0e-7,
                max_iterations=30,
            )
            code = int(ops.analyze(steps, dt))
            if code == 0:
                physical_endpoint = _physical_endpoint_basis(
                    artifact,
                    info,
                    drift_limit=float(
                        config["model"]["dynamic_instability_drift"]
                    ),
                )
                if physical_endpoint:
                    break
                continue
            # OpenSees may have committed part of a multi-step call before
            # returning a failure.  Inspect that last valid state before any
            # numerical recovery; a physical endpoint takes precedence.
            physical_endpoint = _physical_endpoint_basis(
                artifact,
                info,
                drift_limit=float(
                    config["model"]["dynamic_instability_drift"]
                ),
            )
            if physical_endpoint:
                break
            recovered = False
            for recovery_level, (
                substep,
                algorithm,
                test_type,
                tolerance,
                iterations,
            ) in enumerate(recovery_plan, start=1):
                remaining = end_time - float(ops.getTime())
                if remaining <= tolerance_time:
                    recovered = True
                    break
                local_dt = min(record_dt / substep, remaining)
                _set_transient_strategy(
                    algorithm,
                    test_type=test_type,
                    tolerance=tolerance,
                    max_iterations=iterations,
                )
                local_code = int(ops.analyze(1, local_dt))
                attempt_log.append(
                    {
                        "recovery_level": recovery_level,
                        "substep_factor": substep,
                        "algorithm": algorithm,
                        "test_type": test_type,
                        "tolerance": tolerance,
                        "max_iterations": iterations,
                        "status": (
                            "local_recovery_success"
                            if local_code == 0
                            else "numerical_failure"
                        ),
                        "termination_time_s": float(ops.getTime()),
                        "analysis_dt_s": local_dt,
                    }
                )
                maximum_recovery = max(maximum_recovery, recovery_level)
                if local_code == 0:
                    physical_endpoint = _physical_endpoint_basis(
                        artifact,
                        info,
                        drift_limit=float(
                            config["model"]["dynamic_instability_drift"]
                        ),
                    )
                    recovered = True
                    break
            if physical_endpoint:
                break
            if not recovered:
                terminal_code = code if code != 0 else -1
                break
        ops.remove("recorders")
        node_data = _load_recorder(node_path)
        force_data = _load_recorder(force_path)
    finally:
        try:
            ops.remove("recorders")
        except Exception:
            pass
        node_path.unlink(missing_ok=True)
        force_path.unlink(missing_ok=True)
    if node_data.size == 0:
        # Preserve a diagnostic t=0 history when no transient step commits.
        # This keeps an immediate numerical failure visible to paired
        # validation without misclassifying it as physical collapse.
        time_s = np.asarray([0.0], dtype=float)
        displacement = np.zeros(
            (1, artifact.story_count, 2), dtype=float
        )
        force_x = np.zeros((1, artifact.story_count), dtype=float)
        force_y = np.zeros((1, artifact.story_count), dtype=float)
    else:
        time_s = node_data[:, 0]
        displacement = node_data[:, 1:].reshape(
            node_data.shape[0], artifact.story_count, 2
        )
        force_time = force_data[:, 0]
        if force_time.shape != time_s.shape or not np.allclose(
            force_time, time_s, rtol=0.0, atol=record_dt * 1.0e-5
        ):
            raise RuntimeError(
                "MDOF node and story-force recorder times differ"
            )
        force_x, force_y = _parse_element_forces(
            force_data, artifact.story_count
        )
    summary, arrays = _response_summary(
        artifact,
        time_s,
        displacement[:, :, 0],
        displacement[:, :, 1],
        force_x,
        force_y,
    )
    completed_fraction = min(
        max(float(time_s[-1]) / max(total_duration, 1.0e-12), 0.0), 1.0
    )
    instability_limit = float(config["model"]["dynamic_instability_drift"])
    ultimate_reached = any(
        direction.get("first_ultimate_time_s") is not None
        for row in summary["story_states"]
        for direction in (row["X"], row["Y"])
    )
    tangent_instability = bool(
        summary["residual_branch_tangent_instability_x"]
        or summary["residual_branch_tangent_instability_y"]
    )
    collapse_mechanisms = []
    if summary["max_midr"] >= instability_limit:
        collapse_mechanisms.append("drift_limit")
    if ultimate_reached:
        collapse_mechanisms.append("story_spring_ultimate")
    if tangent_instability:
        collapse_mechanisms.append("nonpositive_tangent_with_pdelta")
    collapse_mechanisms.extend(
        mechanism
        for mechanism in physical_endpoint
        if mechanism not in collapse_mechanisms
    )
    summary["collapse_mechanisms"] = collapse_mechanisms
    summary["tangent_instability_observed"] = tangent_instability
    if collapse_mechanisms:
        status = "dynamic_instability"
    elif terminal_code != 0 or completed_fraction < 1.0 - 1.0e-8:
        status = "numerical_failure"
    else:
        status = "success"
    total_steps = int(math.ceil(total_duration / record_dt))
    if not attempt_log:
        attempt_log.append(
            {
                "recovery_level": 0,
                "substep_factor": 1,
                "algorithm": "Newton",
                "test_type": "EnergyIncr",
                "tolerance": 1.0e-7,
                "max_iterations": 30,
                "status": status,
                "termination_time_s": float(time_s[-1]),
                "completed_fraction": completed_fraction,
                "analysis_dt_s": record_dt,
                "completed_steps": int(round(completed_fraction * total_steps)),
                "total_steps": total_steps,
                "committed_time_increment_s": float(time_s[-1]),
            }
        )
    for attempt in attempt_log:
        attempt["max_midr"] = summary["max_midr"]
        attempt["target_im_g"] = target_im_g
        attempt.setdefault("completed_fraction", completed_fraction)
        attempt.setdefault(
            "completed_steps", int(round(completed_fraction * total_steps))
        )
        attempt.setdefault("total_steps", total_steps)
    return {
        "status": status,
        "max_midr": summary["max_midr"],
        "completed_steps": int(round(completed_fraction * total_steps)),
        "completed_samples": int(time_s.size),
        "total_steps": total_steps,
        "analysis_dt_s": record_dt,
        "termination_time_s": float(time_s[-1]),
        "completed_fraction": completed_fraction,
        "recovery_level": maximum_recovery,
        "attempts": attempt_log,
        "modal_period_x_s": info.modal_period_x_s,
        "modal_period_y_s": info.modal_period_y_s,
        "modal_x_mode": info.x_mode,
        "modal_y_mode": info.y_mode,
        "modal_x_effective_mass_ratio": (
            artifact.direction_x.target_effective_mass_ratios[0]
        ),
        "modal_y_effective_mass_ratio": (
            artifact.direction_y.target_effective_mass_ratios[0]
        ),
        "modal_cumulative_x_effective_mass_ratio": float(
            sum(artifact.direction_x.target_effective_mass_ratios)
        ),
        "modal_cumulative_y_effective_mass_ratio": float(
            sum(artifact.direction_y.target_effective_mass_ratios)
        ),
        "modal_identification_method": "bidirectional-2n-directional-mass",
        "modal_eigen_solver_requested": "fullGenLapack",
        "modal_eigen_solver_used": "fullGenLapack",
        "modal_eigen_solver_fallback": False,
        "dynamic_instability_interpretation": (
            "MDOF drift, story-spring ultimate, or nonpositive "
            "material-plus-PDelta tangent endpoint"
        ),
        "dynamic_instability_basis": (
            "+".join(collapse_mechanisms)
            if collapse_mechanisms
            else "not_classified"
        ),
        "collapse_scope": str(config["model"]["collapse_scope"]),
        "target_im_g": target_im_g,
        "component_response": None,
        "analysis_engine": "mdof",
        "mdof_artifact_hash": artifact.artifact_hash,
        "mdof_summary": summary,
        "_history_arrays": arrays,
        "runtime_s_solver": time.perf_counter() - started,
    }


def atomic_write_history(
    path: str | Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Atomically write a compressed float64 MDOF history bundle."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.npz", dir=target.parent
    )
    os.close(descriptor)
    try:
        normalized = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in arrays.items()
        }
        np.savez_compressed(temporary_name, **normalized)
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run_spo(
    artifact: MDOFArtifact,
    *,
    direction: str,
    mode_number: int = 1,
    damping_ratio: float = 0.0,
    displacement_increment_m: float = 0.001,
    maximum_story_drift_increment: float = 0.0001,
    maximum_roof_drift: float = 0.12,
    maximum_steps: int = 10000,
    maximum_runtime_s: float = 120.0,
    near_zero_tangent_ratio: float = 0.005,
) -> dict[str, Any]:
    """Run one mode-proportional diagnostic reduced-model pushover."""
    normalized_direction = direction.upper()
    if normalized_direction not in {"X", "Y"}:
        raise ValueError("MDOF SPO direction must be X or Y")
    if (
        displacement_increment_m <= 0.0
        or maximum_story_drift_increment <= 0.0
        or maximum_roof_drift <= 0.0
        or maximum_steps <= 0
        or maximum_runtime_s <= 0.0
    ):
        raise ValueError("MDOF SPO controls must be positive")
    if not 0.0 <= near_zero_tangent_ratio <= 0.05:
        raise ValueError(
            "MDOF SPO near-zero tangent ratio must lie in [0, 0.05]"
        )
    info = build_model(artifact, damping_ratio=damping_ratio)
    reduction = (
        artifact.direction_x
        if normalized_direction == "X"
        else artifact.direction_y
    )
    if not 1 <= int(mode_number) <= min(
        3, len(reduction.target_mode_shapes)
    ):
        raise ValueError("MDOF SPO mode must be one of the first three modes")
    dof = 1 if normalized_direction == "X" else 2
    shape = np.asarray(
        reduction.target_mode_shapes[int(mode_number) - 1], dtype=float
    )
    if shape[-1] < 0.0:
        shape *= -1.0
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if normalized_direction == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    weights = masses * shape
    scale = float(np.sum(np.abs(weights)))
    if scale <= 0.0:
        raise RuntimeError("MDOF modal pushover pattern is zero")
    weights /= scale
    modal_mass = float(shape @ (masses * shape))
    if modal_mass <= 0.0:
        raise RuntimeError("MDOF modal projection mass is non-positive")
    ops = _ops()
    ops.timeSeries("Linear", 20)
    ops.pattern("Plain", 20, 20)
    for node, force in zip(info.master_nodes, weights):
        load = [0.0, 0.0]
        load[dof - 1] = float(force)
        ops.load(node, *load)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 60, 0)
    ops.algorithm("Newton")
    modal_control_index = int(np.argmax(np.abs(shape)))
    control_shape = float(shape[modal_control_index])
    story_pattern = np.cumsum(weights[::-1])[::-1]
    active_springs = (
        artifact.direction_x.springs
        if normalized_direction == "X"
        else artifact.direction_y.springs
    )
    adaptive_higher_mode_control = int(mode_number) > 1

    analysis_increment = min(
        displacement_increment_m,
        maximum_story_drift_increment * min(artifact.story_heights_m),
    )

    def recovery_controls() -> tuple[list[tuple[int, float]], float | None]:
        """Rank nonsingular fallback control DOFs at the current state.

        A higher-mode roof displacement can reach a local extremum when one
        story crosses yield even though the generalized force is still
        increasing.  Continuing to prescribe roof displacement then makes the
        static equations singular.  Keep the modal control floor as long as it
        converges; only after solver exhaustion rank alternate floors using
        the exact shear-chain tangent sensitivity.  This avoids changing
        control coordinates unnecessarily on otherwise regular SPO branches.
        """
        if not adaptive_higher_mode_control:
            return [], None
        current = np.asarray(
            [ops.nodeDisp(node, dof) for node in info.master_nodes],
            dtype=float,
        )
        delta = np.diff(np.concatenate(([0.0], current)))
        tangent = np.asarray(
            [
                _spring_backbone_tangent(spring, abs(float(value)))
                + spring.geometric_stiffness_kn_m
                for spring, value in zip(active_springs, delta)
            ],
            dtype=float,
        )
        safe_tangent = np.where(
            np.abs(tangent) >= 1.0e-9,
            tangent,
            np.copysign(1.0e-9, tangent + 1.0e-30),
        )
        story_sensitivity = story_pattern / safe_tangent
        floor_sensitivity = np.cumsum(story_sensitivity)
        order = np.argsort(-np.abs(floor_sensitivity))
        controls = [
            (
                int(index),
                (
                    1.0
                    if float(floor_sensitivity[index]) >= 0.0
                    else -1.0
                ),
            )
            for index in order
            if abs(float(floor_sensitivity[index])) > 1.0e-15
        ]
        maximum_floor_sensitivity = float(
            np.max(np.abs(floor_sensitivity))
        )
        if maximum_floor_sensitivity <= 1.0e-15:
            return controls, None
        load_increment = analysis_increment / maximum_floor_sensitivity
        current_factor = abs(float(ops.getLoadFactor(20)))
        load_increment = min(
            load_increment,
            0.02 * max(current_factor, 1.0),
        )
        return controls, max(load_increment, 1.0e-9)

    def near_zero_tangent_basis() -> list[str]:
        """Detect an incipient story mechanism for static SPO only."""
        current = np.asarray(
            [ops.nodeDisp(node, dof) for node in info.master_nodes],
            dtype=float,
        )
        delta = np.diff(np.concatenate(([0.0], current)))
        for spring, value in zip(active_springs, delta):
            if abs(float(value)) < spring.yield_deformation_m:
                continue
            net_tangent = (
                _spring_backbone_tangent(spring, abs(float(value)))
                + spring.geometric_stiffness_kn_m
            )
            if (
                net_tangent
                <= near_zero_tangent_ratio
                * spring.initial_stiffness_kn_m
            ):
                return [
                    "near_zero_story_tangent_with_pdelta_"
                    f"{normalized_direction.lower()}"
                ]
        return []

    control_index = modal_control_index
    control_sign = 1.0 if control_shape >= 0.0 else -1.0
    control_node = info.master_nodes[control_index]
    ops.integrator(
        "DisplacementControl",
        control_node,
        dof,
        control_sign * analysis_increment,
    )
    ops.analysis("Static")
    modal_displacement_limit = (
        maximum_roof_drift * artifact.floor_elevations_m[-1]
    )
    roof = [0.0]
    modal_displacement = [0.0]
    base_shear = [0.0]
    generalized_force = [0.0]
    floor_displacements = [np.zeros(artifact.story_count, dtype=float)]
    status = "success"
    endpoint_basis: list[str] = []
    requested_increment = analysis_increment
    endpoint_resolution = analysis_increment / 64.0
    minimum_increment = analysis_increment / 1024.0
    control_floors_used = {control_index + 1}
    endpoint_trigger_maximum_midr: float | None = None
    accepted_steps = 0
    load_control_recovery_steps = 0
    guard_started = time.perf_counter()
    guard_triggered: str | None = None
    while modal_displacement[-1] < modal_displacement_limit:
        if accepted_steps >= maximum_steps:
            status = "numerical_failure"
            guard_triggered = "maximum_steps"
            break
        if time.perf_counter() - guard_started >= maximum_runtime_s:
            status = "numerical_failure"
            guard_triggered = "maximum_runtime_s"
            break
        code = -1
        accepted_increment = requested_increment
        alternate_controls, load_increment = recovery_controls()
        controls = [(control_index, control_sign)]
        controls.extend(
            candidate
            for candidate in alternate_controls
            if candidate != (control_index, control_sign)
        )
        for candidate_index, candidate_sign in controls:
            trial_increment = requested_increment
            while trial_increment + 1.0e-15 >= minimum_increment:
                candidate_node = info.master_nodes[candidate_index]
                ops.integrator(
                    "DisplacementControl",
                    candidate_node,
                    dof,
                    candidate_sign * trial_increment,
                )
                # Higher-mode load patterns contain force reversals and can
                # pass through abrupt IMK branch transitions even while the
                # generalized force is still increasing.  Use the same
                # general recovery family as the Full-3D SPO.
                for algorithm, arguments in (
                    ("Newton", ()),
                    ("NewtonLineSearch", (0.8,)),
                    ("KrylovNewton", ()),
                    ("BFGS", ()),
                    ("ModifiedNewton", ("-initial",)),
                ):
                    ops.test("NormDispIncr", 1.0e-7, 100, 0)
                    ops.algorithm(algorithm, *arguments)
                    code = int(ops.analyze(1))
                    if code == 0:
                        accepted_increment = trial_increment
                        control_index = candidate_index
                        control_sign = candidate_sign
                        control_floors_used.add(control_index + 1)
                        break
                if code == 0:
                    break
                trial_increment *= 0.5
            if code == 0:
                break
        if code != 0 and load_increment is not None:
            trial_load_increment = load_increment
            minimum_load_increment = load_increment / 64.0
            while (
                trial_load_increment + 1.0e-15
                >= minimum_load_increment
            ):
                ops.integrator("LoadControl", trial_load_increment)
                for algorithm, arguments in (
                    ("Newton", ()),
                    ("NewtonLineSearch", (0.8,)),
                    ("KrylovNewton", ()),
                    ("BFGS", ()),
                    ("ModifiedNewton", ("-initial",)),
                ):
                    ops.test("NormDispIncr", 1.0e-7, 100, 0)
                    ops.algorithm(algorithm, *arguments)
                    code = int(ops.analyze(1))
                    if code == 0:
                        accepted_increment = analysis_increment
                        load_control_recovery_steps += 1
                        break
                if code == 0:
                    break
                trial_load_increment *= 0.5
        ops.algorithm("Newton")
        if code != 0:
            endpoint_basis = _physical_endpoint_basis(
                artifact,
                info,
                drift_limit=maximum_roof_drift,
            )
            if not endpoint_basis:
                endpoint_basis = near_zero_tangent_basis()
            if not endpoint_basis:
                current = np.asarray(
                    [
                        ops.nodeDisp(node, dof)
                        for node in info.master_nodes
                    ],
                    dtype=float,
                )
                current_delta = np.diff(
                    np.concatenate(([0.0], current))
                )
                if any(
                    abs(float(value))
                    >= (
                        spring.capping_deformation_m
                        - max(
                            endpoint_resolution,
                            1.0e-6
                            * spring.capping_deformation_m,
                        )
                    )
                    for value, spring in zip(
                        current_delta, active_springs
                    )
                ):
                    endpoint_basis = [
                        "capping_transition_within_static_resolution_"
                        f"{normalized_direction.lower()}"
                    ]
            status = (
                "model_endpoint"
                if endpoint_basis
                else "numerical_failure"
            )
            break
        requested_increment = min(
            analysis_increment,
            max(accepted_increment * 2.0, minimum_increment),
        )
        accepted_steps += 1
        current = np.asarray(
            [ops.nodeDisp(node, dof) for node in info.master_nodes],
            dtype=float,
        )
        factor = float(ops.getLoadFactor(20))
        endpoint_basis = _physical_endpoint_basis(
            artifact,
            info,
            drift_limit=maximum_roof_drift,
        )
        if not endpoint_basis:
            endpoint_basis = near_zero_tangent_basis()
        current_delta = np.diff(np.concatenate(([0.0], current)))
        current_midr = float(
            np.max(
                np.abs(
                    current_delta
                    / np.asarray(artifact.story_heights_m, dtype=float)
                )
            )
        )
        if endpoint_basis:
            endpoint_trigger_maximum_midr = current_midr
        if endpoint_basis and current_midr > 1.05 * maximum_roof_drift:
            status = "model_endpoint"
            break
        roof.append(abs(float(current[-1])))
        modal_displacement.append(
            abs(float(current @ (masses * shape)) / modal_mass)
        )
        base_shear.append(abs(factor * float(np.sum(weights))))
        generalized_force.append(abs(factor))
        floor_displacements.append(current)
        if endpoint_basis:
            status = "model_endpoint"
            break
        if (
            len(roof) > 2
            and generalized_force[-1] <= 0.2 * max(generalized_force)
        ):
            status = "model_endpoint"
            endpoint_basis = ["postpeak_20pct_strength"]
            break
    displacement = np.asarray(floor_displacements, dtype=float)
    delta = np.diff(
        np.concatenate(
            (np.zeros((displacement.shape[0], 1)), displacement), axis=1
        ),
        axis=1,
    )
    idr = delta / np.asarray(artifact.story_heights_m).reshape(1, -1)
    return {
        "building_id": artifact.building_id,
        "analysis_engine": "mdof",
        "artifact_hash": artifact.artifact_hash,
        "direction": normalized_direction,
        "mode_number": int(mode_number),
        "target_period_s": float(
            reduction.target_periods_s[int(mode_number) - 1]
        ),
        "target_effective_mass_ratio": float(
            reduction.target_effective_mass_ratios[int(mode_number) - 1]
        ),
        "target_participation_factor": float(
            reduction.target_participation_factors[int(mode_number) - 1]
        ),
        "control_floor": modal_control_index + 1,
        "control_strategy": (
            "fixed_modal_with_sensitivity_ranked_recovery"
            if adaptive_higher_mode_control
            else "fixed_modal_control_floor"
        ),
        "control_floors_used": sorted(control_floors_used),
        "accepted_steps": accepted_steps,
        "load_control_recovery_steps": load_control_recovery_steps,
        "analysis_guard_triggered": guard_triggered,
        "near_zero_tangent_ratio": near_zero_tangent_ratio,
        "control_mode_shape_ordinate": control_shape,
        "requested_displacement_increment_m": displacement_increment_m,
        "analysis_displacement_increment_m": analysis_increment,
        "maximum_story_drift_increment": maximum_story_drift_increment,
        "modal_load_pattern": weights.tolist(),
        "modal_displacement_definition": "abs(phi.T M u)/(phi.T M phi)",
        "status": status,
        "endpoint_basis": endpoint_basis,
        "roof_displacement_m": roof,
        "modal_displacement_m": modal_displacement,
        "base_shear_kn": base_shear,
        "modal_generalized_force_kn": generalized_force,
        "floor_displacement_m": displacement.tolist(),
        "story_deformation_m": delta.tolist(),
        "story_drift_ratio": idr.tolist(),
        "maximum_midr": float(np.max(np.abs(idr))),
        "endpoint_trigger_maximum_midr": endpoint_trigger_maximum_midr,
    }


def _compact_spo_ml_features(
    artifact: MDOFArtifact,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Convert one modal SPO curve into compact dimensionless ML inputs."""
    from .spo import fit_trilinear

    displacement = np.asarray(
        result["modal_displacement_m"], dtype=float
    )
    generalized = np.asarray(
        result["modal_generalized_force_kn"], dtype=float
    )
    unique = np.concatenate(
        ([True], np.diff(displacement) > 1.0e-12)
    )
    displacement = displacement[unique]
    generalized = generalized[unique]
    total_weight = float(sum(artifact.gravity_floor_loads_kn))
    total_height = float(artifact.floor_elevations_m[-1])
    prefix = (
        f"{str(result['direction']).lower()}_"
        f"mode_{int(result['mode_number'])}"
    )
    features: dict[str, float | bool | str | None] = {
        f"{prefix}_period_ratio_tn_t1": None,
        f"{prefix}_effective_mass_ratio": float(
            result["target_effective_mass_ratio"]
        ),
        f"{prefix}_participation_factor": float(
            result["target_participation_factor"]
        ),
        f"{prefix}_qy_over_weight": None,
        f"{prefix}_qc_over_weight": None,
        f"{prefix}_dy_over_height": None,
        f"{prefix}_dc_over_height": None,
        f"{prefix}_du_over_height": None,
        f"{prefix}_capping_ductility": None,
        f"{prefix}_postyield_stiffness_ratio": None,
        f"{prefix}_postcapping_stiffness_ratio": None,
        f"{prefix}_capping_drift_concentration": None,
        f"{prefix}_idealization_valid": False,
        f"{prefix}_qpeak_over_weight": None,
        f"{prefix}_dpeak_over_height": None,
        f"{prefix}_dterminal_over_height": None,
        f"{prefix}_terminal_strength_ratio": None,
    }
    reduction = (
        artifact.direction_x
        if result["direction"] == "X"
        else artifact.direction_y
    )
    features[f"{prefix}_period_ratio_tn_t1"] = float(
        result["target_period_s"] / reduction.target_periods_s[0]
    )
    if displacement.size >= 2 and float(np.max(generalized)) > 0.0:
        peak_index = int(np.argmax(generalized))
        peak_force = float(generalized[peak_index])
        features.update(
            {
                f"{prefix}_qpeak_over_weight": peak_force / total_weight,
                f"{prefix}_dpeak_over_height": float(
                    displacement[peak_index]
                )
                / total_height,
                f"{prefix}_dterminal_over_height": float(
                    displacement[-1]
                )
                / total_height,
                f"{prefix}_terminal_strength_ratio": float(
                    generalized[-1]
                )
                / peak_force,
            }
        )
    if displacement.size < 10 or float(np.max(generalized)) <= 0.0:
        features[f"{prefix}_idealization_message"] = (
            "insufficient modal SPO response"
        )
        return {
            "direction": result["direction"],
            "mode_number": int(result["mode_number"]),
            "features": features,
        }
    try:
        fit = fit_trilinear(displacement, generalized)
        initial_stiffness = max(
            float(fit["initial_stiffness_kn_m"]), 1.0e-12
        )
        postcapping = (
            float(fit["vu_kn"]) - float(fit["vc_kn"])
        ) / max(float(fit["du_m"]) - float(fit["dc_m"]), 1.0e-12)
        capping_index = int(
            np.argmin(np.abs(displacement - float(fit["dc_m"])))
        )
        drift = np.asarray(result["story_drift_ratio"], dtype=float)
        roof_drift = max(
            float(displacement[capping_index]) / total_height, 1.0e-12
        )
        concentration = float(
            np.max(np.abs(drift[capping_index])) / roof_drift
        )
        features.update(
            {
                f"{prefix}_qy_over_weight": float(fit["vy_kn"])
                / total_weight,
                f"{prefix}_qc_over_weight": float(fit["vc_kn"])
                / total_weight,
                f"{prefix}_dy_over_height": float(fit["dy_m"])
                / total_height,
                f"{prefix}_dc_over_height": float(fit["dc_m"])
                / total_height,
                f"{prefix}_du_over_height": float(fit["du_m"])
                / total_height,
                f"{prefix}_capping_ductility": float(fit["dc_m"])
                / max(float(fit["dy_m"]), 1.0e-12),
                f"{prefix}_postyield_stiffness_ratio": float(
                    fit["postyield_stiffness_final_kn_m"]
                )
                / initial_stiffness,
                f"{prefix}_postcapping_stiffness_ratio": (
                    postcapping / initial_stiffness
                ),
                f"{prefix}_capping_drift_concentration": concentration,
                f"{prefix}_idealization_valid": bool(
                    fit["trilinear_quality_valid"]
                ),
                f"{prefix}_idealization_message": str(
                    fit["idealization_method"]
                ),
            }
        )
    except Exception as exc:
        features[f"{prefix}_idealization_message"] = (
            f"{type(exc).__name__}: {exc}"
        )
    return {
        "direction": result["direction"],
        "mode_number": int(result["mode_number"]),
        "features": features,
    }


def _full3d_modal_spo_from_calibration(
    artifact: MDOFArtifact,
    *,
    direction: str,
    mode_number: int,
) -> dict[str, Any]:
    """Load one already-computed Full-3D modal SPO calibration curve.

    The Full-3D static analyses are required to build the reduced artifact, so
    using their cached curves as canonical ML inputs adds no structural solve.
    Modal displacement is the mass-weighted projection onto the same Full-3D
    target mode used to define the load pattern.
    """
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Full-3D modal SPO direction must be X or Y")
    reduction = (
        artifact.direction_x if normalized == "X" else artifact.direction_y
    )
    if not 1 <= int(mode_number) <= min(
        3, len(reduction.target_mode_shapes)
    ):
        raise ValueError("Full-3D modal SPO mode must be 1, 2, or 3")
    if not reduction.raw_curve_path:
        raise FileNotFoundError(
            f"{artifact.building_id} {normalized}: calibration curve missing"
        )
    curve_path = Path(reduction.raw_curve_path)
    if not curve_path.is_file():
        raise FileNotFoundError(
            f"{artifact.building_id} {normalized}: {curve_path} is missing"
        )
    pattern_name = (
        "first_mode" if int(mode_number) == 1 else f"mode_{mode_number}"
    )
    prefix = f"nonlinear__{pattern_name}__"
    required = (
        "pattern",
        "floor_displacement_m",
        "story_deformation_m",
        "story_shear_kn",
        "roof_displacement_m",
        "base_shear_kn",
        "load_factor",
    )
    with np.load(curve_path) as stored:
        missing = [
            key for key in required if f"{prefix}{key}" not in stored.files
        ]
        if missing:
            raise KeyError(
                f"{curve_path}: missing Full-3D modal curve arrays: "
                + ", ".join(missing)
            )
        arrays = {
            key: np.asarray(stored[f"{prefix}{key}"], dtype=float)
            for key in required
        }
    floor_displacement = arrays["floor_displacement_m"]
    story_deformation = arrays["story_deformation_m"]
    if floor_displacement.ndim != 2 or floor_displacement.shape[1] != (
        artifact.story_count
    ):
        raise ValueError("Full-3D modal floor-displacement shape is invalid")
    if story_deformation.shape != floor_displacement.shape:
        raise ValueError("Full-3D modal story-deformation shape is invalid")
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if normalized == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    shape = np.asarray(
        reduction.target_mode_shapes[int(mode_number) - 1], dtype=float
    )
    denominator = float(shape @ (masses * shape))
    if denominator <= 0.0:
        raise ValueError("Full-3D modal projection mass is non-positive")
    modal_displacement = np.abs(
        (floor_displacement * masses) @ shape / denominator
    )
    generalized_force = np.abs(arrays["load_factor"])
    if modal_displacement.shape != generalized_force.shape:
        raise ValueError("Full-3D modal force/displacement lengths differ")
    story_drift = story_deformation / np.asarray(
        artifact.story_heights_m, dtype=float
    ).reshape(1, -1)
    peak_force = float(np.max(generalized_force))
    postpeak_reached = bool(
        generalized_force.size > 2
        and int(np.argmax(generalized_force)) < generalized_force.size - 1
        and float(generalized_force[-1]) <= 0.80 * peak_force
    )
    return {
        "building_id": artifact.building_id,
        "analysis_engine": "full3d",
        "feature_role": "canonical_modal_spo_ml_input",
        "curve_source": "full3d_reduction_calibration_cache",
        "source_model_hash": artifact.source_model_hash,
        "source_signature": artifact.source_signature,
        "artifact_hash": artifact.artifact_hash,
        "direction": normalized,
        "mode_number": int(mode_number),
        "target_period_s": float(
            reduction.target_periods_s[int(mode_number) - 1]
        ),
        "target_effective_mass_ratio": float(
            reduction.target_effective_mass_ratios[int(mode_number) - 1]
        ),
        "target_participation_factor": float(
            reduction.target_participation_factors[int(mode_number) - 1]
        ),
        "modal_load_pattern": arrays["pattern"].tolist(),
        "modal_displacement_definition": (
            "abs(phi.T M u)/(phi.T M phi)"
        ),
        "modal_displacement_m": modal_displacement.tolist(),
        "modal_generalized_force_kn": generalized_force.tolist(),
        "roof_displacement_m": np.abs(
            arrays["roof_displacement_m"]
        ).tolist(),
        "base_shear_kn": np.abs(arrays["base_shear_kn"]).tolist(),
        "floor_displacement_m": floor_displacement.tolist(),
        "story_deformation_m": story_deformation.tolist(),
        "story_drift_ratio": story_drift.tolist(),
        "maximum_midr": float(np.max(np.abs(story_drift))),
        "status": "postpeak_reached" if postpeak_reached else "terminal_state",
        "endpoint_basis": (
            ["postpeak_full3d_calibration_curve"]
            if postpeak_reached
            else ["full3d_calibration_curve_terminal_state"]
        ),
        "raw_curve_path": str(curve_path.absolute()),
    }


def _spo_calibration_quality(
    artifact: MDOFArtifact,
) -> dict[str, dict[str, Any]]:
    quality: dict[str, dict[str, Any]] = {}
    for reduction in (artifact.direction_x, artifact.direction_y):
        count = min(
            3,
            len(reduction.mode_period_errors),
            len(reduction.target_effective_mass_ratios),
        )
        masses = np.asarray(
            reduction.target_effective_mass_ratios[:count], dtype=float
        )
        errors = np.asarray(
            reduction.mode_period_errors[:count], dtype=float
        )
        quality[reduction.direction] = {
            "target_periods_s": list(reduction.target_periods_s[:3]),
            "reduced_periods_s": list(reduction.reduced_periods_s[:3]),
            "period_errors": list(reduction.mode_period_errors[:3]),
            "mode_macs": list(reduction.mode_macs[:3]),
            "effective_mass_weighted_period_error": (
                float(np.sum(masses * errors) / np.sum(masses))
                if count and float(np.sum(masses)) > 0.0
                else None
            ),
            "elastic_core_drift_nrmse": float(
                reduction.elastic_core_drift_nrmse
            ),
            "nonlinear_backbone_nrmse": float(
                reduction.nonlinear_backbone_nrmse
            ),
            "story_higher_mode_compatibility_nrmse": [
                float(spring.higher_mode_compatibility_nrmse)
                for spring in reduction.springs
            ],
            "story_effective_higher_mode_weights": [
                float(spring.effective_higher_mode_weight)
                for spring in reduction.springs
            ],
        }
    return quality


def _spo_feature_payload(
    artifact: MDOFArtifact,
    compact_rows: list[dict[str, Any]],
    *,
    feature_source: str,
) -> dict[str, Any]:
    diagnostic_features = {
        key: value
        for row in compact_rows
        for key, value in row["features"].items()
    }
    core_suffixes = (
        "_period_ratio_tn_t1",
        "_effective_mass_ratio",
        "_qpeak_over_weight",
        "_dpeak_over_height",
        "_dterminal_over_height",
        "_terminal_strength_ratio",
    )
    flattened = {
        key: value
        for key, value in diagnostic_features.items()
        if key.endswith(core_suffixes)
    }
    complete = (
        len(compact_rows) == 6
        and {
            (str(row["direction"]), int(row["mode_number"]))
            for row in compact_rows
        }
        == {
            (direction, mode_number)
            for direction in ("X", "Y")
            for mode_number in (1, 2, 3)
        }
    )
    return {
        "schema_version": MDOF_SPO_FEATURE_SCHEMA_VERSION,
        "feature_profile": "modal_envelope_36",
        "feature_source": feature_source,
        "building_id": artifact.building_id,
        "source_model_hash": artifact.source_model_hash,
        "source_signature": artifact.source_signature,
        "artifact_hash": artifact.artifact_hash,
        "complete_three_mode_bidirectional": complete,
        "feature_count": len(flattened),
        "features": flattened,
        "core_features_complete": all(
            value is not None
            and (
                not isinstance(value, float)
                or math.isfinite(value)
            )
            for value in flattened.values()
        ),
        "all_modal_idealizations_valid": all(
            bool(
                row["features"].get(
                    f"{str(row['direction']).lower()}_"
                    f"mode_{int(row['mode_number'])}_"
                    "idealization_valid",
                    False,
                )
            )
            for row in compact_rows
        ),
        "diagnostic_feature_count": len(diagnostic_features),
        "diagnostic_features": diagnostic_features,
        "calibration_quality": _spo_calibration_quality(artifact),
        "modal_spo": compact_rows,
    }


def run_spo_batch(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    direction: str = "both",
    mode: str = "all",
) -> dict[str, Any]:
    """Run diagnostic reduced-model SPOs without touching Full 3D SPO rows."""
    normalized = direction.lower()
    if normalized not in {"x", "y", "both"}:
        raise ValueError("MDOF SPO direction must be x, y, or both")
    if mode not in {"1", "2", "3", "all"}:
        raise ValueError("MDOF SPO mode must be 1, 2, 3, or all")
    modes = (1, 2, 3) if mode == "all" else (int(mode),)
    artifact_root = (
        Path(config["run_dir"]) / str(mdof_config(config)["artifact_subdir"])
    )
    if building_ids:
        paths = [artifact_root / f"{value}.json" for value in building_ids]
    else:
        paths = sorted(artifact_root.glob("*.json"))
    results = []
    failures = []
    for path in paths:
        building_id = path.stem
        try:
            artifact = load_artifact(config, building_id)
            building_spo_results = []
            for current in (
                ("X", "Y") if normalized == "both" else (normalized.upper(),)
            ):
                for mode_number in modes:
                    result = run_spo(
                        artifact,
                        direction=current,
                        mode_number=mode_number,
                        displacement_increment_m=float(
                            config["model"]["spo_displacement_step_m"]
                        ),
                        maximum_story_drift_increment=float(
                            mdof_config(config).get(
                                "modal_spo_maximum_story_drift_increment",
                                0.0001,
                            )
                        ),
                        maximum_roof_drift=float(
                            mdof_config(config)
                            .get("calibration", {})
                            .get("nonlinear_maximum_roof_drift", 0.10)
                        ),
                        maximum_steps=int(
                            mdof_config(config).get(
                                "modal_spo_maximum_steps", 10000
                            )
                        ),
                        maximum_runtime_s=float(
                            mdof_config(config).get(
                                "modal_spo_maximum_runtime_s", 120.0
                            )
                        ),
                        near_zero_tangent_ratio=float(
                            mdof_config(config).get(
                                "modal_spo_near_zero_tangent_ratio",
                                0.005,
                            )
                        ),
                    )
                    output = (
                        Path(config["run_dir"])
                        / "mdof"
                        / "spo"
                        / building_id
                        / (
                            f"{current.lower()}_mode_"
                            f"{mode_number}.json"
                        )
                    )
                    atomic_write_json(output, result)
                    if mode_number == 1:
                        atomic_write_json(
                            output.with_name(f"{current.lower()}.json"),
                            result,
                        )
                    results.append(
                        {
                            "building_id": building_id,
                            "direction": current,
                            "mode_number": mode_number,
                            "status": result["status"],
                            "analysis_valid": (
                                result["status"]
                                in {"success", "model_endpoint"}
                            ),
                            "endpoint_basis": result["endpoint_basis"],
                            "maximum_midr": result["maximum_midr"],
                            "result_path": str(output.absolute()),
                        }
                    )
                    building_spo_results.append(result)
            mdof_compact_rows = [
                _compact_spo_ml_features(artifact, result)
                for result in building_spo_results
            ]
            full3d_spo_results = []
            for result in building_spo_results:
                reference = _full3d_modal_spo_from_calibration(
                    artifact,
                    direction=str(result["direction"]),
                    mode_number=int(result["mode_number"]),
                )
                reference_output = (
                    Path(config["run_dir"])
                    / "mdof"
                    / "spo"
                    / building_id
                    / "full3d_reference"
                    / (
                        f"{str(result['direction']).lower()}_mode_"
                        f"{int(result['mode_number'])}.json"
                    )
                )
                atomic_write_json(reference_output, reference)
                full3d_spo_results.append(reference)
            canonical_rows = [
                _compact_spo_ml_features(artifact, result)
                for result in full3d_spo_results
            ]
            complete = len(canonical_rows) == 6
            feature_path = (
                Path(config["run_dir"])
                / "mdof"
                / "spo"
                / building_id
                / (
                    "ml_features.json"
                    if complete
                    else "ml_features_partial.json"
                )
            )
            atomic_write_json(
                feature_path,
                _spo_feature_payload(
                    artifact,
                    canonical_rows,
                    feature_source=(
                        "full3d_modal_static_calibration_cache"
                    ),
                ),
            )
            diagnostic_feature_path = feature_path.with_name(
                (
                    "ml_features_mdof_diagnostic.json"
                    if complete
                    else "ml_features_mdof_diagnostic_partial.json"
                )
            )
            atomic_write_json(
                diagnostic_feature_path,
                _spo_feature_payload(
                    artifact,
                    mdof_compact_rows,
                    feature_source="mdof_reduced_model_diagnostic",
                ),
            )
        except Exception as exc:
            failures.append(
                {
                    "building_id": building_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return {
        "analysis_engine": "mdof_diagnostic",
        "canonical_feature_source": (
            "full3d_modal_static_calibration_cache"
        ),
        "run_count": len(results),
        "physical_endpoint_count": sum(
            row["status"] == "model_endpoint" for row in results
        ),
        "numerical_failure_count": sum(
            row["status"] == "numerical_failure" for row in results
        ),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
