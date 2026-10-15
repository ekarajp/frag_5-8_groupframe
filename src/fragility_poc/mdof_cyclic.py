"""Development-only cyclic calibration for MDOF deterioration parameters."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .db import connect, transaction, upsert_many
from .io_utils import atomic_write_json, read_json, stable_hash
from .mdof import (
    MDOFArtifact,
    StorySpringParameters,
    atomic_write_history,
    load_artifact,
    save_artifact,
)
from .mdof_calibration import (
    _analyze_displacement_step,
    _backbone_force,
    _configure_static,
    _story_cut_shears,
    _story_displacements,
)
from .structural import (
    _ops,
    build_model as build_full3d_model,
    run_gravity,
)

CYCLIC_CALIBRATION_SCHEMA_VERSION = (
    "full3d-cyclic-hierarchical-deterioration-v1"
)
CYCLIC_PARAMETER_NAMES = (
    "lambda_s",
    "lambda_c",
    "lambda_a",
    "lambda_k",
)


def cyclic_config(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config.get("mdof", {}).get("calibration", {})
    settings = dict(calibration.get("cyclic", {}))
    settings.setdefault(
        "roof_drift_protocol",
        [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06],
    )
    settings.setdefault("cycles_per_level", 2)
    settings.setdefault("patterns", ["first_mode", "uniform"])
    settings.setdefault("displacement_increment_m", 0.005)
    settings.setdefault("minimum_displacement_increment_m", 0.000078125)
    settings.setdefault("maximum_steps_per_target", 5000)
    settings.setdefault("lambda_min", 0.05)
    settings.setdefault("lambda_max", 20.0)
    settings.setdefault("ridge_alpha", 1.0)
    settings.setdefault("raw_subdir", "mdof/cyclic_calibration")
    return settings


def _cyclic_history_path(
    config: dict[str, Any],
    building_id: str,
    direction: str,
    pattern: str,
) -> Path:
    return (
        Path(config["run_dir"])
        / str(cyclic_config(config)["raw_subdir"])
        / building_id
        / f"{direction.lower()}_{pattern}.npz"
    )


def _cyclic_signature(
    building: dict[str, Any],
    artifact: MDOFArtifact,
    config: dict[str, Any],
    direction: str,
    pattern_name: str,
) -> str:
    return stable_hash(
        {
            "schema": CYCLIC_CALIBRATION_SCHEMA_VERSION,
            "model_hash": building.get("model_hash"),
            "artifact_hash": artifact.artifact_hash,
            "direction": direction,
            "pattern": pattern_name,
            "cyclic_config": cyclic_config(config),
            "model_config": config["model"],
        }
    )


def _cyclic_history_is_current(path: Path, signature: str) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        return read_json(metadata_path).get("cyclic_signature") == signature
    except Exception:
        return False


def _cyclic_pattern(
    artifact: MDOFArtifact,
    direction: str,
    pattern_name: str,
) -> np.ndarray:
    reduction = (
        artifact.direction_x if direction == "X" else artifact.direction_y
    )
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if direction == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    if pattern_name == "first_mode":
        values = masses * np.asarray(
            reduction.target_mode_shapes[0], dtype=float
        )
    elif pattern_name == "uniform":
        values = masses
    else:
        raise ValueError(f"Unsupported cyclic pattern: {pattern_name}")
    if float(np.sum(values)) < 0.0:
        values = -values
    norm = float(np.sum(np.abs(values)))
    if norm <= 0.0:
        raise RuntimeError("Cyclic lateral load pattern is zero")
    return values / norm


def _protocol_targets(height_m: float, settings: dict[str, Any]) -> list[float]:
    targets = []
    for ratio in settings["roof_drift_protocol"]:
        amplitude = float(ratio) * height_m
        for _cycle in range(int(settings["cycles_per_level"])):
            targets.extend((amplitude, -amplitude))
    targets.append(0.0)
    return targets


def run_full3d_cyclic_history(
    building: dict[str, Any],
    artifact: MDOFArtifact,
    config: dict[str, Any],
    *,
    direction: str,
    pattern_name: str,
) -> dict[str, Any]:
    """Run one displacement-controlled Full-3D cyclic calibration history."""
    settings = cyclic_config(config)
    info = build_full3d_model(building, config)
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"{building['building_id']}: gravity error exceeds 1%"
        )
    dof = 1 if direction == "X" else 2
    pattern = _cyclic_pattern(artifact, direction, pattern_name)
    ops = _ops()
    ops.timeSeries("Linear", 80)
    ops.pattern("Plain", 80, 80)
    for node, force in zip(info.master_nodes, pattern):
        load = [0.0] * 6
        load[dof - 1] = float(force)
        ops.load(node, *load)
    _configure_static()
    control_node = info.master_nodes[-1]
    height = float(info.floor_elevations_m[-1])
    floor_rows = [np.zeros(artifact.story_count, dtype=float)]
    delta_rows = [np.zeros(artifact.story_count, dtype=float)]
    shear_rows = [np.zeros(artifact.story_count, dtype=float)]
    roof_rows = [0.0]
    base_rows = [0.0]
    target_rows = [0]
    reduction_rows = []
    termination = "protocol_complete"
    started = time.perf_counter()
    for target_index, target in enumerate(
        _protocol_targets(height, settings), start=1
    ):
        for _step in range(int(settings["maximum_steps_per_target"])):
            current = float(ops.nodeDisp(control_node, dof))
            remaining = float(target) - current
            if abs(remaining) <= float(
                settings["minimum_displacement_increment_m"]
            ):
                break
            increment = math.copysign(
                min(
                    abs(remaining),
                    float(settings["displacement_increment_m"]),
                ),
                remaining,
            )
            code, _accepted, level = _analyze_displacement_step(
                control_node,
                dof,
                increment,
                float(settings["minimum_displacement_increment_m"]),
            )
            reduction_rows.append(level)
            if code != 0:
                termination = "physical_or_numerical_endpoint"
                break
            floor, delta = _story_displacements(info, dof)
            factor = float(ops.getLoadFactor(80))
            story, base, _equilibrium = _story_cut_shears(
                info,
                dof,
                load_factor=factor,
                absolute=False,
            )
            floor_rows.append(floor)
            delta_rows.append(delta)
            shear_rows.append(story)
            roof_rows.append(float(floor[-1]))
            base_rows.append(float(base))
            target_rows.append(target_index)
        if termination != "protocol_complete":
            break
    path = _cyclic_history_path(
        config, str(building["building_id"]), direction, pattern_name
    )
    arrays = {
        "floor_displacement_m": np.asarray(floor_rows, dtype=float),
        "story_deformation_m": np.asarray(delta_rows, dtype=float),
        "story_shear_kn": np.asarray(shear_rows, dtype=float),
        "roof_displacement_m": np.asarray(roof_rows, dtype=float),
        "base_shear_kn": np.asarray(base_rows, dtype=float),
        "protocol_target_index": np.asarray(target_rows, dtype=float),
    }
    atomic_write_history(path, arrays)
    result = {
        "building_id": str(building["building_id"]),
        "direction": direction,
        "pattern": pattern_name,
        "termination": termination,
        "step_count": len(roof_rows) - 1,
        "maximum_recovery_level": max(reduction_rows, default=0),
        "runtime_s": time.perf_counter() - started,
        "history_path": str(path.absolute()),
        "cyclic_signature": _cyclic_signature(
            building, artifact, config, direction, pattern_name
        ),
    }
    atomic_write_json(path.with_suffix(".json"), result)
    return result


def _turning_indices(deformation: np.ndarray) -> np.ndarray:
    values = np.asarray(deformation, dtype=float)
    if values.size < 3:
        return np.asarray([], dtype=int)
    increments = np.diff(values)
    sign = np.sign(increments)
    for index in range(1, sign.size):
        if sign[index] == 0.0:
            sign[index] = sign[index - 1]
    turning = np.flatnonzero(sign[:-1] * sign[1:] < 0.0) + 1
    return np.unique(np.concatenate((turning, [values.size - 1])))


def _observed_lambdas(
    spring: StorySpringParameters,
    deformation: np.ndarray,
    total_force: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> dict[str, float]:
    deformation = np.asarray(deformation, dtype=float)
    material_force = np.asarray(total_force, dtype=float) - (
        spring.geometric_stiffness_kn_m * deformation
    )
    increments = np.diff(deformation)
    incremental_energy = np.abs(
        0.5
        * (material_force[1:] + material_force[:-1])
        * increments
    )
    normalized_energy = np.concatenate(
        (
            [0.0],
            np.cumsum(incremental_energy)
            / max(
                spring.yield_force_kn * spring.yield_deformation_m,
                1.0e-12,
            ),
        )
    )
    turning = _turning_indices(deformation)
    if turning.size == 0:
        return {name: maximum for name in CYCLIC_PARAMETER_NAMES}
    residual_deformation = (
        spring.capping_deformation_m
        + (
            spring.capping_force_kn - spring.residual_force_kn
        )
        / max(abs(spring.post_capping_slope_kn_m), 1.0e-12)
    )
    envelope = _backbone_force(
        np.abs(deformation[turning]),
        stiffness=spring.initial_stiffness_kn_m,
        yield_force=spring.yield_force_kn,
        capping_ratio=(
            spring.capping_force_kn / spring.yield_force_kn
        ),
        capping_deformation=spring.capping_deformation_m,
        residual_ratio=(
            spring.residual_force_kn / spring.capping_force_kn
        ),
        residual_deformation=residual_deformation,
        ultimate_deformation=spring.ultimate_deformation_m,
    )
    strength_ratio = np.clip(
        np.abs(material_force[turning]) / np.maximum(envelope, 1.0e-12),
        0.0,
        1.0,
    )
    strength_samples = normalized_energy[turning] / np.maximum(
        1.0 - strength_ratio, 0.02
    )
    valid_strength = (
        np.abs(deformation[turning]) >= spring.yield_deformation_m
    )
    lambda_s = (
        float(np.median(strength_samples[valid_strength]))
        if np.any(valid_strength)
        else maximum
    )
    stiffness_samples = []
    for index in turning:
        stop = min(index + 4, deformation.size - 1)
        if stop <= index:
            continue
        dx = float(deformation[stop] - deformation[index])
        if abs(dx) <= 1.0e-12:
            continue
        observed_stiffness = abs(
            float(material_force[stop] - material_force[index]) / dx
        )
        ratio = float(
            np.clip(
                observed_stiffness / spring.initial_stiffness_kn_m,
                0.0,
                1.0,
            )
        )
        stiffness_samples.append(
            float(normalized_energy[index]) / max(1.0 - ratio, 0.02)
        )
    lambda_k = (
        float(np.median(stiffness_samples))
        if stiffness_samples
        else maximum
    )
    lambda_s = float(np.clip(lambda_s, minimum, maximum))
    lambda_k = float(np.clip(lambda_k, minimum, maximum))
    return {
        "lambda_s": lambda_s,
        "lambda_c": float(
            np.clip(math.sqrt(lambda_s * lambda_k), minimum, maximum)
        ),
        "lambda_a": float(
            np.clip(0.5 * (lambda_s + lambda_k), minimum, maximum)
        ),
        "lambda_k": lambda_k,
    }


def _descriptor(
    building: dict[str, Any],
    artifact: MDOFArtifact,
    story: int,
    direction: str,
) -> list[float]:
    beam_area = (
        2.0
        * float(building["beam_bars_per_face"])
        * math.pi
        * float(building["beam_bar_diameter_m"]) ** 2
        / 4.0
    )
    beam_ratio = beam_area / (
        float(building["beam_b_m"]) * float(building["beam_h_m"])
    )
    column_area = (
        float(building["column_bar_count"])
        * math.pi
        * float(building["column_bar_diameter_m"]) ** 2
        / 4.0
    )
    column_ratio = column_area / (
        float(building["column_b_m"]) * float(building["column_h_m"])
    )
    return [
        math.log(max(float(building["number_of_bays"]), 1.0)),
        math.log(max(float(building["floor_mass_kn_s2_m"]), 1.0e-12)),
        math.log(max(float(building["t1_s"]), 1.0e-12)),
        float(building["axial_ratio"]),
        float(building.get("scwb_strength_ratio") or 1.0),
        beam_ratio,
        column_ratio,
        float(story + 1) / artifact.story_count,
        1.0 if direction == "Y" else 0.0,
    ]


def _fit_hierarchical_models(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    minimum: float,
    maximum: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    features = np.asarray([row["features"] for row in rows], dtype=float)
    building_ids = np.asarray([row["building_id"] for row in rows])
    models: dict[str, Any] = {}
    loo_errors: dict[str, float] = {}
    for name in CYCLIC_PARAMETER_NAMES:
        labels = np.log(
            np.clip(
                np.asarray([row[name] for row in rows], dtype=float),
                minimum,
                maximum,
            )
        )
        errors = []
        for held_out in sorted(set(building_ids)):
            train = building_ids != held_out
            test = ~train
            if int(np.sum(train)) < features.shape[1] + 1:
                prediction = np.full(
                    int(np.sum(test)), float(np.median(labels[train]))
                )
            else:
                fold = make_pipeline(
                    StandardScaler(), Ridge(alpha=alpha)
                )
                fold.fit(features[train], labels[train])
                prediction = fold.predict(features[test])
            errors.extend(np.abs(prediction - labels[test]).tolist())
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features, labels)
        models[name] = model
        loo_errors[name] = float(np.median(errors)) if errors else math.inf
    return models, loo_errors


def _model_payload(models: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for name, pipeline in models.items():
        scaler = pipeline.named_steps["standardscaler"]
        ridge = pipeline.named_steps["ridge"]
        payload[name] = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
        }
    return payload


def _update_artifact(
    artifact: MDOFArtifact,
    building: dict[str, Any],
    models: dict[str, Any],
    *,
    minimum: float,
    maximum: float,
    regression_signature: str,
    loo_errors: dict[str, float],
    config: dict[str, Any],
) -> MDOFArtifact:
    def direction(label: str):
        reduction = (
            artifact.direction_x if label == "X" else artifact.direction_y
        )
        springs = []
        for story, spring in enumerate(reduction.springs):
            features = np.asarray(
                [_descriptor(building, artifact, story, label)], dtype=float
            )
            values = {
                name: float(
                    np.clip(
                        math.exp(float(models[name].predict(features)[0])),
                        minimum,
                        maximum,
                    )
                )
                for name in CYCLIC_PARAMETER_NAMES
            }
            springs.append(replace(spring, **values))
        return replace(reduction, springs=tuple(springs))

    flags = tuple(
        flag
        for flag in artifact.applicability_flags
        if flag != "cyclic-calibration-pending"
    )
    metrics = {
        **artifact.calibration_metrics,
        "cyclic_calibration_status": "hierarchical-development-fit",
        "cyclic_calibration_schema": CYCLIC_CALIBRATION_SCHEMA_VERSION,
        "cyclic_regression_signature": regression_signature,
        "cyclic_leave_one_building_out_median_abs_log_error": loo_errors,
    }
    return replace(
        artifact,
        direction_x=direction("X"),
        direction_y=direction("Y"),
        applicability_flags=flags,
        calibration_metrics=metrics,
        certified=False,
        certification_path=None,
    ).with_hash()


def calibrate_cyclic_development(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Fit deterioration on development buildings and predict holdouts."""
    settings = cyclic_config(config)
    all_ids = (
        list(manifest["development_buildings"])
        + list(manifest["holdout_buildings"])
    )
    marks = ",".join("?" for _ in all_ids)
    with connect(config["database_path"]) as connection:
        buildings = {
            str(row["building_id"]): dict(row)
            for row in connection.execute(
                "SELECT b.*, s.t1_s FROM building_catalog b "
                "JOIN spo_features s USING(building_id) "
                f"WHERE b.building_id IN ({marks})",
                all_ids,
            )
        }
    observations = []
    runs = []
    failures = []
    for building_id in manifest["development_buildings"]:
        building = buildings[building_id]
        artifact = load_artifact(
            config, building_id, require_certified=False
        )
        for direction in ("X", "Y"):
            reduction = (
                artifact.direction_x
                if direction == "X"
                else artifact.direction_y
            )
            for pattern in settings["patterns"]:
                path = _cyclic_history_path(
                    config, building_id, direction, str(pattern)
                )
                signature = _cyclic_signature(
                    building,
                    artifact,
                    config,
                    direction,
                    str(pattern),
                )
                try:
                    if not (
                        resume
                        and _cyclic_history_is_current(path, signature)
                    ):
                        runs.append(
                            run_full3d_cyclic_history(
                                building,
                                artifact,
                                config,
                                direction=direction,
                                pattern_name=str(pattern),
                            )
                        )
                    with np.load(path) as data:
                        deformation = np.asarray(
                            data["story_deformation_m"], dtype=float
                        )
                        shear = np.asarray(
                            data["story_shear_kn"], dtype=float
                        )
                    for story, spring in enumerate(reduction.springs):
                        values = _observed_lambdas(
                            spring,
                            deformation[:, story],
                            shear[:, story],
                            minimum=float(settings["lambda_min"]),
                            maximum=float(settings["lambda_max"]),
                        )
                        observations.append(
                            {
                                "building_id": building_id,
                                "direction": direction,
                                "pattern": str(pattern),
                                "story": story + 1,
                                "features": _descriptor(
                                    building,
                                    artifact,
                                    story,
                                    direction,
                                ),
                                **values,
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "building_id": building_id,
                            "direction": direction,
                            "pattern": str(pattern),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
    if failures:
        return {
            "completed": False,
            "failure_count": len(failures),
            "failures": failures,
            "runs": runs,
        }
    if not observations:
        raise RuntimeError("No cyclic development observations were produced")
    models, loo_errors = _fit_hierarchical_models(
        observations,
        alpha=float(settings["ridge_alpha"]),
        minimum=float(settings["lambda_min"]),
        maximum=float(settings["lambda_max"]),
    )
    model_payload = _model_payload(models)
    regression_signature = stable_hash(
        {
            "schema": CYCLIC_CALIBRATION_SCHEMA_VERSION,
            "development_buildings": manifest["development_buildings"],
            "observations": observations,
            "model": model_payload,
        }
    )
    model_path = (
        Path(config["run_dir"])
        / str(settings["raw_subdir"])
        / "hierarchical_regression.json"
    )
    atomic_write_json(
        model_path,
        {
            "schema_version": CYCLIC_CALIBRATION_SCHEMA_VERSION,
            "regression_signature": regression_signature,
            "development_buildings": manifest["development_buildings"],
            "holdout_buildings": manifest["holdout_buildings"],
            "leave_one_building_out_median_abs_log_error": loo_errors,
            "model": model_payload,
            "observation_count": len(observations),
        },
    )
    updated = []
    database_rows = []
    now = datetime.now(timezone.utc).isoformat()
    for building_id in all_ids:
        artifact = _update_artifact(
            load_artifact(config, building_id, require_certified=False),
            buildings[building_id],
            models,
            minimum=float(settings["lambda_min"]),
            maximum=float(settings["lambda_max"]),
            regression_signature=regression_signature,
            loo_errors=loo_errors,
            config=config,
        )
        saved = save_artifact(artifact, config)
        updated.append(building_id)
        database_rows.append(
            {
                "building_id": building_id,
                "artifact_path": str(saved.absolute()),
                "artifact_hash": artifact.artifact_hash,
                "source_signature": artifact.source_signature,
                "schema_version": artifact.schema_version,
                "story_count": artifact.story_count,
                "certified": 0,
                "certification_path": None,
                "applicability_flags_json": json.dumps(
                    artifact.applicability_flags, sort_keys=True
                ),
                "calibration_metrics_json": json.dumps(
                    artifact.calibration_metrics, sort_keys=True
                ),
                "created_utc": now,
            }
        )
    with transaction(config["database_path"]) as connection:
        upsert_many(
            connection,
            "mdof_artifacts",
            database_rows,
            ("building_id",),
        )
    return {
        "completed": True,
        "failure_count": 0,
        "development_building_count": len(
            manifest["development_buildings"]
        ),
        "updated_artifact_count": len(updated),
        "observation_count": len(observations),
        "leave_one_building_out_median_abs_log_error": loo_errors,
        "regression_signature": regression_signature,
        "model_path": str(model_path.absolute()),
        "runs": runs,
    }
