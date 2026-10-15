"""Strict, fail-closed validation of bidirectional nonlinear MDOF artifacts.

The validation workflow is deliberately separate from production IDA.  It
selects buildings and records without response labels, can execute paired
Full-3D/MDOF comparison cases, evaluates the locked acceptance criteria, and
only then writes a certificate.  Missing evidence is a failed gate, never an
implicit pass.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

from .active_ida import heuristic_capacity_prediction
from .db import connect, initialize, transaction, upsert_many
from .ida import (
    NLTHA_ANALYSIS_SCHEMA_VERSION,
    NLTHA_RECOVERY_POLICY_VERSION,
    _checked_scale_factor,
    _single_nltha_local_recovery,
    ida_curve_analysis_signature,
)
from .fragility import fit_building_fragility
from .io_utils import atomic_write_json, read_json, stable_hash
from .mdof import (
    MDOF_ANALYSIS_SCHEMA_VERSION,
    MDOF_CERTIFICATION_SCHEMA_VERSION,
    MDOFArtifact,
    artifact_path,
    atomic_write_history,
    load_artifact,
    mdof_config,
    modal_assurance_criterion,
    save_artifact,
    shear_stiffness_matrix,
)

VALIDATION_CASE_SCHEMA_VERSION = (
    "mdof-full3d-comparison-case-v2-independent-engine-signatures"
)
VALIDATION_MANIFEST_SCHEMA_VERSION = "mdof-validation-manifest-v1"


def validation_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(mdof_config(config).get("validation", {}))
    defaults = {
        "building_count": 12,
        "development_count": 8,
        "holdout_count": 4,
        "record_pair_count": 4,
        "dense_ida_holdout_count": 3,
        "period_error_first_max": 0.02,
        "first_mode_mac_min": 0.98,
        "higher_mode_period_error_max": 0.05,
        "three_mode_mass_weighted_period_error_max": 0.05,
        "higher_mode_mac_min": 0.90,
        "elastic_drift_nrmse_max": 0.05,
        "backbone_nrmse_median_max": 0.10,
        "backbone_nrmse_p90_max": 0.15,
        "midr_median_abs_log_error_max": 0.07,
        "midr_p90_abs_log_error_max": 0.15,
        "drift_profile_nrmse_median_max": 0.10,
        "critical_story_exact_min": 0.85,
        "critical_story_within_one_min": 0.95,
        "history_correlation_min": 0.90,
        "roof_base_median_error_max": 0.10,
        "collapse_capacity_ratio_min": 0.90,
        "collapse_capacity_ratio_max": 1.10,
        "fragility_median_error_max": 0.10,
        "fragility_dispersion_abs_error_max": 0.05,
        "minimum_nltha_speedup": 20.0,
        "maximum_break_even_nltha_count": 20,
        "fragility_bootstrap_count": 500,
    }
    for key, value in defaults.items():
        settings.setdefault(key, value)
    return settings


def _validation_root(config: dict[str, Any]) -> Path:
    return Path(config["run_dir"]) / "mdof" / "validation"


def _standardized_matrix(
    rows: list[dict[str, Any]],
    names: tuple[str, ...],
) -> np.ndarray:
    values = np.asarray(
        [
            [
                float(row[name])
                if row.get(name) is not None
                and math.isfinite(float(row[name]))
                else math.nan
                for name in names
            ]
            for row in rows
        ],
        dtype=float,
    )
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        fill = (
            float(np.median(values[finite, column]))
            if np.any(finite)
            else 0.0
        )
        values[~finite, column] = fill
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(
        values, 0.25, axis=0
    )
    standard_deviation = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-12, scale, standard_deviation)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (values - center) / scale


def deterministic_maximin(
    rows: list[dict[str, Any]],
    *,
    count: int,
    identifier: str,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return a deterministic space-filling subset without response labels."""
    if count < 1 or len(rows) < count:
        raise ValueError(
            f"Maximin selection needs at least {count} candidates; "
            f"received {len(rows)}"
        )
    ordered = sorted(rows, key=lambda row: str(row[identifier]))
    features = _standardized_matrix(ordered, feature_names)
    centroid = np.mean(features, axis=0)
    distance_to_centroid = np.linalg.norm(features - centroid, axis=1)
    first = max(
        range(len(ordered)),
        key=lambda index: (
            float(distance_to_centroid[index]),
            str(ordered[index][identifier]),
        ),
    )
    chosen = [first]
    while len(chosen) < count:
        remaining = [
            index for index in range(len(ordered)) if index not in chosen
        ]
        next_index = max(
            remaining,
            key=lambda index: (
                float(
                    min(
                        np.linalg.norm(features[index] - features[prior])
                        for prior in chosen
                    )
                ),
                str(ordered[index][identifier]),
            ),
        )
        chosen.append(next_index)
    return [ordered[index] for index in chosen]


def _candidate_buildings(
    config: dict[str, Any],
    building_ids: list[str] | None,
) -> list[dict[str, Any]]:
    parameters: list[Any] = []
    where = "b.valid=1 AND b.selected=1 AND s.valid=1"
    if building_ids:
        marks = ",".join("?" for _ in building_ids)
        where += f" AND b.building_id IN ({marks})"
        parameters.extend(building_ids)
    with connect(config["database_path"]) as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT b.*, s.t1_s, s.vy_kn, s.dy_m, s.vc_kn, "
                "s.dc_m, s.vu_kn, s.du_m, "
                "s.x_mode_effective_mass_ratio, "
                "s.analysis_signature AS spo_analysis_signature "
                "FROM building_catalog b "
                "JOIN spo_features s USING(building_id) "
                f"WHERE {where} ORDER BY b.building_id",
                parameters,
            )
        ]


def _candidate_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    with connect(config["database_path"]) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM ground_motion_catalog "
                "WHERE valid=1 AND source_set NOT LIKE 'PWSA%' "
                "ORDER BY pair_id"
            )
        ]
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(
            row.get("physical_pair_hash")
            or f"{row.get('sha256_x')}:{row.get('sha256_y')}"
        )
        unique.setdefault(identity, row)
    return list(unique.values())


def prepare_validation_manifest(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
) -> dict[str, Any]:
    initialize(config["database_path"])
    settings = validation_config(config)
    count = int(settings["building_count"])
    development_count = int(settings["development_count"])
    holdout_count = int(settings["holdout_count"])
    if development_count + holdout_count != count:
        raise ValueError(
            "MDOF validation development+holdout counts must equal total"
        )
    buildings = deterministic_maximin(
        _candidate_buildings(config, building_ids),
        count=count,
        identifier="building_id",
        feature_names=(
            "number_of_bays",
            "floor_mass_kn_s2_m",
            "t1_s",
            "vy_kn",
            "scwb_strength_ratio",
            "beam_bars_per_face",
            "column_bar_count",
            "axial_ratio",
        ),
    )
    pairs = deterministic_maximin(
        _candidate_pairs(config),
        count=int(settings["record_pair_count"]),
        identifier="pair_id",
        feature_names=(
            "conditioning_period_s",
            "duration_s",
            "pga_x_g",
            "pga_y_g",
            "dt_s",
        ),
    )
    building_tokens = [str(row["building_id"]) for row in buildings]
    pair_tokens = [str(row["pair_id"]) for row in pairs]
    payload = {
        "schema_version": VALIDATION_MANIFEST_SCHEMA_VERSION,
        "selection_method": (
            "deterministic_maximin_label_blind_geometry_mass_modal_"
            "strength_scwb_reinforcement_axial_and_gm_metadata"
        ),
        "development_buildings": building_tokens[:development_count],
        "holdout_buildings": building_tokens[development_count:],
        "record_pairs": pair_tokens,
        "dense_ida_holdout_buildings": building_tokens[
            development_count : development_count
            + int(settings["dense_ida_holdout_count"])
        ],
        "building_candidate_hash": stable_hash(
            [
                {
                    key: row.get(key)
                    for key in (
                        "building_id",
                        "model_hash",
                        "number_of_bays",
                        "floor_mass_kn_s2_m",
                        "t1_s",
                        "vy_kn",
                        "scwb_strength_ratio",
                        "beam_bars_per_face",
                        "column_bar_count",
                        "axial_ratio",
                    )
                }
                for row in _candidate_buildings(config, building_ids)
            ]
        ),
        "pair_candidate_hash": stable_hash(
            [
                {
                    key: row.get(key)
                    for key in (
                        "pair_id",
                        "physical_pair_hash",
                        "sha256_x",
                        "sha256_y",
                        "conditioning_period_s",
                        "duration_s",
                        "pga_x_g",
                        "pga_y_g",
                    )
                }
                for row in _candidate_pairs(config)
            ]
        ),
    }
    payload["manifest_hash"] = stable_hash(payload)
    path = _validation_root(config) / "manifest.json"
    atomic_write_json(path, payload)
    return payload


def _directional_reduced_modes(
    artifact: MDOFArtifact,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    reduction = (
        artifact.direction_x if direction == "X" else artifact.direction_y
    )
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if direction == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    net = np.asarray(
        [
            spring.initial_stiffness_kn_m
            + spring.geometric_stiffness_kn_m
            for spring in reduction.springs
        ],
        dtype=float,
    )
    eigenvalues, shapes = eigh(
        shear_stiffness_matrix(net), np.diag(masses)
    )
    periods = 2.0 * math.pi / np.sqrt(eigenvalues)
    normalized = shapes.T.copy()
    for index in range(normalized.shape[0]):
        scale = float(np.max(np.abs(normalized[index])))
        normalized[index] /= max(scale, np.finfo(float).eps)
        if float(normalized[index] @ masses) < 0.0:
            normalized[index] *= -1.0
    return periods, normalized


def _artifact_static_rows(
    artifact: MDOFArtifact,
) -> list[dict[str, Any]]:
    rows = []
    for label, reduction in (
        ("X", artifact.direction_x),
        ("Y", artifact.direction_y),
    ):
        periods, shapes = _directional_reduced_modes(artifact, label)
        target_periods = np.asarray(reduction.target_periods_s, dtype=float)
        target_shapes = np.asarray(reduction.target_mode_shapes, dtype=float)
        mode_count = min(len(target_periods), len(periods), 3)
        for mode in range(mode_count):
            rows.append(
                {
                    "building_id": artifact.building_id,
                    "direction": label,
                    "mode": mode + 1,
                    "period_error": float(
                        abs(periods[mode] - target_periods[mode])
                        / target_periods[mode]
                    ),
                    "mac": modal_assurance_criterion(
                        shapes[mode], target_shapes[mode]
                    ),
                    "effective_mass_ratio": float(
                        reduction.target_effective_mass_ratios[mode]
                    ),
                    "elastic_drift_nrmse": float(
                        reduction.elastic_drift_nrmse
                    ),
                    "elastic_core_drift_nrmse": float(
                        reduction.elastic_core_drift_nrmse
                    ),
                    "elastic_higher_mode_drift_nrmse": float(
                        reduction.elastic_higher_mode_drift_nrmse
                    ),
                    "elastic_fit_selection": (
                        reduction.elastic_fit_selection
                    ),
                    "backbone_nrmse": float(
                        reduction.nonlinear_backbone_nrmse
                    ),
                    "applicability_flags": list(
                        artifact.applicability_flags
                    ),
                    "artifact_hash": artifact.artifact_hash,
                }
            )
    return rows


def _case_token(target_im_g: float) -> str:
    return f"{target_im_g:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def _case_paths(
    config: dict[str, Any],
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> dict[str, Path]:
    root = (
        _validation_root(config)
        / "cases"
        / building_id
        / pair_id
        / f"im_{_case_token(target_im_g)}"
    )
    return {
        "full3d_json": root / "full3d.json",
        "full3d_npz": root / "full3d.npz",
        "mdof_json": root / "mdof.json",
        "mdof_npz": root / "mdof.npz",
    }


def _case_signature(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    artifact: MDOFArtifact,
    config: dict[str, Any],
    *,
    engine: str,
) -> str:
    payload = {
        "schema": VALIDATION_CASE_SCHEMA_VERSION,
        "analysis_engine": engine,
        "recovery_policy": NLTHA_RECOVERY_POLICY_VERSION,
        "model_hash": building.get("model_hash"),
        "pair_id": pair["pair_id"],
        "sha256_x": pair.get("sha256_x"),
        "sha256_y": pair.get("sha256_y"),
        "target_im_g": float(target_im_g),
        "model_config": config["model"],
    }
    if engine == "mdof":
        payload.update(
            {
                "analysis_schema": MDOF_ANALYSIS_SCHEMA_VERSION,
                "artifact_hash": artifact.artifact_hash,
                "damping_ratio": mdof_config(config)["damping_ratio"],
            }
        )
    elif engine == "full3d":
        payload["analysis_schema"] = NLTHA_ANALYSIS_SCHEMA_VERSION
    else:
        raise ValueError("Validation engine must be full3d or mdof")
    return stable_hash(payload)


def _write_case_result(
    json_path: Path,
    history_path: Path,
    result: dict[str, Any],
    *,
    signature: str,
    engine: str,
) -> None:
    history = result.pop("_history_arrays", None)
    if history is None:
        raise RuntimeError(
            f"{engine} validation run did not return response histories"
        )
    atomic_write_history(history_path, history)
    summary = (
        result.get("full3d_summary")
        if engine == "full3d"
        else result.get("mdof_summary")
    )
    atomic_write_json(
        json_path,
        {
            "schema_version": VALIDATION_CASE_SCHEMA_VERSION,
            "case_signature": signature,
            "analysis_engine": engine,
            "status": result["status"],
            "max_midr": result["max_midr"],
            "runtime_s": result.get(
                "runtime_s_solver", result.get("runtime_s")
            ),
            "recovery_level": result.get("recovery_level"),
            "completed_fraction": result.get("completed_fraction"),
            "dynamic_instability_basis": result.get(
                "dynamic_instability_basis"
            ),
            "summary": summary,
            "history_path": str(history_path.absolute()),
        },
    )


def _case_is_current(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        history = Path(str(payload["history_path"]))
        return (
            payload.get("case_signature") == signature
            and history.is_file()
        )
    except Exception:
        return False


def _run_validation_case(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    resume: bool,
) -> None:
    artifact = load_artifact(
        config, str(building["building_id"]), require_certified=False
    )
    paths = _case_paths(
        config,
        str(building["building_id"]),
        str(pair["pair_id"]),
        target_im_g,
    )
    _unscaled, scale_factor = _checked_scale_factor(
        pair, float(building["t1_s"]), target_im_g, config
    )
    for engine in ("full3d", "mdof"):
        signature = _case_signature(
            building,
            pair,
            target_im_g,
            artifact,
            config,
            engine=engine,
        )
        json_path = paths[f"{engine}_json"]
        history_path = paths[f"{engine}_npz"]
        if resume and _case_is_current(json_path, signature):
            continue
        local = deepcopy(config)
        local["ida"]["analysis_engine"] = engine
        local["ida"]["component_response_recording"]["enabled"] = False
        local["ida"]["component_response_recording"][
            "backfill_existing_checkpoints"
        ] = False
        local["mdof"]["allow_uncertified_diagnostic"] = True
        local["mdof"]["validation"]["capture_full3d_histories"] = True
        started = time.perf_counter()
        result = _single_nltha_local_recovery(
            building,
            pair,
            target_im_g,
            local,
            scale_factor=scale_factor,
        )
        result.setdefault("runtime_s", time.perf_counter() - started)
        _write_case_result(
            json_path,
            history_path,
            result,
            signature=signature,
            engine=engine,
        )


def _load_manifest_rows(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    building_ids = (
        list(manifest["development_buildings"])
        + list(manifest["holdout_buildings"])
    )
    pair_ids = list(manifest["record_pairs"])
    building_marks = ",".join("?" for _ in building_ids)
    pair_marks = ",".join("?" for _ in pair_ids)
    with connect(config["database_path"]) as connection:
        buildings = {
            str(row["building_id"]): dict(row)
            for row in connection.execute(
                "SELECT b.*, s.t1_s, s.vy_kn, s.dy_m, s.vc_kn, "
                "s.dc_m, s.vu_kn, s.du_m, "
                "s.x_mode_effective_mass_ratio "
                "FROM building_catalog b JOIN spo_features s "
                "USING(building_id) "
                f"WHERE b.building_id IN ({building_marks})",
                building_ids,
            )
        }
        pairs = {
            str(row["pair_id"]): dict(row)
            for row in connection.execute(
                "SELECT * FROM ground_motion_catalog "
                f"WHERE pair_id IN ({pair_marks})",
                pair_ids,
            )
        }
    return buildings, pairs


def execute_validation_cases(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    buildings, pairs = _load_manifest_rows(config, manifest)
    failures = []
    completed = 0
    for building_id in (
        list(manifest["development_buildings"])
        + list(manifest["holdout_buildings"])
    ):
        building = buildings[building_id]
        prediction = heuristic_capacity_prediction(
            building, config["ida"]["controller"]
        )
        targets = (
            0.5 * prediction["IO"],
            prediction["IO"],
            prediction["LS"],
            prediction["CP"],
            1.25 * prediction["CP"],
        )
        for pair_id in manifest["record_pairs"]:
            for target in targets:
                try:
                    _run_validation_case(
                        building,
                        pairs[pair_id],
                        float(target),
                        config,
                        resume=resume,
                    )
                    completed += 1
                except Exception as exc:
                    failures.append(
                        {
                            "building_id": building_id,
                            "pair_id": pair_id,
                            "target_im_g": float(target),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
    return {
        "completed_case_count": completed,
        "failure_count": len(failures),
        "failures": failures,
    }


def execute_dense_ida_validation(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run complete Full-3D and MDOF IDA curves for blind holdouts."""
    from .ida import _persist_curve, run_ida_curve

    buildings, pairs = _load_manifest_rows(config, manifest)
    failures = []
    curves = []
    for building_id in manifest["dense_ida_holdout_buildings"]:
        for pair_id in manifest["record_pairs"]:
            for engine in ("full3d", "mdof"):
                local = deepcopy(config)
                local["ida"]["analysis_engine"] = engine
                if engine == "mdof":
                    local.setdefault("mdof", {})[
                        "allow_uncertified_diagnostic"
                    ] = True
                try:
                    curve = run_ida_curve(
                        buildings[building_id],
                        pairs[pair_id],
                        local,
                        resume=resume,
                    )
                    _persist_curve(local, curve)
                    curves.append(
                        {
                            "building_id": building_id,
                            "pair_id": pair_id,
                            "analysis_engine": engine,
                            "valid": bool(curve["valid"]),
                            "run_count": len(curve["runs"]),
                            "capacity_count": len(curve["capacities"]),
                            "message": curve["message"],
                        }
                    )
                    if not curve["valid"]:
                        failures.append(
                            {
                                "building_id": building_id,
                                "pair_id": pair_id,
                                "analysis_engine": engine,
                                "error_type": "InvalidIDACurve",
                                "message": curve["message"],
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "building_id": building_id,
                            "pair_id": pair_id,
                            "analysis_engine": engine,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
    result = {
        "completed_curve_count": len(curves),
        "expected_curve_count": (
            len(manifest["dense_ida_holdout_buildings"])
            * len(manifest["record_pairs"])
            * 2
        ),
        "failure_count": len(failures),
        "failures": failures,
        "curves": curves,
    }
    atomic_write_json(
        _validation_root(config) / "dense_ida_execution.json", result
    )
    return result


def _relative_error(observed: float, reference: float) -> float:
    return abs(float(observed) - float(reference)) / max(
        abs(float(reference)), np.finfo(float).eps
    )


def _profile_nrmse(
    observed_x: list[float],
    observed_y: list[float],
    reference_x: list[float],
    reference_y: list[float],
) -> float:
    observed = np.concatenate((observed_x, observed_y))
    reference = np.concatenate((reference_x, reference_y))
    return float(
        np.linalg.norm(observed - reference)
        / max(np.linalg.norm(reference), np.finfo(float).eps)
    )


def _history_correlations(
    full_path: Path,
    mdof_path: Path,
) -> list[float]:
    with np.load(full_path) as full, np.load(mdof_path) as reduced:
        full_time = np.asarray(full["time_s"], dtype=float)
        reduced_time = np.asarray(reduced["time_s"], dtype=float)
        correlations = []
        for key in ("idr_x", "idr_y"):
            reference = np.asarray(full[key], dtype=float)
            observed = np.asarray(reduced[key], dtype=float)
            for story in range(reference.shape[1]):
                interpolated = np.interp(
                    full_time, reduced_time, observed[:, story]
                )
                if (
                    np.std(reference[:, story]) <= 1.0e-12
                    or np.std(interpolated) <= 1.0e-12
                ):
                    continue
                correlations.append(
                    float(
                        np.corrcoef(reference[:, story], interpolated)[0, 1]
                    )
                )
    return correlations


def _dynamic_rows(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    buildings, _pairs = _load_manifest_rows(config, manifest)
    rows = []
    for building_id in (
        list(manifest["development_buildings"])
        + list(manifest["holdout_buildings"])
    ):
        prediction = heuristic_capacity_prediction(
            buildings[building_id], config["ida"]["controller"]
        )
        targets = (
            0.5 * prediction["IO"],
            prediction["IO"],
            prediction["LS"],
            prediction["CP"],
            1.25 * prediction["CP"],
        )
        for pair_id in manifest["record_pairs"]:
            for target in targets:
                paths = _case_paths(
                    config, building_id, pair_id, float(target)
                )
                if not all(path.is_file() for path in paths.values()):
                    continue
                full = read_json(paths["full3d_json"])
                reduced = read_json(paths["mdof_json"])
                full_summary = full.get("summary") or {}
                reduced_summary = reduced.get("summary") or {}
                if not full_summary or not reduced_summary:
                    continue
                full_midr = float(full_summary["max_midr"])
                reduced_midr = float(reduced_summary["max_midr"])
                noncollapse = (
                    full["status"] == "success"
                    and reduced["status"] == "success"
                )
                correlations = (
                    _history_correlations(
                        paths["full3d_npz"], paths["mdof_npz"]
                    )
                    if noncollapse
                    else []
                )
                roof_errors = [
                    _relative_error(
                        reduced_summary[f"peak_roof_displacement_{axis}_m"],
                        full_summary[f"peak_roof_displacement_{axis}_m"],
                    )
                    for axis in ("x", "y")
                ]
                base_errors = []
                for axis in ("x", "y"):
                    full_value = full_summary.get(
                        f"peak_base_shear_{axis}_kn"
                    )
                    reduced_value = reduced_summary.get(
                        f"peak_base_shear_{axis}_kn"
                    )
                    if full_value is not None and reduced_value is not None:
                        base_errors.append(
                            _relative_error(reduced_value, full_value)
                        )
                rows.append(
                    {
                        "building_id": building_id,
                        "pair_id": pair_id,
                        "target_im_g": float(target),
                        "midr_abs_log_error": abs(
                            math.log(
                                max(reduced_midr, 1.0e-12)
                                / max(full_midr, 1.0e-12)
                            )
                        ),
                        "drift_profile_nrmse": _profile_nrmse(
                            reduced_summary["peak_midr_profile_x"],
                            reduced_summary["peak_midr_profile_y"],
                            full_summary["peak_midr_profile_x"],
                            full_summary["peak_midr_profile_y"],
                        ),
                        "critical_story_difference": abs(
                            int(reduced_summary["critical_story"])
                            - int(full_summary["critical_story"])
                        ),
                        "history_correlations": correlations,
                        "roof_errors": roof_errors,
                        "base_errors": base_errors,
                        "speedup": float(full["runtime_s"])
                        / max(float(reduced["runtime_s"]), 1.0e-12),
                        "full3d_runtime_s": float(full["runtime_s"]),
                        "mdof_runtime_s": float(reduced["runtime_s"]),
                        "noncollapse": noncollapse,
                    }
                )
    return rows


def _capacity_and_fragility_metrics(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    buildings = list(manifest["dense_ida_holdout_buildings"])
    pairs = list(manifest["record_pairs"])
    if not buildings or not pairs:
        return {"complete": False}
    building_marks = ",".join("?" for _ in buildings)
    pair_marks = ",".join("?" for _ in pairs)
    parameters = buildings + pairs
    all_buildings, all_pairs = _load_manifest_rows(config, manifest)
    engine_configs = {}
    for engine in ("full3d", "mdof"):
        local = deepcopy(config)
        local["ida"]["analysis_engine"] = engine
        if engine == "mdof":
            local.setdefault("mdof", {})[
                "allow_uncertified_diagnostic"
            ] = True
        engine_configs[engine] = local
    tables = {
        "full3d": "ida_capacities",
        "mdof": "mdof_ida_capacities",
    }
    current_rows: dict[str, list[dict[str, Any]]] = {}
    with connect(config["database_path"]) as connection:
        for engine, table in tables.items():
            candidates = [
                dict(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE building_id IN ({building_marks}) "
                    f"AND pair_id IN ({pair_marks})",
                    parameters,
                )
            ]
            current_rows[engine] = [
                row
                for row in candidates
                if row.get("analysis_signature")
                == ida_curve_analysis_signature(
                    all_buildings[str(row["building_id"])],
                    all_pairs[str(row["pair_id"])],
                    engine_configs[engine],
                )
            ]
    full_cp = {
        (str(row["building_id"]), str(row["pair_id"])): row
        for row in current_rows["full3d"]
        if row["limit_state"] == "CP" and not bool(row["censored"])
    }
    mdof_cp = {
        (str(row["building_id"]), str(row["pair_id"])): row
        for row in current_rows["mdof"]
        if row["limit_state"] == "CP" and not bool(row["censored"])
    }
    shared_cp = sorted(set(full_cp) & set(mdof_cp))
    capacity_rows = [
        {
            "building_id": key[0],
            "pair_id": key[1],
            "full_capacity": float(full_cp[key]["capacity_im_g"]),
            "mdof_capacity": float(mdof_cp[key]["capacity_im_g"]),
        }
        for key in shared_cp
    ]
    fragility_rows = []
    fragility_errors = []
    for building_id in buildings:
        fits = {}
        for engine in ("full3d", "mdof"):
            rows = [
                row
                for row in current_rows[engine]
                if str(row["building_id"]) == building_id
            ]
            try:
                fits[engine] = fit_building_fragility(
                    building_id,
                    rows,
                    bootstrap_count=int(
                        validation_config(config)[
                            "fragility_bootstrap_count"
                        ]
                    ),
                    seed=int(
                        stable_hash(
                            {
                                "building_id": building_id,
                                "engine": engine,
                                "validation": "dense_ida",
                            }
                        )[:8],
                        16,
                    ),
                    expected_pair_count=len(pairs),
                )
            except Exception as exc:
                fragility_errors.append(
                    {
                        "building_id": building_id,
                        "analysis_engine": engine,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if (
            set(fits) == {"full3d", "mdof"}
            and bool(fits["full3d"]["valid"])
            and bool(fits["mdof"]["valid"])
        ):
            fragility_rows.append(
                {
                    "building_id": building_id,
                    "full_theta": fits["full3d"]["theta_cp_g"],
                    "full_beta": fits["full3d"]["beta_cp"],
                    "mdof_theta": fits["mdof"]["theta_cp_g"],
                    "mdof_beta": fits["mdof"]["beta_cp"],
                }
            )
    expected_capacities = len(buildings) * len(pairs)
    ratios = [
        float(row["mdof_capacity"]) / float(row["full_capacity"])
        for row in capacity_rows
    ]
    theta_errors = [
        _relative_error(row["mdof_theta"], row["full_theta"])
        for row in fragility_rows
    ]
    beta_errors = [
        abs(float(row["mdof_beta"]) - float(row["full_beta"]))
        for row in fragility_rows
    ]
    return {
        "complete": (
            len(capacity_rows) >= expected_capacities
            and len(fragility_rows) >= len(buildings)
        ),
        "capacity_pair_count": len(capacity_rows),
        "expected_capacity_pair_count": expected_capacities,
        "fragility_building_count": len(fragility_rows),
        "expected_fragility_building_count": len(buildings),
        "collapse_capacity_median_ratio": (
            float(np.median(ratios)) if ratios else None
        ),
        "fragility_median_error": (
            float(np.median(theta_errors)) if theta_errors else None
        ),
        "fragility_dispersion_abs_error": (
            float(np.median(beta_errors)) if beta_errors else None
        ),
        "fragility_errors": fragility_errors,
    }


def _gate(name: str, value: Any, passed: bool, threshold: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
    }


def evaluate_validation(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    settings = validation_config(config)
    static_rows = []
    missing_artifacts = []
    building_ids = (
        list(manifest["development_buildings"])
        + list(manifest["holdout_buildings"])
    )
    for building_id in building_ids:
        try:
            static_rows.extend(
                _artifact_static_rows(
                    load_artifact(config, building_id, require_certified=False)
                )
            )
        except Exception as exc:
            missing_artifacts.append(
                f"{building_id}: {type(exc).__name__}: {exc}"
            )
    first = [row for row in static_rows if row["mode"] == 1]
    higher = [row for row in static_rows if row["mode"] > 1]
    backbone = [
        row["backbone_nrmse"]
        for row in static_rows
        if row["mode"] == 1
    ]
    dynamic_rows = _dynamic_rows(config, manifest)
    expected_dynamic = (
        len(building_ids) * len(manifest["record_pairs"]) * 5
    )
    midr_errors = [row["midr_abs_log_error"] for row in dynamic_rows]
    profile_errors = [row["drift_profile_nrmse"] for row in dynamic_rows]
    critical = [
        row["critical_story_difference"] for row in dynamic_rows
    ]
    correlations = [
        value
        for row in dynamic_rows
        for value in row["history_correlations"]
    ]
    roof_base_errors = [
        value
        for row in dynamic_rows
        for value in row["roof_errors"] + row["base_errors"]
    ]
    speedups = [row["speedup"] for row in dynamic_rows]
    break_even_counts = []
    for building_id in building_ids:
        building_cases = [
            row for row in dynamic_rows if row["building_id"] == building_id
        ]
        if not building_cases:
            continue
        artifact = load_artifact(
            config, building_id, require_certified=False
        )
        calibration_runtime = artifact.calibration_metrics.get(
            "calibration_runtime_s"
        )
        saved_per_run = float(
            np.median(
                [
                    row["full3d_runtime_s"] - row["mdof_runtime_s"]
                    for row in building_cases
                ]
            )
        )
        if calibration_runtime is not None and saved_per_run > 0.0:
            break_even_counts.append(
                float(calibration_runtime) / saved_per_run
            )
    capacity = _capacity_and_fragility_metrics(config, manifest)
    first_period_max = max(
        (row["period_error"] for row in first), default=math.inf
    )
    first_mac_min = min((row["mac"] for row in first), default=-math.inf)
    higher_period_max = max(
        (row["period_error"] for row in higher), default=math.inf
    )
    directional_mode_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in static_rows:
        directional_mode_rows.setdefault(
            (str(row["building_id"]), str(row["direction"])), []
        ).append(row)
    weighted_period_errors = []
    for rows in directional_mode_rows.values():
        masses = np.asarray(
            [float(row["effective_mass_ratio"]) for row in rows],
            dtype=float,
        )
        errors = np.asarray(
            [float(row["period_error"]) for row in rows], dtype=float
        )
        if float(np.sum(masses)) > 0.0:
            weighted_period_errors.append(
                float(np.sum(masses * errors) / np.sum(masses))
            )
    weighted_period_max = max(
        weighted_period_errors, default=math.inf
    )
    higher_mac_min = min(
        (row["mac"] for row in higher), default=-math.inf
    )
    elastic_max = max(
        (row["elastic_drift_nrmse"] for row in first), default=math.inf
    )
    elastic_core_max = max(
        (row["elastic_core_drift_nrmse"] for row in first),
        default=math.inf,
    )
    backbone_median = (
        float(np.median(backbone)) if backbone else math.inf
    )
    backbone_p90 = (
        float(np.quantile(backbone, 0.90)) if backbone else math.inf
    )
    applicability_clear = bool(
        static_rows
        and all(
            "cyclic-calibration-pending"
            not in row["applicability_flags"]
            for row in first
        )
    )
    gates = [
        _gate(
            "artifact_completeness",
            len(first),
            not missing_artifacts and len(first) == 2 * len(building_ids),
            f"{2 * len(building_ids)} first directional modes",
        ),
        _gate(
            "cyclic_calibration_complete",
            applicability_clear,
            applicability_clear,
            "no cyclic-calibration-pending flags",
        ),
        _gate(
            "first_mode_period_error",
            first_period_max,
            first_period_max <= float(settings["period_error_first_max"]),
            f"<= {settings['period_error_first_max']}",
        ),
        _gate(
            "first_mode_mac",
            first_mac_min,
            first_mac_min >= float(settings["first_mode_mac_min"]),
            f">= {settings['first_mode_mac_min']}",
        ),
        _gate(
            "three_mode_mass_weighted_period_error",
            weighted_period_max,
            bool(weighted_period_errors)
            and weighted_period_max
            <= float(
                settings[
                    "three_mode_mass_weighted_period_error_max"
                ]
            ),
            "<= "
            f"{settings['three_mode_mass_weighted_period_error_max']}",
        ),
        _gate(
            "higher_mode_mac",
            higher_mac_min,
            bool(higher)
            and higher_mac_min >= float(settings["higher_mode_mac_min"]),
            f">= {settings['higher_mode_mac_min']}",
        ),
        _gate(
            "elastic_core_pattern_drift_nrmse",
            elastic_core_max,
            elastic_core_max
            <= float(settings["elastic_drift_nrmse_max"]),
            f"<= {settings['elastic_drift_nrmse_max']}",
        ),
        _gate(
            "elastic_multi_pattern_drift_nrmse",
            elastic_max,
            elastic_max <= float(settings["elastic_drift_nrmse_max"]),
            f"<= {settings['elastic_drift_nrmse_max']}",
        ),
        _gate(
            "backbone_nrmse_median",
            backbone_median,
            backbone_median
            <= float(settings["backbone_nrmse_median_max"]),
            f"<= {settings['backbone_nrmse_median_max']}",
        ),
        _gate(
            "backbone_nrmse_p90",
            backbone_p90,
            backbone_p90
            <= float(settings["backbone_nrmse_p90_max"]),
            f"<= {settings['backbone_nrmse_p90_max']}",
        ),
        _gate(
            "dynamic_case_completeness",
            len(dynamic_rows),
            len(dynamic_rows) == expected_dynamic,
            f"{expected_dynamic} paired cases",
        ),
    ]
    if dynamic_rows:
        midr_median = float(np.median(midr_errors))
        midr_p90 = float(np.quantile(midr_errors, 0.90))
        profile_median = float(np.median(profile_errors))
        critical_exact = float(np.mean(np.asarray(critical) == 0))
        critical_within_one = float(np.mean(np.asarray(critical) <= 1))
        correlation_min = min(correlations, default=-math.inf)
        roof_base_median = float(
            np.median(roof_base_errors)
            if roof_base_errors
            else math.inf
        )
        speedup_median = float(np.median(speedups))
        break_even_max = max(break_even_counts, default=math.inf)
    else:
        midr_median = math.inf
        midr_p90 = math.inf
        profile_median = math.inf
        critical_exact = -math.inf
        critical_within_one = -math.inf
        correlation_min = -math.inf
        roof_base_median = math.inf
        speedup_median = -math.inf
        break_even_max = math.inf
    gates.extend(
        [
            _gate(
                "midr_median_abs_log_error",
                midr_median,
                midr_median
                <= float(settings["midr_median_abs_log_error_max"]),
                f"<= {settings['midr_median_abs_log_error_max']}",
            ),
            _gate(
                "midr_p90_abs_log_error",
                midr_p90,
                midr_p90
                <= float(settings["midr_p90_abs_log_error_max"]),
                f"<= {settings['midr_p90_abs_log_error_max']}",
            ),
            _gate(
                "peak_drift_profile_nrmse_median",
                profile_median,
                profile_median
                <= float(settings["drift_profile_nrmse_median_max"]),
                f"<= {settings['drift_profile_nrmse_median_max']}",
            ),
            _gate(
                "critical_story_exact",
                critical_exact,
                critical_exact
                >= float(settings["critical_story_exact_min"]),
                f">= {settings['critical_story_exact_min']}",
            ),
            _gate(
                "critical_story_within_one",
                critical_within_one,
                critical_within_one
                >= float(settings["critical_story_within_one_min"]),
                f">= {settings['critical_story_within_one_min']}",
            ),
            _gate(
                "noncollapse_history_correlation_min",
                correlation_min,
                bool(correlations)
                and correlation_min
                >= float(settings["history_correlation_min"]),
                f">= {settings['history_correlation_min']}",
            ),
            _gate(
                "roof_and_base_shear_median_error",
                roof_base_median,
                roof_base_median
                <= float(settings["roof_base_median_error_max"]),
                f"<= {settings['roof_base_median_error_max']}",
            ),
            _gate(
                "median_nltha_speedup",
                speedup_median,
                speedup_median >= float(settings["minimum_nltha_speedup"]),
                f">= {settings['minimum_nltha_speedup']}",
            ),
            _gate(
                "calibration_break_even_nltha_count",
                break_even_max,
                bool(break_even_counts)
                and break_even_max
                <= float(settings["maximum_break_even_nltha_count"]),
                f"<= {settings['maximum_break_even_nltha_count']}",
            ),
            _gate(
                "dense_ida_and_fragility_completeness",
                capacity.get("complete", False),
                bool(capacity.get("complete", False)),
                "3 holdout buildings x 4 pairs plus fragility rows",
            ),
        ]
    )
    ratio = capacity.get("collapse_capacity_median_ratio")
    theta_error = capacity.get("fragility_median_error")
    beta_error = capacity.get("fragility_dispersion_abs_error")
    gates.extend(
        [
            _gate(
                "collapse_capacity_median_ratio",
                ratio,
                ratio is not None
                and float(settings["collapse_capacity_ratio_min"])
                <= ratio
                <= float(settings["collapse_capacity_ratio_max"]),
                (
                    f"{settings['collapse_capacity_ratio_min']}.."
                    f"{settings['collapse_capacity_ratio_max']}"
                ),
            ),
            _gate(
                "fragility_median_error",
                theta_error,
                theta_error is not None
                and theta_error
                <= float(settings["fragility_median_error_max"]),
                f"<= {settings['fragility_median_error_max']}",
            ),
            _gate(
                "fragility_dispersion_abs_error",
                beta_error,
                beta_error is not None
                and beta_error
                <= float(settings["fragility_dispersion_abs_error_max"]),
                f"<= {settings['fragility_dispersion_abs_error_max']}",
            ),
        ]
    )
    return {
        "static_rows": static_rows,
        "dynamic_case_count": len(dynamic_rows),
        "expected_dynamic_case_count": expected_dynamic,
        "capacity_and_fragility": capacity,
        "modal_diagnostics": {
            "maximum_individual_higher_mode_period_error": (
                higher_period_max
            ),
            "maximum_three_mode_mass_weighted_period_error": (
                weighted_period_max
            ),
        },
        "missing_artifacts": missing_artifacts,
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
    }


def _certify_artifacts(
    config: dict[str, Any],
    building_ids: list[str],
    report_path: Path,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for building_id in building_ids:
        artifact = load_artifact(
            config, building_id, require_certified=False
        )
        certified = replace(
            artifact,
            certified=True,
            certification_path=str(report_path.absolute()),
        ).with_hash()
        path = save_artifact(certified, config)
        rows.append(
            {
                "building_id": building_id,
                "artifact_path": str(path.absolute()),
                "artifact_hash": certified.artifact_hash,
                "source_signature": certified.source_signature,
                "schema_version": certified.schema_version,
                "story_count": certified.story_count,
                "certified": 1,
                "certification_path": str(report_path.absolute()),
                "applicability_flags_json": json.dumps(
                    certified.applicability_flags, sort_keys=True
                ),
                "calibration_metrics_json": json.dumps(
                    certified.calibration_metrics, sort_keys=True
                ),
                "created_utc": now,
            }
        )
    with transaction(config["database_path"]) as connection:
        upsert_many(
            connection, "mdof_artifacts", rows, ("building_id",)
        )


def validate_mdof(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Prepare/evaluate validation and optionally execute the costly cases."""
    manifest = prepare_validation_manifest(
        config, building_ids=building_ids
    )
    all_buildings = (
        list(manifest["development_buildings"])
        + list(manifest["holdout_buildings"])
    )
    if execute:
        from .mdof_calibration import build_artifacts
        from .mdof_cyclic import calibrate_cyclic_development

        artifact_calibration = build_artifacts(
            config,
            building_ids=all_buildings,
            resume=resume,
        )
        if artifact_calibration["failure_count"] == 0 and len(
            artifact_calibration["artifacts"]
        ) == len(all_buildings):
            cyclic = calibrate_cyclic_development(
                config, manifest, resume=resume
            )
        else:
            cyclic = {
                "completed": False,
                "failure_count": int(
                    artifact_calibration["failure_count"]
                ),
                "failures": list(artifact_calibration["failures"]),
                "message": (
                    "Cyclic calibration was not started because one or "
                    "more validation MDOF artifacts could not be built"
                ),
            }
        dynamic = (
            execute_validation_cases(config, manifest, resume=resume)
            if cyclic.get("completed") is True
            else {
                "completed_case_count": 0,
                "failure_count": 1,
                "failures": [
                    {
                        "message": (
                            "Dynamic comparisons were not started because "
                            "cyclic development calibration failed"
                        )
                    }
                ],
            }
        )
        expected_dynamic = (
            len(all_buildings)
            * len(manifest["record_pairs"])
            * 5
        )
        if (
            dynamic.get("failure_count") == 0
            and dynamic.get("completed_case_count") == expected_dynamic
        ):
            dense_ida = execute_dense_ida_validation(
                config, manifest, resume=resume
            )
        else:
            dense_ida = {
                "completed_curve_count": 0,
                "expected_curve_count": (
                    len(manifest["dense_ida_holdout_buildings"])
                    * len(manifest["record_pairs"])
                    * 2
                ),
                "failure_count": 1,
                "failures": [
                    {
                        "message": (
                            "Dense holdout IDA was not started because the "
                            "240 paired comparison cases were incomplete"
                        )
                    }
                ],
            }
        execution = {
            "artifact_calibration": artifact_calibration,
            "cyclic_calibration": cyclic,
            "dense_ida": dense_ida,
            **dynamic,
        }
    else:
        execution = {
            "completed_case_count": 0,
            "failure_count": 0,
            "failures": [],
            "message": "evaluation-only; pass execute=True to run cases",
        }
    metrics = evaluate_validation(config, manifest)
    certificate_id = stable_hash(
        {
            "schema": MDOF_CERTIFICATION_SCHEMA_VERSION,
            "manifest_hash": manifest["manifest_hash"],
            "metrics": metrics,
        }
    )
    report = {
        "certificate_id": certificate_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": MDOF_CERTIFICATION_SCHEMA_VERSION,
        "manifest": manifest,
        "execution": execution,
        "metrics": metrics,
        "passed": bool(metrics["passed"]),
        "production_ida_authorized": bool(metrics["passed"]),
        "failure_policy": (
            "Thresholds remain locked; failed or missing holdout evidence "
            "keeps MDOF diagnostic-only and requires Full 3D fallback."
        ),
    }
    report_path = _validation_root(config) / "certificate.json"
    atomic_write_json(report_path, report)
    if report["passed"]:
        _certify_artifacts(config, all_buildings, report_path)
    with transaction(config["database_path"]) as connection:
        upsert_many(
            connection,
            "mdof_validation_certificates",
            [
                {
                    "certificate_id": certificate_id,
                    "created_utc": report["created_utc"],
                    "schema_version": MDOF_CERTIFICATION_SCHEMA_VERSION,
                    "development_buildings_json": json.dumps(
                        manifest["development_buildings"], sort_keys=True
                    ),
                    "holdout_buildings_json": json.dumps(
                        manifest["holdout_buildings"], sort_keys=True
                    ),
                    "record_pairs_json": json.dumps(
                        manifest["record_pairs"], sort_keys=True
                    ),
                    "metrics_json": json.dumps(metrics, sort_keys=True),
                    "passed": int(report["passed"]),
                    "report_path": str(report_path.absolute()),
                }
            ],
            ("certificate_id",),
        )
    return {
        "passed": report["passed"],
        "production_ida_authorized": report[
            "production_ida_authorized"
        ],
        "certificate_id": certificate_id,
        "report_path": str(report_path.absolute()),
        "execution": execution,
        "failed_gates": [
            gate for gate in metrics["gates"] if not gate["passed"]
        ],
    }
