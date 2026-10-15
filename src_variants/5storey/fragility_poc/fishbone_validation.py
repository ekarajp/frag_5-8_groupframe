"""Full-3D versus Fishbone-v2 validation with exact Full-3D cache reuse.

This module deliberately treats the prior MDOF validation directory only as
an immutable Full-3D cache namespace.  MDOF results are neither loaded nor
evaluated.  Missing Full-3D cases may be filled without starting the removed
MDOF engine.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .active_ida import heuristic_capacity_prediction
from .config import load_config
from .db import connect
from .fishbone import FishboneArtifact, load_artifact
from .fishbone_dynamic import (
    load_cached_pair_result,
    pair_case_signature,
    result_path as fishbone_result_path,
    run_pair_nltha,
    save_pair_result,
)
from .ida import _checked_scale_factor, _single_nltha_local_recovery
from .io_utils import atomic_write_json, read_json, stable_hash
from .mdof_validation import (
    _case_is_current,
    _case_paths,
    _case_signature,
    _write_case_result,
)


FISHBONE_VALIDATION_CASE_SCHEMA_VERSION = (
    "full3d-fishbone-comparison-v1-exact-full3d-cache-reuse"
)


def validation_root(config: dict[str, Any]) -> Path:
    return Path(config["run_dir"]) / "fishbone_v2" / "validation"


def source_manifest_path(config: dict[str, Any]) -> Path:
    return Path(config["run_dir"]) / "mdof" / "validation" / "manifest.json"


def load_locked_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Copy the label-blind split without inheriting the old MDOF workflow."""

    source = read_json(source_manifest_path(config))
    payload = {
        "schema_version": "fishbone-validation-manifest-v1",
        "selection_method": source["selection_method"],
        "development_buildings": list(source["development_buildings"]),
        "holdout_buildings": list(source["holdout_buildings"]),
        "dense_ida_holdout_buildings": list(
            source["dense_ida_holdout_buildings"]
        ),
        "record_pairs": list(source["record_pairs"]),
        "source_manifest_hash": source["manifest_hash"],
        "source_manifest_path": str(source_manifest_path(config).absolute()),
        "full3d_cache_policy": (
            "reuse only when legacy Full3D case signature and history match; "
            "ignore all MDOF result files"
        ),
    }
    payload["manifest_hash"] = stable_hash(payload)
    target = validation_root(config) / "manifest.json"
    # This function is called by every isolated validation process.  The
    # locked split is immutable, so avoid replacing an identical file on the
    # NAS: concurrent readers otherwise race while renaming their temporary
    # files on Windows.  A mismatched file is still replaced fail-closed.
    if target.is_file():
        try:
            existing = read_json(target)
        except (OSError, ValueError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("manifest_hash") == payload["manifest_hash"]
        ):
            return existing
    try:
        atomic_write_json(target, payload)
    except PermissionError:
        # Another isolated process may have completed the same immutable
        # write between the existence check and os.replace().
        existing = read_json(target)
        if (
            isinstance(existing, dict)
            and existing.get("manifest_hash") == payload["manifest_hash"]
        ):
            return existing
        raise
    return payload


def load_rows(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    building_ids = list(
        dict.fromkeys(
            list(manifest["development_buildings"])
            + list(manifest["holdout_buildings"])
        )
    )
    pair_ids = list(dict.fromkeys(manifest["record_pairs"]))
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
    if set(buildings) != set(building_ids):
        raise KeyError("Locked validation buildings are incomplete")
    if set(pairs) != set(pair_ids):
        raise KeyError("Locked validation record pairs are incomplete")
    return buildings, pairs


def sparse_targets(
    building: dict[str, Any], config: dict[str, Any]
) -> tuple[float, ...]:
    prediction = heuristic_capacity_prediction(
        building, config["ida"]["controller"]
    )
    return (
        0.5 * float(prediction["IO"]),
        float(prediction["IO"]),
        float(prediction["LS"]),
        float(prediction["CP"]),
        1.25 * float(prediction["CP"]),
    )


def comparison_path(
    config: dict[str, Any],
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> Path:
    token = (
        f"{target_im_g:.8f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )
    return (
        validation_root(config)
        / "cases"
        / building_id
        / pair_id
        / f"im_{token}"
        / "comparison.json"
    )


def _full3d_signature(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
) -> str:
    # The Full-3D branch never reads ``artifact``.  Passing None preserves the
    # exact historical signature and therefore enables safe cache reuse.
    return _case_signature(
        building,
        pair,
        target_im_g,
        None,  # type: ignore[arg-type]
        config,
        engine="full3d",
    )


def full3d_cache_status(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    paths = _case_paths(
        config,
        str(building["building_id"]),
        str(pair["pair_id"]),
        target_im_g,
    )
    signature = _full3d_signature(building, pair, target_im_g, config)
    current = bool(
        _case_is_current(paths["full3d_json"], signature)
        and paths["full3d_npz"].is_file()
    )
    return {
        "current": current,
        "signature": signature,
        "metadata_path": paths["full3d_json"],
        "history_path": paths["full3d_npz"],
    }


def ensure_full3d_case(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    status = full3d_cache_status(
        building, pair, target_im_g, config
    )
    if status["current"]:
        payload = read_json(status["metadata_path"])
        _reject_zero_response_recorder_anomaly(payload, "full3d")
        return {**status, "cache_reused": True, "payload": payload}

    _unscaled, scale_factor = _checked_scale_factor(
        pair, float(building["t1_s"]), target_im_g, config
    )
    local = deepcopy(config)
    local["ida"]["analysis_engine"] = "full3d"
    local["ida"]["component_response_recording"]["enabled"] = False
    local["ida"]["component_response_recording"][
        "backfill_existing_checkpoints"
    ] = False
    # This flag requests only standard floor/story histories from Full 3D.
    # No MDOF solve is started.
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
        status["metadata_path"],
        status["history_path"],
        result,
        signature=str(status["signature"]),
        engine="full3d",
    )
    payload = read_json(status["metadata_path"])
    _reject_zero_response_recorder_anomaly(payload, "full3d")
    return {**status, "current": True, "cache_reused": False, "payload": payload}


def ensure_fishbone_case(
    artifact: FishboneArtifact,
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    damping_ratio = float(config["model"]["damping_ratio"])
    damping_model = str(config["model"]["damping_model"])
    instability_drift = float(
        config["model"]["dynamic_instability_drift"]
    )
    signature_data = pair_case_signature(
        artifact,
        pair,
        target_im_g,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=instability_drift,
    )
    expected = str(signature_data["case_signature"])
    cached = load_cached_pair_result(
        config["run_dir"],
        artifact.building_id,
        str(pair["pair_id"]),
        target_im_g,
        expected_signature=expected,
    )
    if cached is not None:
        _reject_zero_response_recorder_anomaly(cached, "fishbone")
        return {
            "cache_reused": True,
            "signature": expected,
            "payload": cached,
            "metadata_path": fishbone_result_path(
                config["run_dir"],
                artifact.building_id,
                str(pair["pair_id"]),
                target_im_g,
            ),
            "history_path": Path(str(cached["history_path"])),
            "signature_data": signature_data,
        }
    result = run_pair_nltha(
        artifact,
        pair,
        target_im_g,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=instability_drift,
        maximum_runtime_s=float(
            config["model"].get(
                "nltha_analysis_guard_max_runtime_s", 900.0
            )
        ),
    )
    metadata, history = save_pair_result(result, config["run_dir"])
    payload = read_json(metadata)
    _reject_zero_response_recorder_anomaly(payload, "fishbone")
    return {
        "cache_reused": False,
        "signature": expected,
        "payload": payload,
        "metadata_path": metadata,
        "history_path": history,
        "signature_data": signature_data,
    }


def _reject_zero_response_recorder_anomaly(
    payload: dict[str, Any], engine: str
) -> None:
    completed = float(payload.get("completed_fraction") or 0.0)
    maximum = float(payload.get("max_midr") or 0.0)
    if (
        str(payload.get("status")) == "numerical_failure"
        and completed >= 1.0 - 1.0e-8
        and math.isclose(maximum, 0.0, abs_tol=1.0e-15)
    ):
        raise RuntimeError(
            f"{engine}: record ended with a zero-response recorder anomaly"
        )


def _relative_error(observed: float, reference: float) -> float:
    return (float(observed) - float(reference)) / max(
        abs(float(reference)), np.finfo(float).eps
    )


def _nrmse(observed: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(observed - reference)
        / max(np.linalg.norm(reference), np.finfo(float).eps)
    )


def _history_metrics(
    full: dict[str, np.ndarray],
    fishbone: dict[str, np.ndarray],
    axis: str,
) -> list[dict[str, Any]]:
    full_time = np.asarray(full["time_s"], dtype=float)
    fish_time = np.asarray(fishbone["time_s"], dtype=float)
    full_idr = np.asarray(full[f"idr_{axis}"], dtype=float)
    fish_idr = np.asarray(
        fishbone[f"story_drift_ratio_{axis}"], dtype=float
    )
    upper_time = min(float(full_time[-1]), float(fish_time[-1]))
    mask = full_time <= upper_time + 1.0e-12
    reference_time = full_time[mask]
    reference = full_idr[mask]
    rows = []
    for story in range(reference.shape[1]):
        observed = np.interp(
            reference_time, fish_time, fish_idr[:, story]
        )
        target = reference[:, story]
        correlation = None
        if np.std(observed) > 1.0e-12 and np.std(target) > 1.0e-12:
            correlation = float(np.corrcoef(target, observed)[0, 1])
        rows.append(
            {
                "axis": axis.upper(),
                "story": story + 1,
                "correlation": correlation,
                "history_nrmse": _nrmse(observed, target),
                "overlap_end_time_s": upper_time,
            }
        )
    return rows


def compare_case(
    config: dict[str, Any],
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    artifact: FishboneArtifact,
    full3d: dict[str, Any],
    fishbone: dict[str, Any],
) -> dict[str, Any]:
    full_payload = dict(full3d["payload"])
    fish_payload = dict(fishbone["payload"])
    full_summary = dict(full_payload.get("summary") or {})
    if not full_summary:
        raise RuntimeError("Full-3D validation summary is missing")
    with np.load(full3d["history_path"]) as values:
        full_arrays = {
            key: np.asarray(values[key], dtype=float) for key in values.files
        }
    with np.load(fishbone["history_path"]) as values:
        fish_arrays = {
            key: np.asarray(values[key], dtype=float)
            for key in values.files
        }

    full_profiles = {
        axis: np.asarray(
            full_summary[f"peak_midr_profile_{axis}"], dtype=float
        )
        for axis in ("x", "y")
    }
    fish_profiles = {
        axis: np.max(
            np.abs(fish_arrays[f"story_drift_ratio_{axis}"]), axis=0
        )
        for axis in ("x", "y")
    }
    full_midr = float(full_summary["max_midr"])
    fish_midr = float(fish_payload["max_midr"])
    history_rows = _history_metrics(full_arrays, fish_arrays, "x")
    history_rows.extend(_history_metrics(full_arrays, fish_arrays, "y"))
    full_physical = str(full_payload.get("status")) == "dynamic_instability"
    fish_physical = str(fish_payload.get("status")) == "dynamic_instability"
    full_numerical = str(full_payload.get("status")) == "numerical_failure"
    fish_numerical = str(fish_payload.get("status")) == "numerical_failure"
    full_critical = int(full_summary["critical_story"])
    fish_critical = int(fish_payload["critical_story"])
    roof_errors = {}
    base_errors = {}
    for axis in ("x", "y"):
        roof_errors[axis] = _relative_error(
            fish_payload[f"maximum_roof_displacement_{axis}_m"],
            full_summary[f"peak_roof_displacement_{axis}_m"],
        )
        base_errors[axis] = _relative_error(
            fish_payload[f"maximum_base_shear_{axis}_kn"],
            full_summary[f"peak_base_shear_{axis}_kn"],
        )
    full_unscaled_sa, full_scale_factor = _checked_scale_factor(
        pair, float(building["t1_s"]), target_im_g, config
    )
    fish_scale_factor = float(
        fishbone["signature_data"]["scale_factor"]
    )
    comparison_signature = stable_hash(
        {
            "schema": FISHBONE_VALIDATION_CASE_SCHEMA_VERSION,
            "full3d_signature": full3d["signature"],
            "fishbone_signature": fishbone["signature"],
        }
    )
    payload: dict[str, Any] = {
        "schema_version": FISHBONE_VALIDATION_CASE_SCHEMA_VERSION,
        "comparison_signature": comparison_signature,
        "building_id": artifact.building_id,
        "pair_id": str(pair["pair_id"]),
        "target_im_g": float(target_im_g),
        "artifact_hash": artifact.artifact_hash,
        "full3d_reference_period_s": float(building["t1_s"]),
        "fishbone_reference_period_s": float(
            fishbone["signature_data"]["reference_period_s"]
        ),
        "full3d_unscaled_sa_g": float(full_unscaled_sa),
        "fishbone_unscaled_sa_g": float(
            fishbone["signature_data"]["unscaled_sa_g"]
        ),
        "full3d_scale_factor": float(full_scale_factor),
        "fishbone_scale_factor": fish_scale_factor,
        "scale_factor_relative_difference": _relative_error(
            fish_scale_factor, float(full_scale_factor)
        ),
        "full3d_case_signature": full3d["signature"],
        "fishbone_case_signature": fishbone["signature"],
        "full3d_cache_reused": bool(full3d["cache_reused"]),
        "fishbone_cache_reused": bool(fishbone["cache_reused"]),
        "full3d_status": str(full_payload.get("status")),
        "fishbone_status": str(fish_payload.get("status")),
        "full3d_physical_collapse": full_physical,
        "fishbone_physical_collapse": fish_physical,
        "full3d_numerical_failure": full_numerical,
        "fishbone_numerical_failure": fish_numerical,
        "collapse_class_match": full_physical == fish_physical,
        "numerical_class_match": full_numerical == fish_numerical,
        "full3d_midr": full_midr,
        "fishbone_midr": fish_midr,
        "midr_signed_relative_error": _relative_error(
            fish_midr, full_midr
        ),
        "midr_abs_log_error": abs(
            math.log(max(fish_midr, 1.0e-12) / max(full_midr, 1.0e-12))
        ),
        "drift_profile_nrmse_x": _nrmse(
            fish_profiles["x"], full_profiles["x"]
        ),
        "drift_profile_nrmse_y": _nrmse(
            fish_profiles["y"], full_profiles["y"]
        ),
        "drift_profile_nrmse_combined": _nrmse(
            np.concatenate((fish_profiles["x"], fish_profiles["y"])),
            np.concatenate((full_profiles["x"], full_profiles["y"])),
        ),
        "full3d_peak_midr_profile_x": full_profiles["x"].tolist(),
        "full3d_peak_midr_profile_y": full_profiles["y"].tolist(),
        "fishbone_peak_midr_profile_x": fish_profiles["x"].tolist(),
        "fishbone_peak_midr_profile_y": fish_profiles["y"].tolist(),
        "full3d_critical_story": full_critical,
        "fishbone_critical_story": fish_critical,
        "critical_story_difference": abs(full_critical - fish_critical),
        "critical_story_exact": full_critical == fish_critical,
        "critical_story_within_one": abs(full_critical - fish_critical) <= 1,
        "full3d_critical_direction": str(
            full_summary["critical_direction"]
        ),
        "fishbone_critical_direction": str(
            fish_payload["critical_direction"]
        ),
        "critical_direction_match": str(
            full_summary["critical_direction"]
        )
        == str(fish_payload["critical_direction"]),
        "roof_signed_relative_error_x": roof_errors["x"],
        "roof_signed_relative_error_y": roof_errors["y"],
        "base_signed_relative_error_x": base_errors["x"],
        "base_signed_relative_error_y": base_errors["y"],
        "history_by_story": history_rows,
        "history_correlation_mean": (
            float(
                np.mean(
                    [
                        row["correlation"]
                        for row in history_rows
                        if row["correlation"] is not None
                    ]
                )
            )
            if any(row["correlation"] is not None for row in history_rows)
            else None
        ),
        "history_nrmse_mean": float(
            np.mean([row["history_nrmse"] for row in history_rows])
        ),
        "full3d_runtime_s": float(full_payload["runtime_s"]),
        "fishbone_runtime_s": float(
            fish_payload.get("runtime_s_solver", math.nan)
        ),
        "speedup": float(full_payload["runtime_s"])
        / max(
            float(fish_payload.get("runtime_s_solver", math.nan)),
            np.finfo(float).eps,
        ),
        "full3d_completed_fraction": full_payload.get(
            "completed_fraction"
        ),
        "fishbone_completed_fraction": fish_payload.get(
            "completed_fraction"
        ),
        "full3d_recovery_level": full_payload.get("recovery_level"),
        "fishbone_recovery_attempt_count": fish_payload.get(
            "recovery_attempt_count"
        ),
        "fishbone_collapse_basis": fish_payload.get("collapse_basis"),
        "full3d_metadata_path": str(
            Path(full3d["metadata_path"]).absolute()
        ),
        "full3d_history_path": str(
            Path(full3d["history_path"]).absolute()
        ),
        "fishbone_metadata_path": str(
            Path(fishbone["metadata_path"]).absolute()
        ),
        "fishbone_history_path": str(
            Path(fishbone["history_path"]).absolute()
        ),
    }
    path = comparison_path(
        config,
        artifact.building_id,
        str(pair["pair_id"]),
        target_im_g,
    )
    atomic_write_json(path, payload)
    payload["comparison_path"] = str(path.absolute())
    return payload


def run_validation_case(
    config: dict[str, Any],
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
) -> dict[str, Any]:
    artifact = load_artifact(
        config, str(building["building_id"]), require_certified=False
    )
    if artifact.source_model_hash != str(building["model_hash"]):
        raise RuntimeError("Fishbone source model hash is stale")
    full3d = ensure_full3d_case(
        building, pair, target_im_g, config
    )
    fishbone = ensure_fishbone_case(
        artifact, pair, target_im_g, config
    )
    return compare_case(
        config,
        building,
        pair,
        target_im_g,
        artifact,
        full3d,
        fishbone,
    )


def load_case_inputs(
    config_path: str | Path,
    building_id: str,
    pair_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    manifest = load_locked_manifest(config)
    buildings, pairs = load_rows(config, manifest)
    if building_id not in buildings:
        raise KeyError(f"{building_id}: outside locked manifest")
    if pair_id not in pairs:
        raise KeyError(f"{pair_id}: outside locked manifest")
    return config, buildings[building_id], pairs[pair_id]
