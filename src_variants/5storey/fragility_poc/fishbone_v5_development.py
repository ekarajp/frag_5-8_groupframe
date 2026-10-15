"""Development-only dynamic evidence utilities for Fishbone Candidate v5.

This module deliberately uses a separate namespace from the accepted v4
baseline and marks every comparison as development evidence.  Blind-holdout
artifacts are rejected by the calling scripts and are never accepted here as
calibration data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .fishbone import FishboneArtifact
from .fishbone_validation import _history_metrics, _nrmse, _relative_error
from .ida import _checked_scale_factor
from .io_utils import atomic_write_json, stable_hash


COMPARISON_SCHEMA_VERSION = "full3d-fishbone-v5-development-comparison-v1"
HOLDOUT_COMPARISON_SCHEMA_VERSION = (
    "full3d-fishbone-v5-blind-holdout-comparison-v1"
)


def _im_token(target_im_g: float) -> str:
    return f"{target_im_g:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def case_paths(
    run_dir: str | Path,
    campaign_hash: str,
    candidate_label: str,
    building_id: str,
    pair_id: str,
    target_im_g: float,
    *,
    split: str = "development",
) -> tuple[Path, Path]:
    if split not in {"development", "blind_holdout"}:
        raise ValueError(f"Unsupported Fishbone v5 evidence split: {split}")
    root = (
        Path(run_dir)
        / "fishbone_v5"
        / split
        / campaign_hash
        / "dynamic"
        / candidate_label
        / building_id
        / pair_id
        / f"im_{_im_token(target_im_g)}"
    )
    return root / "case.json", root / "case.npz"


def comparison_path(
    run_dir: str | Path,
    campaign_hash: str,
    candidate_label: str,
    building_id: str,
    pair_id: str,
    target_im_g: float,
    *,
    split: str = "development",
) -> Path:
    if split not in {"development", "blind_holdout"}:
        raise ValueError(f"Unsupported Fishbone v5 evidence split: {split}")
    return (
        Path(run_dir)
        / "fishbone_v5"
        / split
        / campaign_hash
        / "comparisons"
        / candidate_label
        / building_id
        / pair_id
        / f"im_{_im_token(target_im_g)}"
        / "comparison.json"
    )


def compare_development_case(
    config: dict[str, Any],
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    artifact: FishboneArtifact,
    full3d: dict[str, Any],
    fishbone: dict[str, Any],
    *,
    campaign_hash: str,
    candidate_label: str,
    candidate_definition_hash: str,
    split: str = "development",
) -> dict[str, Any]:
    if split not in {"development", "blind_holdout"}:
        raise ValueError(f"Unsupported Fishbone v5 evidence split: {split}")
    comparison_schema = (
        COMPARISON_SCHEMA_VERSION
        if split == "development"
        else HOLDOUT_COMPARISON_SCHEMA_VERSION
    )
    full_payload = dict(full3d["payload"])
    fish_payload = dict(fishbone["payload"])
    full_summary = dict(full_payload.get("summary") or {})
    if not full_summary:
        raise RuntimeError("Full3D validation summary is missing")
    with np.load(full3d["history_path"]) as values:
        full_arrays = {
            key: np.asarray(values[key], dtype=float) for key in values.files
        }
    with np.load(fishbone["history_path"]) as values:
        fish_arrays = {
            key: np.asarray(values[key], dtype=float) for key in values.files
        }
    full_profiles = {
        axis: np.asarray(full_summary[f"peak_midr_profile_{axis}"], dtype=float)
        for axis in ("x", "y")
    }
    fish_profiles = {
        axis: np.max(np.abs(fish_arrays[f"story_drift_ratio_{axis}"]), axis=0)
        for axis in ("x", "y")
    }
    history_rows = _history_metrics(full_arrays, fish_arrays, "x")
    history_rows.extend(_history_metrics(full_arrays, fish_arrays, "y"))
    full_midr = float(full_summary["max_midr"])
    fish_midr = float(fish_payload["max_midr"])
    full_critical = int(full_summary["critical_story"])
    fish_critical = int(fish_payload["critical_story"])
    full_physical = str(full_payload.get("status")) == "dynamic_instability"
    fish_physical = str(fish_payload.get("status")) == "dynamic_instability"
    full_numerical = str(full_payload.get("status")) == "numerical_failure"
    fish_numerical = str(fish_payload.get("status")) == "numerical_failure"
    roof_errors = {
        axis: _relative_error(
            fish_payload[f"maximum_roof_displacement_{axis}_m"],
            full_summary[f"peak_roof_displacement_{axis}_m"],
        )
        for axis in ("x", "y")
    }
    base_errors = {
        axis: _relative_error(
            fish_payload[f"maximum_base_shear_{axis}_kn"],
            full_summary[f"peak_base_shear_{axis}_kn"],
        )
        for axis in ("x", "y")
    }
    full_unscaled_sa, full_scale_factor = _checked_scale_factor(
        pair, float(building["t1_s"]), target_im_g, config
    )
    comparison_signature = stable_hash(
        {
            "schema": comparison_schema,
            "campaign_hash": campaign_hash,
            "candidate_label": candidate_label,
            "candidate_definition_hash": candidate_definition_hash,
            "full3d_signature": full3d["signature"],
            "fishbone_signature": fishbone["signature"],
        }
    )
    correlations = [
        float(row["correlation"])
        for row in history_rows
        if row["correlation"] is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": comparison_schema,
        "comparison_signature": comparison_signature,
        "campaign_hash": campaign_hash,
        "candidate_label": candidate_label,
        "candidate_definition_hash": candidate_definition_hash,
        "building_id": artifact.building_id,
        "pair_id": str(pair["pair_id"]),
        "target_im_g": float(target_im_g),
        "artifact_hash": artifact.artifact_hash,
        "full3d_reference_period_s": float(building["t1_s"]),
        "fishbone_reference_period_s": float(fish_payload["reference_period_s"]),
        "full3d_unscaled_sa_g": float(full_unscaled_sa),
        "fishbone_unscaled_sa_g": float(fish_payload["unscaled_sa_g"]),
        "full3d_scale_factor": float(full_scale_factor),
        "fishbone_scale_factor": float(fish_payload["scale_factor"]),
        "scale_factor_relative_difference": _relative_error(
            float(fish_payload["scale_factor"]), float(full_scale_factor)
        ),
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
        "midr_signed_relative_error": _relative_error(fish_midr, full_midr),
        "midr_abs_log_error": abs(
            math.log(max(fish_midr, 1.0e-12) / max(full_midr, 1.0e-12))
        ),
        "drift_profile_nrmse_x": _nrmse(fish_profiles["x"], full_profiles["x"]),
        "drift_profile_nrmse_y": _nrmse(fish_profiles["y"], full_profiles["y"]),
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
        "full3d_critical_direction": str(full_summary["critical_direction"]),
        "fishbone_critical_direction": str(fish_payload["critical_direction"]),
        "critical_direction_match": str(full_summary["critical_direction"])
        == str(fish_payload["critical_direction"]),
        "roof_signed_relative_error_x": roof_errors["x"],
        "roof_signed_relative_error_y": roof_errors["y"],
        "base_signed_relative_error_x": base_errors["x"],
        "base_signed_relative_error_y": base_errors["y"],
        "history_by_story": history_rows,
        "history_correlation_mean": float(np.mean(correlations))
        if correlations
        else None,
        "history_nrmse_mean": float(
            np.mean([float(row["history_nrmse"]) for row in history_rows])
        ),
        "full3d_runtime_s": float(full_payload["runtime_s"]),
        "fishbone_runtime_s": float(fish_payload["runtime_s_solver"]),
        "speedup": float(full_payload["runtime_s"])
        / max(float(fish_payload["runtime_s_solver"]), np.finfo(float).eps),
        "full3d_completed_fraction": full_payload.get("completed_fraction"),
        "fishbone_completed_fraction": fish_payload.get("completed_fraction"),
        "fishbone_coupling_policy": fish_payload.get("coupling_policy"),
        "fishbone_limitations": fish_payload.get("limitations"),
        "full3d_metadata_path": str(Path(full3d["metadata_path"]).absolute()),
        "full3d_history_path": str(Path(full3d["history_path"]).absolute()),
        "fishbone_metadata_path": str(Path(fishbone["metadata_path"]).absolute()),
        "fishbone_history_path": str(Path(fishbone["history_path"]).absolute()),
        "split": split,
        "development_only": split == "development",
        "uses_holdout_evidence": split == "blind_holdout",
        "uses_holdout_evidence_for_tuning": False,
        "uses_ground_motion_evidence": True,
        "uses_ida_evidence": False,
    }
    path = comparison_path(
        config["run_dir"],
        campaign_hash,
        candidate_label,
        artifact.building_id,
        str(pair["pair_id"]),
        target_im_g,
        split=split,
    )
    atomic_write_json(path, payload)
    payload["comparison_path"] = str(path.absolute())
    return payload
