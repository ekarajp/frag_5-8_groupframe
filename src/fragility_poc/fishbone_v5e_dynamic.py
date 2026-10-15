"""Development NLTHA for the preregistered v5E DBE5 grouped Fishbone.

The excitation, damping, mass, P-Delta, convergence recovery and output
contract are inherited from the established grouped-frame solver.  Only the
physical-member numerical formulation is changed to the preregistered five
serial displacement-based subelements.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .fishbone import FishboneArtifact
from .fishbone_v4_dynamic import _combine_directional_results, _run_direction
from .fishbone_v5e_formulation import V5E_FORMULATION_SCHEMA_VERSION
from .fishbone_v5e_grouped_frame import build_grouped_frame_model_dbe5
from .ground_motion import load_acceleration, pair_sa_geomean_g
from .io_utils import stable_hash


FISHBONE_V5E_NLTHA_SCHEMA_VERSION = (
    "bidirectional-fishbone-v5e-grouped-2n-dbe5-persistent-v1"
)
SOLVER_POLICY = "persistent-primary-reconfigure-on-failure-v1"


def pair_case_signature(
    artifact: FishboneArtifact,
    pair: dict[str, Any],
    target_im_g: float,
    *,
    candidate_definition_hash: str,
    damping_ratio: float,
    damping_model: str,
    dynamic_instability_drift: float,
) -> dict[str, float | str]:
    period_x = float(artifact.target_modal_x.periods_s[0])
    period_y = float(artifact.target_modal_y.periods_s[0])
    reference_period = math.sqrt(period_x * period_y)
    unscaled_sa = float(pair_sa_geomean_g(pair, reference_period))
    scale_factor = float(target_im_g / unscaled_sa)
    signature = stable_hash(
        {
            "schema_version": FISHBONE_V5E_NLTHA_SCHEMA_VERSION,
            "member_formulation": V5E_FORMULATION_SCHEMA_VERSION,
            "solver_policy": SOLVER_POLICY,
            "artifact_hash": artifact.artifact_hash,
            "source_model_hash": artifact.source_model_hash,
            "candidate_definition_hash": candidate_definition_hash,
            "pair_id": str(pair["pair_id"]),
            "sha256_x": str(pair.get("sha256_x") or ""),
            "sha256_y": str(pair.get("sha256_y") or ""),
            "target_im_g": float(target_im_g),
            "scale_factor": scale_factor,
            "damping_ratio": float(damping_ratio),
            "damping_model": str(damping_model),
            "dynamic_instability_drift": float(dynamic_instability_drift),
            "coupling_policy": "2n-block-diagonal-independent-direction",
        }
    )
    return {
        "reference_period_s": reference_period,
        "unscaled_sa_g": unscaled_sa,
        "scale_factor": scale_factor,
        "case_signature": signature,
    }


def run_pair_nltha(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    candidate_definition_hash: str,
    maximum_runtime_s_per_direction: float = 900.0,
) -> dict[str, Any]:
    time_x, acceleration_x_g = load_acceleration(
        pair["component_x_path"], pair["units"]
    )
    time_y, acceleration_y_g = load_acceleration(
        pair["component_y_path"], pair["units"]
    )
    if not np.allclose(time_x, time_y, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: X/Y time vectors differ")
    damping_ratio = float(config["model"]["damping_ratio"])
    damping_model = str(config["model"]["damping_model"])
    instability = float(config["model"]["dynamic_instability_drift"])
    signature_data = pair_case_signature(
        artifact,
        pair,
        target_im_g,
        candidate_definition_hash=candidate_definition_hash,
        damping_ratio=damping_ratio,
        damping_model=damping_model,
        dynamic_instability_drift=instability,
    )
    started = time.perf_counter()
    directional: dict[str, dict[str, Any]] = {}
    for direction, acceleration in (("X", acceleration_x_g), ("Y", acceleration_y_g)):
        directional[direction] = _run_direction(
            artifact,
            building,
            config,
            direction,
            acceleration,
            record_dt_s=float(pair["dt_s"]),
            scale_factor=float(signature_data["scale_factor"]),
            damping_ratio=damping_ratio,
            damping_model=damping_model,
            dynamic_instability_drift=instability,
            maximum_runtime_s=maximum_runtime_s_per_direction,
            minimum_substep_factor=64,
            persistent_primary_analysis=True,
            model_builder=build_grouped_frame_model_dbe5,
        )
    arrays, summary = _combine_directional_results(
        artifact, directional["X"], directional["Y"]
    )
    return {
        "schema_version": FISHBONE_V5E_NLTHA_SCHEMA_VERSION,
        "analysis_engine": "fishbone_v5e_grouped_2n_dbe5",
        "member_formulation": V5E_FORMULATION_SCHEMA_VERSION,
        "solver_policy": SOLVER_POLICY,
        "building_id": artifact.building_id,
        "artifact_hash": artifact.artifact_hash,
        "source_model_hash": artifact.source_model_hash,
        "candidate_definition_hash": candidate_definition_hash,
        "pair_id": str(pair["pair_id"]),
        "target_im_g": float(target_im_g),
        "reference_period_s": float(signature_data["reference_period_s"]),
        "unscaled_sa_g": float(signature_data["unscaled_sa_g"]),
        "scale_factor": float(signature_data["scale_factor"]),
        "achieved_im_g": float(target_im_g),
        "case_signature": str(signature_data["case_signature"]),
        **summary,
        "directional_results": {
            direction: {
                key: value
                for key, value in result.items()
                if key not in {"arrays", "recovery_log"}
            }
            for direction, result in directional.items()
        },
        "runtime_s_solver": float(time.perf_counter() - started),
        "coupling_policy": "2n-block-diagonal-independent-direction",
        "limitations": [
            "no cross-direction axial-history coupling",
            "no torsional degree of freedom",
            "no bidirectional member interaction surface",
        ],
        "physics_changed_from_v5d": False,
        "numerical_member_formulation_changed_from_v5d": True,
        "_history_arrays": arrays,
        "_recovery_logs": {
            direction: result["recovery_log"]
            for direction, result in directional.items()
        },
    }
