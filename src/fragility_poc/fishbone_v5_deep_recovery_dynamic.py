"""Development-only v5C-R12 solver with deeper local substepping.

The physical grouped-frame model is identical to ``v5cr12_reconfigured``.
The only change is that, after the existing dt/64 recovery ladder is
exhausted, the solver may try dt/128 and dt/256.  This is an exact-prefix
extension: cases that converge before the new stages follow the same path as
the parent candidate.
"""

from __future__ import annotations

from typing import Any

from .fishbone import FishboneArtifact
from .fishbone_v5_reconfigured_dynamic import (
    pair_case_signature,
    run_pair_nltha as _run_reconfigured_pair,
)


FISHBONE_V5_DEEP_RECOVERY_NLTHA_SCHEMA_VERSION = (
    "bidirectional-fishbone-v5-grouped-2n-reconfigure-deep-substep-v1"
)
SOLVER_POLICY = "reconfigure-every-step-recovery-dt256-v1"


def run_pair_nltha(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    candidate_definition_hash: str,
    maximum_runtime_s_per_direction: float = 1800.0,
) -> dict[str, Any]:
    result = _run_reconfigured_pair(
        artifact,
        building,
        pair,
        target_im_g,
        config,
        candidate_definition_hash=candidate_definition_hash,
        maximum_runtime_s_per_direction=maximum_runtime_s_per_direction,
        minimum_substep_factor=256,
    )
    result["schema_version"] = FISHBONE_V5_DEEP_RECOVERY_NLTHA_SCHEMA_VERSION
    result["analysis_engine"] = "fishbone_v5_grouped_2n_reconfigured_deep_recovery"
    result["solver_policy"] = SOLVER_POLICY
    result["physics_changed_from_v5cr12"] = False
    result["exact_prefix_parent_solver"] = "reconfigure-every-step-recovery-v1"
    result["new_recovery_stages"] = ["dt/128", "dt/256"]
    return result


__all__ = [
    "FISHBONE_V5_DEEP_RECOVERY_NLTHA_SCHEMA_VERSION",
    "SOLVER_POLICY",
    "pair_case_signature",
    "run_pair_nltha",
]
