"""Preregistered numerical formulation helpers for Fishbone candidate v5E.

The active v5D solver remains the production/default formulation.  This
module contains only pure geometry logic so the v5E member subdivision can be
reviewed and tested before any OpenSees model is changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fragility_poc.io_utils import read_json, sha256_file, stable_hash


V5E_FORMULATION_SCHEMA_VERSION = (
    "fishbone-v5e-dbe5-legendre2-physical-hinge-zone-preserving-v1"
)


def verify_preregistration_amendments(
    amendment_path: str | Path, amendment: dict[str, Any]
) -> None:
    """Fail closed when a preregistration bundle does not bind its sources."""

    entries = amendment.get("amendments")
    if entries is None:
        return
    if not isinstance(entries, list) or not entries:
        raise ValueError("The v5E amendment bundle is empty")
    root = Path(amendment_path).absolute().parent
    preregistered = bool(
        amendment.get("created_before_any_v5e_solver_result", False)
        or amendment.get("created_before_any_candidate_solver_result", False)
    )
    if not preregistered:
        raise ValueError("The candidate amendment bundle is not preregistered")
    expected_plan_hash = str(amendment.get("base_plan_sha256", "")).lower()
    if not expected_plan_hash:
        raise ValueError("The v5E amendment bundle does not bind its base plan")
    plan_path = root / "preregistered_pilot_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if sha256_file(plan_path).lower() != expected_plan_hash:
        raise ValueError(f"Stale v5E preregistered pilot plan: {plan_path}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid v5E amendment-bundle entry")
        source = Path(str(entry.get("path", "")))
        source = source if source.is_absolute() else root / source
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = stable_hash(read_json(source))
        expected = str(entry.get("stable_hash", ""))
        if observed != expected:
            raise ValueError(f"Stale v5E amendment source: {source}")


@dataclass(frozen=True)
class MemberSubdivision:
    """Five serial subelements for one physical member.

    Segment 0 and segment 4 are the only nonlinear fibre hinge zones.  The
    three middle segments cover the elastic interior exactly.  This prevents
    the common error of copying an end hinge into every numerical element.
    """

    member_length_m: float
    hinge_length_i_m: float
    hinge_length_j_m: float
    segment_lengths_m: tuple[float, float, float, float, float]
    cumulative_fractions: tuple[float, float, float, float, float, float]
    nonlinear_segment_indices: tuple[int, int] = (0, 4)
    elastic_segment_indices: tuple[int, int, int] = (1, 2, 3)

    def validate(self) -> None:
        values = (
            self.member_length_m,
            self.hinge_length_i_m,
            self.hinge_length_j_m,
            *self.segment_lengths_m,
            *self.cumulative_fractions,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Member subdivision contains a non-finite value")
        if self.member_length_m <= 0.0:
            raise ValueError("Physical member length must be positive")
        if any(length <= 0.0 for length in self.segment_lengths_m):
            raise ValueError("Every v5E numerical subelement must be positive")
        if not math.isclose(
            sum(self.segment_lengths_m),
            self.member_length_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("v5E subelements do not span the physical member")
        if not math.isclose(
            self.segment_lengths_m[0],
            self.hinge_length_i_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            self.segment_lengths_m[-1],
            self.hinge_length_j_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("v5E changed a physical end-hinge length")
        if self.cumulative_fractions[0] != 0.0 or not math.isclose(
            self.cumulative_fractions[-1], 1.0, abs_tol=1.0e-12
        ):
            raise ValueError("v5E member fractions must span [0, 1]")
        if any(
            right <= left
            for left, right in zip(
                self.cumulative_fractions[:-1],
                self.cumulative_fractions[1:],
            )
        ):
            raise ValueError("v5E member fractions must increase strictly")
        if self.nonlinear_segment_indices != (0, 4):
            raise ValueError("Only the two physical end zones may be nonlinear")
        if self.elastic_segment_indices != (1, 2, 3):
            raise ValueError("The physical interior must remain elastic")


def hinge_zone_subdivision(
    *,
    member_length_m: float,
    hinge_length_i_m: float,
    hinge_length_j_m: float,
) -> MemberSubdivision:
    """Return the locked five-subelement v5E geometry.

    The interior is divided into three equal numerical elements.  No clipping
    or building-specific repair is allowed: invalid hinge geometry fails
    closed because altering it would change the physical model.
    """

    length = float(member_length_m)
    hinge_i = float(hinge_length_i_m)
    hinge_j = float(hinge_length_j_m)
    if not all(math.isfinite(value) for value in (length, hinge_i, hinge_j)):
        raise ValueError("Member and hinge lengths must be finite")
    if length <= 0.0 or hinge_i <= 0.0 or hinge_j <= 0.0:
        raise ValueError("Member and hinge lengths must be positive")
    interior = length - hinge_i - hinge_j
    if interior <= 0.0:
        raise ValueError("Physical hinge zones consume the complete member")
    third = interior / 3.0
    segments = (hinge_i, third, third, third, hinge_j)
    cumulative = [0.0]
    cursor = 0.0
    for segment in segments:
        cursor += segment
        cumulative.append(cursor / length)
    cumulative[-1] = 1.0
    result = MemberSubdivision(
        member_length_m=length,
        hinge_length_i_m=hinge_i,
        hinge_length_j_m=hinge_j,
        segment_lengths_m=segments,
        cumulative_fractions=tuple(cumulative),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class OpenSeesMemberSubdivision:
    """Tags and geometry created for one v5E physical member."""

    physical_element_tag: int
    representative_element_tag: int
    element_tags: tuple[int, int, int, int, int]
    node_tags: tuple[int, int, int, int, int, int]
    internal_node_tags: tuple[int, int, int, int]
    integration_tags: tuple[int, int, int, int, int]
    section_tags: tuple[int, int, int, int, int]
    subdivision: MemberSubdivision


def build_opensees_dbe5_member(
    ops: Any,
    *,
    physical_element_tag: int,
    node_i: int,
    node_j: int,
    transformation_tag: int,
    nonlinear_section_i: int,
    nonlinear_section_j: int,
    elastic_section: int,
    member_length_m: float,
    hinge_length_i_m: float,
    hinge_length_j_m: float,
    plane_fixity: tuple[int, int, int, int, int, int],
) -> OpenSeesMemberSubdivision:
    """Create one hinge-zone-preserving five-element OpenSees member.

    The original physical element tag is retained on the first hinge segment
    so existing story-force aggregation can keep one representative per
    physical member.  Extra tags use reserved deterministic namespaces.
    """

    tag = int(physical_element_tag)
    if tag <= 0:
        raise ValueError("OpenSees physical element tag must be positive")
    if len(plane_fixity) != 6 or any(value not in (0, 1) for value in plane_fixity):
        raise ValueError("v5E plane fixity must contain six binary values")
    subdivision = hinge_zone_subdivision(
        member_length_m=member_length_m,
        hinge_length_i_m=hinge_length_i_m,
        hinge_length_j_m=hinge_length_j_m,
    )
    coordinates_i = tuple(float(value) for value in ops.nodeCoord(int(node_i)))
    coordinates_j = tuple(float(value) for value in ops.nodeCoord(int(node_j)))
    if len(coordinates_i) != 3 or len(coordinates_j) != 3:
        raise ValueError("v5E requires three-dimensional OpenSees nodes")
    geometric_length = math.sqrt(
        sum((right - left) ** 2 for left, right in zip(coordinates_i, coordinates_j))
    )
    if not math.isclose(
        geometric_length,
        subdivision.member_length_m,
        rel_tol=1.0e-9,
        abs_tol=1.0e-10,
    ):
        raise ValueError("v5E member length differs from node geometry")

    internal_node_tags = tuple(100_000_000 + 10 * tag + index for index in range(1, 5))
    for internal_tag, fraction in zip(
        internal_node_tags, subdivision.cumulative_fractions[1:-1]
    ):
        coordinates = tuple(
            left + fraction * (right - left)
            for left, right in zip(coordinates_i, coordinates_j)
        )
        ops.node(internal_tag, *coordinates)
        ops.fix(internal_tag, *plane_fixity)

    node_tags = (int(node_i), *internal_node_tags, int(node_j))
    element_tags = (
        tag,
        *(200_000_000 + 10 * tag + index for index in range(1, 5)),
    )
    integration_tags = tuple(
        300_000_000 + 10 * tag + index for index in range(5)
    )
    section_tags = (
        int(nonlinear_section_i),
        int(elastic_section),
        int(elastic_section),
        int(elastic_section),
        int(nonlinear_section_j),
    )
    for index, (element, integration, section) in enumerate(
        zip(element_tags, integration_tags, section_tags)
    ):
        ops.beamIntegration("Legendre", integration, section, 2)
        ops.element(
            "dispBeamColumn",
            element,
            node_tags[index],
            node_tags[index + 1],
            int(transformation_tag),
            integration,
        )
    return OpenSeesMemberSubdivision(
        physical_element_tag=tag,
        representative_element_tag=tag,
        element_tags=element_tags,
        node_tags=node_tags,
        internal_node_tags=internal_node_tags,
        integration_tags=integration_tags,
        section_tags=section_tags,
        subdivision=subdivision,
    )
