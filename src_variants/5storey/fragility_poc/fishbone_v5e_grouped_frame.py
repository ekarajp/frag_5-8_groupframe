"""Development-only v5E displacement-based grouped-frame formulation.

This module wraps the locked v5D grouped-frame builder, removes only its
active-direction force-based physical members before gravity is analysed, and
replaces each with the preregistered five-subelement formulation.  It is
intentionally restricted to the story-isolation pilot until its development
gates pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fishbone import FishboneArtifact, FishboneModelInfo
from .fishbone_v4_grouped_frame import (
    GroupedFrameAudit,
    build_grouped_frame_model,
)
from .fishbone_v5e_formulation import (
    V5E_FORMULATION_SCHEMA_VERSION,
    OpenSeesMemberSubdivision,
    build_opensees_dbe5_member,
)
from .structural import _ops, plastic_hinge_layout


@dataclass(frozen=True)
class V5EGroupedFrameAudit:
    schema_version: str
    direction: str
    physical_column_count: int
    physical_beam_count: int
    numerical_column_element_count: int
    numerical_beam_element_count: int
    nonlinear_hinge_zone_count: int
    elastic_interior_element_count: int
    gravity_pattern_rebuilt: bool
    physical_hinge_lengths_changed: bool
    full3d_analysis_run_count: int


def _median_hinge_lengths(
    pairs: list[tuple[float, float]],
) -> tuple[float, float]:
    return (
        float(np.median([pair[0] for pair in pairs])),
        float(np.median([pair[1] for pair in pairs])),
    )


def _group_section_tag(
    *, group_offset: int, source_tag: int, multiplicity: int
) -> int:
    """Mirror the base builder's Parallel-section tag policy exactly."""

    if int(multiplicity) <= 0:
        raise ValueError("Grouped-frame multiplicity must be positive")
    return int(source_tag) if int(multiplicity) == 1 else int(group_offset + source_tag)


def build_grouped_frame_model_dbe5(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    config: dict[str, Any],
    direction: str,
) -> tuple[FishboneModelInfo, GroupedFrameAudit, V5EGroupedFrameAudit]:
    """Build the preregistered v5E story-isolation pilot model.

    The base builder is used to retain sections, grouping, nodes, masses and
    all source checks.  Its unanalysed gravity pattern and physical active
    members are then replaced deterministically before gravity analysis.
    """

    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("v5E direction must be X or Y")
    info, base_audit = build_grouped_frame_model(
        artifact,
        building,
        config,
        normalized,
        preserve_all_frame_lines=False,
        include_elastic_transverse_beams=False,
        include_nonlinear_transverse_beams=False,
        include_transverse_torsion=False,
    )
    ops = _ops()
    stories = int(building["stories"])
    bays = int(building["number_of_bays"])
    grid = bays + 1
    height = float(building["story_height_m"])
    bay = float(building["bay_width_m"])
    group_count = int(base_audit.reduced_frame_group_count)
    column_tags = np.asarray(info.column_elements, dtype=int).reshape(
        stories, group_count, grid
    )
    active_beams = info.beam_elements_x or info.beam_elements_y
    beam_tags = np.asarray(active_beams, dtype=int).reshape(
        stories, group_count, bays
    )
    endpoints = {
        int(tag): tuple(int(value) for value in ops.eleNodes(int(tag)))
        for tag in (*info.column_elements, *active_beams)
    }

    # No analysis has occurred.  Remove the unanalysed gravity pattern first,
    # then replace only the active physical members.  Section, material, node
    # and mass objects remain identical to v5D.
    ops.remove("loadPattern", 1)
    for tag in (*info.column_elements, *active_beams):
        ops.remove("element", int(tag))

    layout = plastic_hinge_layout(building, config)
    plane_fixity = (
        (0, 1, 0, 1, 0, 1)
        if normalized == "X"
        else (1, 0, 0, 0, 1, 1)
    )
    column_members: dict[int, OpenSeesMemberSubdivision] = {}
    beam_members: dict[int, OpenSeesMemberSubdivision] = {}
    for group, parallel_lines in enumerate(base_audit.group_source_line_indices):
        group_offset = 1_000_000 * (group + 1)
        multiplicity = int(base_audit.group_multiplicities[group])
        beam_nonlinear_section = _group_section_tag(
            group_offset=group_offset, source_tag=101, multiplicity=multiplicity
        )
        column_nonlinear_section = _group_section_tag(
            group_offset=group_offset, source_tag=102, multiplicity=multiplicity
        )
        beam_elastic_section = _group_section_tag(
            group_offset=group_offset, source_tag=111, multiplicity=multiplicity
        )
        column_elastic_section = _group_section_tag(
            group_offset=group_offset, source_tag=112, multiplicity=multiplicity
        )
        for story in range(stories):
            for active_index in range(grid):
                source_indices = (
                    tuple(active_index * grid + line for line in parallel_lines)
                    if normalized == "X"
                    else tuple(line * grid + active_index for line in parallel_lines)
                )
                hinge_i, hinge_j = _median_hinge_lengths(
                    [
                        layout.column_hinge_lengths_m[story][index]
                        for index in source_indices
                    ]
                )
                tag = int(column_tags[story, group, active_index])
                node_i, node_j = endpoints[tag]
                column_members[tag] = build_opensees_dbe5_member(
                    ops,
                    physical_element_tag=tag,
                    node_i=node_i,
                    node_j=node_j,
                    transformation_tag=1,
                    nonlinear_section_i=column_nonlinear_section,
                    nonlinear_section_j=column_nonlinear_section,
                    elastic_section=column_elastic_section,
                    member_length_m=height,
                    hinge_length_i_m=hinge_i,
                    hinge_length_j_m=hinge_j,
                    plane_fixity=plane_fixity,
                )
            for active_bay in range(bays):
                source_indices = (
                    tuple(line * bays + active_bay for line in parallel_lines)
                    if normalized == "X"
                    else tuple(
                        bays * grid + line * bays + active_bay
                        for line in parallel_lines
                    )
                )
                hinge_i, hinge_j = _median_hinge_lengths(
                    [
                        layout.beam_hinge_lengths_m[story][index]
                        for index in source_indices
                    ]
                )
                tag = int(beam_tags[story, group, active_bay])
                node_i, node_j = endpoints[tag]
                beam_members[tag] = build_opensees_dbe5_member(
                    ops,
                    physical_element_tag=tag,
                    node_i=node_i,
                    node_j=node_j,
                    transformation_tag=2,
                    nonlinear_section_i=beam_nonlinear_section,
                    nonlinear_section_j=beam_nonlinear_section,
                    elastic_section=beam_elastic_section,
                    member_length_m=bay,
                    hinge_length_i_m=hinge_i,
                    hinge_length_j_m=hinge_j,
                    plane_fixity=plane_fixity,
                )

    model = config["model"]
    live_fraction = float(model["live_load_mass_fraction"])
    density = float(model["concrete_density_kn_m3"])
    area_load = float(building["dead_load_kn_m2"]) + live_fraction * float(
        building["live_load_kn_m2"]
    )
    beam_self_weight = (
        float(building["beam_b_m"]) * float(building["beam_h_m"]) * density
    )
    column_segment_weight = (
        height
        * float(building["column_b_m"])
        * float(building["column_h_m"])
        * density
    )
    # The original time series remains in the domain when its load pattern is
    # removed; only recreate the pattern to avoid a duplicate time-series tag.
    ops.pattern("Plain", 1, 1)
    for group, (name, multiplicity) in enumerate(
        zip(base_audit.group_names, base_audit.group_multiplicities)
    ):
        adjacency = 1 if str(name).startswith("edge") else 2
        group_offset = 1_000_000 * (group + 1)
        active_uniform = (
            area_load * float(adjacency) * bay / 4.0 + beam_self_weight
        ) * int(multiplicity)
        # Adjacency is reconstructed from the immutable group name, matching
        # the locked edge=1/interior=2 base-builder convention.
        for story in range(stories):
            for active_bay in range(bays):
                tag = int(beam_tags[story, group, active_bay])
                for segment_tag in beam_members[tag].element_tags:
                    ops.eleLoad(
                        "-ele",
                        segment_tag,
                        "-type",
                        "-beamUniform",
                        0.0,
                        -active_uniform,
                        0.0,
                    )
            for active_index in range(grid):
                active_adjacency = 1 if active_index in {0, bays} else 2
                orthogonal_uniform = (
                    area_load * active_adjacency * bay / 4.0
                    + beam_self_weight
                )
                missing_orthogonal = (
                    int(multiplicity)
                    * float(adjacency)
                    * orthogonal_uniform
                    * bay
                    / 2.0
                )
                vertical = -(
                    missing_orthogonal
                    + int(multiplicity) * column_segment_weight
                )
                node = group_offset + (story + 1) * 1000 + active_index + 1
                if normalized == "X":
                    ops.load(node, 0.0, 0.0, vertical, 0.0, 0.0, 0.0)
                else:
                    ops.load(node, 0.0, 0.0, vertical, 0.0, 0.0, 0.0)

    if not math.isclose(
        float(base_audit.expected_gravity_kn),
        float(base_audit.applied_gravity_kn),
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise ValueError("v5E inherited an inconsistent gravity source")
    audit = V5EGroupedFrameAudit(
        schema_version=V5E_FORMULATION_SCHEMA_VERSION,
        direction=normalized,
        physical_column_count=len(column_members),
        physical_beam_count=len(beam_members),
        numerical_column_element_count=5 * len(column_members),
        numerical_beam_element_count=5 * len(beam_members),
        nonlinear_hinge_zone_count=2 * (len(column_members) + len(beam_members)),
        elastic_interior_element_count=3 * (len(column_members) + len(beam_members)),
        gravity_pattern_rebuilt=True,
        physical_hinge_lengths_changed=False,
        full3d_analysis_run_count=0,
    )
    return info, base_audit, audit
