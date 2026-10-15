"""Development-only axial-flexure-aware grouped frame-line reduction.

This module is deliberately separate from the active Fishbone-v3 builder.
For one horizontal direction it condenses identical parallel Full3D frame
lines into at most two coincident planar groups (edge and interior).  Every
group retains the complete bay topology and the source RC fibre sections.
OpenSees ``Parallel`` sections add the response of the physical members in a
group, so axial force, flexure and their P-M interaction remain coupled.

The first implementation is a unidirectional static/modal research candidate.
It is not used by production NLTHA/IDA and does not open blind-holdout data.
Units are kN, m, seconds and radians.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import KSC_TO_KN_M2
from .fishbone import FishboneArtifact, FishboneModelInfo
from .structural import (
    _elastic_rectangular_section,
    _ops,
    _rc_section,
    build_model as build_full3d_model,
    beam_bar_positions,
    beam_reinforcement_fibers,
    define_hysteretic_steel_material,
    hognestad_concrete_parameters,
    mander_rectangular_confined_parameters,
    perimeter_bar_positions,
    plastic_hinge_elastic_interior_length_m,
    plastic_hinge_layout,
    rectangular_torsional_constant,
    run_gravity as run_full3d_gravity,
)


FISHBONE_V4_GROUPED_FRAME_SCHEMA_VERSION = (
    "fishbone-v4-spatial-lattice-v7-nonlinear-transverse-identity-development"
)


@dataclass(frozen=True)
class GroupedFrameAudit:
    direction: str
    group_names: tuple[str, ...]
    group_multiplicities: tuple[int, ...]
    group_source_line_indices: tuple[tuple[int, ...], ...]
    physical_frame_line_count: int
    reduced_frame_group_count: int
    source_column_count: int
    reduced_column_element_count: int
    source_active_beam_count: int
    reduced_beam_element_count: int
    expected_gravity_kn: float
    applied_gravity_kn: float
    maximum_hinge_length_group_spread_m: float
    transverse_torsion_element_count: int
    transverse_torsion_stiffness_knm_rad: float
    elastic_transverse_beam_element_count: int
    nonlinear_transverse_beam_element_count: int


def _metadata(building: dict[str, Any]) -> dict[str, Any]:
    raw = building.get("design_metadata_json", "{}")
    if isinstance(raw, str):
        return dict(json.loads(raw or "{}"))
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError("design_metadata_json must be JSON text or a mapping")


def _define_source_sections(
    building: dict[str, Any], config: dict[str, Any]
) -> dict[str, float | int]:
    """Define the same base RC fibre sections as the Full3D source model."""

    ops = _ops()
    model = config["model"]
    metadata = _metadata(building)
    detailing = model["material_detailing"]
    fc = float(building["fc_ksc"]) * KSC_TO_KN_M2
    fy = float(
        building.get("steel_fy_ksc", config["building"]["steel_fy_ksc"])
    ) * KSC_TO_KN_M2
    beam_width = float(building["beam_b_m"])
    beam_depth = float(building["beam_h_m"])
    column_width = float(building["column_b_m"])
    column_depth = float(building["column_h_m"])
    beam_bar_diameter = float(building["beam_bar_diameter_m"])
    column_bar_diameter = float(building["column_bar_diameter_m"])
    beam_bar_area = math.pi * beam_bar_diameter**2 / 4.0
    column_bar_area = math.pi * column_bar_diameter**2 / 4.0
    clear_cover = float(detailing["clear_cover_m"])
    beam_hoop_diameter = float(
        metadata.get(
            "beam_stirrup_diameter_m", detailing["beam_hoop_diameter_m"]
        )
    )
    beam_hoop_spacing = float(
        metadata.get(
            "beam_stirrup_spacing_m", detailing["beam_hoop_spacing_m"]
        )
    )
    beam_hoop_legs = int(
        metadata.get(
            "beam_stirrup_legs", detailing["beam_hoop_legs_each_axis"]
        )
    )
    column_hoop_diameter = float(
        metadata.get(
            "column_hoop_diameter_m", detailing["column_hoop_diameter_m"]
        )
    )
    column_hoop_spacing = float(
        metadata.get(
            "column_hoop_spacing_m", detailing["column_hoop_spacing_m"]
        )
    )
    column_hoop_legs_x = int(
        metadata.get("column_hoop_legs_x", detailing["column_hoop_legs_x"])
    )
    column_hoop_legs_y = int(
        metadata.get("column_hoop_legs_y", detailing["column_hoop_legs_y"])
    )
    beam_transverse_fy = float(
        metadata.get(
            "beam_stirrup_fy_ksc",
            config["building"]["transverse_steel_fy_ksc"],
        )
    ) * KSC_TO_KN_M2
    column_transverse_fy = float(
        metadata.get(
            "column_hoop_fy_ksc",
            config["building"]["transverse_steel_fy_ksc"],
        )
    ) * KSC_TO_KN_M2

    bars_per_face = int(building["beam_bars_per_face"])
    beam_top_count = int(metadata.get("beam_top_bar_count", bars_per_face))
    beam_bottom_count = int(
        metadata.get("beam_bottom_bar_count", bars_per_face)
    )
    beam_side_count = int(metadata.get("beam_side_bar_count_each", 0))
    beam_top_diameter = float(
        metadata.get("beam_top_bar_diameter_m", beam_bar_diameter)
    )
    beam_bottom_diameter = float(
        metadata.get("beam_bottom_bar_diameter_m", beam_bar_diameter)
    )
    beam_side_diameter = float(
        metadata.get("beam_side_bar_diameter_m", beam_bar_diameter)
    )
    beam_top_layers = int(metadata.get("beam_top_bar_layers", 1))
    beam_bottom_layers = int(metadata.get("beam_bottom_bar_layers", 1))
    beam_fibers = beam_reinforcement_fibers(
        width_m=beam_width,
        depth_m=beam_depth,
        clear_cover_m=clear_cover,
        stirrup_diameter_m=beam_hoop_diameter,
        top_bar_count=beam_top_count,
        top_bar_diameter_m=beam_top_diameter,
        top_bar_layers=beam_top_layers,
        bottom_bar_count=beam_bottom_count,
        bottom_bar_diameter_m=beam_bottom_diameter,
        bottom_bar_layers=beam_bottom_layers,
        side_bar_count_each=beam_side_count,
        side_bar_diameter_m=beam_side_diameter,
        minimum_clear_spacing_m=float(
            detailing.get("beam_minimum_clear_bar_spacing_m", 0.025)
        ),
        nominal_maximum_aggregate_size_m=float(
            detailing["nominal_maximum_coarse_aggregate_size_m"]
        ),
        minimum_bars_per_layer=int(
            config["building"]["beam_section_search"].get(
                "minimum_bars_per_layer", 2
            )
        ),
        three_layer_minimum_depth_m=float(
            config["building"]["beam_section_search"].get(
                "three_layer_minimum_depth_m", 0.50
            )
        ),
    )
    beam_equivalent_diameter = max(
        beam_top_diameter, beam_bottom_diameter, beam_side_diameter
    )
    beam_bar_half_width = (
        beam_width / 2.0
        - clear_cover
        - beam_hoop_diameter
        - beam_equivalent_diameter / 2.0
    )
    beam_bar_half_depth = (
        beam_depth / 2.0
        - clear_cover
        - beam_hoop_diameter
        - beam_equivalent_diameter / 2.0
    )
    beam_bar_layout = beam_bar_positions(
        beam_top_count
        if beam_top_layers == 1
        else math.ceil(beam_top_count / beam_top_layers),
        beam_side_count,
        half_width_m=beam_bar_half_width,
        half_depth_m=beam_bar_half_depth,
        bottom_bar_count=(
            beam_bottom_count
            if beam_bottom_layers == 1
            else math.ceil(beam_bottom_count / beam_bottom_layers)
        ),
    )
    column_bar_half_width = (
        column_width / 2.0
        - clear_cover
        - column_hoop_diameter
        - column_bar_diameter / 2.0
    )
    column_bar_half_depth = (
        column_depth / 2.0
        - clear_cover
        - column_hoop_diameter
        - column_bar_diameter / 2.0
    )
    column_bar_count = int(building["column_bar_count"])
    column_bar_layout = perimeter_bar_positions(
        column_bar_count,
        half_width_m=column_bar_half_width,
        half_depth_m=column_bar_half_depth,
    )

    hognestad = model["hognestad"]
    cover_parameters = hognestad_concrete_parameters(
        fc,
        peak_strain=float(hognestad["peak_strain"]),
        ultimate_strain=float(hognestad["ultimate_strain"]),
        ultimate_strength_ratio=float(hognestad["ultimate_strength_ratio"]),
    )
    confinement = model["mander_confinement"]
    common = {
        "clear_cover_m": clear_cover,
        "eps_co": float(hognestad["peak_strain"]),
        "transverse_ultimate_strain": float(
            confinement["transverse_ultimate_strain"]
        ),
        "minimum_ultimate_strain": float(
            confinement["minimum_ultimate_strain"]
        ),
        "maximum_ultimate_strain": float(
            confinement["maximum_ultimate_strain"]
        ),
    }
    beam_confined = mander_rectangular_confined_parameters(
        fc,
        width_m=beam_width,
        depth_m=beam_depth,
        hoop_diameter_m=beam_hoop_diameter,
        hoop_spacing_m=beam_hoop_spacing,
        transverse_legs_x=beam_hoop_legs,
        transverse_legs_y=beam_hoop_legs,
        longitudinal_bar_diameter_m=beam_equivalent_diameter,
        longitudinal_bar_area_m2=(
            math.pi * beam_equivalent_diameter**2 / 4.0
        ),
        bar_positions=beam_bar_layout,
        longitudinal_area_total_m2=sum(row[2] for row in beam_fibers),
        transverse_fy_kn_m2=beam_transverse_fy,
        **common,
    )
    column_confined = mander_rectangular_confined_parameters(
        fc,
        width_m=column_width,
        depth_m=column_depth,
        hoop_diameter_m=column_hoop_diameter,
        hoop_spacing_m=column_hoop_spacing,
        transverse_legs_x=column_hoop_legs_x,
        transverse_legs_y=column_hoop_legs_y,
        longitudinal_bar_diameter_m=column_bar_diameter,
        longitudinal_bar_area_m2=column_bar_area,
        bar_positions=column_bar_layout,
        transverse_fy_kn_m2=column_transverse_fy,
        **common,
    )
    ec = 2.0 * fc / abs(cover_parameters[1])
    shear_modulus = ec / (2.0 * 1.20)

    cover_base, beam_core_base, column_core_base = 11, 12, 13
    cover, beam_core, column_core = 1, 2, 4
    beam_steel_base, beam_steel = 31, 3
    column_steel_base, column_steel = 51, 5
    ops.uniaxialMaterial("Concrete01", cover_base, *cover_parameters)
    ops.uniaxialMaterial(
        "Concrete04",
        beam_core_base,
        -beam_confined.fcc_kn_m2,
        -beam_confined.eps_cc,
        -beam_confined.eps_cu,
        beam_confined.ec_kn_m2,
    )
    ops.uniaxialMaterial(
        "Concrete04",
        column_core_base,
        -column_confined.fcc_kn_m2,
        -column_confined.eps_cc,
        -column_confined.eps_cu,
        column_confined.ec_kn_m2,
    )
    failure_at_cu = bool(
        model.get(
            "concrete_failure_at_epsilon_cu",
            model.get("concrete_failure_at_ultimate_strain", True),
        )
    )
    if failure_at_cu:
        ops.uniaxialMaterial(
            "MinMax", cover, cover_base, "-min", cover_parameters[3]
        )
        ops.uniaxialMaterial(
            "MinMax", beam_core, beam_core_base, "-min", -beam_confined.eps_cu
        )
        ops.uniaxialMaterial(
            "MinMax",
            column_core,
            column_core_base,
            "-min",
            -column_confined.eps_cu,
        )
    else:
        cover, beam_core, column_core = (
            cover_base,
            beam_core_base,
            column_core_base,
        )
    define_hysteretic_steel_material(
        ops,
        base_material_tag=beam_steel_base,
        material_tag=beam_steel,
        fy_kn_m2=fy,
        model_config=model,
    )
    define_hysteretic_steel_material(
        ops,
        base_material_tag=column_steel_base,
        material_tag=column_steel,
        fy_kn_m2=fy,
        model_config=model,
    )
    _rc_section(
        ops,
        section_tag=101,
        concrete_core_tag=beam_core,
        concrete_cover_tag=cover,
        steel_tag=beam_steel,
        width_m=beam_width,
        depth_m=beam_depth,
        clear_cover_m=clear_cover,
        transverse_bar_diameter_m=beam_hoop_diameter,
        bar_area_m2=beam_bar_area,
        top_bar_count=beam_top_count,
        bottom_bar_count=beam_bottom_count,
        side_bar_count_each=beam_side_count,
        shear_modulus_kn_m2=shear_modulus,
        fiber_mesh=dict(model["fiber_mesh"]),
        longitudinal_fibers=tuple(
            (row[0], row[1], row[2]) for row in beam_fibers
        ),
    )
    _rc_section(
        ops,
        section_tag=102,
        concrete_core_tag=column_core,
        concrete_cover_tag=cover,
        steel_tag=column_steel,
        width_m=column_width,
        depth_m=column_depth,
        clear_cover_m=clear_cover,
        transverse_bar_diameter_m=column_hoop_diameter,
        bar_area_m2=column_bar_area,
        top_bar_count=0,
        bottom_bar_count=0,
        perimeter_bar_count=column_bar_count,
        shear_modulus_kn_m2=shear_modulus,
        fiber_mesh=dict(model["fiber_mesh"]),
    )
    _elastic_rectangular_section(
        ops,
        section_tag=111,
        elastic_modulus_kn_m2=ec,
        shear_modulus_kn_m2=shear_modulus,
        width_m=beam_width,
        depth_m=beam_depth,
    )
    _elastic_rectangular_section(
        ops,
        section_tag=112,
        elastic_modulus_kn_m2=ec,
        shear_modulus_kn_m2=shear_modulus,
        width_m=column_width,
        depth_m=column_depth,
    )
    return {
        "beam_width_m": beam_width,
        "beam_depth_m": beam_depth,
        "column_width_m": column_width,
        "column_depth_m": column_depth,
        "elastic_modulus_kn_m2": ec,
        "shear_modulus_kn_m2": shear_modulus,
        "beam_torsional_constant_m4": (
            rectangular_torsional_constant(beam_width, beam_depth)
        ),
    }


def _group_rows(
    number_of_bays: int,
    *,
    split_edges: bool = False,
    preserve_all: bool = False,
) -> tuple[tuple[str, int, tuple[int, ...], int], ...]:
    if preserve_all:
        return tuple(
            (
                f"frame_line_{line}",
                1,
                (line,),
                1 if line in {0, number_of_bays} else 2,
            )
            for line in range(number_of_bays + 1)
        )
    if split_edges:
        low = ("edge_low", 1, (0,), 1)
        high = ("edge_high", 1, (number_of_bays,), 1)
        if number_of_bays == 1:
            return (low, high)
        return (
            low,
            (
                "interior",
                number_of_bays - 1,
                tuple(range(1, number_of_bays)),
                2,
            ),
            high,
        )
    edge = ("edge", 2, (0, number_of_bays), 1)
    if number_of_bays == 1:
        return (edge,)
    return (
        edge,
        (
            "interior",
            number_of_bays - 1,
            tuple(range(1, number_of_bays)),
            2,
        ),
    )


def _parallel_section(
    tag: int, source_tag: int, multiplicity: int
) -> int:
    if multiplicity == 1:
        return source_tag
    _ops().section("Parallel", tag, *([source_tag] * multiplicity))
    return tag


def build_grouped_frame_model(
    artifact: FishboneArtifact,
    building: dict[str, Any],
    config: dict[str, Any],
    direction: str,
    *,
    orthogonal_gravity_moments_knm: np.ndarray | None = None,
    orthogonal_gravity_vertical_kn: np.ndarray | None = None,
    preserve_all_frame_lines: bool = False,
    include_elastic_transverse_beams: bool = False,
    include_nonlinear_transverse_beams: bool = False,
    include_transverse_torsion: bool = True,
) -> tuple[FishboneModelInfo, GroupedFrameAudit]:
    """Build one high-fidelity grouped planar frame from the Full3D rules."""

    artifact.validate()
    normalized = direction.upper()
    if normalized not in {"X", "Y"}:
        raise ValueError("Grouped frame direction must be X or Y")
    if str(building["building_id"]) != artifact.building_id:
        raise ValueError("Building and Fishbone artifact IDs differ")
    if str(building.get("model_hash", "")) != artifact.source_model_hash:
        raise ValueError("Grouped frame source model hash is stale")
    stories = int(building["stories"])
    bays = int(building["number_of_bays"])
    if stories != artifact.story_count or bays != artifact.number_of_bays:
        raise ValueError("Grouped frame topology differs from the artifact")
    grid = bays + 1
    height = float(building["story_height_m"])
    bay = float(building["bay_width_m"])
    ops = _ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    sections = _define_source_sections(building, config)
    # Full3D local-axis conventions are retained verbatim.
    ops.geomTransf("PDelta", 1, 1.0, 0.0, 0.0)
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)

    active_dof = 1 if normalized == "X" else 2
    floor_nodes: list[int] = []
    for floor in range(1, stories + 1):
        tag = 100 + floor
        floor_nodes.append(tag)
        ops.node(tag, 0.0, 0.0, floor * height)
        fixity = [1, 1, 1, 1, 1, 1]
        fixity[active_dof - 1] = 0
        ops.fix(tag, *fixity)
        mass = (
            artifact.floor_masses_x_kn_s2_m[floor - 1]
            if normalized == "X"
            else artifact.floor_masses_y_kn_s2_m[floor - 1]
        )
        mass_row = [1.0e-12] * 6
        mass_row[active_dof - 1] = mass
        ops.mass(tag, *mass_row)

    layout = plastic_hinge_layout(building, config)
    integration = str(config["model"]["beam_integration"])
    max_iterations = int(
        config["model"].get("force_beam_column_max_iterations", 100)
    )
    tolerance = float(
        config["model"].get("force_beam_column_tolerance", 1.0e-8)
    )
    signed_gravity_moments = (
        None
        if orthogonal_gravity_moments_knm is None
        else np.asarray(orthogonal_gravity_moments_knm, dtype=float)
    )
    if signed_gravity_moments is not None and signed_gravity_moments.shape != (
        stories,
        grid,
        grid,
    ):
        raise ValueError("Orthogonal gravity-moment grid is inconsistent")
    signed_gravity_vertical = (
        None
        if orthogonal_gravity_vertical_kn is None
        else np.asarray(orthogonal_gravity_vertical_kn, dtype=float)
    )
    if signed_gravity_vertical is not None and signed_gravity_vertical.shape != (
        stories,
        grid,
        grid,
    ):
        raise ValueError("Orthogonal gravity-force grid is inconsistent")
    if (signed_gravity_moments is None) != (signed_gravity_vertical is None):
        raise ValueError(
            "Exact orthogonal gravity moment and vertical force must be paired"
        )
    group_rows = _group_rows(
        bays,
        split_edges=signed_gravity_moments is not None,
        preserve_all=preserve_all_frame_lines,
    )
    base_nodes: list[int] = []
    column_by_story: list[list[int]] = [[] for _ in range(stories)]
    beam_by_story: list[list[int]] = [[] for _ in range(stories)]
    maximum_lp_spread = 0.0
    element_tag = 1
    if include_elastic_transverse_beams and include_nonlinear_transverse_beams:
        raise ValueError("Choose elastic or nonlinear transverse beams, not both")
    spatial_lattice = bool(
        include_elastic_transverse_beams
        or include_nonlinear_transverse_beams
    )
    if spatial_lattice and (
        not preserve_all_frame_lines
        or any(row[1] != 1 for row in group_rows)
        or tuple(row[2][0] for row in group_rows) != tuple(range(grid))
    ):
        raise ValueError(
            "Spatial transverse beams require every physical frame line"
        )

    def column_source_indices(active_index: int, parallel_lines: tuple[int, ...]) -> tuple[int, ...]:
        if normalized == "X":
            return tuple(active_index * grid + line for line in parallel_lines)
        return tuple(line * grid + active_index for line in parallel_lines)

    def beam_source_indices(active_bay: int, parallel_lines: tuple[int, ...]) -> tuple[int, ...]:
        if normalized == "X":
            return tuple(line * bays + active_bay for line in parallel_lines)
        offset = bays * grid
        return tuple(offset + line * bays + active_bay for line in parallel_lines)

    for group_index, (name, multiplicity, parallel_lines, adjacency) in enumerate(group_rows):
        group_offset = 1_000_000 * (group_index + 1)
        beam_section = _parallel_section(
            group_offset + 101, 101, multiplicity
        )
        column_section = _parallel_section(
            group_offset + 102, 102, multiplicity
        )
        beam_elastic = _parallel_section(
            group_offset + 111, 111, multiplicity
        )
        column_elastic = _parallel_section(
            group_offset + 112, 112, multiplicity
        )
        node_rows: list[list[int]] = []
        for level in range(stories + 1):
            level_nodes: list[int] = []
            for active_index in range(grid):
                tag = group_offset + level * 1000 + active_index + 1
                coordinate = active_index * bay
                transverse_coordinate = float(np.mean(parallel_lines)) * bay
                if normalized == "X":
                    # Coincident frame groups permit an exact zeroLength
                    # representation of the transverse-beam torsional links.
                    ops.node(
                        tag,
                        coordinate,
                        transverse_coordinate if spatial_lattice else 0.0,
                        level * height,
                    )
                    plane_fixity = (
                        (0, 1, 0, 0, 0, 1)
                        if signed_gravity_moments is not None or spatial_lattice
                        else (0, 1, 0, 1, 0, 1)
                    )
                else:
                    ops.node(
                        tag,
                        transverse_coordinate if spatial_lattice else 0.0,
                        coordinate,
                        level * height,
                    )
                    plane_fixity = (
                        (1, 0, 0, 0, 0, 1)
                        if signed_gravity_moments is not None or spatial_lattice
                        else (1, 0, 0, 0, 1, 1)
                    )
                if level == 0:
                    ops.fix(tag, 1, 1, 1, 1, 1, 1)
                    base_nodes.append(tag)
                else:
                    ops.fix(tag, *plane_fixity)
                    ops.equalDOF(floor_nodes[level - 1], tag, active_dof)
                level_nodes.append(tag)
            node_rows.append(level_nodes)

        for story in range(stories):
            for active_index in range(grid):
                indices = column_source_indices(active_index, parallel_lines)
                pairs = [
                    layout.column_hinge_lengths_m[story][index]
                    for index in indices
                ]
                values_i = [pair[0] for pair in pairs]
                values_j = [pair[1] for pair in pairs]
                maximum_lp_spread = max(
                    maximum_lp_spread,
                    max(values_i) - min(values_i),
                    max(values_j) - min(values_j),
                )
                hinge_i = float(np.median(values_i))
                hinge_j = float(np.median(values_j))
                plastic_hinge_elastic_interior_length_m(
                    member_length_m=height,
                    hinge_length_i_m=hinge_i,
                    hinge_length_j_m=hinge_j,
                    integration=integration,
                )
                integration_tag = group_offset + 100_000 + element_tag
                ops.beamIntegration(
                    integration,
                    integration_tag,
                    column_section,
                    hinge_i,
                    column_section,
                    hinge_j,
                    column_elastic,
                )
                ops.element(
                    "forceBeamColumn",
                    element_tag,
                    node_rows[story][active_index],
                    node_rows[story + 1][active_index],
                    1,
                    integration_tag,
                    "-iter",
                    max_iterations,
                    tolerance,
                )
                column_by_story[story].append(element_tag)
                element_tag += 1
            for active_bay in range(bays):
                indices = beam_source_indices(active_bay, parallel_lines)
                pairs = [
                    layout.beam_hinge_lengths_m[story][index]
                    for index in indices
                ]
                values_i = [pair[0] for pair in pairs]
                values_j = [pair[1] for pair in pairs]
                maximum_lp_spread = max(
                    maximum_lp_spread,
                    max(values_i) - min(values_i),
                    max(values_j) - min(values_j),
                )
                hinge_i = float(np.median(values_i))
                hinge_j = float(np.median(values_j))
                plastic_hinge_elastic_interior_length_m(
                    member_length_m=bay,
                    hinge_length_i_m=hinge_i,
                    hinge_length_j_m=hinge_j,
                    integration=integration,
                )
                integration_tag = group_offset + 100_000 + element_tag
                ops.beamIntegration(
                    integration,
                    integration_tag,
                    beam_section,
                    hinge_i,
                    beam_section,
                    hinge_j,
                    beam_elastic,
                )
                ops.element(
                    "forceBeamColumn",
                    element_tag,
                    node_rows[story + 1][active_bay],
                    node_rows[story + 1][active_bay + 1],
                    2,
                    integration_tag,
                    "-iter",
                    max_iterations,
                    tolerance,
                )
                beam_by_story[story].append(element_tag)
                element_tag += 1

    elastic_transverse_beam_elements: list[list[int]] = [
        [] for _ in range(stories)
    ]
    nonlinear_transverse_beam_elements: list[list[int]] = [
        [] for _ in range(stories)
    ]
    if spatial_lattice:
        beam_area = float(sections["beam_width_m"]) * float(
            sections["beam_depth_m"]
        )
        beam_iy = (
            float(sections["beam_width_m"])
            * float(sections["beam_depth_m"]) ** 3
            / 12.0
        )
        beam_iz = (
            float(sections["beam_depth_m"])
            * float(sections["beam_width_m"]) ** 3
            / 12.0
        )
        for line in range(bays):
            for story in range(stories):
                floor = story + 1
                for active_index in range(grid):
                    first_node = (
                        1_000_000 * (line + 1)
                        + floor * 1000
                        + active_index
                        + 1
                    )
                    second_node = (
                        1_000_000 * (line + 2)
                        + floor * 1000
                        + active_index
                        + 1
                    )
                    tag = (
                        8_000_000
                        + line * 100_000
                        + floor * 100
                        + active_index
                    )
                    if include_elastic_transverse_beams:
                        ops.element(
                            "elasticBeamColumn",
                            tag,
                            first_node,
                            second_node,
                            beam_area,
                            float(sections["elastic_modulus_kn_m2"]),
                            float(sections["shear_modulus_kn_m2"]),
                            float(sections["beam_torsional_constant_m4"]),
                            beam_iy,
                            beam_iz,
                            2,
                        )
                        elastic_transverse_beam_elements[story].append(tag)
                    else:
                        if normalized == "X":
                            source_index = (
                                bays * grid
                                + active_index * bays
                                + line
                            )
                        else:
                            source_index = active_index * bays + line
                        hinge_i, hinge_j = (
                            layout.beam_hinge_lengths_m[story][source_index]
                        )
                        plastic_hinge_elastic_interior_length_m(
                            member_length_m=bay,
                            hinge_length_i_m=hinge_i,
                            hinge_length_j_m=hinge_j,
                            integration=integration,
                        )
                        integration_tag = 70_000_000 + tag
                        ops.beamIntegration(
                            integration,
                            integration_tag,
                            101,
                            hinge_i,
                            101,
                            hinge_j,
                            111,
                        )
                        ops.element(
                            "forceBeamColumn",
                            tag,
                            first_node,
                            second_node,
                            2,
                            integration_tag,
                            "-iter",
                            max_iterations,
                            tolerance,
                        )
                        nonlinear_transverse_beam_elements[story].append(tag)

    transverse_torsion_elements: list[int] = []
    transverse_torsion_stiffness = 0.0
    if (
        len(group_rows) >= 2
        and not spatial_lattice
        and include_transverse_torsion
    ):
        # Under X sway, omitted Y beams couple the RY rotations of adjacent
        # X frame lines through elastic torsion (and vice versa).  All
        # interior frame lines have identical rotation by symmetry, so only
        # the two edge-to-interior boundary segments deform.  Their stiffness
        # is therefore 2GJ/L in parallel.  This term is derived directly from
        # the source beam section and introduces no fitted parameter.
        single_link_stiffness = (
            float(sections["shear_modulus_kn_m2"])
            * float(sections["beam_torsional_constant_m4"])
            / bay
        )
        # The legacy two-group topology combines both physical edges, hence
        # its one link represents two boundary beams in parallel.  With
        # signed gravity moments the edges remain separate and each adjacent
        # group pair represents one physical boundary beam.
        transverse_torsion_stiffness = (
            2.0 * single_link_stiffness
            if len(group_rows) == 2 and signed_gravity_moments is None
            else single_link_stiffness
        )
        rotation_dof = 5 if normalized == "X" else 4
        for link_index in range(len(group_rows) - 1):
            torsion_material = 9_100_000 + link_index
            ops.uniaxialMaterial(
                "Elastic", torsion_material, transverse_torsion_stiffness
            )
            for floor in range(1, stories + 1):
                for active_index in range(grid):
                    first_node = (
                        1_000_000 * (link_index + 1)
                        + floor * 1000
                        + active_index
                        + 1
                    )
                    second_node = (
                        1_000_000 * (link_index + 2)
                        + floor * 1000
                        + active_index
                        + 1
                    )
                    tag = (
                        9_500_000
                        + link_index * 100_000
                        + floor * 100
                        + active_index
                    )
                    ops.element(
                        "zeroLength",
                        tag,
                        first_node,
                        second_node,
                        "-mat",
                        torsion_material,
                        "-dir",
                        rotation_dof,
                    )
                    transverse_torsion_elements.append(tag)

    live_fraction = float(config["model"]["live_load_mass_fraction"])
    density = float(config["model"]["concrete_density_kn_m3"])
    area_load = float(building["dead_load_kn_m2"]) + live_fraction * float(
        building["live_load_kn_m2"]
    )
    beam_self_weight = (
        float(sections["beam_width_m"])
        * float(sections["beam_depth_m"])
        * density
    )
    column_segment_weight = (
        height
        * float(sections["column_width_m"])
        * float(sections["column_depth_m"])
        * density
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    applied_floor = 0.0
    beam_cursor = [0 for _ in range(stories)]
    for group_index, (_name, multiplicity, _lines, adjacency) in enumerate(group_rows):
        group_offset = 1_000_000 * (group_index + 1)
        active_uniform = (
            area_load * adjacency * bay / 4.0 + beam_self_weight
        ) * multiplicity
        for story in range(stories):
            for active_bay in range(bays):
                tag = beam_by_story[story][
                    group_index * bays + active_bay
                ]
                ops.eleLoad(
                    "-ele", tag, "-type", "-beamUniform", 0.0,
                    -active_uniform, 0.0,
                )
                applied_floor += active_uniform * bay / stories
            for active_index in range(grid):
                active_adjacency = 1 if active_index in {0, bays} else 2
                orthogonal_uniform = (
                    area_load * active_adjacency * bay / 4.0
                    + beam_self_weight
                )
                missing_orthogonal = (
                    0.0
                    if spatial_lattice
                    else multiplicity
                    * adjacency
                    * orthogonal_uniform
                    * bay
                    / 2.0
                )
                if signed_gravity_vertical is None:
                    vertical = -(
                        missing_orthogonal
                        + multiplicity * column_segment_weight
                    )
                elif normalized == "X":
                    vertical = float(
                        np.sum(
                            signed_gravity_vertical[
                                story, active_index, list(_lines)
                            ]
                        )
                    ) - multiplicity * column_segment_weight
                else:
                    vertical = float(
                        np.sum(
                            signed_gravity_vertical[
                                story, list(_lines), active_index
                            ]
                        )
                    ) - multiplicity * column_segment_weight
                orthogonal_moment = 0.0
                if signed_gravity_moments is not None:
                    if normalized == "X":
                        orthogonal_moment = float(
                            np.sum(
                                signed_gravity_moments[
                                    story, active_index, list(_lines)
                                ]
                            )
                        )
                    else:
                        orthogonal_moment = float(
                            np.sum(
                                signed_gravity_moments[
                                    story, list(_lines), active_index
                                ]
                            )
                        )
                node = group_offset + (story + 1) * 1000 + active_index + 1
                if normalized == "X":
                    ops.load(
                        node,
                        0.0,
                        0.0,
                        vertical,
                        orthogonal_moment,
                        0.0,
                        0.0,
                    )
                else:
                    ops.load(
                        node,
                        0.0,
                        0.0,
                        vertical,
                        0.0,
                        orthogonal_moment,
                        0.0,
                    )
                applied_floor += -vertical / stories

    if spatial_lattice:
        transverse_by_story = (
            elastic_transverse_beam_elements
            if include_elastic_transverse_beams
            else nonlinear_transverse_beam_elements
        )
        for story, tags in enumerate(transverse_by_story):
            cursor = 0
            for _line in range(bays):
                for active_index in range(grid):
                    active_adjacency = (
                        1 if active_index in {0, bays} else 2
                    )
                    orthogonal_uniform = (
                        area_load * active_adjacency * bay / 4.0
                        + beam_self_weight
                    )
                    ops.eleLoad(
                        "-ele",
                        tags[cursor],
                        "-type",
                        "-beamUniform",
                        0.0,
                        -orthogonal_uniform,
                        0.0,
                    )
                    cursor += 1

    # applied_floor was accumulated over all storeys and normalized inside
    # the loop; recompute explicitly to avoid depending on story count.
    area_floor = area_load * (bays * bay) ** 2
    beam_floor = 2 * bays * grid * bay * beam_self_weight
    column_floor = grid**2 * column_segment_weight
    applied_per_floor = area_floor + beam_floor + column_floor
    expected_per_floor = float(artifact.gravity_floor_loads_kn[0])
    if not math.isclose(
        applied_per_floor,
        expected_per_floor,
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise ValueError("Grouped frame gravity source differs from Full3D")
    expected_gravity = float(sum(artifact.gravity_floor_loads_kn))
    applied_gravity = applied_per_floor * stories
    info = FishboneModelInfo(
        base_node=base_nodes[0],
        pdelta_base_node=base_nodes[0],
        floor_nodes=tuple(floor_nodes),
        column_elements=tuple(tag for rows in column_by_story for tag in rows),
        beam_elements_x=(
            tuple(tag for rows in beam_by_story for tag in rows)
            if normalized == "X"
            else ()
        ),
        beam_elements_y=(
            tuple(tag for rows in beam_by_story for tag in rows)
            if normalized == "Y"
            else ()
        ),
        column_hinge_elements=(),
        beam_hinge_elements=(),
        pdelta_elements=(),
        floor_elevations_m=artifact.floor_elevations_m,
        story_heights_m=artifact.story_heights_m,
        gravity_applied_kn=applied_gravity,
    )
    audit = GroupedFrameAudit(
        direction=normalized,
        group_names=tuple(row[0] for row in group_rows),
        group_multiplicities=tuple(row[1] for row in group_rows),
        group_source_line_indices=tuple(row[2] for row in group_rows),
        physical_frame_line_count=grid,
        reduced_frame_group_count=len(group_rows),
        source_column_count=grid**2 * stories,
        reduced_column_element_count=len(info.column_elements),
        source_active_beam_count=bays * grid * stories,
        reduced_beam_element_count=len(
            info.beam_elements_x or info.beam_elements_y
        ),
        expected_gravity_kn=expected_gravity,
        applied_gravity_kn=applied_gravity,
        maximum_hinge_length_group_spread_m=maximum_lp_spread,
        transverse_torsion_element_count=len(transverse_torsion_elements),
        transverse_torsion_stiffness_knm_rad=transverse_torsion_stiffness,
        elastic_transverse_beam_element_count=sum(
            len(row) for row in elastic_transverse_beam_elements
        ),
        nonlinear_transverse_beam_element_count=sum(
            len(row) for row in nonlinear_transverse_beam_elements
        ),
    )
    return info, audit


def extract_full3d_orthogonal_gravity_actions(
    building: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return exact omitted-beam nodal actions after Full3D gravity.

    This is a one-time physics preprocessor, not a nonlinear calibration.  X
    reduction omits Y beams, whose equivalent vertical and global-MX nodal
    loads are retained; Y reduction analogously retains vertical and
    global-MY actions from X beams.  The negative element resisting force is
    the load that the removed member applies to its retained end node.
    """

    import time

    started = time.perf_counter()
    info = build_full3d_model(building, config)
    build_runtime = time.perf_counter() - started
    equilibrium_error = run_full3d_gravity(info)
    gravity_runtime = time.perf_counter() - started - build_runtime
    bays = int(building["number_of_bays"])
    grid = bays + 1
    stories = int(building["stories"])
    x_beam_count = bays * grid
    output = {
        "moment_X": np.zeros((stories, grid, grid), dtype=float),
        "moment_Y": np.zeros((stories, grid, grid), dtype=float),
        "vertical_X": np.zeros((stories, grid, grid), dtype=float),
        "vertical_Y": np.zeros((stories, grid, grid), dtype=float),
        "column_axial_end_kn": np.zeros(
            (stories, grid, grid, 2), dtype=float
        ),
    }
    ops = _ops()
    for story, tags in enumerate(info.beam_element_tags_by_storey):
        node_index = {
            int(node): index for index, node in enumerate(info.floor_nodes[story])
        }
        for direction, selected, moment_indices in (
            ("X", tags[x_beam_count:], (3, 9)),
            ("Y", tags[:x_beam_count], (4, 10)),
        ):
            for tag in selected:
                nodes = tuple(int(node) for node in ops.eleNodes(tag))
                force = tuple(float(value) for value in ops.eleForce(tag))
                for node, force_index, moment_index in zip(
                    nodes, (2, 8), moment_indices
                ):
                    flat = node_index[node]
                    ix, iy = divmod(flat, grid)
                    output[f"vertical_{direction}"][story, ix, iy] += (
                        -force[force_index]
                    )
                    output[f"moment_{direction}"][story, ix, iy] += (
                        -force[moment_index]
                    )
        for column_index, tag in enumerate(
            info.column_element_tags_by_storey[story]
        ):
            force = tuple(float(value) for value in ops.eleForce(tag))
            ix, iy = divmod(column_index, grid)
            output["column_axial_end_kn"][story, ix, iy, :] = (
                abs(force[2]),
                abs(force[8]),
            )
    vertical_x = float(np.sum(output["vertical_X"]))
    vertical_y = float(np.sum(output["vertical_Y"]))
    return output, {
        "build_runtime_s": build_runtime,
        "gravity_runtime_s": gravity_runtime,
        "gravity_equilibrium_relative_error": float(equilibrium_error),
        "equivalent_omitted_vertical_x_kn": vertical_x,
        "equivalent_omitted_vertical_y_kn": vertical_y,
        "equivalent_vertical_direction_relative_difference": abs(
            vertical_x - vertical_y
        ) / max(abs(vertical_x), abs(vertical_y), 1.0),
    }


def grouped_frame_gravity_column_end_axials_kn(
    info: FishboneModelInfo,
    audit: GroupedFrameAudit,
) -> np.ndarray:
    """Read grouped-frame post-gravity column end compression magnitudes."""

    grid = audit.physical_frame_line_count
    per_story = audit.reduced_frame_group_count * grid
    tags = tuple(int(tag) for tag in info.column_elements)
    if len(tags) % per_story:
        raise ValueError("Grouped-frame column topology is inconsistent")
    stories = len(tags) // per_story
    result = np.zeros(
        (stories, audit.reduced_frame_group_count, grid, 2), dtype=float
    )
    ops = _ops()
    cursor = 0
    for story in range(stories):
        for group in range(audit.reduced_frame_group_count):
            for active_index in range(grid):
                force = tuple(float(value) for value in ops.eleForce(tags[cursor]))
                result[story, group, active_index, :] = (
                    abs(force[2]),
                    abs(force[8]),
                )
                cursor += 1
    return result


def grouped_source_column_end_axials_kn(
    source: np.ndarray,
    audit: GroupedFrameAudit,
) -> np.ndarray:
    """Aggregate Full3D column axials in the exact reduced group ordering."""

    values = np.asarray(source, dtype=float)
    grid = audit.physical_frame_line_count
    if values.ndim != 4 or values.shape[1:] != (grid, grid, 2):
        raise ValueError("Full3D column axial-force grid is inconsistent")
    rows = []
    for lines in audit.group_source_line_indices:
        if audit.direction == "X":
            rows.append(np.sum(values[:, :, list(lines), :], axis=2))
        else:
            rows.append(np.sum(values[:, list(lines), :, :], axis=1))
    return np.stack(rows, axis=1)


def run_grouped_frame_gravity(
    info: FishboneModelInfo, audit: GroupedFrameAudit
) -> float:
    """Commit gravity and return base-reaction equilibrium error."""

    ops = _ops()
    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 80, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("Grouped frame gravity analysis did not converge")
    ops.reactions()
    # Query all fixed nodes rather than the first support retained in the
    # compatibility dataclass.
    reaction = 0.0
    for node in ops.getNodeTags():
        if int(node) >= 1_000_000 and int(node) % 1000 <= 100:
            reaction += float(ops.nodeReaction(int(node), 3))
    error = abs(abs(reaction) - audit.expected_gravity_kn) / max(
        audit.expected_gravity_kn, 1.0
    )
    ops.loadConst("-time", 0.0)
    return float(error)


def grouped_frame_directional_modes(
    artifact: FishboneArtifact,
    info: FishboneModelInfo,
    direction: str,
    *,
    requested_modes: int | None = None,
) -> dict[str, Any]:
    """Identify the first three mass-participating modes in one direction."""

    normalized = direction.upper()
    dof = 1 if normalized == "X" else 2
    masses = np.asarray(
        artifact.floor_masses_x_kn_s2_m
        if normalized == "X"
        else artifact.floor_masses_y_kn_s2_m,
        dtype=float,
    )
    count = int(requested_modes or artifact.story_count)
    values = np.asarray(
        _ops().eigen("-fullGenLapack", count), dtype=float
    )
    rows = []
    total_mass = float(np.sum(masses))
    for mode_index, eigenvalue in enumerate(values, 1):
        if not math.isfinite(float(eigenvalue)) or eigenvalue <= 0.0:
            continue
        shape = np.asarray(
            [
                float(_ops().nodeEigenvector(node, mode_index, dof))
                for node in info.floor_nodes
            ],
            dtype=float,
        )
        denominator = float(shape @ (masses * shape))
        numerator = float(np.sum(masses * shape))
        effective = (
            numerator**2 / denominator / total_mass
            if denominator > 1.0e-18
            else 0.0
        )
        if effective <= 1.0e-8:
            continue
        scale = float(np.max(np.abs(shape)))
        normalized_shape = shape / max(scale, 1.0e-18)
        if normalized_shape[-1] < 0.0:
            normalized_shape *= -1.0
        rows.append(
            {
                "global_mode_index": mode_index,
                "period_s": 2.0 * math.pi / math.sqrt(float(eigenvalue)),
                "effective_mass_ratio": effective,
                "mode_shape": normalized_shape.tolist(),
            }
        )
        if len(rows) == min(3, artifact.story_count):
            break
    if len(rows) < min(3, artifact.story_count):
        raise RuntimeError("Grouped frame lacks three translational modes")
    return {
        "schema_version": FISHBONE_V4_GROUPED_FRAME_SCHEMA_VERSION,
        "direction": normalized,
        "modes": rows,
        "cumulative_effective_mass_ratio": float(
            sum(row["effective_mass_ratio"] for row in rows)
        ),
    }
