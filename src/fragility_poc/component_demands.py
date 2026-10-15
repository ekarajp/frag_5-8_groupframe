"""Compact member-end response envelopes for FEMA-oriented IDA diagnostics.

The nonlinear model already computes section and element responses at every
committed analysis step.  This module asks OpenSees' compiled envelope
recorders to retain minima, maxima, and absolute maxima for every beam and
column end without writing full response histories.  One compressed artifact
is produced for every fixed-IM NLTHA checkpoint.

The FEMA 356 ratios reported here are flexural component diagnostics within
the project's declared flexure-controlled model scope.  They are not a claim
that unmodelled shear, joint, bond, splice, or torsional failure modes have
been simulated.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G_STD
from .structural import _ops

COMPONENT_RESPONSE_SCHEMA_VERSION = (
    "member-end-envelope-v1-fema356-flexural-diagnostic"
)
FEMA356_REFERENCE = (
    "FEMA 356 (2000), Tables 6-7 and 6-8; primary RC frame components"
)


def _temporary_recorder_path(prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    os.close(descriptor)
    path = Path(name)
    # OpenSees recorders create/truncate their own output.  Removing the empty
    # mkstemp file avoids Windows sharing and append-semantics surprises.
    path.unlink(missing_ok=True)
    return path


def _flatten_tags(
    tags_by_storey: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return tuple(tag for storey in tags_by_storey for tag in storey)


def install_component_envelope_recorders(info: Any) -> dict[str, Any]:
    """Install compiled envelope recorders for every member and floor.

    The caller must create ground-motion time series 101 and 102 first.  All
    returned temporary paths must be finalized or cleaned even after a failed
    analysis.
    """
    ops = _ops()
    beam_tags = _flatten_tags(info.beam_element_tags_by_storey)
    column_tags = _flatten_tags(info.column_element_tags_by_storey)
    if not beam_tags or not column_tags:
        raise ValueError("Component recording requires beam and column tags")
    beam_section_count = len(info.beam_integration_points_m)
    column_section_count = len(info.column_integration_points_m)
    if beam_section_count < 2 or column_section_count < 2:
        raise ValueError("Every nonlinear member must have two end sections")

    paths: dict[str, Path] = {}
    gravity_local_forces: dict[str, dict[int, tuple[float, ...]]] = {}

    def element_recorder(
        name: str,
        tags: tuple[int, ...],
        *response: Any,
    ) -> None:
        path = _temporary_recorder_path(f"fragility_{name}_")
        paths[name] = path
        ops.recorder(
            "EnvelopeElement",
            "-file",
            str(path),
            "-precision",
            16,
            "-ele",
            *tags,
            *response,
        )

    for family, tags, section_count in (
        ("beam", beam_tags, beam_section_count),
        ("column", column_tags, column_section_count),
    ):
        family_gravity_forces: dict[int, tuple[float, ...]] = {}
        for tag in tags:
            local_force = tuple(
                float(value)
                for value in ops.eleResponse(tag, "localForce")
            )
            if len(local_force) != 12 or not all(
                math.isfinite(value) for value in local_force
            ):
                raise ValueError(
                    f"Element {tag} has an invalid post-gravity local-force "
                    "vector"
                )
            family_gravity_forces[int(tag)] = local_force
        gravity_local_forces[family] = family_gravity_forces
        element_recorder(
            f"{family}_section_i", tags, "section", 1, "deformation"
        )
        element_recorder(
            f"{family}_section_j",
            tags,
            "section",
            section_count,
            "deformation",
        )
        element_recorder(f"{family}_plastic", tags, "plasticDeformation")
        element_recorder(f"{family}_basic", tags, "basicDeformation")
        element_recorder(f"{family}_force", tags, "localForce")

    # Node acceleration under UniformExcitation is relative by default.
    # Supplying the corresponding Path time series adds the ground component
    # and therefore records absolute floor acceleration.
    for direction, dof, time_series in (("x", 1, 101), ("y", 2, 102)):
        path = _temporary_recorder_path(
            f"fragility_floor_accel_{direction}_"
        )
        paths[f"floor_accel_{direction}"] = path
        ops.recorder(
            "EnvelopeNode",
            "-file",
            str(path),
            "-precision",
            16,
            "-timeSeries",
            time_series,
            "-node",
            *info.master_nodes,
            "-dof",
            dof,
            "accel",
        )

    return {
        "schema": COMPONENT_RESPONSE_SCHEMA_VERSION,
        "paths": paths,
        "beam_tags": beam_tags,
        "column_tags": column_tags,
        "gravity_local_forces_kn": gravity_local_forces,
        "beam_section_count": beam_section_count,
        "column_section_count": column_section_count,
    }


def cleanup_component_envelope_recorders(bundle: dict[str, Any] | None) -> None:
    if not bundle:
        return
    for path in bundle.get("paths", {}).values():
        Path(path).unlink(missing_ok=True)


def _read_envelope(
    path: Path,
    *,
    item_count: int,
    response_width: int,
) -> np.ndarray:
    """Return [minimum, maximum, absolute maximum, item, response]."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Empty component envelope recorder: {path}")
    data = np.loadtxt(path, dtype=float, ndmin=2)
    expected_columns = item_count * response_width
    if data.shape != (3, expected_columns):
        raise ValueError(
            f"Unexpected envelope shape {data.shape}; "
            f"expected (3, {expected_columns}) for {path.name}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite component envelope in {path}")
    return data.reshape(3, item_count, response_width)


def _value_triplet(data: np.ndarray, index: int) -> dict[str, float]:
    return {
        "min": float(data[0, index]),
        "max": float(data[1, index]),
        "absmax": float(abs(data[2, index])),
    }


def _member_metadata(
    tags_by_storey: tuple[tuple[int, ...], ...],
    hinge_lengths_by_storey: tuple[
        tuple[tuple[float, float], ...], ...
    ],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for storey, (tags, lengths) in enumerate(
        zip(tags_by_storey, hinge_lengths_by_storey), start=1
    ):
        if len(tags) != len(lengths):
            raise ValueError("Member tags and hinge lengths are inconsistent")
        for index, (tag, hinge_pair) in enumerate(zip(tags, lengths)):
            rows[int(tag)] = {
                "storey": storey,
                "member_index_in_storey": index,
                "hinge_i_m": float(hinge_pair[0]),
                "hinge_j_m": float(hinge_pair[1]),
            }
    return rows


def _interpolate_table(
    table: dict[tuple[float, float], tuple[float, float, float]],
    x: float,
    y: float,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> dict[str, float]:
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    xc = min(max(float(x), x0), x1)
    yc = min(max(float(y), y0), y1)
    tx = (xc - x0) / max(x1 - x0, 1.0e-12)
    ty = (yc - y0) / max(y1 - y0, 1.0e-12)
    values: list[float] = []
    for state_index in range(3):
        q00 = table[(x0, y0)][state_index]
        q01 = table[(x0, y1)][state_index]
        q10 = table[(x1, y0)][state_index]
        q11 = table[(x1, y1)][state_index]
        low = q00 + ty * (q01 - q00)
        high = q10 + ty * (q11 - q10)
        values.append(float(low + tx * (high - low)))
    return dict(zip(("IO", "LS", "CP"), values))


_BEAM_FEMA_TABLE = {
    True: {
        (0.0, 3.0): (0.010, 0.020, 0.025),
        (0.0, 6.0): (0.005, 0.010, 0.020),
        (0.5, 3.0): (0.005, 0.010, 0.020),
        (0.5, 6.0): (0.005, 0.005, 0.015),
    },
    False: {
        (0.0, 3.0): (0.005, 0.010, 0.020),
        (0.0, 6.0): (0.0015, 0.005, 0.010),
        (0.5, 3.0): (0.005, 0.010, 0.010),
        (0.5, 6.0): (0.0015, 0.005, 0.005),
    },
}

_COLUMN_FEMA_TABLE = {
    True: {
        (0.1, 3.0): (0.005, 0.015, 0.020),
        (0.1, 6.0): (0.005, 0.012, 0.016),
        (0.4, 3.0): (0.003, 0.012, 0.015),
        (0.4, 6.0): (0.003, 0.010, 0.012),
    },
    False: {
        (0.1, 3.0): (0.005, 0.005, 0.006),
        (0.1, 6.0): (0.005, 0.004, 0.005),
        (0.4, 3.0): (0.002, 0.002, 0.003),
        (0.4, 6.0): (0.002, 0.002, 0.002),
    },
}


def _effective_depth_m(
    *,
    h_m: float,
    hoop_diameter_m: float,
    bar_diameter_m: float,
    cover_m: float,
) -> float:
    return max(
        float(h_m)
        - float(cover_m)
        - float(hoop_diameter_m)
        - 0.5 * float(bar_diameter_m),
        0.5 * float(h_m),
    )


def _normalized_shear_fema(
    shear_kn: float,
    *,
    b_m: float,
    d_m: float,
    fc_ksc: float,
) -> float:
    # V/(b d sqrt(fc')) evaluated in US customary units, as used by FEMA 356.
    v_lb = abs(float(shear_kn)) * 224.80894387096
    b_in = float(b_m) * 39.370078740157
    d_in = float(d_m) * 39.370078740157
    fc_psi = float(fc_ksc) * 14.223343307
    return v_lb / max(b_in * d_in * math.sqrt(fc_psi), 1.0e-12)


def _transverse_conforming(
    metadata: dict[str, Any],
    *,
    family: str,
    d_m: float,
    shear_demand_kn: float,
    fy_transverse_ksc: float,
) -> tuple[bool, float]:
    if family == "beam":
        diameter = float(metadata.get("beam_stirrup_diameter_m", 0.0))
        spacing = float(metadata.get("beam_stirrup_spacing_m", math.inf))
        legs = int(metadata.get("beam_stirrup_legs", 0))
    else:
        diameter = float(metadata.get("column_hoop_diameter_m", 0.0))
        spacing = float(metadata.get("column_hoop_spacing_m", math.inf))
        legs = min(
            int(metadata.get("column_hoop_legs_x", 0)),
            int(metadata.get("column_hoop_legs_y", 0)),
        )
    av_m2 = max(legs, 0) * math.pi * diameter**2 / 4.0
    fy_kn_m2 = float(fy_transverse_ksc) * 9.80665
    vs_kn = (
        av_m2 * fy_kn_m2 * float(d_m) / spacing
        if spacing > 0.0 and math.isfinite(spacing)
        else 0.0
    )
    conforming = (
        spacing <= float(d_m) / 3.0 + 1.0e-12
        and vs_kn >= 0.75 * abs(float(shear_demand_kn)) - 1.0e-12
    )
    return bool(conforming), float(vs_kn)


def _beam_rho_difference_ratio(
    building: dict[str, Any],
    metadata: dict[str, Any],
    *,
    d_m: float,
    config: dict[str, Any],
) -> float:
    b_m = float(building["beam_b_m"])
    top_area = float(metadata.get("beam_top_steel_area_m2", 0.0))
    bottom_area = float(metadata.get("beam_bottom_steel_area_m2", 0.0))
    rho_top = top_area / max(b_m * d_m, 1.0e-12)
    rho_bottom = bottom_area / max(b_m * d_m, 1.0e-12)
    fc_mpa = float(building["fc_ksc"]) * 0.0980665
    fy_ksc = float(
        building.get("steel_fy_ksc", config["building"]["steel_fy_ksc"])
    )
    fy_mpa = fy_ksc * 0.0980665
    es_mpa = float(
        config["model"]["steel_hysteretic"]["elastic_modulus_mpa"]
    )
    eps_y = fy_mpa / es_mpa
    beta1 = min(max(0.85 - 0.05 * max(fc_mpa - 28.0, 0.0) / 7.0, 0.65), 0.85)
    rho_bal = (
        0.85
        * beta1
        * fc_mpa
        / max(fy_mpa, 1.0e-12)
        * 0.003
        / (0.003 + eps_y)
    )
    # The two bending signs exchange tension and compression faces.  FEMA
    # capacity decreases as this ratio increases, so retain the governing sign.
    rho_differences = (
        (rho_bottom - rho_top) / max(rho_bal, 1.0e-12),
        (rho_top - rho_bottom) / max(rho_bal, 1.0e-12),
    )
    return float(max(rho_differences))


def _fema_limits(
    *,
    family: str,
    building: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any],
    shear_demand_kn: float,
    axial_compression_kn: float,
    dynamic_axial_compression_kn: float | None = None,
) -> dict[str, Any]:
    cover_m = float(
        config["model"]
        .get("material_detailing", {})
        .get("clear_cover_m", 0.04)
    )
    fc_ksc = float(building["fc_ksc"])
    if family == "beam":
        b_m = float(building["beam_b_m"])
        h_m = float(building["beam_h_m"])
        bar_diameter = float(building["beam_bar_diameter_m"])
        hoop_diameter = float(metadata.get("beam_stirrup_diameter_m", 0.0))
        transverse_fy = float(
            metadata.get(
                "beam_stirrup_fy_ksc",
                config["building"].get("stirrup_fy_ksc", 2400.0),
            )
        )
        fema_design_shear_kn = float(
            metadata.get(
                "beam_capacity_design_shear_kn",
                metadata.get(
                    "beam_exact_design_shear_kn",
                    shear_demand_kn,
                ),
            )
        )
        fema_shear_basis = (
            "catalogue_capacity_design_beam_shear"
            if "beam_capacity_design_shear_kn" in metadata
            else "catalogue_exact_design_beam_shear"
        )
    else:
        b_m = float(building["column_b_m"])
        h_m = float(building["column_h_m"])
        bar_diameter = float(building["column_bar_diameter_m"])
        hoop_diameter = float(metadata.get("column_hoop_diameter_m", 0.0))
        transverse_fy = float(
            metadata.get(
                "column_hoop_fy_ksc",
                config["building"].get("stirrup_fy_ksc", 2400.0),
            )
        )
        fema_design_shear_kn = float(
            metadata.get(
                "column_capacity_design_shear_kn",
                shear_demand_kn,
            )
        )
        fema_shear_basis = (
            "catalogue_capacity_design_column_shear"
            if "column_capacity_design_shear_kn" in metadata
            else "dynamic_envelope_fallback_missing_catalogue_shear"
        )
    d_m = _effective_depth_m(
        h_m=h_m,
        hoop_diameter_m=hoop_diameter,
        bar_diameter_m=bar_diameter,
        cover_m=cover_m,
    )
    normalized_shear = _normalized_shear_fema(
        fema_design_shear_kn,
        b_m=b_m,
        d_m=d_m,
        fc_ksc=fc_ksc,
    )
    conforming, transverse_shear_kn = _transverse_conforming(
        metadata,
        family=family,
        d_m=d_m,
        shear_demand_kn=fema_design_shear_kn,
        fy_transverse_ksc=transverse_fy,
    )
    if family == "beam":
        condition = _beam_rho_difference_ratio(
            building, metadata, d_m=d_m, config=config
        )
        limits = _interpolate_table(
            _BEAM_FEMA_TABLE[conforming],
            condition,
            normalized_shear,
            x_bounds=(0.0, 0.5),
            y_bounds=(3.0, 6.0),
        )
        axial_ratio = 0.0
    else:
        ag_m2 = float(building["column_b_m"]) * float(
            building["column_h_m"]
        )
        fc_kn_m2 = fc_ksc * 9.80665
        gravity_axial_ratio = max(float(axial_compression_kn), 0.0) / max(
            ag_m2 * fc_kn_m2, 1.0e-12
        )
        dynamic_axial_ratio = max(
            float(
                dynamic_axial_compression_kn
                if dynamic_axial_compression_kn is not None
                else axial_compression_kn
            ),
            0.0,
        ) / max(
            ag_m2 * fc_kn_m2,
            1.0e-12,
        )
        # FEMA's P/Agfc table parameter must be member-specific.  Capture the
        # exact post-gravity element force before UniformExcitation rather than
        # applying one governing catalogue ratio to every storey and grid
        # column.  The independent dynamic axial envelope is retained for
        # review but is not combined with a plastic-rotation peak that may
        # occur at a different instant.
        axial_ratio = gravity_axial_ratio
        limits = _interpolate_table(
            _COLUMN_FEMA_TABLE[conforming],
            axial_ratio,
            normalized_shear,
            x_bounds=(0.1, 0.4),
            y_bounds=(3.0, 6.0),
        )
        if axial_ratio > 0.70:
            limits = (
                {"IO": 0.0, "LS": 0.005, "CP": 0.010}
                if conforming
                else {"IO": 0.0, "LS": 0.0, "CP": 0.0}
            )
        condition = axial_ratio
    return {
        "limits_rad": limits,
        "transverse_reinforcement_conforming": conforming,
        "transverse_nominal_shear_kn": transverse_shear_kn,
        "normalized_shear_fema": normalized_shear,
        "fema_design_shear_kn": fema_design_shear_kn,
        "fema_design_shear_basis": fema_shear_basis,
        "dynamic_shear_envelope_kn": float(shear_demand_kn),
        "axial_load_ratio_p_agfc": axial_ratio,
        "dynamic_axial_envelope_ratio_p_agfc": (
            dynamic_axial_ratio if family == "column" else 0.0
        ),
        "axial_ratio_basis": (
            "post_gravity_member_local_force_after_equilibrium"
            if family == "column"
            else "not_applicable_to_beam"
        ),
        "beam_rho_difference_over_rho_bal": (
            condition if family == "beam" else None
        ),
        "effective_depth_m": d_m,
        "reference": FEMA356_REFERENCE,
        "scope": "flexure-controlled primary component diagnostic",
    }


def _residual_story_drift(info: Any) -> dict[str, Any]:
    ops = _ops()
    floor_displacements = np.asarray(
        [
            [float(ops.nodeDisp(tag, 1)), float(ops.nodeDisp(tag, 2))]
            for tag in info.master_nodes
        ],
        dtype=float,
    )
    base = np.zeros((1, 2), dtype=float)
    storey_relative = np.diff(
        np.concatenate((base, floor_displacements), axis=0), axis=0
    )
    elevations = np.asarray(info.floor_elevations_m, dtype=float)
    heights = np.diff(np.concatenate((np.asarray([0.0]), elevations)))
    components = storey_relative / heights.reshape(-1, 1)
    return {
        "floor_residual_displacements_m": floor_displacements.tolist(),
        "storey_residual_drift_components": components.tolist(),
        "maximum_absolute_residual_midr": float(np.max(np.abs(components))),
    }


def finalize_component_envelope_recorders(
    bundle: dict[str, Any],
    *,
    info: Any,
    building: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Read every envelope and return detailed rows plus a compact summary."""
    paths = {key: Path(value) for key, value in bundle["paths"].items()}
    beam_tags = tuple(bundle["beam_tags"])
    column_tags = tuple(bundle["column_tags"])
    arrays: dict[str, np.ndarray] = {}
    for family, tags in (("beam", beam_tags), ("column", column_tags)):
        count = len(tags)
        arrays[f"{family}_section_i"] = _read_envelope(
            paths[f"{family}_section_i"],
            item_count=count,
            response_width=4,
        )
        arrays[f"{family}_section_j"] = _read_envelope(
            paths[f"{family}_section_j"],
            item_count=count,
            response_width=4,
        )
        arrays[f"{family}_plastic"] = _read_envelope(
            paths[f"{family}_plastic"],
            item_count=count,
            response_width=6,
        )
        arrays[f"{family}_basic"] = _read_envelope(
            paths[f"{family}_basic"],
            item_count=count,
            response_width=6,
        )
        arrays[f"{family}_force"] = _read_envelope(
            paths[f"{family}_force"],
            item_count=count,
            response_width=12,
        )
    floor_x = _read_envelope(
        paths["floor_accel_x"],
        item_count=len(info.master_nodes),
        response_width=1,
    )
    floor_y = _read_envelope(
        paths["floor_accel_y"],
        item_count=len(info.master_nodes),
        response_width=1,
    )
    metadata = json.loads(building.get("design_metadata_json") or "{}")
    gravity_local_forces = bundle.get("gravity_local_forces_kn", {})
    family_metadata = {
        "beam": _member_metadata(
            info.beam_element_tags_by_storey,
            info.beam_hinge_lengths_m,
        ),
        "column": _member_metadata(
            info.column_element_tags_by_storey,
            info.column_hinge_lengths_m,
        ),
    }
    member_ends: list[dict[str, Any]] = []
    for family, tags in (("beam", beam_tags), ("column", column_tags)):
        section_arrays = {
            "I": arrays[f"{family}_section_i"],
            "J": arrays[f"{family}_section_j"],
        }
        plastic = arrays[f"{family}_plastic"]
        basic = arrays[f"{family}_basic"]
        force = arrays[f"{family}_force"]
        for member_index, tag in enumerate(tags):
            member = family_metadata[family][int(tag)]
            gravity_force = tuple(
                float(value)
                for value in gravity_local_forces.get(family, {}).get(
                    int(tag), ()
                )
            )
            if len(gravity_force) != 12:
                raise ValueError(
                    f"Missing post-gravity local forces for element {tag}"
                )
            for end in ("I", "J"):
                if end == "I":
                    rotation_z_index, rotation_y_index = 1, 3
                    force_offset = 0
                    axial_compression = max(
                        float(force[1, member_index, 0]), 0.0
                    )
                    gravity_axial_compression = max(
                        gravity_force[0], 0.0
                    )
                else:
                    rotation_z_index, rotation_y_index = 2, 4
                    force_offset = 6
                    axial_compression = max(
                        -float(force[0, member_index, 6]), 0.0
                    )
                    gravity_axial_compression = max(
                        -gravity_force[6], 0.0
                    )
                section = section_arrays[end][:, member_index, :]
                shear_y = _value_triplet(force[:, member_index, :], force_offset + 1)
                shear_z = _value_triplet(force[:, member_index, :], force_offset + 2)
                shear_demand = math.hypot(
                    shear_y["absmax"], shear_z["absmax"]
                )
                plastic_z = _value_triplet(
                    plastic[:, member_index, :], rotation_z_index
                )
                plastic_y = _value_triplet(
                    plastic[:, member_index, :], rotation_y_index
                )
                chord_z = _value_triplet(
                    basic[:, member_index, :], rotation_z_index
                )
                chord_y = _value_triplet(
                    basic[:, member_index, :], rotation_y_index
                )
                plastic_demand = max(
                    plastic_z["absmax"], plastic_y["absmax"]
                )
                fema = _fema_limits(
                    family=family,
                    building=building,
                    config=config,
                    metadata=metadata,
                    shear_demand_kn=shear_demand,
                    axial_compression_kn=gravity_axial_compression,
                    dynamic_axial_compression_kn=axial_compression,
                )
                ratios = {
                    state: (
                        plastic_demand / limit
                        if limit > 0.0
                        # A zero tabulated acceptance limit means any
                        # nonzero plastic rotation is unacceptable. Keep the
                        # serialized diagnostic finite while preserving that
                        # governing meaning.
                        else (1.0e12 if plastic_demand > 0.0 else 0.0)
                    )
                    for state, limit in fema["limits_rad"].items()
                }
                member_ends.append(
                    {
                        "member_type": family,
                        "element_tag": int(tag),
                        "storey": int(member["storey"]),
                        "member_index_in_storey": int(
                            member["member_index_in_storey"]
                        ),
                        "end": end,
                        "hinge_length_m": float(
                            member[f"hinge_{end.lower()}_m"]
                        ),
                        "section_axial_strain": _value_triplet(section, 0),
                        "section_curvature_z_per_m": _value_triplet(
                            section, 1
                        ),
                        "section_curvature_y_per_m": _value_triplet(
                            section, 2
                        ),
                        "section_torsional_deformation": _value_triplet(
                            section, 3
                        ),
                        "plastic_rotation_z_rad": plastic_z,
                        "plastic_rotation_y_rad": plastic_y,
                        "plastic_rotation_absmax_rad": plastic_demand,
                        "chord_rotation_z_rad": chord_z,
                        "chord_rotation_y_rad": chord_y,
                        "chord_rotation_absmax_rad": max(
                            chord_z["absmax"], chord_y["absmax"]
                        ),
                        "axial_force_kn": _value_triplet(
                            force[:, member_index, :], force_offset
                        ),
                        "axial_compression_envelope_kn": axial_compression,
                        "post_gravity_axial_compression_kn": (
                            gravity_axial_compression
                        ),
                        "shear_y_kn": shear_y,
                        "shear_z_kn": shear_z,
                        "shear_resultant_absmax_kn": shear_demand,
                        "torsion_knm": _value_triplet(
                            force[:, member_index, :], force_offset + 3
                        ),
                        "moment_y_knm": _value_triplet(
                            force[:, member_index, :], force_offset + 4
                        ),
                        "moment_z_knm": _value_triplet(
                            force[:, member_index, :], force_offset + 5
                        ),
                        "fema356": fema,
                        "fema356_flexural_utilization": ratios,
                    }
                )

    governing: dict[str, dict[str, Any]] = {}
    for state in ("IO", "LS", "CP"):
        row = max(
            member_ends,
            key=lambda item: float(
                item["fema356_flexural_utilization"][state]
            ),
        )
        governing[state] = {
            "maximum_utilization": float(
                row["fema356_flexural_utilization"][state]
            ),
            "member_type": row["member_type"],
            "element_tag": row["element_tag"],
            "storey": row["storey"],
            "end": row["end"],
            "plastic_rotation_demand_rad": row[
                "plastic_rotation_absmax_rad"
            ],
            "plastic_rotation_limit_rad": row["fema356"]["limits_rad"][
                state
            ],
        }
    floor_rows = []
    for floor_index, node_tag in enumerate(info.master_nodes):
        x = _value_triplet(floor_x[:, :, 0], floor_index)
        y = _value_triplet(floor_y[:, :, 0], floor_index)
        floor_rows.append(
            {
                "floor": floor_index + 1,
                "master_node": int(node_tag),
                "absolute_acceleration_x_m_s2": x,
                "absolute_acceleration_y_m_s2": y,
                "absolute_acceleration_resultant_absmax_g": (
                    math.hypot(x["absmax"], y["absmax"]) / G_STD
                ),
                "absolute_acceleration_max_component_g": (
                    max(x["absmax"], y["absmax"]) / G_STD
                ),
            }
        )
    residual = _residual_story_drift(info)
    summary = {
        "member_end_count": len(member_ends),
        "beam_end_count": sum(
            row["member_type"] == "beam" for row in member_ends
        ),
        "column_end_count": sum(
            row["member_type"] == "column" for row in member_ends
        ),
        "fema356_flexural_governing": governing,
        "peak_floor_acceleration_max_component_g": max(
            (
                row["absolute_acceleration_max_component_g"]
                for row in floor_rows
            ),
            default=0.0,
        ),
        "peak_floor_acceleration_resultant_g": max(
            (
                row["absolute_acceleration_resultant_absmax_g"]
                for row in floor_rows
            ),
            default=0.0,
        ),
        **residual,
    }
    return {
        "schema": COMPONENT_RESPONSE_SCHEMA_VERSION,
        "fema_reference": FEMA356_REFERENCE,
        "member_ends": member_ends,
        "floor_acceleration_envelopes": floor_rows,
        "summary": summary,
    }


def atomic_write_component_response(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def component_response_is_complete(result: dict[str, Any]) -> bool:
    if not bool(result.get("component_response_complete", False)):
        return False
    if result.get("component_response_schema") != COMPONENT_RESPONSE_SCHEMA_VERSION:
        return False
    path_text = str(result.get("component_response_path") or "").strip()
    if not path_text:
        return False
    path = Path(path_text)
    return path.is_file() and path.stat().st_size > 0
