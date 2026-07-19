from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import core
except ImportError:  # pragma: no cover - direct script execution fallback
    import core


INTERFERENCE_STATUSES = {"interference", "possible_interference"}


@dataclass(frozen=True)
class CollisionSettings:
    enabled: bool = True
    fail_on_interference: bool = True
    contact_tolerance: float = 1e-3
    common_volume_tolerance: float = 1e-3
    residual_good_tolerance: float = 1e-4
    residual_bad_tolerance: float = 1e-2
    search_free_rotation: bool = True
    rotation_sample_count: int = 24


@dataclass(frozen=True)
class CollisionResolution:
    transform: list[list[float]]
    accepted: bool
    adjusted: bool
    analysis: dict[str, Any]
    reason: str | None = None


def resolve_collision(
    solver: core.PairAssemblySolver,
    *,
    fixed_step_path: str | Path,
    moving_step_path: str | Path,
    transform: Sequence[Sequence[float]],
    settings: CollisionSettings,
) -> CollisionResolution:
    matrix = _as_matrix4(transform)
    if not settings.enabled:
        return CollisionResolution(_matrix_to_list(matrix), True, False, {"enabled": False})

    occ = _require_occ()
    fixed_shape = _load_step_shape(occ, Path(fixed_step_path))
    moving_source_shape = _load_step_shape(occ, Path(moving_step_path))
    moving_shape = _transform_shape(occ, moving_source_shape, matrix)

    baseline_metrics = _measure_shape_interference(
        occ,
        fixed_shape,
        moving_shape,
        contact_tolerance=settings.contact_tolerance,
        common_volume_tolerance=settings.common_volume_tolerance,
    )
    baseline_diagnosis = solver.diagnose_transform(matrix)
    rotation_hint = _infer_free_rotation_hint(solver)

    analysis: dict[str, Any] = {
        "enabled": True,
        "status": baseline_metrics["status"],
        "classification": _classify_collision(
            baseline_metrics,
            baseline_diagnosis,
            rotation_hint,
            settings=settings,
            relief=None,
        ),
        "baseline_metrics": baseline_metrics,
        "final_metrics": baseline_metrics,
        "transform_diagnosis": baseline_diagnosis,
        "rotation_hint": rotation_hint,
        "rotation_relief": None,
        "settings": {
            "contact_tolerance": float(settings.contact_tolerance),
            "common_volume_tolerance": float(settings.common_volume_tolerance),
            "residual_good_tolerance": float(settings.residual_good_tolerance),
            "residual_bad_tolerance": float(settings.residual_bad_tolerance),
            "search_free_rotation": bool(settings.search_free_rotation),
            "rotation_sample_count": int(settings.rotation_sample_count),
            "fail_on_interference": bool(settings.fail_on_interference),
        },
    }

    if baseline_metrics["status"] not in INTERFERENCE_STATUSES:
        return CollisionResolution(_matrix_to_list(matrix), True, False, analysis)

    relief = None
    if (
        settings.search_free_rotation
        and rotation_hint
        and not rotation_hint["azimuth_locked"]
    ):
        relief = _search_rotation_relief(
            occ,
            solver,
            fixed_shape=fixed_shape,
            moving_shape=moving_shape,
            baseline_transform=matrix,
            baseline_metrics=baseline_metrics,
            baseline_diagnosis=baseline_diagnosis,
            axis_point=rotation_hint["axis_point"],
            axis_direction=rotation_hint["axis_direction"],
            settings=settings,
        )
        analysis["rotation_relief"] = relief

    if relief and _relief_is_acceptable(relief, settings):
        final_transform = _as_matrix4(relief["suggested_transform"])
        final_metrics = relief["best_metrics"]
        final_diagnosis = relief["best_diagnosis"]
        analysis.update(
            {
                "status": final_metrics["status"],
                "classification": "auto_relief_applied",
                "final_metrics": final_metrics,
                "transform_diagnosis": final_diagnosis,
                "applied_angle_deg": relief["best_angle_deg"],
                "applied_transform": _matrix_to_list(final_transform),
            }
        )
        return CollisionResolution(_matrix_to_list(final_transform), True, True, analysis)

    analysis["classification"] = _classify_collision(
        baseline_metrics,
        baseline_diagnosis,
        rotation_hint,
        settings=settings,
        relief=relief,
    )
    if not settings.fail_on_interference:
        return CollisionResolution(_matrix_to_list(matrix), True, False, analysis)

    reason = (
        f"Rejected transform because collision status is '{baseline_metrics['status']}' "
        f"and no non-interfering relief transform was found."
    )
    return CollisionResolution(_matrix_to_list(matrix), False, False, analysis, reason=reason)


def _require_occ() -> dict[str, object]:
    try:
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.gp import gp_Trsf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing pythonocc-core/OCC; run inside the CAD environment.") from exc
    return locals()


def _load_step_shape(occ: dict[str, object], path: Path):
    reader = occ["STEPControl_Reader"]()
    if reader.ReadFile(str(path)) != occ["IFSelect_RetDone"] or reader.TransferRoots() == 0:
        raise RuntimeError(f"Failed to read STEP: {path}")
    return reader.OneShape()


def _transform_shape(occ: dict[str, object], shape, transform: Sequence[Sequence[float]]):
    matrix = _as_matrix4(transform)
    trsf = occ["gp_Trsf"]()
    trsf.SetValues(*(float(matrix[row, col]) for row in range(3) for col in range(4)))
    return occ["BRepBuilderAPI_Transform"](shape, trsf, True).Shape()


def _as_matrix4(transform: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got shape {matrix.shape}.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"Expected homogeneous last row [0, 0, 0, 1], got {matrix[3].tolist()}.")
    return matrix


def _matrix_to_list(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float).tolist()]


def _compose_transforms(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    left_matrix = _matrix_to_list(_as_matrix4(left))
    right_matrix = _matrix_to_list(_as_matrix4(right))
    return [
        [
            float(sum(left_matrix[row][inner] * right_matrix[inner][col] for inner in range(4)))
            for col in range(4)
        ]
        for row in range(4)
    ]


def _axis_rotation_transform(
    axis_point: Sequence[float],
    axis_direction: Sequence[float],
    angle_deg: float,
) -> list[list[float]]:
    axis = np.asarray(axis_direction, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        raise ValueError("Cannot build a rotation around a zero-length axis.")
    axis = axis / axis_norm

    angle_rad = math.radians(float(angle_deg))
    cos_value = math.cos(angle_rad)
    sin_value = math.sin(angle_rad)
    outer = np.outer(axis, axis)
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    rotation = cos_value * np.eye(3) + (1.0 - cos_value) * outer + sin_value * skew

    point = np.asarray(axis_point, dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return _matrix_to_list(transform)


def _shape_min_distance(occ: dict[str, object], shape_a, shape_b) -> Optional[float]:
    try:
        tool = occ["BRepExtrema_DistShapeShape"](shape_a, shape_b)
        tool.Perform()
        if not tool.IsDone():
            return None
        return float(tool.Value())
    except Exception:
        return None


def _shape_common_volume(occ: dict[str, object], shape_a, shape_b) -> Optional[float]:
    try:
        common_builder = occ["BRepAlgoAPI_Common"](shape_a, shape_b)
        common_builder.Build()
        if not common_builder.IsDone():
            return None
        props = occ["GProp_GProps"]()
        occ["brepgprop_VolumeProperties"](common_builder.Shape(), props)
        return float(abs(props.Mass()))
    except Exception:
        return None


def _shape_bbox(occ: dict[str, object], shape) -> Optional[tuple[float, float, float, float, float, float]]:
    try:
        box = occ["Bnd_Box"]()
        box.SetGap(0.0)
        occ["brepbndlib_Add"](shape, box)
        if box.IsVoid():
            return None
        x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
        return (
            float(x_min),
            float(y_min),
            float(z_min),
            float(x_max),
            float(y_max),
            float(z_max),
        )
    except Exception:
        return None


def _bbox_overlap_volume(
    bbox_a: Optional[tuple[float, float, float, float, float, float]],
    bbox_b: Optional[tuple[float, float, float, float, float, float]],
) -> Optional[float]:
    if bbox_a is None or bbox_b is None:
        return None

    dx = max(0.0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[0], bbox_b[0]))
    dy = max(0.0, min(bbox_a[4], bbox_b[4]) - max(bbox_a[1], bbox_b[1]))
    dz = max(0.0, min(bbox_a[5], bbox_b[5]) - max(bbox_a[2], bbox_b[2]))
    return float(dx * dy * dz)


def _measure_shape_interference(
    occ: dict[str, object],
    fixed_shape,
    moving_shape,
    *,
    contact_tolerance: float,
    common_volume_tolerance: float,
    use_common_volume: bool = True,
) -> dict[str, Any]:
    min_distance = _shape_min_distance(occ, fixed_shape, moving_shape)
    common_volume = _shape_common_volume(occ, fixed_shape, moving_shape) if use_common_volume else None
    fixed_bbox = _shape_bbox(occ, fixed_shape)
    moving_bbox = _shape_bbox(occ, moving_shape)
    bbox_overlap_volume = _bbox_overlap_volume(fixed_bbox, moving_bbox)

    touching = min_distance is not None and min_distance <= float(contact_tolerance)
    has_interference = common_volume is not None and common_volume > float(common_volume_tolerance)
    possible_interference = (
        common_volume is None
        and touching
        and bbox_overlap_volume is not None
        and bbox_overlap_volume > 0.0
    )

    if has_interference:
        status = "interference"
    elif possible_interference:
        status = "possible_interference"
    elif touching:
        status = "contact"
    else:
        status = "clearance"

    return {
        "status": status,
        "has_interference": bool(has_interference),
        "possible_interference": bool(possible_interference),
        "touching": bool(touching),
        "min_distance": None if min_distance is None else float(min_distance),
        "common_volume": None if common_volume is None else float(common_volume),
        "bbox_overlap_volume": None if bbox_overlap_volume is None else float(bbox_overlap_volume),
    }


def _pair_constraint_feature_rows(solver: core.PairAssemblySolver) -> list[tuple[object, object, object]]:
    rows: list[tuple[object, object, object]] = []
    for constraint in solver.constraints:
        moving_ref, fixed_ref = solver._split_constraint(constraint)
        moving_feature = solver._feature_for(moving_ref)
        fixed_feature = solver._feature_for(fixed_ref)
        rows.append((constraint, moving_feature, fixed_feature))
    return rows


def _infer_free_rotation_hint(solver: core.PairAssemblySolver) -> Optional[dict[str, Any]]:
    axial_constraints: list[str] = []
    axis_point: Optional[np.ndarray] = None
    axis_direction: Optional[np.ndarray] = None
    azimuth_lock_constraints: list[str] = []

    for constraint, moving_feature, fixed_feature in _pair_constraint_feature_rows(solver):
        if (
            isinstance(moving_feature, core.CylinderFeature)
            and isinstance(fixed_feature, core.CylinderFeature)
            and constraint.kind in {"coincident", "concentric"}
        ):
            axial_constraints.append(constraint.name)
            if axis_point is None:
                axis_point = np.asarray(fixed_feature.axis_point, dtype=float)
                axis_direction = np.asarray(fixed_feature.axis, dtype=float)

    if axis_point is None or axis_direction is None:
        return None

    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    for constraint, moving_feature, fixed_feature in _pair_constraint_feature_rows(solver):
        if not (
            isinstance(moving_feature, core.PlaneFeature)
            and isinstance(fixed_feature, core.PlaneFeature)
        ):
            continue

        moving_alignment = abs(float(np.dot(moving_feature.normal, axis_direction)))
        fixed_alignment = abs(float(np.dot(fixed_feature.normal, axis_direction)))
        if moving_alignment < 0.98 and fixed_alignment < 0.98:
            if constraint.kind in {"coincident", "parallel", "perpendicular", "angle", "distance"}:
                azimuth_lock_constraints.append(constraint.name)

    return {
        "axis_point": axis_point.round(6).tolist(),
        "axis_direction": axis_direction.round(6).tolist(),
        "axial_constraints": axial_constraints,
        "azimuth_locked": bool(azimuth_lock_constraints),
        "azimuth_lock_constraints": azimuth_lock_constraints,
    }


def _interference_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    status = str(metrics.get("status") or "")
    if status == "interference":
        rank = 2.0
    elif status == "possible_interference":
        rank = 1.0
    else:
        rank = 0.0

    common_volume = float(metrics.get("common_volume") or 0.0)
    bbox_overlap = float(metrics.get("bbox_overlap_volume") or 0.0)
    min_distance = float(metrics.get("min_distance") or 0.0)
    return (rank, common_volume, bbox_overlap, -min_distance)


def _search_rotation_relief(
    occ: dict[str, object],
    solver: core.PairAssemblySolver,
    *,
    fixed_shape,
    moving_shape,
    baseline_transform: Sequence[Sequence[float]],
    baseline_metrics: dict[str, Any],
    baseline_diagnosis: dict[str, Any],
    axis_point: Sequence[float],
    axis_direction: Sequence[float],
    settings: CollisionSettings,
) -> dict[str, Any]:
    sample_count = max(int(settings.rotation_sample_count), 2)
    best_angle = 0.0
    best_metrics = baseline_metrics
    best_diagnosis = baseline_diagnosis
    best_transform = _as_matrix4(baseline_transform)
    residual_acceptance_tolerance = max(
        float(settings.residual_good_tolerance) * 10.0,
        float(baseline_diagnosis["max_error"]) + float(settings.residual_good_tolerance),
    )

    for step_index in range(1, sample_count):
        angle_deg = 360.0 * float(step_index) / float(sample_count)
        rotation_transform = _axis_rotation_transform(axis_point, axis_direction, angle_deg)
        candidate_transform = _compose_transforms(rotation_transform, baseline_transform)
        candidate_diagnosis = solver.diagnose_transform(candidate_transform)
        if float(candidate_diagnosis["max_error"]) > residual_acceptance_tolerance:
            continue

        candidate_shape = _transform_shape(occ, moving_shape, rotation_transform)
        candidate_metrics = _measure_shape_interference(
            occ,
            fixed_shape,
            candidate_shape,
            contact_tolerance=settings.contact_tolerance,
            common_volume_tolerance=settings.common_volume_tolerance,
            use_common_volume=False,
        )
        if _interference_score(candidate_metrics) < _interference_score(best_metrics):
            best_angle = float(angle_deg)
            best_metrics = candidate_metrics
            best_diagnosis = candidate_diagnosis
            best_transform = _as_matrix4(candidate_transform)

    improves = _interference_score(best_metrics) < _interference_score(baseline_metrics)
    return {
        "sample_count": int(sample_count),
        "baseline_status": baseline_metrics["status"],
        "best_angle_deg": float(best_angle),
        "improves_interference": bool(improves),
        "best_metrics": best_metrics,
        "best_diagnosis": best_diagnosis,
        "suggested_transform": _matrix_to_list(best_transform),
    }


def _relief_is_acceptable(relief: dict[str, Any], settings: CollisionSettings) -> bool:
    if not relief.get("improves_interference"):
        return False
    if relief["best_metrics"]["status"] not in {"clearance", "contact"}:
        return False
    return float(relief["best_diagnosis"]["max_error"]) <= float(settings.residual_good_tolerance) * 10.0


def _classify_collision(
    metrics: dict[str, Any],
    diagnosis: dict[str, Any],
    rotation_hint: Optional[dict[str, Any]],
    *,
    settings: CollisionSettings,
    relief: Optional[dict[str, Any]],
) -> str:
    if metrics["status"] == "clearance":
        return "clearance"
    if metrics["status"] == "contact":
        return "contact_without_volume_overlap"

    max_error = float(diagnosis["max_error"])
    if max_error >= float(settings.residual_bad_tolerance):
        return "likely_solver_or_constraint_issue"
    if relief and _relief_is_acceptable(relief, settings):
        return "likely_unconstrained_rotation"
    if relief:
        return "unresolved_interference_after_rotation_search"
    if rotation_hint and not rotation_hint["azimuth_locked"]:
        return "possible_unconstrained_rotation"
    if max_error <= float(settings.residual_good_tolerance):
        return "constraints_satisfied_but_physical_interference"
    return "possible_solver_or_constraint_issue"
