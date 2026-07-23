import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


PLANE_VECTOR_SIZE = 10.0
CYLINDER_AXIS_SIZE = 10.0


def _require_solvespace():
    try:
        from python_solvespace import Constraint, Entity, ResultFlag, SolverSystem, make_quaternion
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Missing python-solvespace. Install it in the active environment before running this script."
        ) from exc
    return SolverSystem, ResultFlag, Entity, Constraint, make_quaternion


def _require_occ():
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Missing pythonocc-core/OCC. Run this script inside an environment that contains pythonocc-core."
        ) from exc

    return {
        "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "brepgprop_SurfaceProperties": brepgprop_SurfaceProperties,
        "GeomAbs_Cone": GeomAbs_Cone,
        "GProp_GProps": GProp_GProps,
        "GeomAbs_Cylinder": GeomAbs_Cylinder,
        "GeomAbs_Plane": GeomAbs_Plane,
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPControl_Reader": STEPControl_Reader,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_REVERSED": TopAbs_REVERSED,
        "TopExp_Explorer": TopExp_Explorer,
        "topods": topods,
    }


def normalize(vector: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    length = np.linalg.norm(array)
    if length < eps:
        raise ValueError("Zero-length vector is not allowed.")
    return array / length


def arbitrary_perpendicular(direction: Sequence[float]) -> np.ndarray:
    direction = normalize(direction)
    candidates = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    for candidate in candidates:
        trial = np.cross(direction, candidate)
        if np.linalg.norm(trial) > 1e-8:
            return normalize(trial)
    raise ValueError("Failed to build a perpendicular direction.")


def build_plane_basis(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    normal = normalize(normal)
    u_dir = arbitrary_perpendicular(normal)
    v_dir = normalize(np.cross(normal, u_dir))
    return u_dir, v_dir


def rigid_transform(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray:
    if points_src.shape != points_dst.shape:
        raise ValueError("Source and destination point sets must have the same shape.")
    if len(points_src) < 3:
        raise ValueError("At least three points are required to recover a stable 4x4 rigid transform.")

    center_src = points_src.mean(axis=0)
    center_dst = points_dst.mean(axis=0)
    covariance = (points_src - center_src).T @ (points_dst - center_dst)
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt_mat.T @ u_mat.T))])
    rotation = vt_mat.T @ correction @ u_mat.T
    translation = center_dst - rotation @ center_src

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rigid_transform_from_tripod(points_src: np.ndarray, points_dst: np.ndarray) -> List[List[float]]:
    if points_src.shape != points_dst.shape:
        raise ValueError("Source and destination point sets must have the same shape.")
    if len(points_src) < 3:
        raise ValueError("At least three points are required to recover a stable 4x4 rigid transform.")

    src_points = [list(map(float, point)) for point in points_src.tolist()]
    dst_points = [list(map(float, point)) for point in points_dst.tolist()]
    i0, i1, i2 = PairAssemblySolver._choose_anchor_triangle([np.asarray(point, dtype=float) for point in points_src])

    def vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [float(a[idx] - b[idx]) for idx in range(3)]

    def dot(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(a[idx] * b[idx] for idx in range(3)))

    def cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [
            float(a[1] * b[2] - a[2] * b[1]),
            float(a[2] * b[0] - a[0] * b[2]),
            float(a[0] * b[1] - a[1] * b[0]),
        ]

    def unit(vector: Sequence[float]) -> List[float]:
        length = math.sqrt(dot(vector, vector))
        if length < 1e-12:
            raise ValueError("Cannot normalize a zero-length vector while building rigid transform.")
        return [float(value / length) for value in vector]

    def transpose(mat3: Sequence[Sequence[float]]) -> List[List[float]]:
        return [[float(mat3[col][row]) for col in range(3)] for row in range(3)]

    def mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [float(sum(a[row][k] * b[k][col] for k in range(3))) for col in range(3)]
            for row in range(3)
        ]

    def mat_vec_mul(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
        return [float(sum(a[row][k] * v[k] for k in range(3))) for row in range(3)]

    def basis_from_triangle(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float]) -> List[List[float]]:
        ex = unit(vec_sub(p1, p0))
        temp = vec_sub(p2, p0)
        ez = unit(cross(ex, temp))
        ey = unit(cross(ez, ex))
        return [
            [ex[0], ey[0], ez[0]],
            [ex[1], ey[1], ez[1]],
            [ex[2], ey[2], ez[2]],
        ]

    src_basis = basis_from_triangle(src_points[i0], src_points[i1], src_points[i2])
    dst_basis = basis_from_triangle(dst_points[i0], dst_points[i1], dst_points[i2])
    rotation = mat_mul(dst_basis, transpose(src_basis))
    rotated_origin = mat_vec_mul(rotation, src_points[i0])
    translation = [float(dst_points[i0][idx] - rotated_origin[idx]) for idx in range(3)]

    return [
        rotation[0] + [translation[0]],
        rotation[1] + [translation[1]],
        rotation[2] + [translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_list(matrix) -> List[List[float]]:
    rounded: List[List[float]] = []
    for row in matrix:
        rounded_row: List[float] = []
        for value in row:
            number = round(float(value), 4)
            if abs(number) < 5e-5:
                number = 0.0
            rounded_row.append(number)
        rounded.append(rounded_row)
    return rounded


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(base_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path


def build_step_filename_index(step_dir: Optional[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not step_dir or not os.path.isdir(step_dir):
        return index

    for name in os.listdir(step_dir):
        lower_name = name.lower()
        if not lower_name.endswith((".stp", ".step")):
            continue
        full_path = os.path.join(step_dir, name)
        stem = os.path.splitext(name)[0]
        index[lower_name] = full_path
        index[stem.lower()] = full_path
    return index


def resolve_part_step_path(part: dict, step_dir: Optional[str], base_dir: str, index: Dict[str, str]) -> str:
    candidates: List[str] = []

    for key in ("step_path", "step_file"):
        value = part.get(key)
        if value:
            candidates.append(str(value))

    for key in ("part_name", "part_id"):
        value = part.get(key)
        if value:
            candidates.append(str(value))

    for raw in candidates:
        resolved = resolve_path(raw, base_dir)
        if os.path.isfile(resolved):
            return resolved

        basename = os.path.basename(raw).lower()
        stem = os.path.splitext(os.path.basename(raw))[0].lower()
        if basename in index:
            return index[basename]
        if stem in index:
            return index[stem]

    part_name = part.get("part_name") or part.get("part_id") or "<unknown>"
    if step_dir and os.path.isdir(step_dir):
        raise FileNotFoundError(f"Failed to locate STEP for part '{part_name}' inside '{step_dir}'.")
    raise FileNotFoundError(f"Failed to resolve STEP path for part '{part_name}'.")


def build_parts(payload: dict, step_dir: Optional[str], base_dir: str) -> Dict[str, str]:
    parts = payload.get("parts") or []
    if not parts:
        raise ValueError("The input payload does not contain any parts.")

    step_index = build_step_filename_index(step_dir)
    part_paths: Dict[str, str] = {}
    for part in parts:
        part_name = str(part.get("part_name") or part.get("part_id") or "").strip()
        if not part_name:
            continue
        part_paths[part_name] = resolve_part_step_path(part, step_dir=step_dir, base_dir=base_dir, index=step_index)

    if not part_paths:
        raise ValueError("No usable parts were found in the input payload.")
    return part_paths


def build_parts_from_payload(parts: Sequence[dict], step_dir: Optional[str], base_dir: str) -> Dict[str, str]:
    return build_parts({"parts": list(parts)}, step_dir=step_dir, base_dir=base_dir)


@dataclass(frozen=True)
class PlaneFeature:
    point: np.ndarray
    normal: np.ndarray
    u_dir: np.ndarray
    v_dir: np.ndarray


@dataclass(frozen=True)
class CylinderFeature:
    axis_point: np.ndarray
    axis: np.ndarray
    radius: float
    surface_point: np.ndarray


Feature = Union[PlaneFeature, CylinderFeature]


@dataclass(frozen=True)
class ConstraintRef:
    part_name: str
    face_index: int
    face_type: str


@dataclass(frozen=True)
class PairConstraint:
    name: str
    kind: str
    source_kind: str
    value: float
    orientation: int
    refs: Tuple[ConstraintRef, ConstraintRef]


@dataclass
class PlaneEntities:
    point: object
    u_point: Optional[object]
    v_point: Optional[object]
    normal_tip: object
    normal_line: object
    workplane: Optional[object]
    initial_points: np.ndarray
    support_entities: Tuple[object, ...]

    @property
    def point_entities(self) -> Tuple[object, ...]:
        return self.support_entities


@dataclass
class CylinderEntities:
    axis_point: object
    axis_end: object
    radial_point: object
    axis_line: object
    initial_points: np.ndarray

    @property
    def point_entities(self) -> Tuple[object, object, object]:
        return (self.axis_point, self.axis_end, self.radial_point)


EntityBundle = Union[PlaneEntities, CylinderEntities]


class StepFeatureExtractor:
    def __init__(self) -> None:
        self.occ = _require_occ()
        self._faces_cache: Dict[str, List[object]] = {}

    def _iter_faces(self, shape) -> Iterable[object]:
        explorer = self.occ["TopExp_Explorer"](shape, self.occ["TopAbs_FACE"])
        while explorer.More():
            yield self.occ["topods"].Face(explorer.Current())
            explorer.Next()

    def _load_faces(self, step_path: str) -> List[object]:
        if step_path in self._faces_cache:
            return self._faces_cache[step_path]

        reader = self.occ["STEPControl_Reader"]()
        status = reader.ReadFile(step_path)
        if status != self.occ["IFSelect_RetDone"]:
            raise RuntimeError(f"Failed to read STEP file: {step_path}")
        transferred = reader.TransferRoots()
        if transferred == 0:
            raise RuntimeError(f"STEP TransferRoots failed for: {step_path}")

        shape = reader.OneShape()
        faces = list(self._iter_faces(shape))
        self._faces_cache[step_path] = faces
        return faces

    def extract_feature(self, step_path: str, face_index: int, face_type: str) -> Feature:
        faces = self._load_faces(step_path)
        if face_index < 0 or face_index >= len(faces):
            raise IndexError(f"Face index {face_index} is out of range for STEP '{step_path}'.")

        face = faces[face_index]
        adaptor = self.occ["BRepAdaptor_Surface"](face, True)
        surface_type = adaptor.GetType()
        expected = face_type.strip().lower()
        axial_surface_types = (self.occ["GeomAbs_Cylinder"], self.occ["GeomAbs_Cone"])

        if expected == "plane" and surface_type != self.occ["GeomAbs_Plane"]:
            raise TypeError(f"Face {face_index} in '{step_path}' is not a plane.")
        if expected == "cylinder" and surface_type not in axial_surface_types:
            raise TypeError(
                f"Face {face_index} in '{step_path}' is not a cylinder-like axial surface."
            )

        props = self.occ["GProp_GProps"]()
        self.occ["brepgprop_SurfaceProperties"](face, props)
        center = props.CentreOfMass()
        face_center = np.array([float(center.X()), float(center.Y()), float(center.Z())], dtype=float)

        if surface_type == self.occ["GeomAbs_Plane"]:
            plane = adaptor.Plane()
            axis = plane.Axis()
            direction = axis.Direction()
            normal = normalize([direction.X(), direction.Y(), direction.Z()])
            if face.Orientation() == self.occ["TopAbs_REVERSED"]:
                normal = -normal
            u_dir, v_dir = build_plane_basis(normal)
            return PlaneFeature(point=face_center, normal=normal, u_dir=u_dir, v_dir=v_dir)

        if surface_type in axial_surface_types:
            if surface_type == self.occ["GeomAbs_Cylinder"]:
                axial_surface = adaptor.Cylinder()
            else:
                axial_surface = adaptor.Cone()
            axis = axial_surface.Axis()
            direction = axis.Direction()
            location = axis.Location()
            axis_location = np.array([float(location.X()), float(location.Y()), float(location.Z())], dtype=float)
            axis_dir = normalize([direction.X(), direction.Y(), direction.Z()])
            if face.Orientation() == self.occ["TopAbs_REVERSED"]:
                axis_dir = -axis_dir
            projection_length = float(np.dot(face_center - axis_location, axis_dir))
            axis_point = axis_location + projection_length * axis_dir
            return CylinderFeature(
                axis_point=axis_point,
                axis=axis_dir,
                radius=float(np.linalg.norm(face_center - axis_point)),
                surface_point=face_center,
            )

        raise TypeError(
            f"Face {face_index} in '{step_path}' has unsupported type '{surface_type}'. "
            "Only Plane and cylinder-like axial surfaces are supported."
        )


def first_face_index(element: dict, face_index_base: int = 0) -> int:
    for key in ("matched_face_idx", "face_idx"):
        value = element.get(key)
        if isinstance(value, list) and value:
            return int(value[0]) - face_index_base
        if isinstance(value, int):
            return int(value) - face_index_base
    raise ValueError(f"Constraint element does not contain a face index: {element}")


def normalize_constraint_kind(constraint: dict) -> str:
    raw_kind = str(constraint.get("constraint_type") or constraint.get("source_constraint_type") or "").strip().lower()
    source_kind = str(constraint.get("source_constraint_type") or "").strip().lower()
    params = constraint.get("params") or constraint.get("parameters") or {}
    raw_value = float(params.get("raw_value", 0.0) or 0.0)
    elements = constraint.get("elements") or []
    face_types = [str(element.get("face_type") or "").strip().lower() for element in elements]
    all_cylinders = len(face_types) == 2 and all(face_type == "cylinder" for face_type in face_types)

    if raw_kind == "concentric":
        return "concentric"
    if raw_kind in {"coincident", "coincidence"} and all_cylinders:
        return "concentric"
    if source_kind == "angle":
        if raw_kind == "perpendicular" and abs(raw_value - 90.0) < 1e-6:
            return "perpendicular"
        if raw_kind in {"parallel", "parallelism"} and abs(raw_value) < 1e-6:
            return "parallel"
        return "angle"
    if raw_kind in {"coincident", "coincidence"} and source_kind == "distance":
        return "distance"
    if raw_kind == "distance":
        return "distance"
    if raw_kind in {"coincident", "coincidence"}:
        return "coincident"
    if raw_kind in {"parallel", "parallelism"}:
        return "parallel"
    if raw_kind == "perpendicular":
        return "perpendicular"
    if raw_kind == "angle":
        return "angle"
    return raw_kind


def constraint_orientation(params: dict) -> int:
    raw_orientation = params.get("orientation")
    if raw_orientation not in {None, ""}:
        return int(raw_orientation or 0)

    raw_alignment = params.get("alignment")
    if raw_alignment in {None, ""}:
        return 0

    alignment = int(raw_alignment)
    if alignment == 0:
        return 1
    if alignment == 1:
        return 2
    return alignment


def extract_constraints(data: dict, face_index_base: int = 0) -> List[PairConstraint]:
    raw_constraints = data.get("internal_constraints") or data.get("constraints") or []
    constraints: List[PairConstraint] = []

    for index, item in enumerate(raw_constraints):
        elements = item.get("elements") or []
        if len(elements) != 2:
            continue

        refs = []
        for element in elements:
            refs.append(
                ConstraintRef(
                    part_name=str(element.get("part_name") or element.get("part_id") or "").strip(),
                    face_index=first_face_index(element, face_index_base=face_index_base),
                    face_type=str(element.get("face_type") or "").strip(),
                )
            )

        params = item.get("params") or item.get("parameters") or {}
        constraints.append(
            PairConstraint(
                name=str(item.get("constraint_name") or f"constraint_{index + 1}"),
                kind=normalize_constraint_kind(item),
                source_kind=str(item.get("source_constraint_type") or item.get("constraint_type") or ""),
                value=float(params.get("raw_value", 0.0) or 0.0),
                orientation=constraint_orientation(params),
                refs=(refs[0], refs[1]),
            )
        )

    return constraints


def select_pair_constraints(
    constraints: Sequence[PairConstraint],
    part_order: Sequence[str],
    fixed_part: Optional[str] = None,
    moving_part: Optional[str] = None,
) -> Tuple[str, str, List[PairConstraint]]:
    by_pair: Dict[frozenset, List[PairConstraint]] = defaultdict(list)
    part_rank = {name: index for index, name in enumerate(part_order)}

    for constraint in constraints:
        part_a = constraint.refs[0].part_name
        part_b = constraint.refs[1].part_name
        if not part_a or not part_b or part_a == part_b:
            continue
        by_pair[frozenset((part_a, part_b))].append(constraint)

    if not by_pair:
        raise ValueError("No usable two-part constraints were found in the input data.")

    def ordered_pair(pair_key: frozenset) -> Tuple[str, str]:
        parts = list(pair_key)
        parts.sort(key=lambda name: (part_rank.get(name, math.inf), name))
        return parts[0], parts[1]

    if fixed_part and moving_part:
        pair_key = frozenset((fixed_part, moving_part))
        selected = by_pair.get(pair_key, [])
        if not selected:
            raise ValueError(f"No constraints were found between '{fixed_part}' and '{moving_part}'.")
        return fixed_part, moving_part, list(selected)

    candidate_keys = list(by_pair)
    if fixed_part:
        candidate_keys = [key for key in candidate_keys if fixed_part in key]
    if moving_part:
        candidate_keys = [key for key in candidate_keys if moving_part in key]
    if not candidate_keys:
        raise ValueError("The requested fixed/moving part selection does not match any available part pair.")

    best_key = max(candidate_keys, key=lambda key: (len(by_pair[key]), tuple(ordered_pair(key))))
    first_part, second_part = ordered_pair(best_key)

    if fixed_part and not moving_part:
        moving_candidate = second_part if first_part == fixed_part else first_part
        return fixed_part, moving_candidate, list(by_pair[best_key])
    if moving_part and not fixed_part:
        fixed_candidate = second_part if first_part == moving_part else first_part
        return fixed_candidate, moving_part, list(by_pair[best_key])

    return first_part, second_part, list(by_pair[best_key])


class PairAssemblySolver:
    def __init__(
        self,
        fixed_part: str,
        moving_part: str,
        part_paths: Dict[str, str],
        constraints: Sequence[PairConstraint],
        extractor: Optional[StepFeatureExtractor] = None,
    ) -> None:
        self.fixed_part = fixed_part
        self.moving_part = moving_part
        self.part_paths = part_paths
        self.constraints = list(constraints)
        self.extractor = extractor or StepFeatureExtractor()

        if self.fixed_part not in self.part_paths:
            raise KeyError(f"Unknown fixed part '{self.fixed_part}'.")
        if self.moving_part not in self.part_paths:
            raise KeyError(f"Unknown moving part '{self.moving_part}'.")
        if not self.constraints:
            raise ValueError("At least one constraint is required to solve a part pair.")

        SolverSystem, ResultFlag, Entity, Constraint, make_quaternion = _require_solvespace()
        self.SolverSystem = SolverSystem
        self.ResultFlag = ResultFlag
        self.Entity = Entity
        self.Constraint = Constraint
        self.make_quaternion = make_quaternion

        self.features: Dict[Tuple[str, int], Feature] = {}

    def _feature_for(self, ref: ConstraintRef) -> Feature:
        key = (ref.part_name, ref.face_index)
        if key not in self.features:
            step_path = self.part_paths[ref.part_name]
            self.features[key] = self.extractor.extract_feature(step_path, ref.face_index, ref.face_type)
        return self.features[key]

    def _create_fixed_plane(self, system, feature: PlaneFeature) -> PlaneEntities:
        point = feature.point
        u_point = point + feature.u_dir * PLANE_VECTOR_SIZE
        v_point = point + feature.v_dir * PLANE_VECTOR_SIZE
        normal_tip = point + feature.normal * PLANE_VECTOR_SIZE

        point_entity = system.add_point_3d(*point.tolist())
        u_entity = system.add_point_3d(*u_point.tolist())
        v_entity = system.add_point_3d(*v_point.tolist())
        normal_tip_entity = system.add_point_3d(*normal_tip.tolist())
        normal_line = system.add_line_3d(point_entity, normal_tip_entity)

        quaternion = self.make_quaternion(
            float(feature.u_dir[0]),
            float(feature.u_dir[1]),
            float(feature.u_dir[2]),
            float(feature.v_dir[0]),
            float(feature.v_dir[1]),
            float(feature.v_dir[2]),
        )
        normal_entity = system.add_normal_3d(*quaternion)
        workplane = system.add_work_plane(point_entity, normal_entity)

        system.distance(u_entity, workplane, 0.0, self.Entity.FREE_IN_3D)
        system.distance(v_entity, workplane, 0.0, self.Entity.FREE_IN_3D)
        system.distance(normal_tip_entity, workplane, PLANE_VECTOR_SIZE, self.Entity.FREE_IN_3D)

        return PlaneEntities(
            point=point_entity,
            u_point=u_entity,
            v_point=v_entity,
            normal_tip=normal_tip_entity,
            normal_line=normal_line,
            workplane=workplane,
            initial_points=np.asarray([point, u_point, v_point, normal_tip], dtype=float),
            support_entities=(point_entity, u_entity, v_entity, normal_tip_entity),
        )

    def _create_moving_plane(self, system, feature: PlaneFeature) -> PlaneEntities:
        point = feature.point
        u_point = point + feature.u_dir * PLANE_VECTOR_SIZE
        v_point = point + feature.v_dir * PLANE_VECTOR_SIZE
        normal_tip = point + feature.normal * PLANE_VECTOR_SIZE

        point_entity = system.add_point_3d(*point.tolist())
        u_entity = system.add_point_3d(*u_point.tolist())
        v_entity = system.add_point_3d(*v_point.tolist())
        normal_tip_entity = system.add_point_3d(*normal_tip.tolist())
        normal_line = system.add_line_3d(point_entity, normal_tip_entity)

        quaternion = self.make_quaternion(
            float(feature.u_dir[0]),
            float(feature.u_dir[1]),
            float(feature.u_dir[2]),
            float(feature.v_dir[0]),
            float(feature.v_dir[1]),
            float(feature.v_dir[2]),
        )
        normal_entity = system.add_normal_3d(*quaternion)
        workplane = system.add_work_plane(point_entity, normal_entity)

        system.distance(u_entity, workplane, 0.0, self.Entity.FREE_IN_3D)
        system.distance(v_entity, workplane, 0.0, self.Entity.FREE_IN_3D)
        system.distance(normal_tip_entity, workplane, PLANE_VECTOR_SIZE, self.Entity.FREE_IN_3D)

        return PlaneEntities(
            point=point_entity,
            u_point=u_entity,
            v_point=v_entity,
            normal_tip=normal_tip_entity,
            normal_line=normal_line,
            workplane=workplane,
            initial_points=np.asarray([point, u_point, v_point, normal_tip], dtype=float),
            support_entities=(point_entity, u_entity, v_entity, normal_tip_entity),
        )

    def _create_fixed_cylinder(self, system, feature: CylinderFeature) -> CylinderEntities:
        axis_point = feature.axis_point
        axis_end = axis_point + feature.axis * CYLINDER_AXIS_SIZE
        radial_point = feature.surface_point

        axis_point_entity = system.add_point_3d(*axis_point.tolist())
        axis_end_entity = system.add_point_3d(*axis_end.tolist())
        radial_point_entity = system.add_point_3d(*radial_point.tolist())
        axis_line = system.add_line_3d(axis_point_entity, axis_end_entity)

        return CylinderEntities(
            axis_point=axis_point_entity,
            axis_end=axis_end_entity,
            radial_point=radial_point_entity,
            axis_line=axis_line,
            initial_points=np.asarray([axis_point, axis_end, radial_point], dtype=float),
        )

    def _create_moving_cylinder(self, system, feature: CylinderFeature) -> CylinderEntities:
        axis_point = feature.axis_point
        axis_end = axis_point + feature.axis * CYLINDER_AXIS_SIZE
        radial_point = feature.surface_point

        axis_point_entity = system.add_point_3d(*axis_point.tolist())
        axis_end_entity = system.add_point_3d(*axis_end.tolist())
        radial_point_entity = system.add_point_3d(*radial_point.tolist())
        axis_line = system.add_line_3d(axis_point_entity, axis_end_entity)

        return CylinderEntities(
            axis_point=axis_point_entity,
            axis_end=axis_end_entity,
            radial_point=radial_point_entity,
            axis_line=axis_line,
            initial_points=np.asarray([axis_point, axis_end, radial_point], dtype=float),
        )

    def _create_bundle(self, system, feature: Feature, fixed: bool) -> EntityBundle:
        if isinstance(feature, PlaneFeature):
            return self._create_fixed_plane(system, feature) if fixed else self._create_moving_plane(system, feature)
        if isinstance(feature, CylinderFeature):
            return self._create_fixed_cylinder(system, feature) if fixed else self._create_moving_cylinder(system, feature)
        raise TypeError(f"Unsupported feature type: {type(feature)}")

    def _rigidify_moving_part(self, system, moving_bundles: Dict[int, EntityBundle]) -> Tuple[np.ndarray, List[object]]:
        unique_entities: List[object] = []
        unique_points: List[np.ndarray] = []
        seen_params = set()

        for bundle in moving_bundles.values():
            for entity, point in zip(bundle.point_entities, bundle.initial_points):
                key = tuple(system.params(entity.params))
                if key in seen_params:
                    continue
                seen_params.add(key)
                unique_entities.append(entity)
                unique_points.append(np.asarray(point, dtype=float))

        if len(unique_entities) < 3:
            raise ValueError("At least three support points are required to rigidify the moving part.")

        anchor_indices = self._choose_anchor_triangle(unique_points)
        i0, i1, i2 = anchor_indices
        anchor_pairs = ((i0, i1), (i0, i2), (i1, i2))
        for start, end in anchor_pairs:
            distance = float(np.linalg.norm(unique_points[start] - unique_points[end]))
            system.distance(unique_entities[start], unique_entities[end], distance, self.Entity.FREE_IN_3D)

        for index in range(len(unique_entities)):
            if index in anchor_indices:
                continue
            for anchor_index in anchor_indices:
                distance = float(np.linalg.norm(unique_points[index] - unique_points[anchor_index]))
                if distance > 1e-9:
                    system.distance(unique_entities[index], unique_entities[anchor_index], distance, self.Entity.FREE_IN_3D)

        return np.asarray(unique_points, dtype=float), unique_entities

    @staticmethod
    def _choose_anchor_triangle(points: Sequence[np.ndarray]) -> Tuple[int, int, int]:
        if len(points) < 3:
            raise ValueError("Need at least three points to define a rigid anchor triangle.")

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                baseline = points[j] - points[i]
                if np.linalg.norm(baseline) < 1e-9:
                    continue
                for k in range(j + 1, len(points)):
                    area = np.linalg.norm(np.cross(baseline, points[k] - points[i]))
                    if area > 1e-8:
                        return i, j, k

        raise ValueError("The moving support points are degenerate; a non-collinear anchor triangle is required.")

    def _split_constraint(self, constraint: PairConstraint) -> Tuple[ConstraintRef, ConstraintRef]:
        ref_a, ref_b = constraint.refs
        if ref_a.part_name == self.moving_part and ref_b.part_name == self.fixed_part:
            return ref_a, ref_b
        if ref_b.part_name == self.moving_part and ref_a.part_name == self.fixed_part:
            return ref_b, ref_a
        raise ValueError(
            f"Constraint '{constraint.name}' does not belong to the selected pair '{self.fixed_part}'/'{self.moving_part}'."
        )

    @staticmethod
    def _rotation_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
        axis = normalize(axis)
        x_value, y_value, z_value = axis
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        one_minus_cos = 1.0 - cos_value
        return np.asarray(
            [
                [
                    cos_value + x_value * x_value * one_minus_cos,
                    x_value * y_value * one_minus_cos - z_value * sin_value,
                    x_value * z_value * one_minus_cos + y_value * sin_value,
                ],
                [
                    y_value * x_value * one_minus_cos + z_value * sin_value,
                    cos_value + y_value * y_value * one_minus_cos,
                    y_value * z_value * one_minus_cos - x_value * sin_value,
                ],
                [
                    z_value * x_value * one_minus_cos - y_value * sin_value,
                    z_value * y_value * one_minus_cos + x_value * sin_value,
                    cos_value + z_value * z_value * one_minus_cos,
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation_between_vectors(source: Sequence[float], target: Sequence[float]) -> np.ndarray:
        source_vec = normalize(source)
        target_vec = normalize(target)
        dot_value = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))

        if dot_value > 1.0 - 1e-10:
            return np.eye(3)
        if dot_value < -1.0 + 1e-10:
            return PairAssemblySolver._rotation_from_axis_angle(arbitrary_perpendicular(source_vec), math.pi)

        axis = np.cross(source_vec, target_vec)
        return PairAssemblySolver._rotation_from_axis_angle(axis, math.acos(dot_value))

    @staticmethod
    def _basis_from_direction_pair(primary: Sequence[float], secondary: Sequence[float]) -> np.ndarray:
        first = normalize(primary)
        second_raw = np.asarray(secondary, dtype=float)
        second_projected = second_raw - float(np.dot(second_raw, first)) * first
        if np.linalg.norm(second_projected) < 1e-8:
            second_projected = arbitrary_perpendicular(first)
        second = normalize(second_projected)
        third = normalize(np.cross(first, second))
        return np.asarray(
            [
                [float(first[0]), float(second[0]), float(third[0])],
                [float(first[1]), float(second[1]), float(third[1])],
                [float(first[2]), float(second[2]), float(third[2])],
            ],
            dtype=float,
        )

    @staticmethod
    def _mat3_mul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(sum(float(left[row][idx]) * float(right[idx][col]) for idx in range(3)))
                    for col in range(3)
                ]
                for row in range(3)
            ],
            dtype=float,
        )

    @staticmethod
    def _mat3_transpose(matrix: Sequence[Sequence[float]]) -> np.ndarray:
        return np.asarray(
            [[float(matrix[col][row]) for col in range(3)] for row in range(3)],
            dtype=float,
        )

    @staticmethod
    def _mat3_vec_mul(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> np.ndarray:
        return np.asarray(
            [
                float(sum(float(matrix[row][idx]) * float(vector[idx]) for idx in range(3)))
                for row in range(3)
            ],
            dtype=float,
        )

    @staticmethod
    def _is_independent_direction(source: Sequence[float], target: Sequence[float], tol: float = 1e-8) -> bool:
        return np.linalg.norm(np.cross(normalize(source), normalize(target))) > tol

    @staticmethod
    def _solve_small_linear_system(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> np.ndarray:
        size = len(values)
        augmented = [
            [float(matrix[row][col]) for col in range(size)] + [float(values[row])]
            for row in range(size)
        ]

        for col in range(size):
            pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
            if abs(augmented[pivot][col]) < 1e-10:
                raise ValueError("Singular small linear system.")
            if pivot != col:
                augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

            pivot_value = augmented[col][col]
            for idx in range(col, size + 1):
                augmented[col][idx] /= pivot_value

            for row in range(size):
                if row == col:
                    continue
                factor = augmented[row][col]
                if abs(factor) < 1e-14:
                    continue
                for idx in range(col, size + 1):
                    augmented[row][idx] -= factor * augmented[col][idx]

        return np.asarray([augmented[row][size] for row in range(size)], dtype=float)

    @staticmethod
    def _independent_rows(rows: Sequence[np.ndarray], values: Sequence[float]) -> Tuple[List[np.ndarray], List[float]]:
        independent: List[np.ndarray] = []
        independent_values: List[float] = []
        orthonormal_basis: List[np.ndarray] = []

        for row, value in zip(rows, values):
            residual = np.asarray(row, dtype=float)
            for basis_vector in orthonormal_basis:
                residual = residual - float(np.dot(residual, basis_vector)) * basis_vector

            length = float(np.linalg.norm(residual))
            if length <= 1e-9:
                continue

            orthonormal_basis.append(residual / length)
            independent.append(np.asarray(row, dtype=float))
            independent_values.append(float(value))

        return independent, independent_values

    @staticmethod
    def _least_norm_translation(rows: Sequence[np.ndarray], values: Sequence[float]) -> np.ndarray:
        independent, independent_values = PairAssemblySolver._independent_rows(rows, values)
        if not independent:
            return np.zeros(3, dtype=float)

        gram = [
            [float(np.dot(row_a, row_b)) for row_b in independent]
            for row_a in independent
        ]
        multipliers = PairAssemblySolver._solve_small_linear_system(gram, independent_values)
        translation = np.zeros(3, dtype=float)
        for row, multiplier in zip(independent, multipliers):
            translation = translation + float(multiplier) * np.asarray(row, dtype=float)
        return translation

    @staticmethod
    def _rotation_from_direction_pairs(direction_pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        pairs = [(normalize(source), normalize(target)) for source, target in direction_pairs]
        if not pairs:
            return np.eye(3)

        primary_source, primary_target = pairs[0]
        for secondary_source, secondary_target in pairs[1:]:
            source_independent = PairAssemblySolver._is_independent_direction(primary_source, secondary_source)
            target_independent = PairAssemblySolver._is_independent_direction(primary_target, secondary_target)
            if not source_independent or not target_independent:
                continue

            source_basis = PairAssemblySolver._basis_from_direction_pair(primary_source, secondary_source)
            target_basis = PairAssemblySolver._basis_from_direction_pair(primary_target, secondary_target)
            return PairAssemblySolver._mat3_mul(target_basis, PairAssemblySolver._mat3_transpose(source_basis))

        return PairAssemblySolver._rotation_between_vectors(primary_source, primary_target)

    def _analytic_direction_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        direction_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        for constraint in self.constraints:
            moving_ref, fixed_ref = self._split_constraint(constraint)
            moving_feature = self._feature_for(moving_ref)
            fixed_feature = self._feature_for(fixed_ref)

            if isinstance(moving_feature, PlaneFeature) and isinstance(fixed_feature, PlaneFeature):
                if constraint.kind == "coincident":
                    if int(constraint.orientation) == 0:
                        continue
                    normal_sign = -1.0 if int(constraint.orientation) == 2 else 1.0
                    direction_pairs.append((moving_feature.normal, normal_sign * fixed_feature.normal))
                    continue

                if constraint.kind == "distance":
                    dot_value = float(np.dot(moving_feature.normal, fixed_feature.normal))
                    if abs(dot_value) > 1e-6:
                        normal_sign = 1.0 if dot_value >= 0.0 else -1.0
                        direction_pairs.append((moving_feature.normal, normal_sign * fixed_feature.normal))
                    continue

                if constraint.kind == "parallel":
                    dot_value = float(np.dot(moving_feature.normal, fixed_feature.normal))
                    normal_sign = 1.0 if dot_value >= 0.0 else -1.0
                    direction_pairs.append((moving_feature.normal, normal_sign * fixed_feature.normal))
                    continue

                if constraint.kind == "angle":
                    if abs(constraint.value) < 1e-8 or abs(abs(constraint.value) - 180.0) < 1e-8:
                        normal_sign = -1.0 if abs(abs(constraint.value) - 180.0) < 1e-8 else 1.0
                        direction_pairs.append((moving_feature.normal, normal_sign * fixed_feature.normal))
                    # Non-zero/non-180 angles constrain the remaining twist. They
                    # are checked after candidate transforms are built.
                    continue

                if constraint.kind == "perpendicular":
                    continue

                raise NotImplementedError(
                    f"{constraint.name}: analytic fallback does not support plane-plane '{constraint.kind}'."
                )

            if isinstance(moving_feature, CylinderFeature) and isinstance(fixed_feature, CylinderFeature):
                if constraint.kind in {"coincident", "concentric"}:
                    axis_sign = 1.0 if float(np.dot(moving_feature.axis, fixed_feature.axis)) >= 0.0 else -1.0
                    direction_pairs.append((moving_feature.axis, axis_sign * fixed_feature.axis))
                    continue

                raise NotImplementedError(
                    f"{constraint.name}: analytic fallback does not support cylinder-cylinder '{constraint.kind}'."
                )

            raise NotImplementedError(
                f"{constraint.name}: analytic fallback does not support "
                f"{type(moving_feature).__name__} <-> {type(fixed_feature).__name__}."
            )

        return direction_pairs

    def _analytic_translation_candidates(
        self, rotation: np.ndarray
    ) -> List[Tuple[np.ndarray, List[Tuple[str, float]]]]:
        import itertools

        base_rows: List[np.ndarray] = []
        base_values: List[float] = []
        distance_options: List[List[Tuple[str, np.ndarray, float]]] = []

        for constraint in self.constraints:
            moving_ref, fixed_ref = self._split_constraint(constraint)
            moving_feature = self._feature_for(moving_ref)
            fixed_feature = self._feature_for(fixed_ref)

            if isinstance(moving_feature, PlaneFeature) and isinstance(fixed_feature, PlaneFeature):
                normal = normalize(fixed_feature.normal)
                moved_point_without_translation = self._mat3_vec_mul(rotation, moving_feature.point)

                if constraint.kind == "coincident":
                    base_rows.append(normal)
                    base_values.append(float(np.dot(normal, fixed_feature.point - moved_point_without_translation)))
                    continue

                if constraint.kind == "distance":
                    raw_distance = abs(float(constraint.value))
                    signed_before_translation = float(
                        np.dot(normal, moved_point_without_translation - fixed_feature.point)
                    )
                    if raw_distance < 1e-9:
                        candidate_distances = [0.0]
                    else:
                        preferred_sign = 1.0 if signed_before_translation >= 0.0 else -1.0
                        preferred_distance = preferred_sign * raw_distance
                        candidate_distances = [preferred_distance, -preferred_distance]

                    options: List[Tuple[str, np.ndarray, float]] = []
                    for signed_distance in candidate_distances:
                        rhs = float(
                            signed_distance
                            + np.dot(normal, fixed_feature.point)
                            - np.dot(normal, moved_point_without_translation)
                        )
                        options.append((constraint.name, normal, rhs))
                    distance_options.append(options)
                    continue

                continue

            if isinstance(moving_feature, CylinderFeature) and isinstance(fixed_feature, CylinderFeature):
                if constraint.kind in {"coincident", "concentric"}:
                    axis = normalize(fixed_feature.axis)
                    moved_axis_point = self._mat3_vec_mul(rotation, moving_feature.axis_point)
                    delta = fixed_feature.axis_point - moved_axis_point
                    for basis_vector in (arbitrary_perpendicular(axis), normalize(np.cross(axis, arbitrary_perpendicular(axis)))):
                        base_rows.append(np.asarray(basis_vector, dtype=float))
                        base_values.append(float(np.dot(basis_vector, delta)))

        option_products = itertools.product(*distance_options) if distance_options else [()]
        candidates: List[Tuple[np.ndarray, List[Tuple[str, float]]]] = []

        for option_product in option_products:
            rows = list(base_rows)
            values = list(base_values)
            signs: List[Tuple[str, float]] = []
            for name, row, rhs in option_product:
                rows.append(np.asarray(row, dtype=float))
                values.append(float(rhs))
                signs.append((name, float(rhs)))

            if not rows:
                candidates.append((np.zeros(3, dtype=float), signs))
                continue

            translation = self._least_norm_translation(rows, values)
            candidates.append((translation, signs))

        return candidates

    def _analytic_constraint_errors(
        self, rotation: np.ndarray, translation: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        errors: List[Tuple[str, str, float]] = []

        for constraint in self.constraints:
            moving_ref, fixed_ref = self._split_constraint(constraint)
            moving_feature = self._feature_for(moving_ref)
            fixed_feature = self._feature_for(fixed_ref)

            if isinstance(moving_feature, PlaneFeature) and isinstance(fixed_feature, PlaneFeature):
                moved_point = self._mat3_vec_mul(rotation, moving_feature.point) + translation
                moved_normal = normalize(self._mat3_vec_mul(rotation, moving_feature.normal))
                fixed_normal = normalize(fixed_feature.normal)
                signed_offset = float(np.dot(fixed_normal, moved_point - fixed_feature.point))

                if constraint.kind == "coincident":
                    errors.append((constraint.name, "plane_offset", abs(signed_offset)))
                    if int(constraint.orientation) != 0:
                        target_normal = -fixed_normal if int(constraint.orientation) == 2 else fixed_normal
                        errors.append(
                            (
                                constraint.name,
                                "plane_normal",
                                float(np.linalg.norm(moved_normal - target_normal)),
                            )
                        )
                    continue

                if constraint.kind == "distance":
                    errors.append((constraint.name, "plane_distance", abs(abs(signed_offset) - abs(float(constraint.value)))))
                    if abs(float(np.dot(moving_feature.normal, fixed_feature.normal))) > 1e-6:
                        errors.append(
                            (
                                constraint.name,
                                "plane_parallel",
                                abs(1.0 - abs(float(np.dot(moved_normal, fixed_normal)))),
                            )
                        )
                    continue

                if constraint.kind == "parallel":
                    errors.append((constraint.name, "plane_parallel", abs(1.0 - abs(float(np.dot(moved_normal, fixed_normal))))))
                    continue

                if constraint.kind == "angle":
                    dot_value = float(np.clip(np.dot(moved_normal, fixed_normal), -1.0, 1.0))
                    angle = math.degrees(math.acos(dot_value))
                    errors.append((constraint.name, "plane_angle", abs(angle - abs(float(constraint.value)))))
                    continue

            if isinstance(moving_feature, CylinderFeature) and isinstance(fixed_feature, CylinderFeature):
                moved_axis_point = self._mat3_vec_mul(rotation, moving_feature.axis_point) + translation
                moved_axis = normalize(self._mat3_vec_mul(rotation, moving_feature.axis))
                fixed_axis = normalize(fixed_feature.axis)
                axis_parallel_error = abs(1.0 - abs(float(np.dot(moved_axis, fixed_axis))))
                axis_separation = float(np.linalg.norm(np.cross(moved_axis_point - fixed_feature.axis_point, fixed_axis)))
                errors.append((constraint.name, "axis_parallel", axis_parallel_error))
                errors.append((constraint.name, "axis_separation", axis_separation))

        return errors

    def diagnose_transform(
        self,
        transform: Sequence[Sequence[float]],
    ) -> Dict[str, object]:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform matrix, got shape {matrix.shape}.")
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
            raise ValueError("Transform must be a rigid homogeneous 4x4 matrix.")

        rotation = np.asarray(matrix[:3, :3], dtype=float)
        translation = np.asarray(matrix[:3, 3], dtype=float)
        errors = self._analytic_constraint_errors(rotation, translation)
        constraint_errors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for constraint_name, metric_name, error_value in errors:
            constraint_errors[constraint_name].append((metric_name, float(error_value)))

        by_constraint = []
        max_error = 0.0
        for constraint in self.constraints:
            entries = constraint_errors.get(constraint.name, [])
            constraint_max = max((value for _, value in entries), default=0.0)
            max_error = max(max_error, constraint_max)
            by_constraint.append(
                {
                    "name": constraint.name,
                    "kind": constraint.kind,
                    "value": float(constraint.value),
                    "orientation": int(constraint.orientation),
                    "max_error": float(constraint_max),
                    "metrics": [
                        {"name": metric_name, "value": float(metric_value)}
                        for metric_name, metric_value in entries
                    ],
                }
            )

        return {
            "transform": matrix_to_list(matrix),
            "max_error": float(max_error),
            "errors": [
                {
                    "constraint": constraint_name,
                    "metric": metric_name,
                    "value": float(error_value),
                }
                for constraint_name, metric_name, error_value in errors
            ],
            "constraints": by_constraint,
        }

    def _solve_analytically(self) -> List[List[float]]:
        rotation = self._rotation_from_direction_pairs(self._analytic_direction_pairs())
        best_transform: Optional[List[List[float]]] = None
        best_error = math.inf
        best_errors: List[Tuple[str, str, float]] = []

        for translation, _ in self._analytic_translation_candidates(rotation):
            errors = self._analytic_constraint_errors(rotation, translation)
            max_error = max((error for _, _, error in errors), default=0.0)
            if max_error < best_error:
                best_error = max_error
                best_errors = errors
                best_transform = [
                    [float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]), float(translation[0])],
                    [float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]), float(translation[1])],
                    [float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]), float(translation[2])],
                    [0.0, 0.0, 0.0, 1.0],
                ]

        if best_transform is not None and best_error <= 1e-4:
            return best_transform

        detail = "; ".join(f"{name}:{kind}={value:.6g}" for name, kind, value in best_errors)
        raise RuntimeError(f"Analytic fallback could not satisfy the selected constraints. {detail}")

    def _apply_plane_plane_distance(self, system, moving: PlaneEntities, fixed: PlaneEntities, value: float) -> None:
        if fixed.workplane is None:
            raise ValueError("Fixed plane bundle does not have a workplane.")
        system.distance(moving.point, fixed.workplane, value, self.Entity.FREE_IN_3D)

    def _apply_plane_plane_angle(self, system, moving: PlaneEntities, fixed: PlaneEntities, angle_value: float) -> None:
        system.add_constraint(
            self.Constraint.ANGLE,
            self.Entity.FREE_IN_3D,
            float(angle_value),
            self.Entity.NONE,
            self.Entity.NONE,
            moving.normal_line,
            fixed.normal_line,
        )

    def _apply_plane_plane_parallel(self, system, moving: PlaneEntities, fixed: PlaneEntities) -> None:
        # `Constraint.PARALLEL` on these 3D helper lines can trigger a native
        # SolveSpace assertion and kill the Python process. Express the same
        # intent through a 0-degree angle instead so failures stay as Python
        # exceptions rather than crashing the notebook kernel.
        self._apply_plane_plane_angle(system, moving, fixed, 0.0)

    def _apply_plane_plane_coincident(
        self,
        system,
        moving: PlaneEntities,
        fixed: PlaneEntities,
        orientation: int,
    ) -> None:
        self._apply_plane_plane_distance(system, moving, fixed, 0.0)
        # A true plane-plane coincidence needs both coplanarity and aligned
        # plane normals. Without the angle constraint, SolveSpace can settle on
        # a coplanar but flipped orientation that still passes the distance test.
        if int(orientation) == 0:
            return
        angle_value = 180.0 if int(orientation) == 2 else 0.0
        self._apply_plane_plane_angle(system, moving, fixed, angle_value)

    def _apply_plane_plane_perpendicular(self, system, moving: PlaneEntities, fixed: PlaneEntities) -> None:
        system.add_constraint(
            self.Constraint.PERPENDICULAR,
            self.Entity.FREE_IN_3D,
            0.0,
            self.Entity.NONE,
            self.Entity.NONE,
            moving.normal_line,
            fixed.normal_line,
        )

    def _apply_cylinder_cylinder_concentric(
        self, system, moving: CylinderEntities, fixed: CylinderEntities
    ) -> None:
        system.coincident(moving.axis_point, fixed.axis_line, self.Entity.FREE_IN_3D)
        system.coincident(moving.axis_end, fixed.axis_line, self.Entity.FREE_IN_3D)

    def _apply_plane_cylinder_tangent(
        self,
        system,
        plane: PlaneEntities,
        cylinder: CylinderEntities,
        cylinder_feature: CylinderFeature,
        orientation: int,
    ) -> None:
        if plane.workplane is None:
            raise ValueError("Plane bundle does not have a workplane.")
        radius = abs(float(cylinder_feature.radius))
        signed_radius = -radius if int(orientation) == 2 else radius
        system.distance(cylinder.axis_point, plane.workplane, signed_radius, self.Entity.FREE_IN_3D)
        system.distance(cylinder.axis_end, plane.workplane, signed_radius, self.Entity.FREE_IN_3D)

    def _apply_cylinder_cylinder_tangent(
        self,
        system,
        moving: CylinderEntities,
        fixed: CylinderEntities,
        moving_feature: CylinderFeature,
        fixed_feature: CylinderFeature,
    ) -> None:
        distance = abs(float(moving_feature.radius)) + abs(float(fixed_feature.radius))
        system.add_constraint(
            self.Constraint.ANGLE,
            self.Entity.FREE_IN_3D,
            0.0,
            self.Entity.NONE,
            self.Entity.NONE,
            moving.axis_line,
            fixed.axis_line,
        )
        system.distance(moving.axis_point, fixed.axis_line, distance, self.Entity.FREE_IN_3D)
        system.distance(moving.axis_end, fixed.axis_line, distance, self.Entity.FREE_IN_3D)

    def _apply_constraint(
        self,
        system,
        constraint: PairConstraint,
        moving_bundles: Dict[int, EntityBundle],
        fixed_bundles: Dict[int, EntityBundle],
    ) -> None:
        moving_ref, fixed_ref = self._split_constraint(constraint)
        moving_feature = self._feature_for(moving_ref)
        fixed_feature = self._feature_for(fixed_ref)
        moving_bundle = moving_bundles[moving_ref.face_index]
        fixed_bundle = fixed_bundles[fixed_ref.face_index]

        if isinstance(moving_feature, PlaneFeature) and isinstance(fixed_feature, PlaneFeature):
            if constraint.kind == "tangent":
                self._apply_plane_plane_coincident(system, moving_bundle, fixed_bundle, constraint.orientation)
                return
            if constraint.kind == "coincident":
                self._apply_plane_plane_coincident(
                    system,
                    moving_bundle,
                    fixed_bundle,
                    constraint.orientation,
                )
                return
            if constraint.kind == "distance":
                self._apply_plane_plane_distance(system, moving_bundle, fixed_bundle, float(constraint.value))
                return
            if constraint.kind == "perpendicular":
                self._apply_plane_plane_perpendicular(system, moving_bundle, fixed_bundle)
                return
            if constraint.kind == "angle":
                if abs(constraint.value) < 1e-8 or abs(abs(constraint.value) - 180.0) < 1e-8:
                    raise NotImplementedError(
                        f"{constraint.name}: plane-plane angle {constraint.value} degrees is not supported in the simplified solver."
                    )
                self._apply_plane_plane_angle(system, moving_bundle, fixed_bundle, float(constraint.value))
                return
            if constraint.kind == "parallel":
                raise NotImplementedError(
                    f"{constraint.name}: pure plane-plane parallel constraints are intentionally not supported in the simplified solver."
                )

        if isinstance(moving_feature, CylinderFeature) and isinstance(fixed_feature, CylinderFeature):
            if constraint.kind in {"coincident", "concentric"}:
                self._apply_cylinder_cylinder_concentric(system, moving_bundle, fixed_bundle)
                return
            if constraint.kind == "tangent":
                self._apply_cylinder_cylinder_tangent(
                    system,
                    moving_bundle,
                    fixed_bundle,
                    moving_feature,
                    fixed_feature,
                )
                return

        if isinstance(moving_feature, CylinderFeature) and isinstance(fixed_feature, PlaneFeature):
            if constraint.kind == "tangent":
                self._apply_plane_cylinder_tangent(
                    system,
                    fixed_bundle,
                    moving_bundle,
                    moving_feature,
                    constraint.orientation,
                )
                return

        if isinstance(moving_feature, PlaneFeature) and isinstance(fixed_feature, CylinderFeature):
            if constraint.kind == "tangent":
                self._apply_plane_cylinder_tangent(
                    system,
                    moving_bundle,
                    fixed_bundle,
                    fixed_feature,
                    constraint.orientation,
                )
                return

        raise NotImplementedError(
            f"{constraint.name}: unsupported constraint '{constraint.kind}' for "
            f"{type(moving_feature).__name__} <-> {type(fixed_feature).__name__}."
        )

    def _solve_solvespace_debug_legacy(self) -> np.ndarray:
        system = self.SolverSystem()
        system.set_group(1)

        fixed_bundles: Dict[int, EntityBundle] = {}
        moving_bundles: Dict[int, EntityBundle] = {}

        for constraint in self.constraints:
            moving_ref, fixed_ref = self._split_constraint(constraint)

            if fixed_ref.face_index not in fixed_bundles:
                fixed_feature = self._feature_for(fixed_ref)
                fixed_bundles[fixed_ref.face_index] = self._create_bundle(system, fixed_feature, fixed=True)

        system.set_group(2)

        for constraint in self.constraints:
            moving_ref, fixed_ref = self._split_constraint(constraint)

            if moving_ref.face_index not in moving_bundles:
                moving_feature = self._feature_for(moving_ref)
                moving_bundles[moving_ref.face_index] = self._create_bundle(system, moving_feature, fixed=False)

        source_points, moving_entities = self._rigidify_moving_part(system, moving_bundles)

        # --- 暴力调试模式：把日志塞进报错信息里 ---
        batch_history = []
        for constraint in self.constraints:
            try:
                self._apply_constraint(system, constraint, moving_bundles=moving_bundles, fixed_bundles=fixed_bundles)
                batch_history.append(f"APPLIED: {constraint.name} ({constraint.kind})")
            except Exception as exc:
                error_msg = "\n" + "=" * 50 + "\n"
                error_msg += "【调试追踪日志】\n"
                if batch_history:
                    error_msg += "\n".join(batch_history) + "\n"
                error_msg += f"❌ 约束应用失败: '{constraint.name}' (类型: {constraint.kind})\n"
                error_msg += f"代码异常: {exc}\n"
                error_msg += "=" * 50
                raise RuntimeError(error_msg) from exc

        result = system.solve()
        if result != self.ResultFlag.OKAY:
            error_msg = "\n" + "=" * 50 + "\n"
            error_msg += "【调试追踪日志】\n"
            if batch_history:
                error_msg += "\n".join(batch_history) + "\n"
            error_msg += "❌ 整体求解失败\n"
            error_msg += f"内部状态码: {result}, 冲突实体 ID: {system.failures()}\n"
            error_msg += "=" * 50
            raise RuntimeError(error_msg)

        solved_points = np.asarray([system.params(entity.params) for entity in moving_entities], dtype=float)
        return rigid_transform_from_tripod(source_points, solved_points)

        

    def solve(self) -> np.ndarray:
        system = self.SolverSystem()
        system.set_group(1)

        fixed_bundles: Dict[int, EntityBundle] = {}
        moving_bundles: Dict[int, EntityBundle] = {}

        for constraint in self.constraints:
            _, fixed_ref = self._split_constraint(constraint)
            if fixed_ref.face_index not in fixed_bundles:
                fixed_feature = self._feature_for(fixed_ref)
                fixed_bundles[fixed_ref.face_index] = self._create_bundle(system, fixed_feature, fixed=True)

        system.set_group(2)

        for constraint in self.constraints:
            moving_ref, _ = self._split_constraint(constraint)
            if moving_ref.face_index not in moving_bundles:
                moving_feature = self._feature_for(moving_ref)
                moving_bundles[moving_ref.face_index] = self._create_bundle(system, moving_feature, fixed=False)

        source_points, moving_entities = self._rigidify_moving_part(system, moving_bundles)

        batch_history = []
        for constraint in self.constraints:
            try:
                self._apply_constraint(system, constraint, moving_bundles=moving_bundles, fixed_bundles=fixed_bundles)
                batch_history.append(f"APPLIED: {constraint.name} ({constraint.kind})")
            except Exception as exc:
                error_msg = "\n" + "=" * 50 + "\n"
                error_msg += "Failed while applying constraints.\n"
                if batch_history:
                    error_msg += "\n".join(batch_history) + "\n"
                error_msg += f"Constraint: {constraint.name} ({constraint.kind})\n"
                error_msg += f"Exception: {exc}\n"
                error_msg += "=" * 50
                raise RuntimeError(error_msg) from exc

        result = system.solve()
        if result != self.ResultFlag.OKAY:
            try:
                return self._solve_analytically()
            except Exception as analytic_exc:
                error_msg = "\n" + "=" * 50 + "\n"
                error_msg += "SolveSpace failed, and the analytic fallback could not recover the transform.\n"
                if batch_history:
                    error_msg += "\n".join(batch_history) + "\n"
                error_msg += f"SolveSpace status: {result}, failure entity IDs: {system.failures()}\n"
                error_msg += f"Analytic fallback error: {analytic_exc}\n"
                error_msg += "=" * 50
                raise RuntimeError(error_msg) from analytic_exc

        solved_points = np.asarray([system.params(entity.params) for entity in moving_entities], dtype=float)
        return rigid_transform_from_tripod(source_points, solved_points)


def solve_pair_from_payload(
    parts_payload: Sequence[dict],
    constraints_payload: Sequence[dict],
    step_dir: Optional[str] = None,
    base_dir: Optional[str] = None,
    fixed_part: Optional[str] = None,
    moving_part: Optional[str] = None,
    face_index_base: int = 0,
) -> dict:
    base_dir = os.path.abspath(base_dir or os.getcwd())
    part_paths = build_parts_from_payload(parts_payload, step_dir=step_dir, base_dir=base_dir)
    constraints = extract_constraints({"constraints": list(constraints_payload)}, face_index_base=face_index_base)
    ordered_part_names = [part.get("part_name") or part.get("part_id") for part in parts_payload]
    fixed_name, moving_name, pair_constraints = select_pair_constraints(
        constraints,
        part_order=[str(name) for name in ordered_part_names if name],
        fixed_part=fixed_part,
        moving_part=moving_part,
    )

    solver = PairAssemblySolver(
        fixed_part=fixed_name,
        moving_part=moving_name,
        part_paths={fixed_name: part_paths[fixed_name], moving_name: part_paths[moving_name]},
        constraints=pair_constraints,
    )
    transform = solver.solve()

    return {
        "fixed_part": fixed_name,
        "moving_part": moving_name,
        "constraint_names": [constraint.name for constraint in pair_constraints],
        "transform": matrix_to_list(transform),
    }


def solve_pair_from_json(
    assembly_json_path: str,
    step_dir: Optional[str] = None,
    fixed_part: Optional[str] = None,
    moving_part: Optional[str] = None,
    face_index_base: int = 0,
) -> dict:
    data = load_json(assembly_json_path)
    json_dir = os.path.dirname(os.path.abspath(assembly_json_path))
    part_paths = build_parts(data, step_dir=step_dir, base_dir=json_dir)
    constraints = extract_constraints(data, face_index_base=face_index_base)
    ordered_part_names = [part.get("part_name") or part.get("part_id") for part in (data.get("parts") or [])]
    fixed_name, moving_name, pair_constraints = select_pair_constraints(
        constraints,
        part_order=[str(name) for name in ordered_part_names if name],
        fixed_part=fixed_part,
        moving_part=moving_part,
    )

    solver = PairAssemblySolver(
        fixed_part=fixed_name,
        moving_part=moving_name,
        part_paths={fixed_name: part_paths[fixed_name], moving_name: part_paths[moving_name]},
        constraints=pair_constraints,
    )
    transform = solver.solve()

    return {
        "assembly_json": os.path.abspath(assembly_json_path),
        "fixed_part": fixed_name,
        "moving_part": moving_name,
        "constraint_names": [constraint.name for constraint in pair_constraints],
        "transform": matrix_to_list(transform),
    }


def diagnose_pair_transform_from_json(
    assembly_json_path: str,
    transform: Sequence[Sequence[float]],
    step_dir: Optional[str] = None,
    fixed_part: Optional[str] = None,
    moving_part: Optional[str] = None,
    face_index_base: int = 0,
) -> dict:
    data = load_json(assembly_json_path)
    json_dir = os.path.dirname(os.path.abspath(assembly_json_path))
    part_paths = build_parts(data, step_dir=step_dir, base_dir=json_dir)
    constraints = extract_constraints(data, face_index_base=face_index_base)
    ordered_part_names = [part.get("part_name") or part.get("part_id") for part in (data.get("parts") or [])]
    fixed_name, moving_name, pair_constraints = select_pair_constraints(
        constraints,
        part_order=[str(name) for name in ordered_part_names if name],
        fixed_part=fixed_part,
        moving_part=moving_part,
    )

    solver = PairAssemblySolver(
        fixed_part=fixed_name,
        moving_part=moving_name,
        part_paths={fixed_name: part_paths[fixed_name], moving_name: part_paths[moving_name]},
        constraints=pair_constraints,
    )
    diagnosis = solver.diagnose_transform(transform)
    diagnosis["assembly_json"] = os.path.abspath(assembly_json_path)
    diagnosis["fixed_part"] = fixed_name
    diagnosis["moving_part"] = moving_name
    diagnosis["constraint_names"] = [constraint.name for constraint in pair_constraints]
    return diagnosis


def solve_from_payload(
    parts_payload: Sequence[dict],
    constraints_payload: Sequence[dict],
    step_dir: Optional[str] = None,
    base_dir: Optional[str] = None,
    fixed_part: Optional[str] = None,
    moving_part: Optional[str] = None,
    face_index_base: int = 0,
) -> dict:
    return solve_pair_from_payload(
        parts_payload=parts_payload,
        constraints_payload=constraints_payload,
        step_dir=step_dir,
        base_dir=base_dir,
        fixed_part=fixed_part,
        moving_part=moving_part,
        face_index_base=face_index_base,
    )


def solve_from_json(
    assembly_json_path: str,
    step_dir: Optional[str] = None,
    fixed_part: Optional[str] = None,
    moving_part: Optional[str] = None,
    face_index_base: int = 0,
) -> dict:
    return solve_pair_from_json(
        assembly_json_path=assembly_json_path,
        step_dir=step_dir,
        fixed_part=fixed_part,
        moving_part=moving_part,
        face_index_base=face_index_base,
    )


def _default_sample_json() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "（5，11）多轴机械臂（已改）1_multi axis robot arm.json")
    return candidate if os.path.isfile(candidate) else None


def _default_sample_step_dir() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidate = os.path.join(
        repo_root,
        "step_export",
        "（5，11）多轴机械臂（已改）1_multi axis robot arm",
        "result",
        "retrieved_steps",
    )
    return candidate if os.path.isdir(candidate) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a two-part assembly from STEP + face constraints using python-solvespace."
    )
    parser.add_argument(
        "--assembly-json",
        default=_default_sample_json(),
        help="Assembly JSON path. If omitted and the bundled sample exists, the sample is used.",
    )
    parser.add_argument(
        "--step-dir",
        default=_default_sample_step_dir(),
        help="Optional STEP directory used to resolve part STEP files by filename.",
    )
    parser.add_argument(
        "--fixed-part",
        help="Name of the fixed part. If omitted, the script auto-selects a pair and fixes the earlier part in input order.",
    )
    parser.add_argument(
        "--moving-part",
        help="Name of the moving part. If omitted, the script auto-selects a pair with the most constraints.",
    )
    parser.add_argument(
        "--face-index-base",
        type=int,
        default=0,
        choices=(0, 1),
        help="Whether face indices in constraints are 0-based or 1-based.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, the result is printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.assembly_json:
        raise SystemExit("Provide --assembly-json or place the sample JSON next to this script.")

    payload = solve_pair_from_json(
        assembly_json_path=args.assembly_json,
        step_dir=args.step_dir,
        fixed_part=args.fixed_part,
        moving_part=args.moving_part,
        face_index_base=args.face_index_base,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
