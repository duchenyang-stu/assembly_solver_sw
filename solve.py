from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import core
except ImportError:  # pragma: no cover - direct script execution fallback
    import core

from .collision import CollisionSettings, resolve_collision
from .input_loader import AssemblyInput, PairJob


@dataclass(frozen=True)
class SolveRecord:
    index: int
    fixed_part: str
    moving_part: str
    constraint_names: list[str]
    constraint_kinds: list[str]
    status: str
    solver_mode: str
    transform: list[list[float]] | None = None
    solver_used: str | None = None
    max_constraint_error: float | None = None
    primary_error: str | None = None
    collision: dict[str, Any] | None = None
    collision_adjusted: bool | None = None
    rejected_transform: list[list[float]] | None = None
    selected_candidate: str | None = None
    selected_variant: str | None = None
    candidate_results: list[dict[str, Any]] | None = None
    error: str | None = None
    traceback: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class StrictPairAssemblySolver(core.PairAssemblySolver):
    """Use SolveSpace only; strict failures stay failures."""

    def solve(self) -> np.ndarray:
        system = self.SolverSystem()
        system.set_group(1)
        fixed_bundles = {}
        moving_bundles = {}

        for constraint in self.constraints:
            _, fixed_ref = self._split_constraint(constraint)
            if fixed_ref.face_index not in fixed_bundles:
                fixed_bundles[fixed_ref.face_index] = self._create_bundle(
                    system, self._feature_for(fixed_ref), fixed=True
                )

        system.set_group(2)
        for constraint in self.constraints:
            moving_ref, _ = self._split_constraint(constraint)
            if moving_ref.face_index not in moving_bundles:
                moving_bundles[moving_ref.face_index] = self._create_bundle(
                    system, self._feature_for(moving_ref), fixed=False
                )

        source_points, moving_entities = self._rigidify_moving_part(system, moving_bundles)
        history: list[str] = []
        for constraint in self.constraints:
            self._apply_constraint(system, constraint, moving_bundles, fixed_bundles)
            history.append(f"APPLIED: {constraint.name} ({constraint.kind})")

        result = system.solve()
        if result != self.ResultFlag.OKAY:
            detail = "\n".join(history)
            raise RuntimeError(
                f"SolveSpace failed in strict mode. Status: {result}; "
                f"failure entity IDs: {system.failures()}\n{detail}"
            )

        solved_points = np.asarray([system.params(entity.params) for entity in moving_entities], dtype=float)
        return np.asarray(core.rigid_transform_from_tripod(source_points, solved_points), dtype=float)


@dataclass
class _CandidateOutcome:
    label: str
    constraints: list[core.PairConstraint]
    flipped_constraints: list[str]
    omitted_constraints: list[str]
    variant_label: str | None = None
    solver: StrictPairAssemblySolver | None = None
    transform: list[list[float]] | np.ndarray | None = None
    solver_used: str | None = None
    primary_error: str | None = None
    max_constraint_error: float | None = None
    collision: dict[str, Any] | None = None
    collision_adjusted: bool | None = None
    error: str | None = None
    traceback: str | None = None

    @property
    def has_transform(self) -> bool:
        return self.transform is not None and self.solver is not None and self.error is None


def solve_jobs(
    assembly: AssemblyInput,
    jobs: list[PairJob],
    *,
    solver_mode: str = "solvespace-then-analytic",
    max_error: float = 1e-4,
    reject_high_error: bool = False,
    avoid_interference: bool = True,
    fail_on_interference: bool = True,
    contact_tolerance: float = 1e-3,
    common_volume_tolerance: float = 1e-3,
    residual_good_tolerance: float = 1e-4,
    residual_bad_tolerance: float = 1e-2,
    search_free_rotation: bool = True,
    rotation_sample_count: int = 24,
) -> list[SolveRecord]:
    if solver_mode not in {"solvespace", "analytic", "solvespace-then-analytic"}:
        raise ValueError("solver_mode must be one of: solvespace, analytic, solvespace-then-analytic")

    records: list[SolveRecord] = []
    for job in jobs:
        names = [constraint.name for constraint in job.constraints]
        kinds = [constraint.kind for constraint in job.constraints]
        settings = CollisionSettings(
            enabled=avoid_interference,
            # Candidate selection needs metrics for every transform; final
            # accept/reject happens after all candidates are scored.
            fail_on_interference=False,
            contact_tolerance=contact_tolerance,
            common_volume_tolerance=common_volume_tolerance,
            residual_good_tolerance=residual_good_tolerance,
            residual_bad_tolerance=residual_bad_tolerance,
            search_free_rotation=search_free_rotation,
            rotation_sample_count=rotation_sample_count,
        )
        outcomes = [
            outcome
            for label, constraints, flipped_constraints, omitted_constraints in _candidate_constraint_sets(job)
            for outcome in _evaluate_candidate(
                assembly,
                job,
                label,
                constraints,
                flipped_constraints,
                omitted_constraints,
                solver_mode,
                max_error=max_error,
                reject_high_error=reject_high_error,
                avoid_interference=avoid_interference,
                settings=settings,
            )
        ]
        _annotate_weak_constraint_suspicion(outcomes)
        candidate_results = [_candidate_summary(outcome) for outcome in outcomes]
        acceptable = [
            outcome
            for outcome in outcomes
            if _candidate_is_acceptable(
                outcome,
                max_error=max_error,
                avoid_interference=avoid_interference,
            )
        ]

        complete_acceptable = [outcome for outcome in acceptable if not outcome.omitted_constraints]
        if complete_acceptable:
            outcome = min(complete_acceptable, key=_candidate_score)
            diagnosis = outcome.solver.diagnose_transform(outcome.transform)
            records.append(
                SolveRecord(
                    job.index,
                    job.fixed_part,
                    job.moving_part,
                    names,
                    kinds,
                    "ok",
                    solver_mode,
                    core.matrix_to_list(outcome.transform),
                    solver_used=outcome.solver_used,
                    max_constraint_error=float(diagnosis["max_error"]),
                    primary_error=outcome.primary_error,
                    collision=outcome.collision,
                    collision_adjusted=outcome.collision_adjusted,
                    selected_candidate=outcome.label,
                    selected_variant=outcome.variant_label,
                    candidate_results=candidate_results,
                )
            )
            continue

        if acceptable:
            outcome = min(acceptable, key=_candidate_score)
            diagnosis = outcome.solver.diagnose_transform(outcome.transform)
            records.append(
                SolveRecord(
                    job.index,
                    job.fixed_part,
                    job.moving_part,
                    names,
                    kinds,
                    "partial",
                    solver_mode,
                    core.matrix_to_list(outcome.transform),
                    solver_used=outcome.solver_used,
                    max_constraint_error=float(diagnosis["max_error"]),
                    primary_error=outcome.primary_error,
                    collision=outcome.collision,
                    collision_adjusted=outcome.collision_adjusted,
                    selected_candidate=outcome.label,
                    selected_variant=outcome.variant_label,
                    candidate_results=candidate_results,
                    error=(
                        "Solved a degraded constraint set after omitting: "
                        + ", ".join(outcome.omitted_constraints)
                    ),
                )
            )
            continue

        transform_outcomes = [outcome for outcome in outcomes if outcome.has_transform]
        if transform_outcomes:
            complete_transform_outcomes = [
                outcome for outcome in transform_outcomes if not outcome.omitted_constraints
            ]
            outcome_pool = complete_transform_outcomes or transform_outcomes
            outcome = min(outcome_pool, key=_candidate_score)
            records.append(
                SolveRecord(
                    job.index,
                    job.fixed_part,
                    job.moving_part,
                    names,
                    kinds,
                    "collision" if avoid_interference and fail_on_interference else "error",
                    solver_mode,
                    solver_used=outcome.solver_used,
                    max_constraint_error=outcome.max_constraint_error,
                    primary_error=outcome.primary_error,
                    collision=outcome.collision,
                    collision_adjusted=outcome.collision_adjusted,
                    rejected_transform=core.matrix_to_list(outcome.transform),
                    selected_candidate=outcome.label,
                    selected_variant=outcome.variant_label,
                    candidate_results=candidate_results,
                    error=_candidate_rejection_reason(outcome, avoid_interference=avoid_interference),
                )
            )
            continue

        first_error = next((outcome for outcome in outcomes if outcome.error), outcomes[0])
        records.append(
            SolveRecord(
                job.index,
                job.fixed_part,
                job.moving_part,
                names,
                kinds,
                "error",
                solver_mode,
                selected_candidate=first_error.label,
                candidate_results=candidate_results,
                error=first_error.error or "No candidate produced a transform.",
                traceback=first_error.traceback,
            )
        )
    return records


def _evaluate_candidate(
    assembly: AssemblyInput,
    job: PairJob,
    label: str,
    constraints: list[core.PairConstraint],
    flipped_constraints: list[str],
    omitted_constraints: list[str],
    solver_mode: str,
    *,
    max_error: float,
    reject_high_error: bool,
    avoid_interference: bool,
    settings: CollisionSettings,
) -> list[_CandidateOutcome]:
    error_outcome = _CandidateOutcome(label, constraints, flipped_constraints, omitted_constraints)
    try:
        solver = StrictPairAssemblySolver(
            fixed_part=job.fixed_part,
            moving_part=job.moving_part,
            part_paths={
                job.fixed_part: str(assembly.part_paths[job.fixed_part]),
                job.moving_part: str(assembly.part_paths[job.moving_part]),
            },
            constraints=constraints,
        )
        transform_candidates = _solve_candidates_with_mode(
            solver,
            solver_mode,
            max_error=max_error,
            reject_high_error=reject_high_error,
        )
        if not transform_candidates:
            raise RuntimeError("No transform candidate was produced.")

        outcomes: list[_CandidateOutcome] = []
        for transform, solver_used, primary_error, base_variant_label in transform_candidates:
            try:
                expanded_transforms = _expand_concentric_free_motion_candidates(
                    solver,
                    transform,
                    max_error=max_error,
                    sample_count=settings.rotation_sample_count,
                )
            except Exception:
                expanded_transforms = [(transform, None)]

            for expanded_transform, motion_label in expanded_transforms:
                variant_label = base_variant_label
                if motion_label:
                    variant_label = f"{variant_label}:{motion_label}" if variant_label else motion_label
                outcome = _CandidateOutcome(
                    label,
                    constraints,
                    flipped_constraints,
                    omitted_constraints,
                    variant_label=variant_label,
                )
                collision = None
                collision_adjusted = None
                final_transform = expanded_transform
                if avoid_interference:
                    collision_resolution = resolve_collision(
                        solver,
                        fixed_step_path=assembly.part_paths[job.fixed_part],
                        moving_step_path=assembly.part_paths[job.moving_part],
                        transform=expanded_transform,
                        settings=settings,
                    )
                    collision = collision_resolution.analysis
                    collision_adjusted = bool(collision_resolution.adjusted)
                    final_transform = collision_resolution.transform

                diagnosis = solver.diagnose_transform(final_transform)
                outcome.solver = solver
                outcome.transform = final_transform
                outcome.solver_used = solver_used
                outcome.primary_error = primary_error
                outcome.max_constraint_error = float(diagnosis["max_error"])
                outcome.collision = collision
                outcome.collision_adjusted = collision_adjusted
                outcomes.append(outcome)
        return outcomes
    except Exception as exc:
        error_outcome.error = f"{type(exc).__name__}: {exc}"
        error_outcome.traceback = traceback.format_exc()
    return [error_outcome]


def _candidate_constraint_sets(
    job: PairJob,
) -> list[tuple[str, list[core.PairConstraint], list[str], list[str]]]:
    constraints = _sort_constraints_for_solving(job.constraints)
    candidates: list[tuple[str, list[core.PairConstraint], list[str], list[str]]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    def add_candidate(
        label: str,
        candidate_constraints: list[core.PairConstraint],
        flipped_constraints: list[str] | None = None,
        omitted_constraints: list[str] | None = None,
    ) -> None:
        ordered = _sort_constraints_for_solving(candidate_constraints)
        if not ordered:
            return
        key = tuple((constraint.name, int(constraint.orientation)) for constraint in ordered)
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, ordered, list(flipped_constraints or []), list(omitted_constraints or [])))

    add_candidate("primary", constraints)

    flippable_coincident = [constraint.name for constraint in constraints if _is_oriented_plane_coincident(constraint)]
    flippable_tangent = [
        constraint.name
        for constraint in constraints
        if constraint.kind == "tangent" and int(constraint.orientation) in {1, 2}
    ]
    if flippable_coincident:
        add_candidate(
            "flipped_coincident_orientation",
            _flip_named_orientations(constraints, set(flippable_coincident)),
            flippable_coincident,
        )
    if flippable_tangent:
        add_candidate(
            "flipped_tangent_orientation",
            _flip_named_orientations(constraints, set(flippable_tangent)),
            flippable_tangent,
        )
    if flippable_coincident and flippable_tangent:
        flipped_names = flippable_coincident + flippable_tangent
        add_candidate(
            "flipped_coincident_and_tangent_orientation",
            _flip_named_orientations(constraints, set(flipped_names)),
            flipped_names,
        )

    # Individual flips catch common mixed-orientation cases without exploding the
    # search space on dense prediction sets.
    for constraint_name in (flippable_coincident + flippable_tangent)[:8]:
        add_candidate(
            f"flipped_orientation:{constraint_name}",
            _flip_named_orientations(constraints, {constraint_name}),
            [constraint_name],
        )

    for label, subset in _degraded_constraint_subsets(constraints):
        omitted = [constraint.name for constraint in constraints if constraint.name not in {item.name for item in subset}]
        add_candidate(label, subset, [], omitted)
    return candidates


def _is_oriented_plane_coincident(constraint: core.PairConstraint) -> bool:
    if constraint.kind != "coincident" or int(constraint.orientation) not in {1, 2}:
        return False
    # Many source payloads do not carry face_type, so requiring "plane" here
    # prevents coplanar side alternatives from being explored. Non-planar
    # coincident flips are still scored by residual and collision checks.
    return True


def _constraint_strength_rank(constraint: core.PairConstraint) -> tuple[int, int, str]:
    kind = constraint.kind
    face_types = {ref.face_type.strip().lower() for ref in constraint.refs}
    if kind in {"coincident", "concentric"}:
        tier = 0
    elif kind == "distance":
        tier = 1
    elif kind == "tangent" and face_types == {"plane"}:
        # Plane-plane tangent is modeled as coincidence in this solver.
        tier = 1
    elif kind in {"perpendicular", "angle"}:
        tier = 2
    elif kind == "parallel":
        tier = 3
    elif kind == "tangent":
        tier = 4
    else:
        tier = 5
    return (tier, int(constraint.orientation), constraint.name)


def _sort_constraints_for_solving(
    constraints: list[core.PairConstraint] | tuple[core.PairConstraint, ...],
) -> list[core.PairConstraint]:
    return sorted(list(constraints), key=_constraint_strength_rank)


def _flip_named_orientations(
    constraints: list[core.PairConstraint],
    names: set[str],
) -> list[core.PairConstraint]:
    flipped: list[core.PairConstraint] = []
    for constraint in constraints:
        if constraint.name in names and int(constraint.orientation) in {1, 2}:
            flipped_orientation = 1 if int(constraint.orientation) == 2 else 2
            flipped.append(replace(constraint, orientation=flipped_orientation))
        else:
            flipped.append(constraint)
    return flipped


def _degraded_constraint_subsets(
    constraints: list[core.PairConstraint],
) -> list[tuple[str, list[core.PairConstraint]]]:
    if len(constraints) <= 1:
        return []

    anchors = [
        constraint
        for constraint in constraints
        if constraint.kind in {"coincident", "concentric", "distance"}
        or (constraint.kind == "tangent" and {ref.face_type.strip().lower() for ref in constraint.refs} == {"plane"})
    ]
    non_tangent = [constraint for constraint in constraints if constraint.kind != "tangent"]
    positional_orientation = [
        constraint
        for constraint in constraints
        if constraint.kind in {"coincident", "concentric", "distance", "parallel", "perpendicular", "angle"}
    ]

    subsets: list[tuple[str, list[core.PairConstraint]]] = []
    if anchors and len(anchors) < len(constraints):
        subsets.append(("degraded_anchor_only", anchors))
    if anchors and non_tangent and len(non_tangent) < len(constraints):
        subsets.append(("degraded_without_tangent", non_tangent))
    if positional_orientation and len(positional_orientation) < len(constraints):
        subsets.append(("degraded_without_contact", positional_orientation))
    return subsets


def _candidate_is_acceptable(
    outcome: _CandidateOutcome,
    *,
    max_error: float,
    avoid_interference: bool,
) -> bool:
    if not outcome.has_transform or outcome.max_constraint_error is None:
        return False
    if float(outcome.max_constraint_error) > float(max_error):
        return False
    if not avoid_interference:
        return True
    metrics = _candidate_final_metrics(outcome)
    return metrics.get("status") in {"clearance", "contact"}


def _candidate_score(outcome: _CandidateOutcome) -> tuple[float, float, float, float, float, int, int]:
    metrics = _candidate_final_metrics(outcome)
    status = str(metrics.get("status") or "")
    if status in {"clearance", "contact", ""}:
        collision_rank = 0.0
    elif status == "possible_interference":
        collision_rank = 1.0
    else:
        collision_rank = 2.0
    common_volume = float(metrics.get("common_volume") or 0.0)
    bbox_overlap = float(metrics.get("bbox_overlap_volume") or 0.0)
    min_distance = float(metrics.get("min_distance") or 0.0)
    max_error = float(outcome.max_constraint_error if outcome.max_constraint_error is not None else 1e9)
    omitted_count = len(outcome.omitted_constraints)
    priority = 0 if outcome.label == "primary" else 1
    return (collision_rank, common_volume, bbox_overlap, max_error, -min_distance, omitted_count, priority)


def _candidate_final_metrics(outcome: _CandidateOutcome) -> dict[str, Any]:
    if not outcome.collision:
        return {"status": "clearance", "common_volume": 0.0, "min_distance": None, "bbox_overlap_volume": 0.0}
    return outcome.collision.get("final_metrics") or {}


def _candidate_summary(outcome: _CandidateOutcome) -> dict[str, Any]:
    metrics = {} if outcome.error else _candidate_final_metrics(outcome)
    summary: dict[str, Any] = {
        "label": outcome.label,
        "variant_label": outcome.variant_label,
        "flipped_constraints": outcome.flipped_constraints,
        "status": "error" if outcome.error else "solved",
        "solver_used": outcome.solver_used,
        "constraint_names": [constraint.name for constraint in outcome.constraints],
        "constraint_kinds": [constraint.kind for constraint in outcome.constraints],
        "omitted_constraints": outcome.omitted_constraints,
        "degraded": bool(outcome.omitted_constraints),
        "max_constraint_error": outcome.max_constraint_error,
        "collision_status": metrics.get("status"),
        "common_volume": metrics.get("common_volume"),
        "min_distance": metrics.get("min_distance"),
        "bbox_overlap_volume": metrics.get("bbox_overlap_volume"),
        "collision_adjusted": outcome.collision_adjusted,
        "score": None if outcome.error else list(_candidate_score(outcome)),
    }
    suspicious = getattr(outcome, "suspicious_constraints", None)
    if suspicious:
        summary["suspicious_constraints"] = suspicious
    if outcome.error:
        summary["error"] = outcome.error
    if outcome.collision:
        summary["classification"] = outcome.collision.get("classification")
    return summary


def _candidate_rejection_reason(outcome: _CandidateOutcome, *, avoid_interference: bool) -> str:
    if outcome.error:
        return outcome.error
    metrics = _candidate_final_metrics(outcome)
    if avoid_interference and metrics.get("status") not in {"clearance", "contact"}:
        return (
            f"Rejected all transform candidates; best candidate '{outcome.label}' still has "
            f"collision status '{metrics.get('status')}'."
        )
    return f"Rejected all transform candidates; best candidate '{outcome.label}' did not satisfy tolerances."


def _solve_candidates_with_mode(
    solver: StrictPairAssemblySolver,
    solver_mode: str,
    *,
    max_error: float,
    reject_high_error: bool,
) -> list[tuple[list[list[float]] | np.ndarray, str, str | None, str]]:
    if solver_mode == "solvespace":
        return [(solver.solve(), "solvespace", None, "solvespace")]

    if solver_mode == "analytic":
        transforms = solver._solve_analytic_candidates(max_error=max_error, max_candidates=8)
        if not transforms:
            return [(solver._solve_analytically(), "analytic", None, "analytic_001")]
        return [
            (transform, "analytic", None, f"analytic_{index:03d}")
            for index, transform in enumerate(transforms, start=1)
        ]

    candidates: list[tuple[list[list[float]] | np.ndarray, str, str | None, str]] = []
    primary_error: str | None = None
    try:
        transform = solver.solve()
        if not reject_high_error or float(solver.diagnose_transform(transform)["max_error"]) <= float(max_error):
            candidates.append((transform, "solvespace", None, "solvespace"))
        else:
            primary_error = (
                "SolveSpace returned OK, but max_constraint_error exceeded "
                f"threshold {float(max_error):.6g}."
            )
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"

    try:
        analytic_transforms = solver._solve_analytic_candidates(max_error=max_error, max_candidates=8)
        for index, transform in enumerate(analytic_transforms, start=1):
            candidates.append((transform, "analytic", primary_error, f"analytic_{index:03d}"))
    except Exception:
        if not candidates:
            raise

    return _dedupe_transform_candidates(candidates)


def _dedupe_transform_candidates(
    candidates: list[tuple[list[list[float]] | np.ndarray, str, str | None, str]]
) -> list[tuple[list[list[float]] | np.ndarray, str, str | None, str]]:
    deduped = []
    seen: set[tuple[float, ...]] = set()
    for transform, solver_used, primary_error, label in candidates:
        matrix = np.asarray(transform, dtype=float)
        key = tuple(round(float(value), 8) for value in matrix.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((transform, solver_used, primary_error, label))
    return deduped


def _expand_concentric_free_motion_candidates(
    solver: StrictPairAssemblySolver,
    transform: list[list[float]] | np.ndarray,
    *,
    max_error: float,
    sample_count: int,
) -> list[tuple[np.ndarray, str | None]]:
    axis_info = _first_concentric_axis(solver)
    if axis_info is None:
        return [(np.asarray(transform, dtype=float), None)]

    axis_point, axis_direction, scale = axis_info
    angle_count = min(max(int(sample_count or 0), 2), 6)
    angles = [0.0] + [360.0 * index / float(angle_count) for index in range(1, angle_count)]
    shift_scale = max(float(scale), 1.0)
    shifts = [
        0.0,
        -0.15 * shift_scale,
        0.15 * shift_scale,
    ]

    base_transform = np.asarray(transform, dtype=float)
    candidates: list[tuple[np.ndarray, str | None]] = []
    seen: set[tuple[float, ...]] = set()
    for angle in angles:
        rotation_delta = _axis_rotation_transform(axis_point, axis_direction, angle)
        for shift in shifts:
            translation_delta = np.eye(4, dtype=float)
            translation_delta[:3, 3] = np.asarray(axis_direction, dtype=float) * float(shift)
            candidate = translation_delta @ rotation_delta @ base_transform
            diagnosis = solver.diagnose_transform(candidate)
            if float(diagnosis["max_error"]) > max(float(max_error), float(max_error) * 10.0):
                continue
            key = tuple(round(float(value), 8) for value in candidate.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            label = None if abs(angle) < 1e-12 and abs(shift) < 1e-12 else f"free_a{angle:.1f}_s{shift:.4g}"
            candidates.append((candidate, label))
            if len(candidates) >= 18:
                return candidates
    return candidates or [(base_transform, None)]


def _first_concentric_axis(solver: StrictPairAssemblySolver) -> tuple[np.ndarray, np.ndarray, float] | None:
    points: list[np.ndarray] = []
    radii: list[float] = []
    for constraint in solver.constraints:
        if constraint.kind not in {"concentric", "coincident"}:
            continue
        try:
            moving_ref, fixed_ref = solver._split_constraint(constraint)
            moving_feature = solver._feature_for(moving_ref)
            fixed_feature = solver._feature_for(fixed_ref)
        except Exception:
            continue
        if isinstance(moving_feature, core.CylinderFeature) and isinstance(fixed_feature, core.CylinderFeature):
            points.extend(
                [
                    np.asarray(moving_feature.axis_point, dtype=float),
                    np.asarray(fixed_feature.axis_point, dtype=float),
                    np.asarray(moving_feature.surface_point, dtype=float),
                    np.asarray(fixed_feature.surface_point, dtype=float),
                ]
            )
            radii.extend([abs(float(moving_feature.radius)), abs(float(fixed_feature.radius))])
            scale = max(
                max(radii, default=1.0) * 20.0,
                _point_cloud_scale(points),
                1.0,
            )
            return (
                np.asarray(fixed_feature.axis_point, dtype=float),
                core.normalize(fixed_feature.axis),
                float(scale),
            )
    return None


def _point_cloud_scale(points: list[np.ndarray]) -> float:
    if len(points) < 2:
        return 1.0
    array = np.asarray(points, dtype=float)
    span = np.max(array, axis=0) - np.min(array, axis=0)
    return float(np.linalg.norm(span))


def _axis_rotation_transform(axis_point: np.ndarray, axis_direction: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = core.normalize(axis_direction)
    point = np.asarray(axis_point, dtype=float)
    angle = math.radians(float(angle_deg))
    x_value, y_value, z_value = axis
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    one_minus_cos = 1.0 - cos_value
    rotation = np.asarray(
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
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return transform


def _annotate_weak_constraint_suspicion(outcomes: list[_CandidateOutcome]) -> None:
    complete_collision = [
        outcome
        for outcome in outcomes
        if outcome.has_transform
        and not outcome.omitted_constraints
        and _candidate_final_metrics(outcome).get("status") not in {"clearance", "contact"}
    ]
    if not complete_collision:
        return
    recovered_omissions: set[str] = set()
    for outcome in outcomes:
        if not outcome.has_transform or not outcome.omitted_constraints:
            continue
        metrics = _candidate_final_metrics(outcome)
        if metrics.get("status") in {"clearance", "contact"}:
            recovered_omissions.update(outcome.omitted_constraints)
    if not recovered_omissions:
        return
    for outcome in outcomes:
        if outcome.has_transform:
            outcome.suspicious_constraints = sorted(recovered_omissions)


def _solve_with_mode(
    solver: StrictPairAssemblySolver,
    solver_mode: str,
    *,
    max_error: float,
    reject_high_error: bool,
) -> tuple[list[list[float]] | np.ndarray, str, str | None]:
    if solver_mode == "solvespace":
        return solver.solve(), "solvespace", None
    if solver_mode == "analytic":
        return solver._solve_analytically(), "analytic", None

    try:
        transform = solver.solve()
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
        return solver._solve_analytically(), "analytic", primary_error

    if not reject_high_error:
        return transform, "solvespace", None

    diagnostic_error = float(solver.diagnose_transform(transform)["max_error"])
    if diagnostic_error <= float(max_error):
        return transform, "solvespace", None

    primary_error = (
        f"SolveSpace returned OK, but max_constraint_error {diagnostic_error:.6g} "
        f"exceeded threshold {float(max_error):.6g}."
    )
    return solver._solve_analytically(), "analytic", primary_error
