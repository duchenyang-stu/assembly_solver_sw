from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstructed_solver.batch_run import (
    _batch_result_json,
    _compact_json,
    _convert_constraint_payload,
    _read_json,
    _record_json,
    _record_matrix,
    _resolve_sample_json,
    _sample_status,
    _split_stem,
)
from reconstructed_solver.input_loader import collect_pair_jobs, load_assembly_payload
from reconstructed_solver.solve import solve_jobs
from reconstructed_solver.visualize import save_solution_step


SUPPORTED_PREDICTION_TYPES = {
    "coincident": "Coincident",
    "concentric": "Concentric",
    "parallel": "Parallel",
    "perpendicular": "Perpendicular",
}

TYPE_PRIORITY = {
    "Coincident": 0,
    "Concentric": 1,
    "Parallel": 2,
    "Perpendicular": 3,
}


@dataclass(frozen=True)
class PredictionConstraint:
    index: int
    raw: dict[str, Any]
    name: str
    constraint_type: str
    score: float
    part_a: str
    face_a: int
    part_b: str
    face_b: int
    direction: int

    @property
    def pair_key(self) -> tuple[str, int, str, int, str]:
        return (self.part_a, self.face_a, self.part_b, self.face_b, self.constraint_type)

    @property
    def face_pair_key(self) -> tuple[str, int, str, int]:
        return (self.part_a, self.face_a, self.part_b, self.face_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a compact, solver-consistent subset from model predictions, then "
            "solve/export assemblies with the same flow used for GT constraints."
        )
    )
    parser.add_argument(
        "--predictions-json",
        default=str(Path(__file__).resolve().parent / "predictions_test.json"),
        help="Model prediction JSON. Expected top-level list items with assembly_id and predictions.",
    )
    parser.add_argument(
        "--json-root",
        default="/home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair",
        help="Directory containing original per-sample JSON files. Used for parts and STEP resolution.",
    )
    parser.add_argument(
        "--step-root",
        default="/home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair/step",
        help="Directory containing part STEP/STP files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "prediction_output"),
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Prediction split to process, e.g. test or all.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--assembly-id",
        action="append",
        default=None,
        help="Process only this assembly_id. Can be provided multiple times.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 for easier debugging.",
    )
    parser.add_argument("--face-index-base", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--solver",
        default="solvespace-then-analytic",
        choices=("solvespace", "analytic", "solvespace-then-analytic"),
    )
    parser.add_argument("--max-error", type=float, default=1e-4)
    parser.add_argument("--reject-high-error", action="store_true")
    parser.add_argument("--allow-interference", action="store_true")
    parser.add_argument("--no-free-rotation-search", action="store_true")
    parser.add_argument("--contact-tolerance", type=float, default=1e-3)
    parser.add_argument("--common-volume-tolerance", type=float, default=1e-3)
    parser.add_argument("--rotation-samples", type=int, default=24)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Drop predictions below this score unless fallback needs them.",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=16,
        help="Keep at most this many scored predictions per sample before search.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=24,
        help="Maximum number of constraint subsets evaluated per sample.",
    )
    parser.add_argument(
        "--min-constraints",
        type=int,
        default=1,
        help="Smallest subset size considered during prediction selection.",
    )
    parser.add_argument(
        "--max-constraints",
        type=int,
        default=3,
        help="Largest subset size considered during prediction selection.",
    )
    parser.add_argument(
        "--prefer-mixed-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer subsets that contain both positional and directional constraint types.",
    )
    parser.add_argument(
        "--allow-direction-only",
        action="store_true",
        help=(
            "Allow a solved subset containing only Parallel/Perpendicular constraints even when "
            "Coincident/Concentric predictions are available."
        ),
    )
    parser.add_argument(
        "--export-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export rejected-but-solved transforms just like batch_run.py does.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only perform prediction conversion and candidate selection setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_json = Path(args.predictions_json).resolve()
    json_root = Path(args.json_root).resolve()
    step_root = Path(args.step_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not step_root.is_dir():
        raise NotADirectoryError(step_root)

    predictions = _select_prediction_items(_read_prediction_json(predictions_json), args.split, args.assembly_id)
    predictions = predictions[int(args.offset) :]
    if args.limit is not None:
        predictions = predictions[: int(args.limit)]

    context = {
        "json_root": str(json_root),
        "step_root": str(step_root),
        "output_dir": str(output_dir),
        "face_index_base": int(args.face_index_base),
        "solver_mode": args.solver,
        "max_error": float(args.max_error),
        "reject_high_error": bool(args.reject_high_error),
        "allow_interference": bool(args.allow_interference),
        "contact_tolerance": float(args.contact_tolerance),
        "common_volume_tolerance": float(args.common_volume_tolerance),
        "search_free_rotation": not args.no_free_rotation_search,
        "rotation_samples": int(args.rotation_samples),
        "score_threshold": float(args.score_threshold),
        "max_predictions": int(args.max_predictions),
        "beam_size": int(args.beam_size),
        "min_constraints": int(args.min_constraints),
        "max_constraints": int(args.max_constraints),
        "prefer_mixed_types": bool(args.prefer_mixed_types),
        "allow_direction_only": bool(args.allow_direction_only),
        "export_rejected": bool(args.export_rejected),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
    }
    tasks = [
        {
            "batch_index": batch_index,
            "prediction": item,
            "context": context,
        }
        for batch_index, item in enumerate(predictions, start=1 + int(args.offset))
    ]

    workers = max(1, int(args.workers or 1))
    if workers == 1:
        results = []
        for task in tasks:
            result = _run_prediction_task(task)
            results.append(result)
            if args.stop_on_error and result.get("status") == "error":
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_prediction_task, tasks, chunksize=1))
        if args.stop_on_error:
            first_error = next((idx for idx, item in enumerate(results) if item.get("status") == "error"), None)
            if first_error is not None:
                results = results[: first_error + 1]

    ok_samples = _ok_samples_by_split(results)
    ok_samples_path = output_dir / "ok_prediction_samples_by_split.json"
    ok_samples_path.write_text(json.dumps(ok_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "predictions_json": str(predictions_json),
        "json_root": str(json_root),
        "step_root": str(step_root),
        "output_dir": str(output_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "workers": workers,
        "dry_run": bool(args.dry_run),
        "selection": {
            "score_threshold": float(args.score_threshold),
            "max_predictions": int(args.max_predictions),
            "beam_size": int(args.beam_size),
            "min_constraints": int(args.min_constraints),
            "max_constraints": int(args.max_constraints),
            "prefer_mixed_types": bool(args.prefer_mixed_types),
            "allow_direction_only": bool(args.allow_direction_only),
            "supported_types": sorted(SUPPORTED_PREDICTION_TYPES.values()),
        },
        "artifact_formats": ["step"],
        "sample_count": len(results),
        "success_count": sum(item.get("status") in {"ok", "partial", "rejected_but_exported"} for item in results),
        "ok_count": sum(item.get("status") == "ok" for item in results),
        "dry_run_count": sum(item.get("status") == "dry_run_ok" for item in results),
        "error_count": sum(item.get("status") == "error" for item in results),
        "skipped_count": sum(item.get("status") == "skipped_existing" for item in results),
        "step_count": sum(int(item.get("step_count") or 0) for item in results),
        "ok_samples_json": str(ok_samples_path),
        "results": [_batch_result_json(item) for item in results],
    }
    summary_path = output_dir / "prediction_results.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(summary_path),
                "sample_count": payload["sample_count"],
                "ok_count": payload["ok_count"],
                "success_count": payload["success_count"],
                "dry_run_count": payload["dry_run_count"],
                "error_count": payload["error_count"],
                "skipped_count": payload["skipped_count"],
                "step_count": payload["step_count"],
                "ok_samples": payload["ok_samples_json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if payload["success_count"] == 0 and payload["dry_run_count"] == 0 and payload["step_count"] == 0:
        raise SystemExit(1)


def _run_prediction_task(task: dict[str, Any]) -> dict[str, Any]:
    context = dict(task["context"])
    batch_index = int(task["batch_index"])
    prediction_item = dict(task["prediction"])
    assembly_id = _safe_id(prediction_item.get("assembly_id") or f"prediction_{batch_index}")
    split_name = _split_stem(str(prediction_item.get("split") or "unknown"))
    json_root = Path(context["json_root"])
    step_root = Path(context["step_root"])
    output_dir = Path(context["output_dir"])
    sample_output_dir = output_dir / split_name / assembly_id
    sample_summary = sample_output_dir / f"{assembly_id}.json"

    try:
        sample_json = _resolve_sample_json(f"{assembly_id}.json", json_root)
        if context["skip_existing"] and sample_summary.exists():
            return _compact_json(
                {
                    "batch_index": batch_index,
                    "split": split_name,
                    "assembly_id": assembly_id,
                    "sample_json": str(sample_json),
                    "output_dir": str(sample_output_dir),
                    "status": "skipped_existing",
                    "summary": str(sample_summary),
                }
            )

        result = _process_prediction_sample(
            prediction_item,
            sample_json,
            sample_output_dir,
            sample_summary,
            step_root=step_root,
            face_index_base=int(context["face_index_base"]),
            solver_mode=str(context["solver_mode"]),
            max_error=float(context["max_error"]),
            reject_high_error=bool(context["reject_high_error"]),
            allow_interference=bool(context["allow_interference"]),
            contact_tolerance=float(context["contact_tolerance"]),
            common_volume_tolerance=float(context["common_volume_tolerance"]),
            search_free_rotation=bool(context["search_free_rotation"]),
            rotation_samples=int(context["rotation_samples"]),
            score_threshold=float(context["score_threshold"]),
            max_predictions=int(context["max_predictions"]),
            beam_size=int(context["beam_size"]),
            min_constraints=int(context["min_constraints"]),
            max_constraints=int(context["max_constraints"]),
            prefer_mixed_types=bool(context["prefer_mixed_types"]),
            allow_direction_only=bool(context["allow_direction_only"]),
            export_rejected=bool(context["export_rejected"]),
            dry_run=bool(context["dry_run"]),
        )
        result.update(
            {
                "batch_index": batch_index,
                "split": split_name,
                "assembly_id": assembly_id,
                "summary": str(sample_summary),
            }
        )
    except Exception as exc:
        result = {
            "batch_index": batch_index,
            "split": split_name,
            "assembly_id": assembly_id,
            "sample_json": str((json_root / f"{assembly_id}.json").resolve()),
            "output_dir": str(sample_output_dir),
            "status": "error",
            "summary": str(sample_summary),
            "error": f"{type(exc).__name__}: {exc}",
        }
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        sample_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return _compact_json(result)


def _process_prediction_sample(
    prediction_item: dict[str, Any],
    sample_json: Path,
    sample_output_dir: Path,
    sample_summary: Path,
    *,
    step_root: Path,
    face_index_base: int,
    solver_mode: str,
    max_error: float,
    reject_high_error: bool,
    allow_interference: bool,
    contact_tolerance: float,
    common_volume_tolerance: float,
    search_free_rotation: bool,
    rotation_samples: int,
    score_threshold: float,
    max_predictions: int,
    beam_size: int,
    min_constraints: int,
    max_constraints: int,
    prefer_mixed_types: bool,
    allow_direction_only: bool,
    export_rejected: bool,
    dry_run: bool,
) -> dict[str, Any]:
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    raw_payload = _read_json(sample_json)
    base_payload = _convert_constraint_payload(raw_payload, sample_json, step_root, step_index=None)
    predictions = _prediction_constraints(prediction_item.get("predictions") or [])
    selected_pool, rejected_predictions = _filter_prediction_pool(
        predictions,
        score_threshold=score_threshold,
        max_predictions=max_predictions,
    )
    candidate_sets = _candidate_prediction_sets(
        selected_pool,
        min_constraints=min_constraints,
        max_constraints=max_constraints,
        beam_size=beam_size,
        prefer_mixed_types=prefer_mixed_types,
    )

    if dry_run:
        result = {
            "sample_json": str(sample_json),
            "output_dir": str(sample_output_dir),
            "status": "dry_run_ok",
            "part_count": len(base_payload.get("parts") or []),
            "candidate_pool_count": len(selected_pool),
            "candidate_set_count": len(candidate_sets),
            "prediction_selection": {
                "pool": [_prediction_json(item) for item in selected_pool],
                "rejected": rejected_predictions,
                "candidate_sets": [_candidate_set_json(index, candidates) for index, candidates in enumerate(candidate_sets, start=1)],
            },
        }
        sample_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    if not candidate_sets:
        result = {
            "sample_json": str(sample_json),
            "output_dir": str(sample_output_dir),
            "status": "error",
            "part_count": len(base_payload.get("parts") or []),
            "pair_count": 0,
            "step_count": 0,
            "prediction_selection": {
                "pool": [_prediction_json(item) for item in selected_pool],
                "rejected": rejected_predictions,
            },
            "error": "No usable prediction constraints remained after filtering.",
        }
        sample_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    attempts: list[dict[str, Any]] = []
    best_attempt: dict[str, Any] | None = None
    best_records = None
    best_jobs = None
    best_assembly = None
    best_policy_attempt: dict[str, Any] | None = None
    best_policy_records = None
    best_policy_jobs = None
    best_policy_assembly = None
    pool_has_positional = any(_is_positional_type(prediction.constraint_type) for prediction in selected_pool)

    for candidate_index, candidate_predictions in enumerate(candidate_sets, start=1):
        attempt = _solve_prediction_candidate(
            candidate_index,
            candidate_predictions,
            base_payload,
            sample_json,
            step_root=step_root,
            face_index_base=face_index_base,
            solver_mode=solver_mode,
            max_error=max_error,
            reject_high_error=reject_high_error,
            allow_interference=allow_interference,
            contact_tolerance=contact_tolerance,
            common_volume_tolerance=common_volume_tolerance,
            search_free_rotation=search_free_rotation,
            rotation_samples=rotation_samples,
        )
        attempts.append(attempt["summary"])
        if best_attempt is None or _attempt_score(attempt["summary"]) < _attempt_score(best_attempt):
            best_attempt = attempt["summary"]
            best_records = attempt.get("records")
            best_jobs = attempt.get("jobs")
            best_assembly = attempt.get("assembly")

        policy_ok = _attempt_satisfies_selection_policy(
            attempt["summary"],
            pool_has_positional=pool_has_positional,
            allow_direction_only=allow_direction_only,
        )
        if policy_ok and (
            best_policy_attempt is None
            or _attempt_score(attempt["summary"]) < _attempt_score(best_policy_attempt)
        ):
            best_policy_attempt = attempt["summary"]
            best_policy_records = attempt.get("records")
            best_policy_jobs = attempt.get("jobs")
            best_policy_assembly = attempt.get("assembly")

        if _candidate_status_is_final(attempt["summary"].get("status")) and policy_ok:
            best_attempt = attempt["summary"]
            best_records = attempt.get("records")
            best_jobs = attempt.get("jobs")
            best_assembly = attempt.get("assembly")
            break

    if best_attempt is None:
        raise RuntimeError("Prediction candidate search did not produce any attempts.")

    selection_policy_satisfied = best_policy_attempt is not None
    if selection_policy_satisfied:
        best_attempt = best_policy_attempt
        best_records = best_policy_records
        best_jobs = best_policy_jobs
        best_assembly = best_policy_assembly
    else:
        best_records = None
        best_jobs = None
        best_assembly = None

    predictions_by_index = {prediction.index: prediction for prediction in selected_pool}
    selected_predictions = [
        predictions_by_index[int(item["prediction_index"])]
        for item in best_attempt.get("selected_prediction_refs") or []
        if int(item.get("prediction_index") or 0) in predictions_by_index
    ]
    results = []
    if best_records and best_jobs and best_assembly:
        jobs_by_index = {job.index: job for job in best_jobs}
        for record in best_records:
            matrix, matrix_source = _record_matrix(record)
            should_export = matrix is not None and (record.status == "ok" or export_rejected)
            artifacts = (
                save_solution_step(
                    best_assembly,
                    record,
                    sample_output_dir,
                    transform=matrix,
                    use_pair_subdir=False,
                )
                if should_export
                else {}
            )
            item = _record_json(
                record,
                jobs_by_index[record.index],
                best_assembly,
                matrix=matrix,
                matrix_source=matrix_source,
                step_path=artifacts.get("step"),
            )
            results.append(item)

    status = _sample_status(results) if results else "error"
    if status != "ok" and any(item.get("step") for item in results):
        status = "rejected_but_exported"

    result = {
        "sample_json": str(sample_json),
        "output_dir": str(sample_output_dir),
        "status": status,
        "part_count": len(base_payload.get("parts") or []),
        "pair_count": len(results),
        "step_count": sum(1 for item in results if item.get("step")),
        "prediction_selection": {
            "strategy": "beam_search_with_solver_validation",
            "score_threshold": score_threshold,
            "max_predictions": max_predictions,
            "beam_size": beam_size,
            "min_constraints": min_constraints,
            "max_constraints": max_constraints,
            "allow_direction_only": allow_direction_only,
            "pool_has_positional": pool_has_positional,
            "selection_policy_satisfied": selection_policy_satisfied,
            "candidate_pool_count": len(selected_pool),
            "candidate_set_count": len(candidate_sets),
            "pool": [_prediction_json(item) for item in selected_pool],
            "rejected": rejected_predictions,
            "selected_attempt": best_attempt,
            "selected_predictions": [_prediction_json(item) for item in selected_predictions],
            "attempts": attempts,
        },
        "results": results,
    }
    if not results:
        result["error"] = best_attempt.get("error") or _selection_failure_reason(
            best_attempt,
            pool_has_positional=pool_has_positional,
            allow_direction_only=allow_direction_only,
        )
    sample_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _solve_prediction_candidate(
    candidate_index: int,
    candidate_predictions: Sequence[PredictionConstraint],
    base_payload: dict[str, Any],
    sample_json: Path,
    *,
    step_root: Path,
    face_index_base: int,
    solver_mode: str,
    max_error: float,
    reject_high_error: bool,
    allow_interference: bool,
    contact_tolerance: float,
    common_volume_tolerance: float,
    search_free_rotation: bool,
    rotation_samples: int,
) -> dict[str, Any]:
    candidate_payload = dict(base_payload)
    candidate_payload["internal_constraints"] = [
        _prediction_to_constraint(prediction)
        for prediction in candidate_predictions
    ]
    candidate_payload["prediction_source"] = {
        "sample_json": str(sample_json),
        "candidate_index": candidate_index,
        "predictions": [_prediction_json(prediction) for prediction in candidate_predictions],
    }

    summary = _candidate_set_json(candidate_index, candidate_predictions)
    try:
        assembly = load_assembly_payload(
            candidate_payload,
            sample_json,
            step_dir=step_root,
            face_index_base=face_index_base,
        )
        jobs = collect_pair_jobs(assembly)
        records = solve_jobs(
            assembly,
            jobs,
            solver_mode=solver_mode,
            max_error=max_error,
            reject_high_error=reject_high_error,
            avoid_interference=not allow_interference,
            fail_on_interference=not allow_interference,
            contact_tolerance=contact_tolerance,
            common_volume_tolerance=common_volume_tolerance,
            search_free_rotation=search_free_rotation,
            rotation_sample_count=rotation_samples,
        )
        statuses = [record.status for record in records]
        summary.update(
            {
                "status": _records_status(statuses),
                "pair_count": len(records),
                "record_statuses": statuses,
                "max_constraint_error": max(
                    (
                        float(record.max_constraint_error)
                        for record in records
                        if record.max_constraint_error is not None
                    ),
                    default=None,
                ),
                "step_exportable": any(record.transform is not None or record.rejected_transform is not None for record in records),
            }
        )
        errors = [record.error for record in records if record.error]
        if errors:
            summary["error"] = " | ".join(str(error) for error in errors[:3])
        return {
            "summary": _compact_json(summary),
            "assembly": assembly,
            "jobs": jobs,
            "records": records,
        }
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return {"summary": _compact_json(summary)}


def _read_prediction_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        for key in ("results", "items", "samples", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        raise ValueError(f"Prediction JSON object does not contain a list payload: {path}")
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise TypeError(f"Unsupported prediction JSON top-level type: {type(payload).__name__}")


def _select_prediction_items(
    payload: list[dict[str, Any]],
    split: str,
    assembly_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    requested_splits = {item.strip() for item in str(split or "all").split(",") if item.strip()}
    requested_ids = {str(item).strip() for item in assembly_ids or [] if str(item).strip()}
    results = []
    for item in payload:
        item_split = str(item.get("split") or "unknown")
        item_id = str(item.get("assembly_id") or item.get("id") or "").strip()
        if requested_splits and requested_splits != {"all"} and item_split not in requested_splits:
            continue
        if requested_ids and item_id not in requested_ids:
            continue
        results.append(item)
    return results


def _prediction_constraints(raw_predictions: Iterable[dict[str, Any]]) -> list[PredictionConstraint]:
    constraints: list[PredictionConstraint] = []
    for index, item in enumerate(raw_predictions, start=1):
        normalized_type = _prediction_type(item.get("type") or item.get("type_label"))
        part_a = str(item.get("part_a") or item.get("part_0") or "").strip()
        part_b = str(item.get("part_b") or item.get("part_1") or "").strip()
        if not normalized_type or not part_a or not part_b:
            continue
        try:
            face_a = int(item.get("face_a"))
            face_b = int(item.get("face_b"))
            score = float(item.get("score", 0.0) or 0.0)
            direction = int(item.get("direction", item.get("alignment_label", 0)) or 0)
        except (TypeError, ValueError):
            continue
        constraints.append(
            PredictionConstraint(
                index=index,
                raw=dict(item),
                name=f"Pred{index}_{normalized_type}_{face_a}_{face_b}",
                constraint_type=normalized_type,
                score=score,
                part_a=part_a,
                face_a=face_a,
                part_b=part_b,
                face_b=face_b,
                direction=direction,
            )
        )
    return constraints


def _prediction_type(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in SUPPORTED_PREDICTION_TYPES:
        return SUPPORTED_PREDICTION_TYPES[text]
    label_map = {
        "0": "Coincident",
        "1": "Concentric",
        "2": None,
        "3": "Parallel",
        "4": "Perpendicular",
    }
    return label_map.get(text)


def _filter_prediction_pool(
    predictions: Sequence[PredictionConstraint],
    *,
    score_threshold: float,
    max_predictions: int,
) -> tuple[list[PredictionConstraint], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    deduped: dict[tuple[str, int, str, int, str], PredictionConstraint] = {}
    for prediction in predictions:
        if prediction.part_a == prediction.part_b:
            rejected.append({**_prediction_json(prediction), "reason": "same_part"})
            continue
        if prediction.constraint_type not in TYPE_PRIORITY:
            rejected.append({**_prediction_json(prediction), "reason": "unsupported_type"})
            continue
        existing = deduped.get(prediction.pair_key)
        if existing is None or prediction.score > existing.score:
            if existing is not None:
                rejected.append({**_prediction_json(existing), "reason": "duplicate_lower_score"})
            deduped[prediction.pair_key] = prediction
        else:
            rejected.append({**_prediction_json(prediction), "reason": "duplicate_lower_score"})

    scored = sorted(deduped.values(), key=_prediction_sort_key)
    thresholded = [prediction for prediction in scored if prediction.score >= score_threshold]
    if not thresholded and scored:
        thresholded = scored[: min(max_predictions, len(scored))]
        threshold_fallback = {prediction.index for prediction in thresholded}
    else:
        threshold_fallback = set()

    selected = thresholded[: max(1, int(max_predictions))]
    selected_ids = {prediction.index for prediction in selected}
    for prediction in scored:
        if prediction.index in selected_ids:
            continue
        reason = "below_score_threshold" if prediction.score < score_threshold else "outside_max_predictions"
        if prediction.index in threshold_fallback:
            reason = "threshold_fallback"
        rejected.append({**_prediction_json(prediction), "reason": reason})
    return selected, rejected


def _candidate_prediction_sets(
    pool: Sequence[PredictionConstraint],
    *,
    min_constraints: int,
    max_constraints: int,
    beam_size: int,
    prefer_mixed_types: bool,
) -> list[list[PredictionConstraint]]:
    if not pool:
        return []
    min_size = max(1, int(min_constraints))
    max_size = min(max(min_size, int(max_constraints)), len(pool))
    target_sets: list[tuple[float, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(indices: Sequence[int], bonus: float = 0.0) -> None:
        key = tuple(sorted(set(int(index) for index in indices)))
        if len(key) < min_size or len(key) > max_size or key in seen:
            return
        seen.add(key)
        constraints = [pool[index] for index in key]
        target_sets.append((_subset_rank(constraints, prefer_mixed_types=prefer_mixed_types) + bonus, key))

    for size in range(max_size, min_size - 1, -1):
        add(range(size), bonus=0.0)

    for index in range(len(pool)):
        for size in range(min_size, max_size + 1):
            add(range(index, min(index + size, len(pool))), bonus=0.1)

    for first in range(len(pool)):
        for second in range(first + 1, len(pool)):
            add((first, second), bonus=0.2)
            if max_size >= 3:
                for third in range(second + 1, len(pool)):
                    add((first, second, third), bonus=0.25)

    by_type: dict[str, list[int]] = {}
    for index, prediction in enumerate(pool):
        by_type.setdefault(prediction.constraint_type, []).append(index)
    positional = by_type.get("Coincident", []) + by_type.get("Concentric", [])
    directional = by_type.get("Parallel", []) + by_type.get("Perpendicular", [])
    for first in positional[:4]:
        for second in directional[:4]:
            add((first, second), bonus=-0.2)
            if max_size >= 3:
                for third in range(len(pool)):
                    if third not in {first, second}:
                        add((first, second, third), bonus=-0.15)
                        break

    target_sets.sort(key=lambda item: (item[0], item[1]))
    return [[pool[index] for index in indices] for _, indices in target_sets[: max(1, int(beam_size))]]


def _prediction_to_constraint(prediction: PredictionConstraint) -> dict[str, Any]:
    return {
        "constraint_name": prediction.name,
        "constraint_type": prediction.constraint_type,
        "source_constraint_type": prediction.constraint_type,
        "prediction_score": float(prediction.score),
        "prediction_index": int(prediction.index),
        "params": {
            "alignment": int(prediction.direction),
            "orientation": _prediction_orientation(prediction),
            "raw_value": 0.0,
            "dimensions": [],
            "mate_entity_count": 2,
        },
        "elements": [
            {
                "part_name": prediction.part_a,
                "part_id": prediction.part_a,
                "matched_face_idx": [int(prediction.face_a)],
                "verified_face_idx": [int(prediction.face_a)],
            },
            {
                "part_name": prediction.part_b,
                "part_id": prediction.part_b,
                "matched_face_idx": [int(prediction.face_b)],
                "verified_face_idx": [int(prediction.face_b)],
            },
        ],
        "verified_face_pairs": [
            {
                prediction.part_a: int(prediction.face_a),
                prediction.part_b: int(prediction.face_b),
            }
        ],
    }


def _prediction_orientation(prediction: PredictionConstraint) -> int:
    if prediction.constraint_type == "Coincident":
        return 1 if int(prediction.direction) == 0 else 2
    return 0


def _prediction_json(prediction: PredictionConstraint) -> dict[str, Any]:
    return {
        "index": int(prediction.index),
        "name": prediction.name,
        "type": prediction.constraint_type,
        "score": float(prediction.score),
        "direction": int(prediction.direction),
        "part_a": prediction.part_a,
        "face_a": int(prediction.face_a),
        "part_b": prediction.part_b,
        "face_b": int(prediction.face_b),
    }


def _candidate_set_json(index: int, candidates: Sequence[PredictionConstraint]) -> dict[str, Any]:
    scores = [float(candidate.score) for candidate in candidates]
    return {
        "candidate_index": int(index),
        "constraint_count": len(candidates),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "types": [candidate.constraint_type for candidate in candidates],
        "selected_prediction_refs": [
            {
                "rank": position,
                "prediction_index": candidate.index,
                "name": candidate.name,
                "type": candidate.constraint_type,
                "score": float(candidate.score),
            }
            for position, candidate in enumerate(candidates, start=1)
        ],
    }


def _prediction_sort_key(prediction: PredictionConstraint) -> tuple[float, int, int]:
    return (-float(prediction.score), TYPE_PRIORITY.get(prediction.constraint_type, 99), int(prediction.index))


def _subset_rank(
    constraints: Sequence[PredictionConstraint],
    *,
    prefer_mixed_types: bool,
) -> float:
    scores = [float(constraint.score) for constraint in constraints]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    type_set = {constraint.constraint_type for constraint in constraints}
    positional_count = sum(1 for constraint in constraints if constraint.constraint_type in {"Coincident", "Concentric"})
    directional_count = len(constraints) - positional_count
    duplicate_faces = len(constraints) - len({constraint.face_pair_key for constraint in constraints})
    rank = -mean_score
    rank += 0.02 * len(constraints)
    rank += 0.5 * duplicate_faces
    if prefer_mixed_types and positional_count and directional_count:
        rank -= 0.08
    if "Coincident" in type_set or "Concentric" in type_set:
        rank -= 0.04
    return rank


def _records_status(statuses: Sequence[str]) -> str:
    if not statuses:
        return "error"
    if all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "ok" for status in statuses):
        return "partial"
    if any(status == "collision" for status in statuses):
        return "collision"
    return "error"


def _candidate_status_is_final(status: Any) -> bool:
    return str(status or "") in {"ok", "partial"}


def _candidate_status_has_transform(status: Any) -> bool:
    return str(status or "") in {"ok", "partial", "collision"}


def _attempt_satisfies_selection_policy(
    summary: dict[str, Any],
    *,
    pool_has_positional: bool,
    allow_direction_only: bool,
) -> bool:
    if not _candidate_status_has_transform(summary.get("status")):
        return False
    if allow_direction_only or not pool_has_positional:
        return True
    return any(_is_positional_type(constraint_type) for constraint_type in summary.get("types") or [])


def _is_positional_type(constraint_type: str) -> bool:
    return str(constraint_type) in {"Coincident", "Concentric"}


def _selection_failure_reason(
    summary: dict[str, Any],
    *,
    pool_has_positional: bool,
    allow_direction_only: bool,
) -> str:
    if (
        pool_has_positional
        and not allow_direction_only
        and _candidate_status_has_transform(summary.get("status"))
        and not any(_is_positional_type(constraint_type) for constraint_type in summary.get("types") or [])
    ):
        return (
            "Only direction-only prediction subsets produced a transform. "
            "Pass --allow-direction-only to accept Parallel/Perpendicular-only results."
        )
    return "No prediction candidate produced a solved transform."


def _attempt_score(summary: dict[str, Any]) -> tuple[int, float, float, int]:
    status_rank = {
        "ok": 0,
        "partial": 1,
        "collision": 2,
        "error": 3,
    }.get(str(summary.get("status") or ""), 4)
    max_error = summary.get("max_constraint_error")
    error_value = float(max_error) if max_error is not None else 1e9
    mean_score = float(summary.get("mean_score") or 0.0)
    count = int(summary.get("constraint_count") or 0)
    return (status_rank, error_value, -mean_score, -count)


def _ok_samples_by_split(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in results:
        split = _split_stem(str(item.get("split") or "unknown"))
        grouped.setdefault(split, [])
        if item.get("status") == "ok":
            grouped[split].append(f"{item.get('assembly_id')}.json")
    return grouped


def _safe_id(value: Any) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "sample")).strip("._")
    return safe or "sample"


if __name__ == "__main__":
    main()
