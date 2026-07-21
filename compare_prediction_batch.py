from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstructed_solver.batch_run import (
    _compact_json,
    _convert_constraint_payload,
    _read_json,
    _record_json,
    _record_matrix,
    _sample_status,
    _split_stem,
)
from reconstructed_solver.input_loader import collect_pair_jobs
from reconstructed_solver.prediction_run import (
    _attempt_score,
    _attempt_satisfies_selection_policy,
    _candidate_prediction_sets,
    _candidate_status_is_final,
    _filter_prediction_pool,
    _prediction_constraints,
    _prediction_json,
    _selection_failure_reason,
    _solve_prediction_candidate,
)
from reconstructed_solver.visualize import save_solution_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GT batch outputs with prediction-selected BRepNet constraints for ok_test samples."
    )
    parser.add_argument("--predictions-json", default=str(Path(__file__).resolve().parent / "predictions_test.json"))
    parser.add_argument("--ok-json", default=str(Path(__file__).resolve().parent / "batch_output" / "ok_test.json"))
    parser.add_argument("--gt-output-root", default=str(Path(__file__).resolve().parent / "batch_output"))
    parser.add_argument(
        "--json-root",
        default="/home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair",
    )
    parser.add_argument(
        "--step-root",
        default="/home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair/step",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "ok_test_prediction_compare"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-predictions", type=int, default=16)
    parser.add_argument("--beam-size", type=int, default=24)
    parser.add_argument("--min-constraints", type=int, default=1)
    parser.add_argument("--max-constraints", type=int, default=3)
    parser.add_argument(
        "--prefer-mixed-types",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-direction-only", action="store_true")
    parser.add_argument(
        "--export-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_json = Path(args.predictions_json).resolve()
    ok_json = Path(args.ok_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ok_ids = _load_ok_ids(ok_json, args.split)
    predictions_by_id = {
        str(item.get("assembly_id")): item
        for item in json.loads(predictions_json.read_text(encoding="utf-8-sig"))
        if isinstance(item, dict) and item.get("assembly_id")
    }
    tasks = [
        {
            "batch_index": index,
            "assembly_id": assembly_id,
            "prediction_item": predictions_by_id[assembly_id],
        }
        for index, assembly_id in enumerate(ok_ids, start=1)
        if assembly_id in predictions_by_id
    ]
    tasks = tasks[int(args.offset) :]
    if args.limit is not None:
        tasks = tasks[: int(args.limit)]

    context = {
        "split": str(args.split),
        "json_root": str(Path(args.json_root).resolve()),
        "step_root": str(Path(args.step_root).resolve()),
        "gt_output_root": str(Path(args.gt_output_root).resolve()),
        "output_dir": str(output_dir),
        "face_index_base": int(args.face_index_base),
        "solver_mode": str(args.solver),
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
        "skip_existing": bool(args.skip_existing),
    }
    for task in tasks:
        task["context"] = context

    workers = max(1, int(args.workers or 1))
    if workers == 1:
        results = []
        for task in tasks:
            result = _run_compare_task(task)
            results.append(result)
            _print_progress(len(results), len(tasks), result)
            if args.stop_on_error and result.get("status") == "error":
                break
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_run_compare_task, tasks, chunksize=1):
                results.append(result)
                _print_progress(len(results), len(tasks), result)
                if args.stop_on_error and result.get("status") == "error":
                    break

    summary = {
        "predictions_json": str(predictions_json),
        "ok_json": str(ok_json),
        "gt_output_root": str(Path(args.gt_output_root).resolve()),
        "output_dir": str(output_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": str(args.split),
        "offset": int(args.offset),
        "limit": args.limit,
        "workers": workers,
        "selection": {
            "score_threshold": float(args.score_threshold),
            "max_predictions": int(args.max_predictions),
            "beam_size": int(args.beam_size),
            "min_constraints": int(args.min_constraints),
            "max_constraints": int(args.max_constraints),
            "prefer_mixed_types": bool(args.prefer_mixed_types),
            "allow_direction_only": bool(args.allow_direction_only),
        },
        "sample_count": len(results),
        "ok_count": sum(1 for item in results if item.get("status") == "ok"),
        "predict_ok_count": sum(1 for item in results if item.get("predict_status") == "ok"),
        "predict_collision_count": sum(1 for item in results if item.get("predict_status") == "collision"),
        "predict_error_count": sum(1 for item in results if item.get("predict_status") == "error"),
        "error_count": sum(1 for item in results if item.get("status") == "error"),
        "skipped_count": sum(1 for item in results if item.get("status") == "skipped_existing"),
        "results": results,
    }
    summary_path = output_dir / "comparison_results.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(summary_path),
                "sample_count": summary["sample_count"],
                "ok_count": summary["ok_count"],
                "predict_ok_count": summary["predict_ok_count"],
                "predict_collision_count": summary["predict_collision_count"],
                "predict_error_count": summary["predict_error_count"],
                "error_count": summary["error_count"],
                "skipped_count": summary["skipped_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_compare_task(task: dict[str, Any]) -> dict[str, Any]:
    context = dict(task["context"])
    assembly_id = str(task["assembly_id"])
    split = _split_stem(context["split"])
    output_dir = Path(context["output_dir"]) / split / assembly_id
    summary_path = output_dir / f"{assembly_id}.json"
    if context["skip_existing"] and summary_path.exists():
        return _compact_json(
            {
                "batch_index": int(task["batch_index"]),
                "assembly_id": assembly_id,
                "split": split,
                "status": "skipped_existing",
                "output_dir": str(output_dir),
                "summary": str(summary_path),
            }
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return _process_compare_sample(
            batch_index=int(task["batch_index"]),
            assembly_id=assembly_id,
            prediction_item=dict(task["prediction_item"]),
            output_dir=output_dir,
            summary_path=summary_path,
            context=context,
        )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "batch_index": int(task["batch_index"]),
            "assembly_id": assembly_id,
            "split": split,
            "status": "error",
            "output_dir": str(output_dir),
            "summary": str(summary_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def _process_compare_sample(
    *,
    batch_index: int,
    assembly_id: str,
    prediction_item: dict[str, Any],
    output_dir: Path,
    summary_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    split = _split_stem(context["split"])
    gt_dir = Path(context["gt_output_root"]) / split / assembly_id
    gt_summary_path = gt_dir / f"{assembly_id}.json"
    gt_step_source = gt_dir / "assembled.step"
    if not gt_summary_path.is_file():
        raise FileNotFoundError(gt_summary_path)
    if not gt_step_source.is_file():
        raise FileNotFoundError(gt_step_source)

    gt_step = output_dir / "GT.step"
    shutil.copy2(gt_step_source, gt_step)
    gt_summary = json.loads(gt_summary_path.read_text(encoding="utf-8"))
    gt_item = (gt_summary.get("results") or [{}])[0]

    sample_json = Path(context["json_root"]) / f"{assembly_id}.json"
    step_root = Path(context["step_root"])
    raw_payload = _read_json(sample_json)
    base_payload = _convert_constraint_payload(raw_payload, sample_json, step_root, step_index=None)
    predictions = _prediction_constraints(prediction_item.get("predictions") or [])
    selected_pool, rejected_predictions = _filter_prediction_pool(
        predictions,
        score_threshold=float(context["score_threshold"]),
        max_predictions=int(context["max_predictions"]),
    )
    candidate_sets = _candidate_prediction_sets(
        selected_pool,
        min_constraints=int(context["min_constraints"]),
        max_constraints=int(context["max_constraints"]),
        beam_size=int(context["beam_size"]),
        prefer_mixed_types=bool(context["prefer_mixed_types"]),
    )

    attempts: list[dict[str, Any]] = []
    best_attempt: dict[str, Any] | None = None
    best_records = None
    best_jobs = None
    best_assembly = None
    best_policy_attempt: dict[str, Any] | None = None
    best_policy_records = None
    best_policy_jobs = None
    best_policy_assembly = None
    pool_has_positional = any(
        prediction.constraint_type in {"Coincident", "Concentric"}
        for prediction in selected_pool
    )

    for candidate_index, candidate_predictions in enumerate(candidate_sets, start=1):
        attempt = _solve_prediction_candidate(
            candidate_index,
            candidate_predictions,
            base_payload,
            sample_json,
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
            allow_direction_only=bool(context["allow_direction_only"]),
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
        for item in (best_attempt or {}).get("selected_prediction_refs") or []
        if int(item.get("prediction_index") or 0) in predictions_by_index
    ]

    predict_results = []
    if best_records and best_jobs and best_assembly:
        jobs_by_index = {job.index: job for job in best_jobs}
        for record in best_records:
            matrix, matrix_source = _record_matrix(record)
            should_export = matrix is not None and (record.status == "ok" or bool(context["export_rejected"]))
            artifacts = (
                save_solution_step(
                    best_assembly,
                    record,
                    output_dir,
                    transform=matrix,
                    use_pair_subdir=False,
                )
                if should_export
                else {}
            )
            predicted_step = None
            assembled_step = Path(artifacts["step"]) if artifacts.get("step") else None
            if assembled_step and assembled_step.is_file():
                predicted_step_path = output_dir / "predict_brepnet.step"
                if predicted_step_path.exists():
                    predicted_step_path.unlink()
                assembled_step.replace(predicted_step_path)
                predicted_step = str(predicted_step_path)
            item = _record_json(
                record,
                jobs_by_index[record.index],
                best_assembly,
                matrix=matrix,
                matrix_source=matrix_source,
                step_path=predicted_step,
            )
            if record.collision is not None:
                item["collision"] = record.collision
            predict_results.append(item)

    predict_status = _sample_status(predict_results) if predict_results else "error"
    if predict_status != "ok" and any(item.get("step") for item in predict_results):
        predict_status = "rejected_but_exported"

    result = {
        "batch_index": batch_index,
        "assembly_id": assembly_id,
        "split": split,
        "status": "ok" if predict_results else "error",
        "predict_status": predict_status,
        "sample_json": str(sample_json),
        "output_dir": str(output_dir),
        "GT": _compact_json(
            {
                "status": gt_item.get("status"),
                "solver_used": gt_item.get("solver_used"),
                "max_constraint_error": gt_item.get("max_constraint_error"),
                "matrix_source": gt_item.get("matrix_source"),
                "transform_matrix": gt_item.get("transform_matrix"),
                "constraints": gt_item.get("constraints"),
                "step": str(gt_step),
                "source_summary": str(gt_summary_path),
                "source_step": str(gt_step_source),
            }
        ),
        "predict": {
            "status": predict_status,
            "selection": {
                "strategy": "beam_search_with_solver_validation",
                "score_threshold": float(context["score_threshold"]),
                "max_predictions": int(context["max_predictions"]),
                "beam_size": int(context["beam_size"]),
                "min_constraints": int(context["min_constraints"]),
                "max_constraints": int(context["max_constraints"]),
                "allow_direction_only": bool(context["allow_direction_only"]),
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
            "results": predict_results,
        },
        "summary": str(summary_path),
    }
    if not predict_results:
        result["predict"]["error"] = (best_attempt or {}).get("error") or _selection_failure_reason(
            best_attempt or {},
            pool_has_positional=pool_has_positional,
            allow_direction_only=bool(context["allow_direction_only"]),
        )
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return _compact_json(
        {
            "batch_index": batch_index,
            "assembly_id": assembly_id,
            "split": split,
            "status": result["status"],
            "predict_status": predict_status,
            "output_dir": str(output_dir),
            "summary": str(summary_path),
            "GT_step": str(gt_step),
            "predict_step": _first_predict_step(predict_results),
            "error": result.get("predict", {}).get("error"),
        }
    )


def _load_ok_ids(path: Path, split: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get(split) or payload.get(_split_stem(split)) or []
    else:
        values = payload
    return [Path(str(item)).stem for item in values]


def _first_predict_step(results: Sequence[dict[str, Any]]) -> str | None:
    for item in results:
        if item.get("step"):
            return str(item["step"])
    return None


def _print_progress(done: int, total: int, result: dict[str, Any]) -> None:
    if done == total or done == 1 or done % 25 == 0:
        print(
            json.dumps(
                {
                    "done": done,
                    "total": total,
                    "assembly_id": result.get("assembly_id"),
                    "status": result.get("status"),
                    "predict_status": result.get("predict_status"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
