from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstructed_solver.input_loader import collect_pair_jobs, load_assembly_payload
from reconstructed_solver.solve import solve_jobs
from reconstructed_solver.visualize import FIXED_PART_COLOR, MOVING_PART_COLOR, save_solution_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch solve dataset split JSON files and export colored assembled STEP files."
    )
    parser.add_argument("--split-json", default="/home/xiazhen/cad/AssemblyTry/pipline_model/reconstructed_solver/dataset_splits_5type_common_brepnet.json", help="Dataset split JSON, e.g. dataset_splits_5type_common_brepnet.json.")
    parser.add_argument(
        "--split",
        default="all",
        help="Split to process: train, val, test, or all. Multiple splits can be comma-separated.",
    )
    parser.add_argument("--json-root", default="/home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair", help="Directory containing the per-sample constraint JSON files.")
    parser.add_argument("--step-root", default="/home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair/step", help="Directory containing part STEP/STP files.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "batch_output"))
    parser.add_argument(
        "--flat-pair-output",
        action="store_true",
        help="Write assembled.step directly in each sample output directory instead of a numbered pair subdirectory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes used to process samples. Use 1 for sequential execution.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many JSON files after split selection.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many JSON files after split selection.")
    parser.add_argument("--face-index-base", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--solver",
        default="solvespace-then-analytic",
        choices=("solvespace", "analytic", "solvespace-then-analytic"),
        help="Solver mode. Default tries SolveSpace first, then analytic recovery if SolveSpace fails.",
    )
    parser.add_argument("--max-error", type=float, default=1e-4)
    parser.add_argument("--reject-high-error", action="store_true")
    parser.add_argument("--allow-interference", action="store_true")
    parser.add_argument("--no-free-rotation-search", action="store_true")
    parser.add_argument(
        "--gt-original-semantics",
        action="store_true",
        help=(
            "Evaluate Coincident and Tangent constraints with their original GT orientations by disabling "
            "orientation-flip candidates. Does not change Concentric orientation handling."
        ),
    )
    parser.add_argument(
        "--no-coincident-orientation-flip",
        action="store_true",
        help="Disable candidates that flip Coincident orientation.",
    )
    parser.add_argument(
        "--no-tangent-orientation-flip",
        action="store_true",
        help="Disable candidates that flip Tangent orientation.",
    )
    parser.add_argument(
        "--unoriented-concentric",
        action="store_true",
        help="Treat Concentric constraints as unoriented axes by ignoring their alignment/orientation sign.",
    )
    parser.add_argument("--contact-tolerance", type=float, default=1e-3)
    parser.add_argument("--common-volume-tolerance", type=float, default=1e-3)
    parser.add_argument("--rotation-samples", type=int, default=24)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose per-sample result JSON already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the whole batch when one sample fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve JSON files and STEP paths; do not solve or export STEP artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_json = Path(args.split_json).resolve()
    json_root = Path(args.json_root).resolve() if args.json_root else split_json.parent
    step_root = Path(args.step_root).resolve()
    if not step_root.is_dir():
        raise NotADirectoryError(step_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers or 1))
    split_items = _select_split_items(_read_json(split_json), args.split)
    split_items = split_items[int(args.offset) :]
    if args.limit is not None:
        split_items = split_items[: int(args.limit)]

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
        "allow_coincident_orientation_flip": not (args.gt_original_semantics or args.no_coincident_orientation_flip),
        "allow_tangent_orientation_flip": not (args.gt_original_semantics or args.no_tangent_orientation_flip),
        "use_concentric_orientation": not args.unoriented_concentric,
        "gt_original_semantics": bool(args.gt_original_semantics),
        "rotation_samples": int(args.rotation_samples),
        "flat_pair_output": bool(args.flat_pair_output),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
    }
    tasks = [
        {
            "batch_index": batch_index,
            "split": item["split"],
            "entry": item["entry"],
        }
        for batch_index, item in enumerate(split_items, start=1 + int(args.offset))
    ]

    if workers == 1:
        batch_results = []
        for task in tasks:
            result = _run_sample_task({**task, "context": context})
            batch_results.append(result)
            if args.stop_on_error and result.get("status") == "error":
                break
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(context,),
        ) as executor:
            batch_results = list(executor.map(_run_sample_task, tasks, chunksize=1))
        if args.stop_on_error:
            first_error_index = next(
                (index for index, item in enumerate(batch_results) if item.get("status") == "error"),
                None,
            )
            if first_error_index is not None:
                batch_results = batch_results[: first_error_index + 1]

    generated_at = datetime.now().isoformat(timespec="seconds")
    split_names = _task_split_names(tasks)
    ok_samples = _ok_samples_by_split(batch_results, split_names)
    ok_count = sum(len(items) for items in ok_samples.values())
    ok_samples_path = output_dir / "ok_samples_by_split.json"
    ok_samples_path.write_text(json.dumps(ok_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "split_json": str(split_json),
        "json_root": str(json_root),
        "step_root": str(step_root),
        "output_dir": str(output_dir),
        "generated_at": generated_at,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "dry_run": bool(args.dry_run),
        "flat_pair_output": bool(args.flat_pair_output),
        "workers": workers,
        "artifact_formats": ["step"],
        "gt_original_semantics": bool(args.gt_original_semantics),
        "allow_coincident_orientation_flip": context["allow_coincident_orientation_flip"],
        "allow_tangent_orientation_flip": context["allow_tangent_orientation_flip"],
        "use_concentric_orientation": context["use_concentric_orientation"],
        "sample_count": len(batch_results),
        "ok_count": ok_count,
        "ok_samples_json": str(ok_samples_path),
        "success_count": sum(item.get("status") in {"ok", "partial", "rejected_but_exported"} for item in batch_results),
        "dry_run_count": sum(item.get("status") == "dry_run_ok" for item in batch_results),
        "error_count": sum(item.get("status") == "error" for item in batch_results),
        "skipped_count": sum(item.get("status") == "skipped_existing" for item in batch_results),
        "step_count": sum(int(item.get("step_count") or 0) for item in batch_results),
        "results": batch_results,
    }
    summary_path = output_dir / "batch_results.json"
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
                "workers": payload["workers"],
                "ok_samples": payload["ok_samples_json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if payload["success_count"] == 0 and payload["dry_run_count"] == 0 and payload["step_count"] == 0:
        raise SystemExit(1)


_WORKER_CONTEXT: dict[str, Any] = {}


def _init_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = dict(context)


def _run_sample_task(task: dict[str, Any]) -> dict[str, Any]:
    context = task.get("context") or _WORKER_CONTEXT
    if not context:
        raise RuntimeError("Sample worker context was not initialized.")

    batch_index = int(task["batch_index"])
    split_name = _split_stem(str(task["split"]))
    entry = str(task["entry"])
    json_root = Path(context["json_root"])
    step_root = Path(context["step_root"])
    output_dir = Path(context["output_dir"])

    try:
        sample_json = _resolve_sample_json(entry, json_root)
        assembly_id = _sample_stem(sample_json)
        sample_output_dir = output_dir / split_name / assembly_id
        sample_summary = sample_output_dir / f"{assembly_id}.json"
        if context["skip_existing"] and sample_summary.exists():
            return _batch_result_json(
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

        result = _process_sample(
            sample_json,
            sample_output_dir,
            sample_summary,
            step_index=context.get("step_index"),
            step_root=step_root,
            face_index_base=int(context["face_index_base"]),
            solver_mode=str(context["solver_mode"]),
            max_error=float(context["max_error"]),
            reject_high_error=bool(context["reject_high_error"]),
            allow_interference=bool(context["allow_interference"]),
            contact_tolerance=float(context["contact_tolerance"]),
            common_volume_tolerance=float(context["common_volume_tolerance"]),
            search_free_rotation=bool(context["search_free_rotation"]),
            allow_coincident_orientation_flip=bool(context["allow_coincident_orientation_flip"]),
            allow_tangent_orientation_flip=bool(context["allow_tangent_orientation_flip"]),
            use_concentric_orientation=bool(context["use_concentric_orientation"]),
            rotation_samples=int(context["rotation_samples"]),
            flat_pair_output=bool(context["flat_pair_output"]),
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
        fallback_json = (json_root / Path(entry)).resolve()
        assembly_id = _sample_stem(fallback_json)
        sample_output_dir = output_dir / split_name / assembly_id
        sample_summary = sample_output_dir / f"{assembly_id}.json"
        result = {
            "batch_index": batch_index,
            "split": split_name,
            "assembly_id": assembly_id,
            "sample_json": str(fallback_json),
            "output_dir": str(sample_output_dir),
            "status": "error",
            "summary": str(sample_summary),
            "error": f"{type(exc).__name__}: {exc}",
        }
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        sample_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return _batch_result_json(result)


def _process_sample(
    sample_json: Path,
    sample_output_dir: Path,
    sample_summary: Path,
    *,
    step_index: dict[str, Path] | None,
    step_root: Path,
    face_index_base: int,
    solver_mode: str,
    max_error: float,
    reject_high_error: bool,
    allow_interference: bool,
    contact_tolerance: float,
    common_volume_tolerance: float,
    search_free_rotation: bool,
    allow_coincident_orientation_flip: bool,
    allow_tangent_orientation_flip: bool,
    use_concentric_orientation: bool,
    rotation_samples: int,
    flat_pair_output: bool,
    dry_run: bool,
) -> dict[str, Any]:
    raw_payload = _read_json(sample_json)
    assembly_payload = _convert_constraint_payload(raw_payload, sample_json, step_root, step_index)
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_files(sample_output_dir, sample_summary)

    assembly = load_assembly_payload(
        assembly_payload,
        sample_json,
        step_dir=step_root,
        face_index_base=face_index_base,
    )
    jobs = collect_pair_jobs(assembly)
    if flat_pair_output and len(jobs) != 1:
        raise ValueError(
            f"--flat-pair-output requires exactly one pair per sample, but {sample_json} has {len(jobs)} pairs."
        )

    if dry_run:
        result = {
            "sample_json": str(sample_json),
            "output_dir": str(sample_output_dir),
            "status": "dry_run_ok",
            "part_count": len(assembly.part_paths),
            "pair_count": len(jobs),
            "parts": [_part_json(assembly, name, role=None, color=None) for name in _part_names(assembly)],
            "jobs": [_job_json(job) for job in jobs],
        }
        sample_summary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

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
        allow_coincident_orientation_flip=allow_coincident_orientation_flip,
        allow_tangent_orientation_flip=allow_tangent_orientation_flip,
        use_concentric_orientation=use_concentric_orientation,
        rotation_sample_count=rotation_samples,
    )

    results = []
    jobs_by_index = {job.index: job for job in jobs}
    for record in records:
        matrix, matrix_source = _record_matrix(record)
        artifacts = (
            save_solution_step(
                assembly,
                record,
                sample_output_dir,
                transform=matrix,
                use_pair_subdir=not flat_pair_output,
            )
            if matrix is not None
            else {}
        )
        item = _record_json(
            record,
            jobs_by_index[record.index],
            assembly,
            matrix=matrix,
            matrix_source=matrix_source,
            step_path=artifacts.get("step"),
        )
        results.append(item)

    result = {
        "sample_json": str(sample_json),
        "output_dir": str(sample_output_dir),
        "status": _sample_status(results),
        "part_count": len(assembly.part_paths),
        "pair_count": len(records),
        "step_count": sum(1 for item in results if item.get("step")),
        "results": results,
    }
    sample_summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _batch_result_json(result: dict[str, Any]) -> dict[str, Any]:
    steps = _result_step_paths(result)
    return _compact_json(
        {
            "batch_index": result.get("batch_index"),
            "split": result.get("split"),
            "assembly_id": result.get("assembly_id"),
            "sample_json": result.get("sample_json"),
            "output_dir": result.get("output_dir"),
            "summary": result.get("summary"),
            "status": result.get("status"),
            "pair_count": result.get("pair_count"),
            "step_count": result.get("step_count"),
            "step": steps[0] if len(steps) == 1 else None,
            "steps": steps if len(steps) > 1 else None,
            "error": result.get("error"),
        }
    )


def _result_step_paths(result: dict[str, Any]) -> list[str]:
    paths = []
    for item in result.get("results") or []:
        step_path = item.get("step")
        if step_path:
            paths.append(str(step_path))
    return paths


def _task_split_names(tasks: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for task in tasks:
        name = _split_stem(str(task.get("split") or "unknown"))
        if name not in names:
            names.append(name)
    return names


def _ok_samples_by_split(batch_results: list[dict[str, Any]], split_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {split: [] for split in split_names}
    for item in batch_results:
        split = _split_stem(str(item.get("split") or "unknown"))
        grouped.setdefault(split, [])
        if item.get("status") != "ok":
            continue
        grouped[split].append(Path(str(item.get("sample_json"))).name)
    return grouped


def _remove_stale_files(sample_output_dir: Path, sample_summary: Path) -> None:
    for path in (sample_output_dir / "assembly_input.json", sample_output_dir / "sample_results.json"):
        if path.is_file() and path.resolve() != sample_summary.resolve():
            path.unlink()


def _record_matrix(record) -> tuple[list[list[float]] | None, str | None]:
    if record.transform is not None:
        return record.transform, "transform"
    if record.rejected_transform is not None:
        return record.rejected_transform, "rejected_transform"
    return None, None


def _record_json(
    record,
    job,
    assembly,
    *,
    matrix: list[list[float]] | None,
    matrix_source: str | None,
    step_path: str | None,
) -> dict[str, Any]:
    item = {
        "index": record.index,
        "status": record.status,
        "solver_mode": record.solver_mode,
        "solver_used": record.solver_used,
        "fixed_part": _part_json(assembly, record.fixed_part, role="fixed", color=("blue", FIXED_PART_COLOR)),
        "moving_part": _part_json(assembly, record.moving_part, role="moving", color=("yellow", MOVING_PART_COLOR)),
        "constraints": [_constraint_json(constraint) for constraint in job.constraints],
        "transform_matrix": matrix,
        "matrix_source": matrix_source,
        "max_constraint_error": record.max_constraint_error,
        "selected_candidate": record.selected_candidate,
        "selected_variant": getattr(record, "selected_variant", None),
        "primary_error": record.primary_error,
        "collision": record.collision,
        "collision_adjusted": record.collision_adjusted,
        "candidate_results": record.candidate_results,
        "step": step_path,
        "error": record.error,
    }
    return _compact_json(item)


def _job_json(job) -> dict[str, Any]:
    return {
        "index": job.index,
        "fixed_part": job.fixed_part,
        "moving_part": job.moving_part,
        "constraint_count": len(job.constraints),
        "constraints": [_constraint_json(constraint) for constraint in job.constraints],
    }


def _part_names(assembly) -> list[str]:
    return list(assembly.part_order or assembly.part_paths)


def _part_json(assembly, part_name: str, *, role: str | None, color: tuple[str, str] | None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "part_id": part_name,
        "role": role,
        "step_path": str(assembly.part_paths[part_name]),
    }
    if color is not None:
        color_name, color_hex = color
        item["color"] = {"name": color_name, "hex": color_hex}
    return _compact_json(item)


def _constraint_json(constraint) -> dict[str, Any]:
    return _compact_json(
        {
            "name": constraint.name,
            "kind": constraint.kind,
            "source_kind": constraint.source_kind,
            "value": float(constraint.value),
            "orientation": int(constraint.orientation),
            "refs": [
                _compact_json(
                    {
                        "part_id": ref.part_name,
                        "face_index": int(ref.face_index),
                        "face_type": ref.face_type,
                    }
                )
                for ref in constraint.refs
            ],
        }
    )


def _sample_status(results: list[dict[str, Any]]) -> str:
    if not results:
        return "error"
    statuses = [str(item.get("status") or "") for item in results]
    has_step = any(item.get("step") for item in results)
    if all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "ok" for status in statuses):
        return "partial"
    if has_step:
        return "rejected_but_exported"
    return "error"


def _compact_json(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if value is not None and value != "" and value != []}


def _convert_constraint_payload(
    payload: dict[str, Any],
    sample_json: Path,
    step_root: Path,
    step_index: dict[str, Path] | None,
) -> dict[str, Any]:
    payload = _strip_answer_fields(payload)
    part_names = _extract_part_names(payload)
    constraints = []
    for index, constraint in enumerate(payload.get("internal_constraints") or payload.get("constraints") or []):
        converted = dict(constraint)
        converted.setdefault("constraint_name", f"constraint_{index + 1}")
        converted["elements"] = [_convert_constraint_element(element) for element in constraint.get("elements") or []]
        constraints.append(converted)
        for element in converted["elements"]:
            part_name = str(element.get("part_name") or "").strip()
            if part_name and part_name not in part_names:
                part_names.append(part_name)

    if not part_names:
        raise ValueError(f"No parts found in {sample_json}.")

    parts = []
    for part_name in part_names:
        step_path, step_index = _resolve_part_step(part_name, step_root, step_index)
        parts.append(
            {
                "part_name": part_name,
                "part_id": part_name,
                "step_path": str(step_path),
            }
        )

    return {
        "assembly_uuid": payload.get("assembly_uuid") or sample_json.stem,
        "source_json": str(sample_json),
        "source_step_dir": str(step_root),
        "parts": parts,
        "internal_constraints": constraints,
    }


def _strip_answer_fields(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    for key in (
        "transform",
        "solution_transform",
        "rejected_transform",
        "applied_transform",
        "candidate_transform",
        "ground_truth_transform",
    ):
        stripped.pop(key, None)
    return stripped


def _extract_part_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    raw_parts = payload.get("part")
    if raw_parts is None:
        raw_parts = payload.get("parts")
    if isinstance(raw_parts, dict):
        raw_parts = list(raw_parts.values()) if not _part_name(raw_parts) else [raw_parts]
    if isinstance(raw_parts, (str, int)):
        raw_parts = [raw_parts]
    for raw in raw_parts or []:
        name = _part_name(raw)
        if name and name not in names:
            names.append(name)

    for constraint in payload.get("internal_constraints") or payload.get("constraints") or []:
        for element in constraint.get("elements") or []:
            name = _part_name(element)
            if name and name not in names:
                names.append(name)
    return names


def _part_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("part", "part_name", "part_id", "name", "id"):
            raw = value.get(key)
            if raw:
                return str(raw).strip()
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _convert_constraint_element(element: dict[str, Any]) -> dict[str, Any]:
    converted = dict(element)
    part_name = _part_name(element)
    if part_name:
        converted["part_name"] = part_name
        converted.setdefault("part_id", part_name)
    if "matched_face_idx" not in converted and "verified_face_idx" in converted:
        converted["matched_face_idx"] = converted["verified_face_idx"]
    if "matched_face_idx" not in converted and "face" in converted:
        converted["matched_face_idx"] = converted["face"]
    if "matched_face_idx" not in converted and "face_index" in converted:
        converted["matched_face_idx"] = converted["face_index"]
    return converted


def _build_step_index(step_root: Path) -> dict[str, Path]:
    if not step_root.is_dir():
        raise NotADirectoryError(step_root)
    index: dict[str, Path] = {}
    for path in step_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".step", ".stp"}:
            continue
        resolved = path.resolve()
        keys = {path.name.lower(), path.stem.lower()}
        for key in keys:
            index.setdefault(key, resolved)
    return index


def _resolve_part_step(
    part_name: str,
    step_root: Path,
    step_index: dict[str, Path] | None,
) -> tuple[Path, dict[str, Path] | None]:
    raw = str(part_name).strip()
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve(), step_index

    for candidate in _direct_step_candidates(raw, step_root):
        if candidate.is_file():
            return candidate.resolve(), step_index

    if step_index is None:
        step_index = _build_step_index(step_root)

    candidates = [
        raw.lower(),
        Path(raw).name.lower(),
        Path(raw).stem.lower(),
        f"{raw}.step".lower(),
        f"{raw}.stp".lower(),
    ]
    for candidate in candidates:
        if candidate in step_index:
            return step_index[candidate], step_index
    raise FileNotFoundError(f"Failed to locate STEP/STP for part '{part_name}'.")


def _direct_step_candidates(part_name: str, step_root: Path) -> list[Path]:
    raw_path = Path(part_name)
    names = []
    for value in (part_name, raw_path.name):
        text = str(value).strip()
        if text and text not in names:
            names.append(text)

    candidates: list[Path] = []
    for name in names:
        path = Path(name)
        if path.suffix.lower() in {".step", ".stp"}:
            candidates.append(step_root / path)
            continue
        candidates.append(step_root / path)
        candidates.append(step_root / f"{name}.step")
        candidates.append(step_root / f"{name}.stp")
    return candidates


def _select_split_items(split_payload: dict[str, Any], split: str) -> list[dict[str, str]]:
    requested = [item.strip() for item in str(split or "all").split(",") if item.strip()]
    if not requested or requested == ["all"]:
        keys = [key for key, value in split_payload.items() if isinstance(value, list)]
    else:
        keys = requested

    items: list[dict[str, str]] = []
    for key in keys:
        value = split_payload.get(key)
        if not isinstance(value, list):
            raise KeyError(f"Split '{key}' does not exist or is not a list.")
        split_name = _split_stem(str(key))
        items.extend({"split": split_name, "entry": str(item)} for item in value)
    return items


def _split_stem(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "unknown")).strip("._")
    return safe or "unknown"


def _resolve_sample_json(entry: str, json_root: Path) -> Path:
    raw = Path(entry)
    path = raw if raw.is_absolute() else json_root / raw
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sample_stem(path: Path) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", path.stem).strip("._")
    return safe or "sample"


if __name__ == "__main__":
    main()
