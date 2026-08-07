"""Batch-solve local pair bundles and export inferred and ground-truth STEP files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstructed_solver.input_loader import collect_pair_jobs, load_assembly_payload
from reconstructed_solver.process_local_pair import (
    _adapt_bundle_payload,
    _ground_truth_relative_transform,
    _matrix_json,
    _read_json,
    _transform_difference,
)
from reconstructed_solver.solve import solve_jobs
from reconstructed_solver.summarize_local_pair_results import summarize_results
from reconstructed_solver.visualize import save_solution_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process local pair bundles containing pair.json and co-located STEP files."
    )
    parser.add_argument(
        "--input-root",
        default="/home/xiazhen/cad/AssemblyTry/pipline_model/modules/data/tmp_data",
        help="Root directory searched recursively for pair.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "local_pair_batch_output"),
        help="Root containing one result directory per input bundle.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Worker process count; use 1 for serial processing.")
    parser.add_argument("--limit", type=int, help="Process at most this many bundles after --offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many ordered bundles.")
    parser.add_argument(
        "--sample-ids-file",
        help="Optional text file containing sample IDs to process. One ID per line; non-ID lines are ignored.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples with an existing sample_results.json.")
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
    parser.add_argument("--gt-original-semantics", action="store_true")
    parser.add_argument("--unoriented-concentric", action="store_true")
    parser.add_argument("--contact-tolerance", type=float, default=1e-3)
    parser.add_argument("--common-volume-tolerance", type=float, default=1e-3)
    parser.add_argument("--rotation-samples", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if output_dir == input_root or input_root in output_dir.parents:
        raise ValueError("--output-dir must not be the input root or one of its parents.")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = sorted(path.parent for path in input_root.rglob("pair.json") if output_dir not in path.parents)
    if args.sample_ids_file:
        ids_path = Path(args.sample_ids_file).expanduser().resolve()
        if not ids_path.is_file():
            raise FileNotFoundError(ids_path)
        requested_ids = {
            line.strip()
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith(("#", "[", "-", "=")) and " " not in line.strip()
        }
        bundles_by_id = {str(bundle.relative_to(input_root)): bundle for bundle in bundles}
        missing_ids = sorted(requested_ids - bundles_by_id.keys())
        if missing_ids:
            raise FileNotFoundError(
                f"{len(missing_ids)} requested sample IDs were not found under {input_root}; "
                f"first missing ID: {missing_ids[0]}"
            )
        bundles = [bundles_by_id[sample_id] for sample_id in sorted(requested_ids)]
    else:
        bundles = bundles[int(args.offset) :]
        if args.limit is not None:
            bundles = bundles[: max(0, int(args.limit))]

    context = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "skip_existing": bool(args.skip_existing),
        "face_index_base": int(args.face_index_base),
        "solver_mode": args.solver,
        "max_error": float(args.max_error),
        "reject_high_error": bool(args.reject_high_error),
        "allow_interference": bool(args.allow_interference),
        "contact_tolerance": float(args.contact_tolerance),
        "common_volume_tolerance": float(args.common_volume_tolerance),
        "search_free_rotation": not args.no_free_rotation_search,
        "rotation_sample_count": int(args.rotation_samples),
        "allow_coincident_orientation_flip": not args.gt_original_semantics,
        "allow_tangent_orientation_flip": not args.gt_original_semantics,
        "use_concentric_orientation": not args.unoriented_concentric,
    }
    tasks = [{"bundle_dir": str(bundle), "context": context} for bundle in bundles]
    workers = max(1, int(args.workers))
    if workers == 1:
        results = [_process_bundle(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_bundle, tasks, chunksize=1))

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workers": workers,
        "offset": int(args.offset),
        "limit": args.limit,
        "sample_count": len(results),
        "ok_count": sum(item["status"] == "ok" for item in results),
        "partial_count": sum(item["status"] == "partial" for item in results),
        "error_count": sum(item["status"] == "error" for item in results),
        "skipped_count": sum(item["status"] == "skipped_existing" for item in results),
        "results": results,
    }
    summary_path = output_dir / "batch_results.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    id_summary = summarize_results(output_dir, input_root)
    id_summary_path = output_dir / "result_id_summary.json"
    id_summary_path.write_text(json.dumps(id_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**{key: summary[key] for key in ("sample_count", "ok_count", "partial_count", "error_count", "skipped_count")}, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


def _process_bundle(task: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = Path(task["bundle_dir"])
    context = task["context"]
    input_root = Path(context["input_root"])
    sample_id = str(bundle_dir.relative_to(input_root))
    sample_output = Path(context["output_dir"]) / sample_id
    result_path = sample_output / "sample_results.json"
    if (
        context["skip_existing"]
        and (sample_output / "GT.step").is_file()
        and (sample_output / "inferred.step").is_file()
    ):
        return {"sample_id": sample_id, "status": "skipped_existing", "result": str(result_path)}

    try:
        source_payload = _read_json(bundle_dir / "pair.json")
        assembly = load_assembly_payload(
            _adapt_bundle_payload(source_payload),
            bundle_dir / "pair.json",
            step_dir=bundle_dir,
            face_index_base=int(context["face_index_base"]),
        )
        jobs = collect_pair_jobs(assembly)
        if len(jobs) != 1:
            raise ValueError(f"Expected exactly one constrained pair, found {len(jobs)}.")
        ground_truth = _ground_truth_relative_transform(source_payload)
        if ground_truth is None:
            raise ValueError("Missing fixed/moving transform_matrix values required for GT.step.")

        sample_output.mkdir(parents=True, exist_ok=True)
        records = solve_jobs(
            assembly,
            jobs,
            solver_mode=str(context["solver_mode"]),
            max_error=float(context["max_error"]),
            reject_high_error=bool(context["reject_high_error"]),
            avoid_interference=not bool(context["allow_interference"]),
            fail_on_interference=not bool(context["allow_interference"]),
            contact_tolerance=float(context["contact_tolerance"]),
            common_volume_tolerance=float(context["common_volume_tolerance"]),
            search_free_rotation=bool(context["search_free_rotation"]),
            rotation_sample_count=int(context["rotation_sample_count"]),
            allow_coincident_orientation_flip=bool(context["allow_coincident_orientation_flip"]),
            allow_tangent_orientation_flip=bool(context["allow_tangent_orientation_flip"]),
            use_concentric_orientation=bool(context["use_concentric_orientation"]),
        )
        record = records[0]
        gt_step = _export_step(assembly, record, ground_truth, sample_output, "GT.step")
        inferred_transform = record.transform if record.transform is not None else record.rejected_transform
        inferred_step = (
            _export_step(assembly, record, inferred_transform, sample_output, "inferred.step")
            if inferred_transform is not None
            else None
        )

        status = "ok" if record.status == "ok" and inferred_step else "partial"
        result = {
            "sample_id": sample_id,
            "source_bundle": str(bundle_dir),
            "status": status,
            "ground_truth_relative_moving_to_fixed_4x4": _matrix_json(ground_truth),
            "GT_step": gt_step,
            "inferred_step": inferred_step,
            "record": record.to_json(),
        }
        if inferred_transform is not None:
            result["ground_truth_comparison"] = _transform_difference(ground_truth, inferred_transform)
    except Exception as exc:
        sample_output.mkdir(parents=True, exist_ok=True)
        result = {
            "sample_id": sample_id,
            "source_bundle": str(bundle_dir),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sample_id": sample_id, "status": result["status"], "result": str(result_path), "GT_step": result.get("GT_step"), "inferred_step": result.get("inferred_step"), "error": result.get("error")}


def _export_step(assembly, record, transform, output_dir: Path, name: str) -> str:
    target = output_dir / name
    artifacts = save_solution_step(
        assembly,
        record,
        output_dir,
        transform=transform,
        use_pair_subdir=False,
    )
    temporary = Path(artifacts["step"])
    temporary.replace(target)
    return str(target)


if __name__ == "__main__":
    main()
