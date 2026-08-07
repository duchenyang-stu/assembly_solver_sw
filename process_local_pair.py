"""Solve one co-located pair bundle containing pair.json and its STEP files.

The bundle format keeps both component-to-assembly transforms in pair.json.
The solver operates in the fixed component's frame, so this script derives the
moving-to-fixed ground-truth transform for result comparison without feeding it
back into the solver.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstructed_solver.input_loader import collect_pair_jobs, load_assembly_payload
from reconstructed_solver.solve import SolveRecord, solve_jobs
from reconstructed_solver.visualize import save_solution_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a local pair bundle containing pair.json and co-located STEP files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/home/xiazhen/cad/AssemblyTry/pipline_model/modules/data/tmp_data/000001__cpo-0b6a04ee45869264a576",
        help="Bundle directory containing pair.json and the STEP/STP files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Result directory. Defaults to <input-dir>/solver_output.",
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
    parser.add_argument("--gt-original-semantics", action="store_true")
    parser.add_argument("--unoriented-concentric", action="store_true")
    parser.add_argument("--contact-tolerance", type=float, default=1e-3)
    parser.add_argument("--common-volume-tolerance", type=float, default=1e-3)
    parser.add_argument("--rotation-samples", type=int, default=24)
    parser.add_argument(
        "--no-export-step",
        action="store_true",
        help="Write only the JSON report; do not export an assembled STEP file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the local JSON/STEP bundle and write its summary without solving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    pair_json = input_dir / "pair.json"
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not pair_json.is_file():
        raise FileNotFoundError(f"Expected pair.json in bundle directory: {input_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / "solver_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_payload = _read_json(pair_json)
    assembly_payload = _adapt_bundle_payload(source_payload)
    assembly = load_assembly_payload(
        assembly_payload,
        pair_json,
        step_dir=input_dir,
        face_index_base=args.face_index_base,
    )
    jobs = collect_pair_jobs(assembly)
    ground_truth = _ground_truth_relative_transform(source_payload)

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "pair_json": str(pair_json),
        "output_dir": str(output_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_count": len(assembly.part_paths),
        "parts": [
            {"part_name": name, "step_path": str(path)}
            for name, path in assembly.part_paths.items()
        ],
        "pair_count": len(jobs),
        "ground_truth_relative_moving_to_fixed_4x4": _matrix_json(ground_truth),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        report["jobs"] = [_job_json(job) for job in jobs]
        report["status"] = "dry_run_ok"
    else:
        records = solve_jobs(
            assembly,
            jobs,
            solver_mode=args.solver,
            max_error=args.max_error,
            reject_high_error=args.reject_high_error,
            avoid_interference=not args.allow_interference,
            fail_on_interference=not args.allow_interference,
            contact_tolerance=args.contact_tolerance,
            common_volume_tolerance=args.common_volume_tolerance,
            search_free_rotation=not args.no_free_rotation_search,
            rotation_sample_count=args.rotation_samples,
            allow_coincident_orientation_flip=not args.gt_original_semantics,
            allow_tangent_orientation_flip=not args.gt_original_semantics,
            use_concentric_orientation=not args.unoriented_concentric,
        )
        report["solver"] = args.solver
        report["max_error"] = float(args.max_error)
        report["allow_interference"] = bool(args.allow_interference)
        report["results"] = [
            _record_json(
                assembly,
                record,
                ground_truth=ground_truth,
                output_dir=output_dir,
                export_step=not args.no_export_step,
            )
            for record in records
        ]
        report["success_count"] = sum(record.status == "ok" for record in records)
        report["failure_count"] = len(records) - report["success_count"]
        report["status"] = "ok" if report["success_count"] else "error"

    result_path = output_dir / "local_pair_results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "result": str(result_path),
                "pair_count": report["pair_count"],
                "success_count": report.get("success_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] == "error":
        raise SystemExit(1)


def _ground_truth_relative_transform(payload: dict[str, Any]) -> np.ndarray | None:
    transforms = payload.get("transform_matrix")
    if not isinstance(transforms, dict):
        return None
    fixed = _matrix(transforms.get("fixed_augmented_step_mm_to_root_assembly_mm_4x4"))
    moving = _matrix(transforms.get("moving_augmented_step_mm_to_root_assembly_mm_4x4"))
    if fixed is None or moving is None:
        return None
    return np.linalg.inv(fixed) @ moving


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _adapt_bundle_payload(source: dict[str, Any]) -> dict[str, Any]:
    """Convert the endpoint-based local bundle schema to the solver schema."""
    parts = source.get("parts")
    constraints = source.get("constraints")
    if not isinstance(parts, list) or not isinstance(constraints, list):
        raise ValueError("Local pair bundle requires list-valued 'parts' and 'constraints' fields.")

    converted_constraints = []
    for index, constraint in enumerate(constraints, start=1):
        if not isinstance(constraint, dict):
            raise ValueError(f"Constraint {index} must be a JSON object.")
        type_name = str(constraint.get("type_name") or "").strip()
        face_type = _face_type_for_constraint(type_name)
        endpoints = constraint.get("endpoints")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ValueError(f"Constraint {index} ({type_name or 'unknown'}) must have exactly two endpoints.")
        elements = []
        for endpoint_index, endpoint in enumerate(endpoints, start=1):
            if not isinstance(endpoint, dict):
                raise ValueError(f"Constraint {index} endpoint {endpoint_index} must be a JSON object.")
            part_name = str(endpoint.get("part_name") or "").strip()
            face_indices = endpoint.get("verified_face_indices") or endpoint.get("augmented_face_indices")
            if not part_name or not isinstance(face_indices, list) or not face_indices:
                raise ValueError(
                    f"Constraint {index} endpoint {endpoint_index} requires part_name and verified_face_indices."
                )
            elements.append(
                {
                    "part_name": part_name,
                    "verified_face_idx": int(face_indices[0]),
                    "face_type": face_type,
                }
            )
        params = constraint.get("params") or {}
        converted_constraints.append(
            {
                "constraint_name": str(constraint.get("name") or f"constraint_{index}"),
                "constraint_type": type_name,
                "source_constraint_type": type_name,
                "params": {"alignment": (params.get("alignment") if isinstance(params, dict) else None)},
                "elements": elements,
            }
        )

    return {"parts": parts, "internal_constraints": converted_constraints}


def _face_type_for_constraint(type_name: str) -> str:
    normalized = type_name.strip().lower()
    if normalized == "concentric":
        return "cylinder"
    if normalized in {"coincident", "parallel", "perpendicular", "distance", "angle"}:
        return "plane"
    if normalized == "tangent":
        return ""
    raise ValueError(
        f"Unsupported local-bundle constraint type {type_name!r}; cannot infer its endpoint face type."
    )


def _matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Transform matrix contains a non-finite value.")
    return matrix


def _record_json(
    assembly,
    record: SolveRecord,
    *,
    ground_truth: np.ndarray | None,
    output_dir: Path,
    export_step: bool,
) -> dict[str, Any]:
    item = record.to_json()
    item["fixed_step_path"] = str(assembly.part_paths[record.fixed_part])
    item["moving_step_path"] = str(assembly.part_paths[record.moving_part])
    matrix = record.transform if record.transform is not None else record.rejected_transform
    item["transform_source"] = "transform" if record.transform is not None else "rejected_transform"
    item["transform_matrix"] = _matrix_json(matrix)
    if matrix is not None and ground_truth is not None:
        item["ground_truth_comparison"] = _transform_difference(ground_truth, matrix)
    if export_step and matrix is not None:
        item["artifacts"] = save_solution_step(
            assembly,
            record,
            output_dir,
            transform=matrix,
        )
    return item


def _transform_difference(reference: np.ndarray, candidate: Sequence[Sequence[float]]) -> dict[str, float]:
    actual = _matrix(candidate)
    assert actual is not None
    delta = np.linalg.inv(reference) @ actual
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_error_mm": float(np.linalg.norm(delta[:3, 3])),
        "rotation_error_deg": float(np.degrees(np.arccos(cosine))),
    }


def _matrix_json(matrix: Sequence[Sequence[float]] | np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=float).tolist()


def _job_json(job) -> dict[str, Any]:
    return {
        "index": job.index,
        "fixed_part": job.fixed_part,
        "moving_part": job.moving_part,
        "constraint_names": [constraint.name for constraint in job.constraints],
        "constraint_kinds": [constraint.kind for constraint in job.constraints],
    }


if __name__ == "__main__":
    main()
