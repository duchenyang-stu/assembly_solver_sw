from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstructed_solver.input_loader import collect_pair_jobs, load_assembly
from reconstructed_solver.solve import solve_jobs
from reconstructed_solver.visualize import DEFAULT_VIEWS, save_solution_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve constrained assembly pairs and export assembled STEP plus PNG views.")
    parser.add_argument("--assembly-json", required=True, help="Input assembly JSON.")
    parser.add_argument("--step-dir", help="Directory containing STEP files. Defaults to ./steps when available.")
    parser.add_argument("--fixed-part", help="Solve only this fixed part with --moving-part or best matching pair.")
    parser.add_argument("--moving-part", help="Solve only this moving part with --fixed-part or best matching pair.")
    parser.add_argument("--face-index-base", type=int, default=0, choices=(0, 1))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument(
        "--solver",
        default="solvespace-then-analytic",
        choices=("solvespace", "analytic", "solvespace-then-analytic"),
        help="Solver mode. Default tries SolveSpace first, then analytic recovery if SolveSpace fails.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=1e-4,
        help="Constraint diagnostic tolerance used with --reject-high-error.",
    )
    parser.add_argument(
        "--reject-high-error",
        action="store_true",
        help="In solvespace-then-analytic mode, reject a SolveSpace OK result when max_constraint_error exceeds --max-error.",
    )
    parser.add_argument(
        "--allow-interference",
        action="store_true",
        help="Do not reject transforms whose assembled shapes physically interfere.",
    )
    parser.add_argument(
        "--no-free-rotation-search",
        action="store_true",
        help="When a transform interferes, do not search the unconstrained coaxial rotation for a non-interfering pose.",
    )
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=1e-3,
        help="Distance at or below this value is treated as contact.",
    )
    parser.add_argument(
        "--common-volume-tolerance",
        type=float,
        default=1e-3,
        help="Boolean common volume above this value is treated as physical interference.",
    )
    parser.add_argument(
        "--rotation-samples",
        type=int,
        default=24,
        help="Number of coaxial angle samples used for automatic interference relief.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assembly = load_assembly(
        args.assembly_json,
        step_dir=args.step_dir,
        face_index_base=args.face_index_base,
    )
    jobs = collect_pair_jobs(assembly, fixed_part=args.fixed_part, moving_part=args.moving_part)
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
    )

    result_payload = {
        "assembly_json": str(assembly.assembly_json),
        "step_dir": str(assembly.step_dir) if assembly.step_dir else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "solver_mode": args.solver,
        "max_error": args.max_error,
        "reject_high_error": bool(args.reject_high_error),
        "avoid_interference": not args.allow_interference,
        "allow_interference": bool(args.allow_interference),
        "search_free_rotation": not args.no_free_rotation_search,
        "contact_tolerance": args.contact_tolerance,
        "common_volume_tolerance": args.common_volume_tolerance,
        "rotation_samples": args.rotation_samples,
        "artifact_formats": ["step", "png"],
        "views": list(DEFAULT_VIEWS),
        "pair_count": len(records),
        "success_count": sum(record.status == "ok" for record in records),
        "failure_count": sum(record.status != "ok" for record in records),
        "results": [],
    }

    for record in records:
        item = record.to_json()
        if record.status == "ok":
            item["artifacts"] = save_solution_views(
                assembly,
                record,
                output_dir,
            )
        result_payload["results"].append(item)

    summary_path = output_dir / "solve_results.json"
    summary_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "pair_count": result_payload["pair_count"],
        "success_count": result_payload["success_count"],
        "failure_count": result_payload["failure_count"],
    }, ensure_ascii=False, indent=2))
    if result_payload["success_count"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
