from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import core
except ImportError:  # pragma: no cover - direct script execution fallback
    import core


@dataclass(frozen=True)
class AssemblyInput:
    assembly_json: Path
    payload: dict
    step_dir: Optional[Path]
    part_paths: dict[str, Path]
    constraints: list[core.PairConstraint]
    part_order: list[str]


@dataclass(frozen=True)
class PairJob:
    index: int
    fixed_part: str
    moving_part: str
    constraints: list[core.PairConstraint]


def load_assembly(
    assembly_json: str | Path,
    *,
    step_dir: str | Path | None = None,
    face_index_base: int = 0,
) -> AssemblyInput:
    json_path = Path(assembly_json).resolve()
    payload = core.load_json(str(json_path))
    return load_assembly_payload(
        payload,
        json_path,
        step_dir=step_dir,
        face_index_base=face_index_base,
    )


def load_assembly_payload(
    payload: dict,
    assembly_json: str | Path,
    *,
    step_dir: str | Path | None = None,
    face_index_base: int = 0,
) -> AssemblyInput:
    json_path = Path(assembly_json).resolve()
    resolved_step_dir = _resolve_step_dir(payload, json_path, step_dir)
    part_paths = core.build_parts(
        payload,
        step_dir=str(resolved_step_dir) if resolved_step_dir else None,
        base_dir=str(json_path.parent),
    )
    resolved_paths = {name: Path(path).resolve() for name, path in part_paths.items()}
    missing = [f"{name}: {path}" for name, path in resolved_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing STEP files:\n" + "\n".join(missing))

    constraints = core.extract_constraints(payload, face_index_base=face_index_base)
    part_order = [
        str(name)
        for part in payload.get("parts") or []
        for name in [part.get("part_name") or part.get("part_id")]
        if name
    ]
    return AssemblyInput(json_path, payload, resolved_step_dir, resolved_paths, constraints, part_order)


def collect_pair_jobs(
    assembly: AssemblyInput,
    *,
    fixed_part: str | None = None,
    moving_part: str | None = None,
) -> list[PairJob]:
    if fixed_part or moving_part:
        fixed, moving, constraints = core.select_pair_constraints(
            assembly.constraints,
            part_order=assembly.part_order,
            fixed_part=fixed_part,
            moving_part=moving_part,
        )
        return [PairJob(1, fixed, moving, constraints)]

    rank = {name: index for index, name in enumerate(assembly.part_order)}
    by_pair: dict[frozenset[str], list[core.PairConstraint]] = defaultdict(list)
    for constraint in assembly.constraints:
        left = constraint.refs[0].part_name
        right = constraint.refs[1].part_name
        if left and right and left != right:
            by_pair[frozenset((left, right))].append(constraint)

    def ordered(pair_key: frozenset[str]) -> tuple[str, str]:
        parts = sorted(pair_key, key=lambda name: (rank.get(name, 10**9), name))
        return parts[0], parts[1]

    jobs: list[PairJob] = []
    for pair_key in sorted(by_pair, key=lambda item: (rank.get(ordered(item)[0], 10**9), rank.get(ordered(item)[1], 10**9), ordered(item))):
        fixed, moving = ordered(pair_key)
        jobs.append(PairJob(len(jobs) + 1, fixed, moving, by_pair[pair_key]))
    return jobs


def _resolve_step_dir(payload: dict, json_path: Path, step_dir: str | Path | None) -> Optional[Path]:
    if step_dir:
        path = Path(step_dir).resolve()
        if not path.is_dir():
            raise NotADirectoryError(path)
        return path

    required = {
        Path(str(part.get("step_file") or part.get("step_path") or "")).name.lower()
        for part in payload.get("parts") or []
    }
    required.discard("")
    candidates: list[Path] = []
    for raw in (payload.get("source_step_dir"), json_path.parent / "steps", ROOT / "steps", json_path.parent):
        if raw:
            candidates.append(Path(raw))

    for candidate in candidates:
        if candidate.is_dir() and _contains_steps(candidate, required):
            return candidate.resolve()
    return None


def _contains_steps(directory: Path, required_names: set[str]) -> bool:
    if not required_names:
        return True
    present = {path.name.lower() for path in directory.iterdir() if path.is_file()}
    return required_names.issubset(present)
