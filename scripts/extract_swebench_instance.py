#!/usr/bin/env python3
"""Extract one or more SWE-bench Lite rows into a local JSON dataset file.

The official SWE-bench harness accepts a JSON file as ``--dataset-name``.  This
helper keeps case-study runs reproducible without relying on network access to
download the dataset again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pyarrow.ipc as ipc

REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ARROW_GLOBS = (
    Path.home()
    / ".cache/huggingface/datasets/princeton-nlp___swe-bench_lite/**/swe-bench_lite-{split}.arrow",
    REPO_ROOT / "data/swe-bench/**/swe-bench_lite-{split}.arrow",
    REPO_ROOT / "data/hf_cache/datasets/**/swe-bench_lite-{split}.arrow",
)


def _expand_instance_ids(raw_ids: Iterable[str]) -> List[str]:
    instance_ids: List[str] = []
    for raw in raw_ids:
        for item in raw.split(","):
            item = item.strip()
            if item:
                instance_ids.append(item)
    return instance_ids


def _find_arrow_file(split: str) -> Path:
    matches: List[Path] = []
    for pattern in DEFAULT_ARROW_GLOBS:
        matches.extend(Path("/").glob(str(pattern).lstrip("/").format(split=split)))
    if not matches:
        searched = "\n".join(str(p).format(split=split) for p in DEFAULT_ARROW_GLOBS)
        raise FileNotFoundError(f"Could not find SWE-bench Lite arrow file. Searched:\n{searched}")
    return sorted(matches)[0]


def _load_rows(arrow_file: Path) -> List[dict]:
    with arrow_file.open("rb") as fh:
        return ipc.open_stream(fh).read_all().to_pylist()


def extract_instances(arrow_file: Path, instance_ids: List[str]) -> List[dict]:
    rows = _load_rows(arrow_file)
    row_by_id = {row["instance_id"]: row for row in rows}
    missing = [instance_id for instance_id in instance_ids if instance_id not in row_by_id]
    if missing:
        available_hint = ", ".join(sorted(row_by_id)[:10])
        raise KeyError(f"Missing instance_id(s): {missing}. First available ids: {available_hint}")
    return [row_by_id[instance_id] for instance_id in instance_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", action="append", required=True, help="Instance id; repeat or comma-separate.")
    parser.add_argument("--output", required=True, help="Output JSON dataset path.")
    parser.add_argument("--split", default="test", help="SWE-bench Lite split name.")
    parser.add_argument("--arrow-file", help="Explicit local swe-bench_lite-<split>.arrow file.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact output.")
    args = parser.parse_args()

    instance_ids = _expand_instance_ids(args.instance_id)
    arrow_file = Path(args.arrow_file) if args.arrow_file else _find_arrow_file(args.split)
    rows = extract_instances(arrow_file, instance_ids)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=None if args.indent == 0 else args.indent),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} SWE-bench Lite row(s) from {arrow_file} to {output}")


if __name__ == "__main__":
    main()
