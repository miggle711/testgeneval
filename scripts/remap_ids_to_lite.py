#!/usr/bin/env python3
"""Remap prediction `id` fields from kjain14/testgeneval to kjain14/testgenevallite.

`id` is `<instance_id>-<row index>`, and the row index differs between the two
datasets for the same instance (full dataset: index among 1210 rows; lite:
index among 160). run_evaluation.py matches predictions to dataset tasks via
`tasks_map[prediction["id"]]` (a direct dict lookup, no fallback), so
predictions carrying the full dataset's `id` raise a KeyError against
kjain14/testgenevallite's task map even though their `instance_id` exists in
both datasets.

For each input directory, recursively finds *.jsonl prediction files and
writes a copy with `id` rewritten to kjain14/testgenevallite's `id` for the
same `instance_id`, under a sibling directory (input dir name + suffix).
Originals are left untouched.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def build_lite_id_map() -> dict:
    lite = load_dataset("kjain14/testgenevallite", split="test")
    return {row["instance_id"]: row["id"] for row in lite}


def remap_file(src: Path, dst: Path, id_map: dict) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remapped = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = record["instance_id"]
            if instance_id not in id_map:
                raise ValueError(
                    f"{src}: instance_id {instance_id!r} not found in "
                    "kjain14/testgenevallite"
                )
            record["id"] = id_map[instance_id]
            fout.write(json.dumps(record) + "\n")
            remapped += 1
    return remapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Directories to search recursively for prediction *.jsonl files",
    )
    parser.add_argument(
        "--suffix",
        default="_lite",
        help="Suffix appended to each input directory's name for the output directory",
    )
    args = parser.parse_args()

    id_map = build_lite_id_map()

    for input_dir in args.input_dirs:
        input_dir = input_dir.resolve()
        output_dir = input_dir.parent / f"{input_dir.name}{args.suffix}"
        jsonl_files = sorted(input_dir.rglob("*.jsonl"))
        if not jsonl_files:
            print(f"no .jsonl files found under {input_dir}")
            continue
        for src in jsonl_files:
            dst = output_dir / src.relative_to(input_dir)
            count = remap_file(src, dst, id_map)
            print(
                f"{src.relative_to(input_dir.parent)} -> "
                f"{dst.relative_to(input_dir.parent)} ({count} predictions)"
            )


if __name__ == "__main__":
    main()
