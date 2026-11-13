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
same task, under a sibling directory (input dir name + suffix). Originals are
left untouched.

A task is (instance_id, code_file, test_file), not instance_id alone: two Lite
rows share an instance_id with a full-dataset row but have a different test
file (testgeneval#143). A prediction for such a row is refused, or with
--skip-mismatched left out and reported; it is never remapped to the other
task.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def task_key(row) -> tuple:
    """What identifies a task. instance_id alone does not: the same instance
    can be a different task in the two datasets when its test file differs
    (sympy__sympy-17022 and sympy__sympy-19007, testgeneval#143)."""
    return (row["instance_id"], row["code_file"], row["test_file"])


def build_lite_id_map(full_rows, lite_rows):
    """({full id: Lite id} for full rows that are the same task as a Lite
    row, {full id: Lite id} for full rows whose instance_id is in Lite but
    whose task differs, set of Lite ids)."""
    lite_by_task = {task_key(r): r["id"] for r in lite_rows}
    lite_by_iid = {r["instance_id"]: r["id"] for r in lite_rows}
    same, different = {}, {}
    for r in full_rows:
        if task_key(r) in lite_by_task:
            same[r["id"]] = lite_by_task[task_key(r)]
        elif r["instance_id"] in lite_by_iid:
            different[r["id"]] = lite_by_iid[r["instance_id"]]
    return same, different, set(lite_by_task.values())


def load_maps():
    full = load_dataset("kjain14/testgeneval", split="test")
    lite = load_dataset("kjain14/testgenevallite", split="test")
    return build_lite_id_map(full, lite)


def remap_file(src: Path, dst: Path, maps, skip_mismatched=False):
    """Write src with each prediction's full-dataset id replaced by the Lite
    id of the same task. Returns (written, skipped ids). A prediction whose
    instance is in Lite as a different task is refused, or skipped and
    returned when skip_mismatched is set; one already carrying a Lite id is
    kept; one with no Lite counterpart at all is refused."""
    same, different, lite_ids = maps
    rows, skipped = [], []
    for line in src.open():
        if not line.strip():
            continue
        record = json.loads(line)
        rid = record["id"]
        if rid in same:
            record["id"] = same[rid]
        elif rid in different:
            if not skip_mismatched:
                raise ValueError(
                    f"{src}: {rid} has instance_id {record['instance_id']!r}, which is in "
                    f"kjain14/testgenevallite as a different task ({different[rid]}: another "
                    "test file). Remapping would evaluate it as that task. Pass "
                    "--skip-mismatched to leave it out and report it."
                )
            skipped.append(rid)
            continue
        elif rid not in lite_ids:
            raise ValueError(f"{src}: {rid} ({record['instance_id']!r}) is not a task in kjain14/testgenevallite")
        rows.append(record)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fout:
        for record in rows:
            fout.write(json.dumps(record) + "\n")
    return len(rows), skipped


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
    parser.add_argument(
        "--skip-mismatched",
        action="store_true",
        help="Leave out predictions whose instance is in Lite as a different task "
             "(another test file) and report them, instead of stopping",
    )
    args = parser.parse_args()

    maps = load_maps()

    for input_dir in args.input_dirs:
        input_dir = input_dir.resolve()
        output_dir = input_dir.parent / f"{input_dir.name}{args.suffix}"
        jsonl_files = sorted(input_dir.rglob("*.jsonl"))
        if not jsonl_files:
            print(f"no .jsonl files found under {input_dir}")
            continue
        for src in jsonl_files:
            dst = output_dir / src.relative_to(input_dir)
            count, skipped = remap_file(src, dst, maps, args.skip_mismatched)
            print(
                f"{src.relative_to(input_dir.parent)} -> "
                f"{dst.relative_to(input_dir.parent)} ({count} predictions"
                + (f", {len(skipped)} left out as a different Lite task: {', '.join(skipped)}" if skipped else "")
                + ")"
            )


if __name__ == "__main__":
    main()
