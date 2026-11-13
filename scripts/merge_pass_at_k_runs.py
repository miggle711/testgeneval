# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Merges a stage-1 (k=1) and a stage-2 (k=4) inference run into one k=5
predictions file, for the staged pass@5 plan proposed in fyp-paper#5.

Why this exists: run_api.py's resume logic skips any id already present in
its own output file, so a k1 run cannot be "topped up" in place, and
merge_and_validate.py only concatenates same-k *shards*. This script merges
per instance id across two runs of different k, concatenating
preds[prompt_name] with the stage-1 sample(s) FIRST. That ordering is the
whole point of the staged plan: {setting}_pass_at_1 = any(tests_passed[:1]),
so keeping stage 1 first means the already-reported pass@1 cannot change
when stage 2 lands.

Guarantees enforced here rather than by convention:
  * model_name_or_path must be identical across both files (it embeds the
    temperature, e.g. "..._t=0.8", so this doubles as the config-freeze
    check between stages). run_api.py records only the model's basename,
    so two models with the same basename under different organisations
    can not be told apart here; the project's models all have distinct
    basenames.
  * Neither stage may hold more samples per id than its k (--stage1_k,
    --stage2_k), so a wrong file can not push the merged list past 5.
  * preds_prompts[prompt_name] must be identical per id between the two
    files -- a mismatch means the serialiser or dataset changed between
    stages, which would make the merged samples non-exchangeable.
  * Ids present only in the stage-2 file are EXCLUDED by default: their
    pass@1 would come from a stage-2 sample, silently changing the pass@1
    population after stage 1 was reported. --include_stage2_only opts in.
  * For the same reason, an id whose stage-1 record has an empty preds list
    (run_api.py writes one when the sample's content came back None) is
    kept exactly as stage 1 wrote it, with no stage-2 samples appended. Its
    prompts are still checked against stage 2's.

Non-guarantees, by design:
  * usage lists are concatenated independently of preds, never zipped:
    usage can be absent entirely (pre-instrumentation files) or longer than
    preds (a None-content sample keeps its usage entry but produces no
    pred, see run_api.py's none-content guard). If only one file has usage
    for a record, usage is dropped for that record with a warning, since a
    partial list could be misread as aligned.
  * Merged records can hold fewer than expected_k samples (none-content
    guard); downstream report generation already handles this with its
    len(...) >= k guard, which simply omits pass@k for that instance.

Only reads the two input files and writes a new output file; never
modifies the inputs, so it is safe to re-run.
"""

import argparse
import copy
import json
import os
import sys
from collections import Counter


def read_preds_file(path: str) -> dict:
    """Reads a predictions jsonl into an ordered {id: record} dict.
    Fails on malformed JSON or duplicate ids -- a duplicate id means
    corrupted/interleaved concurrent writes or a double-submitted job
    (same failure classes merge_and_validate.py checks for), and merging
    on top of that would silently pick one of the two records.
    """
    records = {}
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: malformed JSON: {e}")
            curr_id = rec["id"]
            if curr_id in records:
                raise ValueError(
                    f"{path}: duplicate id {curr_id!r} (lines "
                    f"{records[curr_id]['_line']} and {line_no}) -- "
                    f"investigate before merging, do not trust this file"
                )
            rec["_line"] = line_no
            records[curr_id] = rec
    return records


def check_model_consistency(stage1: dict, stage2: dict) -> str:
    """All records in both files must share one model_name_or_path.
    Because run_api.py writes it as f"{name}_t={temperature}", this is
    also the between-stages config-freeze check for temperature.
    """
    names = {
        rec.get("model_name_or_path")
        for rec in list(stage1.values()) + list(stage2.values())
    }
    if len(names) != 1:
        raise ValueError(
            f"model_name_or_path differs across the two runs: {sorted(map(str, names))}. "
            f"The stages were generated under different configs (the suffix "
            f"embeds temperature); these samples are not exchangeable and "
            f"must not be merged."
        )
    return names.pop()


def _check_prompts_match(curr_id, rec1, rec2, prompt_name, allow_prompt_mismatch, stats):
    """The two stages must have seen the same prompt for an id, including
    when stage 1 produced no sample: a mismatch means the serialiser or
    dataset changed between stages."""
    prompt1 = rec1.get("preds_prompts", {}).get(prompt_name)
    prompt2 = rec2.get("preds_prompts", {}).get(prompt_name)
    if prompt1 == prompt2:
        return
    if not allow_prompt_mismatch:
        raise ValueError(
            f"id {curr_id!r}: preds_prompts[{prompt_name!r}] differs "
            f"between the two runs -- the serialiser or dataset "
            f"changed between stages, so the samples are not "
            f"exchangeable. Rerun one stage, or pass "
            f"--allow_prompt_mismatch if this is a known, "
            f"documented difference."
        )
    stats["prompt_mismatch_allowed"] += 1


def _check_sample_counts(curr_id, rec1, rec2, prompt_name, stage1_k, stage2_k):
    """A stage may hold fewer samples than its k (a None-content sample is
    skipped), never more: a k=5 file given as stage 1, say, would push the
    merged list past 5 and evaluation would silently use only the first 5."""
    for label, rec, k in (("stage 1", rec1, stage1_k), ("stage 2", rec2, stage2_k)):
        n = len(rec.get("preds", {}).get(prompt_name) or [])
        if n > k:
            raise ValueError(
                f"id {curr_id!r}: {label} has {n} samples, more than its k={k}. "
                f"Check that the right files were passed as --stage1 and --stage2."
            )


def merge_records(
    stage1: dict,
    stage2: dict,
    prompt_name: str,
    include_stage2_only: bool,
    allow_prompt_mismatch: bool,
    stage1_k: int = 1,
    stage2_k: int = 4,
) -> tuple[list, Counter]:
    """Returns (merged records in stage-1 order, stats Counter)."""
    stats = Counter()
    merged = []

    for curr_id, rec1 in stage1.items():
        out = copy.deepcopy(rec1)
        out.pop("_line", None)
        rec2 = stage2.get(curr_id)

        stage1_preds = rec1.get("preds", {}).get(prompt_name)
        if rec2 is not None:
            _check_prompts_match(curr_id, rec1, rec2, prompt_name, allow_prompt_mismatch, stats)
            _check_sample_counts(curr_id, rec1, rec2, prompt_name, stage1_k, stage2_k)
        if rec2 is None:
            stats["stage1_only"] += 1
        elif not stage1_preds:
            # run_api.py writes a record with an EMPTY preds list when the
            # stage-1 sample came back with None content. Appending stage 2
            # here would move a stage-2 sample into position 0 and give this
            # id a pass@1 it did not have after stage 1, changing the
            # reported pass@1 population. Keep the stage-1 record as is;
            # this id gets neither pass@1 nor pass@5, as before stage 2.
            stats["stage1_empty_not_merged"] += 1
        else:
            preds1 = rec1.get("preds", {}).get(prompt_name)
            preds2 = rec2.get("preds", {}).get(prompt_name)
            if preds1 is None and preds2 is None:
                stats["missing_prompt_both"] += 1
            else:
                # Stage-1 samples first: pass_at_1 reads tests_passed[:1],
                # so this ordering is what freezes the reported pass@1.
                out.setdefault("preds", {})[prompt_name] = (
                    (preds1 or []) + (preds2 or [])
                )
                # The merged row holds both stages' samples, so its run
                # identity is no longer stage 1's alone.
                run1, run2 = rec1.get("run"), rec2.get("run")
                if run1 and run2:
                    out["run"] = {
                        **run1,
                        "num_samples": run1["num_samples"] + run2["num_samples"],
                        "merged_from": [run1, run2],
                    }
                if preds1 is None or preds2 is None:
                    stats["missing_prompt_one_side"] += 1

            usage1 = rec1.get("usage", {}).get(prompt_name) if "usage" in rec1 else None
            usage2 = rec2.get("usage", {}).get(prompt_name) if "usage" in rec2 else None
            if usage1 is not None and usage2 is not None:
                out.setdefault("usage", {})[prompt_name] = usage1 + usage2
            elif usage1 is not None or usage2 is not None:
                # One-sided usage would look aligned but cover only one
                # stage's samples; drop it rather than mislead.
                if "usage" in out:
                    out["usage"].pop(prompt_name, None)
                stats["usage_dropped"] += 1

        merged.append(out)

    stage2_only_ids = [i for i in stage2 if i not in stage1]
    if include_stage2_only:
        for curr_id in stage2_only_ids:
            n = len(stage2[curr_id].get("preds", {}).get(prompt_name) or [])
            if n > stage2_k:
                raise ValueError(
                    f"id {curr_id!r}: stage 2 has {n} samples, more than its k={stage2_k}."
                )
            out = copy.deepcopy(stage2[curr_id])
            out.pop("_line", None)
            merged.append(out)
            stats["stage2_only_included"] += 1
    else:
        stats["stage2_only_excluded"] = len(stage2_only_ids)

    return merged, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1", type=str, required=True,
        help="Stage-1 predictions jsonl (the k=1 run); its samples go first",
    )
    parser.add_argument(
        "--stage2", type=str, required=True,
        help="Stage-2 predictions jsonl (the k=4 top-up run)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Merged k=5 predictions jsonl to write (must not exist unless --force)",
    )
    parser.add_argument("--prompt_name", type=str, default="full")
    parser.add_argument("--stage1_k", type=int, default=1,
                        help="Samples per instance in the stage-1 file (default 1)")
    parser.add_argument("--stage2_k", type=int, default=4,
                        help="Samples per instance in the stage-2 file (default 4)")
    parser.add_argument(
        "--expected_k", type=int, default=5,
        help="Informational: report how many merged records fall short of "
             "this sample count (the none-content guard makes some shortfall "
             "normal; report generation's len(...) >= k guard handles it)",
    )
    parser.add_argument(
        "--include_stage2_only", action="store_true",
        help="Also emit ids present only in the stage-2 file. Off by default "
             "because such ids would take their pass@1 from a stage-2 sample, "
             "changing the pass@1 population after stage 1 was reported.",
    )
    parser.add_argument(
        "--allow_prompt_mismatch", action="store_true",
        help="Proceed when preds_prompts differ between stages for an id "
             "(normally a hard error -- it means non-exchangeable samples)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite --output if it exists")
    args = parser.parse_args(argv)

    for path in (args.stage1, args.stage2):
        # samefile also catches a symlink or hard link to an input, which
        # --force would otherwise overwrite.
        same = (os.path.abspath(path) == os.path.abspath(args.output)
                or (os.path.exists(path) and os.path.exists(args.output)
                    and os.path.samefile(path, args.output)))
        if same:
            print(f"--output must not be an input file: {path}", file=sys.stderr)
            return 1
    if os.path.exists(args.output) and not args.force:
        print(
            f"{args.output} already exists; pass --force to overwrite. "
            f"(Refusing by default -- overwriting a merged file that "
            f"evaluation already consumed is how mixed-data incidents start.)",
            file=sys.stderr,
        )
        return 1

    try:
        stage1 = read_preds_file(args.stage1)
        stage2 = read_preds_file(args.stage2)
        model_name = check_model_consistency(stage1, stage2)
        merged, stats = merge_records(
            stage1, stage2, args.prompt_name,
            args.include_stage2_only, args.allow_prompt_mismatch,
            args.stage1_k, args.stage2_k,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    with open(args.output, "w") as f:
        for rec in merged:
            print(json.dumps(rec), file=f)

    sample_counts = Counter(
        len(rec.get("preds", {}).get(args.prompt_name, [])) for rec in merged
    )
    short = sum(n for k, n in sample_counts.items() if k < args.expected_k)

    print(f"Merged {len(merged)} records into {args.output} (model: {model_name})")
    print(f"Samples per record: {dict(sorted(sample_counts.items()))}")
    if short:
        print(
            f"NOTE: {short} record(s) hold fewer than {args.expected_k} samples "
            f"(expected when the none-content guard dropped samples; those "
            f"instances simply get no pass@{args.expected_k})."
        )
    for key in (
        "stage1_only", "stage1_empty_not_merged",
        "stage2_only_excluded", "stage2_only_included",
        "usage_dropped", "missing_prompt_one_side", "missing_prompt_both",
        "prompt_mismatch_allowed",
    ):
        if stats[key]:
            print(f"  {key}: {stats[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
