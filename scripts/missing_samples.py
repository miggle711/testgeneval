"""Describe missing full-setting samples without attributing their cause.

Read-only and standard-library only. Run once per model/arm directory:
  python scripts/missing_samples.py /path/to/eval_logs --instances-file ids.txt
Add --predictions run.jsonl to distinguish prediction and evaluation gaps.
Add --runner-log run.out (repeatable) for this same run's outer host messages.
Without an instance list, absent logs are counted only within the observed
logs/predictions universe, not the dataset. Output is JSON on stdout.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swebench_docker.constants import (  # noqa: E402
    APPLY_PATCH_FAIL, INSTALL_FAIL, KEY_ID, KEY_PREDICTIONS, TESTS_CONFIG, TESTS_ERROR, TESTS_TIMEOUT,
    UNFILTERED_TESTS_FAILED, UNFILTERED_TESTS_PASSED,
)

CONFIG = re.compile(re.escape(TESTS_CONFIG) + r"\s*([^\s]+)\s+(pred|baseline)\b[^\n]*")
HOST_TIMEOUT = re.compile(r"Container exceeded host timeout of \d+s, killed\.")
RUNNER_EVENT = re.compile(
    r"\[([^\]\n]+)\]\[[^\]\n]+\]\s+(Container exceeded host timeout of \d+s, killed\.|Error running container:[^\n]*|No \.sif found[^\n]*)"
)
ENDINGS = ("timeout_marker", "harness_error", "cut_off_mid_block", "cleanly", "no_full_block")


def input_info(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def full_blocks(text):
    configs = list(CONFIG.finditer(text))
    return [text[m.end():configs[i + 1].start() if i + 1 < len(configs) else len(text)]
            for i, m in enumerate(configs) if m.group(1, 2) == ("full", "pred")]


def end_marker(text):
    return max(text.rfind(UNFILTERED_TESTS_PASSED), text.rfind(UNFILTERED_TESTS_FAILED))


def ending(blocks, text):
    # A timeout in an earlier sample is not evidence about the final sample.
    tail = blocks[-1] if blocks else text
    terminal = end_marker(tail) if blocks else -1
    timeout = max(tail.rfind(TESTS_TIMEOUT), tail.rfind("MutationTimeout"),
                  max((m.start() for m in HOST_TIMEOUT.finditer(tail)), default=-1))
    error = max(tail.rfind(TESTS_ERROR), tail.rfind(APPLY_PATCH_FAIL), tail.rfind(INSTALL_FAIL),
                tail.rfind("Traceback (most recent call last):"),
                tail.rfind("Error running container:"))
    if timeout > max(terminal, error):
        return "timeout_marker"
    if error > terminal:
        return "harness_error"
    if terminal >= 0:
        return "cleanly"
    return "cut_off_mid_block" if blocks else "no_full_block"


def predictions(path):
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        instance = row[KEY_ID]
        if not isinstance(instance, str) or not instance:
            raise ValueError(f"{path}:{number}: id must be a nonempty string")
        payload = row.get(KEY_PREDICTIONS, {})
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{number}: preds must be an object")
        values = payload.get("full")
        if instance in result:
            raise ValueError(f"{path}:{number}: duplicate prediction id {instance}")
        if values is not None and (not isinstance(values, list) or
                                   not all(isinstance(v, str) for v in values)):
            raise ValueError(f"{path}:{number}: preds.full must be a list of strings")
        # Missing setting and absent record are distinguishable from an empty list.
        result[instance] = None if values is None else len(values)
    return result


def summary(rows, k):
    present = [r for r in rows if r["log"] is not None]
    short = [r for r in present if r["full_blocks"] < k]
    gaps = [r for r in rows if r["log"] is None or r["full_blocks"] < k]
    histogram = Counter(r["full_blocks"] for r in present)
    return {
        "instances": len(rows), "logs": len(present),
        "no_log": len(rows) - len(present),
        "logs_by_full_blocks": {str(n): histogram[n] for n in range(k + 1)},
        "logs_over_k": sum(n for blocks, n in histogram.items() if blocks > k),
        "short_log_endings": {name: sum(r["ending"] == name for r in short) for name in ENDINGS},
        "short_or_absent_prediction_counts": dict(sorted(Counter(
            str(r["prediction_count"]) if r["prediction_count"] is not None else r["prediction_status"]
            for r in gaps).items())),
        "short_or_absent_runner_events": dict(sorted(Counter(
            event for r in gaps for event in r["runner_events"]).items())),
    }


def collect(log_dir, k=5, instances_file=None, predictions_file=None, runner_logs=()):
    if not log_dir.is_dir():
        raise ValueError(f"not a log directory: {log_dir}")
    records = {}
    manifest = hashlib.sha256()
    for path in sorted(log_dir.rglob("*.full.eval.log")):
        instance = path.name.split(".", 1)[0]
        if instance in records:
            raise ValueError(f"multiple full logs for {instance}; use one model/arm directory")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        manifest.update(json.dumps([str(path.relative_to(log_dir)), digest]).encode() + b"\n")
        text = raw.decode("utf-8", errors="replace")
        blocks = full_blocks(text)
        records[instance] = {"log": str(path), "log_sha256": digest,
                             "full_blocks": len(blocks),
                             "finished_full_blocks": sum(end_marker(b) >= 0 for b in blocks),
                             "ending": ending(blocks, text)}
    preds = predictions(predictions_file) if predictions_file else {}
    events = {}
    for path in runner_logs:
        for match in RUNNER_EVENT.finditer(path.read_text(errors="replace")):
            instance, message = match.groups()
            event = ("host_timeout" if HOST_TIMEOUT.fullmatch(message) else
                     "missing_image" if message.startswith("No .sif") else "container_error")
            events.setdefault(instance, set()).add(event)
    if instances_file:
        ids = [line.strip() for line in instances_file.read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ids in instance list")
        universe = set(ids)
        outside = sorted((records.keys() | preds.keys()) - universe)
        if outside:
            raise ValueError(f"logs/predictions outside instance list: {outside[:5]}")
        scope = "explicit instance list"
    else:
        universe = records.keys() | preds.keys()
        scope = "observed logs and predictions only; dataset-wide absent logs unknown"
    rows = []
    for instance in sorted(universe):
        row = {"instance": instance, "repo": re.sub(r"-\d+(?:-\d+)?$", "", instance),
               "log": None, "full_blocks": 0, "finished_full_blocks": 0, "ending": "no_log"}
        row.update(records.get(instance, {}))
        row["prediction_count"] = preds.get(instance)
        row["prediction_status"] = ("not_supplied" if predictions_file is None else
                                    "no_record" if instance not in preds else
                                    "no_full_setting" if preds[instance] is None else "available")
        row["runner_events"] = sorted(events.get(instance, ()))
        rows.append(row)
    return {"run": log_dir.name, "k": k, "universe": scope,
            "inputs": {"log_directory": str(log_dir.resolve()), "logs_manifest_sha256": manifest.hexdigest(),
                       "instances": input_info(instances_file) if instances_file else None,
                       "predictions": input_info(predictions_file) if predictions_file else None,
                       "runner_logs": [input_info(p) for p in runner_logs]},
            "notes": ["Block counts are starts; finished_full_blocks counts terminal unfiltered markers.",
                      "A clean short log does not explain why later samples are absent.",
                      "Runner events are same-id evidence, not proof of the final log's cause; retries may exist.",
                      "Without same-run runner logs, host timeout cannot be distinguished from other kills."],
            "overall": summary(rows, k),
            "by_repo": {repo: summary([r for r in rows if r["repo"] == repo], k)
                        for repo in sorted({r["repo"] for r in rows})},
            "instances": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--instances-file", type=Path, help="Expected dataset ids, one per line (the id field, not instance_id).")
    parser.add_argument("--predictions", type=Path, help="This model/arm's predictions JSONL.")
    parser.add_argument("--runner-log", type=Path, action="append", default=[], help="Outer log from this run; repeatable.")
    parser.add_argument("--summary-only", action="store_true", help="Omit individual instance rows from stdout.")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")
    try:
        report = collect(args.log_dir, args.k, args.instances_file, args.predictions, args.runner_log)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.summary_only:
        del report["instances"]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
