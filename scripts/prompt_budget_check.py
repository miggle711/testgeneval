"""Rule 4 check for a kg_only prompt change: which instances would be
excluded for a model with and without it (fyp-paper#7, miggle711/pycodekg#237).

Rule 4 excludes an instance for a model, in both arms, when either arm's
prompt does not fit: the server refuses a request whose prompt plus
max_tokens (the model's OUTPUT_LIMITS entry) exceeds its --max-model-len.
This script builds the final prompts with this checkout's own prompt
classes, the same ones run_api.py uses (InstructPrompt and KGOnlyPrompt),
wraps them in messages as call_chat does, counts tokens with the model's
own tokenizer and chat template, and applies that limit to three prompts
per instance:

  instruct            the whole-file arm (targets from --kg-prompts)
  kg_only, with       kg_prompts.json built normally
  kg_only, without    built with build_kg_prompts.py --without-module-constants

An instance is excluded when instruct or its kg_only prompt is over. The
report pairs the two versions: instances excluded only with the change are
what the change costs; excluded only without it should be none.

The count is the tokenizer's, not the server's: vLLM may differ by a few
tokens (special tokens, template details), so the smallest margin is
reported too. The server's own refusal, recorded by testgeneval#134 as
prompt_too_long, stays the rule in a real run.

Usage (an env with transformers and this repo's dependencies; on M3 the
vLLM env; gated models need an HF token):

    python scripts/prompt_budget_check.py --dataset-path <saved dataset> \\
        --kg-prompts <kg_prompts.json> --kg-prompts-without <kg_prompts_without.json> \\
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct --max-model-len 98304 \\
        --out budget_qwen.csv
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def full_prompts(prompt_info, dataset):
    """{id: full-setting user prompt} and the system message, from a prompt
    class, exactly as run_api.main builds them."""
    built = prompt_info.add_prompts_to_dataset(dataset)["test"]
    return {r["id"]: r["preds_prompts"]["full"] for r in built}, prompt_info.system_message_full


def messages(system, user, no_system_message=False):
    """The chat messages call_chat sends."""
    if no_system_message:
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tokenizer_counter(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    def count(msgs):
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return len(ids)
    return count


def compare(tokens, budget):
    """tokens: {id: {"instruct": n, "with": n, "without": n}}. Returns
    per-instance rows and a summary. A prompt fits when n <= budget, where
    budget = max_model_len - output limit."""
    rows = []
    for i in sorted(tokens):
        t = tokens[i]
        over = {k: t[k] > budget for k in ("instruct", "with", "without")}
        rows.append({"id": i, **{f"{k}_tokens": t[k] for k in ("instruct", "with", "without")},
                     "budget": budget,
                     "excluded_without": over["instruct"] or over["without"],
                     "excluded_with": over["instruct"] or over["with"],
                     "kg_only_margin_with": budget - t["with"]})
    with_ = {r["id"] for r in rows if r["excluded_with"]}
    without = {r["id"] for r in rows if r["excluded_without"]}
    summary = {
        "instances": len(rows), "budget": budget,
        "excluded_without": len(without), "excluded_with": len(with_),
        "excluded_only_with": sorted(with_ - without),
        "excluded_only_without": sorted(without - with_),
        "kg_only_tokens_added": {
            "instances_changed": sum(r["with_tokens"] != r["without_tokens"] for r in rows),
            "max": max((r["with_tokens"] - r["without_tokens"] for r in rows), default=0)},
        "max_tokens": {k: max((r[f"{k}_tokens"] for r in rows), default=0)
                       for k in ("instruct", "with", "without")},
        "smallest_kg_only_margin_with": min((r["kg_only_margin_with"] for r in rows), default=None),
    }
    return rows, summary


def main(argv=None, counter=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", required=True, help="a dataset saved with save_to_disk (test split)")
    ap.add_argument("--kg-prompts", required=True, help="kg_prompts.json as built for the run")
    ap.add_argument("--kg-prompts-without", required=True,
                    help="the same build with --without-module-constants (or without the change measured)")
    ap.add_argument("--model", required=True, help="HF model id: its tokenizer, and its output limit")
    ap.add_argument("--max-model-len", type=int, required=True, help="the server's --max-model-len")
    ap.add_argument("--output-limit", type=int,
                    help="max_tokens sent per request (default: run_api.OUTPUT_LIMITS for --model)")
    ap.add_argument("--no-system-message", action="store_true", help="as run_api's flag")
    ap.add_argument("--out", required=True, help="per-instance CSV")
    args = ap.parse_args(argv)

    from datasets import load_from_disk
    from inference.configs.instruct_prompt import InstructPrompt
    from inference.configs.kg_only_prompt import KGOnlyPrompt
    output_limit = args.output_limit
    if output_limit is None:
        from inference.api.run_api import OUTPUT_LIMITS
        if args.model not in OUTPUT_LIMITS:
            raise SystemExit(f"{args.model} has no OUTPUT_LIMITS entry; pass --output-limit")
        output_limit = OUTPUT_LIMITS[args.model]
    budget = args.max_model_len - output_limit

    dataset = load_from_disk(args.dataset_path)
    with_ids = set(json.load(open(args.kg_prompts)))
    without_ids = set(json.load(open(args.kg_prompts_without)))
    if with_ids != without_ids:
        raise SystemExit(f"the two kg_prompts files cover different instances "
                         f"({len(with_ids ^ without_ids)} differ); build both from the same KGs")
    dataset = dataset.filter(lambda r: r["id"] in with_ids)

    instruct, sys_i = full_prompts(InstructPrompt(kg_prompts_path=args.kg_prompts), dataset)
    kg_with, sys_k = full_prompts(KGOnlyPrompt(args.kg_prompts), dataset)
    kg_without, _ = full_prompts(KGOnlyPrompt(args.kg_prompts_without), dataset)

    count = counter or tokenizer_counter(args.model)
    tokens = {i: {"instruct": count(messages(sys_i, instruct[i], args.no_system_message)),
                  "with": count(messages(sys_k, kg_with[i], args.no_system_message)),
                  "without": count(messages(sys_k, kg_without[i], args.no_system_message))}
              for i in kg_with}
    rows, summary = compare(tokens, budget)
    summary.update({"model": args.model, "max_model_len": args.max_model_len,
                    "output_limit": output_limit})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
