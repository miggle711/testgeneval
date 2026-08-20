# Inference run log

Tracks completion counts and known failure causes per model/arm, since a
model's raw completed-instance count alone doesn't distinguish "ran clean"
from "lost instances to context overflow" -- that distinction matters for
comparing models fairly and should be reported alongside any pass-rate
numbers, not silently absorbed into them.

Full dataset (`kjain14/testgeneval`) is 1210 instances. All runs below are
the `instruct` (baseline) arm, temperature 0, `full` setting only.

## instruct arm

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 1210 / 1210 | 0 | Clean run, no losses. |
| DeepSeek-Coder-6.7B-Instruct | 1100 / 1210 | 110 | 32768 context, largest `instruct` prompts exceed it. |
| DeepSeek-Coder-V2-Lite-Instruct | 1117 / 1210 | 93 | 32768 context. |
| StarCoder2-15B-Instruct-v0.1 | 832 / 1210 | 378 | MAX_MODEL_LEN=16384 (model's real limit, confirmed via vLLM's own derived-max-model-len error). Highest loss rate so far (~31%), consistent with having half the context budget of the 32768-context models. |
| CodeGemma-7B-IT | not completed | -- | Abandoned. MAX_MODEL_LEN=8192 (model's real limit) would produce an even higher loss rate than StarCoder2; deprioritized in favor of medium-tier models instead of forcing a run through it. |
| Meta-Llama-3.1-8B-Instruct | not run | -- | Gated, license access not yet confirmed. |
| Qwen2.5-Coder-32B-Instruct | 1210 / 1210 | 0 | Needs TENSOR_PARALLEL_SIZE=2 (32B doesn't fit one 48GB L40S at float16). All 8 shards completed, merged via `cat` per GUIDE.md, no losses. |
| Codestral-22B-v0.1 | not run | -- | ~44GB at float16, tight against 48GB L40S, real OOM risk. Not yet attempted. |
| Meta-Llama-3.1-70B-Instruct | in progress | -- | Gated, access confirmed. TENSOR_PARALLEL_SIZE=4, 8 shards submitted, running one at a time under the 4-GPU QOS cap. |
| Qwen2.5-72B-Instruct | in progress | -- | ~145GB at float16. TENSOR_PARALLEL_SIZE=4, 8 shards submitted by jliu0290 under his own M3 account (separate GPU quota from mvar0010's runs). |

## kg_only arm

Not started. Needs `kg_prompts.json` built from the full-dataset KG set
(`scripts/build_kg_prompts.py` in `pycodekg`/`repo-kg-construction`) before
any `kg_only` run can happen. KG build (4 GitHub Actions batches) and merge
are done; `kg_prompts.json` generation is the next step.

## Why context-overflow losses happen

The `instruct` arm's prompt shows the model the whole source file as flat
text. Smaller-context models can't fit `instruct`'s largest files (some
source files in this dataset are large) within their real max context,
even with a small requested output budget. This is a genuine, deterministic
model limitation, not a transient failure: `run_api.py`'s `@retry` burns
its retry budget against the same guaranteed-to-fail request every time,
then logs `Failed, skipping...` and the instance is silently absent from
the output file (no `id` written, no error record kept elsewhere).

Confirmed real via `grep -c "Failed, skipping" slurm-<jobid>.out` per shard,
summed across shards, matching each model's exact (1210 - completed) gap.

Whether `kg_only`'s more compact per-function context view hits the same
ceiling as often for these models is an open, real comparison worth making
once `kg_only` runs exist -- may itself be a meaningful finding (KG framing
could structurally avoid this failure mode more often than a flat-file dump
does for the same context-limited models).
