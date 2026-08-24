# Inference run log

Tracks completion counts and known failure causes per model/arm, since a
model's raw completed-instance count alone doesn't distinguish "ran clean"
from "lost instances to context overflow" -- that distinction matters for
comparing models fairly and should be reported alongside any pass-rate
numbers, not silently absorbed into them.

Full dataset (`kjain14/testgeneval`) is 1210 instances. All runs below are
the `instruct` (baseline) arm, temperature 0, `full` setting only.

## instruct arm

### Small (~7-8B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 1210 / 1210 | 0 | Clean run, no losses. |
| DeepSeek-Coder-6.7B-Instruct | 1100 / 1210 | 110 | 32768 context, largest `instruct` prompts exceed it. |
| CodeGemma-7B-IT | not completed | -- | Abandoned. MAX_MODEL_LEN=8192 (model's real limit) would produce an even higher loss rate than StarCoder2; deprioritized in favor of medium-tier models instead of forcing a run through it. |
| Meta-Llama-3.1-8B-Instruct | 1210 / 1210 | 0 | Access was in fact confirmed and a full 8-shard run completed; this row was stale (previously read "not run, license access not yet confirmed"). Corrected 2026-08-22 after verifying real `wc -l` counts on all 8 shard files summed to 1210, merged via `cat` per GUIDE.md. |
| Gemma-3-4B-it | 1121 / 1210 | 89 | From the team's funding proposal shortlist (`docs/funding-proposal.md`), gated (Google's license), access confirmed. Rejects `float16` outright at vLLM startup ("The model type 'gemma3' does not support float16. Reason: Numerical instability."), needed `m3_run_inference.slurm`'s new `DTYPE` env var (`DTYPE=bfloat16`) to run at all. Job 59426744, completed 2026-08-24, ~1h28m. Model's real context window is 128K, but `MAX_MODEL_LEN` wasn't overridden for this run and stayed at the script's default (32768), so all 89 losses are the same context-length `BadRequestError` seen elsewhere in this doc, not a Gemma3-specific issue. Worth a rerun with `MAX_MODEL_LEN` raised to actually use the model's real context budget, if a cleaner comparison point is wanted later. |
| Phi-4 | in progress | -- | From the funding proposal shortlist, not gated. Only 16K context, the smallest of any model run so far (StarCoder2's 16384 is the closest comparison, 31% loss rate), expect a real, meaningful loss count. Job 59425624, submitted 2026-08-24 by wtho0016. |

### Medium (~15-32B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Instruct | 1117 / 1210 | 93 | 32768 context. |
| StarCoder2-15B-Instruct-v0.1 | 832 / 1210 | 378 | MAX_MODEL_LEN=16384 (model's real limit, confirmed via vLLM's own derived-max-model-len error). Highest loss rate so far (~31%), consistent with having half the context budget of the 32768-context models. |
| Qwen2.5-Coder-32B-Instruct | 1210 / 1210 | 0 | Needs TENSOR_PARALLEL_SIZE=2 (32B doesn't fit one 48GB L40S at float16). All 8 shards completed, merged via `cat` per GUIDE.md, no losses. |
| Codestral-22B-v0.1 | 1101 / 1210 | 109 | ~44GB at float16, tight against 48GB L40S, real OOM risk. Two earlier stalled attempts (323/1210) were caused by `/fs04` sitting at 99% capacity, traced via 2625 `Connection error`s in one job's log (`slurm-59362972.out`), the only job in the shared directory with that signature. Freed ~341GB 2026-08-24 by deleting confirmed-complete models' HF caches with each owner's go-ahead, then restarted the run clean from a fresh output file (job 59400996, 2h21m, completed cleanly). All 109 losses confirmed via `grep -c "Failed, skipping"` plus checking the actual error text (normalizing digits and deduplicating over every `API Error` line) to be the same 32768-token context-length `BadRequestError`, no `Connection error`s this run, consistent with DeepSeek-6.7B/DeepSeek-V2-Lite's loss pattern. |
| Qwen3-Coder-30B-A3B-Instruct | in progress | -- | From the funding proposal shortlist, real successor to Qwen2.5-Coder. MoE (128 experts, 8 active, 30.5B total/3.3B active params), not gated. TENSOR_PARALLEL_SIZE=2. 256K native context, well beyond every other model's context ceiling, expect very few losses if any. Job 59425627, submitted 2026-08-24 by wtho0016. |

### Large (~70-110B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| Meta-Llama-3.1-70B-Instruct | 1210 / 1210 | 0 | Gated, access confirmed. TENSOR_PARALLEL_SIZE=4, 8 shards, all completed and merged via `cat` 2026-08-22. Real per-instance token counts logged in the job's own `slurm-*.out` files (via `run_api.py`'s `input_tokens=`/`output_tokens=` logging): 11,077,947 input / 2,350,216 output tokens total across all 1210 instances. Row previously read "in progress"; corrected after verifying real `wc -l` counts. |
| Qwen2.5-72B-Instruct | 1210 / 1210 | 0 | ~145GB at float16. TENSOR_PARALLEL_SIZE=4, 8 shards submitted by jliu0290 under his own M3 account (separate GPU quota from mvar0010's runs), all completed and merged via `cat` 2026-08-22. Real per-instance token counts aren't available here -- jliu0290's `slurm-*.out` logs live under his own directory, not this shared one; worth pulling if exact token counts for this model are ever needed. Row previously read "in progress"; corrected after verifying real `wc -l` counts. |
| Llama-4-Scout-17B-16E-Instruct | 1210 / 1210 | 0 | From the funding proposal shortlist. Despite the "17B" in the name, this is actually 109B total params (17B active, MoE, 16 experts) -- confirmed against the model card, not the name. ~218GB at float16, doesn't fit the account's normal 4x L40S (48GB) quota (192GB total, less than the model needs). Ran instead on the `m3h` partition's H100 nodes (confirmed real: `srun --partition=m3h --qos=m3h --gres=gpu:H100:1` allocated an 80GB H100, `al49`'s association already includes `m3h` in its QOS list), TENSOR_PARALLEL_SIZE=4 across 4x H100 (320GB total). Gated, access confirmed. Job 59426743, completed 2026-08-25, 52m09s, 2.59s/instance average -- by far the fastest large-tier model despite being the largest by total parameter count, since MoE inference cost scales with active params (17B) not total params (109B), plus 4x H100's extra memory bandwidth over the L40S setup used for the dense 70B/72B models. Zero context-overflow losses, the only new funding-proposal model to complete clean. |

Tiers match `docs/EXPERIMENT_PLAN.md`'s Stage 4 shortlist, plus the new
Small/Medium/Large models added to the team's funding proposal
(`docs/funding-proposal.md`, everything except DeepSeek-V3.2, too large
to self-host and out of scope for this M3 run).

All 8 completed `instruct` models above (everything except the 4 new
funding-proposal models, still in progress) have their merged
predictions gzipped locally and backed up to the team's shared Google
Drive folder
(https://drive.google.com/drive/folders/1YWtuRmoFjjiQx6GntXZamsel1iIY0Mb1)
as of 2026-08-24, so a fresh evaluation run doesn't need to re-pull them
from M3.

## kg_only arm

`kg_prompts.json` is in fact already built and complete at the repo root
-- verified 2026-08-22 via `python3 -c "import json; print(len(json.load(open('kg_prompts.json'))))"`,
which returned 1210 (the full dataset). This row previously read "KG build
done, `kg_prompts.json` generation is the next step"; that step was already
finished by the time this was checked.

First real `kg_only` run submitted 2026-08-22: Qwen2.5-Coder-7B-Instruct,
full `kjain14/testgeneval` dataset, temperature 0, job 59370304 (single
job, no sharding). Chosen because this model's `instruct` arm is already
clean (1210/1210, no context-overflow losses), giving a matched head-to-head
comparison once this finishes. Update this row with the real completed
count once the job finishes.

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

## Two different fairness questions, two different denominators

This dataset actually supports two genuinely different comparisons, and
they need different handling of skipped or missing instances. Conflating
them risks either an unfair cross-model comparison or quietly burying a
real result.

**"Which model is inherently more capable at test generation?"** Here,
context-window ceiling is part of the capability being measured. A
smaller-context model failing on large files is a real limitation, the
same one it would have in actual use. Pass rate computed over each
model's own completed subset is legitimate under this framing, but the
denominator (completed out of 1210) needs to be reported alongside the
rate every time. "StarCoder2 passed 70% of the 832 instances it could
attempt" is not the same claim as "StarCoder2 passed 70% of the
benchmark," and a bare percentage collapses that distinction.

**"Does KG-augmented context outperform flat-file context for test
generation, controlling for model?"** This is the actual research
question (see docs/EXPERIMENT_PLAN.md), and fairness is
stricter here. What matters is comparing instruct vs kg_only on the same
instance set, per model. If instruct's flat-file prompt overflows a
model's context on instance X but kg_only's more compact per-function
view doesn't, that isn't noise to average away. It's potentially the
actual finding: KG framing structurally avoiding context-overflow
failures more often than a flat-file dump does, for the same
context-limited model. Silently dropping mismatched instances from both
arms just to force equal denominators would bury a result worth
reporting.

**Where it genuinely gets unfair** is comparing models using each one's
own denominator as if it were the same test. Reporting Model A
(1210/1210) and Model B (832/1210) as bare "pass@1 = X%" without the
denominator implicitly compares performance on different, non-random
instance subsets. Model B's subset skews toward shorter, simpler files,
which can inflate its apparent rate relative to what it would score on
the full set. The fix isn't excluding Model B from the comparison, it's
never reporting a bare percentage without its denominator, and also
reporting pass rate over the intersection of instances every compared
model actually completed, as a second, strictly apples-to-apples number
alongside the per-model number that includes its own context limits.

## "PostprocessingError" can mask a real generation failure

`swebench_docker/swebench_utils.py`'s `classify_error()` is the function
that turns raw test output into the short label written to `test_error`
in results JSON. It has a catch-all branch: if the captured output has no
recognizable error text and the literal word "test" doesn't appear
anywhere in it, it labels the instance `PostprocessingError`, implying
something went wrong parsing the model's output rather than in the
generated test itself.

Diagnosed for real on `django__django-11179-15747` under
Qwen2.5-Coder-7B-Instruct (`instruct` arm): the generated test file
imports `django.test.TestCase` but the class itself subclasses plain
`unittest.TestCase`. Django's test database only gets created and wired
up automatically for `django.test.TestCase`/`TransactionTestCase`; plain
`unittest.TestCase` skips that setup entirely. The generated test then
calls `self.MyModel.objects.create(...)` against a database connection
that was never configured, and the real underlying exception is
`django.core.exceptions.ImproperlyConfigured: settings.DATABASES is
improperly configured. Please supply the NAME value.` That exception
doesn't match any of `classify_error()`'s known patterns, so it falls
through to the generic label.

Confirmed by reproducing directly in the real `kdjain/swe-bench-django_
django-testbed:3.0` container: running the gold test module with the
same `coverage run ./tests/runtests.py` command works cleanly (41/41
passed), so this isn't a testbed or harness bug. It's a real model
output defect (wrong base class for a Django ORM test), not noise to
filter out. Across the one results file checked (Qwen2.5-Coder-7B-
Instruct, `instruct` arm, 134 instances), 46 hit `PostprocessingError`,
21 in astropy and 25 in django, concentrated in those two repos only.
The astropy side hasn't been diagnosed against a matching model+instance
log yet, so it may or may not share this same root cause.

Not fixing `classify_error()` right now since evaluation runs are
actively in flight elsewhere (M3 inference plus at least one teammate
running `run_evaluation.py` locally); changing the classification
mid-run would mean the same underlying error gets labelled differently
depending on when an instance happened to be evaluated, which is its own
consistency problem when comparing results later. Worth fixing between
runs: surface the real exception name (e.g. `ImproperlyConfigured`)
instead of the generic label, so this kind of failure doesn't need
re-diagnosing from scratch next time.

## Evaluation can now run directly on M3, not just locally

Confirmed 2026-08-24: a real evaluation run through the new Apptainer
backend (`run_evaluation.py --backend apptainer`, see testgeneval#2)
produces identical results to the same instance run through Docker.
Same instance (`astropy__astropy-13579-15669`), same numbers: All Tests
Passed, `CoverageLOG` 35.978%, `FunctionCoverageLOG` 4.762%,
`MutationLOG` 12.37%, `FunctionMutationLOG` 0.0%. Not just "it ran
without crashing", the actual scientific output matches, including
`FunctionMutationLOG`, the metric behind the cosmic-ray subprocess fix
in testgeneval#27/#28.

`FunctionCoverageLOG`/`FunctionMutationLOG` are this fork's own addition
on top of the original TestGenEval benchmark (see
`docs/EXPERIMENT_PLAN.md`'s Stage 5), scoped to just the lines of the
patch's target function rather than the whole file. They're the intended
primary comparison metric for RQ2/RQ3, since `kg_only` can structurally
only generate tests for the function it was shown while `instruct` sees
the whole file and could pick up incidental coverage or mutation kills
elsewhere. Whole-file `CoverageLOG`/`MutationLOG` are the original
benchmark's metrics, kept as secondary/contextual numbers. Worth noting
for this one instance: the function-scoped numbers are much lower than
the whole-file ones (4.76% vs 35.98% coverage, 0% vs 12.37% mutation
score), which is structurally plausible on a single instance (the
generated test may exercise the file broadly without deeply exercising
the target function's own lines) but shouldn't be read as a trend from
one data point.

Two real gotchas hit getting this working, both worth knowing before
trying it yourself:
- `load_dataset`'s cache defaults to `$HOME`, not scratch space, and
  `$HOME` has its own small, easy to exhaust quota that `lfs quota`
  can't even inspect (it's not a Lustre filesystem). Set
  `HF_DATASETS_CACHE` to somewhere under scratch, same idea as the
  existing `HF_HOME` fix for model weights.
- A shared git checkout with commits from more than one account can end
  up with `.git` objects owned by different people, breaking `git pull`
  for everyone but each object's original owner, and it's not fixable
  with `setfacl` (would need per-object grants from each owner). Clone
  fresh into your own directory instead of continuing to share one
  checkout across accounts.

Only `astropy_astropy_5.0.sif` exists on M3 so far. The remaining ~142
testbed `.sif` files (one per repo/version combination) still need
building and transferring before this can replace the local Docker
evaluation path at full scale, see testgeneval#2 for the build process
(has to happen off M3, M3 needs `sudo` for `apptainer build`/`pull`
from Docker Hub, which regular accounts don't have).
