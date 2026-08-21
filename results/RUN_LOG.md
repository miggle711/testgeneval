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
| Meta-Llama-3.1-8B-Instruct | not run | -- | Gated, license access not yet confirmed. |

### Medium (~15-32B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Instruct | 1117 / 1210 | 93 | 32768 context. |
| StarCoder2-15B-Instruct-v0.1 | 832 / 1210 | 378 | MAX_MODEL_LEN=16384 (model's real limit, confirmed via vLLM's own derived-max-model-len error). Highest loss rate so far (~31%), consistent with having half the context budget of the 32768-context models. |
| Qwen2.5-Coder-32B-Instruct | 1210 / 1210 | 0 | Needs TENSOR_PARALLEL_SIZE=2 (32B doesn't fit one 48GB L40S at float16). All 8 shards completed, merged via `cat` per GUIDE.md, no losses. |
| Codestral-22B-v0.1 | not run | -- | ~44GB at float16, tight against 48GB L40S, real OOM risk. Not yet attempted. |

### Large (~70-73B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| Meta-Llama-3.1-70B-Instruct | in progress | -- | Gated, access confirmed. TENSOR_PARALLEL_SIZE=4, 8 shards submitted, running one at a time under the 4-GPU QOS cap. |
| Qwen2.5-72B-Instruct | in progress | -- | ~145GB at float16. TENSOR_PARALLEL_SIZE=4, 8 shards submitted by jliu0290 under his own M3 account (separate GPU quota from mvar0010's runs). |

Tiers match `docs/EXPERIMENT_PLAN.md`'s Stage 4 shortlist.

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
