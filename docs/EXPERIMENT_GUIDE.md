# Experiment Guide: generating and evaluating instruct vs kg_only predictions

This is the practical, copy-paste guide for running the full
`instruct` vs `kg_only` comparison — generation *and* evaluation —
across whichever backend a given model actually runs on. The team uses
three:

- **Monash M3** (self-hosted vLLM) — free, university compute.
- **DeepInfra** — paid hosted API, covers most of the shortlisted
  models that don't self-host on M3.
- **OpenAI** — paid API, for models only available that way (e.g.
  GPT-5, per `EXPERIMENT_PLAN.md`).

Read this alongside [GUIDE.md](GUIDE.md) — that's the canonical guide
for the fork as a whole and has the full M3/Slurm instructions this
doc doesn't repeat. The reason one doc can cover DeepInfra and OpenAI
together: `inference/api/run_api.py` talks to a generic
**OpenAI-compatible** chat API. Real OpenAI is its default target; an
env var (`LOCAL_MODEL_BASE_URL`) points the same code at any other
provider exposing that same request shape, DeepInfra included. Same
code, same command shape, different account.

**What's backend-specific vs. generic to the project:**

| Steps | Part of the pipeline | Backend-specific? |
| --- | --- | --- |
| 1 | Picking a backend for your model | — |
| 2–3 | Credentials, model id lookup | Yes — differs between DeepInfra, OpenAI, and M3 |
| 4 | Smoke test | Yes for DeepInfra/OpenAI (not applicable to M3) |
| 5–6 | Docker images, `kg_prompts.json` | No — needed no matter how you generate tests |
| 7–8 | Generation (`instruct` and `kg_only` predictions) | Yes — the command and cost account differ by backend |
| 9 | Evaluation (pytest, coverage, mutation score) | No — free, local Docker, identical regardless of where predictions came from |
| 10–13 | Reading results, scaling up, troubleshooting | Mostly no |

---

## 0. Prerequisites

```bash
cd "<workspace>/testgeneval"
conda env create -f testgeneval.yaml
conda activate testgeneval
```

Do every step below with this environment active. Running
`run_evaluation.py` under a different/system Python is a known source
of a confusing `AttributeError: _ARRAY_API not found` error (numpy/pandas
version mismatch), not a real bug in the script.

---

## 1. Pick the backend for the model you're running

| Backend | Use for | Cost | Where generation happens |
| --- | --- | --- | --- |
| M3 (self-hosted vLLM) | models in the self-hosted shortlist with available GPU headroom | Free | Slurm job — see GUIDE.md's ["Running inference on Monash M3"](GUIDE.md#running-inference-on-monash-m3) section, **not** this doc |
| DeepInfra | most of the paid-API shortlist (see [`funding-proposal.md`](funding-proposal.md)) | Paid, per-token, billed to DeepInfra | Steps 2–8 below, DeepInfra branch |
| OpenAI | models only available that way (e.g. GPT-5) | Paid, per-token, billed to OpenAI | Steps 2–8 below, OpenAI branch |

**If you're running on M3, stop here and follow GUIDE.md instead** —
credential setup and the generation command are different there
(Slurm env vars, not `.env`). Once M3 predictions land in the same
`results/<arm>/<dataset>/preds/` layout, everything from step 9
onward (evaluation, reading results) applies to them exactly the same
as DeepInfra/OpenAI predictions.

The rest of this doc covers the two hosted-API backends, DeepInfra and
OpenAI, side by side, since they share the same code path.

---

## 2. Configure credentials

```bash
cd "<workspace>/testgeneval"
cp .env_template .env
```

### DeepInfra

```bash
LOCAL_MODEL_BASE_URL=https://api.deepinfra.com/v1/openai/
LOCAL_MODEL_API_KEY=<your real DeepInfra API key>
SWEBENCH_DOCKER_FORK_DIR=<absolute path to this testgeneval checkout>
```

- **The trailing slash on `LOCAL_MODEL_BASE_URL` is required.** Without
  it, the OpenAI client concatenates the path wrong and every request
  404s (see the comment at
  [inference/api/run_api.py:374-381](../inference/api/run_api.py#L374-L381)).
- `LOCAL_MODEL_BASE_URL` being set is what routes the code to DeepInfra
  instead of real OpenAI
  ([run_api.py:382-386](../inference/api/run_api.py#L382-L386)).

### OpenAI

```bash
OPENAI_API_KEY=<your real OpenAI API key>
SWEBENCH_DOCKER_FORK_DIR=<absolute path to this testgeneval checkout>
```

Leave `LOCAL_MODEL_BASE_URL` unset — real OpenAI is the code's default
target when that variable isn't present
([run_api.py:387-394](../inference/api/run_api.py#L387-L394)).

### Notes for both

- `SWEBENCH_DOCKER_FORK_DIR` must point at wherever *you* cloned this
  repo — the evaluation containers mount that path in, and it's
  silently wrong if it points somewhere else.
- `.env` is already git-ignored. **Never commit a real API key** — each
  team member creates their own `.env` locally from the template.
- If you personally need to call both backends in the same session
  (e.g. testing DeepInfra and OpenAI back to back), don't keep editing
  `.env` — override at the shell for a single command instead:
  ```bash
  # Forces the OpenAI branch for just this one call, regardless of what .env has:
  LOCAL_MODEL_BASE_URL= LOCAL_MODEL_API_KEY= python -m inference.api.run_api ...
  ```
  An explicitly empty value set in the shell takes precedence over
  `.env` and reads as "unset" to the `if local_base_url:` check in
  `run_api.py`.

Sanity-check whichever one you filled in:

```bash
python -c "
import dotenv, os
dotenv.load_dotenv()
print('base url:', os.environ.get('LOCAL_MODEL_BASE_URL') or '(unset -> real OpenAI)')
print('deepinfra key set:', bool(os.environ.get('LOCAL_MODEL_API_KEY')))
print('openai key set:', bool(os.environ.get('OPENAI_API_KEY')))
"
```

---

## 3. Confirm the exact model id

### DeepInfra

DeepInfra's catalog changes over time and the exact id string matters
(it's sent verbatim in every request):

1. Go to `deepinfra.com/models` and find the model you want.
2. Open its **API** tab — it shows the exact id string to use (usually,
   but not always, the same as the model's Hugging Face id, e.g.
   `meta-llama/Meta-Llama-3.1-8B-Instruct`).
3. Use that exact string, copy-pasted, for `--model_name_or_path`
   below.

Don't assume a model from
[pycodekg's `EXPERIMENT_PLAN.md`](../../pycodekg/docs/EXPERIMENT_PLAN.md)
shortlist is available on DeepInfra under the same name — confirm it
on the site first.

### OpenAI

Use the id exactly as OpenAI documents it (e.g. `gpt-4o-2024-05-13`,
`gpt-5`). One real caveat: `run_api.py` only has hardcoded context
window / output limit / per-token cost numbers for a handful of older
OpenAI models (`MODEL_LIMITS`, `OUTPUT_LIMITS`,
`MODEL_COST_PER_INPUT`/`_OUTPUT` near the top of the file). **`gpt-5`
isn't registered there yet** — it'll silently fall back to a
conservative 32K context / 4096-output default and log `$0` cost
instead of erroring. That's not a correctness bug (the request still
works), but it's worth adding a real entry before trusting a GPT-5
run's numbers — see the pitfalls section.

---

## 4. Smoke-test the connection

Before running the real dataset, send one throwaway request through
the same branching logic `run_api.py` itself uses, so the test
actually reflects whichever backend `.env` currently points at:

```bash
cd "<workspace>/testgeneval"
python -c "
import dotenv, openai, os
dotenv.load_dotenv()
base_url = os.environ.get('LOCAL_MODEL_BASE_URL')
if base_url:
    client = openai.OpenAI(api_key=os.environ.get('LOCAL_MODEL_API_KEY', 'not-needed'), base_url=base_url)
    print(f'Using custom endpoint: {base_url}')
else:
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    print('Using real OpenAI')
resp = client.chat.completions.create(
    model='<exact-model-id-from-step-3>',
    messages=[{'role': 'user', 'content': 'Reply with the single word: pong'}],
    max_tokens=5,
)
print(resp.choices[0].message.content)
print('usage:', resp.usage)
"
```

Expected: prints `pong` (or close to it) and a real `usage` line with
non-zero token counts. A 401 means the key is wrong; a 404 on the
DeepInfra branch usually means the trailing slash on the base URL is
missing, or the model id from step 3 is wrong.

---

## 5. Get the evaluation Docker images

Evaluation (running the generated tests, coverage, mutation testing)
happens in containers, regardless of backend. Pull the pre-built
images rather than building locally:

```bash
cd "<workspace>/testgeneval"
python scripts/pull_images.py --makefile Makefile.testgenevallite
```

Use `Makefile.testgeneval` instead for the full (much larger) dataset,
once the small one is working end-to-end.

---

## 6. (Only for the `kg_only` arm) get `kg_prompts.json`

The knowledge-graph context isn't built in this repo — it's built in
the sibling `pycodekg` repo via its own `scripts/build_kg_prompts.py`,
against the same dataset. Get the resulting `kg_prompts.json` (or
`kg_prompts_depth2.json`) from whoever built it, or build it yourself
following `pycodekg`'s own docs, before doing step 8.

The `instruct` arm (step 7) also reads this file, just to know which
function name to focus on — it still runs without it, but then the two
arms are no longer a matched comparison, so don't skip this if you plan
to run both arms for a real comparison.

---

## 7. Generate predictions — baseline (`instruct`) arm

`run_pipeline.py` won't work for a DeepInfra or not-yet-registered
OpenAI model — its `--model` flag only accepts a fixed, pre-registered
list of names. Call the underlying script directly instead (the same
pattern GUIDE.md uses for evaluating predictions generated on M3):

```bash
cd "<workspace>/testgeneval"
MODEL="<exact-model-id-from-step-3>"
DATASET="kjain14/testgenevallite"

mkdir -p "results/instruct/${DATASET##*/}/preds"

python -m inference.api.run_api \
  --dataset_name_or_path "$DATASET" \
  --model_name_or_path "$MODEL" \
  --output_dir "results/instruct/${DATASET##*/}/preds" \
  --prompt_config instruct \
  --kg_prompts_path /path/to/kg_prompts.json \
  --model_args "temperature=0.2" \
  --num_samples 1 \
  --skip_completion \
  --max_cost 5
```

This exact command works unchanged for either DeepInfra or OpenAI —
which one it actually calls is decided entirely by `.env` (step 2),
not by anything here. Just use the right model id from step 3.

What matters here:
- `kjain14/testgenevallite` is the small (160-instance) dataset — start
  here, not the full 1210-instance `kjain14/testgeneval`, until the
  whole pipeline is confirmed working.
- `--skip_completion` skips the `first`/`last`/`extra` settings this
  fork doesn't use — leaving it off wastes time and money generating
  predictions nothing downstream reads.
- `--num_samples 1` is a pass@1 run. For a pass@k sweep later, raise
  this and set a real temperature (`num_samples=1` at `temperature=0`
  is fine; `num_samples>1` at `temperature=0` just pays for identical
  copies).
- `--max_cost 5` is a **soft** dollar cap (in whatever unit `calc_cost`
  uses for this model — see the cost caveat in section 12) — it stops
  submitting new requests once hit, but in-flight ones still finish.
  Start low on a first run.
- The job resumes automatically if interrupted: rerunning the exact
  same command skips any `id` already present in the output file.

---

## 8. Generate predictions — KG-augmented (`kg_only`) arm

Same command, swap the prompt config:

```bash
cd "<workspace>/testgeneval"
MODEL="<exact-model-id-from-step-3>"
DATASET="kjain14/testgenevallite"

mkdir -p "results/kg_only/${DATASET##*/}/preds"

python -m inference.api.run_api \
  --dataset_name_or_path "$DATASET" \
  --model_name_or_path "$MODEL" \
  --output_dir "results/kg_only/${DATASET##*/}/preds" \
  --prompt_config kg_only \
  --kg_prompts_path /path/to/kg_prompts.json \
  --model_args "temperature=0.2" \
  --num_samples 1 \
  --skip_completion \
  --max_cost 5
```

`--kg_prompts_path` is required here — without a real file, `kg_only`
fails outright (it has no fallback, unlike `instruct`).

---

## 9. Evaluate the generated tests

This runs the generated tests in Docker and scores them. It's
identical no matter which of the three backends (M3, DeepInfra,
OpenAI) produced the predictions file — it doesn't touch any API and
costs nothing beyond your own compute.

```bash
cd "<workspace>/testgeneval"
MODEL_NICK="$(basename "<exact-model-id-from-step-3>")"

mkdir -p "results/instruct/data_logs"
python3 run_evaluation.py \
  --predictions_path "results/instruct/testgenevallite/preds/${MODEL_NICK}_t=0.2__testgenevallite__test.jsonl" \
  --log_dir "results/instruct/data_logs" \
  --swe_bench_tasks kjain14/testgenevallite \
  --num_processes 4
```

Repeat with `results/kg_only/...` for the KG-augmented arm.

`--log_dir` must already exist as a real directory — `run_evaluation.py`
does not create it and fails outright if it's missing (empty
directories aren't tracked by git, so a fresh clone always needs the
`mkdir -p`). If you pulled images from Dockerhub in step 5 rather than
building locally, add `--namespace kdjain` here too.

The exact predictions filename follows
`{model_nickname}_t={temperature}__{dataset}__test.jsonl`
(see `basic_args` in
[run_api.py:409-411](../inference/api/run_api.py#L409-L411)) — if
unsure, list the directory instead of retyping it by hand:

```bash
python3 -c "
import glob
print(glob.glob('results/instruct/testgenevallite/preds/*'))
"
```

---

## 10. Read the results

Once evaluation finishes, `results/<dataset>/` has three files per
model:

- `<model>_full.json` — raw per-instance evaluation data.
- `<model>_summary.json` — pass rates and lexical stats rolled up.
- `<model>_report.json` — the aggregated numbers you actually care
  about, including both whole-file and function-scoped coverage/
  mutation score (`full_av_coverage`, `full_av_function_coverage`, and
  the mutation-score equivalents).

Compare the same keys between the `instruct` run and the `kg_only` run
for the headline comparison.

---

## 11. Scaling up: more models, more temperatures, mixed backends

Once one model works end-to-end through steps 7–10, loop over the real
shortlist and temperature sweep (see
[pycodekg's `EXPERIMENT_PLAN.md`](../../pycodekg/docs/EXPERIMENT_PLAN.md)
and [`funding-proposal.md`](funding-proposal.md) for the agreed list
and budget). Example, assuming every model in the loop uses the same
backend (whatever `.env` currently points at):

```bash
cd "<workspace>/testgeneval"
DATASET="kjain14/testgenevallite"

for MODEL in "<model-id-1>" "<model-id-2>" "<model-id-3>"; do
  for TEMP in 0 0.5 1.0; do
    for ARM in instruct kg_only; do
      MODEL_NICK="$(basename "$MODEL")"
      mkdir -p "results/${ARM}/${DATASET##*/}/preds"
      python -m inference.api.run_api \
        --dataset_name_or_path "$DATASET" \
        --model_name_or_path "$MODEL" \
        --output_dir "results/${ARM}/${DATASET##*/}/preds" \
        --prompt_config "$ARM" \
        --kg_prompts_path /path/to/kg_prompts.json \
        --model_args "temperature=${TEMP}" \
        --num_samples 1 \
        --skip_completion \
        --max_cost 5
    done
  done
done
```

If some models in your run go through DeepInfra and others through
OpenAI, don't mix them in one loop over a single `.env` — either run
each backend's models as a separate loop with its own `.env`, or use
the shell-override trick from step 2 per iteration. M3 models don't go
through this loop at all; they're submitted as separate Slurm jobs per
GUIDE.md and their output files land in the same
`results/<arm>/<dataset>/preds/` directory once copied off M3, ready
for the same evaluation and results steps.

Watch actual spend on the DeepInfra dashboard / OpenAI usage page while
this runs, not just `--max_cost` — see the caveat in section 12 about
why the script's own cost figure can't be trusted for every model.

---

## 12. Common pitfalls

- **Every DeepInfra request 404s**: almost always the missing trailing
  slash on `LOCAL_MODEL_BASE_URL`.
- **`Must provide an api key` error**: `.env` isn't being found — check
  you're running from `testgeneval/` (where `.env` lives). If you meant
  to use OpenAI, make sure `LOCAL_MODEL_BASE_URL` isn't left over from
  an earlier DeepInfra test (`echo $LOCAL_MODEL_BASE_URL` in your
  shell, in case it was exported earlier in the session rather than
  just sitting in `.env`).
- **The script's printed "Total Cost" is `0.00` even though you were
  billed**: expected, not a bug, for any model not explicitly added to
  `MODEL_COST_PER_INPUT`/`MODEL_COST_PER_OUTPUT` in
  [run_api.py:112-131](../inference/api/run_api.py#L112-L131) — every
  DeepInfra model, and newer OpenAI models like `gpt-5` that predate
  this file's last update, fall into that bucket. Treat the DeepInfra
  dashboard / OpenAI usage page as the source of truth for real spend,
  not this log line, unless someone adds real per-token rates to those
  dicts first.
- **A GPT-5 (or other unregistered) run looks like it's truncating
  prompts or cutting completions short**: check `MODEL_LIMITS` /
  `OUTPUT_LIMITS` in the same file — an unregistered model silently
  uses a conservative 32K-context / 4096-output default rather than
  the model's real limits. Add a real entry once you know the actual
  numbers.
- **`--log_dir must exist and point at a directory`**: run the
  `mkdir -p` from step 9 first; it's not created automatically.
- **`AttributeError: _ARRAY_API not found`** running `run_evaluation.py`:
  you're not in the `testgeneval` conda env (see section 0).
- **`kg_only` run fails outright with no output**: missing or wrong
  `--kg_prompts_path` — this arm has no fallback, unlike `instruct`.
- **Model works in a playground/dashboard but the script gets a
  400/404**: re-check the exact id string from step 3 — it's case- and
  slash-sensitive, and DeepInfra's catalog changes over time.

---

## 13. Where to go for more detail

- [GUIDE.md](GUIDE.md) — the full fork guide, including the complete
  M3/Slurm instructions this doc points to but doesn't repeat.
- [EXPERIMENT_PLAN.md in pycodekg](../../pycodekg/docs/EXPERIMENT_PLAN.md)
  — the model shortlist, temperature sweep, and statistical comparison
  plan this is all in service of.
- [funding-proposal.md](funding-proposal.md) — why some models are
  priced as paid API infrastructure (DeepInfra/Mistral/OpenAI) and
  others run free on M3.
- [../../PROJECT-OVERVIEW.md](../../PROJECT-OVERVIEW.md) and
  [../../EXPERIMENT-RUNBOOK.md](../../EXPERIMENT-RUNBOOK.md) — how this
  repo fits into the wider three-repo project.
