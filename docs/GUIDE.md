# Guide to this testgeneval fork

This is a fork of Meta's [TestGenEval](https://github.com/facebookresearch/testgeneval)
benchmark, adapted for a specific comparison: a plain LLM baseline
(`instruct`) against an LLM given structural context pulled from a
knowledge graph (`kg_only`), both asked to generate a complete test file
from scratch for a changed function. The original README still describes
the original benchmark's four settings and general-purpose design; this
doc describes how the fork actually works today, and how to run it.


## What's different in this fork

Upstream TestGenEval supports four settings: `full` (generate a whole
test file) and three completion settings (`first`/`last`/`extra`, each
asking for one more test given part of an existing file). This fork only
uses `full`. The completion settings still exist in the code but nothing
in the current experiment design touches them.

There's also a second axis this fork adds: which *prompt strategy* builds
the `full` prompt. Two are supported, selected with `--prompt_config`:

- **`instruct`** — the original approach. Shows the model the whole code
  file as flat text, plus a line naming the specific function the patch
  changed, so it knows where to focus.
- **`kg_only`** — shows the model a knowledge-graph-derived view of the
  same target function instead of the raw file: the function's own
  source and docstring, its callers, callees, sibling methods, and any
  classes it inherits from or instantiates, all presented as labelled
  sections rather than a flat dump. This content is pre-computed outside
  this repo, by a companion project called `pycodekg`, and handed to
  this repo as a JSON file (`kg_prompts.json`).

Both arms are told to focus on the same function, and neither is shown
any existing test content, the comparison is meant to isolate
*representation* (structured KG sections vs. a flat file), not have the
two arms secretly doing different jobs. If you want the full reasoning
behind that design, it's written up in the `pycodekg` repo's
`docs/EXPERIMENT_PLAN.md`.

Evaluation has also grown a second pair of metrics. Coverage and
mutation score were originally computed over the whole file. This fork
adds function-scoped versions of both, restricted to just the lines of
the function the patch actually touched, because `kg_only` can
structurally only write tests for the function it was shown, while
`instruct` has the whole file in front of it and could pick up
incidental credit elsewhere. Both the whole-file and function-scoped
numbers get written to the results; which one to treat as the headline
number is still an open call.

## Setting up

Clone the repo and create the conda environment from the pinned
dependency file:

```bash
git clone <this fork's URL>
cd testgeneval
conda env create -f testgeneval.yaml
conda activate testgeneval
```

Copy `.env_template` to `.env` and fill it in. `SWEBENCH_DOCKER_FORK_DIR`
specifically needs to point at wherever you cloned this repo — the
evaluation containers mount that path in, so if it's wrong, evaluation
silently can't find its own code.

## Getting the docker images

Evaluation runs each generated test inside a container that already has
the right Python version, the target repo checked out, and coverage/
mutation tooling installed. You need these images before you can
evaluate anything.

Easiest path: pull pre-built images instead of building them yourself:

```bash
python scripts/pull_images.py --makefile Makefile.testgenevallite
```

(swap in `Makefile.testgeneval` for the full, much larger dataset). If
you'd rather build locally instead of pulling, `make -f
Makefile.testgenevallite` does that, but expect it to take a while —
building the full dataset's images can run the better part of a day.

## Running a comparison

`run_pipeline.py` drives both prediction generation and evaluation in
one call. A baseline `instruct` run looks like:

```bash
python run_pipeline.py \
  --results_dir results \
  --dataset_name_or_path kjain14/testgenevallite \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --skip_completion
```

`--skip_completion` matters — without it, the pipeline also tries to run
the three completion settings this fork isn't using, which just wastes
time generating predictions nothing downstream reads.

For the `kg_only` arm, you need `kg_prompts.json` already built (that's
`pycodekg`'s job, via its own `scripts/build_kg_prompts.py`, run from
that repo against the same dataset), then point at it:

```bash
python run_pipeline.py \
  --results_dir results \
  --dataset_name_or_path kjain14/testgenevallite \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --skip_completion \
  --prompt_config kg_only \
  --kg_prompts_path /path/to/kg_prompts.json
```

Worth knowing: `--kg_prompts_path` gets read by *both* arms, not just
`kg_only`. `instruct` reads it too, just for the target function's name,
so it can be told to focus on the same function `kg_only`'s subgraph is
built around. If you're running `instruct` on its own without that file,
it just falls back to generic, unfocused wording — it won't error out,
but the two arms stop being a matched comparison.

If you built images locally rather than pulling them, that's all you
need. If you pulled from Dockerhub, add `--namespace kdjain` so the
pipeline looks for images under the right account.

## Evaluating predictions already generated elsewhere (e.g. on M3)

`run_pipeline.py`'s `--model` flag only accepts a fixed, pre-registered
list of model names. Most of the models actually used for M3 inference
(`Qwen/Qwen2.5-Coder-7B-Instruct`, `deepseek-ai/DeepSeek-Coder-6.7B-Instruct`,
`bigcode/starcoder2-15b-instruct-v0.1`, and so on) aren't in that list,
so `run_pipeline.py` rejects them outright, even though the predictions
file itself is already sitting there ready to evaluate.

Use `run_evaluation.py` directly instead, it's model-agnostic, no
`--model` choices restriction, and just reads a predictions file:

```bash
mkdir -p results/instruct/data_logs
python3 run_evaluation.py \
  --predictions_path results/instruct/<model>__testgeneval__0__test.jsonl \
  --log_dir results/instruct/data_logs \
  --swe_bench_tasks kjain14/testgeneval \
  --num_processes 4
```

`--log_dir` has to already exist as a real directory, `run_evaluation.py`
doesn't create it and fails with `--log_dir must exist and point at a
directory` otherwise. It's not part of the repo (empty directories
aren't tracked by git), so a fresh clone always needs the `mkdir -p`
first, even if it already exists in someone else's checkout of the same
repo.

`--swe_bench_tasks` takes the same dataset name used for inference
(`kjain14/testgenevallite` or `kjain14/testgeneval`). If you pulled
images from Dockerhub rather than building locally, pass `--namespace
kdjain` here too, same account as `pull_images.py`/`run_pipeline.py`.

Run this from the project's own conda env (`conda activate
testgeneval`), not system Python. A stock/Anaconda base Python's numpy
and pandas can have an ABI mismatch that breaks this script's imports
with a confusing, unrelated-looking error (`AttributeError: _ARRAY_API
not found`) rather than anything mentioning the real cause.

## Reading the results

Once a run finishes, `results/<dataset>/` has three files per model:

- `<model>_full.json` — the raw, per-instance evaluation data.
- `<model>_summary.json` — pass rates and some lexical stats (generated
  test length, method count) rolled up.
- `<model>_report.json` — the aggregated numbers, one row per metric.
  This is where both the whole-file and function-scoped coverage/
  mutation figures show up, as separate keys
  (`full_av_coverage`/`full_av_function_coverage`, and similarly for
  mutation score).

## Running inference on Monash M3

Inference for the actual experiment runs on Monash's M3 HPC cluster,
since it has the GPUs and this fork's API-based inference path
(`inference/api/run_api.py`) can talk to a local model server just as
easily as a hosted API. M3 doesn't support Docker, though, so evaluation
has to happen somewhere else, generate predictions on M3, copy them
off, then run `run_evaluation.py` (or the rest of `run_pipeline.py`) on
a machine that has Docker.


### Clone the repo on M3

Do this from the login node.

```bash
cd ~/al49_scratch
mkdir kg-testing
cd kg-testing
git clone https://github.com/miggle711/testgeneval.git
cd testgeneval
```

Put this in scratch space (`al49_scratch`), not your home directory.
Home has a small quota, and this repo plus its dependencies and
downloaded models add up to a lot.

### Build the conda environment (one time)

**`m3_setup_env.slurm`** builds the conda environment the inference job
expects, installing vLLM and this repo's actual runtime dependencies on
a GPU node (vLLM's install needs a GPU to detect CUDA capabilities
correctly, so it can't happen on the shared login node). Submit it with:

```bash
sbatch m3_setup_env.slurm
```

Check on it with `squeue -u <your-username>`, and once it's running,
watch the log with `tail -f slurm-<jobid>.out`. It's done when you see
`Env testgeneval-vllm ready. m3_run_inference.slurm can now use it.`
This takes a few minutes, mostly spent downloading and installing vLLM.

You only need to do this once. If the environment ever gets into a bad
state, just rerun `sbatch m3_setup_env.slurm`, it removes the old
environment first, so it's safe to rerun.

Double check:

```bash
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate testgeneval-vllm
which vllm
```

This should print a real path like
`/home/<username>/.conda/envs/testgeneval-vllm/bin/vllm`. If it prints
nothing, rerun `m3_setup_env.slurm`.

### Running inference on one job

**`m3_run_inference.slurm`** is the actual inference job. It starts vLLM
serving your chosen model locally on the node, waits for it to come up,
then calls `run_api.py` against it exactly the way the local or hosted-API
paths do. Configure it through environment variables rather than editing
the script:

```bash
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" \
DATASET_PATH="kjain14/testgenevallite" \
PROMPT_CONFIG="instruct" \
TEMPERATURE="0" \
sbatch m3_run_inference.slurm
```

`MODEL` needs to be a real HuggingFace model id, and should be one of
the actual shortlisted models (see pycodekg's `docs/EXPERIMENT_PLAN.md`).
`DATASET_PATH` can be a HuggingFace dataset name, which vLLM downloads
directly, or a local path to a dataset saved to disk and transferred
over. `PROMPT_CONFIG` is `instruct` or `kg_only`; `kg_only` also needs
`KG_PROMPTS_PATH` pointing at a real `kg_prompts.json`, built separately
in the pycodekg repo. `TEMPERATURE` is whatever's being tested in the
sensitivity sweep (0, 0.5, 1).

By default requests go to vLLM one at a time, `MAX_CONCURRENCY` (default
1) raises that, so vLLM's continuous batching actually has multiple
requests to work with instead of sitting mostly idle between them. Keep
it at or below the script's own `--max-num-seqs 8` passed to `vllm
serve` -- that's vLLM's hard ceiling on concurrent sequences regardless
of what's requested. Watch vLLM's own `GPU KV cache usage` log line
after raising it; that's the real signal for how much headroom is left,
not a fixed safe number for every model/GPU combination.

```bash
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" \
DATASET_PATH="kjain14/testgenevallite" \
PROMPT_CONFIG="instruct" \
TEMPERATURE="0" \
MAX_CONCURRENCY="8" \
sbatch m3_run_inference.slurm
```

Both scripts default `HF_HOME` to project scratch space rather than your
M3 home directory. Specifically, each user gets their own subfolder
under the shared `al49_scratch`, since that directory is shared across
the whole project, not private to you, and another teammate's cache
might already exist there without write access for you.

Watch the job the same way as setup. It takes roughly 45 seconds per
instance at a typical pace, so the full 160 instance testgenevallite
dataset is a couple of hours in one job. When it's done you'll see
`Done. Predictions written under ...`, and the output lands in
`results/instruct/` (or `results/kg_only/`), named something like
`Qwen2.5-Coder-7B-Instruct__testgenevallite__0__test.jsonl`.

### Running inference sharded, across multiple parallel jobs

The full TestGenEval dataset (1210 instances, not the 160 instance lite
one) would take roughly 16 hours in a single job, a long time to have
riding on one job. Sharding splits the dataset into pieces and runs each
piece as its own job, in parallel. Confirmed on real M3 that this
project can run multiple GPU jobs at once, there's no hard limit
stopping this (checked both by watching several jobs run simultaneously
in `squeue`, and via `sacctmgr show assoc`, which showed no meaningful
job concurrency cap for this account).

Submit the same command several times, changing only `SHARD_ID` each
time and keeping `NUM_SHARDS` the same across all of them:

```bash
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" DATASET_PATH="kjain14/testgenevallite" PROMPT_CONFIG="instruct" TEMPERATURE="0" NUM_SHARDS="4" SHARD_ID="0" sbatch m3_run_inference.slurm
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" DATASET_PATH="kjain14/testgenevallite" PROMPT_CONFIG="instruct" TEMPERATURE="0" NUM_SHARDS="4" SHARD_ID="1" sbatch m3_run_inference.slurm
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" DATASET_PATH="kjain14/testgenevallite" PROMPT_CONFIG="instruct" TEMPERATURE="0" NUM_SHARDS="4" SHARD_ID="2" sbatch m3_run_inference.slurm
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" DATASET_PATH="kjain14/testgenevallite" PROMPT_CONFIG="instruct" TEMPERATURE="0" NUM_SHARDS="4" SHARD_ID="3" sbatch m3_run_inference.slurm
```

Each shard processes only its own slice (with 4 shards on the 160
instance lite dataset, that's 40 instances each) and writes its own
output file with a suffix showing which shard it is, something like
`Qwen2.5-Coder-7B-Instruct__testgenevallite__0__test__shard-0__num_shards-4.jsonl`.

Check `squeue -u <your-username>` and confirm several rows actually show
`R` on different nodes at once, not queued up behind each other.

Once every shard finishes, merge them with `scripts/merge_and_validate.py`
instead of a manual `cat`:

```bash
python3 scripts/merge_and_validate.py \
  --output_dir results/instruct \
  --model_nickname Qwen2.5-Coder-7B-Instruct \
  --dataset testgenevallite \
  --temperature 0 \
  --num_shards 4 \
  --expected_total 160
```

It refuses to merge if any shard file is missing, so it never produces a
silent partial file. It also validates the result afterward: total line
count, plus a check that every `id` is unique, which catches corrupted
or interleaved writes (something that could in principle happen under
`MAX_CONCURRENCY > 1` if two requests write at the same instant).
`--expected_total` is optional and only prints a note, not an error,
when the count differs. Context overflow losses are a real, expected
outcome for some models, not a bug, see `results/RUN_LOG.md`.

Don't set `DELETE_MODEL_AFTER_RUN=1` when running several shards of the
same model in parallel. One shard finishing and deleting the model
cache would break the other shards still running.

### Getting the results off M3

Pull the predictions file down with `scp` from your own machine, not
from inside an M3 SSH session:

```bash
scp <username>@m3.massive.org.au:/home/<username>/al49_scratch/kg-testing/testgeneval/results/instruct/Qwen2.5-Coder-7B-Instruct__testgenevallite__0__test.jsonl .
```

The dedicated transfer node (`m3-dtn.massive.org.au`) is meant for this
and worth trying first. If it gives a `Connection closed` error for no
obvious reason, falling back to the regular login node like above works
fine too, at least for a file this size.

### Worth knowing

The inference script tracks which instances it's already completed
based on the output file. If a job dies partway through or hits its
time limit, resubmit the exact same command and it picks up where it
left off instead of starting over, as long as the earlier output file
is still there.

The larger models in the planned lineup (the 70B/72B tier) don't fit on
a single M3 GPU at full precision. Set `TENSOR_PARALLEL_SIZE` to split
the model across multiple GPUs in one job, and pass a matching
`--gres=gpu:L40S:N` on the `sbatch` command line, since `#SBATCH`
directives can't read shell/env vars. See the script's own header
comment for the exact pattern.

**New teammate on the shared `al49_scratch` directory, jobs die instantly
with no output file at all?** Confirmed real (twice, for two different
teammates) that being added to the `al49` SLURM account/association
(`sacctmgr show associations user=<username>`) does *not* automatically
grant write access to files already in the shared `kg-testing/testgeneval`
directory. A job that can't even create its own `slurm-<jobid>.out`
fails in under a second with exit code `0:53` and produces zero output
anywhere, easy to mistake for a conda-env or script bug rather than a
permissions issue. Confirm directly with `touch
/fs04/scratch2/al49/kg-testing/testgeneval/test_<username>.txt` before
chasing anything else, if that fails with `Permission denied`, the fix
is someone who already has write access running:

```bash
setfacl -R -m u:<new-username>:rwx /fs04/scratch2/al49/kg-testing/testgeneval
setfacl -R -d -m u:<new-username>:rwx /fs04/scratch2/al49/kg-testing/testgeneval
```

(`-R` applies to existing files/directories, the second `-d` command sets
the *default* ACL so new files created afterward also inherit access.)
Expect `Operation not permitted` on some files during this, that's normal
for files owned by other teammates you don't have permission to modify,
not a sign the fix failed, what matters is whether new file creation
works afterward.

Output filenames have several `__` (double underscore) separators
between fields (model name, dataset, temperature, split, shard info).
These are easy to mangle by hand, retyping or copy-pasting across some
terminals silently collapses or drops repeated underscores. Prefer
`glob` over typing the filename out:

```bash
python3 -c "
import glob
print(glob.glob('results/instruct/*<distinctive-part-of-model-name>*'))
"
```

instead of constructing the exact filename yourself.

## Adding a new model

If a model isn't in `run_pipeline.py`'s `--model` choices list, you'll
need to add it there. Beyond that, as long as it's reachable through the
existing inference paths (a hosted API, or a local server exposing an
OpenAI-compatible endpoint the way vLLM does), no other code changes are
needed — both `instruct` and `kg_only` prompt classes are shared across
every model.

## Where things live

- `inference/api/run_api.py` — the actual inference entry point, called
  by both `run_pipeline.py` and the M3 SLURM script.
- `inference/configs/instruct_prompt.py` /
  `inference/configs/kg_only_prompt.py` — the two prompt strategies.
- `swebench_docker/` — evaluation: coverage, mutation testing, and the
  Docker orchestration.
- `creation/` — the scripts that originally built this benchmark from
  SWE-bench; not something you need to touch to run a comparison.
