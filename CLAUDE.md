# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of Meta's [TestGenEval](https://github.com/facebookresearch/testgeneval) benchmark,
adapted for one specific comparison: a plain LLM baseline (`instruct`) against an LLM given
structural context pulled from a knowledge graph (`kg_only`), both asked to generate a complete
test file from scratch for a changed function. See `docs/GUIDE.md` for the full run workflow —
it's the primary operational doc for this fork and should be read before running anything here.
`results/RUN_LOG.md` is the single source of truth for what's actually been run: per-model
completion counts, real failure causes, and job IDs — always check it before assuming a model's
status. The upstream `README.md` still describes the original four-setting benchmark design; this
fork only uses the `full` setting.

The `kg_only` arm's prompts are pre-computed outside this repo by a companion project,
`pycodekg` (repo `repo-kg-construction`), and handed to this repo as a JSON file
(`kg_prompts.json`). This repo does not build knowledge graphs itself. `docs/EXPERIMENT_PLAN.md`
(copied into this fork from `pycodekg`, its canonical source) describes the full cross-repo
design and the locked model shortlist. `docs/funding-proposal.md` extends that shortlist with
newer models the team is adding.

## Commands

Environment: `conda env create -f testgeneval.yaml && conda activate testgeneval`. Copy
`.env_template` to `.env` and set `SWEBENCH_DOCKER_FORK_DIR` to this repo's absolute path —
evaluation containers mount that path in, and get it wrong silently if the var is unset or wrong.

Tests use plain `unittest`, not pytest (pytest isn't a dependency of this repo):

```bash
python -m unittest swebench_docker.test_target_range -v
python -m unittest swebench_docker.test_swebench_utils -v
python -m unittest inference.api.test_shard_ordering -v
python -m unittest inference.api.test_concurrent_inference -v
```

Run one test method: `python -m unittest swebench_docker.test_target_range.TestResolveTargetLineRange.test_single_function_patch`

Full comparison run (generation + evaluation in one call):

```bash
python run_pipeline.py --results_dir results --dataset_name_or_path kjain14/testgenevallite \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --skip_completion
```

`--skip_completion` is required — without it the pipeline also runs the three completion
settings (`first`/`last`/`extra`) this fork doesn't use. For the `kg_only` arm, add
`--prompt_config kg_only --kg_prompts_path /path/to/kg_prompts.json` (built separately by
`pycodekg`). `--kg_prompts_path` is also read by `instruct` (for target-function naming only),
so pass it there too when running a matched comparison.

**Real inference at scale runs on M3 via `m3_run_inference.slurm`**, not `run_pipeline.py`
locally — M3 is where the team actually has GPU access. Configure via env vars, not by editing
the script:

```bash
MAX_CONCURRENCY=8 MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" DATASET_PATH="kjain14/testgeneval" \
  PROMPT_CONFIG="instruct" TEMPERATURE="0" sbatch --gres=gpu:L40S:1 --time=8:00:00 m3_run_inference.slurm
```

Key env vars, see the script's own header comment for the full list: `MAX_CONCURRENCY` (requests
in flight at once, default 1/sequential — real speedup, ~6-7x, confirmed on a 70B model going
from ~130s/instance to ~21s/instance at `MAX_CONCURRENCY=8`), `MAX_NUM_SEQS` (vLLM's own ceiling
on concurrent sequences, `MAX_CONCURRENCY` can never exceed this, default 8 — calibrate this per
model before raising it, see the "Known M3 gotchas" entry on `GPU KV cache usage` below, a value
confirmed safe for one model is not safe to assume for another similarly-sized one), `DTYPE` (default
`float16`, some architectures/quantization schemes reject it — confirmed real for
`google/gemma-3-4b-it` (needs `bfloat16`) and separately for any `gpt-oss-*` model
(`torch.float16 is not supported for quantization method gpt_oss_mxfp4. Supported dtypes:
[torch.bfloat16]`, same fix, `DTYPE=bfloat16`)), `TENSOR_PARALLEL_SIZE` (splits a model across N GPUs in one job, for models too
large for one GPU — must also pass a matching `--gres=gpu:L40S:N` on the `sbatch` command line,
since `#SBATCH` directives can't read env vars), `SHARD_ID`/`NUM_SHARDS` (split the dataset
across parallel jobs — largely superseded by `MAX_CONCURRENCY` now, since a single unsharded job
finishes in a few hours rather than needing 8-way parallelism, though sharding is still viable
for extra resilience against a mid-run crash).

Merge and validate a sharded run's output instead of a manual `cat` + `wc -l`:

```bash
python3 scripts/merge_and_validate.py --output_dir results/instruct \
  --model_nickname Qwen2.5-Coder-7B-Instruct --dataset testgeneval --temperature 0 \
  --num_shards 8 --expected_total 1210
```

Refuses to merge if any shard is missing, validates the result afterward (unique ids, matches
`--expected_total` if given — a mismatch there is just a note, not an error, since context
overflow losses are expected for some models).

Evaluating predictions already generated elsewhere (e.g. inference run on M3, since M3 doesn't
support Docker):

```bash
mkdir -p results/instruct/data_logs   # must exist first, run_evaluation.py won't create it
python3 run_evaluation.py --predictions_path results/instruct/<model>__testgeneval__0__k1__test.jsonl \
  --log_dir results/instruct/data_logs --swe_bench_tasks kjain14/testgeneval --num_processes 4
```

Add `--backend apptainer` to run evaluation directly on M3 instead of locally (confirmed
producing bit-for-bit identical results to Docker for a real instance, see `RUN_LOG.md`). Needs
a `.sif` file already built and present at `APPTAINER_IMAGES_DIR/{repo}_{version}.sif` — M3
itself can't build/pull these (`apptainer build`/`pull` from Docker Hub needs `sudo`, which
regular M3 accounts don't have), so they have to be built elsewhere (e.g. via `docker run --rm
--privileged -v "$PWD:/output" quay.io/singularity/singularity:v3.11.4 build ...`) and
transferred in.

Docker images (needed before evaluating anything locally) — pull rather than build locally,
unless you have the better part of a day:

```bash
python scripts/pull_images.py --makefile Makefile.testgenevallite   # or Makefile.testgeneval for full
```

## Architecture

**Two independent halves that only meet at the predictions JSONL file.** Inference
(`inference/`) generates predictions; evaluation (`swebench_docker/`) scores them by running the
generated tests inside per-repo containers (Docker or Apptainer, see below). They can run on
different machines entirely — this is the actual working setup: inference on Monash's M3 HPC (no
Docker support), evaluation locally where Docker is available, or increasingly directly on M3 via
the Apptainer backend.

**Prompt strategy is a pluggable class, not a branch.** `inference/configs/instruct_prompt.py`
and `inference/configs/kg_only_prompt.py` both implement `add_prompts_to_dataset`,
`postprocess_output`, `system_message` — `inference/api/run_api.py` picks one via
`--prompt_config` and treats them identically after that. Both arms are told to focus on the
same target function (via `kg_prompts.json`'s `target_functions`/`target_classes`, read by both
prompt classes) and neither is shown existing test content — the comparison isolates
*representation* of the code context, not different task framings. Adding a new model needs no
changes to either prompt class, only registering it (if not already reachable via `--model_name_or_path`,
which accepts any real HuggingFace model id for the local/M3 vLLM path) and, for `run_pipeline.py`
specifically, its `--model` choices list.

**Inference requests run concurrently, not sequentially, via `run_api.py`'s `--max_concurrency`.**
A `ThreadPoolExecutor` drives up to N requests in flight at once, each worker thread gets its own
cached `openai.OpenAI` client (never mutates the shared `openai.api_key`/`base_url` module
globals, which would race across threads), and output-file writes are lock-protected to prevent
corrupted/interleaved lines. Defaults to 1 (fully sequential, unchanged from the original
behavior) unless explicitly raised.

**Sharding shards the dataset before filtering completed ids, not after** — see the comment at
`inference/api/run_api.py`'s `main()` shard/filter block. Filtering first would let each shard's
already-different resume point change the dataset size differently before `dataset.shard(...,
contiguous=True)` computes slices, drifting shard boundaries out of alignment and producing
overlapping/gapped shards across a resumed run.

**Evaluation runs two scoring passes per instance**, both over the same container invocation:
whole-file coverage/mutation score (the original TestGenEval metrics), and a function-scoped
variant restricted to the lines the patch actually touched (`swebench_docker/target_range.py`
resolves the target line range from the patch; `context_manager.py`'s
`_log_function_coverage`/`_log_function_mutation_score` compute the scoped numbers). The
function-scoped pair is the intended primary metric for the project's research questions (see
`docs/EXPERIMENT_PLAN.md`'s Stage 5) since `kg_only` can structurally only write tests for the
function it was shown, while `instruct` has the whole file and could pick up incidental credit
elsewhere; whole-file numbers are kept as secondary/contextual.

**Context-overflow losses are silent by design, not a bug.** When a model's real context window
can't fit `instruct`'s flat-file prompt, `run_api.py`'s retry logic burns its budget against a
guaranteed-to-fail request, then skips the instance with no record in the output file itself
(only a log line). `results/RUN_LOG.md` tracks known per-model completion counts and loss causes
separately for exactly this reason — a raw completed-count is not enough to tell "ran clean" from
"lost instances to context overflow" when comparing models. Comparing bare pass percentages
across models with different completion counts is the real fairness trap, not the losses
themselves — see `RUN_LOG.md`'s "Two different fairness questions" section for the full argument.

**`context_manager.py`'s mutation testing runs `cosmic-ray` as a subprocess inside the
per-instance conda testbed env** (`self.cmd_conda_run`), never via direct import — `cosmic_ray`
is not installed in the outer environment `evaluate_instance.py` itself runs in. Any code path
that needs to inspect `cosmic-ray`'s output (e.g. `mutation.sqlite`) must go through that same
subprocess pattern, not `import cosmic_ray` directly. Also wrapped in a broad try/except with a
`FunctionMutationFAIL` log fallback, so a failure in this bonus metric can't take down the whole
instance's evaluation the way it originally did.

**`swebench_docker/run_apptainer.py` mirrors `run_docker.py`'s signature**, selectable via
`run_evaluation.py --backend apptainer`. Real differences from a naive `docker run` ->
`apptainer` translation, each confirmed against a live M3 job: `apptainer run <sif>` doesn't
work (the OCI-to-runscript conversion mangles an absolute Dockerfile `ENTRYPOINT` into a relative
one), use `apptainer exec <sif> <real-entrypoint-path>` instead; the entrypoint's real path
differs by base image (`/opt/entrypoint.sh` for pyenv-based repos, `/home/swe-bench/entrypoint.sh`
for conda-based ones — `PYENV_REPOS` in `constants.py` already distinguishes these); Apptainer
doesn't honor the image's Dockerfile `WORKDIR`, needs explicit `--pwd`.

## Local environment notes

- **`gh` is not on PATH in the sandboxed shell used here**, even though it's installed via
  Homebrew. Use the full path, `/opt/homebrew/bin/gh`, for any `gh` command (PR/issue creation,
  `gh pr view`, etc.) instead of assuming it resolves.
- **A `testgeneval` conda env with this repo's real dependencies already exists locally**, at
  `/opt/anaconda3/envs/testgeneval`. The sandboxed shell's plain `python3`/`conda` don't have
  `datasets`, `tiktoken`, etc. installed and `conda activate` doesn't reliably work in this
  shell either, so call the env's interpreter directly by full path, e.g.
  `/opt/anaconda3/envs/testgeneval/bin/python3 -m unittest inference.api.test_pass_at_k -v`,
  rather than assuming the default `python3` has this repo's deps.

## Known M3 gotchas (not specific to this repo's code, but easy to lose hours to)

- **A teammate submit-only (no commits) sharing another teammate's clone hits git's "dubious
  ownership" safety check on any git command**, since the directory isn't owned by their own
  account. Confirmed real 2026-08-29: `jliu0290` running `git status`/`git pull` inside
  `mvar0010`'s clone got `fatal: detected dubious ownership in repository`. This is different
  from the actual-commit-conflict gotcha below (that one is unfixable without per-object
  grants); this one is just a git safety default, fixed once per teammate via
  `git config --global --add safe.directory <the shared clone's absolute path>` (their own
  global config, doesn't affect the owner or other teammates). This is the confirmed-safe
  pattern for read-only/submit-only sharing (running `sbatch` against someone else's clone
  without ever committing into it yourself). See the shared-clone-vs-commit-conflict gotcha
  immediately below for when a separate clone is required instead.
- **Shared team directories don't inherit write access from SLURM account membership.** Being
  added to `al49`'s SLURM association doesn't grant filesystem ACL access to files already in a
  shared directory — confirmed real for two different teammates, jobs died in under a second with
  zero output and exit code `0:53`. Fix: `setfacl -R -m u:<username>:rwx <dir>` plus the `-d`
  variant for new files, run by someone who already has access.
- **A shared git checkout with commits from multiple accounts breaks `git pull`/`fetch` for
  everyone but each object's original owner** — `.git/objects` files get created owned by
  whoever committed them, no default ACL. Not fixable with `setfacl` (would need per-object
  grants from each owner). Clone fresh into your own directory instead. Confirmed real
  2026-09-16: also blocks *committing*, not just pulling (`fatal: could not open
  '.git/COMMIT_EDITMSG': Permission denied`, owned by another teammate's prior commit attempt
  in the same shared clone) — `rm -f .git/COMMIT_EDITMSG` let git recreate it fresh and
  unblocked that one file, but the underlying problem recurred on the next `git pull`
  (`insufficient permission for adding an object to repository database .git/objects`).
  `git gc` does NOT fix this either — confirmed real: it repacked the objects you own, but
  couldn't unlink the newly-redundant loose copies of objects you don't own (`warning: unable
  to unlink ...: Permission denied`) and separately failed on a ref owned by someone else
  (`cannot lock ref ...: Permission denied` on another teammate's branch), and the very next
  `git pull` hit the identical `insufficient permission` error since new incoming objects still
  need to land in a tree you don't fully own. Real prevention: shared clones (named
  `testgeneval_<username>_main` or similar) should be submit-only from every account except the
  one that created them — never `git add`/`commit` there, only `sbatch`/read predictions. Anyone
  who needs to commit gets their own clone (`testgeneval_<username>_own`).
- **`load_dataset`'s cache defaults to `$HOME`, not scratch space**, and `$HOME` on M3 has its
  own small, easy-to-exhaust quota that `lfs quota` can't even inspect (not a Lustre filesystem).
  Set `HF_DATASETS_CACHE` to somewhere under `al49_scratch`, same idea as the existing `HF_HOME`
  fix for model weights.
- **M3 has GPU types beyond the default L40S** (`--gres=gpu:L40S:N`) — the `m3h` partition has
  H100 nodes (80GB each), useful for models too large to fit the account's normal 4x L40S (192GB)
  quota. Needs both `--partition=m3h` and `--qos=m3h` together (either alone fails).
- **`run_evaluation.py --log_dir` has to already exist as a real directory** — it doesn't create
  it and fails with an unclear-looking error otherwise (`mkdir -p` first). Not part of the repo
  since empty directories aren't tracked by git, so a fresh clone always needs this.
- **`/fs04` disk pressure causes confusing, unrelated-looking failures**, not an obvious "disk
  full" error — confirmed real: a run at 99% filesystem capacity produced 2625 `Connection
  error`s (the vLLM server itself became unreachable) and separately a `PermissionError` on file
  writes. If a job fails with something that doesn't obviously explain itself, check
  `lfs quota -g al49 /fs04` before chasing the error message literally.
- **`hf cache rm` can report "not found" for a model that's genuinely cached**, at least once
  before succeeding on a retry with the same command — root cause never fully pinned down, but
  re-running with a consistent `HF_HOME` across `list`/`rm` calls has worked every time this
  happened. Don't trust a single "not found" as proof the cache is actually empty.
- **`al49_scratch`'s 3.0TB allocation can fill up team-wide, not just per-user** — confirmed real
  2026-08-29: `df -h ~/al49_scratch` showed 100% used (0 available) mid-job, blocking a model
  download with `UserWarning: Not enough free disk space`. Root cause that time was
  `repo-kg-construction/kg_output/` (204GB, the raw per-commit KG files from a full-dataset batch
  build) left in place after the two `kg_prompts_depthN.json` files it fed into were already
  built — safe to delete once those exist, regenerable via the same `sbatch` scripts if ever
  needed again. Check `du -sh ~/al49_scratch/*/` team-wide (not just your own subdirectory) when
  this happens, the pressure can come from anyone's checkout.
- **vLLM's `GPU KV cache usage` log line is the real ground truth for how much concurrency
  headroom a model actually has**, not a number to guess at, and not something safe to
  extrapolate from model size or from another model's confirmed value. Confirmed real
  2026-08-29 on three models, each calibrated separately with `NUM_SAMPLES=5` at temperature
  0.8:
  - Qwen2.5-Coder-7B-Instruct (one L40S): safe at `MAX_NUM_SEQS=32` (stable 20-25% KV cache,
    no spikes), and the script's default of 8 was barely used at all (~6.4%-8.5%). Not yet
    tested above 32; likely still has headroom.
  - Qwen3-Coder-30B-A3B-Instruct (2x L40S, `TENSOR_PARALLEL_SIZE=2`): safe at `MAX_NUM_SEQS=32`
    but closer to its real ceiling, oscillating 41%-60% under sustained load. Don't assume
    much more headroom above 32 for this one without retesting.
  - Meta-Llama-3.1-8B-Instruct (one L40S): **`MAX_NUM_SEQS=32` is not safe** for this model,
    despite being the same size class as Qwen2.5-Coder-7B-Instruct above. It spiked to 97.3%
    KV cache usage, real near-OOM risk, driven by some very long completions (one hit
    `output_tokens=20480`) clustering together in the same batch window. Confirmed safe at
    `MAX_NUM_SEQS=24` instead (stable 30-55%, no spikes). This is the concrete case that
    disproves "similarly-sized models need similar `MAX_NUM_SEQS`" as a shortcut, output-length
    distribution on the actual dataset matters as much as parameter count.

  Always calibrate per model with a short run on `kjain14/testgenevallite` (no need to let it
  finish, just watch the log for a few minutes) before trusting a value for a real production
  run; treat anything approaching ~85-90% as the real danger zone for an OOM crash mid-run, and
  watch for spikes, not just a snapshot reading, since KV cache usage can climb well past an
  early reading once enough long-output requests land in the same window.
- **Two jobs landing on the same M3 node can silently cross-talk if neither sets `VLLM_PORT`
  explicitly**, since it defaults to `8003 + ${SHARD_ID:-0}` and neither job had a different
  `SHARD_ID`. Confirmed real 2026-08-29: two separate `sbatch` submissions (gpt-oss-20B and
  gpt-oss-120B, both without `VLLM_PORT` set) both landed on node `m3h101` and both defaulted to
  port 8003. One job's `run_api.py` client ended up talking to the *other* job's vLLM server,
  producing a wall of `"The model \`openai/gpt-oss-120b\` does not exist."` 404 errors from the
  20B job's server, since it only knows its own model name. The failures are handled safely
  (`process_instance` returns `None` and writes nothing to the output file on failure, confirmed
  by checking the output file afterward, it had zero garbage lines from the crossed run), so
  this doesn't corrupt data, but it does silently waste the whole job's runtime with nothing to
  show for it, and the error message ("model does not exist") looks like a config/model-id typo
  rather than a port collision, easy to misdiagnose. Fix: pass an explicit, distinct
  `VLLM_PORT` on every simultaneous submission (e.g. `VLLM_PORT=8003` / `VLLM_PORT=8004`) rather
  than relying on the default, especially for jobs likely to land on the same node (same
  partition, submitted close together).
- **`MODEL_LIMITS` (in `run_api.py`, used for client-side prompt truncation math) and
  `MAX_MODEL_LEN` (the actual `--max-model-len` passed to `vllm serve`, defaults to 32768) are
  two separate settings that must agree, and nothing checks that they do.** `MODEL_LIMITS`
  can claim a model's full real published context (e.g. 262144 for Qwen3-Coder-30B-A3B-Instruct),
  while the server itself is silently capped much lower, since `MAX_MODEL_LEN`'s 32768 default
  is a real, deliberate safety cap for a different model entirely (Qwen3-4B-Instruct-2507
  genuinely OOM'd at its native 262144 context on one L40S, only 31.62 GiB was available).
  Confirmed real 2026-08-31/09-01: a completed Qwen3-Coder-30B-A3B-Instruct production job
  lost 117 real instances this way, prompts the client thought it had room for got rejected
  outright by the server. Fixing this needs a real `MAX_MODEL_LEN` override per model, sized
  against real GPU memory headroom (`vllm serve`'s own startup log reports real free/used
  memory, `grep -i "Free memory on device\|Actual usage is"`), not a guess, and even a real,
  tested value can still be too low: `MAX_MODEL_LEN=65536` fixed Qwen3-Coder-30B-A3B-Instruct's
  `kg_only` arm (1208/1210, 6 remaining real losses) but a real 57537-token prompt still
  exceeded it on the `instruct` arm, needed raising to `98304`. Absence of this error in a
  partial or cancelled run is not evidence a model is unaffected, only that it had not yet
  reached the problem within however much of the dataset it processed, confirmed real when an
  earlier "clean" reading for Meta-Llama-3.1-8B-Instruct turned out to be from a 99/1210
  partial sample, not a real completed run; a later, real full run showed it genuinely
  affected too (a real 24769-token prompt hit the default 32768 wall).
- **A real, team-wide `/fs04` disk-full incident (2026-09-01/02) crashed two otherwise-healthy
  real production jobs mid-run**, not just blocked new downloads: `OSError: [Errno 122] Disk
  quota exceeded` then `OSError: [Errno 5] Input/output error` trying to write a completed
  result, a hard crash (`process_instance`'s own retry-and-skip handling doesn't cover a
  disk write failure), 15 to 19 real hours of GPU compute lost on jobs that were otherwise
  progressing cleanly. Root cause was genuinely team-wide (`lfs quota -g al49 /fs04` showed
  `3.0T`/`3.0T`, 0 available, and the pressure came predominantly from other, unrelated
  projects sharing the same `al49` allocation, not this project's own usage, confirmed by
  checking real per-account usage: this project's whole team was ~465GB against a 3.0T pool).
  Cleaning your own `~/al49_scratch/<username>/hf-cache/` of dropped models' weights (check
  `du -sh` per model under `hub/`) is worth doing but cannot fully resolve pressure at this
  scale alone; if `df -h ~/al49_scratch` shows `100%`/`0` available, check `lfs quota -g al49
  /fs04` and escalate to whoever administers the group's storage rather than only chasing your
  own directory. Confirmed resolved once the real allocation grew to `5.1T` total, `1.6T` free.
- **A large model's real startup time across many GPUs can exceed the script's own
  `VLLM_STARTUP_TIMEOUT` default (900s), even with that default already once raised from an
  original 300s.** Confirmed real 2026-09-02: `Llama-4-Scout-17B-16E-Instruct` (`TENSOR_
  PARALLEL_SIZE=4`, 4x H100) genuinely needs more than 900s just for one worker's weight
  loading (447.56 real seconds logged) plus real `torch.compile` graph-compilation overhead
  across 4 tensor-parallel workers; the real symptom is repeated `No available shared memory
  broadcast block found in 60 seconds` lines followed by `vllm server did not become ready
  within 900s`, not a crash or config error. Fixed by passing `VLLM_STARTUP_TIMEOUT=1800` on
  the submission. Larger models on more GPUs should be expected to need more startup room,
  not assumed to fit the same default that works for a single-GPU 7-8B model.
- **Fixed 2026-09-17 (testgeneval#43-adjacent): output filenames now include `k{num_samples}`**
  (e.g. `k1` for pass@1, `k5` for pass@5), across `run_api.py`, `run_pipeline.py`,
  `run_huggingface.py`, and `scripts/merge_and_validate.py`. Before this fix, output filenames
  didn't encode `NUM_SAMPLES`, so a real pass@1 job and a later real pass@5 job for the same
  model and arm silently collided on the same file. `run_api.py` built the output filename from
  model, dataset, and temperature only. `existing_ids` then read whatever was already at that
  filename as "already done" regardless of what real sample count those rows actually held.
  Confirmed real 2026-09-07 across four separate cases: three real, silent no-ops
  where a job assigned as pass@5 read the existing real pass@1 file, saw almost everything
  already "complete," and only real gap-filled a handful of missing instances at `k=1`, never
  actually generating real pass@5 data at all (gpt-oss-20B, Qwen3-4B, and twice for
  gpt-oss-120B, the second attempt also independently crashed on a missing `DTYPE=bfloat16`);
  and one real, genuine data-mixing incident (Llama-3.1-8B) where a real pass@1 job and a real
  pass@5 job both actually ran into the same file, producing up to 7 real duplicate rows per
  id, a mix of `k=1` and `k=5` samples tangled together, accumulated across at least 16 separate
  real jobs that had ever written to that one file. Workaround used at the time, now obsolete:
  before submitting a job at a different `NUM_SAMPLES` for a model and arm that already had real
  data, `mv` the existing file aside first (e.g. append `__pass1`/`__pass5`). With `k{num_samples}`
  now in the filename, this manual step is no longer needed -- a pass@1 and pass@5 run for the
  same model/arm/temperature genuinely can't collide anymore. Still worth confirming a resubmitted
  job's own log says `Read 0 already completed ids` when starting a new, previously-never-run
  sample count, as a sanity check. Any file mixed under the old pattern before this fix still
  needs the old cleanup: group and deduplicate by the real `id` field, not `instance_id`, two
  structurally distinct real tasks can share the same `instance_id` (confirmed real:
  `django__django-12091-15824` and `django__django-12091-15825`), so grouping by `instance_id`
  produces a false "duplicate" alarm.
- **A committed evaluation report with a real, correct `generated` count can still mean real
  evaluation never actually happened.** Confirmed real 2026-09-07 across five committed report
  files: every real `total_predictions`/`generated` count matched this project's own
  independently confirmed completion counts exactly, but every real `with_logs` category (and
  every other real outcome category) was completely empty. Traced directly into
  `swebench_docker/swebench_utils.py`: `with_logs` only gets appended to when a real,
  per-instance `.eval.log` file exists on disk, the artifact actual Docker-based test execution
  produces. `swebench_docker/run_docker.py` shells out to the real `docker` CLI directly, wrapped
  in a broad `except Exception` that only logs a warning on failure rather than raising, so if
  the real Docker daemon was not running wherever `run_evaluation.py` was executed, the script
  keeps going, `generated` stays correctly populated (that count is computed upstream of Docker
  entirely), and every real evaluation-outcome category silently stays empty with no visible
  error. Check `with_logs` (or any other real outcome category) directly in a `_report.json`
  before trusting a committed evaluation report reflects real, actual test execution, a nonempty
  `generated`/`total_predictions` count alone is not evidence evaluation ran.
- **Pulling this project's evaluation Docker images (`scripts/pull_images.py`) is fast per
  image but real, cumulative local disk usage can exceed expectations fast.** Confirmed real
  2026-09-07: all 102 real, distinct images in `Makefile.testgenevallite` pulled in roughly 15
  to 20 real minutes total (each individual pull took single-digit real seconds, these are
  pre-built images being downloaded, not built), but some real per-repo testbed images (astropy,
  sympy, sphinx in particular) run 3 to 3.6GB each, not the smaller "dozen-plus images, mostly
  shared base layers" pattern the first few pulls suggested. A real local Docker daemon can hang
  under this disk pressure (`docker images`/`docker system df` timing out with no response);
  confirmed recoverable via a full Docker Desktop restart with no real data loss both times it
  happened, but re-verify the real pulled-image count directly against the Makefile's own real
  target list after any such recovery, a read taken mid-hang can come back falsely near-empty.
  The full, non-lite `Makefile.testgeneval` defines 370 real images, roughly 3.6x more, and is
  very unlikely to fit on a machine that only just fit the lite set.
- **`Qwen/Qwen3-Coder-30B-A3B-Instruct` at `MAX_MODEL_LEN=98304` genuinely needs
  `TENSOR_PARALLEL_SIZE=2` (2x L40S), a single L40S is not enough.** Confirmed real 2026-09-08:
  two separate real submissions that requested only 1 GPU (no `TENSOR_PARALLEL_SIZE`, no
  `--gres` override) both failed identically on two different, otherwise-idle real nodes with
  the exact same byte-for-byte `CUDA out of memory` error (`44.39 GiB` total capacity, `245.31
  MiB` free, `43.63 GiB` already allocated by PyTorch before the request even completed),
  ruling out shared-node contention as the cause. The one real, full production run that
  actually completed cleanly at this `MAX_MODEL_LEN` (job 59668606) used `TENSOR_PARALLEL_SIZE=2`
  the whole time; this project's own real calibration data for this model documents a safe
  `MAX_NUM_SEQS=32` ceiling, but only at 2 GPUs, and that calibration does not explicitly confirm
  it was tested at 98304 specifically, so `MAX_NUM_SEQS=6` (the value the one proven, real
  production run actually used) is the safer real default until 32 is directly confirmed at this
  context length too. Real, working real command:
  `TENSOR_PARALLEL_SIZE=2 MAX_MODEL_LEN=98304 MAX_NUM_SEQS=6 MAX_CONCURRENCY=6 sbatch
  --gres=gpu:L40S:2 ...`. A command missing the `TENSOR_PARALLEL_SIZE`/`--gres` pair for this
  model at this context length will not just run slower, it will not start at all.
- **A completed job's own reported coverage number can be a stale pre-run snapshot, not the
  real final count, if the job's log isn't checked directly.** Confirmed real 2026-09-08:
  Llama-3.1-8B kg_only pass@5 (`59853977`) initially looked like a real 304/1210 (~25%)
  shortfall, matching the exact line count already sitting in the output file. Its own log's
  first lines told the real story: `Read 304 already completed ids from <file>` then
  `Filtered to 906 instances`, meaning 304 was what the job found already done *before it
  started*, not what it produced. It then ran all 906 to genuine completion (real progress
  bar hit `906/906`, script logged `Done!`), but a real cluster of `Request timed out`/
  `RetryError` lines, 356 timeouts collapsing into 96 real `RetryError`s, all bunched in the
  last ~5% of the log with none earlier, meant 96 of those 906 never got a response and were
  never written. Real final count: 304 + (906 − 96) = 1114, confirmed directly against the
  file (`wc -l`, 0 duplicate ids). Always check a job's own `Read N already completed ids` /
  `Filtered to M instances` lines before treating a raw line count as the real final number,
  especially for any resumed job.
- **Not every real completion gap is a timeout worth retrying, some are a genuine, permanent
  context-length mismatch.** Following on from the 96 missing ids above: tokenizing the real
  missing prompts with the model's own tokenizer (`AutoTokenizer.from_pretrained`, inside the
  `testgeneval-vllm` conda env, not the bare login node, which lacks `transformers`) against
  the real input budget (`MAX_MODEL_LEN − MAX_NEW_TOKENS` = `65536 − 48000` = `17536` tokens
  for this run) showed 29 of the 96 have real prompts from 20,790 up to 62,706 tokens, all
  already over budget regardless of any timeout. The other 67 were genuinely within budget
  and failed purely on wall-clock time. A char-count-based estimate would have been directly
  misleading here too, char counts alone (up to 282,905 chars) don't map linearly enough to
  flag the real boundary at this precision, tokenize with the real model tokenizer before
  deciding whether a missing id is fixable by raising a timeout or is structurally excluded by
  the model's fixed context window. Decision made for this project: exclude the 29 genuinely
  over-budget ids rather than shrink `MAX_NEW_TOKENS` to fit them, real generation quality on
  the recoverable 67+ majority matters more than forcing in a handful of outsized instances at
  a reduced output budget; real new denominator for this file is `1210 − 29 = 1181`, not 1210.
- **When `MAX_NEW_TOKENS` is set, `MAX_MODEL_LEN` must be set alongside it and must exceed it,
  never left to the slurm script's default.** Confirmed real 2026-09-10: a Llama-4-Scout
  pass@1 resubmit OOM'd on 2 of 4 GPUs during worker startup, before any inference. Not a
  GPU-count problem (`gres/gpu=4` was correct). The job's own logged args showed
  `'max_model_len': 32768, 'override_generation_config': {'max_new_tokens': 48000}`,
  `MAX_NEW_TOKENS` larger than `MAX_MODEL_LEN` itself, telling vLLM to reserve output-token
  room bigger than the whole context window. `MAX_MODEL_LEN` was never passed on the command
  so it fell to the script default of 32768, while `MAX_NEW_TOKENS=48000` was carried over
  from the testgeneval#46 fix pattern. The model itself supports far more (real
  `text_config.max_position_embeddings` is 10485760). Fixed by passing `MAX_MODEL_LEN=65536`
  explicitly plus dropping to the conservative `MAX_NUM_SEQS=6` (the earlier OOM was at
  `MAX_NUM_SEQS=16` even at the smaller 32768).
- **The default `gpu` partition has a real per-user GPU cap of 4 (`gpuq` QOS,
  `MaxTRESPerUser = gres/gpu=4`), which blocks concurrent large tensor-parallel jobs.**
  Confirmed real 2026-09-10: with two 1-GPU L40S jobs running, a 4-GPU job cannot start until
  both free up, `squeue` shows `PD (QOSMaxGRESPerUser)`, distinct from a plain `(Priority)`
  wait. Check the real cap with `sacctmgr show qos format=Name,MaxTRESPerUser`. Work around by
  moving the large job to the `m3h` H100 partition (`--partition=m3h --qos=m3h
  --gres=gpu:H100:4`), a separate pool that does not count against `gpuq`, though `m3h` has
  its own `gres/gpu=4` cap so two 4-GPU jobs there still queue one behind the other.
- **Pass@5 jobs need a real `--time=48:00:00` floor, not 36h, especially for slow-prompt
  models.** Confirmed real 2026-09-10: three pass@5 jobs (Llama-3.1-8B instruct, Qwen3-4B
  both arms) all hit `State=TIMEOUT` at exactly 1 day 12 hours with `--time=36:00:00`, none
  reached `Done!`, all were making steady real progress the whole time (just too slow to
  finish 1210 instances at 5 samples each). Real measured per-instance rates: Llama-3.1-8B
  instruct 272 s/it (~67h for a full run), Qwen3-4B instruct ~100 s/it (~34h, just over the
  36h budget once startup and retries are added). Because `run_api.py` flushes every
  completed instance immediately, no real work was lost, resubmit with `existing_ids`
  resuming from the partial file and a 48h budget.
- **`run_api.py`'s OpenAI client had no configurable request timeout until testgeneval#49.**
  Confirmed real 2026-09-09/10: a Llama-3.1-8B kg_only pass@5 fixup lost 62 of 96 real
  remaining instances to `Request timed out` / `RetryError[APITimeoutError]`, dominated by
  scikit-learn's large prompts, and 33 of those 62 were ids that had succeeded on the prior
  run, the timeout intermittently claims previously-good instances on retry. The client was
  constructed with no `timeout=` argument at either call site, running on the OpenAI SDK's
  600s default with no env var to change it. Fixed by adding a `REQUEST_TIMEOUT` env var
  (default 600, preserving prior behavior) wired into both `openai.OpenAI(...)` sites and
  `m3_run_inference.slurm`. Set `REQUEST_TIMEOUT=1800` when resubmitting against a
  slow-prompt-heavy subset.
- **gpt-oss models have a fourth, distinct null-content failure separate from reasoning-budget
  exhaustion: `finish_reason=stop` with `content=None`.** Confirmed real 2026-09-10 on
  gpt-oss-20B kg_only pass@5, one id (`django__django-14411-16029`), `completion_tokens=3762`,
  the model emitted a valid stop signal at a modest token count but the Harmony parser still
  found nothing in the final content channel (all output stayed in the reasoning channel).
  Distinct from the `finish_reason=length` reasoning-budget exhaustion seen on gpt-oss-120B.
  Very rare (1 in ~6050 real samples for that arm), noted and accepted rather than resubmitted.
- **`apptainer pull docker://<image>` runs rootless on M3, no sudo needed.** Confirmed real
  2026-09-11: astropy testbed pulled and converted to a 902MB `.sif` in ~9 min with no
  elevated privileges. Only `apptainer build` from a `.def` file needs root. An earlier belief
  (testgeneval#2, this file's own prior notes) that pulls needed sudo was wrong, corrected. Set
  `APPTAINER_CACHEDIR` AND `APPTAINER_TMPDIR` to a real scratch path before pulling, or the
  final `mksquashfs` step fills `/tmp`/`$HOME` and dies with `No space left on device` even
  when scratch has terabytes free.
- **Never invent a scratch path by pattern-matching an existing one's name, confirm it exists
  first.** Confirmed real 2026-09-11: `m3_pull_apptainer_images.slurm`/`m3_run_evaluation.slurm`
  defaulted to `$HOME/al49_scratch2/$USER/...`, invented from the real `HF_HOME` pattern
  (`al49_scratch`, no `2`) without checking. `$HOME`'s only real scratch symlinks are
  `al49 -> /projects/al49` and `al49_scratch -> /scratch/al49`; `al49_scratch2` was never real.
  `mkdir -p` doesn't fail on a bad path assumption, it silently creates a genuine new directory
  under `$HOME` instead. A real bulk `.sif` pull ran 27m40s writing there before being caught,
  and combined with pre-existing legitimate usage (`.conda` 13GB, `.cache` 6.7GB, `.apptainer`
  987MB, `.triton` 533MB against a 20GB total home quota) drove `$HOME` to 100% full, 146MB
  free, a real, active risk to anything else touching home. Fixed: `rm -rf
  "$HOME/al49_scratch2"` (recovers the wrongly-placed files; no data lost since the correct
  real copies already existed on `/fs04/scratch2/al49/$USER/`), `rm -rf "$HOME/.apptainer"`
  (frees the default Apptainer cache, real, ~987MB), both slurm scripts corrected to default
  to `/fs04/scratch2/al49/$USER/...`, the path actually proven working (788GB free) throughout
  this project, not a second guess. Before trusting any new default path in a script:
  `readlink -f`/`ls -la` the parent directly, don't assume from a name that looks plausible.
- **`kdjain`, not `aorwall`, is the Docker Hub namespace with real, correct testbed images.**
  Confirmed real 2026-09-11: `aorwall/swe-bench-<repo>-testbed:<version>` (upstream) lacks
  `cosmic-ray` and this fork's own `swebench_docker`, both added by this repo's own
  Dockerfiles, not upstream's. A mutation run against an `aorwall` `.sif` fails outright with
  `ModuleNotFoundError: cosmic_ray`. `scripts/pull_apptainer_images.py --namespace kdjain`
  (the default) pulls the correct, fork-built images.
- **The Apptainer evaluation backend is correctness-validated against Docker, not just
  "it runs."** Confirmed real 2026-09-11: `pallets__flask-5014-16417`, 5 real samples,
  evaluated through both `--backend apptainer` on M3 and plain Docker locally. Every metric
  (`CoverageLOG`, `FunctionCoverageLOG`, `MutationLOG`, `FunctionMutationLOG`) matched to full
  floating-point precision on every sample. Separately confirmed the pyenv-repo bind-path fix
  (`task_instance.json` must bind unconditionally to `/home/swe-bench/task_instance.json`, not
  next to the entrypoint, or every `django`/`requests`/`scikit-learn` instance fails with
  `ValueError`) with a real `django 3.0` instance, clean run, no crash. See testgeneval#52
  (merged) and testgeneval#53 (remaining: full 126-image pull, real concurrency measurement).
- **Without an explicit `--home`, Apptainer containers share the HOST user's real `$HOME`, not
  an isolated one.** Confirmed real 2026-09-11: a real concurrency test (`NUM_PROCESSES=8`) hit
  24/24 failures, `error: could not lock config file /home/<user>/.gitconfig: File exists`.
  Every container's `entrypoint.sh` runs `git config --global --add safe.directory ...`, and
  with no `--home` set, all 8 concurrent containers raced to write the exact same real host
  file. `--cleanenv` strips environment variables, it does not isolate the filesystem. Every
  earlier validation of the Apptainer backend ran `NUM_PROCESSES=1`, so this never surfaced.
  Fixed in `run_apptainer.py`: `--home` a fresh `tempfile.mkdtemp()` directory per instance,
  cleaned up alongside the existing `task_instance.json` tempfile. Confirmed real 0 races,
  multiple successful concurrent runs on resubmit.
- **`apptainer pull`/`build` crashes on M3 compute nodes with no working fix, only login nodes
  work, and M3 policy forbids a long job there.** Confirmed real 2026-09-11 on 3 different
  compute nodes (`m3e107`, `m3e108`, `m3k032`), same exact crash every time at the
  `mksquashfs` step: `proot error: ptrace(TRACEME): Operation not permitted`. The documented
  `PROOT_NO_SECCOMP=1` workaround does NOT fix it (tested directly). Root cause: no
  subuid/subgid allocation exists for the account (`grep "^$USER:" /etc/subuid /etc/subgid`
  empty on both login and compute nodes; `apptainer exec --fakeroot` reports "User not listed
  in /etc/subuid"), so Apptainer always falls back to the `proot` emulation layer for squashfs
  builds. Login nodes tolerate that fallback (3/3 real pulls succeeded there); compute nodes
  apply additional sandboxing that blocks it (even `cat /proc/sys/kernel/yama/ptrace_scope`
  fails with `Cannot allocate memory` inside a compute-node job). `apptainer exec` (running an
  already-built `.sif`, the real evaluation path) is unaffected, only pull/build hits this. But
  M3's own policy explicitly forbids heavyweight background processes on login nodes, and the
  real full 126-image pull (~19h) is unambiguously that. Real workaround: build `.sif` files
  off-M3 on a machine without this restriction, `rsync` to
  `/fs04/scratch2/al49/$USER/apptainer_images/`. See testgeneval#54.
- **Use the real M3 data-transfer node for large transfers, not a login node.** Confirmed real
  2026-09-12: `m3-dtn.massive.org.au` resolves via direct DNS lookup (`host m3-dtn.massive.org.au`),
  distinct from the `m3.massive.org.au` login-node address used for everything else this
  project. M3 itself distinguishes login (interactive/job-submission only) from DTN
  (large file transfers) node types; a ~105GB real `.sif` transfer belongs on the DTN, not a
  login node. `dtn.massive.org.au`/`dtn1.massive.org.au`/`m3dtn.massive.org.au` do NOT resolve,
  confirm the real hostname with `host`/`nslookup` rather than guessing a plausible-looking one
  (the same mistake that caused the `al49_scratch2` incident above).
- **A Docker container can build real `.sif` files on a Mac, a genuine second source alongside
  a teammate's native Linux machine.** Confirmed real 2026-09-12: `quay.io/singularity/
  singularity:v3.11.4` (a real Docker image containing a real Apptainer/Singularity binary)
  ran `pull --force ... docker://kdjain/swe-bench-astropy_astropy-testbed:4.2` successfully,
  producing a real, correct 949MB `.sif`, ~8-9 min despite running under `amd64`-on-`arm64`
  emulation (Docker Desktop's own real Linux VM, not a shared M3 compute node, so it does not
  carry the same ptrace/seccomp restriction documented above). All 126 real images pulled this
  way, 0 failures, ~105GB total.
- **macOS ships bash 3.2, `mapfile` (bash 4+) silently fails with no useful error.** Confirmed
  real 2026-09-12: a resumable pull-loop script used `mapfile -t ARR < <(...)` and produced
  `mapfile: command not found` followed by `unbound variable` on the array, zero real pulls,
  no useful signal beyond checking the log directly. Portable fix: write the list to a temp
  file, read it with `while IFS= read -r line; do ... done < "$tmpfile"`, works identically on
  bash 3.2 and 4+, no array or `mapfile` needed.
- **A completed job's own script logic finishing does not guarantee `sacct` will show
  `COMPLETED`.** Confirmed real 2026-09-11: a 4h46m concurrency test evaluated all 40 real
  instances successfully (0 races, every `Container ran successfully` line present, the
  script's own final `Done.`/`Build the report with:` lines printed), yet `sacct` showed
  `OUT_OF_ME+`/`ExitCode 0:125`. The real `oom_kill` event fired <0.5s after the very last
  instance's own success line, during the script's exit-time cleanup, not mid-evaluation, and
  not a `generate_report.py` call (that line is only a printed suggestion, never actually
  invoked). Check the real log content (did every real instance succeed, is `.eval.log` data
  present and intact) before trusting a `sacct` failure status alone to mean the real work was
  lost, in this case it wasn't.
- **A second, real data-mixing incident on Llama-3.1-8B instruct pass@5, same class as the
  first.** Confirmed real 2026-09-12: real file had 1289 lines against the 1210-instance
  dataset, immediate red flag from a plain `wc -l`. Audited by grouping on real `id` (not
  `instance_id`): 1199 unique ids, 90 duplicated, both real copies at full `{5: 5}` samples
  (not a `k=1`/`k=5` mix like the first incident). Same real fix pattern: dedupe keeping the
  last real occurrence per id, verify the dedup file independently before touching the
  original, move the original aside as a `.pre_dedup_backup` rather than deleting it. Real,
  final, corrected count: 1199/1210 (99.1%), replacing a stale 1195/1210 the status table had
  carried since before this job's real resubmit even ran. Root cause of the duplication not
  traced to a specific job this time, but the general lesson holds: always re-`wc -l` and
  re-audit a file believed final rather than trusting an earlier snapshot's count.
- **scikit-learn's 5 testbed Dockerfiles are missing `coverage`/`cosmic-ray`, a genuine,
  isolated, pre-existing bug, not an Apptainer artifact.** Confirmed real 2026-09-12 via
  calibration: `scikit-learn__scikit-learn-10198-16700` failed with `FileNotFoundError:
  'coverage'`. All 5 scikit-learn Dockerfiles (`0.20`-`1.4`) install a hardcoded package list
  with no `coverage`/`cosmic-ray` and no `requirements.txt` to fall back on, unlike every other
  repo. A first grep pass (`"pip install coverage"`) over-flagged django and sympy too, both
  real false positives (django uses `requirements.txt`, sympy bundles it into a combined `pip
  install` line), caught before filing anything by checking the Dockerfile AND any sibling
  `requirements.txt` together. Full sweep across all 18 repos, 156 real Dockerfiles: scikit-
  learn is the only genuine gap. Fixed (testgeneval#55), but the already-pulled `.sif` files
  were built from the old, broken Docker Hub images, the Dockerfile fix alone does not take
  effect until those real images are rebuilt/repushed and re-pulled.
- **Calibrate a new backend against every real repo before a full production handoff, not just
  a representative few.** Confirmed real 2026-09-12: astropy/flask/django validated the
  Apptainer backend's correctness and the `--home` concurrency fix, but the other 8 real repos
  (matplotlib, sympy, scikit-learn, sphinx-doc, pylint-dev, pytest-dev, pydata/xarray,
  mwaskom/seaborn) had never been touched. One instance per repo, run sequential + parallel +
  sharded (all three agreed), caught the scikit-learn bug above cheaply, in isolation, before
  a real production run would have hit it mid-batch across possibly hundreds of instances.
  `psf/requests` has real `.sif` files pulled but zero real instances in `kjain14/testgeneval`,
  confirmed directly, not a gap.
- **`scripts/shard_predictions.py` validated end to end for the first time.** Confirmed real
  2026-09-12: 2 real jobs, same 8-instance set split 4+4, writing to the same shared
  `--log_dir` on `/fs04`. Exact split, no overlap, no file collision between the two jobs'
  real `.eval.log` writes. Previously written but never actually exercised.
- **A job that times out repeatedly is not necessarily broken, check the real per-instance
  rate before assuming a bug.** Confirmed real 2026-09-12: Qwen3-4B instruct pass@5 hit a
  second real `TIMEOUT` (36h, then 48h). Real per-instance rate near the end (`128-268s/it`),
  `Running: 16` (fully using `MAX_CONCURRENCY`), `Waiting: 59-64` (genuine backlog), healthy
  GPU KV cache (21-47%), strong generation throughput with no degradation over 48h, and 0 real
  `Request timed out` (ruling out the testgeneval#49 pattern). This is genuine workload scale,
  not a bug: 621 real remaining instances at the confirmed rate need 40+ real hours. Resubmit
  with a real, generous time margin based on the measured rate, don't just double the previous
  budget blindly.
- **`results/{instruct,kg_only}/` accumulates several real files per model, only one of which
  is live — check the plain filename, not a suffixed one.** Confirmed real 2026-09-12: checked
  Meta-Llama-3.1-8B-Instruct's `__test__pass5.jsonl` (324/304 lines) and wrongly concluded real
  work was missing/lost, when the actual live file (`__test.jsonl`, no suffix — the exact path
  `run_api.py` reads/writes via `OUTPUT_DIR/{model}__{dataset}__{temp}__test.jsonl`) was already
  correctly at 1199/1208. The naming convention, reverse-engineered from RUN_LOG.md's real
  incident write-ups (not documented anywhere before this; **the filename shape itself changed
  2026-09-17 to add `k{num_samples}`, see the "output filenames now include k{num_samples}"
  entry above, the pattern below is the pre-fix historical shape**):
    - `{model}__{dataset}__{temp}__test.jsonl` (no extra suffix) = **the only live file**.
      `run_api.py`'s own resume logic (`Read N already completed ids from <this path>`) always
      reads/writes here. This is the one to `wc -l` for a real status check.
    - `__pass1.jsonl` / `__pass5.jsonl` = point-in-time **snapshots**, usually made right before
      a resubmit changed `NUM_SAMPLES` (e.g. "moved aside as `__pass1.jsonl` before resubmitting
      with `NUM_SAMPLES=5`") or produced by a real dedup/split fix. Once made, they are frozen
      and go stale the moment the live file changes again — never trust one for current status.
    - `_original.jsonl` / `.pre_dedup_backup` = the pre-fix copy kept before a corruption/dedup
      fix was applied to the live file, kept for audit trail, never touched again afterward.
    - A `_corrupted_output_limit_YYYYMMDD/` (or similarly dated) subdirectory = an entire
      superseded run, archived wholesale rather than deleted.
  When in doubt about which file is real, check `run_api.py`'s own resume log line
  (`grep "Read.*already completed ids" slurm-<jobid>.out`) against a fresh `wc -l` on the exact
  path it names — that's the only fully trustworthy real cross-check.
- **Adding a new per-instance metric to evaluation (e.g. testgeneval#77's `any_pass_at_1`)
  doesn't retroactively apply to existing logs, and a cheap targeted rerun to backfill it
  creates a real directory-merge problem.** `get_logs_eval` reads every metric for one instance
  (coverage, mutation, and now `any_tests_passed`) from the *same* single `.eval.log` file, and
  `generate_report.py`/`get_eval_report` only reads from one `LOG_DIR` at a time. So: (1) any
  log written before the new metric's code landed simply won't have it, real re-evaluation is
  required, re-reading the old log is not enough; (2) `--skip_existing` only checks file
  *existence*, not content, so re-running against the *same* `LOG_DIR` to backfill just the new
  metric would find every instance's log already there and skip all of them, producing nothing
  new; (3) the real fix, writing to a *new* `LOG_DIR` so the rerun genuinely executes, then
  creates a second directory with the new metric but none of the original's expensive-to-compute
  data (e.g. mutation scores, ~4-8x more expensive per instance than a coverage-only run, see the
  `SKIP_MUTATION` gotcha above), two disjoint directories, neither of which alone gives the full
  picture, and nothing in this codebase currently merges them. A cheap, mutation-skipped backfill
  run is still worth it over a full expensive rerun, but needs a real merge step (reading both
  directories per instance and combining fields) before `generate_report.py` can produce one
  complete report, not yet written as of 2026-09-17, tracked in testgeneval#76.
