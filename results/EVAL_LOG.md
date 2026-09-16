# Evaluation run log

Tracks real evaluation status per model/arm: whether `run_evaluation.py
--backend apptainer` (via `m3_run_evaluation.slurm`) has actually been
run, real completion counts, and real failure causes found along the
way. Separate from `RUN_LOG.md`, which tracks inference (generation)
only -- a model showing "Done" there means its predictions exist, not
that they have been evaluated. Split out on 2026-09-14 once this file's
content had grown enough to make the distinction genuinely useful (see
`RUN_LOG.md`'s own header: it is explicitly inference-only).

See `docs/GUIDE.md` for the overall workflow and testgeneval#56 for the
real team handoff plan (who is evaluating which model/arm).

## Real evaluation status per model/arm (pass@5, the primary metric)

Updated 2026-09-16. Real sub-issues per person: #57 (jliu0290), #58
(wtho0016), #59 (wlee0060), #62 (mvar0010), all linked under #56.

| Model | Arm | Real evaluation status |
|---|---|---|
| gpt-oss-20B | instruct (pass@1, pipeline validation only) | Done (full scale, 2026-09-14), see below. Not the real pass@5 metric. |
| gpt-oss-20B | instruct (100-instance dry run) | Done (2026-09-13), superseded by the full-scale run above |
| gpt-oss-120B | instruct | wtho0016, in progress (#58) |
| gpt-oss-120B | kg_only | wtho0016, in progress (#58) |
| gpt-oss-20B | instruct | wlee0060, in progress (#59), resubmitted after a first real 10/10 TIMEOUT, confirmed RUNNING (jobs 60167462-471, all 10/10) |
| gpt-oss-20B | kg_only | wlee0060, in progress (#59), same resubmit |
| Qwen3-Coder-30B | instruct | jliu0290, in progress (#57), hit and fixed a real second scikit-learn 1.4 bug (#60) and a real catastrophic-backtracking hang (#61) along the way |
| Qwen3-Coder-30B | kg_only | jliu0290, in progress (#57) |
| Llama-4-Scout | instruct | **Done, 100%**, mvar0010 (#62) |
| Llama-4-Scout | kg_only | **Done, 100%**, mvar0010 (#62) |
| Qwen3-4B | kg_only | **Done, 100%**, mvar0010 (#62) |
| Llama-3.1-8B | kg_only | **Done, 1208/1208**, mvar0010 (#62), confirmed via real .eval.log count matching combined shard size |
| Llama-3.1-8B | instruct | 1048/1210, mvar0010 (#62), resubmitted for the real remainder |
| Qwen3-4B | instruct | Not yet run, predictions complete (1199/1210, see RUN_LOG.md's 2026-09-14 entry), evaluation not yet started |

**None of the pass@5 rows above are actually complete yet.** The
gpt-oss-20B row marked "Done" is a real, full-scale run of the
*pass@1* file, used deliberately to stress-test the pipeline
(sharding, parallelism, `generate_report.py`) before trusting it for
the real metric; it does not count as gpt-oss-20B instruct's real
pass@5 evaluation, which still needs to happen separately against
that file.

### Real evaluation has not actually happened for any of this project's data

jliu0290 committed real evaluation report files
(`f222d1d`, `ff4ba30`) for five real model/arm combinations
(Llama-4-Scout both arms, Qwen3-Coder-30B both arms, gpt-oss-120B
kg_only). Every real `_summary.json`'s `total_predictions` count
matched this project's own independently confirmed real completion
counts exactly, so the real inference side of these reports is
genuine. But every real `_report.json`'s `with_logs` category, and
every other real outcome category (`install_fail`, `test_errored`,
`test_timeout`, `mutation_timeout`), came back completely empty across
all five files. Traced directly into `swebench_docker/swebench_utils.py`
(line 590): `with_logs` is only appended to when a real, per-instance
`.eval.log` file exists on disk, the artifact real Docker-based test
execution produces. None do. Confirmed this is not a code bug, the
report-generation logic is working exactly as written, real evaluation
(the actual Docker container run per instance) never happened for any
of these five real files.

Traced the likely real cause into `swebench_docker/run_docker.py`:
this fork shells out to the real `docker` CLI directly via
`asyncio.create_subprocess_exec`, wrapped in a broad `except Exception`
that only logs a warning on failure rather than raising. If the real
Docker daemon was not running wherever this was executed, every
real container-launch attempt would fail exactly this way: silent,
logged only as a warning, `generated` still correctly populated
(upstream of Docker entirely), every real evaluation-outcome category
staying empty. Not yet confirmed directly with jliu0290 what command or
environment produced these reports. No real pass@k, coverage, or
mutation score exists anywhere in this project as of this writing, this
remains the single largest real gap between the current, near-complete
real inference dataset and an actual, reportable result.

### A real, local attempt to build the evaluation Docker images

Investigated reviving the stale `feat/apptainer-backend` branch
(Docker-free evaluation, runnable on M3 itself) as an alternative to
needing a working Docker install. Real, current M3 disk headroom is
genuinely large enough now (`/fs04` at roughly 3.8TB used of 5.1TB,
about 1.3TB real free), removing what was originally thought to be the
main blocker. The real remaining blocker turned out to be local
machine storage instead: `scripts/pull_images.py --makefile
Makefile.testgenevallite` (102 real, distinct pre-built Docker images,
pulled rather than built, matching this project's own documented
"easiest path") is real, fast per-image (each pull took single-digit
real seconds), but real total local disk usage climbed faster than
expected, some real per-repo testbed images (astropy, sympy, sphinx)
run 3 to 3.6GB each, unlike the earlier assumed "dozen-plus" images at
a smaller average size.

The real local Docker daemon hung twice under this real disk pressure
(basic commands like `docker images`/`docker system df` timing out with
no response), both times recovered cleanly via a full Docker Desktop
restart with no real data loss, confirmed by re-verifying the real
pulled-image count directly against the Makefile's own real target list
after each recovery rather than trusting a mid-hang read (one such read
falsely showed only 1 real image present during the second hang; the
real count was unaffected once the daemon actually recovered). All 102
real `testgenevallite` images were eventually confirmed present via a
direct diff against the Makefile's real target list. The real, full
`testgeneval` (not lite) dataset defines 370 images, roughly 3.6x more;
given how much local space 102 images alone consumed, the full dataset
is very unlikely to be viable on this same machine without either
substantially more real local storage or moving the work to a machine
with more of it. The 102 real testgenevallite images were deleted again
afterward to restore local headroom, this was a real, deliberate
feasibility check, not a completed setup.

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

**The "needs sudo for pull" belief above was wrong, corrected 2026-09-11.**
`apptainer pull docker://<image>` runs rootless on an M3 login node, no
`sudo` required. Only `apptainer build` from a `.def` file needs root
(M3's own docs' `sudo apptainer build alpine.sif docker://alpine`
example is about `build`, not `pull`, and doesn't generalize the way it
first reads). Confirmed real: an astropy testbed pulled and converted
to a 902MB `.sif` in ~9 min with no elevated privileges. This changes
the real, practical picture from "images have to be built off M3 and
transferred" to "images can be pulled directly on M3."

## Apptainer backend merged, correctness-validated against Docker, and a real home-quota incident (2026-09-11)

The `feat/apptainer-backend` branch above (89 commits behind `main`,
untouched since 2026-08-24) was rebased conflict-free and merged as
testgeneval#52. Real, additional findings from bringing it current and
actually exercising it, beyond the rootless-pull correction above.

**Three real bugs found in review, all fixed before merge:**
- `task_instance.json` was bound next to the entrypoint
  (`{entrypoint_parent}/task_instance.json`), but `evaluate_instance.py`
  reads it from a hardcoded `/home/swe-bench/task_instance.json` with no
  real fallback (a base64 `INSTANCE` env var this path never sets). For
  a conda repo (astropy, flask, sympy, ...) the entrypoint parent
  happens to already be `/home/swe-bench`, so this worked by luck. For
  every `PYENV_REPOS` repo (django, requests, scikit-learn), the
  entrypoint parent is `/opt`, so the file landed at
  `/opt/task_instance.json` and every one of those instances failed
  with `ValueError`. Fixed by binding unconditionally to
  `/home/swe-bench/task_instance.json`, matching `run_docker.py`.
- The `swebench_docker` bind was read-write; `run_docker.py` mounts it
  `:ro`. Low severity (a stray `__pycache__` write into the host's
  checked-out fork), fixed to match.
- `pull_apptainer_images.py`'s repo-name regex (`[a-z0-9_]+`) couldn't
  match a hyphen, silently dropping 4 of 12 repos (`pylint-dev_pylint`,
  `pytest-dev_pytest`, `scikit-learn_scikit-learn`, `sphinx-doc_sphinx`,
  ~49 of the real 126 distinct testbed images) with no error, the
  script just reported a wrong, lower "77 distinct images" count. Fixed
  to `[a-z0-9_-]+` plus a sanity check that exits loudly if a full
  Makefile parses fewer than 8 repos.

**Correctness validated with an exact match against Docker, not just
"it runs."** `pallets__flask-5014-16417`, 5 real samples (from
Qwen2.5-Coder-7B-Instruct), evaluated through both backends on the
fixed code. Every metric on every sample (`CoverageLOG`,
`FunctionCoverageLOG`, `MutationLOG`, `FunctionMutationLOG`) matched to
full floating-point precision between the Apptainer run on M3 (15m38s)
and a Docker run on a local machine (28m23s). cosmic-ray's mutation
selection is apparently deterministic given the same code, so even the
mutation numbers lined up exactly, not just coverage. Flask is a conda
repo, so this validated the conda bind path specifically; the pyenv
bind fix was separately confirmed working with a real `django 3.0`
instance (`django__django-10730-15721`), which completed cleanly
(634.57s, no `ValueError`, real coverage/mutation numbers computed) on
the same day.

Real `.sif` sizes: astropy 5.1 (the largest real Docker testbed image,
~3.6GB) converts to 902MB-949MB depending on namespace. **Must pull
from the `kdjain` namespace, not `aorwall`**, confirmed real: an
`aorwall` testbed image lacks `cosmic-ray` and this repo's
`swebench_docker` (both are added by this fork's own Dockerfiles, not
upstream's), so a mutation run against one fails outright with
`ModuleNotFoundError: cosmic_ray`.

**Real, corrected image count: 126 distinct testbed images across 12
repos, not 77/8** (the pre-regex-fix miscount). At ~900MB worst case,
~110GB total, well under the real 788GB free on `/fs04` at the time.

### A real, active home-quota incident during the first full bulk pull

`m3_pull_apptainer_images.slurm`'s default paths for
`APPTAINER_IMAGES_DIR`/`APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` (and the
matching defaults in `m3_run_evaluation.slurm`) were written as
`$HOME/al49_scratch2/$USER/...`, a path that was never real, invented
rather than checked, confirmed real 2026-09-11: `$HOME`'s only actual
scratch-related symlinks are `al49 -> /projects/al49` and
`al49_scratch -> /scratch/al49` (no `2`), and neither matches
`/fs04/scratch2/al49/$USER`, the path this whole project has actually
used and confirmed working (788GB free) all along. `mkdir -p` doesn't
fail on a nonexistent parent the way a bad path assumption might
suggest, it just silently creates a genuine new directory tree under
`$HOME` instead. The first real bulk pull job ran for 27m40s writing
real `.sif` files there before being caught, and combined with
pre-existing legitimate usage (`.conda` 13GB, `.cache` 6.7GB,
`.apptainer` cache 987MB, `.triton` 533MB against a 20GB total home
quota), drove `$HOME` to 100% full, 146MB free, a real, active risk to
anything else touching home (shell config, SLURM's own bookkeeping,
other jobs). Caught before it broke anything else. Fixed: `rm -rf
"$HOME/al49_scratch2"` recovered the wrongly-placed files (no real data
lost, the correct real `.sif` copies already existed on `/fs04` from
earlier manual testing), `rm -rf "$HOME/.apptainer"` recovered the
default Apptainer cache (real, freed ~987MB), and both slurm scripts
were corrected to default to `/fs04/scratch2/al49/$USER/...`, the path
actually proven to work, not a second guess. `scripts/
pull_apptainer_images.py` and `run_apptainer.py`'s own `$HOME/
apptainer_images` fallback (used only if `APPTAINER_IMAGES_DIR` is
unset and either is invoked directly, not through the now-fixed slurm
wrappers) got warning comments rather than a changed default, since
guessing a new default without testing it is exactly the mistake that
caused this.

Lesson: never invent a scratch path by pattern-matching an existing
one's *name* (`HF_HOME`'s `al49_scratch` inspired the invented
`al49_scratch2`) without confirming the real path exists, `readlink -f`
or `ls -la` on the parent first. A path that merely looks plausible can
silently succeed at creating itself under `$HOME` instead of failing
loudly, and the real damage doesn't show up until the small home quota
is nearly gone.

Filed as testgeneval#53 to track the remaining real work: the full
126-image pull at scale (in progress, job 60003349 after two earlier
attempts, one killed by the path bug), a real concurrency/scale
measurement (the current ~2-3 cores/instance estimate is from a single
`top` snapshot, not measured under load), and this pyenv check
(completed, see above, kept in the issue as a record).

### A real concurrency race found and fixed: every container shared the host's real $HOME

Started the real concurrency measurement (testgeneval#53 item 3): 40
real instances (astropy, flask, django, limited to the 3 `.sif` files
pulled so far) through `m3_run_evaluation.slurm` with
`NUM_PROCESSES=8`. Real, immediate result: **24 failures, 0
successes**, every one identical:

```
error: could not lock config file /home/<user>/.gitconfig: File exists
```

Real cause: `entrypoint.sh` runs `git config --global --add
safe.directory ...` inside every container. Without an explicit
`--home`, Apptainer maps the container's `$HOME` to the HOST user's
real `$HOME`, not an isolated directory (`--cleanenv` strips
environment variables, it does not isolate the filesystem). With 8
containers running concurrently, all 8 raced to write the exact same
real host file, `/home/<user>/.gitconfig`; git's own file locking meant
only one writer at a time could win, every other concurrent attempt
failed outright. Every earlier validation of this backend (astropy,
flask, django, all in #52) ran `NUM_PROCESSES=1`, so this never
surfaced, this is exactly the kind of bug a real concurrency test
exists to catch and did.

Fixed in `run_apptainer.py`: pass `--home` to a fresh
`tempfile.mkdtemp()` directory per instance, cleaned up in the same
`finally` block that already removes the `task_instance.json` tempfile.
Real, confirmed result on resubmit: 0 `could not lock config file`
errors, multiple real successful container runs (3+ confirmed within
the first 40 minutes of a still-running job). Merged directly.

### Real, hard limit on where `.sif` files can be built: not M3 compute nodes, not M3 login nodes either

Attempting the real, full 126-image pull (testgeneval#53 item 1) as an
`sbatch` job surfaced a second, separate, more serious real problem.
Every `apptainer pull` attempt on a `comp`-partition compute node
failed identically, at the `mksquashfs` SIF-creation step:

```
FATAL: ... while creating squashfs: /usr/libexec/apptainer/bin/mksquashfs command failed: exit status 1: proot error: ptrace(TRACEME): Operation not permitted
```

Reproduced on 3 different real compute nodes (`m3e107`, `m3e108`,
`m3k032`), same exact crash every time; the documented `proot`
workaround, `PROOT_NO_SECCOMP=1`, does not fix it, confirmed by testing
it directly and getting the identical crash on a third node. Root cause
traced directly: `grep "^$USER:" /etc/subuid /etc/subgid` returns
empty on both a login node and a compute node, and `apptainer exec
--fakeroot` reports "User not listed in /etc/subuid, trying
root-mapped namespace". With no real subuid/subgid allocation, Apptainer
has no path to genuine unprivileged user namespaces anywhere on this
account, so it always falls back to the `proot` emulation layer for
squashfs builds specifically. Login nodes tolerate that fallback
(confirmed: 3/3 real pulls succeeded there earlier); compute nodes
apply additional sandboxing that blocks it outright, confirmed real:
even `cat /proc/sys/kernel/yama/ptrace_scope` fails with `Cannot
allocate memory` inside a real compute-node job, consistent with a
genuine, additional confinement layer not present on login nodes.

This only affects `apptainer pull`/`build` (the squashfs-creation
step). `apptainer exec` (running an already-built `.sif`, the actual
evaluation path validated in #52) is unaffected, every real evaluation
run in this project's history worked fine on compute nodes.

But M3's own usage policy explicitly forbids heavyweight, long-running
background processes on login nodes ("We will kill any heavyweight
processes... repeat offense may revoke access"), and the real full
126-image pull (~19h at the measured ~9 min/image rate) is
unambiguously that kind of job. So neither compute nodes (crash) nor
login nodes (forbidden) can run the real bulk pull on M3 as things
stand. Filed as testgeneval#54, cross-linked from #53. Real, chosen
workaround: build the `.sif` files off-M3 on a teammate's own Linux
machine (no such restriction there), transfer to
`/fs04/scratch2/al49/$USER/apptainer_images/` via `rsync`, the exact
plan testgeneval#2 assumed originally, before the real discovery that
rootless pull worked at all turned out to only be true on login nodes.
Also worth an M3 helpdesk request for a real subuid/subgid allocation
as the durable fix, which would remove this constraint for good, not
filed as of this writing.

### Real concurrency measured on a 24-core node: 8 is already oversubscribed

With the `--home` race fixed above, real resource usage was measured
directly (`top -bn1`, `free -h`, `nproc` on the live compute node) with
`NUM_PROCESSES=8` running 40 real instances, ~47 minutes into a still-
running job, 6 real successes confirmed by that point, 0 races.

Real node: `m3e105`, 24 physical cores (`nproc`), 1.5TB RAM. Real,
concrete numbers at that snapshot:

- `load average: 93.47, 95.18, 92.85` on a 24-core node, roughly **4x
  the physical core count**. `%Cpu(s)` showed `45.2 us, 11.1 sy, 43.1
  id`, the node was genuinely ~43% idle despite that load average,
  meaning many real processes were runnable but queued for a core, a
  real sign of CPU contention, not yet total gridlock.
- Real per-instance CPU varied far more than the earlier single-
  snapshot estimate (~2-3 cores) suggested: individual `coverage`
  processes at that moment ranged from `514.3%` down to `47.6%` CPU.
  The high outlier is consistent with `pytest-xdist` (confirmed present
  in the testbed Dockerfiles) spawning multiple parallel test workers
  for a single mutant inside one container, briefly using 5+ cores at
  once, not a flat, predictable per-instance cost.
- Real memory was never a constraint: `1.3Ti free` of `1.5Ti` total, 24
  cores were the real bottleneck, not RAM, at any concurrency level
  tested so far.
- Real concurrent work observed: ~7 active `cosmic-ray` process groups
  at that snapshot, close to the requested `NUM_PROCESSES=8` (some
  naturally finish/start between snapshots).

**Real, corrected planning number: `NUM_PROCESSES=8` already
oversubscribes a single 24-core `comp` node for this workload.** The
earlier, unmeasured estimate (~2-3 cores/instance, ~48-64 safe
concurrent under the 256-CPU per-user QOS cap) assumed a flatter,
lower per-instance cost than what real load shows; the pytest-xdist
spikes mean a real safe per-node concurrency is closer to
`NUM_PROCESSES=4-6`, not 8, until xdist's own worker count is
separately constrained inside the container (not yet attempted). This
matters directly for `m3_run_evaluation.slurm`'s default
`NUM_PROCESSES=8` and any real production eval run's sizing across
several nodes.

### The concurrency test finished: 40/40 real successes, 0 races, one real (minor) OOM at exit

The concurrency job above (job 60004046) kept running past the 47-
minute snapshot documented earlier. Real, final `sacct` state:
`OUT_OF_ME+` (`OUT_OF_MEMORY`), `ExitCode 0:125`, 4h46m elapsed. Read
at face value this looks like the job failed; the real log tells a
different, better story.

Every one of the real 40 instances completed: `grep -c "Container ran
successfully"` = 40, `grep -c "could not lock config file"` = 0. The
script's own final lines printed correctly (`Done. Per-instance logs
in ...`, `Build the report with: ...`), meaning the real work loop
finished on its own terms, it was not killed mid-evaluation. The real
`oom_kill` event fired at `22:30:17.709`, less than half a second after
the very last instance's own `Container ran successfully` line at
`22:30:17.326`. `generate_report.py` is never actually invoked by
`m3_run_evaluation.slurm`, line 130's `echo "Build the report with:
..."` is only a suggestion printed to the user, not a real call, so
this was not a report-generation OOM. Most likely real cause: a
transient memory spike during the script's own exit-time cleanup after
a long run (Apptainer overlay teardown for the last container, plus
Python's own garbage collection across everything the 4h46m run had
accumulated), not a failure of the evaluation work itself.

Real, honest conclusion: this is the single cleanest, most conclusive
result of the whole concurrency investigation, not a new problem. The
`--home` isolation fix holds completely under sustained real load (0
races across all 40 instances, not just an early sample), and the
OOM is a minor, cosmetic tail-end issue, not something that lost any
real data (every instance's `.eval.log` in
`/fs04/scratch2/al49/mvar0010/concurrency_test_logs` is real and
intact). Practical fix for future long real runs: raise
`m3_run_evaluation.slurm`'s `--mem` request slightly, or make
`generate_report.py` a real, explicit step in the script rather than
leaving it as a printed suggestion, so its own memory needs are
accounted for rather than competing with whatever's still winding down
from the eval loop. Not yet applied, low priority given the real work
itself completed successfully.

### A second, independent .sif source: built off-M3 via Docker, not just a teammate's native Linux machine

While waiting for a teammate to build the full 126-image set natively
(the real workaround chosen in testgeneval#54), a second, independent
path was found and validated on a Mac: `quay.io/singularity/
singularity:v3.11.4`, a real, existing Docker image containing a real
Singularity/Apptainer binary. Docker Desktop on macOS runs a genuine
Linux VM under the hood, distinct from a shared M3 compute node, so it
does not carry the same ptrace/seccomp restriction documented above.
Confirmed real: `docker run ... pull --force ... docker://kdjain/
swe-bench-astropy_astropy-testbed:4.2` produced a real, correct 949MB
`.sif`, matching the same real size seen from the M3 login-node pull,
in ~8-9 minutes despite running under `amd64`-on-`arm64` emulation
(the image is `linux/amd64`, the Mac is Apple Silicon).

A resumable wrapper script (`apptainer_pull_docker.sh`, kept local, not
committed, since it is Mac-specific and not part of the M3 workflow)
looped this over the same real 126-image Makefile list, using the same
atomic-rename-on-success pattern as `scripts/pull_apptainer_images.py`
so a killed/interrupted run never leaves a file that looks done but
isn't. One real portability bug found and fixed along the way: macOS
ships bash 3.2 (Apple has not updated it in years over licensing), and
`mapfile` (a bash 4+ builtin) silently failed with `command not found`
followed by `unbound variable` on the very first run, producing zero
real pulls with no useful error surfaced until the log was checked
directly. Fixed by replacing the `mapfile`-into-array pattern with a
temp file plus a `while read` loop, which needs no array and works
identically on bash 3.2.

Real, final result: **all 126 images pulled successfully, 0 failures**,
~105GB total on disk, matching the ~110GB estimate. Confirms the
off-M3 build path works from either a native Linux machine or a Mac
with Docker Desktop, giving two independent, real sources for the same
`.sif` set. Transfer to M3 uses the actual data-transfer node
(`m3-dtn.massive.org.au`, confirmed via direct DNS lookup, not a
login node) via `rsync`, per M3's own real distinction between login
and DTN nodes for large transfers.

### A second, real data-mixing incident found: Llama-3.1-8B instruct pass@5, 90 duplicate ids

A routine status check (`wc -l` on the real output file) found
1289 lines against a 1210-instance dataset, an immediate real red flag
given this exact model already had one genuine mixing incident earlier
in this project (see the 2026-09-07 section above). Audited the same
way: grouped by real `id`, not `instance_id`. Real result: 1199 unique
ids, 90 of them appearing twice, both real copies at full `{5: 5}`
samples each (not a `k=1`/`k=5` mix like the earlier incident, genuine
duplicate full rows this time). `1199 + 90 = 1289` reconciles exactly
against the line count, confirming no other real corruption beyond the
duplication itself.

Fixed the same way as the earlier incident: grouped every real row by
`id`, kept the last real occurrence per id, wrote a `.dedup` file,
verified it independently (`malformed: 0`, `1199` unique ids, `{5:
1199}` distribution) before touching the original. Original moved
aside as `...jsonl.pre_dedup_backup` rather than deleted, dedup file
promoted to the real filename. Real, final state: Llama-3.1-8B
instruct pass@5 is genuinely clean at **1199/1210 (99.1%)**, corrected
from the stale, wrong `1195/1210` figure the status table carried
before this was caught (that number in turn came from a still-earlier,
pre-dedup snapshot of this same file, from before job 59968456's real
resubmit even ran). Root cause of the duplication not confirmed
(most likely something touched this plain filename a second time
without the usual pre-move-aside step, the same class of collision
documented earlier this project, but not traced to a specific real job
this time).

### Calibration before team handoff: 8 real, previously-untested repos, sequential + parallel + sharded, found one real, isolated bug

Before handing the Apptainer backend to the team for full production
runs, calibrated it against one real instance from each of the 8
remaining, previously-untested repos (matplotlib, sympy, scikit-learn,
sphinx-doc, pylint-dev, pytest-dev, pydata/xarray, mwaskom/seaborn),
the ones not already exercised by the astropy/flask/django validation
in #52 and the concurrency test above. `psf/requests` has real,
pulled `.sif` files but zero real instances in the `kjain14/testgeneval`
dataset itself, confirmed directly, so 8 is the real, complete
remaining set, not a gap.

Ran the same real 8-instance predictions file three ways: sequential
(`NUM_PROCESSES=1`, one job), parallel (`NUM_PROCESSES=4`, one job),
and sharded (`scripts/shard_predictions.py --num-shards 2`, two
separate real jobs writing to the same shared `--log_dir`), the first
real test of the sharding path end to end, never previously exercised.
All three produced the identical real result: 7/8 real instances
succeeded cleanly, 1/8 (`scikit-learn__scikit-learn-10198-16700`)
failed with a real `FileNotFoundError: 'coverage'`. Sharding itself
worked correctly, the 4+4 split was exact with no overlap, both shard
jobs wrote their real `.eval.log` files to the shared directory with
no collision.

Traced the real scikit-learn failure to a genuine, pre-existing gap in
the testbed image definitions: all 5 scikit-learn Dockerfiles
(`0.20`-`1.4`) install a hardcoded package list with no
`coverage`/`cosmic-ray`, and scikit-learn has no `requirements.txt` to
fall back on, unlike every other repo. An initial grep pass
(`"pip install coverage"`) over-flagged django (13/13) and sympy
(13/13) as also missing it; both were real false positives (django
installs via a separate `requirements.txt`, sympy bundles `coverage`
into a combined `pip install` line on the version checked), caught and
corrected before filing anything. A full, corrected sweep (does
`coverage` appear anywhere in the Dockerfile OR a sibling
`requirements.txt`) across all 18 real repos, 156 real Dockerfiles,
confirmed scikit-learn's 5 files are the only genuine gap. Filed as
testgeneval#55, fixed (added `coverage cosmic-ray` to all 5), with the
real caveat that the already-pulled `.sif` files were built from the
old, broken Docker Hub images, the Dockerfile fix alone does not take
effect until those real images are rebuilt/repushed and re-pulled.

Real, practical value confirmed: calibration caught one genuine,
repo-specific bug in isolation, cheaply, before any real production
run would have hit it mid-batch across possibly hundreds of real
scikit-learn instances.

**Follow-up (2026-09-12/13): all 5 scikit-learn versions genuinely
fixed.** A second, independent real bug was found while fixing the
first: scikit-learn 0.20/0.21/0.22/1.3's real, unmodified upstream
`aorwall` base testbed images never actually compiled the real C
extension (`sklearn/__check_build/_check_build.so`, confirmed 0 `.so`
files anywhere in the tree via direct `apptainer exec` inspection on
all 4 real, unmodified images), so any test importing `sklearn` at all
hit `ImportError: No module named 'sklearn.__check_build._check_build'`.
1.4 was unaffected (already had 136 real `.so` files). Fixed by adding
`RUN python setup.py build_ext --inplace` after the post-pip-install
`git checkout` line in each of the 4 affected Dockerfiles. Verified
real end-to-end on M3: 0.20's fixed `.sif` (rebuilt locally via
`docker build --platform linux/amd64`, pushed to a real, accessible
Docker Hub namespace since the original `kdjain` images are broken,
pulled back as `.sif` via `quay.io/singularity/singularity`, transferred
to M3) ran the real, previously-failing instance
(`scikit-learn__scikit-learn-10198-16700`) to full completion: 610 real
mutants, `MutationLOG: 3.93%`, `FunctionMutationLOG: 0.0%` (a real,
legitimate score, not an error). 1.3 verified locally (136 real `.so`
files, clean Docker build on the newer Python 3.9 stack). All 4 fixed
`.sif` files transferred to M3, replacing the old broken ones at the
same real path. **Do not re-run `scripts/pull_apptainer_images.py` for
scikit-learn specifically**, it would silently restore the old, broken
`kdjain`-namespace images (see the warning comment added directly to
`m3_run_evaluation.slurm`).

### Qwen3-4B instruct pass@5's real, third timeout: confirmed genuine workload scale, not a bug

The redone job above (`59964291`, resubmitted at `--time=48:00:00`
after an earlier real 36h TIMEOUT) hit a second real `TIMEOUT`, 2 days
0h1m elapsed, real file at 589/1210 lines (up from 311 at start).
Investigated directly rather than just raising the time budget again:
real per-instance rate near the end of the run ranged `128.54s/it` to
`267.69s/it` (from the script's own progress bar), `Running: 16 reqs`
(fully using the configured `MAX_CONCURRENCY=16`), `Waiting: 59-64
reqs` (a real, substantial backlog, not idle capacity), `GPU KV cache
usage` a healthy 21-47% throughout, real generation throughput staying
strong (300-620 tokens/s) with no degradation over the full 48h run,
and only 2 real `RetryError`s, 0 `Request timed out` (ruling out the
testgeneval#49 timeout pattern entirely for this job).

Real, honest conclusion: this is not a bug, a hang, or server
degradation, it is genuinely this much real work at this real,
confirmed rate. Qwen3-4B instruct's real prompts are large (already
documented elsewhere in this project, mean ~11265 input tokens vs
kg_only's ~5001), and 621 real remaining instances at the confirmed
~200-270s/it real rate need on the order of 40+ real hours, more than
either of the two time budgets tried so far. Resubmitted as
`60021634` with `--time=60:00:00`, a real, generous margin above the
estimate given the observed per-instance variance. Sharding the
remaining instances across 2 real concurrent GPU jobs was considered
and deliberately not used this time, the real gain (~2x wall-clock)
did not justify the added real complexity of extracting exactly which
ids are still missing and confirming no overlap, for what is
ultimately a one-time job. As of this writing (2026-09-14), this job
is still running, real progress 700/1210.

### File-naming note: which prediction file is actually live (real confusion this caused, 2026-09-12)

`results/{instruct,kg_only}/` accumulates several real files per model
over a project's life, only one of which is ever live. Checked
Meta-Llama-3.1-8B-Instruct's `__test__pass5.jsonl` (324/304 real
lines) and, without cross-checking further, wrongly concluded this
model's real pass@5 work was missing or lost. The actual live file
(`__test.jsonl`, no suffix, the exact path `run_api.py` reads/writes
via `OUTPUT_DIR/{model}__{dataset}__{temp}__test.jsonl`) was already
correctly at 1199/1208, confirmed both by a fresh `wc -l` and by the
real resume job's own `Read 1199 already completed ids from <path>`
log line agreeing exactly.

The real naming convention, reverse-engineered from this project's own
past fixes (not written down anywhere before this entry):
- `{model}__{dataset}__{temp}__test.jsonl` (no extra suffix) is the
  only live file. This is the one to check for real current status.
- `__pass1.jsonl` / `__pass5.jsonl` are point-in-time snapshots, made
  when a resubmit changed `NUM_SAMPLES` or a dedup/split fix ran, and
  frozen from that moment on.
- `_original.jsonl` / `.pre_dedup_backup` are the pre-fix copy kept
  before a corruption/dedup fix, for audit trail only.
- A dated `_corrupted_*`/`_pre_*` subdirectory is an entire superseded
  run, archived wholesale.

Real, practical rule: when checking a model's real status, always
`wc -l` the plain, unsuffixed filename, and if a resume job is
involved, cross-check against its own
`grep "Read.*already completed ids" slurm-<jobid>.out` line, which
names the exact path and count it actually used.

Follow-up (2026-09-13): all 21 real snapshot files and 7 superseded
subdirectories under `results/{instruct,kg_only}/` were individually
verified (each candidate's expected live counterpart confirmed to
exist and be newer, plus a direct grep across the codebase confirming
zero scripts reference the suffixed filenames) then archived into
`results/_archive/{instruct,kg_only}/`, leaving only the live
`__test.jsonl` files visible in a plain `ls`.

### Real finding: django's testbed has no pytest, and most generated tests for it import pytest anyway (2026-09-13)

Before handing the pipeline to the team, ran a real pre-handoff dry
run: 100 real instances, randomly sampled across 10 distinct repos
from gpt-oss-20B's already-complete instruct pass@1 file (confirmed
clean, 1210/1210), sharded 4 ways (`scripts/shard_predictions.py`),
submitted as 4 real concurrent `m3_run_evaluation.slurm` jobs
(`60029929`-`60029932`) writing to one shared `--log_dir`, a genuine
test of both sharding and cross-job parallelism at a more realistic
scale than the earlier 8-instance calibration.

Found a real, severe, repo-specific gap partway through: 41 of the
first 52 completed logs (79%) failed with
`ModuleNotFoundError: No module named 'pytest'`. Broken down by repo,
this was not spread evenly: **42 of 43 sampled django instances**
hit it, every other repo in the sample (matplotlib, pytest-dev,
mwaskom/seaborn, astropy) was clean. Confirmed at the run's real
completion: 59/100 total instances hit this exact pattern, matching
the earlier partial-sample rate closely.

Traced the real source carefully rather than assuming (this looked at
first like the same class of bug as testgeneval#55's scikit-learn
`coverage` gap, but is not): confirmed directly that `django`'s base
source at the exact eval commit has zero files importing `pytest`
(`grep -rl '^import pytest' tests/` inside the real `.sif`, 0 matches)
and that the real, live `kjain14/testgeneval` dataset's own
`test_patch` field for the specific instance checked (731 real chars)
also does not import `pytest`. The actual source: **4 of 5 real
`gpt-oss-20b` samples for that instance wrote `import pytest` directly
into their own generated test file** (genuine model output, not a
testbed or dataset defect). django's real test suite has used plain
`unittest` for its entire history, but pytest is common enough
elsewhere in the Python ecosystem that models write it out of habit
without checking what's actually available.

**Decision: left as-is, not fixed.** Unlike the scikit-learn
`coverage`/compiled-extension gaps (genuine omissions in an otherwise
complete testbed, fixed in testgeneval#55), django's lack of `pytest`
is not an oversight; its real test infrastructure is deliberately
`unittest`-only project-wide. Installing `pytest` into the testbed
would change what is being measured (silently rewarding any model
that happens to write pytest-style tests, regardless of whether that
matches the codebase's actual conventions) rather than fixing a
broken environment. Documenting this prominently instead: **expect
django's real pass@k/coverage/mutation numbers to be substantially
suppressed for any model that defaults to pytest-style test
generation, for reasons unrelated to test quality.** If this
distortion turns out to matter for the real RQ2/RQ3 analysis, revisit
as a disclosed limitation in the writeup rather than a pipeline fix.

### Sharding fix: stratify by repo, not plain round-robin (2026-09-13)

The dry run above's real per-shard timing was uneven purely from an
uneven repo mix: 4 equally-sized (25-instance) shards took 1h56m to
6h55m (a 3.5x real spread), even though the source file was already
randomly shuffled, because plain round-robin (`i % num_shards`)
straight down an already-shuffled file can still let one shard draw a
disproportionate share of a slow repo's instances (one shard drew 7 of
the sample's sympy instances against another's 3). Fixed
`scripts/shard_predictions.py` (commit c6672a1) to group by repo first,
then round-robin within each group. Verified against the real dry-run
repo distribution: django went from an 8/15/12/8 split to 11/11/11/10;
sympy from 7/3/3/4 to 5/4/4/4.

### Full-scale (1210-instance) validation before team handoff: sharding, evaluation, and generate_report.py all confirmed working (2026-09-14)

Followed the 100-instance dry run with a genuine full-scale real test:
`gpt-oss-20B` instruct pass@1 (already confirmed clean, 1210/1210),
sharded 10 ways with the newly-fixed stratified `shard_predictions.py`
(commit c6672a1), submitted as 10 real concurrent
`m3_run_evaluation.slurm` jobs (`60040993`-`60041002`) against the full
account CPU quota (confirmed via `sacctmgr show qos`: `MaxTRESPU
cpu=256,gres/gpu=4` per user, no account-wide `GrpTRES` cap on top, so
256/24 = 10 concurrent jobs is a real, hard, per-user ceiling, not a
guess).

**Real timing.** Wall-clock for the slowest shard was ~22.5h, well past
what a naive early-rate extrapolation suggested (~9h). Real
per-instance rate varied enormously by repo: matplotlib/astropy
instances (the bulk of the first ~4 hours) ran 1200-11700s each; once
those repos fully drained, sympy/scikit-learn (measured directly from
completed instances) averaged ~1293s/instance. The stratified sharding
fix did its real job: no shard was left disproportionately holding a
slow repo's instances (spot-checked completion counts stayed within a
narrow range across all 10 shards throughout the run, unlike the
~8/15/12/8 django spread the old round-robin produced on the smaller
dry run).

**Real resource check, mid-run.** Confirmed directly (not assumed):
load average 49-82 and 107-422GiB memory used, out of 128 real cores
and 1.5TiB real memory, on each of the 4 real nodes hosting the 10
jobs (3-4 jobs per node, since these are 128-core nodes, not the
24-core ones the earlier `NUM_PROCESSES=8` oversubscription finding
was measured on). No real CPU or memory pressure at any point.

**Real completion.** All 10 shards reached `Container ran successfully`
for their full real instance count (1210 total, confirmed against
`ls .../fullrun_logs/*.log | wc -l`). 7 of 10 hit the same cosmetic
end-of-job OOM seen before (each one's own log shows `Done. Per-instance
logs in ...` and `Build the report with: ...` printed before the OOM
line, confirming real work finished cleanly beforehand in every case);
2 exited cleanly with no OOM; the last one was still in `RUNNING` state
by the time all logs existed (real work done, exit bookkeeping lagged).

**`generate_report.py` run at real full scale for the first time.**
Previously only exercised on the 100-instance dry run. Ran clean
against all 1210 real logs (`total_predictions: 1210`,
`no_generation`/`install_fail`/`reset_failed` all `0` across the board,
only `test_errored: 3`, `test_timeout: 1`, `mutation_timeout: 83`
(about 6.9%, consistent with the dry run's ~6% rate, not a new
problem). Real top-line numbers closely tracked the 100-instance dry
run's own report (`full_pass_at_1: 0.267` here vs `0.24` there), a
real, reassuring sign the smaller sample was already representative.

**Conclusion: the full apptainer-backend evaluation pipeline, the
stratified sharding fix, and `generate_report.py` are all confirmed
working correctly at genuine production scale**, not just in
isolated/small-sample tests. This is the strongest real validation
this pipeline has had before the team's own production runs
(testgeneval#56).

### First real pass@5 evaluation runs submitted, and a deliberate NUM_PROCESSES=8 test (2026-09-14)

With the pipeline validated, submitted the first 5 real pass@5 files
(mvar0010's testgeneval#56 assignments: Llama-4-Scout both arms,
Qwen3-4B kg_only, Llama-3.1-8B both arms, the last of these completed
just after #56 was filed). Each file sharded 2 ways
(`scripts/shard_predictions.py`, the stratified fix), 10 real jobs
total, using the full per-user job quota.

**Deliberately tested `NUM_PROCESSES=8` for the first time**, double
the value validated in the full-scale run above. Real justification:
the full-scale run's own resource check found only 49-82 load average
on 128-core nodes at `NUM_PROCESSES=4` with 3-4 jobs per node, real
headroom well below the point the original `NUM_PROCESSES=8`
oversubscription finding was measured at (a 24-core node, a
structurally different, smaller real constraint). Not yet confirmed
whether 8 actually delivers a real, proportional throughput gain on
these bigger nodes, or just adds concurrent load without reducing real
wall-clock, since the per-instance work itself (mutation testing) is
not obviously parallelizable further within one instance.

**Real timeout risk assessed before submitting, not after.** At the
new shard size (~600-607 instances each, roughly 5x the full-scale
run's 121-instance shards), naive extrapolation from the validated
`NUM_PROCESSES=4` per-instance rate put real completion at 56-112
hours, well past the script's default 24h `--time` limit -- these were
knowingly submitted at the default anyway, since `--skip_existing`
(already unconditional in `m3_run_evaluation.slurm`) makes a timeout
genuinely low-cost: it checks each real instance's `.eval.log` in
`LOG_DIR` and skips already-done ones, so a resubmit of the identical
command after a timeout picks up exactly where it left off, no real
progress lost. Chose to accept the timeout risk rather than
pre-emptively over-shard or extend `--time` speculatively, and monitor
real progress directly instead.

Real job IDs: `60078548`/`60078549` (Llama-4-Scout instruct),
`60078550`/`60078551` (Llama-4-Scout kg_only), `60078552`/`60078553`
(Qwen3-4B kg_only), `60078554`/`60078555` (Llama-3.1-8B kg_only),
`60078556`/`60078557` (Llama-3.1-8B instruct). All 10 started running
within 2 minutes of submission, spread across 8 distinct real compute
nodes.

### Three real, sequential per-account setup gaps found onboarding the first teammate (2026-09-14)

Handed off the first three real per-model assignments as GitHub
sub-issues (testgeneval#57/#58/#59, linked under #56). wtho0016 was
the first to actually try running them, and hit three separate, real,
sequential blockers, each only surfacing once the previous one was
fixed:

1. **No `.sif` testbed images under her own account.** Real error:
   `No .sif files in APPTAINER_IMAGES_DIR
   (/fs04/scratch2/al49/wtho0016/apptainer_images)`. Pulling all 126
   herself would take ~19h. Fixed by pointing `APPTAINER_IMAGES_DIR`
   at mvar0010's already-populated copy instead, confirmed genuinely
   group-readable (`getfacl`: `group::r-x` on both
   `/fs04/scratch2/al49/mvar0010` and its `apptainer_images`
   subdirectory, and her real group membership includes `al49`,
   confirmed via `id wtho0016`).

2. **No `testgeneval` conda env under her own account.** Real error:
   `EnvironmentNameNotFound: Could not find conda environment:
   testgeneval`. Each person's conda env is genuinely per-account (no
   equivalent to the `.sif`-directory sharing trick, envs live under
   each person's own `$HOME`/conda install), so this really does need
   a real, one-time `conda env create -f testgeneval.yaml` per
   teammate. Confirmed it's the plain `testgeneval` env this needs,
   not `testgeneval-vllm` (`m3_run_evaluation.slurm`'s own
   `CONDA_ENV` default is `testgeneval`; `testgeneval-vllm` is
   `m3_run_inference.slurm`'s default, a heavier, inference-only env
   nobody doing evaluation-only work needs).

3. **A corrupted, incomplete HuggingFace dataset cache in her own
   `$HOME`.** Real error, after the first two fixes let the job
   actually start doing real work: `OSError: Cannot find data file`,
   pointing at a `.incomplete/` path under
   `~/.cache/huggingface/datasets/kjain14___testgeneval/...`. Traced
   to real, genuine `$HOME` quota pressure, not bad luck: her own
   `du -sh` showed `.cache` at 5.4GB and `.conda` at 13GB, a combined
   18.4GB against M3's real ~20GB `$HOME` quota (the same real ceiling
   documented in the earlier `$HOME`/`al49_scratch2` incident), almost
   certainly what interrupted the original download partway through.
   `lfs quota` cannot report real usage against `$HOME` directly
   (confirmed again here, same limitation documented before: `/home`
   is "not on a mounted Lustre filesystem"), and mvar0010 could not
   inspect her `$HOME` contents directly either (real, expected
   per-user permission denial, unlike the shared `al49` scratch
   directories). Fixed two ways: cleared the real stale artifact
   (`rm -rf ~/.cache/huggingface/datasets/downloads`, confirmed by her
   own `ls` to have left only a stray, empty `.lock` file, no real
   partial data), and, since simply retrying into the same tight
   `$HOME` risked the identical interruption recurring, pointed
   `HF_DATASETS_CACHE` at a fresh directory under her own scratch
   space (`/fs04/scratch2/al49/wtho0016/hf-datasets-cache`) instead,
   added to the `sbatch --export` list alongside
   `APPTAINER_IMAGES_DIR`.

All three fixes were folded directly into testgeneval#57/#58/#59's
instructions (not just left as a reply to wtho0016), on the
expectation that jliu0290 and wlee0060 would hit the identical three
gaps on their own first real attempts otherwise, since none of them
have ever run this pipeline under their own M3 accounts before.

### A second, real scikit-learn 1.4 bug found via a teammate's real run: never rebuilt after #55, and also needed build_ext (2026-09-15, testgeneval#60)

With 4 teammates' real jobs running in parallel, jliu0290 hit a real
`FileNotFoundError: [Errno 2] No such file or directory: 'coverage'`
on `scikit-learn__scikit-learn-26644-16933`, the exact same error
class as testgeneval#55, but on 1.4, which that issue's own
investigation had concluded was clean (136 real `.so` files, no fix
needed, confirmed at the time by checking the already-published
`-testbed:1.4` tag directly).

Traced to two separate, real gaps:

1. **1.4's real `.sif` on M3 genuinely predated the coverage/cosmic-ray
   fix (commit 436d9b0), confirmed directly**: `apptainer exec` into
   the live M3 file, `pyenv which coverage`/`pyenv which cosmic-ray`
   both returned "command not found" under both pyenv versions present
   in the image. The earlier #55 fix commit did touch 1.4's Dockerfile
   too (its own message says "5 scikit-learn Dockerfiles"), but the
   real `.sif` on M3 was pulled/verified before that commit existed in
   this session's timeline and was never rebuilt afterward the way
   0.20/0.21/0.22/1.3 were.

2. **A genuinely fresh rebuild from the now-current Dockerfile also
   lost the compiled C extension**, a real regression found while
   fixing #1 (0 real `.so` files in the fresh build, versus the 136
   the earlier, already-published tag had). Root cause: the real base
   image `aorwall/swe-bench-scikit-learn_scikit-learn:bookworm-slim`
   has 0 real `.so` files (confirmed directly, same finding #55
   already made for the other 4 versions), and 1.4's own Dockerfile
   never had a `build_ext` step. The earlier #55 verification checked
   the already-published `-testbed:1.4` tag (which had the extension
   compiled by whatever process built that specific tag upstream, not
   by this fork's own Dockerfile), so a real gap in this fork's own
   Dockerfile went unnoticed until an actual fresh rebuild exposed it.

Fixed the same way as #55's other 4 versions: added
`RUN python setup.py build_ext --inplace` to
`docker/scikit-learn__scikit-learn/1.4/Dockerfile` (commit aa55d5d).
Rebuilt locally (`--platform linux/amd64`), verified both fixes
directly before pushing (136 real `.so` files, `coverage`/
`cosmic-ray` both present), pushed to Docker Hub
(`miggy711/swe-bench-scikit-learn_scikit-learn-testbed:1.4`), pulled
back as a real `.sif` via `quay.io/singularity/singularity`,
transferred to M3, replacing the broken copy in place at the same
path every teammate's jobs already point at. Re-verified on the live
M3 file after transfer: `coverage`/`cosmic-ray` present, 136 real
`.so` files, matching the local pre-transfer verification exactly.

No command changes needed on any teammate's end, since the fix
replaced the file in place. Any real scikit-learn 1.4 results from
before the transfer (2026-09-15, ~13:25 local time) are suspect and
may need re-running; anything submitted after should be clean.

### Real catastrophic-backtracking regex bug found via a teammate's genuine 21h hang, fixed, checked for silent corruption across the whole project (2026-09-15/16, testgeneval#61)

jliu0290 hit a real, reproducible infinite hang: `django__django-12396`
consumed ~99% of one CPU core continuously for ~21 hours, no child
processes, `timeout=3600` never firing, manually cancelled. Confirmed
directly (not just from the description): `timeout 30 python3 -c
"..."` reproducing `extract_preamble_classes_and_functions()` against
the exact real generated test that hung (13,523 real chars, `@patch`
decorator chains) genuinely killed the process at the 30s mark, exit
code 124.

Root cause: `class_pattern`/`test_method_pattern`/`test_function_pattern`'s
decorator-repeat group had the shape `(\s*@...\s*)*`, a classic real
catastrophic-backtracking pattern (two independently-backtracking
`\s*` at each end of a repeated group). Not a subprocess call, pure
Python regex execution in the main evaluation process, so no
subprocess-level timeout could ever interrupt it.

Fixed by anchoring each repeat on a literal `\n` instead of `\s*`,
removing the ambiguity. Verified directly against the same real
hanging content: `class_pattern.finditer()` went from a real >10s
hang to 0.3ms (18 real matches), `test_method_pattern.finditer()`
from >10s to 0.16ms (12 real matches).

**Checked the real, complete scope before deciding whether a project-
wide rerun was needed**, rather than assume: ran both the old and new
regex against every real generated test in every real prediction file
in the project (70,345 real predictions across 23 files). Real
result: **0 mismatches** between old and new regex output on every
input that did not hang the old one, confirming the bug's only real
symptom is hanging, never silently wrong parsing, so no real evaluation
data anywhere has been corrupted by this. **17 real hangs found**
(12 distinct real instance ids, spanning django/sphinx/xarray/
matplotlib/scikit-learn/pylint): `django__django-14727-16057`,
`django__django-12503-15855`, `sphinx-doc__sphinx-9128-17042`,
`django__django-12396-15845`, `django__django-13615-15954`,
`django__django-15521-16119`, `django__django-16749-16207`,
`pydata__xarray-6394-16538`, `matplotlib__matplotlib-23140-16282`,
`scikit-learn__scikit-learn-7760-16935`, `sphinx-doc__sphinx-9155-17043`,
`django__django-14996-16075`, `pylint-dev__pylint-5201-16594`.

**Decision: no project-wide rerun needed.** These 12 real instances
just need their real evaluation to actually run (or re-run, if a
prior attempt hung and was manually killed/skipped) once the fix
merges. Filed as PR #61, held open for real review rather than
merged directly, since this touches shared parsing code every real
evaluation run depends on.
