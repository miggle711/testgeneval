# Alternative to run_docker.py for M3, which does not support Docker but
# does support Apptainer. Every Docker-derived testbed image still has to
# be converted to a .sif file first, this module only handles running an
# already-present .sif, the same way run_docker.py only handles running an
# already-pulled Docker image. Use scripts/pull_apptainer_images.py to
# populate APPTAINER_IMAGES_DIR.
#
# `apptainer pull docker://<image>` works rootless on an M3 login node,
# confirmed real 2026-09-11 (astropy testbed pulled and converted to a
# 902MB .sif in ~9 min, no sudo). Only `apptainer build` from a .def file
# needs root, which is not what this path does. `APPTAINER_CACHEDIR` AND
# `APPTAINER_TMPDIR` both have to point at scratch before pulling, or the
# final mksquashfs step fills /tmp or $HOME and dies with "No space left
# on device" even when scratch has terabytes free (confirmed real, same
# day). An earlier version of this comment and testgeneval#2 said M3
# needs sudo for the pull; that was wrong.
#
# Three real differences from a naive docker run -> apptainer translation,
# each confirmed against a live M3 job (testgeneval#2):
#   1. `apptainer run <sif>` doesn't work -- the OCI-to-runscript
#      conversion mangles an absolute Dockerfile ENTRYPOINT into a
#      relative one (`./entrypoint.sh`), which then resolves against
#      whatever directory the host shell happened to be in, not anything
#      inside the container. `apptainer exec <sif> <real-entrypoint-path>`
#      sidesteps the broken runscript entirely.
#   2. The entrypoint's real absolute path differs by base image:
#      /opt/entrypoint.sh for pyenv-based repos (this repo's own
#      docker/pyenv/Dockerfile), /home/swe-bench/entrypoint.sh for
#      conda-based ones (third-party aorwall/swe-bench-conda base image,
#      not built from anything in this repo). PYENV_REPOS in
#      constants.py already distinguishes these for other reasons.
#   3. Apptainer does not honor the image's Dockerfile WORKDIR --
#      `os.getcwd()` inside the container comes back as the *host's*
#      invoking directory. --pwd has to be set explicitly to the
#      entrypoint's parent directory, or evaluate_instance.py's
#      `python -m swebench_docker.evaluate_instance` can't resolve the
#      swebench_docker package (it's imported relative to cwd).
#   4. Without an explicit --home, Apptainer maps the container's $HOME
#      to the HOST user's real $HOME, not an isolated directory. Every
#      earlier validation of this backend ran one instance at a time, so
#      this never surfaced; a real NUM_PROCESSES>1 concurrency test
#      (testgeneval#53) found it directly: two containers running
#      entrypoint.sh concurrently both execute `git config --global
#      --add safe.directory ...`, and both raced to write the SAME real
#      host file, /home/<user>/.gitconfig, every one but the first
#      losing with "could not lock config file: File exists". Fixed by
#      passing --home to a fresh tempfile.mkdtemp() directory per
#      instance, cleaned up in the same finally block that removes the
#      task_instance.json tempfile.

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time

import dotenv
from swebench_docker.constants import (
    KEY_PREDICTIONS,
    MAP_VERSION_TO_INSTALL,
    MUTATION_TIMEOUT,
    OVERHEAD_TIMEOUT,
    PYENV_REPOS,
)

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Margin added on top of `timeout` to get the host-side wall-clock cap.
# - `timeout` alone doesn't bound the host: it's only passed INTO the
#   container as APPTAINERENV_TIMEOUT, so a container that hangs before
#   reaching that logic runs forever (confirmed real: 5.1h scikit-learn
#   hang, Slurm OOM-killed at 180G).
# - Deriving from `timeout` instead of a fixed constant means raising
#   TIMEOUT in m3_run_evaluation.slurm for a slow repo actually raises
#   the host cap too, not just the in-container one.
# - The margin gives the container's own timeout logic a head start, so
#   its clean failure wins the race in the normal case, not a generic
#   host-side SIGKILL.
# Set APPTAINER_HOST_TIMEOUT directly to override with a fixed value.
APPTAINER_HOST_TIMEOUT_MARGIN = int(
    os.environ.get("APPTAINER_HOST_TIMEOUT_MARGIN", 300)
)


# Repos whose test command needs more space than --writable-tmpfs gives. That
# area is capped by Apptainer's `sessiondir max size`, 64 MB on M3, and sphinx's
# `tox` run fills it while packaging, so its tests never start. An overlay
# directory keeps the image's contents and puts writes on real disk instead.
REPOS_NEEDING_DISK_OVERLAY = {"sphinx-doc/sphinx"}


def _writable_args(overlay_dir):
    """Apptainer arguments for the container's writable layer."""
    if overlay_dir:
        return ["--overlay", overlay_dir]
    return ["--writable-tmpfs"]


class ApptainerUnavailableError(RuntimeError):
    """The apptainer binary cannot be launched, so no instance can be evaluated.

    Kept separate from a single container failing: that is routine and
    only costs one instance, this one would leave every outcome category
    of the report empty while its generated count still looks complete.
    """


def _entrypoint_path(repo: str) -> str:
    """Real absolute entrypoint path inside the container, which differs
    by base image (see module docstring point 2). Not something the host
    side can introspect from the .sif without actually running it, so
    this has to be a lookup, kept in sync with PYENV_REPOS.
    """
    if repo in PYENV_REPOS:
        return "/opt/entrypoint.sh"
    return "/home/swe-bench/entrypoint.sh"


# Steps that use OVERHEAD_TIMEOUT (no limit of their own), per the call sites
# in context_manager.py: per instance, the setup checks, the git operations
# that apply the patch, and the mutant-count script; per test run, `coverage
# json`; per sample with mutation, the scripts that skip non-target mutants
# and score the rest (or cr-rate and cr-report at whole-file scope).
OVERHEAD_STEPS_PER_INSTANCE = 12
OVERHEAD_STEPS_PER_RUN = 1
OVERHEAD_STEPS_PER_MUTATION = 2


def filter_runs(sample) -> int:
    """How many test-command runs filtering makes for one sample: one per test
    method and test function, parsed with the same function evaluate_instance
    uses (full_processing), so the count can't drift from what actually runs.
    Pattern matching only; no generated code runs on the host."""
    from swebench_docker.evaluate_instance import extract_preamble_classes_and_functions
    try:
        _, classes, functions = extract_preamble_classes_and_functions(sample or "", None)
    except Exception:
        return 1
    return len(functions) + sum(len(methods) for _, methods, _ in classes)


def host_timeout_for(timeout: int, n_samples: int, skip_mutation: bool, samples=None,
                     filtered: bool = True) -> int:
    """Host-side wall-clock budget for one container, which evaluates all of an
    instance's samples in turn (fyp-paper#7, open item 4: a budget of one
    `timeout` per sample killed sphinx-7305 and sphinx-10353 before their last
    samples were evaluated, while every step inside was within its own limit).

    With the samples' texts, the budget is

        B = B_instance + sum over samples of B_sample + margin

    built from limits the harness enforces inside the container:
    - B_instance: the setup command (`timeout`); with mutation, the target
      mutant count (`cosmic-ray init`, `timeout`); and its overhead steps;
    - B_sample: its test runs, each up to `timeout` plus `coverage json`
      (overhead): with filtering (the `full` setting), one per test method and
      function plus the combined run; otherwise one. With mutation,
      `cosmic-ray init` (`timeout`), the mutation run (MUTATION_TIMEOUT) and
      its reporting steps (overhead);
    - overhead steps are bounded by OVERHEAD_TIMEOUT each.

    It is a ceiling: steps normally finish far inside their limits, so fast
    instances aren't held longer; it only decides when a hung container is
    killed. Without the texts (samples=None), the earlier estimate of one
    `timeout` plus MUTATION_TIMEOUT per sample. APPTAINER_HOST_TIMEOUT
    overrides with a fixed value either way.
    """
    override = os.environ.get("APPTAINER_HOST_TIMEOUT")
    if override:
        return int(override)
    if samples is None:
        per_sample = timeout + (0 if skip_mutation else MUTATION_TIMEOUT)
        return max(1, n_samples) * per_sample + APPTAINER_HOST_TIMEOUT_MARGIN
    o = OVERHEAD_TIMEOUT
    b_instance = timeout + OVERHEAD_STEPS_PER_INSTANCE * o + (0 if skip_mutation else timeout)
    mutation = 0 if skip_mutation else timeout + MUTATION_TIMEOUT + OVERHEAD_STEPS_PER_MUTATION * o
    per_run = timeout + OVERHEAD_STEPS_PER_RUN * o
    b_samples = sum(((filter_runs(sample) + 1) if filtered else 1) * per_run + mutation
                    for sample in samples)
    return b_instance + max(b_samples, per_run) + APPTAINER_HOST_TIMEOUT_MARGIN


def parse_slurm_time(text):
    """Seconds in a Slurm time string ([D-]HH:MM:SS, MM:SS or SS), or None
    for UNLIMITED, NOT_SET or anything unreadable."""
    text = (text or "").strip()
    try:
        days, _, rest = text.rpartition("-")
        parts = [int(x) for x in rest.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        h, m, sec = parts[-3:]
        return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + sec
    except ValueError:
        return None


def slurm_seconds_left():
    """Seconds left in the enclosing Slurm job, or None outside one."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return None
    return parse_slurm_time(out)


async def run_apptainer_evaluation(
    task_instance: dict,
    namespace: str,
    log_dir: str,
    setting: str,
    ind: int,
    timeout: int = 60,
    verbose: bool = False,
    base64_instance: bool = True,
    only_baseline: bool = False,
    skip_mutation: bool = False,
):
    repo_name = task_instance["repo"].replace("/", "_")

    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
        task_instance["version"]
    ]

    # TODO: Change this when deciding
    if "packages" in specifications and specifications["packages"] == "environment.yml":
        container_log_dir = "/home/swe-bench/logs"
    else:
        container_log_dir = "/opt/logs"

    # Where converted .sif files live on M3, one file per
    # {repo_name}_{version}.sif, e.g. astropy_astropy_5.0.sif -- matches the
    # repo_name sanitization run_docker.py already uses. There is no default
    # (#122): it used to fall back to $HOME/apptainer_images, where M3 has no
    # images, so every instance was skipped with a warning and the run exited
    # 0 having evaluated nothing. Read per call, so a value exported after
    # import counts.
    images_dir = os.environ.get("APPTAINER_IMAGES_DIR")
    if not images_dir:
        raise ValueError(
            "APPTAINER_IMAGES_DIR must be set to the directory holding the "
            ".sif testbed images (on M3, under /fs04/scratch2/<project>/$USER)."
        )
    sif_path = os.path.join(
        images_dir, f"{repo_name}_{task_instance['version']}.sif"
    )
    if not os.path.exists(sif_path):
        logger.warning(
            f"[{task_instance['id']}][{sif_path}]  No .sif found for this "
            f"repo/version, skipping. Build and transfer it first (see "
            f"testgeneval#2)."
        )
        return

    entrypoint = _entrypoint_path(task_instance["repo"])
    entrypoint_parent = os.path.dirname(entrypoint)

    swebench_docker_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR")
    if not swebench_docker_fork_dir:
        raise ValueError(
            "SWEBENCH_DOCKER_FORK_DIR must be set for the Apptainer path -- "
            "unlike run_docker.py, there's no base64-INSTANCE fallback "
            "here, task_instance.json is always bind-mounted in."
        )

    tmpfile_path = tempfile.mktemp(suffix=".json")
    with open(tmpfile_path, "w+") as f:
        json.dump(task_instance, f)

    # A real, per-instance directory for --home, not the container default.
    # Without an explicit --home, Apptainer maps the container's $HOME to
    # the HOST user's real $HOME (confirmed real 2026-09-11: running two
    # instances concurrently, both executed `git config --global --add
    # safe.directory ...` inside entrypoint.sh, and both raced to write
    # the SAME real host file, /home/<user>/.gitconfig -- one always lost
    # with "could not lock config file: File exists". --cleanenv strips
    # environment variables, it does not isolate the filesystem. Every
    # earlier validation of this backend ran NUM_PROCESSES=1, so this
    # never surfaced until a real concurrency test did.
    home_dir = tempfile.mkdtemp(prefix="apptainer-home-")

    # evaluate_instance.py reads task_instance.json from a hardcoded
    # /home/swe-bench/task_instance.json (with a base64 INSTANCE env var
    # as its only fallback, which this path never sets). run_docker.py
    # mounts it to that exact path regardless of base image; do the same
    # rather than putting it next to the entrypoint. A pyenv image
    # (django, requests, scikit-learn) has no /home/swe-bench, but
    # --writable-tmpfs lets Apptainer create the mount point.
    task_json_target = "/home/swe-bench/task_instance.json"

    overlay_dir = None
    if task_instance["repo"] in REPOS_NEEDING_DISK_OVERLAY:
        overlay_dir = tempfile.mkdtemp(prefix="apptainer-overlay-")

    apptainer_command = [
        "apptainer",
        "exec",
        *_writable_args(overlay_dir),
        "--cleanenv",
        "--home",
        home_dir,
        "--pwd",
        entrypoint_parent,
        "-B",
        f"{log_dir}:{container_log_dir}",
        # :ro to match run_docker.py's deliberate read-only mount of
        # swebench_docker -- keeps a __pycache__ or stray artifact write
        # out of the user's checked-out fork on the host.
        "-B",
        f"{swebench_docker_fork_dir}/swebench_docker:{entrypoint_parent}/swebench_docker:ro",
        "-B",
        f"{tmpfile_path}:{task_json_target}",
        sif_path,
        entrypoint,
    ]

    # TESTBED_NAME, REPO_DIR, and IMAGE_TYPE are already baked into every
    # testbed image via its own Dockerfile's ENV directives (confirmed
    # across docker/*/*/Dockerfile) -- same as run_docker.py, which never
    # overrides them either. Only the per-run values that vary by
    # instance/settings need to be passed through here.
    env = {
        **os.environ,
        "APPTAINERENV_LOG_DIR": container_log_dir,
        "APPTAINERENV_SETTING": setting,
        "APPTAINERENV_IND": str(ind),
        "APPTAINERENV_TIMEOUT": str(timeout),
        "APPTAINERENV_ONLY_BASELINE": str(only_baseline),
        "APPTAINERENV_SKIP_MUTATION": str(skip_mutation),
        "APPTAINERENV_MUTATION_SCOPE": os.environ.get("MUTATION_SCOPE", "file"),
        "APPTAINERENV_MUTATION_TIMEOUT": str(MUTATION_TIMEOUT),
        "APPTAINERENV_OVERHEAD_TIMEOUT": str(OVERHEAD_TIMEOUT),
    }

    cmd_string = " ".join(apptainer_command)

    if verbose:
        logger.info(cmd_string)

    start_time = time.time()

    # Derived from this call's own `timeout` and sample count, so raising
    # TIMEOUT in m3_run_evaluation.slurm raises the host cap too, and a
    # pass@5 container gets room for all five samples (see host_timeout_for).
    samples = task_instance.get(KEY_PREDICTIONS, {}).get(setting)
    n_samples = len(samples) if isinstance(samples, list) else 1
    texts = samples if isinstance(samples, list) else [samples]
    host_timeout = host_timeout_for(timeout, n_samples, skip_mutation, samples=texts,
                                    filtered=(setting == "full"))
    logger.info(f"[{task_instance['id']}] host timeout {host_timeout}s "
                f"(sum of the inner limits over {n_samples} sample(s))")
    left = slurm_seconds_left()
    if left is not None and host_timeout > left:
        # The watchdog is no longer the binding limit: Slurm may end the job
        # first, cutting this container's later samples.
        logger.warning(f"[{task_instance['id']}] host timeout {host_timeout}s exceeds the "
                       f"{left}s left in Slurm job {os.environ.get('SLURM_JOB_ID')}; "
                       f"this container may be cut by the job's time limit")

    try:
        process = await asyncio.create_subprocess_exec(
            *apptainer_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=host_timeout
            )
        except asyncio.TimeoutError:
            # start_new_session=True puts the child in its own process
            # group; kill the whole group, not just the apptainer wrapper,
            # or the container process it spawned can survive as an
            # orphan and keep eating memory -- the exact failure mode this
            # timeout exists to prevent.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            logger.warning(
                f"[{task_instance['id']}][{sif_path}]  Container exceeded "
                f"host timeout of {host_timeout}s, killed."
            )
            return
        str_stdout = stdout.decode() if stdout else ""
        str_stderr = stderr.decode() if stderr else ""

        elapsed_time = time.time() - start_time

        if process.returncode != 0:
            logger.warning(
                f"[{task_instance['id']}][{sif_path}]  Error running container:"
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"Stdout - {str_stdout}")
            logger.warning(f"Stderr - {str_stderr}")

        elif "Evaluation succeeded" not in str_stdout:
            logger.warning(
                f"[{task_instance['id']}][{sif_path}]  Container ran successfully in {elapsed_time} seconds, but evaluation failed."
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"stdout - {str_stdout}")
        else:
            logger.info(
                f"[{task_instance['id']}][{sif_path}]  Container ran successfully in {elapsed_time} seconds."
            )
    except FileNotFoundError as e:
        raise ApptainerUnavailableError(
            "The apptainer binary was not found on PATH, nothing can be "
            "evaluated. Run on a node that has Apptainer available."
        ) from e
    except Exception as e:
        logger.warning(
            f"[{task_instance['id']}][{sif_path}]  Error running container: {e}"
        )
    finally:
        # Under real OOM pressure (job 60941163, 2026-10-14), the kernel killing
        # a task's subprocess mid-flight can leave this tmpfile already gone by
        # the time cleanup runs here, raising FileNotFoundError and crashing the
        # whole evaluation run instead of just this one instance.
        try:
            os.unlink(tmpfile_path)
        except FileNotFoundError:
            pass
        shutil.rmtree(home_dir, ignore_errors=True)
        if overlay_dir:
            shutil.rmtree(overlay_dir, ignore_errors=True)
