# Alternative to run_docker.py for M3, which does not support Docker but
# does support Apptainer. Every Docker-derived testbed image still has to
# be converted to a .sif file and pulled onto M3 by hand first (M3 requires
# sudo for `apptainer pull`/`build` from Docker Hub, which regular accounts
# don't have -- confirmed real, see testgeneval#2), this module only
# handles running an already-present .sif, the same way run_docker.py only
# handles running an already-pulled Docker image.
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

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time

import dotenv
from swebench_docker.constants import MAP_VERSION_TO_INSTALL, PYENV_REPOS

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Where converted .sif files live on M3, one file per
# {repo_name}_{version}.sif, e.g. astropy_astropy_5.0.sif -- matches the
# repo_name sanitization run_docker.py already uses (task_instance["repo"]
# with "/" replaced by "_").
APPTAINER_IMAGES_DIR = os.environ.get(
    "APPTAINER_IMAGES_DIR", os.path.expanduser("~/apptainer_images")
)


def _entrypoint_path(repo: str) -> str:
    """Real absolute entrypoint path inside the container, which differs
    by base image (see module docstring point 2). Not something the host
    side can introspect from the .sif without actually running it, so
    this has to be a lookup, kept in sync with PYENV_REPOS.
    """
    if repo in PYENV_REPOS:
        return "/opt/entrypoint.sh"
    return "/home/swe-bench/entrypoint.sh"


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

    sif_path = os.path.join(
        APPTAINER_IMAGES_DIR, f"{repo_name}_{task_instance['version']}.sif"
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

    apptainer_command = [
        "apptainer",
        "exec",
        "--writable-tmpfs",
        "--cleanenv",
        "--pwd",
        entrypoint_parent,
        "-B",
        f"{log_dir}:{container_log_dir}",
        "-B",
        f"{swebench_docker_fork_dir}/swebench_docker:{entrypoint_parent}/swebench_docker",
        "-B",
        f"{tmpfile_path}:{entrypoint_parent}/task_instance.json",
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
    }

    cmd_string = " ".join(apptainer_command)

    if verbose:
        logger.info(cmd_string)

    start_time = time.time()

    try:
        process = await asyncio.create_subprocess_exec(
            *apptainer_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        stdout, stderr = await process.communicate()
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
    except Exception as e:
        logger.warning(
            f"[{task_instance['id']}][{sif_path}]  Error running container: {e}"
        )
    finally:
        os.unlink(tmpfile_path)
