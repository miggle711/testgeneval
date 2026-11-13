# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/run_evaluation.py
"""Run evaluation"""
import argparse
import asyncio
import hashlib
import json
import logging
import os

from swebench_docker.constants import (
    KEY_ID,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTIONS,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
    MAP_REPO_TO_TEST_FRAMEWORK,
)
from swebench_docker.run_apptainer import run_apptainer_evaluation
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.swebench_utils import get_instances, get_test_directives
from swebench_docker.utils import get_eval_refs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")


def validate_predictions(predictions_path, tasks_ids):
    # Check that predictions file exists
    if not any([predictions_path.endswith(x) for x in [".jsonl"]]):
        raise ValueError("Predictions path must be .jsonl file")
    predictions = get_instances(predictions_path)
    not_in_tasks = []
    # Check that predictions are correctly formatted
    for pred in predictions:
        if any(
            [
                x not in pred
                for x in [KEY_ID, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTIONS]
            ]
        ):
            raise ValueError(
                f"Every prediction must have {KEY_INSTANCE_ID}, {KEY_MODEL}, {KEY_ID}, and {KEY_PREDICTIONS} fields"
            )
        if pred[KEY_ID] not in tasks_ids:
            not_in_tasks.append(pred[KEY_ID])
    # Check that instance IDs specified by predictions exist
    if len(not_in_tasks) > 0:
        logger.warning(
            "Predictions for the following instance_ids were not "
            + "found in the tasks file and will not be considered: "
            + ", ".join(not_in_tasks)
        )


def completed_samples(log_file: str) -> int:
    """How many samples a `full` setting log finished. Each finished sample
    ends with one Unfiltered Tests Passed/Failed marker; a container killed
    partway leaves fewer than there are samples."""
    with open(log_file, errors="replace") as f:
        content = f.read()
    return content.count(UNFILTERED_TESTS_PASSED) + content.count(UNFILTERED_TESTS_FAILED)


def set_aside_partial_log(log_file: str) -> str:
    """Rename an incomplete log to <log>.partial (or .partial.N if that is
    taken), keeping it for audit, so a rerun writes a fresh log rather than
    adding samples to the old one. Returns the new path."""
    dest = f"{log_file}.partial"
    n = 1
    while os.path.exists(dest):
        dest = f"{log_file}.partial.{n}"
        n += 1
    os.rename(log_file, dest)
    return dest


SOURCE_SUFFIX = ".source.json"


def log_source(samples, skip_mutation: bool) -> dict:
    """What a log's results come from: the generated samples it scores and
    the settings that change what gets scored. A log is named by id, model
    and setting only, so the two arms, two sample counts or two mutation
    settings of one model all map to the same file name."""
    return {
        "samples_sha256": hashlib.sha256(
            json.dumps(samples, sort_keys=True).encode()
        ).hexdigest(),
        "num_samples": len(samples) if isinstance(samples, list) else 1,
        "skip_mutation": skip_mutation,
        "mutation_scope": os.environ.get("MUTATION_SCOPE", "file"),
    }


def _log_path(log_dir: str, prediction: dict, setting: str) -> str:
    return os.path.join(
        log_dir, f"{prediction[KEY_ID]}.{prediction[KEY_MODEL]}.{setting}.eval.log"
    )


def check_log_sources(predictions: list, log_dir: str, skip_mutation: bool) -> None:
    """Raise if log_dir already holds a log for one of these instances that
    was produced from other samples or settings. Without this the second run
    is either skipped as already done or replaces the first run's log, and
    both look like a normal run. Compared per instance, so shards and
    resubmits of the same predictions still share a directory. Logs written
    before this check existed have no record and are not checked."""
    clashes = []
    for p in predictions:
        for setting, samples in p.get(KEY_PREDICTIONS, {}).items():
            log_file = _log_path(log_dir, p, setting)
            record = log_file + SOURCE_SUFFIX
            if not (os.path.exists(log_file) and os.path.exists(record)):
                continue
            with open(record) as f:
                recorded = json.load(f)
            expected = log_source(samples, skip_mutation)
            differing = sorted(k for k in expected if recorded.get(k) != expected[k])
            if differing:
                clashes.append((p[KEY_ID], differing, recorded.get("predictions_path")))
    if clashes:
        instance, differing, recorded_from = clashes[0]
        raise ValueError(
            f"{len(clashes)} instance(s) in {log_dir} already have a log from "
            f"different predictions or settings. First: {instance} differs in "
            f"{', '.join(differing)} (its log was evaluated from {recorded_from}). "
            f"A log directory holds one run: one arm, one sample count, one "
            f"mutation setting. Use a separate --log_dir for this run."
        )


def record_log_sources(
    predictions: list, log_dir: str, skip_mutation: bool, predictions_path: str
) -> None:
    """Write each log's source record before its container starts."""
    for p in predictions:
        for setting, samples in p.get(KEY_PREDICTIONS, {}).items():
            record = _log_path(log_dir, p, setting) + SOURCE_SUFFIX
            source = {**log_source(samples, skip_mutation), "predictions_path": predictions_path}
            # Replace in one step: a job killed mid-write must not leave a
            # truncated record that blocks the resubmit.
            tmp = f"{record}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(source, f)
            os.replace(tmp, record)




def make_log_dir_writable(log_dir: str) -> None:
    """Make log_dir world-writable, so containers that do not run as its
    owner can write logs there.

    Only the owner may change a directory's mode, so an unconditional chmod
    failed with PermissionError on a directory another account owns, even
    one that was already 777 (#120: all 20 of a teammate's shards failed
    this way). Nothing to do when the mode is already 777; when the chmod is
    not allowed, warn and carry on, since group or other write access may
    still be enough.
    """
    if os.stat(log_dir).st_mode & 0o777 == 0o777:
        return
    try:
        os.chmod(log_dir, 0o777)
    except PermissionError:
        logger.warning(
            f"Could not make {log_dir} world-writable (it belongs to another "
            f"account). Containers can write logs there only if its "
            f"permissions already allow it."
        )


async def main(
    predictions_path: str,
    swe_bench_tasks: str,
    namespace: str,
    log_dir: str,
    skip_existing: bool = False,
    timeout: int = 900,
    num_processes: int = -1,
    skip_mutation: bool = False,
    backend: str = "docker",
):
    """
    Runs evaluation on predictions for each model/repo/version combination.

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        namespace (str): Docker repository namespace.
        log_dir (str): Path to the directory where logs will be saved.
        skip_existing (bool): Whether to skip evaluations for predictions that already have logs.
        timeout (int): Timeout for each evaluation.
        num_processes (int): Number of processes to run in parallel (-1 = unlimited)

    Raises:
        ValueError: If log_dir is not a directory, testbed is not a directory, or swe_bench_tasks does not exist.
    """
    # Validate arguments
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")

    make_log_dir_writable(log_dir)

    tasks = list(get_eval_refs(swe_bench_tasks).values())

    # Verify arguments are formatted correctly
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_ID]: t for t in tasks}
    predictions_path = os.path.abspath(predictions_path)
    validate_predictions(predictions_path, [t[KEY_ID] for t in tasks])

    predictions = get_instances(predictions_path)

    if len(predictions) == 0:
        logger.info("No predictions to evaluate")
        return

    check_log_sources(predictions, log_dir, skip_mutation)

    # Remove predictions that have already been evaluated
    if skip_existing:
        # Skip logs that already exist
        predictions_filtered = []
        for p in predictions:
            all_exist = True
            if KEY_PREDICTIONS not in p:
                continue
            for setting in p[KEY_PREDICTIONS]:
                log_file = _log_path(log_dir, p, setting)
                log_file_name = os.path.basename(log_file)
                if not os.path.exists(log_file):
                    all_exist = False
                    break
                # A log existing is not enough: a container killed partway
                # leaves a log with fewer finished samples than there are.
                samples = p[KEY_PREDICTIONS][setting]
                expected = len(samples) if isinstance(samples, list) else 1
                if setting == "full" and completed_samples(log_file) < expected:
                    moved = set_aside_partial_log(log_file)
                    logger.info(
                        f"{log_file_name}: fewer than {expected} samples finished, "
                        f"re-evaluating (partial log kept as {os.path.basename(moved)})"
                    )
                    all_exist = False
                    break
            if not all_exist:
                predictions_filtered.append(p)
        if len(predictions_filtered) == 0:
            logger.info(f"All predictions already exist, skipping")
            return
        else:
            logger.info(
                f"# of predictions to evaluate: {len(predictions_filtered)} "
                + f"({len(predictions) - len(predictions_filtered)} already evaluated)"
            )
            predictions = predictions_filtered
    else:
        logger.info(f"# of predictions to evaluate: {len(predictions)}")

    record_log_sources(predictions, log_dir, skip_mutation, predictions_path)

    task_instances = []

    # Set the relevant data on task_instances
    for prediction in predictions:
        task = tasks_map[prediction[KEY_ID]]

        test_type = MAP_REPO_TO_TEST_FRAMEWORK[task["repo"]]
        test_directives = get_test_directives(task)
        test_cmd = f"{test_type} {' '.join(test_directives)}"

        task_instances.append(
            {
                "repo": task["repo"],
                "version": task["version"],
                "base_commit": task["base_commit"],
                KEY_ID: prediction[KEY_ID],
                KEY_INSTANCE_ID: prediction[KEY_INSTANCE_ID],
                KEY_MODEL: prediction[KEY_MODEL],
                KEY_PREDICTIONS: prediction[KEY_PREDICTIONS],
                "preds_context": task["preds_context"],
                "test_patch": task["test_patch"],
                "test_file": task["test_file"],
                "code_file": task["code_file"],
                "patch": task["patch"],
                "test_directives": test_directives,
                "test_cmd": test_cmd,
            }
        )

    task_instances = sorted(task_instances, key=lambda x: x[KEY_ID])

    run_evaluation_fn = (
        run_apptainer_evaluation if backend == "apptainer" else run_docker_evaluation
    )

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(task_instances))
    tasks = []
    for task_instance in task_instances:
        if task_instance[KEY_PREDICTIONS]:

            async def run_throttled(*args, **kwargs):
                async with sem:
                    return await run_evaluation_fn(
                        *args,
                        **kwargs,
                        only_baseline=False,
                        skip_mutation=skip_mutation,
                    )

            for ind, setting in enumerate(task_instance[KEY_PREDICTIONS]):
                task = asyncio.create_task(
                    run_throttled(
                        task_instance, namespace, log_dir, setting, ind, timeout=timeout
                    )
                )
                tasks.append(task)
        else:
            logger.info(f"[{task_instance[KEY_ID]}] No prediction found.")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_path", type=str, help="Path to predictions file", required=True
    )
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory", required=True
    )
    parser.add_argument(
        "--swe_bench_tasks",
        type=str,
        help="Path to dataset file or HF datasets name",
        required=True,
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Docker repository namespace",
        required=False,
        default="aorwall",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="(Optional) Skip existing logs"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="(Optional) Timeout in seconds (default: 60)",
        default=60,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="(Optional) Number of processes to run in parallel (-1 for unlimited)",
        default=-1,
    )
    parser.add_argument(
        "--skip_mutation", action="store_true", help="(Optional) Skip mutation"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["docker", "apptainer"],
        default="docker",
        help="(Optional) Container backend. 'docker' (default) is "
             "unchanged from before. 'apptainer' is for M3, which does "
             "not support Docker -- requires .sif files already present "
             "in APPTAINER_IMAGES_DIR (populate with "
             "scripts/pull_apptainer_images.py; apptainer pull works "
             "rootless on an M3 login node, see "
             "swebench_docker/run_apptainer.py and testgeneval#2).",
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
