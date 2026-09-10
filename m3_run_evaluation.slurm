#!/bin/bash
#SBATCH --job-name="testgeneval-eval"
#SBATCH --partition=comp
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-eval-%j.out

# Runs testgeneval evaluation on an M3 CPU node via the Apptainer
# backend (run_evaluation.py --backend apptainer). No GPU: evaluation
# is Docker/Apptainer container execution of the generated tests plus
# cosmic-ray mutation testing, all CPU-bound.
#
# Prerequisites (see swebench_docker/run_apptainer.py and testgeneval#2):
#   - The .sif testbed images must already be present in
#     APPTAINER_IMAGES_DIR. Populate with:
#       python scripts/pull_apptainer_images.py --makefile Makefile.testgeneval
#     (rootless on a login node, or as its own CPU sbatch job for the
#     full 77-image set; ~9 min each, ~12h total).
#   - CONDA_ENV: a conda env on M3 with this repo's evaluation deps
#     (datasets, python-dotenv). Not the vLLM env; a plain one is fine.
#
# TODO before running:
#   PREDICTIONS_PATH: the predictions .jsonl to evaluate. For a large
#     run, shard it first (scripts/shard_predictions.py) and submit one
#     job per shard.
#   LOG_DIR: where per-instance .eval.log files land. MUST be on shared
#     scratch, not /tmp -- /tmp is node-local, so sharded jobs on
#     different nodes would each write to a different, invisible /tmp.
#     generate_report.py reads every .eval.log from one directory.
#   SWE_BENCH_TASKS: dataset id or path (default kjain14/testgeneval).
#   NUM_PROCESSES: concurrent container evaluations. Each peaks at
#     ~2-3 CPUs during a mutant's pytest run (measured 2026-09-11 on
#     astropy). 24 cpus-per-task / ~3 = ~8 is a safe start; raise after
#     watching real CPU and disk on one node. Per-user QOS cap is 256
#     CPUs total across all your jobs (sacctmgr show qos), so ~4 of
#     these jobs at NUM_PROCESSES up to ~20 each is the ceiling.
#   TIMEOUT: per-instance seconds (default 3600). cosmic-ray mutation
#     on a large repo genuinely takes 20-40 min; do not set this low.
#   SKIP_MUTATION: set to 1 to skip cosmic-ray entirely (coverage only).
#     Cuts per-instance time to ~3-5 min, but drops FunctionMutationLOG,
#     the primary RQ2/RQ3 metric.
#
# APPTAINER_TMPDIR is forced to node-local /tmp below (fast local disk,
# private to this job) -- the ephemeral squashfs overlay churn from many
# concurrent containers would otherwise hammer shared Lustre scratch and
# become the bottleneck (Apptainer disk-pressure hangs are documented in
# results/RUN_LOG.md). The .sif files and the pull cache stay on scratch.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-testgeneval}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-CHANGE_ME}"
LOG_DIR="${LOG_DIR:-$HOME/al49_scratch2/$USER/eval_logs}"
SWE_BENCH_TASKS="${SWE_BENCH_TASKS:-kjain14/testgeneval}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TIMEOUT="${TIMEOUT:-3600}"
SKIP_MUTATION="${SKIP_MUTATION:-0}"
NAMESPACE="${NAMESPACE:-kdjain}"

# Where the converted .sif testbed images live. One file per
# {repo_name}_{version}.sif (see run_apptainer.py). On scratch: they're
# large, read-only, and read once per container start.
APPTAINER_IMAGES_DIR="${APPTAINER_IMAGES_DIR:-$HOME/al49_scratch2/$USER/apptainer_images}"

# run_apptainer.py binds this repo's swebench_docker into the container.
SWEBENCH_DOCKER_FORK_DIR="${SWEBENCH_DOCKER_FORK_DIR:-$SLURM_SUBMIT_DIR}"

if [ "$PREDICTIONS_PATH" = "CHANGE_ME" ]; then
    echo "PREDICTIONS_PATH must be set to a real predictions .jsonl" >&2
    exit 1
fi
if [ ! -d "$APPTAINER_IMAGES_DIR" ] || [ -z "$(ls -A "$APPTAINER_IMAGES_DIR"/*.sif 2>/dev/null)" ]; then
    echo "No .sif files in APPTAINER_IMAGES_DIR ($APPTAINER_IMAGES_DIR)." >&2
    echo "Run scripts/pull_apptainer_images.py first." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# Node-local scratch for Apptainer's ephemeral overlay/tmp work. /tmp on
# an M3 compute node is local disk, wiped at job end, not shared Lustre.
APPTAINER_TMPDIR="/tmp/$USER-apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

export APPTAINER_TMPDIR
export APPTAINER_CACHEDIR="$HOME/al49_scratch2/$USER/apptainer-cache"
export APPTAINER_IMAGES_DIR
export SWEBENCH_DOCKER_FORK_DIR
mkdir -p "$APPTAINER_CACHEDIR"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"

EVAL_ARGS=(
    --predictions_path "$PREDICTIONS_PATH"
    --log_dir "$LOG_DIR"
    --swe_bench_tasks "$SWE_BENCH_TASKS"
    --backend apptainer
    --namespace "$NAMESPACE"
    --num_processes "$NUM_PROCESSES"
    --timeout "$TIMEOUT"
    --skip_existing
)
if [ "$SKIP_MUTATION" = "1" ]; then
    EVAL_ARGS+=(--skip_mutation)
fi

echo "Evaluating $PREDICTIONS_PATH"
echo "  log_dir=$LOG_DIR"
echo "  num_processes=$NUM_PROCESSES timeout=$TIMEOUT skip_mutation=$SKIP_MUTATION"
echo "  APPTAINER_TMPDIR=$APPTAINER_TMPDIR (node-local)"
echo "  images=$APPTAINER_IMAGES_DIR ($(ls "$APPTAINER_IMAGES_DIR"/*.sif 2>/dev/null | wc -l) .sif files)"

python run_evaluation.py "${EVAL_ARGS[@]}"

echo "Done. Per-instance logs in $LOG_DIR"
echo "Build the report with: python generate_report.py --log_dir $LOG_DIR ..."
