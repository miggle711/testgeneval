#!/bin/bash
#SBATCH --job-name="testgeneval-inference"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Runs testgeneval inference (instruct vs kg_only) on an M3 GPU node.
# vLLM serves an OpenAI-compatible endpoint, then this repo's own
# inference.api.run_api calls it, same as the local/Groq paths.
#
# TODO before running:
#   CONDA_ENV: conda env on M3 with vllm plus this repo's deps
#     (datasets, openai, tiktoken, python-dotenv). Install on a GPU
#     node, not the login node.
#   HF_HOME: defaults to al49 scratch space now, override if needed.
#   MODEL: real HuggingFace model id for the model this job is for.
#   PROMPT_CONFIG / KG_PROMPTS_PATH: kg_only needs a kg_prompts.json
#     built locally (pycodekg's build_kg_prompts.py) and transferred.
#   DATASET_PATH: filtered dataset, transferred to M3.
#   TEMPERATURE: 0 / 0.5 / 1 per the sensitivity sweep.
#   MAX_CONCURRENCY: requests run_api.py keeps in flight at once. vLLM
#     serves requests one at a time by default, leaving continuous
#     batching and GPU KV cache mostly unused. Keep it <= --max-num-seqs
#     below (8) -- vLLM won't run more concurrent sequences regardless
#     of what's requested. Defaults to 1 (unchanged, sequential); watch
#     vLLM's own GPU KV cache usage log line before raising it on a new
#     model/GPU combination.
#   SHARD_ID / NUM_SHARDS: optional, split the dataset across several
#     parallel jobs for a large run (confirmed real on M3 that this
#     project can run multiple GPU jobs at once, see testgeneval#23).
#     Submit one job per shard id, same NUM_SHARDS each time.
#   TENSOR_PARALLEL_SIZE: for models too large for one GPU (e.g. a 32B
#     model needs ~64GB at float16, more than one L40S's 48GB). Splits
#     the model itself across N GPUs in one job, not N separate jobs
#     like sharding. The #SBATCH --gres line above only requests 1 GPU
#     by default -- override it at submission time to match, e.g.:
#       TENSOR_PARALLEL_SIZE=2 sbatch --gres=gpu:L40S:2 m3_run_inference.slurm
#     #SBATCH directives can't read shell/env vars, so --gres has to be
#     passed on the sbatch command line, not just as this env var.
#
# Model storage: the planned models range ~15GB to ~140GB at float16,
# easily exceeding the team's model storage budget if several are
# cached at once. DELETE_MODEL_AFTER_RUN=1 frees this run's model
# after inference finishes. Defaults off, since back to back jobs on
# the same model shouldn't force a redundant redownload.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-testgeneval-vllm}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PROMPT_CONFIG="${PROMPT_CONFIG:-instruct}"          # instruct | kg_only
KG_PROMPTS_PATH="${KG_PROMPTS_PATH:-kg_prompts.json}"
DATASET_PATH="${DATASET_PATH:-CHANGE_ME}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results/${PROMPT_CONFIG}}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"   # requests in flight at once; match vLLM's --max-num-seqs below
# For splitting a large dataset across parallel jobs (confirmed real
# on M3: this project can run several GPU jobs at once, see
# testgeneval#23). Both unset by default, matching run_api.py's own
# --shard_id/--num_shards default of processing everything in one job.
SHARD_ID="${SHARD_ID:-}"
NUM_SHARDS="${NUM_SHARDS:-}"
# Each shard needs its own vLLM port so parallel jobs landing on the
# same node don't collide. Offsets from a base port by shard id.
VLLM_PORT="${VLLM_PORT:-$((8003 + ${SHARD_ID:-0}))}"
# GPUs to split the model across for one job (see the TENSOR_PARALLEL_SIZE
# note above the TODO block). 1 keeps the existing single-GPU behavior.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# Some models default to a very large max sequence length that needs
# more KV cache than a single GPU has (confirmed real on M3:
# Qwen3-4B-Instruct-2507 defaults to 262144, needs 36 GiB, only
# 31.62 GiB was available). This experiment's prompts and generated
# tests are nowhere near that long, so cap it.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Some models' chat templates reject a system role entirely (confirmed
# real on M3: bigcode/starcoder2-15b-instruct-v0.1 400s on every single
# request with "System messages are not allowed in this template").
# Set NO_SYSTEM_MESSAGE=1 for those, folds the system message into the
# user turn instead.
NO_SYSTEM_MESSAGE="${NO_SYSTEM_MESSAGE:-0}"
DELETE_MODEL_AFTER_RUN="${DELETE_MODEL_AFTER_RUN:-0}"   # 1 = delete this model's HF cache when done
# Scratch space, not $HOME. al49_scratch is shared across the whole
# project, not per user, so this needs its own subfolder under it or
# it can collide with a teammate's existing, differently owned cache
# (confirmed real on M3: another teammate's hf-cache already existed
# there, not group writable, every download failed with a permission
# error).
HF_HOME="${HF_HOME:-/home/$USER/al49_scratch/$USER/hf-cache}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME"
export HF_HOME

# flashinfer (a vllm dependency) JIT compiles some CUDA kernels on
# first use, which needs nvcc. The conda env has PyTorch's bundled
# CUDA runtime but not a full toolkit, nvcc isn't on PATH without
# this (confirmed real on M3: "/usr/local/cuda/bin/nvcc: No such
# file or directory").
module load cuda/12.8
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# The env's own libstdc++ is newer than the system one, but isn't
# found first without this. Without it, vllm fails to import
# (confirmed real on M3: sqlite3 pulls in libicui18n, which needs a
# CXXABI version the system libstdc++.so.6 doesn't have).
# LD_LIBRARY_PATH isn't always already set, :- keeps set -u happy.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# dirname "$0" is unreliable under sbatch, SLURM can stage or invoke
# the script in a way that leaves it resolving to "." (confirmed
# real on M3: python -m inference.api.run_api failed with
# ModuleNotFoundError, not actually in the repo directory).
# SLURM sets SLURM_SUBMIT_DIR to the real directory sbatch was run
# from, use that instead.
cd "$SLURM_SUBMIT_DIR"

# vLLM's own model loading (via transformers' hub code) doesn't reliably
# pick up `hf auth login`'s cached token under sbatch, even though the
# hf CLI does, export it explicitly so gated models (Llama, CodeGemma)
# actually authenticate. Falls back to the CLI's cached token file if
# HF_TOKEN isn't already set.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN

# Start vLLM, serving an OpenAI-compatible endpoint on this node.
# CUDA_VISIBLE_DEVICES lists every GPU this job's own SLURM allocation
# gives it (0..TENSOR_PARALLEL_SIZE-1, SLURM isolates each job to its
# own view starting at 0 regardless of the physical device numbers on
# the node), so vLLM's tensor parallelism has all of them to split the
# model across.
CUDA_VISIBLE_DEVICES=$(seq 0 $((TENSOR_PARALLEL_SIZE - 1)) | paste -sd, -) vllm serve "$MODEL" \
    --trust-remote-code \
    --dtype float16 \
    --max-num-seqs 8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --port "$VLLM_PORT" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Poll /v1/models instead of a fixed sleep, and fail loudly if the
# server never comes up instead of running the full dataset against
# a dead endpoint (confirmed real on M3: vllm crashed at startup,
# but the old fixed sleep 60 had no way to notice and run_api.py
# silently "processed" the whole dataset against nothing, tenacity
# retrying and giving up per instance -- see testgeneval#20).
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"
elapsed=0
until curl -s -f "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vllm server process exited before becoming ready. Check the log above for its real startup error." >&2
        exit 1
    fi
    if [ "$elapsed" -ge "$VLLM_STARTUP_TIMEOUT" ]; then
        echo "vllm server did not become ready within ${VLLM_STARTUP_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "vllm server ready after ${elapsed}s."

# Only the full setting is in scope, skip completion.
EXTRA_ARGS=(--skip_completion --kg_prompts_path "$KG_PROMPTS_PATH")
if [ "$PROMPT_CONFIG" = "kg_only" ]; then
    EXTRA_ARGS+=(--prompt_config kg_only)
fi
if [ -n "$SHARD_ID" ] && [ -n "$NUM_SHARDS" ]; then
    EXTRA_ARGS+=(--shard_id "$SHARD_ID" --num_shards "$NUM_SHARDS")
fi
if [ "$NO_SYSTEM_MESSAGE" = "1" ]; then
    EXTRA_ARGS+=(--no_system_message)
fi

LOCAL_MODEL_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1/" \
    python -m inference.api.run_api \
    --model_name_or_path "$MODEL" \
    --dataset_name_or_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 1 \
    --max_concurrency "$MAX_CONCURRENCY" \
    --model_args "temperature=${TEMPERATURE}" \
    "${EXTRA_ARGS[@]}"

echo "Done. Predictions written under $OUTPUT_DIR"
echo "Transfer that output back off M3 (e.g. scp/rsync) for Docker-based"
echo "evaluation, since M3 does not support Docker -- see run_evaluation.py"
echo "which must run elsewhere."

# Optional: free this model's disk space now that inference is done.
# Unsafe to combine with sharding if other shards of the same model
# might still be running, this would delete the cache out from under
# them. Leave off (default) when running shards in parallel.
if [ "$DELETE_MODEL_AFTER_RUN" = "1" ]; then
    echo "DELETE_MODEL_AFTER_RUN=1 -- removing HuggingFace cache for $MODEL"
    # Uses huggingface_hub's own cache resolver instead of guessing a path.
    python3 -c "
from huggingface_hub import scan_cache_dir
model_id = '$MODEL'
cache = scan_cache_dir()
matches = [r for r in cache.repos if r.repo_id == model_id]
if not matches:
    print(f'No cached files found for {model_id}, nothing to delete.')
else:
    for repo in matches:
        size_gb = repo.size_on_disk / 1e9
        print(f'Deleting {repo.repo_id} ({size_gb:.1f} GB) from {repo.repo_path}')
        strategy = cache.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
        strategy.execute()
"
fi
