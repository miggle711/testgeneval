#!/bin/bash
#SBATCH --job-name="testgeneval-inference"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
VLLM_PORT="${VLLM_PORT:-8003}"
TEMPERATURE="${TEMPERATURE:-0.2}"
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

# Start vLLM, serving an OpenAI-compatible endpoint on this node.
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --trust-remote-code \
    --dtype float16 \
    --max-num-seqs 8 \
    --port "$VLLM_PORT" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Fixed wait for the server to come up. Placeholder, poll /v1/models
# instead if model load time varies a lot across models.
sleep 60

# Only the full setting is in scope, skip completion.
EXTRA_ARGS=(--skip_completion --kg_prompts_path "$KG_PROMPTS_PATH")
if [ "$PROMPT_CONFIG" = "kg_only" ]; then
    EXTRA_ARGS+=(--prompt_config kg_only)
fi

LOCAL_MODEL_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1/" \
    python -m inference.api.run_api \
    --model_name_or_path "$MODEL" \
    --dataset_name_or_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 1 \
    --model_args "temperature=${TEMPERATURE}" \
    "${EXTRA_ARGS[@]}"

echo "Done. Predictions written under $OUTPUT_DIR"
echo "Transfer that output back off M3 (e.g. scp/rsync) for Docker-based"
echo "evaluation, since M3 does not support Docker -- see run_evaluation.py"
echo "which must run elsewhere."

# Optional: free this model's disk space now that inference is done.
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
