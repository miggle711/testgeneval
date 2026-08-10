#!/bin/bash
#SBATCH --job-name="testgeneval-inference"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Runs LLM inference for the testgeneval comparison (instruct vs kg_only arms)
# on an M3 GPU node, via vLLM serving an OpenAI-compatible endpoint (matches
# Aaron's working M3 setup: https://zenodo.org/records/18893780), then
# calling this repo's own inference.api.run_api (the same script already
# proven against local MLX and Groq -- one generation path for every
# environment instead of a second, parallel script).
#
# TODO before running:
#   - CONDA_ENV: name of a conda env on M3 with vllm pip-installed, plus
#     this repo's own deps (datasets, openai, tiktoken, python-dotenv --
#     see testgeneval.crossplatform.yaml in the sibling pycodekg repo for
#     the known-working non-vllm package list). No dedicated vllm module
#     exists on M3 (checked docs.erc.monash.edu/Compute/HPC/M3/Software/) --
#     `pip install vllm` into this env once, on a GPU node (per M3's Python
#     in HPC guidance: GPU-dependent packages should be installed on a GPU
#     node, not the login node).
#   - MODEL: real HuggingFace model id (e.g. Qwen/Qwen3-4B-Instruct-2507,
#     matching Aaron's example -- swap for whichever of the ~8 models this
#     job run is for).
#   - PROMPT_CONFIG / KG_PROMPTS_PATH: set per which arm this job runs
#     (instruct needs no extra args; kg_only needs a kg_prompts.json built
#     locally via pycodekg's scripts/build_kg_prompts.py and transferred
#     to M3 first -- this job does NOT build the KG itself)
#   - DATASET_PATH: the filtered dataset (e.g. testgenevallite_django,
#     produced locally via datasets.Dataset.save_to_disk) transferred to M3
#   - TEMPERATURE: 0 / 0.5 / 1 per the planned sensitivity sweep
#
# Model storage: the ~8 planned open-weight models range from ~15GB
# (7-8B) to ~140GB (70B) at float16 -- keeping several cached at once can
# exceed the team's 150GB model-storage budget (e.g. 5 of the originally
# discussed models total ~263GB together). DELETE_MODEL_AFTER_RUN=1
# removes this run's HuggingFace cache entry once inference finishes, so
# each model's full slice of jobs (all temperatures x both arms) can run
# back-to-back without manual cleanup, at the cost of re-downloading if a
# later job needs the same model again. Defaults to NOT deleting, since a
# job that's one of several back-to-back runs on the same model shouldn't
# force a redundant re-download for the next one.

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

mkdir -p "$OUTPUT_DIR"

module load miniforge3
conda activate "$CONDA_ENV"

cd "$(dirname "$0")"

# ---- Start vLLM, serving an OpenAI-compatible endpoint on this node ----
# Matches Aaron's working command (https://zenodo.org/records/18893780):
#   CUDA_VISIBLE_DEVICES=0 vllm serve <model> --trust-remote-code \
#     --dtype float16 --max-num-seqs 8 --port 8003
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --trust-remote-code \
    --dtype float16 \
    --max-num-seqs 8 \
    --port "$VLLM_PORT" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Wait for the server to actually be up before calling it -- a fixed sleep
# is a placeholder; poll /v1/models instead if model load time varies a lot
# across the ~8 planned models.
sleep 60

# ---- Run inference via this repo's proven run_api.py ----
# NOTE: base_url needs a trailing slash (LOCAL_MODEL_BASE_URL=.../v1/, not
# .../v1) -- the openai client concatenates the path without a separator
# otherwise and every request 404s. This bit us with the local MLX/Groq
# setup earlier; Aaron's example client code omits /v1 entirely, which is
# a separate, real bug worth flagging to him if his script is used as-is.
EXTRA_ARGS=(--skip_full)
if [ "$PROMPT_CONFIG" = "kg_only" ]; then
    EXTRA_ARGS+=(--prompt_config kg_only --kg_prompts_path "$KG_PROMPTS_PATH")
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

# ---- Optional: free this model's disk space now that inference is done ----
if [ "$DELETE_MODEL_AFTER_RUN" = "1" ]; then
    echo "DELETE_MODEL_AFTER_RUN=1 -- removing HuggingFace cache for $MODEL"
    # huggingface-cli's own cache tooling resolves the real cache dir for
    # this exact model id rather than guessing a path -- safer than a
    # hand-built rm -rf against ~/.cache/huggingface/hub/models--org--name,
    # which breaks silently if HF's cache-dir naming scheme ever changes.
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
