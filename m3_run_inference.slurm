#!/bin/bash
#SBATCH --job-name="testgeneval-inference"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Runs LLM inference for the testgeneval comparison (instruct vs kg_only arms)
# on an M3 GPU node, via Ollama serving an OpenAI-compatible endpoint, then
# calling this repo's own inference.api.run_api (the same script already
# proven against local MLX and Groq -- one generation path for every
# environment instead of a second, parallel script).
#
# TODO before running:
#   - PROJECT: set to your real M3 project id (used for conda's scratch dirs)
#   - OLLAMA_MODEL: pick a real Ollama model tag (e.g. llama3.1:8b-instruct-q8_0)
#   - CONDA_ENV: name of a conda env with this repo's Python deps installed
#     (datasets, openai, tiktoken, python-dotenv, transformers, etc. --
#     see testgeneval.crossplatform.yaml in the sibling pycodekg repo for
#     the known-working package list, though M3 needs its own env, not a
#     copy of the local one)
#   - PROMPT_CONFIG / KG_PROMPTS_PATH: set per which arm this job runs
#     (instruct needs no extra args; kg_only needs a kg_prompts.json built
#     locally via pycodekg's scripts/build_kg_prompts.py and transferred
#     to M3 first -- this job does NOT build the KG itself)
#   - DATASET_PATH: the filtered dataset (e.g. testgenevallite_django,
#     produced locally via datasets.Dataset.save_to_disk) transferred to M3

set -euo pipefail

PROJECT="${PROJECT:-CHANGE_ME}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b-instruct-q8_0}"
CONDA_ENV="${CONDA_ENV:-testgeneval}"
PROMPT_CONFIG="${PROMPT_CONFIG:-instruct}"          # instruct | kg_only
KG_PROMPTS_PATH="${KG_PROMPTS_PATH:-kg_prompts.json}"
DATASET_PATH="${DATASET_PATH:-CHANGE_ME}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results/${PROMPT_CONFIG}}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

mkdir -p "$OUTPUT_DIR"

# ---- Start Ollama, serving an OpenAI-compatible endpoint on this node ----
# https://docs.erc.monash.edu/Compute/HPC/M3/Software/Ollama/
module load ollama
SCRATCH_FOLDER_PATH="/scratch/${PROJECT}/${USER}"
export OLLAMA_MODELS="${SCRATCH_FOLDER_PATH}/ollama-models"
mkdir -p "$OLLAMA_MODELS"
export APPTAINERENV_OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"

# $OLLAMA_SIF is defined by the ollama module itself.
apptainer run --nv "$OLLAMA_SIF" &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

# Wait for the server to actually be up before pulling/running a model --
# a fixed sleep is a placeholder; poll instead if this proves flaky.
sleep 15

apptainer exec --nv "$OLLAMA_SIF" ollama pull "$OLLAMA_MODEL"

# Run the model as a background server process so it stays warm for the
# whole run_api.py invocation below, rather than one-shotting a single
# prompt via `ollama run`.
apptainer exec --nv "$OLLAMA_SIF" ollama run "$OLLAMA_MODEL" --keepalive 60m &
sleep 10

# ---- Run inference via this repo's proven run_api.py ----
module load miniforge3
conda activate "$CONDA_ENV"

cd "$(dirname "$0")"

EXTRA_ARGS=(--skip_full)
if [ "$PROMPT_CONFIG" = "kg_only" ]; then
    EXTRA_ARGS+=(--prompt_config kg_only --kg_prompts_path "$KG_PROMPTS_PATH")
fi

LOCAL_MODEL_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1/" \
    python -m inference.api.run_api \
    --model_name_or_path "$OLLAMA_MODEL" \
    --dataset_name_or_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 1 \
    "${EXTRA_ARGS[@]}"

echo "Done. Predictions written under $OUTPUT_DIR"
echo "Transfer that output back off M3 (e.g. scp/rsync) for Docker-based"
echo "evaluation, since M3 does not support Docker -- see run_evaluation.py"
echo "which must run elsewhere."
