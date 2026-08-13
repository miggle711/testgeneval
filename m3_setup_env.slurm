#!/bin/bash
#SBATCH --job-name="testgeneval-env-setup"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out

# One time setup for the conda env m3_run_inference.slurm expects
# (CONDA_ENV, testgeneval-vllm by default). Run this once, then every
# inference job just does a fast conda activate.
#
# vllm is installed here, on a GPU node, not the login node, since its
# install involves CUDA linked wheels and GPU capability detection.
#
# vllm needs Python 3.10+. This repo's own crossplatform.yaml pins
# python=3.9 for unrelated reasons (its dependency set covers this
# whole repo, not just the inference path), so this env is built with
# its own newer Python instead of matching that file.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-testgeneval-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

module load miniforge3
# conda activate needs the shell hook sourced first in a non
# interactive script, module load alone doesn't always wire it up.
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
conda activate "$CONDA_ENV"

# vllm first, then everything else unpinned on openai and numpy so
# pip's resolver keeps whatever versions vllm actually needs. Pinning
# openai==1.35.9 here fought vllm's own openai>=2.0.0 requirement and
# pip silently downgraded it anyway (confirmed real on M3). run_api.py
# only uses openai's older module level API style (openai.api_key =
# ..., openai.chat.completions.create), confirmed that still works
# fine on openai 2.x. numpy is imported in run_api.py but never
# actually used, so it doesn't need a specific version either.
pip install vllm

pip install \
    "python-dotenv==1.0.1" \
    "tiktoken==0.7.0" \
    "datasets==2.18.0" \
    "frozendict==2.4.4" \
    "tenacity==8.4.2" \
    "tqdm==4.66.4"

echo "Env $CONDA_ENV ready. m3_run_inference.slurm can now use it."
