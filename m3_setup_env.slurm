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
#
# 3.12, not just 3.10+: vllm's flashinfer dependency uses
# array.array[int] as a real type hint, which only became
# subscriptable in Python 3.12 (confirmed real on M3, 3.11 crashed
# loading the model with "type 'array.array' is not subscriptable").

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-testgeneval-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

module load miniforge3
# conda activate needs the shell hook sourced first in a non
# interactive script, module load alone doesn't always wire it up.
source "$(conda info --base)/etc/profile.d/conda.sh"

# conda create fails outright if the env already exists. Remove any
# prior attempt first so this script can be re-run after a fix,
# not just run once.
conda env remove -y -n "$CONDA_ENV" 2>/dev/null || true

conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
conda activate "$CONDA_ENV"

# The env's own libstdc++ is newer than the system one, but isn't
# found first without this. Needed for vllm to import at all
# (confirmed real on M3, see m3_run_inference.slurm's comment).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

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
