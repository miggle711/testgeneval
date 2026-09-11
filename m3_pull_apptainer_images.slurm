#!/bin/bash
#SBATCH --job-name="testgeneval-pull-sif"
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-pull-sif-%j.out

# Pulls every testbed Docker image the given Makefile references and
# converts each to a .sif for the Apptainer evaluation backend. One-time
# setup for run_evaluation.py --backend apptainer (see testgeneval#2).
#
# apptainer pull works rootless on M3 (confirmed 2026-09-11: astropy
# testbed pulled and converted to a 902MB .sif in ~9 min, no sudo). The
# full testgeneval Makefile has 126 distinct testbed images across 12
# repos, ~110GB of .sif, ~19h total -- hence a 48h sbatch job not an
# login-node run that could be killed.
#
# TODO before running:
#   MAKEFILE: Makefile.testgeneval (full, 126 images) or
#     Makefile.testgenevallite (38). Default: full.
#   NAMESPACE: kdjain (default) is the fork's Docker Hub account, whose
#     testbed images carry cosmic-ray and this repo's swebench_docker.
#     The upstream aorwall images do NOT (confirmed real: a mutation run
#     against an aorwall .sif fails with ModuleNotFoundError: cosmic_ray).
#   CONDA_ENV: any conda env on M3 with python 3 (only needs stdlib for
#     the pull script itself). Default: testgeneval.
#
# APPTAINER_CACHEDIR and APPTAINER_TMPDIR are both forced onto scratch
# below -- if either defaults to /tmp or $HOME the final mksquashfs step
# fills it and dies with "No space left on device" even when scratch has
# terabytes free (confirmed real, same day).

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-testgeneval}"
MAKEFILE="${MAKEFILE:-Makefile.testgeneval}"
APPTAINER_IMAGES_DIR="${APPTAINER_IMAGES_DIR:-$HOME/al49_scratch2/$USER/apptainer_images}"

export APPTAINER_CACHEDIR="$HOME/al49_scratch2/$USER/apptainer-cache"
export APPTAINER_TMPDIR="$HOME/al49_scratch2/$USER/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$APPTAINER_IMAGES_DIR"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"

echo "Pulling .sif images from $MAKEFILE into $APPTAINER_IMAGES_DIR"
df -h "$APPTAINER_IMAGES_DIR"

python scripts/pull_apptainer_images.py \
    --makefile "$MAKEFILE" \
    --out-dir "$APPTAINER_IMAGES_DIR"

echo "Done."
ls -lh "$APPTAINER_IMAGES_DIR"
du -sh "$APPTAINER_IMAGES_DIR"
