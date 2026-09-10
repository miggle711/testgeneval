# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pull every testbed Docker image a Makefile references and convert each to
a .sif file for the Apptainer evaluation backend (run_evaluation.py
--backend apptainer, see swebench_docker/run_apptainer.py and
testgeneval#2).

Runs rootless on an M3 login node -- `apptainer pull docker://<image>`
does not need sudo (only `apptainer build` from a .def file does).
Confirmed real 2026-09-11: an astropy testbed pulled and converted to a
902MB .sif in ~9 min.

Two environment variables MUST point at scratch before running this, or
the final mksquashfs step fills /tmp or $HOME and dies with "No space
left on device" even when scratch has terabytes free:

    export APPTAINER_CACHEDIR="/fs04/scratch2/<proj>/$USER/apptainer-cache"
    export APPTAINER_TMPDIR="/fs04/scratch2/<proj>/$USER/apptainer-tmp"

Output filenames match what run_apptainer.py expects:
{repo_name}_{version}.sif, e.g. astropy_astropy_5.1.sif, where repo_name
is task_instance["repo"] with "/" replaced by "_".

Full testgeneval Makefile defines 77 distinct testbed images (~40GB of
.sif, ~11-12h total at ~9 min each); testgenevallite defines 38.

Usage:
    python scripts/pull_apptainer_images.py \
        --makefile Makefile.testgeneval \
        --out-dir "$APPTAINER_IMAGES_DIR"
"""

import argparse
import os
import re
import subprocess
import sys

# swe-bench-<repo_underscored>-testbed:<version>, e.g.
# swe-bench-astropy_astropy-testbed:5.1 -> repo_name "astropy_astropy",
# version "5.1". The repo part already carries the underscore form
# run_apptainer.py wants, so no further mapping is needed.
_TESTBED_RE = re.compile(
    r"(?P<namespace>aorwall|kdjain)/swe-bench-(?P<repo>[a-z0-9_]+)-testbed:(?P<version>[0-9.]+)"
)


def parse_makefile(makefile_path):
    """Return a sorted, de-duplicated list of (namespace, repo, version)
    for every testbed image the Makefile references."""
    seen = set()
    with open(makefile_path) as f:
        for line in f:
            for m in _TESTBED_RE.finditer(line):
                seen.add((m.group("namespace"), m.group("repo"), m.group("version")))
    return sorted(seen)


def sif_name(repo, version):
    return f"{repo}_{version}.sif"


def pull_one(namespace, repo, version, out_dir, force=False):
    dest = os.path.join(out_dir, sif_name(repo, version))
    image = f"{namespace}/swe-bench-{repo}-testbed:{version}"
    if os.path.exists(dest) and not force:
        print(f"  skip (exists): {dest}")
        return True
    print(f"  pull: docker://{image} -> {dest}")
    try:
        subprocess.run(
            ["apptainer", "pull", "--force", dest, f"docker://{image}"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {image} ({e})", file=sys.stderr)
        # Leave any partial file for inspection rather than deleting it
        # blindly; a rerun with --force will overwrite.
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--makefile", default="Makefile.testgeneval", help="Path to the Makefile to read image references from.")
    parser.add_argument("--out-dir", default=os.environ.get("APPTAINER_IMAGES_DIR", os.path.expanduser("~/apptainer_images")), help="Directory to write .sif files into (default: $APPTAINER_IMAGES_DIR).")
    parser.add_argument("--force", action="store_true", help="Re-pull and overwrite .sif files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be pulled and exit.")
    args = parser.parse_args()

    if not os.path.exists(args.makefile):
        sys.exit(f"Makefile not found: {args.makefile}")

    for var in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        val = os.environ.get(var)
        if not val:
            print(f"WARNING: {var} is not set. If it defaults to /tmp or "
                  f"$HOME, large pulls will fail with 'No space left on "
                  f"device'. Set it to a scratch path first.", file=sys.stderr)
        elif not os.path.isdir(val):
            print(f"WARNING: {var}={val} is not an existing directory.", file=sys.stderr)

    images = parse_makefile(args.makefile)
    print(f"{len(images)} distinct testbed images in {args.makefile}")

    if args.dry_run:
        for ns, repo, version in images:
            print(f"  docker://{ns}/swe-bench-{repo}-testbed:{version} -> "
                  f"{os.path.join(args.out_dir, sif_name(repo, version))}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    ok, failed = 0, []
    for i, (ns, repo, version) in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {repo} {version}")
        if pull_one(ns, repo, version, args.out_dir, force=args.force):
            ok += 1
        else:
            failed.append(f"{ns}/swe-bench-{repo}-testbed:{version}")

    print(f"\ndone: {ok} ok, {len(failed)} failed")
    if failed:
        print("failed images (rerun the script to retry, existing .sif files are skipped):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
