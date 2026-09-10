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

Full testgeneval Makefile defines 126 distinct testbed images across
12 repos (~110GB of .sif, ~19h total at ~9 min each); testgenevallite
defines a subset.

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
#
# The Makefile's `docker build -t <ns>/...` lines name the *build*
# targets, usually `aorwall/...`. But the aorwall (upstream) testbed
# images do NOT carry cosmic-ray or this repo's swebench_docker, so a
# mutation run against one fails with ModuleNotFoundError: cosmic_ray
# (confirmed real 2026-09-11). The fork's own pre-built images, at
# `kdjain/...`, do. So --namespace defaults to kdjain and the tag read
# from the Makefile is rewritten to it, the same rewrite pull_images.py
# does.
#
# The repo part is [a-z0-9_-]+, NOT [a-z0-9_]+: 4 of the 12 repos have
# a hyphen (pylint-dev_pylint, pytest-dev_pytest, scikit-learn_scikit-learn,
# sphinx-doc_sphinx) and dropping the hyphen from the class silently
# skips all ~49 of their testbed images with no error. Regex backtracking
# still resolves the "-testbed:" anchor correctly for every name shape:
#   swe-bench-pylint-dev_pylint-testbed:2.10        -> pylint-dev_pylint, 2.10
#   swe-bench-astropy_astropy-testbed:5.1           -> astropy_astropy, 5.1
#   swe-bench-scikit-learn_scikit-learn-testbed:1.3 -> scikit-learn_scikit-learn, 1.3
_TESTBED_RE = re.compile(
    r"(?:aorwall|kdjain)/swe-bench-(?P<repo>[a-z0-9_-]+)-testbed:(?P<version>[0-9.]+)"
)


def parse_makefile(makefile_path):
    """Return a sorted, de-duplicated list of (repo, version) for every
    testbed image the Makefile references. The namespace in the Makefile
    is ignored; the caller supplies the one to actually pull from."""
    seen = set()
    with open(makefile_path) as f:
        for line in f:
            for m in _TESTBED_RE.finditer(line):
                seen.add((m.group("repo"), m.group("version")))
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
    parser.add_argument("--namespace", default="kdjain", help="Docker Hub namespace to pull testbed images from (default: kdjain, the fork's account whose images carry cosmic-ray; aorwall's do not).")
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
    repos = sorted({repo for repo, _ in images})
    print(f"{len(images)} distinct testbed images across {len(repos)} repos "
          f"in {args.makefile} (pulling from namespace '{args.namespace}')")
    print(f"  repos: {', '.join(repos)}")

    # The full testgeneval Makefile covers 12 repos; testgenevallite a
    # subset. Fewer than 8 almost certainly means the repo-name regex
    # silently dropped the hyphenated ones (pylint-dev, pytest-dev,
    # scikit-learn, sphinx-doc) -- catch it here, not after a multi-hour
    # pull that quietly omits a third of the dataset.
    if "testgenevallite" not in os.path.basename(args.makefile) and len(repos) < 8:
        print(f"WARNING: only {len(repos)} repos parsed from a full "
              f"Makefile that should have 12. The repo-name regex may be "
              f"dropping hyphenated repo names. Not continuing.",
              file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for repo, version in images:
            print(f"  docker://{args.namespace}/swe-bench-{repo}-testbed:{version} -> "
                  f"{os.path.join(args.out_dir, sif_name(repo, version))}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    ok, failed = 0, []
    for i, (repo, version) in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {repo} {version}")
        if pull_one(args.namespace, repo, version, args.out_dir, force=args.force):
            ok += 1
        else:
            failed.append(f"{args.namespace}/swe-bench-{repo}-testbed:{version}")

    print(f"\ndone: {ok} ok, {len(failed)} failed")
    if failed:
        print("failed images (rerun the script to retry, existing .sif files are skipped):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
