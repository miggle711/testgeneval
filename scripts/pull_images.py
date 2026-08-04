# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import re
import subprocess


# Function to parse the Makefile and extract image information
def parse_makefile(makefile_path):
    images = []
    docker_build_regex = re.compile(
        r"docker build .* -t (?P<image>[^\s]+) -f (?P<dockerfile>[^\s]+)"
    )

    with open(makefile_path, "r") as f:
        for line in f:
            match = docker_build_regex.search(line)
            if match:
                images.append(
                    (
                        match.group("image").replace("aorwall/", "kdjain/"),
                        match.group("dockerfile"),
                    )
                )
    return images


def pull_images(images):
    # Was previously `images[:2]` -- pulled only the first 2 of ~100 images
    # (just the shared base layers, no real per-repo testbed image), with no
    # comment or upstream discussion explaining the limit. Removed so this
    # actually pulls the full image set the Makefile defines.
    for image, _ in images:
        try:
            image_updated = image.replace("aorwall/", "kdjain/")
            subprocess.run(["docker", "pull", image_updated], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error pulling image: {image}")
            print(e)
            continue


# Main script
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build, push, and optionally clean Docker images based on a Makefile."
    )
    parser.add_argument(
        "--makefile", default="Makefile.testgeneval", help="Path to the Makefile."
    )
    args = parser.parse_args()

    makefile_path = args.makefile
    if not os.path.exists(makefile_path):
        print(f"Makefile not found at {makefile_path}")
        exit(1)

    images = parse_makefile(makefile_path)
    if not images:
        print("No images found in the Makefile.")
        exit(1)

    pull_images(images)
