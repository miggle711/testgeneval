import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from swebench_docker import run_apptainer
from swebench_docker.run_apptainer import (
    REPOS_NEEDING_DISK_OVERLAY,
    ApptainerUnavailableError,
    _writable_args,
    run_apptainer_evaluation,
)

TASK_INSTANCE = {
    "repo": "django/django",
    "version": "4.1",
    "id": "django__django-15380-1",
    "instance_id": "django__django-15380",
}


class TestRunApptainerSurfacesUnavailableApptainer(unittest.TestCase):
    # Only logging a missing binary would leave a report with a complete
    # generated count and empty outcome categories, as a stopped Docker
    # daemon did.
    def run_with(self, subprocess_exec):
        images_dir = tempfile.TemporaryDirectory()
        self.addCleanup(images_dir.cleanup)
        open(os.path.join(images_dir.name, "django_django_4.1.sif"), "w").close()
        with patch.dict(os.environ, {"SWEBENCH_DOCKER_FORK_DIR": "/fork",
                                     "APPTAINER_IMAGES_DIR": images_dir.name}), \
             patch(
                 "swebench_docker.run_apptainer.asyncio.create_subprocess_exec",
                 subprocess_exec,
             ):
            asyncio.run(
                run_apptainer_evaluation(TASK_INSTANCE, "ns", "/tmp/logs", "full", 0)
            )

    def test_missing_apptainer_binary_raises(self):
        with self.assertRaises(ApptainerUnavailableError):
            self.run_with(AsyncMock(side_effect=FileNotFoundError("apptainer")))

    def test_single_container_failure_does_not_abort_the_run(self):
        process = AsyncMock()
        process.returncode = 255
        process.communicate.return_value = (b"FATAL: container creation failed\n", None)
        with self.assertLogs("swebench_docker.run_apptainer", level="WARNING"):
            self.run_with(AsyncMock(return_value=process))


class TestImagesDirIsRequired(unittest.TestCase):
    # Without it every instance used to be skipped as "No .sif found" in
    # $HOME/apptainer_images and the run exited 0 having evaluated nothing.
    def run_without_images_dir(self):
        env = {k: v for k, v in os.environ.items() if k != "APPTAINER_IMAGES_DIR"}
        env["SWEBENCH_DOCKER_FORK_DIR"] = "/fork"
        with patch.dict(os.environ, env, clear=True):
            asyncio.run(run_apptainer_evaluation(TASK_INSTANCE, "ns", "/tmp/logs", "full", 0))

    def test_unset_images_dir_raises_instead_of_skipping(self):
        with self.assertRaises(ValueError) as caught:
            self.run_without_images_dir()
        self.assertIn("APPTAINER_IMAGES_DIR", str(caught.exception))

    def test_the_directory_set_at_call_time_is_used(self):
        images_dir = tempfile.TemporaryDirectory()
        self.addCleanup(images_dir.cleanup)
        with patch.dict(os.environ, {"APPTAINER_IMAGES_DIR": images_dir.name}):
            with self.assertLogs("swebench_docker.run_apptainer", level="WARNING") as logs:
                asyncio.run(
                    run_apptainer_evaluation(TASK_INSTANCE, "ns", "/tmp/logs", "full", 0)
                )
        self.assertIn(images_dir.name, "\n".join(logs.output))

class TestWritableArgs(unittest.TestCase):
    def test_default_is_the_in_memory_layer(self):
        self.assertEqual(_writable_args(None), ["--writable-tmpfs"])

    def test_overlay_directory_replaces_it(self):
        self.assertEqual(_writable_args("/tmp/x"), ["--overlay", "/tmp/x"])

    def test_only_sphinx_needs_the_overlay(self):
        self.assertEqual(REPOS_NEEDING_DISK_OVERLAY, {"sphinx-doc/sphinx"})


if __name__ == "__main__":
    unittest.main()
