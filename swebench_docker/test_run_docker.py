"""run_docker_evaluation must raise when Docker itself is unavailable.

A missing CLI or stopped daemon fails every instance the same way. Only
logging it left run_evaluation.py exiting 0 with reports whose generated
count was complete and whose outcome categories were all empty.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from swebench_docker.run_docker import DockerUnavailableError, run_docker_evaluation

TASK_INSTANCE = {
    "repo": "django/django",
    "version": "4.1",
    "id": "django__django-15380-1",
    "instance_id": "django__django-15380",
}

# Verbatim docker CLI output with no daemon running, old and new wording.
DAEMON_DOWN_OLD = (
    "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
    "Is the docker daemon running?\n"
)
DAEMON_DOWN_NEW = (
    "failed to connect to the docker API at unix:///Users/u/.docker/run/docker.sock; "
    "check if the path is correct and if the daemon is running: dial unix "
    "/Users/u/.docker/run/docker.sock: connect: no such file or directory\n"
)


def fake_process(returncode, output):
    process = AsyncMock()
    process.returncode = returncode
    process.communicate.return_value = (output.encode(), None)
    return process


def run_with(subprocess_exec):
    env = {k: v for k, v in os.environ.items() if k != "SWEBENCH_DOCKER_FORK_DIR"}
    with patch.dict(os.environ, env, clear=True), patch(
        "swebench_docker.run_docker.asyncio.create_subprocess_exec", subprocess_exec
    ):
        asyncio.run(run_docker_evaluation(TASK_INSTANCE, "ns", "/tmp/logs", "full", 0))


class TestRunDockerSurfacesUnavailableDocker(unittest.TestCase):
    def test_unavailable_docker_raises(self):
        cases = {
            "cli not installed": AsyncMock(side_effect=FileNotFoundError("docker")),
            "daemon down, old wording": AsyncMock(
                return_value=fake_process(125, DAEMON_DOWN_OLD)
            ),
            "daemon down, new wording": AsyncMock(
                return_value=fake_process(1, DAEMON_DOWN_NEW)
            ),
        }
        for name, subprocess_exec in cases.items():
            with self.subTest(name):
                with self.assertRaises(DockerUnavailableError):
                    run_with(subprocess_exec)

    def test_single_container_failure_does_not_abort_the_run(self):
        # One bad image must still cost one instance, not the whole run.
        missing_image = (
            "Unable to find image 'ns/swe-bench-django_django-testbed:4.1' locally\n"
            "docker: Error response from daemon: pull access denied\n"
        )
        with self.assertLogs("swebench_docker.run_docker", level="WARNING"):
            run_with(AsyncMock(return_value=fake_process(125, missing_image)))


if __name__ == "__main__":
    unittest.main()
