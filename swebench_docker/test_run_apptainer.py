import unittest

from swebench_docker.run_apptainer import REPOS_NEEDING_DISK_OVERLAY, _writable_args


class TestWritableArgs(unittest.TestCase):
    def test_default_is_the_in_memory_layer(self):
        self.assertEqual(_writable_args(None), ["--writable-tmpfs"])

    def test_overlay_directory_replaces_it(self):
        self.assertEqual(_writable_args("/tmp/x"), ["--overlay", "/tmp/x"])

    def test_only_sphinx_needs_the_overlay(self):
        self.assertEqual(REPOS_NEEDING_DISK_OVERLAY, {"sphinx-doc/sphinx"})


if __name__ == "__main__":
    unittest.main()
