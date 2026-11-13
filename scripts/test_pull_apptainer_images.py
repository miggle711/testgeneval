"""scikit-learn's testbed images come from the rebuilt namespace, not from
--namespace, whose scikit-learn images are broken (testgeneval#55, #123)."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
import pull_apptainer_images as pull  # noqa: E402

MAKEFILE = (
    "\tdocker build -t aorwall/swe-bench-astropy_astropy-testbed:5.1 .\n"
    "\tdocker build -t aorwall/swe-bench-scikit-learn_scikit-learn-testbed:1.3 .\n"
)


def run_main(argv):
    out = io.StringIO()
    with patch.object(sys, "argv", ["pull_apptainer_images.py", *argv]), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        pull.main()
    return out.getvalue()


class SklearnNamespaceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # "testgenevallite" in the name skips the 8-repo sanity check
        self.makefile = os.path.join(self.dir, "Makefile.testgenevallite")
        with open(self.makefile, "w") as f:
            f.write(MAKEFILE)

    def test_dry_run_lists_sklearn_from_the_rebuilt_namespace(self):
        out = run_main(["--makefile", self.makefile, "--out-dir", self.dir, "--dry-run"])
        self.assertIn("docker://miggy711/swe-bench-scikit-learn_scikit-learn-testbed:1.3", out)
        self.assertIn("docker://kdjain/swe-bench-astropy_astropy-testbed:5.1", out)
        self.assertNotIn("kdjain/swe-bench-scikit-learn", out)

    def test_pull_requests_sklearn_from_the_rebuilt_namespace(self):
        with patch.object(pull.subprocess, "run") as run:
            run_main(["--makefile", self.makefile, "--out-dir", self.dir, "--force"])
        pulled = [c.args[0][-1] for c in run.call_args_list]
        self.assertEqual(sorted(pulled), [
            "docker://kdjain/swe-bench-astropy_astropy-testbed:5.1",
            "docker://miggy711/swe-bench-scikit-learn_scikit-learn-testbed:1.3",
        ])


if __name__ == "__main__":
    unittest.main()
