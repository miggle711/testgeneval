# Copyright (c) Meta Platforms, Inc. and affiliates.

import signal
import unittest

from swebench_docker.evaluate_instance import extract_preamble_classes_and_functions

# Small readable fixture for correctness assertions (not the hang trigger
# below -- decorator count alone doesn't reproduce catastrophic backtracking
# with re.MULTILINE's ^ anchors; see CATASTROPHIC_BACKTRACK_TRIGGER).
STACKED_DECORATORS = """import unittest
from unittest.mock import patch


@patch('a.b.c')
@patch('a.b.d')
@patch('a.b.e')
class FooTests(unittest.TestCase):
    @patch('x.y.z')
    @patch('x.y.w')
    def test_something(self):
        pass

    def test_other(self):
        pass
"""

# Verified real trigger for the old (\s*@[\w\.\(\)\', ]+\s*)* shape
# (testgeneval#61): the ambiguity isn't decorator count, it's internal
# whitespace inside decorator args, matched by both the outer \s* and the
# inner [\w\.\(\)\', ]+ char class (space is in both), which the old
# pattern can partition exponentially many ways. On the old pattern this
# hangs past several seconds at 18 decorators when no class ever matches
# (forcing a full backtrack search); confirmed instant (<1ms) on the fixed
# pattern up to 40 decorators.
CATASTROPHIC_BACKTRACK_TRIGGER = (
    "".join(f"@patch({' ' * 8}a.b.{i}{' ' * 8})\n" for i in range(18)) + "notaclass\n"
)

CRLF_DECORATOR = "@patch('x')\r\nclass Foo(unittest.TestCase):\r\n    def test_it(self):\r\n        pass\r\n"


def _run_with_timeout(fn, timeout_s):
    def _handler(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class TestExtractPreambleTerminates(unittest.TestCase):
    def test_stacked_decorators_terminate_quickly(self):
        # Old regex shape caught catastrophic backtracking on exactly this
        # kind of input; this must not hang.
        try:
            preamble, classes, test_functions = _run_with_timeout(
                lambda: extract_preamble_classes_and_functions(STACKED_DECORATORS, None),
                2,
            )
        except TimeoutError:
            self.fail("extract_preamble_classes_and_functions hung (catastrophic backtracking regression)")

        self.assertEqual(len(classes), 1)
        self.assertIn("FooTests", classes[0][0])
        method_names = [m[0] for m in classes[0][1]]
        self.assertTrue(any("test_something" in m for m in method_names))
        self.assertTrue(any("test_other" in m for m in method_names))

    def test_catastrophic_backtrack_trigger_terminates_quickly(self):
        # This specific shape is a verified real trigger for the old
        # pattern (see comment on CATASTROPHIC_BACKTRACK_TRIGGER above).
        try:
            _run_with_timeout(
                lambda: extract_preamble_classes_and_functions(
                    CATASTROPHIC_BACKTRACK_TRIGGER, None
                ),
                2,
            )
        except TimeoutError:
            self.fail("extract_preamble_classes_and_functions hung (catastrophic backtracking regression)")


class TestExtractPreambleCRLF(unittest.TestCase):
    def test_crlf_decorator_line_is_not_silently_dropped(self):
        preamble, classes, test_functions = _run_with_timeout(
            lambda: extract_preamble_classes_and_functions(CRLF_DECORATOR, None), 2
        )
        self.assertEqual(len(classes), 1)
        # The decorator must still be part of the matched class header,
        # not silently dropped into the unparsed preamble.
        self.assertIn("@patch", classes[0][0])


if __name__ == "__main__":
    unittest.main()
