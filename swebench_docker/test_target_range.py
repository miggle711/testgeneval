# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from swebench_docker.target_range import resolve_target_line_range

MOD_SOURCE = """def foo():
    x = 1
    x += 1
    return x


def bar():
    y = 2
    return y


class C:
    def method(self):
        z = 5
        return z
"""

MULTI_HUNK_PATCH = """diff --git a/mod.py b/mod.py
index a0abe96..f95279a 100644
--- a/mod.py
+++ b/mod.py
@@ -1,5 +1,6 @@
 def foo():
     x = 1
+    x += 1
     return x


@@ -10,4 +11,5 @@ def bar():

 class C:
     def method(self):
-        return 1
+        z = 5
+        return z
"""

SINGLE_FN_SOURCE = "def foo():\n    x = 1\n    x += 1\n    return x\n"
SINGLE_FN_PATCH = """diff --git a/mod.py b/mod.py
index a0abe96..f95279a 100644
--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,4 @@
 def foo():
     x = 1
+    x += 1
     return x
"""

MODULE_LEVEL_SOURCE = "MODULE_CONST = 1\ndef foo():\n    x = 1\n"
MODULE_LEVEL_PATCH = """diff --git a/mod.py b/mod.py
index a0abe96..f95279a 100644
--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,3 @@
+MODULE_CONST = 1
 def foo():
     x = 1
"""

DELETION_ONLY_SOURCE = "def foo():\n    x = 1\n    return x\n"
DELETION_ONLY_PATCH = """diff --git a/mod.py b/mod.py
index a0abe96..f95279a 100644
--- a/mod.py
+++ b/mod.py
@@ -1,4 +1,3 @@
 def foo():
     x = 1
-    x += 1
     return x
"""

MULTI_FILE_SOURCE = "def foo():\n    x = 999\n    return x\n"
MULTI_FILE_PATCH = """diff --git a/models.py b/models.py
index a366f20..042d686 100644
--- a/models.py
+++ b/models.py
@@ -1,3 +1,3 @@
 def foo():
-    x = 1
+    x = 999
     return x
diff --git a/test_models.py b/test_models.py
index b6e3c44..ccc7fa6 100644
--- a/test_models.py
+++ b/test_models.py
@@ -1,4 +1,4 @@
 def bar():
     y = 1
-    z = 2
+    z = 888
     return y
"""

SPACE_IN_PATH_SOURCE = "def foo():\n    return 999\n"
SPACE_IN_PATH_PATCH = (
    "diff --git a/weird dir/foo.py b/weird dir/foo.py\n"
    "index c2119dc..9a313fc 100644\n"
    "--- a/weird dir/foo.py\t\n"
    "+++ b/weird dir/foo.py\t\n"
    "@@ -1,2 +1,2 @@\n"
    " def foo():\n"
    "-    return 1\n"
    "+    return 999\n"
)


MULTI_HUNK_WITH_GAP_SOURCE = """def foo():
    x = 1
    x += 1
    return x


def untouched():
    a = 1
    b = 2
    c = 3
    return a + b + c


class C:
    def method(self):
        z = 5
        return z
"""

MULTI_HUNK_WITH_GAP_PATCH = """diff --git a/mod.py b/mod.py
index a0abe96..f95279a 100644
--- a/mod.py
+++ b/mod.py
@@ -1,5 +1,6 @@
 def foo():
     x = 1
+    x += 1
     return x


@@ -13,4 +14,5 @@ def bar():

 class C:
     def method(self):
-        return 1
+        z = 5
+        return z
"""


class TestResolveTargetLineRange(unittest.TestCase):
    def test_multiple_hunks_return_separate_function_ranges(self):
        result = resolve_target_line_range(MOD_SOURCE, MULTI_HUNK_PATCH, "mod.py")
        self.assertEqual(result, [(1, 4), (13, 15)])

    def test_untouched_function_between_two_touched_functions_is_excluded(self):
        # Regression test for the merge bug: foo() and C.method() are
        # touched, untouched() sits between them and must not be folded
        # into the reported ranges just because its lines fall inside
        # the old min/max span.
        result = resolve_target_line_range(
            MULTI_HUNK_WITH_GAP_SOURCE, MULTI_HUNK_WITH_GAP_PATCH, "mod.py"
        )
        self.assertEqual(result, [(1, 4), (15, 17)])
        untouched_range = (7, 11)
        for start, end in result:
            self.assertFalse(start <= untouched_range[0] <= end)
            self.assertFalse(start <= untouched_range[1] <= end)

    def test_single_function_patch(self):
        result = resolve_target_line_range(SINGLE_FN_SOURCE, SINGLE_FN_PATCH, "mod.py")
        self.assertEqual(result, [(1, 4)])

    def test_module_level_only_change_returns_none(self):
        result = resolve_target_line_range(
            MODULE_LEVEL_SOURCE, MODULE_LEVEL_PATCH, "mod.py"
        )
        self.assertIsNone(result)

    def test_wrong_file_in_patch_returns_none(self):
        result = resolve_target_line_range(SINGLE_FN_SOURCE, SINGLE_FN_PATCH, "other.py")
        self.assertIsNone(result)

    def test_pure_deletion_anchors_to_enclosing_function(self):
        result = resolve_target_line_range(
            DELETION_ONLY_SOURCE, DELETION_ONLY_PATCH, "mod.py"
        )
        self.assertEqual(result, [(1, 3)])

    def test_empty_patch_returns_none(self):
        result = resolve_target_line_range(SINGLE_FN_SOURCE, "", "mod.py")
        self.assertIsNone(result)

    def test_invalid_source_returns_none(self):
        result = resolve_target_line_range("def foo(:\n", SINGLE_FN_PATCH, "mod.py")
        self.assertIsNone(result)

    def test_code_file_is_not_matched_as_substring_of_another_file(self):
        # models.py must not pick up test_models.py's hunk just because
        # "models.py" is a substring of "test_models.py".
        result = resolve_target_line_range(
            MULTI_FILE_SOURCE, MULTI_FILE_PATCH, "models.py"
        )
        self.assertEqual(result, [(1, 3)])

    def test_path_with_space_still_matches_despite_trailing_tab(self):
        # git appends a trailing tab to +++/--- lines for paths with
        # spaces; the exact-match must strip it, not just substring it.
        result = resolve_target_line_range(
            SPACE_IN_PATH_SOURCE, SPACE_IN_PATH_PATCH, "weird dir/foo.py"
        )
        self.assertEqual(result, [(1, 2)])


if __name__ == "__main__":
    unittest.main()
