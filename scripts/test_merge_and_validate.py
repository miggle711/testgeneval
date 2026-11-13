import importlib.util
import json
import os
import tempfile
import unittest

# scripts/ isn't a package, so import by path.
_SPEC = importlib.util.spec_from_file_location(
    "merge_and_validate",
    os.path.join(os.path.dirname(__file__), "merge_and_validate.py"),
)
merge_and_validate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_and_validate)


def write_shard(directory, name, rows):
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class TestMergeAndValidateDuplicates(unittest.TestCase):
    def merged_counts(self, shard_rows):
        with tempfile.TemporaryDirectory() as d:
            shards = [
                write_shard(d, f"shard-{i}.jsonl", rows)
                for i, rows in enumerate(shard_rows)
            ]
            merged = os.path.join(d, "merged.jsonl")
            merge_and_validate.merge(shards, merged)
            return merge_and_validate.validate(merged)

    def test_shared_instance_id_is_not_a_duplicate(self):
        # Real pair from the dataset: two distinct tasks, one instance_id.
        total, unique = self.merged_counts([
            [{"id": "django__django-12091-15824", "instance_id": "django__django-12091"}],
            [{"id": "django__django-12091-15825", "instance_id": "django__django-12091"}],
        ])
        self.assertEqual((total, unique), (2, 2))

    def test_same_id_in_two_shards_is_a_duplicate(self):
        row = {"id": "django__django-12091-15824", "instance_id": "django__django-12091"}
        total, unique = self.merged_counts([[row], [row]])
        self.assertEqual((total, unique), (2, 1))


if __name__ == "__main__":
    unittest.main()
