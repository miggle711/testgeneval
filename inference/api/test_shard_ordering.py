# Regression test: shards must not overlap when different shards'
# output files have different numbers of already-completed ids
# (e.g. a resumed job). See run_api.py's main(), the fix is
# sharding before filtering existing_ids, not after.

import unittest

from datasets import Dataset


def shard_then_filter(dataset, existing_ids, num_shards, shard_id):
    d = dataset.shard(num_shards, shard_id, contiguous=True)
    if existing_ids:
        d = d.filter(
            lambda x: x["id"] not in existing_ids,
            load_from_cache_file=False,
        )
    return d


class TestShardOrdering(unittest.TestCase):
    def test_shard_then_filter_produces_disjoint_shards(self):
        dataset = Dataset.from_dict({"id": [f"item-{i}" for i in range(20)]})

        # shard 0 resumed with 3 already-completed ids, shard 1 fresh
        shard0 = shard_then_filter(dataset, {"item-0", "item-1", "item-2"}, 2, 0)
        shard1 = shard_then_filter(dataset, set(), 2, 1)

        self.assertEqual(set(shard0["id"]) & set(shard1["id"]), set())


if __name__ == "__main__":
    unittest.main()
