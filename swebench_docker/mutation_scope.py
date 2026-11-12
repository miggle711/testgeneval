"""Restrict cosmic-ray to the target function's mutants (testgeneval#10).

Both arms are asked to test one function, so the primary mutation score
counts only mutants inside it. With MUTATION_SCOPE=function, every work item
outside the target function is marked skipped right after `cosmic-ray init`,
so `cosmic-ray exec` runs only the target function's mutants. Before this,
the function score was filtered out of a whole-file run: when that run hit
the mutation timeout, the target function's unfinished mutants silently
dropped out of the score.

Whether the target function holds any mutant at all decides whether an
instance is eligible (fyp-paper#7, rule 3). That is a fact about the patched
source, so count_script reads it from a work db that no generated test has
touched, once per instance.

cosmic_ray is only installed in the per-repo testbed env, so these are
script texts run there with `python -c`, not code imported here.
"""

from typing import List, Tuple

FILE = "file"
FUNCTION = "function"
SCOPES = (FILE, FUNCTION)

# The per-sample mutation run's work db, and the one the per-instance count
# uses. They are separate files because `cosmic-ray init` empties the db it
# is given.
RUN_DB = "mutation.sqlite"
COUNT_DB = "mutation_count.sqlite"


def _in_range_ids(code_file: str, ranges: List[Tuple[int, int]], db: str = RUN_DB) -> str:
    """Script lines setting `in_range` to the job ids in work db `db` with a
    mutation in a target range of code_file (a mutation's start line, as
    cosmic-ray records it)."""
    clause = " or ".join(f"({start} <= spec.start_pos[0] <= {end})" for start, end in ranges)
    return (
        "from cosmic_ray.work_db import WorkDB\n"
        f"db = WorkDB({db!r}, WorkDB.Mode.open)\n"
        "in_range = {\n"
        "    item.job_id\n"
        "    for item in db.work_items\n"
        "    for spec in item.mutations\n"
        f"    if str(spec.module_path) == {code_file!r}\n"
        f"    and ({clause})\n"
        "}\n"
    )


def count_script(code_file: str, ranges: List[Tuple[int, int]]) -> str:
    """Count the mutants inside the target ranges in COUNT_DB, changing
    nothing.

    Prints `TARGET_MUTANT_COUNT <n>`. Run on a db that `cosmic-ray init` has
    just filled, n is what every later sample's FunctionMutationTotal must
    equal: both select from the same work items with the same ranges.
    """
    return _in_range_ids(code_file, ranges, COUNT_DB) + (
        "try:\n"
        "    print(f'TARGET_MUTANT_COUNT {len(in_range)}')\n"
        "finally:\n"
        "    db.close()\n"
    )


def skip_outside_script(code_file: str, ranges: List[Tuple[int, int]]) -> str:
    """Mark every work item outside the target ranges skipped.

    Prints `FUNC_MUTANTS_IN_RANGE <n>`, the number left to run.
    """
    return _in_range_ids(code_file, ranges) + (
        "from cosmic_ray.work_item import WorkResult, WorkerOutcome\n"
        "try:\n"
        "    outside = [item.job_id for item in db.pending_work_items if item.job_id not in in_range]\n"
        "    result = WorkResult(worker_outcome=WorkerOutcome.SKIPPED,\n"
        "                        output='outside the target function')\n"
        "    if hasattr(db, 'set_multiple_results'):\n"
        "        db.set_multiple_results(outside, result)\n"
        "    else:\n"
        "        for job_id in outside:\n"
        "            db.set_result(job_id, result)\n"
        "    print(f'FUNC_MUTANTS_IN_RANGE {len(in_range)}')\n"
        "finally:\n"
        "    db.close()\n"
    )


def score_script(code_file: str, ranges: List[Tuple[int, int]]) -> str:
    """Kill rate over the target function's mutants that have a result.

    Prints `FUNC_MUTATION_RESULT <score> <done> <total>`, where done is how
    many of the total target mutants finished (fewer after a timeout), or
    `FUNC_MUTATION_NO_MUTANTS_IN_RANGE <total>` when none finished (total 0:
    the target holds no mutant).
    """
    return _in_range_ids(code_file, ranges) + (
        "try:\n"
        "    results = dict(db.results)\n"
        "    done = [j for j in in_range if j in results\n"
        "            and results[j].worker_outcome != 'skipped']\n"
        "    if done:\n"
        "        kills = sum(1 for j in done if results[j].is_killed)\n"
        "        print(f'FUNC_MUTATION_RESULT {kills / len(done) * 100} {len(done)} {len(in_range)}')\n"
        "    else:\n"
        "        print(f'FUNC_MUTATION_NO_MUTANTS_IN_RANGE {len(in_range)}')\n"
        "finally:\n"
        "    db.close()\n"
    )
