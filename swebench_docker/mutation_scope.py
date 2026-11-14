"""Restrict cosmic-ray to the target function's mutants (testgeneval#10).

Both arms are asked to test one function, so the primary mutation score
counts only mutants inside it. With MUTATION_SCOPE=function, every work item
outside the target function is marked skipped right after `cosmic-ray init`,
so `cosmic-ray exec` runs only the target function's mutants. Before this,
the function score was filtered out of a whole-file run: when that run hit
the mutation timeout, the target function's unfinished mutants silently
dropped out of the score.

cosmic_ray is only installed in the per-repo testbed env, so these are
script texts run there with `python -c`, not code imported here.
"""

from typing import List, Tuple

FILE = "file"
FUNCTION = "function"
SCOPES = (FILE, FUNCTION)


def _in_range_ids(code_file: str, ranges: List[Tuple[int, int]]) -> str:
    """Script lines setting `in_range` to the job ids with a mutation in a
    target range of code_file (a mutation's start line, as cosmic-ray
    records it)."""
    clause = " or ".join(f"({start} <= spec.start_pos[0] <= {end})" for start, end in ranges)
    return (
        "from cosmic_ray.work_db import WorkDB\n"
        "db = WorkDB('mutation.sqlite', WorkDB.Mode.open)\n"
        "in_range = {\n"
        "    item.job_id\n"
        "    for item in db.work_items\n"
        "    for spec in item.mutations\n"
        f"    if str(spec.module_path) == {code_file!r}\n"
        f"    and ({clause})\n"
        "}\n"
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
    """Kill rate over the target function's mutants that cosmic-ray tested.

    Only mutants whose worker outcome is "normal" count: the test command ran
    against the mutant and passed or failed. cosmic-ray's own `is_killed`
    treats every outcome but SURVIVED as a kill, so a mutant it could not
    test (worker outcome "no-test", "exception" or "abnormal", for example
    when the mutated module fails to import in the worker) would otherwise
    count as killed and raise the score.

    Prints `FUNC_MUTATION_UNTESTED <n>`, the target mutants with a result
    other than "normal" or "skipped", then either
    `FUNC_MUTATION_RESULT <score> <done> <total>`, where done is how many of
    the total target mutants were tested (fewer after a timeout or when some
    were untested), or `FUNC_MUTATION_NO_MUTANTS_IN_RANGE <total>` when none
    was tested (total 0: the target holds no mutant).
    """
    return _in_range_ids(code_file, ranges) + (
        "try:\n"
        "    results = dict(db.results)\n"
        "    finished = [j for j in in_range if j in results\n"
        "                and results[j].worker_outcome != 'skipped']\n"
        "    done = [j for j in finished if results[j].worker_outcome == 'normal']\n"
        "    print(f'FUNC_MUTATION_UNTESTED {len(finished) - len(done)}')\n"
        "    if done:\n"
        "        kills = sum(1 for j in done if results[j].is_killed)\n"
        "        print(f'FUNC_MUTATION_RESULT {kills / len(done) * 100} {len(done)} {len(in_range)}')\n"
        "    else:\n"
        "        print(f'FUNC_MUTATION_NO_MUTANTS_IN_RANGE {len(in_range)}')\n"
        "finally:\n"
        "    db.close()\n"
    )
