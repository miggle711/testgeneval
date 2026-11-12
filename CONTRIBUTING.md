# Contributing

This is a fork of [facebookresearch/testgeneval](https://github.com/facebookresearch/testgeneval), extended for a research project on patch anchored knowledge graph retrieval for LLM based test generation. See `README.md` for what the fork adds, and `CLAUDE.md` for how to run things and the real gotchas the team has hit on M3.

## Setup

```
conda env create -f testgeneval.yaml
conda activate testgeneval
```

Copy `.env_template` to `.env` and set `SWEBENCH_DOCKER_FORK_DIR` to this repo's absolute path.

Tests use plain `unittest`, not pytest:

```
python -m unittest swebench_docker.test_run_apptainer -v
```

Without the conda env active, call its interpreter directly, e.g. `/opt/anaconda3/envs/testgeneval/bin/python3 -m unittest ...` (see `CLAUDE.md`).

## Branches and commits

- Branch from a fresh `main`.
- Name the branch with a prefix that says what it is: `fix/`, `feat/`, `docs/`, `chore/`, `refactor/` or `test/`.
- Start the commit title with the same prefix: `fix: ...`, `feat: ...`.
- Keep one concern per PR.

## Pull requests

Link the issue in the first line (`Fixes #N` or, if the PR doesn't complete it, `Part of #N`).

The description should show the change on real data, not just describe it:

1. **Problem:** the real failure, with the actual error output or log excerpt, not a paraphrase. Name the real cost if you know it (e.g. "lost 117 instances for Qwen3-Coder-30B instruct, job 59573905").
2. **Fix:** what changed and why this is the right layer to fix it at.
3. **After:** the same scenario with the new behaviour, ideally from a real run, not a hypothetical.
4. **Tests:** what's covered and the suite result.

`.github/workflows/tests.yml` runs the `swebench_docker` and `inference` unit test suites on every PR and push to `main`. It's a real CI gate the test suites must pass, not a stand-in for M3 validation, it mocks out any real network/GPU calls and can't exercise the Apptainer backend, inference against a real vLLM server, or evaluation at all. There's no branch protection requiring it to pass before merge, but a red run should be treated as a real blocker regardless.

CI passing is necessary but not sufficient for anything touching inference, evaluation, or the Apptainer backend. Before merging one of those:

- Run the change against something real on M3 (a smoke job, a real instance, a real model), not just a synthetic case.
- Post the job id and what you checked in the log, not just the final summary line. A reviewer should be able to pull the same job's log and see the same thing you're claiming.
- If the change could affect reported numbers (timeouts, pass@k scoring, mutation scope, prompt truncation, anything that changes what gets counted), say so explicitly and link to the relevant tracking issue in `fyp-paper` if one exists.

### Scripts and numbers behind a claim

Every number or "I checked that" in a PR description should be reproducible by someone else. Say what you ran (the real command, the real job id, the real file path on scratch), not just the result. A path that only exists on your own M3 account doesn't count as reproducible, say where it lives (shared scratch, which group) or include the script.

### Tests

Write a test because a real code path matters, not to raise a coverage number. One test per real behaviour; parametrize variants of the same behaviour instead of copy-pasting a near-identical test. Skip a test that only restates the code.

A real path is worth a test when it's: something this project's own code actually exercises in a real run (check, don't assume, a file can be unused legacy machinery inherited from upstream), something a real incident has already hit (`CLAUDE.md`'s gotchas and `results/RUN_LOG.md` are the record of these), or logic that's cheap to check without GPU/M3 access but currently only gets caught by an expensive real failure (a config contradiction that only surfaces as a GPU OOM after allocation, say). A regression test must fail without the fix, check this by stashing the source change and running the test.

### Stacked PRs

When a PR depends on another that's still open:

- Base it on the other PR's branch, so the diff shows only its own change.
- Say so in the description, and the merge order.
- After the base PR merges, retarget and rebase.

## Issues

Give a real repro: the command, the actual error or log output, and the instance/job id if there is one. "This doesn't work" without a real example makes it much harder for anyone else to pick up.
