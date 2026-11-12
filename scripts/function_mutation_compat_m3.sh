#!/usr/bin/env bash
# Recheck the failing Django instance in its real image, using a fresh clone.
# Set FUNCMUT_COMPAT_SCRATCH to a new scratch directory; use it again for check.
set -euo pipefail
ROOT="${FUNCMUT_COMPAT_SCRATCH:?Set FUNCMUT_COMPAT_SCRATCH to a fresh scratch directory}"
OLD="${FUNCMUT_ORIGINAL_SCRATCH:-/fs04/scratch2/al49/$USER}"
REPO="$ROOT/testgeneval"
IMAGES="${APPTAINER_IMAGES_DIR:-/fs04/scratch2/al49/mvar0010/apptainer_images}"
case "${1:-}" in
submit)
    [ ! -e "$ROOT" ] || { echo "Refusing to reuse $ROOT; choose a fresh directory" >&2; exit 1; }
    mkdir -p "$ROOT/logs" "$ROOT/hf-cache"
    git clone --single-branch --branch feat/target-function-mutation https://github.com/miggle711/testgeneval.git "$REPO"
    git -C "$REPO" merge-base --is-ancestor b4b7ccc HEAD
    git -C "$REPO" rev-parse HEAD > "$ROOT/revision.txt"
    python3 - "$OLD/funcmut/kg_k5.jsonl" "$ROOT/predictions.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
rows = [r for r in rows if r['id'] == 'django__django-11514-15783']
assert len(rows) == 1 and len(rows[0]['preds']['full']) == 5, 'Missing or unexpected repro predictions'
with open(sys.argv[2], 'w') as out:
    out.write(json.dumps(rows[0]) + '\n')
PY
    export COMPAT_REPO="$REPO" COMPAT_IMAGES="$IMAGES"
    export PREDICTIONS_PATH="$ROOT/predictions.jsonl" LOG_DIR="$ROOT/logs"
    export HF_HOME="$ROOT/hf-cache" HF_DATASETS_CACHE="$ROOT/hf-cache/datasets"
    export APPTAINER_IMAGES_DIR="$IMAGES" MUTATION_SCOPE=function SKIP_MUTATION=0 NUM_PROCESSES=1
    export TIMEOUT=3600 MUTATION_TIMEOUT=3600 CONDA_ENV=testgeneval SBATCH_EXPORT=ALL
    export SWEBENCH_DOCKER_FORK_DIR="$REPO" SWE_BENCH_TASKS=kjain14/testgeneval
    unset APPTAINER_HOST_TIMEOUT
    cat > "$ROOT/run.sh" <<'BATCH'
#!/usr/bin/env bash
set -euo pipefail
cd "$COMPAT_REPO"
# The version is read from the same Django 3.0 image as the failing sample.
apptainer exec --cleanenv --no-home "$COMPAT_IMAGES/django__django_3.0.sif" \
    /opt/pyenv/versions/3.6.15/bin/python -c \
    'import pkg_resources; from cosmic_ray.work_db import WorkDB; print("cosmic-ray:",pkg_resources.get_distribution("cosmic-ray").version,"bulk API:",hasattr(WorkDB,"set_multiple_results"),"single API:",hasattr(WorkDB,"set_result"))'
bash m3_run_evaluation.slurm
BATCH
    cd "$REPO"
    sbatch --parsable --partition=comp --cpus-per-task=8 --mem=32G --time=04:00:00 \
        --job-name=fm-compat --output="$ROOT/slurm-%j.out" "$ROOT/run.sh" > "$ROOT/job.txt"
    echo "Submitted $(cat "$ROOT/job.txt"); check with the same FUNCMUT_COMPAT_SCRATCH"
    ;;
check)
    printf 'Code: '; cat "$ROOT/revision.txt"
    sacct -j "$(cat "$ROOT/job.txt")" --format=JobID,JobName,State,Elapsed -X
    python3 - "$ROOT" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
for p in sorted(root.glob('slurm-*.out')):
    for line in p.read_text(errors='replace').splitlines():
        if 'cosmic-ray:' in line:
            print(line)
paths = sorted((root/'logs').rglob('*.full.eval.log'))
if not paths:
    print('WAIT: no instance log yet')
    raise SystemExit(0)
if len(paths) != 1:
    print('FAIL: expected one instance log, found', len(paths))
    raise SystemExit(1)
text = paths[0].read_text(errors='replace')
blocks = [b for b in text.split('>>>>> Tests config')[1:] if b.split('\n',1)[0].strip() == 'full pred']
ended = sum('>>>>> Unfiltered Tests Passed' in b or '>>>>> Unfiltered Tests Failed' in b for b in blocks)
mut = [b for b in blocks if 'Running mutation testing' in b]
failed = [b for b in blocks if 'FunctionMutationFAIL' in b]
def number(key, block):
    m = re.search(r'\b'+key+r': (\d+)', block)
    return int(m.group(1)) if m else None
complete = [b for b in mut if number('FunctionMutationNum',b) is not None
            and number('FunctionMutationNum',b) == number('FunctionMutationTotal',b)
            and number('FunctionMutationTotal',b) > 0 and 'FunctionMutationLOG:' in b]
print('full blocks:',len(blocks),'finished:',ended,'mutation:',len(mut),'complete:',len(complete),'failures:',len(failed))
if failed:
    for b in failed:
        for line in b.splitlines():
            if 'FunctionMutationFAIL' in line:
                print(line)
    print('FAIL: function mutation still fails')
    raise SystemExit(1)
if ended < 5:
    print('WAIT: evaluation incomplete; inspect Slurm state for a failed job')
    raise SystemExit(0)
ok = len(blocks) == 5 and len(mut) > 0 and len(complete) == len(mut) and 'MutationTimeout' not in text
print('PASS: all five samples finished and every mutation run completed' if ok else 'FAIL: incomplete target mutation or unexpected sample count')
raise SystemExit(0 if ok else 1)
PY
    ;;
*) echo "usage: $0 submit|check" >&2; exit 2 ;;
esac
