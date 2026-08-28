# Experiment Plan: `instruct` vs `kg_only` on TestGenEval

Sole copy as of 2026-08-24. A duplicate previously lived in the
`repo-kg-construction`/`pycodekg` repo but had drifted badly out of sync
(missing RQ1/RQ4, the locked model shortlist, the statistical analysis
plan, and other since-decided content) and was deleted rather than kept
in sync. This document covers Stages 3 onward (prompt construction,
inference, evaluation); Stages 1 and 2 (knowledge graph construction and
per-instance subgraph extraction) describe work that happens in the
`pycodekg` repo, which no longer has its own copy of this plan.

This document describes the current design of the comparison experiment for
the Patch-Aware Knowledge Graph Retrieval for LLM-Based Repository-Level
Test Generation project: a KG-augmented test-generation approach (`kg_only`)
against a standard, retrieval-free baseline (`instruct`), evaluated on
TestGenEval (TGE).

## Scope

The experiment targets **test generation, not test completion**, 
no existing test content is shown to either arm.
Concretely, this means using **TestGenEval's `full` setting**: given a code
file, generate a complete test file from scratch. TGE's other three
settings (`first`/`last`/`extra`) are completion tasks that show part of an
existing test file and ask for one more test to be added; they don't fit a
generation-only scope and are not part of the primary result.

**Open decision:** whether the completion settings are dropped from the
codebase entirely or kept available as a secondary/appendix comparison.
Current direction is `full`-only as the primary and, likely, only result.

## Research questions

Locked 2026-08-23 as a full four-RQ set. RQ1 and RQ4 previously had no
dedicated evaluation methodology in this document. The four RQs
themselves stand; the specific procedures now attached to them ("RQ1
validation design" and "RQ4 instrumentation" below) are proposals only,
not yet signed off by the team, see "Open decisions requiring
sign-off."

> **RQ1 (KG Construction Quality):** How accurately does the
> repository-level KG capture code structure, dependencies, and API
> relationships?

> **RQ2 (Patch-Based Structural Retrieval):** How effectively can code
> patches identify affected entities and retrieve relevant KG subgraphs?

> **RQ3 (Test Generation Correctness and Coverage):** To what extent does
> patch-aware KG subgraph retrieval improve the correctness and coverage of
> LLM-generated test cases compared to retrieval-free baselines?

> **RQ4 (Efficiency):** What is the computational cost of the proposed
> approach relative to the quality gains achieved?

TestGenEval instances already
come with `code_file` and `test_file` pre-selected by the benchmark itself,
before either arm ever sees a prompt. The "find the relevant file in a
large, unbounded repository" problem the original motivation describes is
already solved by the benchmark's own construction, for both arms equally.
RQ2 is therefore scoped, within this experiment, to a narrower and more
concrete question than "search a whole repository": given the file TGE
already selected, how well does patch-based analysis identify the specific
function(s) within it that changed, and how well does KG traversal from
that point recover genuinely relevant structural context (callers, callees,
class relationships) versus missing it.

RQ2 is answered as **retrieval characterization, not "relevance"**:
patch-to-seed localization success rate, nodes/edges retrieved, cross-file
node proportion, and caller/callee/sibling/class distribution, alongside
token count. Whether that retrieved context is actually *useful* is left
entirely to RQ3, not claimed twice under RQ2 as well.

RQ3 is the direct behavioural question: does the structured context
`kg_only` retrieves lead to better generated tests than `instruct` reading
the file directly. Both function-scoped and whole-file
correctness/coverage/mutation-score numbers are reported (see Stage 5);
treating the function-scoped figures as primary is a proposal, not yet
signed off by the team, see "Open decisions requiring sign-off."

**Framing note:** this experiment evaluates patch-aware KG retrieval as a
combined context-selection approach against a retrieval-free, focal-file
baseline. It is a system-level comparison, not a causal isolation of graph
structure from cross-file access or structured presentation. Because
`kg_only`'s context can include cross-file code the `instruct` baseline
has no way to see, an observed difference cannot by itself be attributed
specifically to *graph-based* retrieval versus *repository-level access*
versus *structured formatting*. Isolating those would need a semantic
text-RAG baseline and targeted component ablations, both explicitly out of
scope here (see Known limitations).

## Stage 1: Knowledge graph construction

A structural knowledge graph is built for each repository at the exact
commit a dataset instance references (`base_commit`). Every `.py` file is
parsed via AST into graph nodes (files, classes, methods, functions, test
functions) and directed edges (containment, calls, attribute access,
inheritance, existing-test relationships, usage, imports). This produces
one whole-repository graph per commit, reused across every instance that
shares a commit.

## Stage 2: Per-instance subgraph extraction

For each instance, the patch's changed function(s)/class(es) are identified
by resolving the patch's changed lines against real AST-derived line
ranges, checked against both the file's pre-patch and reconstructed
post-patch state. The patch itself is never shown to either arm as prompt
content, it is used only to locate which part of the file the instance is
about. The resolved function/class names, matched to real KG nodes, become
the **seed(s)** for that instance (an instance can have more than one seed
when a patch touches multiple functions).

A bounded breadth-first search from the seed(s): depth 2, over
containment/calls/access/inheritance edges, import edges excluded by
default, collects the surrounding structural context: the seed's own
source, its callers, its callees, and sibling methods on the same class.
The seed's own source is taken from the post-patch version of the file, so
both arms are evaluated against, and shown, the same corrected code for the
function actually under test. The resulting subgraph passes a set of
structural validation checks (no isolated nodes, no dangling edges, seed
connectivity, no duplicate edges) before use.

Existing-test content is deliberately **not** part of either arm's
context, under the `full`-only scope, generation is meant to be a
from-scratch task for both arms, so neither is shown any pre-existing test
material, KG-retrieved or otherwise.

## Stage 3: Prompt construction

Both arms are asked to generate a complete test file for the same target
function(s), with no existing test content shown to either.

- **`instruct`** (baseline): the whole code file, presented as flat text,
  with an instruction naming the specific function(s) the patch changed as
  the focus of testing.
- **`kg_only`**: the seed function's own source, docstring, and structural
  context (callers, callees, sibling methods, and any classes it inherits
  from or instantiates), presented as labelled, KG-derived sections rather
  than a flat file dump.

Both arms are told to focus on the same named function(s), the KG
pipeline's own patch-seeded identification is the single source of truth
for this, shared by both arms rather than reimplemented independently on
each side. This keeps the comparison about *representation* (structured
KG sections vs. a flat file) rather than the two arms silently being asked
to do different jobs.

One deliberate, retained asymmetry: `kg_only`'s structural context is not
restricted to the seed's own file. In the sample checked, the large
majority of `kg_only`'s context nodes are drawn from files other than the
seed's, this is treated as the approach's actual value proposition
(surfacing real cross-file relationships a single-file baseline has no way
to access), not an imbalance to correct. Restricting it to same-file
content to make the comparison "fairer" would remove the one capability
that distinguishes a KG-based approach from `instruct` in the first place.

## Stage 4: Inference

**Revised 2026-08-28, supersedes the 2026-08-22 shortlist/sweep below.**
The team confirmed a new model shortlist and sampling config for the main
results. Main results use **temperature 0.2 with BFS depth 2 for a pass@1
number, and temperature 0.8 with depth 2 for pass@k where k equals 5** (5
independent samples per instance, at least one passing counts as a pass),
not the 0/0.5/1.0 sweep described below, which is now specific to the
BFS-depth ablation only (see that section). Quantization is acceptable
for models that need it to fit M3's GPUs; see the quantization note at
the end of this section for what that trades off.

**Revised shortlist** (2026-08-28), replacing the 2026-08-22 locked
shortlist immediately below:

| Model | Tier | Notes |
|---|---|---|
| GPT-5 | XL | OpenAI API, not self-hosted on M3. Following TestGenEval appendix D.1 (arxiv.org/html/2410.00752v2). |
| Llama-3.1-8B-Instruct | Small | FP16. Confirmed real 2026-08-29: runs on 1x L40S, not H100 (corrects an earlier speculative "1x H100" written before any real testing happened). MAX_NUM_SEQS=24 confirmed safe on one L40S; MAX_NUM_SEQS=32 spiked to 97.3% GPU KV cache usage in testing, real near-OOM risk, don't use 32 for this model. |
| Qwen2.5-Coder-7B-Instruct | Small | No smaller Qwen3-Coder exists; also in ULT. Confirmed real 2026-08-29: runs on 1x L40S, MAX_NUM_SEQS=32 confirmed safe (stable 20-25% GPU KV cache usage, real headroom left even at that value). |
| gpt-oss-20B | Medium | Native MXFP4 (see quantization note below). Real model id `openai/gpt-oss-20b`, 21B total/3.6B active params (MoE), model card states it runs within ~16GB of memory via MXFP4, corrects an earlier speculative "1x H100" written before checking the real model card: 1x L40S (48GB) has real headroom for this, no H100 needed. |
| Qwen3-Coder-30B-A3B-Instruct | Medium | Successor to Qwen2.5-Coder, self-hostable. Confirmed real 2026-08-29: runs on 2x L40S with TENSOR_PARALLEL_SIZE=2, not H100 (corrects an earlier speculative "2x H100"). MAX_NUM_SEQS=32 confirmed safe but close to this model's real ceiling (41-60% GPU KV cache usage under sustained load), don't assume much headroom above 32 without retesting. |
| gpt-oss-120B | Large | Native MXFP4 (see quantization note below). Real model id `openai/gpt-oss-120b`, 117B total/5.1B active params (MoE), model card states it fits on a single 80GB GPU, corrects an earlier speculative "2x H100": 1x H100 should be enough, not 2x. Not yet run or verified on M3. |
| Llama-4-Scout-17B-16E-Instruct | Large | Meta's current-gen model (Apr 2025), cheaper than the 3.1 model it supplements. Also the current candidate for the BFS-depth/temperature sensitivity ablation, pending confirmation once real pass@1/pass@k numbers are in across this shortlist (see BFS-depth ablation section). Already confirmed real 2026-08-25 (see results/RUN_LOG.md): needs 4x H100 (~218GB at float16, doesn't fit the account's normal 4x L40S 192GB quota), TENSOR_PARALLEL_SIZE=4. |

Every model needs a fresh `instruct` and `kg_only` run at both temp=0.2
(pass@1) and temp=0.8 (pass@k=5): 4 runs per model, 28 runs total across
the 7 models above. None of the runs completed so far in `results/RUN_LOG.md`
use either of these temperatures, so none of them count toward this
revised plan.

**pass@k=5 is currently blocked**, not just unimplemented cheaply.
`run_api.py` hardcodes `num_samples` to 1 for this fork's `full` setting
regardless of `--num_samples` (see the "Samples per configuration" note
below, which already flagged this same root cause for the older sweep).
Tracked in [testgeneval#32](https://github.com/miggle711/testgeneval/issues/32).
Evaluation's own container-side code (`evaluate_instance.py`'s
`full_processing`) already loops over a list of samples correctly, so the
fix is confined to the inference side. Needs to land before any pass@5
run can be submitted.

**Quantization note:** not one uniform scheme across the shortlist.
gpt-oss-20B/120B are only released in MXFP4 (OpenAI's native 4-bit
format for these models; there's no fp16 version of the open weights to
quantize differently). The remaining self-hosted models (Llama-3.1-8B,
Qwen2.5-Coder-7B, Qwen3-Coder-30B, Llama-4-Scout) do ship at fp16/bf16
and could share one chosen scheme (e.g. AWQ or GPTQ int4) consistently
within that group. GPT-5 is API-only, so serving precision is entirely
OpenAI's choice, unknown and uncontrolled from this side. Quantization is
lossy (some model/scheme-dependent quality trade for memory headroom and
usually inference speed), so this three-way split (MXFP4, a chosen
uniform scheme, and an uncontrolled API) needs to be reported explicitly
alongside results, not glossed over as one blanket policy. Otherwise
cross-model comparisons would be confounded by differing precision
regimes on top of whatever the models' own capability differences are.

**Older shortlist and sweep, for the record (locked 2026-08-22, superseded
2026-08-28 above):**

Both arms' prompts are sent to each model under evaluation across a
temperature sweep of **0, 0.5, and 1.0**, to check sensitivity to sampling
temperature as well as model choice. Generated test files are stored for
evaluation.

Models are drawn from open-source coding LLMs, spanning small, medium, and
large size tiers, with all candidates below confirmed to exist and be
accessible:

**Small (~7-8B params)**
- Qwen2.5-Coder-7B-Instruct
- DeepSeek-Coder-6.7B-Instruct
- Meta-Llama-3.1-8B-Instruct (gated, requires accepting Meta's license)
- CodeGemma-7B-IT

**Medium (~15-32B params)**
- Qwen2.5-Coder-32B-Instruct
- StarCoder2-15B-Instruct-v0.1
- Codestral-22B-v0.1
- DeepSeek-Coder-V2-Lite-Instruct (16B MoE, ~2.4B active)

**Large (~70-73B params)**
- Meta-Llama-3.1-70B-Instruct (gated)
- Qwen2.5-72B-Instruct (~145GB at float16)

Two substantially larger models (DeepSeek-Coder-V2-Instruct at 235.7B
params, Qwen3-Coder-480B-A35B-Instruct at 480.2B params) were considered
and set aside, their storage footprint at float16 (roughly 470GB and
960GB respectively) is well beyond what the available compute/storage
setup is expected to support. Qwen2.5-72B-Instruct is listed above as a
same-tier substitute.

**Locked shortlist** (2026-08-22, set for the funding/budget request):

- **Small:** Qwen2.5-Coder-7B-Instruct, DeepSeek-Coder-6.7B-Instruct
- **Medium:** Qwen2.5-Coder-32B-Instruct, Codestral-22B-v0.1. Both likely
  need `TENSOR_PARALLEL_SIZE=2` (2x L40S), Qwen2.5-Coder-32B is ~64GB at
  float16, over one 48GB card, and Codestral-22B at ~44GB is too tight
  once KV cache is added.
- **Large:** open, priced as three funding-request options (Qwen2.5-72B-Instruct
  only / both it and Llama-3.1-70B-Instruct / neither), since the
  multi-GPU tensor-parallel setup for 70B+ models on this M3 account is
  untested (see the M3 guide's "Worth knowing" section) and shouldn't be
  folded into a single guessed number.

Dropped from the small/medium tiers: Meta-Llama-3.1-8B-Instruct (gated,
no code-specialization benefit worth the license-approval wait),
CodeGemma-7B-IT (least-benchmarked of the small candidates, redundant
once two code-specialized small models are already covered),
StarCoder2-15B-Instruct-v0.1 (needs the `NO_SYSTEM_MESSAGE=1` workaround
for a known chat-template rejection, see the M3 guide). DeepSeek-Coder-V2-Lite-Instruct
is a possible optional 5th medium-tier model if a cheap extra data point
is wanted later.

**Samples per configuration:** 1 sample at T=0 (deterministic, so repeats
add no signal there), 5 samples at T=0.5 and T=1.0 (genuinely
non-deterministic, so repeats are needed to estimate real variance
rather than report a single noisy draw), both arms. That's 11 samples/arm
times 2 arms, 22 full passes per model, matching `docs/funding-proposal.md`'s
priced run matrix. (This corrects an earlier version of this section
that said 3 samples at the non-deterministic temperatures instead of 5;
5 is the figure actually priced and locked in the funding proposal.)
Implementing real repeats needs either a small code change or a
separate output path per repeat, since `run_api.py` currently hardcodes
1 sample for the `full` setting regardless of what's requested, not
just resubmitting the same job multiple times.

Inference is planned to run on Monash's M3 HPC cluster. M3 does not
support Docker, so evaluation (Stage 5) will either run in a ported Apptainer image or 
run in a separate environment.

## BFS-depth ablation

**Status: proposed, pending team sign-off.** Whether this ablation runs
at all, and the specific depths/subset/models below, have not been
confirmed by the team; see "Open decisions requiring sign-off."

**Depth 3 is now excluded from the depth sweep** (confirmed 2026-08-27,
see [pycodekg#140](https://github.com/miggle711/pycodekg/issues/140)):
BFS depth beyond 2 hops never reaches the rendered `kg_only` prompt at
all. `LLMSerializer._build_context_section` filters every output category
(`callers`, `callees`, `related`, `sibling_methods`) to content within
1-2 hops of the seed, regardless of how far BFS actually explored.
Confirmed empirically across 258 real seeds spanning 7 repos: 0/258
showed any difference between depth-2 and depth-3 rendered output, while
the raw BFS-visited node count kept growing sharply at depth 3 in every
case. Depth **{1, 2}** is therefore the only range where depth has a
measurable effect on what the model actually sees. A depth-3 arm would
be a guaranteed no-op, real compute spent to reproduce depth-2's exact
prompt content.

**Ablation model: not yet confirmed** (as of 2026-08-28). The team's
current intent is to run this ablation on whichever model from the
revised Stage 4 shortlist turns out most performant, once real pass@1/
pass@k numbers exist across that shortlist, not decided in advance.
Llama-4-Scout-17B-16E-Instruct is the current leading candidate (see
Stage 4), but this is not locked. A 6-job temp/depth matrix (temp
0/0.5/1 by depth 1/2) was submitted for Llama-4-Scout on 2026-08-27 as a
pipeline test, then cancelled before consuming real H100 time once it
was confirmed the model choice isn't locked in yet. Revisit once the
Stage 4 main-results batch produces comparable numbers across models.

An additive experiment answering a question more central to this
paper's actual contribution than temperature sensitivity: why depth 2,
specifically, for the bounded BFS subgraph expansion in Stage 2? Runs
`kg_only` only, since BFS depth has no meaning for `instruct`, which does
no graph retrieval, at depths **{1, 2}** (previously {1, 2, 3}, see the
depth-3 exclusion above), on three models spanning the shortlist's size
tiers rather than the full shortlist, on a **150-instance representative
subset** (~12.4% of the 1210-instance dataset) rather than the full
dataset, since this supports a secondary, design-justifying claim rather
than the paper's primary result. (The three-model framing here predates
the 2026-08-28 single-best-performing-model framing described just
above; whether the ablation covers three models or one is itself part
of what's still pending team confirmation.)

**Temperature for the ablation itself: T=0, 1 sample per instance.**
Chosen independently of the primary run's sweep, since this experiment
is characterizing depth's effect, not temperature's; holding temperature
fixed and deterministic keeps that comparison clean rather than adding
a second varying dimension to a supplementary result. Revisit this if a
depth-by-temperature interaction turns out to matter.

Cost derivation is in `docs/funding-proposal.md`'s "BFS-depth ablation"
section: negligible (~$3.54 AUD) relative to the rest of the budget,
since it scales with a small subset times three models times one
temperature rather than the full shortlist times the full dataset times
the sweep.

Measured per depth: retrieved node/edge count, prompt token count,
target-function coverage, and target-function mutation score. Reported
descriptively as a trend across depths 1 and 2 (previously 1/2/3, see
the depth-3 exclusion above; see "Statistical analysis plan" below), not
treated as a significance-testing exercise, since
the goal is characterizing a design choice, not establishing an effect.

## Stage 5: Evaluation

Each generated test file is evaluated in an isolated environment against
the real, current repository state, along three dimensions:

- **Functional correctness**: whether generated tests execute
  successfully, reported both as "at least one test passes" and "every
  test passes", to distinguish partial-suite failures from total ones.
- **Structural coverage**: the proportion of source lines exercised by
  passing tests.
- **Mutation score**: synthetic bugs are injected into the source and the
  generated suite is re-run against each mutant; the fraction of injected
  bugs caught measures whether the tests meaningfully validate behaviour,
  not just execute without error.

Coverage and mutation score are each computed at two scopes: the
conventional whole-file measurement, and a function-scoped measurement
restricted to the lines of the specific function the instance's patch
touched. Both are recorded. Treating the function-scoped figures as the
primary comparison numbers for RQ2/RQ3 has been proposed, since `kg_only`
is structurally only able to generate tests for the function it was given,
while `instruct` has the whole file available and could pick up incidental
coverage or mutation kills elsewhere in the file unrelated to the function
actually under test, but this is not yet a team-confirmed decision.
Whole-file figures would be kept as secondary/contextual numbers rather
than dropped either way, since they remain informative about overall
generated test quality on the file as a whole.

**Proposed, pending team sign-off:** the write-up would report both, as
two named, distinct comparisons: function-scoped coverage/mutation score
as the primary outcomes for RQ2/RQ3, whole-file figures alongside as
secondary/contextual numbers, not dropped. See "Open decisions requiring
sign-off."

Since Docker is unavailable on M3, evaluation runs via an Apptainer port
of the TestGenEval Docker images instead (`swebench_docker/run_apptainer.py`,
selected via `run_evaluation.py --backend apptainer`). Validated for real
2026-08-23: after two fixes (`--writable-tmpfs`, since Apptainer mounts
`.sif` images read-only by default unlike Docker's writable layer; and
`--cleanenv`, since Apptainer leaks the host's environment into the
container by default unlike Docker's clean one), a full real evaluation
of `astropy__astropy-13579` ran end-to-end on M3 and produced real
output: `coverage.json`, whole-file coverage 35.98%, function-scoped
coverage 4.76%, all filtered tests passing. This runs on M3 itself, at
no additional cost, for any repo/version with a `.sif` image already
built. As of this check, only one exists (`astropy_astropy_5.0.sif`);
TestGenEval spans roughly 30 repos, so the remaining `.sif` images still
need building (off-M3, then transferred over, since M3 requires sudo to
build/pull them directly) before a full evaluation pass across the whole
dataset is possible. A local workstation or cloud VM running Docker
remains a fallback only for whichever repos' `.sif` images aren't built
in time, not the default path.

Test execution, mutation testing in particular, is comparatively slow per instance, so this affects how
long a full evaluation pass takes; not yet measured at scale since only
one real instance has been run through this path so far.

## Statistical analysis plan

**Status: proposed, pending team sign-off.** The analysis choices below
(paired bootstrap CIs, McNemar's test, per-model reporting, etc.) have not
been confirmed by the team; see "Open decisions requiring sign-off."

Predefined 2026-08-23, before the primary run, to avoid choosing an
analysis after seeing results. **Experimental unit: the TestGenEval
instance, paired across `instruct` and `kg_only`.** This governs every
row below, since a paired design (same instance, both arms) supports
paired-difference methods that an unpaired design wouldn't.

| Outcome type | Analysis |
|---|---|
| Primary continuous (target-function mutation score, target-function coverage) | Paired differences (Δᵢ = KG − Instruct), 95% paired bootstrap CI |
| Secondary continuous (whole-file mutation/coverage) | Same paired-bootstrap approach |
| Binary (Any Pass / All Pass) | McNemar's test + report discordant pair counts (KG-wins / baseline-wins), not just a p-value |
| Efficiency (token count, latency) | Primarily descriptive, paired, CIs where meaningful |
| RQ1 (edge precision) | Overall + per-relationship-type precision with binomial 95% CIs; separately report the exact/ambiguous/dropped resolution distribution |
| BFS ablation | Descriptive trend across depths 1 and 2 (context size, token cost, quality); not a significance-testing exercise |

**Multi-model / multi-sample handling:** report per-model, not pooled,
unless an aggregation method is explicitly justified in the write-up.
Since the primary run is now 1 sample per instance at T=0 (see Stage 4),
there's no repeated-sample structure to model there; the BFS ablation is
also 1 sample per instance at T=0, so the same applies.

## RQ1 validation design

**Status: proposed, pending team sign-off.** The sample size,
relationship-type split, and validation method below have not been
confirmed by the team; see "Open decisions requiring sign-off."

RQ1 is answered by **stratified manual validation**, not the resolver's
own confidence tags alone (those, exact/ambiguous/dropped, are useful
as a cheap, automatic first-order signal, reported as a distribution
across the dataset, but "the resolver believes this match is exact" is
not the same claim as "this match is correct").

- **80 manually validated relationships**, stratified across
  relationship type. Suggested split ~50 function calls / 40
  attribute-access / 30 inheritance / 30 instantiation / remainder
  other, adjusted proportionally to what the constructed KG actually
  contains once real counts are available.
- Reported as a **stratified validation sample**, not a definitive
  ground truth: precision is reported with binomial 95% CIs (see
  Statistical analysis plan above), not as a bare percentage.
- **Why 80, not a larger sample (e.g. 170):** at ~2 to 3 minutes per
  annotation, 80 is ~3 to 4 hours including setup and reconciliation;
  170 approaches a full working day for diminishing marginal CI-width
  improvement. That time is better spent on the BFS ablation, pipeline
  integrity work, or writing.

## RQ4 instrumentation

**Status: proposed, pending team sign-off.** What "efficiency" means for
RQ4 and which metrics below actually get collected have not been
confirmed by the team; see "Open decisions requiring sign-off."

Cheap to collect alongside inference, no separate experimental pass
needed, just logging that isn't currently in place:

- **KG build time**, measured separately for AST parsing vs. edge
  resolution (as already promised in the paper's Methodology), not as a
  single combined number.
- **Per-instance token count** of the serialized KG context sent to the
  model.
- Distinguish **one-time KG construction cost** (per repository/commit,
  shared across every instance that references that commit) from
  **per-instance retrieval + inference cost** (paid every time, not
  amortized).
- Compare `kg_only` vs `instruct` prompt token counts directly. The
  funding proposal's existing token-cost measurements are the starting
  point for this comparison.

## Known limitations

- **Knowledge graph completeness.** Static analysis currently misses a
  handful of call-resolution patterns: certain dynamic-dispatch/registry
  patterns, chained attribute access on inferred types, some framework
  dispatch conventions (e.g. Django's class-based-view `.as_view()`),
  constructor calls, and dunder/operator-overload calls (e.g. `Q() | Q()`
  routing through `__or__`). These affect how complete the retrieved
  call-graph context is for a given seed, independent of the fairness
  design above, and are tracked as separate KG-quality issues rather than
  correctness bugs in the experiment design itself.
- **Call-graph richness varies per instance.** For a seed with few real
  call-graph connections, `kg_only`'s retrieved context leans more heavily
  on sibling methods and structural relationships than on callers/callees.
  How "rich" a given instance's retrieved context is is not uniform across
  the dataset, and is worth reporting as a property of the sample rather
  than treated as a uniform baseline.
- **RQ2's "issue reports" clause does not apply to this dataset.** TGE's
  schema has no issue-report/problem-statement field at all, so RQ2 as
  originally worded (patches *and* issue reports) is answered here using
  only the patch signal, which is the only one the dataset provides.
- **System-level comparison, not causal isolation (locked 2026-08-23).**
  The experiment evaluates the combined patch-aware KG retrieval approach
  against a focal-file, retrieval-free baseline as a system-level
  intervention, rather than causally isolating individual components.
  Because the retrieved KG context may include cross-file code
  unavailable to the focal-file baseline, observed differences cannot be
  attributed specifically to graph-based structural selection versus
  repository-level context access or structured presentation alone. A
  semantic repository-retrieval (text-RAG) baseline and targeted
  component ablations would be required to isolate these effects, and
  are left to future work.
- **Explicitly out of scope for this paper** (locked 2026-08-23): a
  text-RAG baseline arm, SWT-Bench, issue-report-based retrieval, the
  test completion settings (`first`/`last`/`extra`), and component-level
  causal attribution of why KG context helps (structure vs. scope vs.
  representation are bundled together, not isolated from one another).

## Open decisions requiring sign-off

1. ~~Whether the completion settings (`first`/`last`/`extra`) are dropped
   from the codebase entirely, or retained as an available secondary
   comparison.~~ Resolved 2026-08-23: dropped from primary scope, listed
   under "Explicitly out of scope" above. `first`/`last`/`extra` hand the
   model part of an existing test file, which is structurally incompatible
   with the "no existing test content shown to either arm" principle the
   whole experiment is built around (see Scope).
2. ~~The concrete run matrix for Stage 4, final model shortlist and sample
   counts per configuration.~~ Small/medium tiers locked 2026-08-22 (see
   Stage 4 above). Large tier: both Llama-3.1-70B-Instruct and
   Qwen2.5-72B-Instruct are in fact already complete (1210/1210 each,
   verified 2026-08-22, `TENSOR_PARALLEL_SIZE=4` confirmed working in
   practice); the multi-GPU risk this item used to flag no longer
   applies, this is sunk, already-spent compute.
3. ~~Whether Stage 5 evaluation runs on a local workstation or a shared,
   Docker-capable cluster.~~ Resolved 2026-08-22: evaluation runs on M3
   itself via the Apptainer backend (see Stage 5 above), validated on a
   real instance. The MacBook Pro (M2 Max, Apple Silicon) Docker/Rosetta
   path is no longer the default plan, kept only as a fallback for
   repos whose `.sif` image isn't built in time.
4. Whether Stage 5's function-scoped coverage/mutation numbers are
   reported as the sole primary metric, or alongside whole-file numbers
   as two named comparisons. Proposed (not team-signed-off): function-
   scoped primary, whole-file secondary and reported alongside, see the
   Research questions section's RQ3 note above. Reopened 2026-08-24
   pending team confirmation.
5. ~~This document was mirrored from the `repo-kg-construction`/`pycodekg`
   repo; changes made here needed porting to that repo's canonical
   copy.~~ Resolved 2026-08-24: the `pycodekg` copy had drifted badly out
   of sync and was deleted rather than kept in sync going forward. This
   is now the sole copy of the experiment plan (see the note at the top
   of this file).
6. Whether to drop Stage 4's temperature sweep (0/0.5/1.0) down to a
   single setting, redirecting the freed compute/budget elsewhere (e.g.
   the BFS-depth ablation). Sweep stays as the current default pending
   this decision, to be decided by the team.
7. The RQ1 validation procedure (sample size, relationship-type split,
   validation method), see "RQ1 validation design" above. Drafted as a
   proposal 2026-08-23, not yet signed off by the team.
8. What "efficiency" means for RQ4 and which metrics actually get
   collected, see "RQ4 instrumentation" above. Drafted as a proposal
   2026-08-23, not yet signed off by the team.
9. Whether the BFS-depth ablation runs at all, and if so, its depths/
   subset size/selection procedure/models, see "BFS-depth ablation"
   above. Drafted as a proposal 2026-08-23, not yet signed off by the
   team.
10. The statistical analysis plan in full (paired bootstrap CIs,
    McNemar's test, per-model reporting, multi-sample handling), see
    "Statistical analysis plan" above. Drafted as a proposal 2026-08-23,
    not yet signed off by the team.
