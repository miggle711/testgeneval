# Funding proposal

Patch-Aware Knowledge Graph Retrieval for LLM-Based Repository-Level
Test Generation (ENG4702 FYP). Covers the compute needed to finish the
`instruct` vs `kg_only` comparison described in
`docs/EXPERIMENT_PLAN.md`.

## What this is funding

A dollar budget for the inference side of the complete experiment —
every model, both arms, every temperature — priced as paid API
infrastructure. Evaluation (coverage + mutation testing) is separate: 
it runs on M3 at no cost, validated for real (see below), so it isn't 
part of this dollar figure.

## Experiment scope

Compare a KG-augmented test-generation approach (`kg_only`) against a
retrieval-free baseline (`instruct`) on TestGenEval's `full` setting
(generate a complete test file from scratch), across a locked model
shortlist and a temperature sensitivity sweep.

**Model shortlist:**

| Tier | Models |
|---|---|
| Small | Qwen2.5-Coder-7B-Instruct, DeepSeek-Coder-6.7B-Instruct, Gemma-3-4B-it |
| Medium | Qwen2.5-Coder-32B-Instruct, Codestral-22B-v0.1, StarCoder2-15B-Instruct-v0.1, DeepSeek-Coder-V2-Lite-Instruct, Qwen3-Coder-30B-A3B-Instruct, Phi-4 |
| Large | Llama-3.1-70B-Instruct, Llama-4-Scout-17B-16E-Instruct, Qwen2.5-72B-Instruct, DeepSeek-V3.2 (API-only, see below) |

**How this shortlist was arrived at.** Three papers were checked
directly for their own model choices:

- **TestGenEval** (Meta, Oct 2024) — this project's own benchmark.
  Evaluated Llama 3.1 (8B/70B/405B), GPT-4o, and Codestral; Codestral
  generated the most passing tests of the models it tested.
- **ULT / UnLeakedTestbench** (Huang et al., 2025; subsequently accepted by TOSEM) — relevant as recent unit-test-generation benchmarking evidence, but not directly equivalent to this project's task. ULT evaluates function-level test generation on 3,909 curated real-world Python functions, with emphasis on high cyclomatic complexity and reducing test contamination. It evaluates contemporary open-weight coding models including CodeLlama, DeepSeek-Coder, Gemma 3, Qwen2.5-Coder, and Phi-4-mini. Unlike this project, ULT does not evaluate patch-anchored repository-level retrieval or cross-file KG context. Its value here is primarily in identifying contemporary open models used for unit-test generation and providing recent evidence about model performance on realistic test-generation tasks.
- **Knowledge Graph Based Repository-Level Code Generation**
  (arXiv:2505.14394) — the paper closest to this project's actual
  research question. Built almost entirely on closed-source models
  (GPT-4, GPT-4o, Claude 3.5 Sonnet), open models only as weaker
  baselines.

| Model | Tier | Status | Why |
|---|---|---|---|
| Codestral-22B-v0.1 | Medium | Original, paper precedent | TestGenEval's own best-performing model |
| Llama-3.1-70B-Instruct | Large | Original, paper precedent | Directly evaluated in TestGenEval |
| Qwen2.5-Coder-7B-Instruct | Small | Kept, still current | No smaller Qwen3-Coder exists; also in ULT |
| Qwen2.5-Coder-32B-Instruct | Medium | Kept, still current | Also in ULT; supplemented below |
| StarCoder2-15B-Instruct-v0.1 | Medium | Kept, still current | BigCode hasn't released a StarCoder3 |
| Qwen2.5-72B-Instruct | Large | Kept, still current | No official Qwen3-72B-Instruct exists |
| DeepSeek-Coder-6.7B-Instruct | Small | Kept, discontinued line | DeepSeek folded "Coder" into its general line after V2; also in ULT; 1100/1210 already free on M3 |
| DeepSeek-Coder-V2-Lite-Instruct | Medium | Kept, discontinued line | Same discontinued line; 1117/1210 already free on M3 |
| Qwen3-Coder-30B-A3B-Instruct | Medium | Added, current generation | Real successor to Qwen2.5-Coder; self-hostable on M3 |
| Llama-4-Scout-17B-16E-Instruct | Large | Added, current generation | Meta's real current gen (Apr 2025); cheaper than the 3.1 model it supplements |
| DeepSeek-V3.2 | Large | Added, current generation, API-only | Real successor to the discontinued Coder line; too large (670B+) to self-host on M3 |
| Gemma-3-4B-it | Small | Added, from related work | In ULT; new model family not previously covered |
| Phi-4 | Medium | Added, from related work, substituted | ULT used Phi-4-mini; that exact variant is unpriced everywhere checked, so the real, priced full Phi-4 (14B) is used instead |

**Sampling design:** 1 sample per instance at T=0 (deterministic —
repeats add no signal), 5 samples per instance at T=0.5 and T=1.0
(genuinely non-deterministic, so repeats are needed to estimate real
variance rather than report a single noisy draw). This mirrors the
original upstream TestGenEval codebase's own convention
(`num_samples = 1 if temperature == 0.2 else 5` in `run_pipeline_all.py`),
adapted to this project's actual temperature values (0/0.5/1.0, not
0.2) and to the fact this project only uses the `full` setting, which
the current code hardcodes to 1 sample regardless — implementing real
repeats needs either a small code change or a separate output path per
repeat, not just resubmitting the same job multiple times.

## Cost of the complete experiment

**Run matrix per model:** 1 sample at T=0 + 5 samples at T=0.5 + 5
samples at T=1.0, for both the `instruct` and `kg_only` arms — 22
full 1210-instance passes per model in total (1+5+5 = 11 per arm, ×2
arms).

**Token basis:** every model is priced against the same real,
measured per-instance token count — 9,155.3 input / 1,942.3 output
tokens per instance on average (11,077,947 input / 2,350,216 output
tokens across a real, complete 1210-instance run, taken from
`response.usage.prompt_tokens`/`completion_tokens`, the model's own
real tokenizer, not an estimate). This one real measurement is applied
across every model as the token basis, since the prompt text is the
same regardless of which model reads it; only each model's own price
per token differs. Real per-model token counts would sharpen this
further but aren't available yet for any model besides the one this
was measured on.

**Per-model cost, one full 1210-instance pass**, input and output
priced separately so the arithmetic is checkable (rates verified live
against each provider's own API, 2026-08-23, not third-party
aggregators):

| Model | Input $/M | Output $/M | Input cost/pass | Output cost/pass | Cost/pass |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | $0.025 | $0.05 | $0.28 | $0.12 | $0.39 |
| DeepSeek-Coder-6.7B-Instruct | — | — | — | — | **unpriced** |
| Qwen2.5-Coder-32B-Instruct | $0.06 | $0.15 | $0.66 | $0.35 | $1.02 |
| Codestral-22B-v0.1 | $0.30 | $0.90 | $3.32 | $2.12 | $5.44 |
| StarCoder2-15B-Instruct-v0.1 | $0.15 | $0.15 | $1.66 | $0.35 | $2.01 |
| Llama-3.1-70B-Instruct | $0.40 | $0.40 | $4.43 | $0.94 | $5.37 |
| Qwen2.5-72B-Instruct | $0.36 | $0.40 | $3.99 | $0.94 | $4.93 |
| DeepSeek-Coder-V2-Lite-Instruct | — | — | — | — | **unpriced** |
| Qwen3-Coder-30B-A3B-Instruct | $0.07 | $0.26 | $0.78 | $0.61 | $1.39 |
| DeepSeek-V3.2 | $0.26 | $0.38 | $2.88 | $0.89 | $3.77 |
| Gemma-3-4B-it | $0.05 | $0.10 | $0.55 | $0.24 | $0.79 |
| Phi-4 | $0.07 | $0.14 | $0.78 | $0.33 | $1.10 |
| Llama-4-Scout-17B-16E-Instruct | $0.10 | $0.30 | $1.11 | $0.71 | $1.81 |

**Full experiment cost** (cost/pass × 22 passes per model, summed
across the 11 priced models):

| Model | Cost (× 22 passes) |
|---|---|
| Qwen2.5-Coder-7B-Instruct | $8.68 |
| Qwen2.5-Coder-32B-Instruct | $22.38 |
| Codestral-22B-v0.1 | $119.65 |
| StarCoder2-15B-Instruct-v0.1 | $44.31 |
| Llama-3.1-70B-Instruct | $118.17 |
| Qwen2.5-72B-Instruct | $108.42 |
| Qwen3-Coder-30B-A3B-Instruct | $30.53 |
| DeepSeek-V3.2 | $82.99 |
| Gemma-3-4B-it | $17.36 |
| Phi-4 | $24.30 |
| Llama-4-Scout-17B-16E-Instruct | $39.88 |
| **Total (11 of 13 models)** | **≈$617 USD (~$926 AUD)** |

Two models in the shortlist — DeepSeek-Coder-6.7B-Instruct and
DeepSeek-Coder-V2-Lite-Instruct — genuinely have no API host found on
any provider checked (DeepInfra, Groq, Together, Fireworks). Both are
older/smaller DeepSeek releases most providers have stopped carrying.
No dollar figure exists for these two without either finding another
provider or self-hosting them.

**Evaluation (coverage + mutation testing) runs on M3, at no cost.**
This was validated for real: the Apptainer backend
(`swebench_docker/run_apptainer.py`, `run_evaluation.py --backend
apptainer`) successfully ran a complete evaluation including mutation
testing against a real instance on 2026-08-23, producing real output
(`coverage.json`, real pass/fail results, real timing). It isn't
included as a dollar cost in this document because it doesn't need to
be — it's free M3 compute, the same as most of the inference work
already done for this project.

**Reference figure, if M3 ever became unavailable for evaluation** (not
part of the ask, kept here so the number exists if it's ever needed):
the real timed run used 922.97 CPU-seconds (`user`+`sys` time — actual
compute consumed, the right basis regardless of how many cores a job is
split across) for one instance with mutation testing, ≈0.2564 CPU-hours.
At the full scale needed (22 passes × 1210 instances × 11 priced models
= 292,820 evaluations), that's ≈75,079 CPU-hours. Priced against the
cheapest real cloud CPU rate checked (Hetzner-class, ~$0.004/vCPU-hour,
verified against public 2026 pricing) rather than a mainstream provider
(AWS-class pricing runs roughly 10x higher at this scale, which would
exceed this entire budget on its own): **≈$305 USD (~$458 AUD)**.

## BFS-depth ablation

A small ablation answering "why depth 2?" directly, rather than leaving it an unjustified default — a question a reviewer would otherwise raise, and arguably more central to the paper's contribution than temperature sensitivity. This is additive to the existing run matrix: the 0/0.5/1.0 temperature sweep for the main `instruct`-vs-`kg_only` comparison stays as-is. Runs `kg_only` only (BFS depth has no meaning for `instruct`, which does no graph retrieval), at depths **{1, 2, 3}**, 1 sample per instance at **T=0** — chosen independently of the primary run's sweep, since this ablation characterizes depth's effect, not temperature's, and holding temperature fixed keeps that comparison clean — on **three models spanning the shortlist's tiers** rather than all 13 — enough to check whether the depth-vs-quality trend holds across model size, without scaling a supplementary characterization experiment to the full shortlist's cost:

| Tier | Model |
|---|---|
| Small | Qwen2.5-Coder-7B-Instruct |
| Medium | Qwen2.5-Coder-32B-Instruct |
| Large | Qwen2.5-72B-Instruct |

**Subset size:** 150 instances (~12.4% of the 1210-instance dataset) —
representative, not exhaustive, since this supports a secondary claim
(the depth-2 design choice), not the paper's primary result.

**Cost basis, and its real limit:** the same real per-instance token
count used throughout this document (9,155.3 input / 1,942.3 output,
measured at the current default depth, 2) is applied to all three
depths as a proxy. This is a real approximation, not a measurement —
depth 1's serialized context is very likely smaller and depth 3's
larger, since BFS depth directly changes how much structural context
gets included in the prompt. The ablation itself is what produces the
real per-depth token counts; this estimate necessarily predates that
data and should be treated as directional, not exact.

**Cost:** 150 instances × 3 depths = 450 `kg_only` calls per model,
priced at each model's already-established per-instance rate (cost/pass
÷ 1210, from the per-model table above):

| Model | Cost/instance | × 450 calls |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | $0.000322 | $0.15 |
| Qwen2.5-Coder-32B-Instruct | $0.000843 | $0.38 |
| Qwen2.5-72B-Instruct | $0.004074 | $1.83 |
| **Total** | | **≈$2.36 USD (~$3.54 AUD)** |

Negligible relative to the rest of the budget — the ablation is cheap
because it runs a small subset against three models, not the full
shortlist against the full dataset. The BFS traversal itself (across
all three depths) is free M3 compute, identical to the rest of KG
construction; only the resulting LLM calls carry a dollar cost.

**Additive to the priced inference line below, not a replacement for
any part of it** — the $926 AUD figure reflects the confirmed 0/0.5/1.0
temperature sweep for the primary `instruct`-vs-`kg_only` comparison,
unaffected by this ablation. The ablation's own $3.54 AUD is a separate
line in the budget table below.

## Final budget request

DeepSeek-Coder-6.7B-Instruct and DeepSeek-Coder-V2-Lite-Instruct have
no API host (see the shortlist rationale above), but need no dollar
cost either — both already run on M3 for free, both are already mostly
complete there (1100/1210, 1117/1210), and M3 can keep running whatever
instances are left the same way. No rented-GPU line is needed for them.

**General contingency**: 25% of the priced inference cost ($926 AUD) =
**≈$231.50 AUD** — the 5-sample repeat mechanism isn't implemented yet
(`run_api.py` hardcodes 1 sample for `full` regardless of what's
requested), and three new models (Llama-4-Scout, Gemma-3, Phi-4) have
never run through this pipeline, which has a real history of per-model
surprises (e.g. StarCoder2's chat-template rejection, `docs/GUIDE.md`).

**Requested: ≈$1,161.04 AUD**

| Component | Amount | How it was derived |
|---|---|---|
| Priced inference, 11 models, full run matrix | $926 AUD | Real token count × real per-model API rates (see table above) — 0/0.5/1.0 temperature sweep |
| BFS-depth ablation, 3 models × 150-instance subset × 3 depths, T=0 | $3.54 AUD | See "BFS-depth ablation" section above — additive, independent of the sweep above |
| General contingency, 25% | $231.50 AUD | Standard buffer rate + repeat-mechanism/new-model risk × the priced-inference row only |
| **Total requested** | **≈$1,161.04 AUD** | |

**Where this goes:** two subscriptions cover all 11 priced models.
`inference/api/run_api.py` already supports either via
`LOCAL_MODEL_BASE_URL`/`LOCAL_MODEL_API_KEY`, no code changes needed.

| Provider | Covers | Share of priced inference |
|---|---|---|
| DeepInfra | 10 of 11 models (everything except Codestral) | $746 AUD |
| Mistral AI (La Plateforme) | Codestral-22B-v0.1 | $180 AUD |

## Disclaimers

- **Data provenance**: the real per-instance token count this whole
  inference cost table is built on (11,077,947 input / 2,350,216 output
  tokens) was measured from a real completed run on M3. Inference is 
  priced as paid API infrastructure in this document; evaluation runs 
  on M3 for free (see above).
- Real timing/cost data: one real per-instance token count (measured on
  one model, Llama-3.1-70B-Instruct, applied to every other model as a
  proxy — see the token basis note above) and two real evaluation
  timing samples (one without mutation testing, ~36 seconds; one with,
  11m20.696s / 922.97 CPU-seconds). Every other model's token count is
  still a proxy, not independently measured — treat the inference
  totals as a solid planning estimate, not an exact quote.
- API pricing was pulled live on 2026-08-23 and is subject to change.
