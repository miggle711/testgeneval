# Inference run log

Tracks completion counts and known failure causes per model/arm, since a
model's raw completed-instance count alone doesn't distinguish "ran clean"
from "lost instances to context overflow" -- that distinction matters for
comparing models fairly and should be reported alongside any pass-rate
numbers, not silently absorbed into them.

Full dataset (`kjain14/testgeneval`) is 1210 instances. All runs below are
the `instruct` (baseline) arm, temperature 0, `full` setting only.

## instruct arm

### Small (~7-8B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 1210 / 1210 | 0 | Clean run, no losses. Concurrency calibration (2026-08-29, one L40S): MAX_NUM_SEQS=8 (the script's default) never went past ~8.5% GPU KV cache usage even at full concurrency. Retested at MAX_NUM_SEQS=32: stable in a 20-25% band, no spikes, confirmed safe; still not close to this model's real ceiling, unlike Qwen3-Coder-30B-A3B-Instruct's ~50-60% at the same value. A higher MAX_NUM_SEQS is likely fine too but hasn't been tested yet. |
| DeepSeek-Coder-6.7B-Instruct | 1100 / 1210 | 110 | 32768 context, largest `instruct` prompts exceed it. |
| CodeGemma-7B-IT | not completed | -- | Abandoned. MAX_MODEL_LEN=8192 (model's real limit) would produce an even higher loss rate than StarCoder2; deprioritized in favor of medium-tier models instead of forcing a run through it. |
| Meta-Llama-3.1-8B-Instruct | 1210 / 1210 | 0 | Access was in fact confirmed and a full 8-shard run completed; this row was stale (previously read "not run, license access not yet confirmed"). Corrected 2026-08-22 after verifying real `wc -l` counts on all 8 shard files summed to 1210, merged via `cat` per GUIDE.md. Concurrency calibration (2026-08-29, one L40S): MAX_NUM_SEQS=32 is not safe for this model, real risk confirmed, GPU KV cache usage spiked to 97.3% early in a NUM_SAMPLES=5/temperature=0.8 test (unlike Qwen2.5-Coder-7B-Instruct at the same size class and MAX_NUM_SEQS, this model produced some very long completions, e.g. output_tokens=20480, that clustered together and drove cache demand hard). Survived that spike without crashing, but too close to the edge to trust for a real run. Retested at MAX_NUM_SEQS=24: stayed in a 30-55% band with no spike, confirmed safe. Use 24, not 32, for this model. |
| Gemma-3-4B-it | 1121 / 1210 | 89 | From the team's funding proposal shortlist (`docs/funding-proposal.md`), gated (Google's license), access confirmed. Rejects `float16` outright at vLLM startup ("The model type 'gemma3' does not support float16. Reason: Numerical instability."), needed `m3_run_inference.slurm`'s new `DTYPE` env var (`DTYPE=bfloat16`) to run at all. Job 59426744, completed 2026-08-24, ~1h28m. Model's real context window is 128K, but `MAX_MODEL_LEN` wasn't overridden for this run and stayed at the script's default (32768), so all 89 losses are the same context-length `BadRequestError` seen elsewhere in this doc, not a Gemma3-specific issue. Worth a rerun with `MAX_MODEL_LEN` raised to actually use the model's real context budget, if a cleaner comparison point is wanted later. |
| Phi-4 | 885 / 1210 | 325 | From the funding proposal shortlist, not gated. Only 16K context, the smallest of any model run so far. First attempt (job 59425627) crashed at vLLM startup: `max_model_len (32768) is greater than the derived max_model_len (16384)`, the script's default `MAX_MODEL_LEN` exceeds this model's real limit, same class of fix StarCoder2 needed. Resubmitted with `MAX_MODEL_LEN=16384` (job 59429296) by wtho0016, completed 2026-08-25, 2h43m06s. 325 losses (~27%) confirmed via `grep -c "Failed, skipping"` plus checking the actual error text to be the same context-length `BadRequestError` pattern, comparable to StarCoder2's ~31% at the same 16384 window. |

### Medium (~15-32B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Instruct | 1117 / 1210 | 93 | 32768 context. |
| StarCoder2-15B-Instruct-v0.1 | 832 / 1210 | 378 | MAX_MODEL_LEN=16384 (model's real limit, confirmed via vLLM's own derived-max-model-len error). Highest loss rate so far (~31%), consistent with having half the context budget of the 32768-context models. |
| Qwen2.5-Coder-32B-Instruct | 1210 / 1210 | 0 | Needs TENSOR_PARALLEL_SIZE=2 (32B doesn't fit one 48GB L40S at float16). All 8 shards completed, merged via `cat` per GUIDE.md, no losses. |
| Codestral-22B-v0.1 | 1101 / 1210 | 109 | ~44GB at float16, tight against 48GB L40S, real OOM risk. Two earlier stalled attempts (323/1210) were caused by `/fs04` sitting at 99% capacity, traced via 2625 `Connection error`s in one job's log (`slurm-59362972.out`), the only job in the shared directory with that signature. Freed ~341GB 2026-08-24 by deleting confirmed-complete models' HF caches with each owner's go-ahead, then restarted the run clean from a fresh output file (job 59400996, 2h21m, completed cleanly). All 109 losses confirmed via `grep -c "Failed, skipping"` plus checking the actual error text (normalizing digits and deduplicating over every `API Error` line) to be the same 32768-token context-length `BadRequestError`, no `Connection error`s this run, consistent with DeepSeek-6.7B/DeepSeek-V2-Lite's loss pattern. |
| Qwen3-Coder-30B-A3B-Instruct | 1210 / 1210 | 0 | From the funding proposal shortlist, real successor to Qwen2.5-Coder. MoE (128 experts, 8 active, 30.5B total/3.3B active params), not gated. TENSOR_PARALLEL_SIZE=2. 256K native context, well beyond every other model's context ceiling, confirmed zero losses as expected. Job 59425624, submitted by wtho0016, completed 2026-08-25, 3h19m40s (~9.9s/instance). Concurrency calibration (2026-08-29, 2x L40S): MAX_NUM_SEQS=32 (up from the script's default of 8) works and stays safe, GPU KV cache usage oscillated between ~41% and ~60% under sustained full-concurrency load (Running: 32 reqs, Waiting: 120+ throughout), never approaching the ~80-85% danger zone. Unlike the 7B model, this one doesn't have obvious further headroom at this MAX_NUM_SEQS; treat 32 as close to this model's real ceiling at TENSOR_PARALLEL_SIZE=2 rather than an underused floor. First attempt at MAX_NUM_SEQS=32 timed out at vLLM startup (300s default, weight loading alone took 229s); fixed by raising VLLM_STARTUP_TIMEOUT to 900s, since made the script's own default (testgeneval#36). |
| gpt-oss-20B | not completed | -- | From the model shortlist revision (2026-08-28), suggested by Aaron. Real model id `openai/gpt-oss-20b`, 21B total/3.6B active params (MoE), native MXFP4. Needs 1x H100 (not L40S: MXFP4 requires GPU compute capability >= 9.0, L40S is 8.9, would fall back to dequantizing to bf16 at ~4x the memory, real risk on a 48GB card, see docs/EXPERIMENT_PLAN.md's shortlist table). First smoke test attempt (job 59557962) crashed at vLLM config validation, not a real hardware/quota issue: `torch.float16 is not supported for quantization method gpt_oss_mxfp4. Supported dtypes: [torch.bfloat16]`. Same class of fix `google/gemma-3-4b-it` needed, `DTYPE=bfloat16`. Second attempt (job 59562182, with DTYPE=bfloat16) passed the config check but hit a different real problem: it and the gpt-oss-120B smoke test (job 59562183) both landed on the same node (`m3h101`) and neither had `VLLM_PORT` set, so both defaulted to port 8003. The 120B job's client ended up talking to the 20B job's server, producing a wall of "model does not exist" 404s for the 120B job, while the 20B job itself was serving real requests the whole time under a misleading appearance of failure. Confirmed the failures were handled safely (no garbage written to the output file, `process_instance` returns `None` on failure and writes nothing), just wasted runtime. Fixed by resubmitting both with explicit distinct `VLLM_PORT` (8003 for 20B, 8004 for 120B), job 59562654. This retest did reach its own model, but revealed a third, real, unrelated problem: only 109/160 instances completed (51 lost), all to `Error: 'NoneType' object has no attribute 'replace'`, a client-side crash in `postprocess_python_output` when `message.content` comes back `None`. 71/84 real successful requests (85%) hit exactly the default 4096 output token cap, this model isn't in `OUTPUT_LIMITS`. Root cause: gpt-oss's "harmony" response format emits reasoning content in a separate channel before the final answer, and vLLM's Harmony parser only populates `message.content` from the final channel, a low `max_tokens` (this fork's default is 4096) makes an empty/null response likely if the reasoning channel eats the whole budget first. Confirmed as a known, real vLLM/gpt-oss issue via vllm-project/vllm#32791 and the model's own HuggingFace discussion page, not a bug specific to this fork. Filed as testgeneval#40, needs `OUTPUT_LIMITS` raised for both gpt-oss models before a real production run. |

### Large (~70-110B params)

| Model | Completed | Context-overflow losses | Notes |
|---|---|---|---|
| Meta-Llama-3.1-70B-Instruct | 1210 / 1210 | 0 | Gated, access confirmed. TENSOR_PARALLEL_SIZE=4, 8 shards, all completed and merged via `cat` 2026-08-22. Real per-instance token counts logged in the job's own `slurm-*.out` files (via `run_api.py`'s `input_tokens=`/`output_tokens=` logging): 11,077,947 input / 2,350,216 output tokens total across all 1210 instances. Row previously read "in progress"; corrected after verifying real `wc -l` counts. |
| Qwen2.5-72B-Instruct | 1210 / 1210 | 0 | ~145GB at float16. TENSOR_PARALLEL_SIZE=4, 8 shards submitted by jliu0290 under his own M3 account (separate GPU quota from mvar0010's runs), all completed and merged via `cat` 2026-08-22. Real per-instance token counts aren't available here -- jliu0290's `slurm-*.out` logs live under his own directory, not this shared one; worth pulling if exact token counts for this model are ever needed. Row previously read "in progress"; corrected after verifying real `wc -l` counts. |
| Llama-4-Scout-17B-16E-Instruct | 1210 / 1210 | 0 | From the funding proposal shortlist. Despite the "17B" in the name, this is actually 109B total params (17B active, MoE, 16 experts) -- confirmed against the model card, not the name. ~218GB at float16, doesn't fit the account's normal 4x L40S (48GB) quota (192GB total, less than the model needs). Ran instead on the `m3h` partition's H100 nodes (confirmed real: `srun --partition=m3h --qos=m3h --gres=gpu:H100:1` allocated an 80GB H100, `al49`'s association already includes `m3h` in its QOS list), TENSOR_PARALLEL_SIZE=4 across 4x H100 (320GB total). Gated, access confirmed. Job 59426743, completed 2026-08-25, 52m09s, 2.59s/instance average -- by far the fastest large-tier model despite being the largest by total parameter count, since MoE inference cost scales with active params (17B) not total params (109B), plus 4x H100's extra memory bandwidth over the L40S setup used for the dense 70B/72B models. Zero context-overflow losses, the only new funding-proposal model to complete clean. |
| gpt-oss-120B | not completed | -- | From the model shortlist revision (2026-08-28), suggested by Aaron. Real model id `openai/gpt-oss-120b`, 117B total/5.1B active params (MoE), native MXFP4. Needs 1x H100 (fits on a single 80GB GPU per the model card, but only because MXFP4 requires GPU compute capability >= 9.0, same requirement as gpt-oss-20B above). First smoke test attempt (job 59557964) crashed with the identical error as gpt-oss-20B: `torch.float16 is not supported for quantization method gpt_oss_mxfp4. Supported dtypes: [torch.bfloat16]`. Fix: `DTYPE=bfloat16`. Second attempt (job 59562183) never actually reached its own model: it landed on the same node as the gpt-oss-20B retest (job 59562182, see that row above for the full port-collision story) and, with no `VLLM_PORT` set on either job, its `run_api.py` client ended up talking to gpt-oss-20B's server instead, producing a wall of "model does not exist" 404s. Fixed by resubmitting with an explicit distinct `VLLM_PORT=8004` (job 59562655, alongside gpt-oss-20B's retest on 8003), confirmed real `200 OK` responses for `openai/gpt-oss-120b` specifically this time. |

Tiers match `docs/EXPERIMENT_PLAN.md`'s Stage 4 shortlist, plus the new
Small/Medium/Large models added to the team's funding proposal
(`docs/funding-proposal.md`, everything except DeepSeek-V3.2, too large
to self-host and out of scope for this M3 run).

All 8 completed `instruct` models above (everything except the 4 new
funding-proposal models, still in progress) have their merged
predictions gzipped locally and backed up to the team's shared Google
Drive folder
(https://drive.google.com/drive/folders/1YWtuRmoFjjiQx6GntXZamsel1iIY0Mb1)
as of 2026-08-24, so a fresh evaluation run doesn't need to re-pull them
from M3.

## kg_only arm

`kg_prompts.json` is in fact already built and complete at the repo root
-- verified 2026-08-22 via `python3 -c "import json; print(len(json.load(open('kg_prompts.json'))))"`,
which returned 1210 (the full dataset). This row previously read "KG build
done, `kg_prompts.json` generation is the next step"; that step was already
finished by the time this was checked.

First real `kg_only` run submitted 2026-08-22: Qwen2.5-Coder-7B-Instruct,
full `kjain14/testgeneval` dataset, temperature 0, job 59370304 (single
job, no sharding). Chosen because this model's `instruct` arm is already
clean (1210/1210, no context-overflow losses), giving a matched head-to-head
comparison once this finishes. Update this row with the real completed
count once the job finishes.

This run, and every other `kg_only` result in this document so far, used a
`kg_prompts.json` built before four real determinism/correctness fixes on
the pycodekg side (miggle711/pycodekg#144, #145, #146, and
miggle711/testgeneval#31's target-range fix on this repo's own evaluation
code). A rebuilt, fixed `kg_prompts_depth1.json`/`kg_prompts_depth2.json`
pair now exists (built on M3 directly, see pycodekg's `m3_build_kgs.slurm`/
`m3_build_kg_prompts.slurm`, full 1210-instance dataset, all four fixes
included), superseding the old file. Confirmed via a direct diff that
`target_functions`/`target_classes` (the only fields `instruct` reads from
this file) are identical between old and new for all 1210 instances, so
existing `instruct` runs' target-function naming is still valid and does
not need to be rerun on that basis alone. `kg_only` results do need
rerunning against the new file, since the fixes changed the actual
structural context content, not just target-function naming.

**The forward-looking run plan (model shortlist, sampling config, ablation
scope) lives in `docs/EXPERIMENT_PLAN.md`, not here** — this document
previously had a "Main results plan" section here that duplicated and, in
places, conflicted with EXPERIMENT_PLAN.md's own Stage 4/BFS-ablation
sections (different ablation model count, different depths, different
subset size). Removed 2026-08-28 in favor of a single source of truth;
see EXPERIMENT_PLAN.md for the current plan and RUN_LOG.md's own role
description at the top of this file for what belongs here instead (real
completion counts and failure causes for runs that have actually
happened).

## Current batch: temperature 0.2/0.8, depth 2, real OUTPUT_LIMITS (2026-08-29)

**Real status snapshot as of 2026-09-02, not a history, just what is
actually usable right now.** Everything below this table has the full
story of how each row got here, real job IDs and real causes; this is
just the current state, updated as jobs progress. `t=0.2` means
temperature 0.2 (pass@1), `t=0.8` means temperature 0.8 (pass@k=5,
`NUM_SAMPLES=5`).

**Real, team-wide infrastructure incident, 2026-09-01/02, worth flagging
here since it took down real production jobs, not this project's own
bug:** `/fs04` genuinely hit 100% capacity (`lfs quota -g al49 /fs04`
showed `3.0T`/`3.0T`, 0 available), and two real, otherwise-healthy
production jobs (59597322, 59597542, both real `t=0.8` runs with 15 to
19 real hours of progress) crashed hard with
`OSError: [Errno 122] Disk quota exceeded` then
`OSError: [Errno 5] Input/output error` trying to write a completed
result, a real crash, not the retry-and-skip path `process_instance`
already handles cleanly for a bad API response. Confirmed resolved as
of 2026-09-02, the real allocation is now `5.1T` total with `1.6T`
free.

| Model | Arm | Temp | Status | Real note |
|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | instruct | 0.2 | Paused | Blocked on testgeneval#43 (repetition loops), not resubmitted |
| Meta-Llama-3.1-8B-Instruct | kg_only | 0.2 | Paused | Same as above |
| Meta-Llama-3.1-8B-Instruct | instruct | 0.8 | Running (59670063) | Real fix applied (`MAX_MODEL_LEN=65536`), submitted by mvar0010 directly rather than jliu0290, since jliu0290's own HuggingFace account did not yet have real access to this gated model approved (two real attempts, 59669230/59669298, both failed on a 401 gated repo error in under 40 seconds each) |
| Meta-Llama-3.1-8B-Instruct | kg_only | 0.8 | Running (59670064) | Same real fix and same reassignment reason as instruct above |
| Qwen3-Coder-30B-A3B-Instruct | instruct | 0.2 | Paused | Blocked on testgeneval#43 |
| Qwen3-Coder-30B-A3B-Instruct | kg_only | 0.2 | Paused | Blocked on testgeneval#43 |
| Qwen3-Coder-30B-A3B-Instruct | instruct | 0.8 | Resubmit ready at `MAX_MODEL_LEN=98304` | The `65536` fix genuinely was not enough, a real 57537 input token prompt hit that wall too before the disk crash, real max seen since (from the completed kg_only file) is around 68389 estimated tokens, resubmitting with real margin above both, `MAX_NUM_SEQS` lowered to 6 to compensate for the larger real KV cache reservation, mvar0010 running this one directly given the real risk of a third underestimate |
| Qwen3-Coder-30B-A3B-Instruct | kg_only | 0.8 | **COMPLETED, real, mostly clean (59597323)** | 1208/1210 real instances, 6 real context length errors, both from prompts exceeding even 65536, a real, large improvement over the pre-fix 25 lost, not perfectly clean, accepted as a real, small, disclosed residual rather than re-run |
| Qwen3-4B-Instruct-2507 | instruct | 0.2 | Paused | Blocked on testgeneval#43 |
| Qwen3-4B-Instruct-2507 | kg_only | 0.2 | Paused | Blocked on testgeneval#43 |
| Qwen3-4B-Instruct-2507 | instruct | 0.8 | Running (59669630) | Real fix applied (`MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS` raised to 20 given the real unused concurrency headroom found earlier, KV cache was only 12 to 25% at the old `MAX_NUM_SEQS=8`), running under jliu0290's account |
| Qwen3-4B-Instruct-2507 | kg_only | 0.8 | Running (59669631) | Same real fix, running under jliu0290's account. The earlier corrupted file (job 59573909, 45/1210 real losses) was moved aside to `results/kg_only/_context_mismatch_20260902/` first so this resubmit does not inherit it via `existing_ids` |
| gpt-oss-20B | instruct | 0.2 | Real `testgenevallite`-scale data only, final | Job 59566582 COMPLETED, real truncation fix confirmed (0 requests near the 48000 cap), but a real, separate, external vLLM/harmony quirk still loses 16/160 (10%) instances regardless of budget, confirmed final via real `completion_tokens` values (2682 to 29948, nowhere near the cap), see testgeneval#40. No full `testgeneval`-scale run yet |
| gpt-oss-120B | any | any | Blocked, real attempts failed, retry ready | mvar0010 taking this over directly from jliu0290 (real `DTYPE=bfloat16` fix ready to submit under mvar0010's own account instead), given jliu0290 is now focused on the Llama-3.1-8B/Qwen3-4B batch above. Real team-wide disk space issue that blocked the last attempt (job 59594023) is now resolved |
| Llama-4-Scout-17B-16E-Instruct | instruct | 0.8 | Queued (59670222) | First real attempt (59621719) failed on a real `vllm server did not become ready within 900s`, this model's real 4 GPU tensor-parallel load (weight loading alone took 447.56 real seconds, plus real torch.compile overhead) genuinely needs more than the script's 900s `VLLM_STARTUP_TIMEOUT` default. Resubmitted by wtho0016 with `VLLM_STARTUP_TIMEOUT=1800`, not yet confirmed sufficient |
| Llama-4-Scout-17B-16E-Instruct | kg_only | 0.8 | Queued (59670226) | Same real cause as instruct above (first attempt 59621720), same fix, resubmitted by wtho0016 |
| Llama-4-Scout-17B-16E-Instruct | any | 0.2 | Paused | Blocked on testgeneval#43 |

**Real division of labor as of 2026-09-02**, since three real M3 accounts are now
running different parts of this batch in parallel, updated as the real, actual
assignments shifted through the day: mvar0010 is running the
Qwen3-Coder-30B-A3B-Instruct `instruct` resubmit directly (the higher-risk
`MAX_MODEL_LEN=98304` value), gpt-oss-120B (taken over from jliu0290, a real
`t=0.2` calibration/smoke test on `testgenevallite`, not counted toward the
paused `t=0.2` production batch, since it exists only to verify the
`DTYPE`/`MAX_NUM_SEQS` fix works), and, as of later in the day, both
Meta-Llama-3.1-8B-Instruct `t=0.8` jobs too, reassigned from jliu0290 after
their own HuggingFace account turned out not to have real access to this
gated model approved yet. jliu0290 is running the 2 real Qwen3-4B-Instruct-2507
`t=0.8` jobs under their own account (59669630 instruct, 59669631 kg_only),
both started cleanly. wtho0016 is assigned Llama-4-Scout's 2 real `t=0.8`
resubmits with the longer startup
timeout, not yet resubmitted as of this writing. A real command from
jliu0290 at `t=0.2` for Meta-Llama-3.1-8B-Instruct was caught and held before
submission, since real `t=0.2` production data stays paused for every model
pending testgeneval#43's team decision, regardless of which account would
run it.

The rest of this section has the full real story behind the table above,
job IDs and real causes for each row's status. This whole batch is a
distinct phase from every table earlier in this document: those used
temperature 0 (pass@1 only) and, for `instruct`/some earlier `kg_only`
rows, predate the `OUTPUT_LIMITS` fix (testgeneval#41) entirely, so
their completion counts are not directly comparable to this batch. This
batch follows `EXPERIMENT_PLAN.md`'s revised Stage 4 config: temperature
0.2/depth 2 for pass@1, temperature 0.8/depth 2 (`NUM_SAMPLES=5`) for
pass@k=5, `KG_PROMPTS_PATH=kg_prompts_depth2.json`.

**Real, large-scale truncation bug discovered before this batch started
running clean.** `OUTPUT_LIMITS` (`inference/api/run_api.py`) had no real
entries for any shortlist model, so every model fell back to the default
4096 output tokens. Measured on real completed `instruct`/`kg_only` pass@1
data at the time (since discarded, see below): Meta-Llama-3.1-8B-Instruct
55-63% of completions near/at the cap, Qwen3-Coder-30B-A3B-Instruct
24-33%. Confirmed real via AST-parse plus manual inspection, not just a
char-length proxy: some near-cap completions were outright syntax errors
from mid-token cutoff, others parsed as valid Python but were degenerate
repetition-loop garbage the model fell into near the length ceiling.
Filed as testgeneval#41 (broader than testgeneval#40's gpt-oss-specific
finding, same root cause and fix location). All affected data
(`Meta-Llama-3.1-8B-Instruct`/`Qwen3-Coder-30B-A3B-Instruct` `instruct`
and `kg_only`, both temperatures) moved to
`results/instruct/_corrupted_output_limit_20260829/` and
`results/kg_only/_corrupted_output_limit_20260829/` rather than deleted,
and is being regenerated under the fix.

Fixed 2026-08-29 (`51b2370`, `e1fc573`, `1c1d003` on `main`): real
`OUTPUT_LIMITS`/`MODEL_LIMITS` entries per model, keyed by real
`model_name_or_path`. 8000 tokens for Meta-Llama-3.1-8B-Instruct,
Qwen3-Coder-30B-A3B-Instruct, Llama-4-Scout-17B-16E-Instruct
(directly measured: real max legitimate, non-truncated completion length
across those three models' pre-fix data was 5828/5042/3582 tokens
respectively, 8000 leaves real margin above all three) and, by
extrapolation only (no completed data yet), Qwen3-4B-Instruct-2507.
gpt-oss-20B/120B needed a much larger, separately-calibrated budget, see
their own rows below. Also added a guard against `response.choices[i]
.message.content` coming back `None` (confirmed real for gpt-oss, see
testgeneval#40) before it reaches `postprocess_fn`, so one bad sample in
an `n>1` batch no longer discards the whole instance's other, real
completions.

`Qwen2.5-Coder-7B-Instruct` dropped from this batch: its real native
context (32768, no YaRN, confirmed via the model's own `config.json`) is
below this dataset's real max prompt size (~43K tokens, measured across
the full `testgeneval` dataset for both arms). Filed as testgeneval#42.
`Qwen3-4B-Instruct-2507` (native 262144 context, confirmed via
`config.json`) takes its place in the shortlist instead, chosen after
checking every smaller Qwen2.5-Coder variant (1.5B/3B/7B/AWQ) and
sharing the same 32768 ceiling regardless of size, and `CodeQwen1.5-7B-
Chat` only reaching 65536, still short of the real max. Not yet
independently calibrated for `MAX_NUM_SEQS`; running conservatively at
the script's own default (8) for its first real production run.

**Real per-instance test:** hit one real config bug resubmitting this
batch, worth flagging so it isn't repeated -- `KG_PROMPTS_PATH` defaults
to the relative filename `kg_prompts.json`, which does not exist in this
checkout (only `kg_prompts_depth1.json`/`kg_prompts_depth2.json` do,
from the earlier M3 KG build). Two `kg_only` jobs failed in under 90
seconds with `FileNotFoundError: [Errno 2] No such file or directory:
'kg_prompts.json'` before this was caught. Pass `KG_PROMPTS_PATH`
explicitly on every submission, including `instruct` (which reads it
more loosely for target-function naming, so it will not hard-crash the
same way, but should still get the right file for a valid comparison).

**gpt-oss-20B output budget calibration**, real data, `kjain14/testgenevallite`
(84 real instances). First attempt (job 59565919, `MAX_CONCURRENCY=1`,
the script's default) was too slow to be worth waiting on; cancelled and
resubmitted at `MAX_NUM_SEQS=32`/`MAX_CONCURRENCY=32` (job 59565984,
untested before this, chosen as a moderate starting point, confirmed
safe: GPU KV cache usage stayed under ~12% throughout, real headroom
left). Job 59565984's real data, first pass at the still-unmeasured
32000 placeholder from testgeneval#41's initial fix: real completions
distribution (48 fresh instances, `existing_ids` skipped the rest as
already done from an earlier smoke test) min=2022, median=8871,
p90=22852, max=30763 tokens, but 10/48 (21%) still hit the
`None`-content guard, all `finish_reason=length`, confirming 32000 was
genuinely too low, not just untested. Raised to 48000 (`e1fc573`),
old output moved aside to `results/instruct/_pre_48k_calibration_20260829/`,
rerun clean on the full 84 instances (job 59566179): max observed
dropped to 31651 (real margin under 48000 now), but 5/160 requests
(3.1%, `finish_reason=length`) still hit the guard, concentrated in
sympy (4/5). Added `completion_tokens` directly to the guard's warning
log (`1c1d003`) to diagnose these without relying on log-line-adjacency
correlation (unreliable under `MAX_CONCURRENCY>1`, confirmed: all 5
warnings from job 59566179 printed consecutively at the very end of the
log, not spread through it live). Old pre-diagnostic-logging output
moved aside to `results/instruct/_pre_diagnostic_run_20260829/`. Real
per-instance comparison against `Meta-Llama-3.1-8B-Instruct`/
`Qwen3-Coder-30B-A3B-Instruct`'s pre-fix data for the same 5 instance ids:
3/5 (`sympy__sympy-13471`, `sympy__sympy-14024`, `astropy__astropy-7746`)
were solved cleanly by both other models well under 8000 tokens, so
gpt-oss's failure there looks like its own reasoning-budget cost, not
genuine task difficulty. The other 2/5 (`sympy__sympy-21171`,
`sympy__sympy-22714`) failed for the other two models as well (one a
real repetition-loop failure, the other an unexplained syntax error,
neither near their own 4096-token pre-fix cap), suggesting a real,
cross-model hard tail unrelated to output budget. A second diagnostic
run (job 59566582 under mvar0010) is queued on the `m3h` H100 partition
to get real `completion_tokens` values for these specific failures
before deciding whether to raise the cap again; real queue estimate at
submission time was ~18-25 hours out due to `m3h` priority contention,
not something fixable from this side.

**gpt-oss-120B calibration**, same real methodology, real code, on a
teammate's (`jliu0290`) own M3 account/GPU quota rather than mvar0010's,
to run in parallel rather than compete for the same account's H100
priority slot. First submission (job 59570494) predated `jliu0290`
pulling `main` in the shared clone; caught before it started (still
queued, ~24-25hr real scheduler estimate at the time) and cancelled,
though a check afterward found the shared clone already had the real
fix present (`git log` confirmed `HEAD` at `1c1d003`), so this was
precautionary rather than a real bug avoided. Hit git's "dubious
ownership" safety check running `git status`/`git pull` as a non-owner
in `mvar0010`'s clone; resolved per-teammate via `git config --global
--add safe.directory <path>`, one-time, does not affect the clone
owner or other teammates. Resubmission (job 59570693) was accidentally
submitted from `mvar0010`'s own login session instead of `jliu0290`'s,
landing it under the wrong account/quota and defeating the point of
running it in parallel; cancelled once caught. Real, correctly-attributed
job is 59570724, submitted by `jliu0290` under their own account,
`VLLM_PORT=8005` (distinct from the 20B diagnostic's 8003, avoiding a
repeat of the earlier port-collision incident), same 48000 starting
budget and `MAX_NUM_SEQS=16` (more conservative than 20B's 32, since
120B has not been tested at all and has roughly double the active
params). Not yet started as of this writing (also queued behind `m3h`
priority contention). No completed data yet for this model in the
current batch.

Shared clone (`/fs04/scratch2/al49/kg-testing/testgeneval_mvar0010_main`)
used for teammate submissions rather than separate clones, since this is
submit-only (no one but mvar0010 commits into it) -- confirmed safe,
different from the commit-conflict gotcha in `CLAUDE.md`. Directory-level
ACL grants added for `jliu0290`, `wlee0060`, `wtho0016` (`rwx`, plus
`default:` entries for inherited access on new files). Teammates need
`git config --global --add safe.directory <path>` once in their own
global git config to clear git's "dubious ownership" check on a
directory they do not own; this is a one-time, per-teammate, local
config change, not a repo-wide setting.

**L40S production batch status** (12 jobs, `mvar0010`'s account, matches
the confirmed-safe `MAX_NUM_SEQS` values from the concurrency calibration
rows above where available). First submission attempt (jobs 59566758
through 59566770) had two real problems: the `kg_only` jobs hit the
`KG_PROMPTS_PATH` default bug described above (jobs 59566760/59566761,
both `Meta-Llama-3.1-8B-Instruct kg_only`, `FAILED` in ~75-80s each,
confirmed via `sacct`), and cancelling/resubmitting individual jobs
left the batch in an inconsistent state (some jobs correctly fixed,
others still carrying the broken default, one Qwen3-4B `kg_only` pair
never resubmitted at all). Cancelled the entire batch (`scancel
59566758 59566759 59566760 59566761 59566762 59566763 59566764
59566765 59566767 59566768 59566769 59566770`, 12 real job IDs; note
the sequence skips 59566766, which was never one of ours -- SLURM
assigned it to an unrelated job from someone else on the shared
cluster in the gap between this batch's two separate submission
commands, not a missing or forgotten job on our end) and resubmitted
clean,
with `KG_PROMPTS_PATH=kg_prompts_depth2.json` passed explicitly on every
job this time: jobs 59566867 through 59566878, at the same
`MAX_NUM_SEQS` values used in the earlier concurrency calibration rows
above (24 for Llama, 32 for Qwen3-Coder-30B, 8 for Qwen3-4B). **These
values turned out to be stale and genuinely unsafe**, discovered real,
not suspected: after ~4 hours, `Meta-Llama-3.1-8B-Instruct instruct`
t=0.2 (59566867) showed sustained 92-99.5% GPU KV cache usage with
frequent `Retrying request to /chat/completions` lines (vLLM preemption
under memory pressure), and its own tqdm progress bar reported a real
extrapolated ETA of `64:12:42` against an `8:00:00` time limit -- 396/1190
instances done at that point, mathematically certain to be killed by
the time limit rather than complete. The denominator (1190, not the
full 1210) was noticed at the time but not investigated until after the
crisis response; confirmed benign once checked: job 59566758 (the first,
earlier-cancelled `Meta-Llama-3.1-8B-Instruct instruct` t=0.2 attempt,
cancelled during the `KG_PROMPTS_PATH` batch cleanup above) had already
written exactly 20 real completions to the shared output file before
being cancelled (`grep -c "output_tokens=" slurm-59566758.out` returned
20), and `existing_ids` correctly picked those up and filtered them out
of 59566867's working set, 1210 minus 20 equals 1190. Confirmed real
resume behavior working as designed, not a bug -- but a reminder that a
cancelled job's partial output is not automatically discarded, later
resubmissions against the same output file inherit whatever it already
wrote, which is exactly the intended behavior for a genuine resume but
worth being aware of when reasoning about a fresh run's real progress
numbers. Root cause: the
earlier calibration
for these three models was done under the old, broken 4096-token
`OUTPUT_LIMITS` default; the real fix raises that to 8000 tokens (see
above), roughly doubling the KV cache each in-flight sequence holds for
its full duration, so a `MAX_NUM_SEQS` value safe at 4096 is not safe at
8000 without retesting. Cancelled all 12 jobs (`scancel 59566867
59566868 59566869 59566870 59566871 59566872 59566873 59566874 59566875
59566876 59566877 59566878`) including the 8 that had not started yet,
on the reasoning that the same root cause applied to them too even
without direct evidence, since restarting them cost nothing (they had
not consumed GPU time).

Recalibrated properly with real `testgenevallite` calibration runs at
the new 8000-token cap, same discipline as the gpt-oss rows above,
before trusting any new value for production: Llama at `MAX_NUM_SEQS=12`
(job 59572150, ran ~48 minutes, n=273 samples, min=33.4% median=68.2%
p90=80.0% max=93.0%, only 1.5% of samples at/above 90%, zero retries),
Qwen3-Coder-30B at `MAX_NUM_SEQS=16` (job 59572810, `COMPLETED` cleanly
in 28:38, n=143 samples, max=94.0%, 2.1% at/above 90%, zero retries),
Qwen3-4B at `MAX_NUM_SEQS=8` unchanged (job 59572811, n=196 samples,
max=46.7%, 0% at/above 90%, clearly safe, this model was never the
problem). Real danger-zone fraction for Llama and Qwen3-Coder-30B
*decreased* as more samples accumulated over time (2.7%→1.5% and
6.1%→2.1% respectively between two checks roughly 20 minutes apart),
the opposite of the climbing-toward-crash pattern seen in the original
crisis, and max values stayed flat rather than climbing -- real evidence
these are stable, bounded values, not a spike that just had not
happened yet.

Resubmitted the full 12-job production batch at the recalibrated values
(jobs 59573898-59573909, same `instruct`/`kg_only` x t=0.2(pass@1)/
t=0.8(pass@5, NUM_SAMPLES=5) x model order as before): Llama at
`MAX_NUM_SEQS=12`/`MAX_CONCURRENCY=12`, Qwen3-Coder-30B at 16/16,
Qwen3-4B at 8/8 (unchanged). Time limits also raised over the original
submission (`--time=12:00:00` for pass@1, `36:00:00` for pass@5, up from
8/24 hours), since lower concurrency likely means slower real wall-clock
throughput than the stale calibration assumed, plus buffer given the
earlier real pass@5 timeout crisis this project already hit once before
(see the "PostprocessingError" / timeout sections elsewhere in this
document).

**A second, real problem showed up once these actually started running,
this time not an infrastructure bug but a real limitation of the
sampling config itself.** All three models hit the same thing at
temperature 0.2 (pass@1): a meaningful chunk of real completions land
near the 8000 token cap again, but this time it is genuine repetition
loop garbage, the model getting stuck repeating the same text over and
over, not legitimate content that needed more room. Checked by hand
across all three models and confirmed real: Llama at 59.5 percent of
its instruct completions (job 59573898, cancelled once found), Qwen3-4B
at 40 to 45 percent across both arms (jobs 59573906/59573908, cancelled),
Qwen3-Coder-30B at around 11 percent (job 59573902, cancelled). Every
one manually checked was real repeated text, not truncation, confirmed
via the same repetition heuristic used elsewhere in this document.
Notably this does not show up at temperature 0.8 at all, every real
sample checked there across all three models has been clean.

This turns out to be a real, already documented problem with running
these models close to greedy decoding, not something specific to this
pipeline. Meta's own community discussion for Llama 3.1 8B reports the
same failure independently. Qwen's own model card is even more direct
about it, they explicitly say not to use greedy decoding since it can
cause endless repetitions, and recommend temperature 0.6 to 0.7 instead
of anything close to 0.2. Full writeup, including the external sources
and the real per model data, is in testgeneval#43.

Given this now shows up in every model actually tested, cancelled every
temperature 0.2 job across all three models (Llama's four, Qwen3-4B's
two, Qwen3-Coder-30B's two) rather than keep producing more data known
to be affected. Left the temperature 0.8 jobs running for Qwen3-4B and
Qwen3-Coder-30B (59573907, 59573909, 59573903, 59573905) since those
have been clean every time checked. This needs a real decision from the
team before any temperature 0.2 job gets resubmitted, whether that
means raising the temperature, adding a penalty, or accepting the
current rate as a disclosed limitation, see testgeneval#43 for the
three options laid out. Update this section with real completion
counts once that decision is made and pass@1 jobs resume.

**Llama-4-Scout-17B-16E-Instruct**, the one shortlist model with no
coverage at all in this batch until now (its only existing data,
documented in the Large tier table above, is from an earlier `t=0`
pass@1-only run, not this batch's `t=0.2`/`t=0.8` scheme). A third
teammate, `wtho0016`, submitted its 4 jobs (59571014-59571017,
`instruct`/`kg_only` x t=0.2/t=0.8) on their own M3 account/GPU quota,
using the shared clone. `TENSOR_PARALLEL_SIZE=4`, `--gres=gpu:H100:4`,
`MAX_NUM_SEQS=8`/`MAX_CONCURRENCY=8` (uncalibrated for this model at
this batch's temperatures, chosen conservative same as Qwen3-4B), 4
distinct `VLLM_PORT` values (8006-8009) to avoid any repeat of the
earlier port-collision incident. All 4 landed on different `m3h` nodes
per `squeue`'s scheduled-node output, so no collision risk between them
regardless. None have started yet as of this writing; real scheduler
estimates at submission time ranged ~50-56 hours out (`--start`),
similar order of magnitude to the other `m3h` jobs' waits, real
priority contention on that partition, not something fixable from this
side.

**Real per-account H100 queue estimates** (via `squeue --start`, useful
context for anyone deciding whether to wait or route more submissions
elsewhere): gpt-oss-20B (59566582, mvar0010) ~17 hours out as of this
writing, notably shorter than gpt-oss-120B (59570724, jliu0290) at ~48
hours and Llama-4-Scout (59571014-59571017, wtho0016) at ~50-56 hours,
despite the latter two being submitted later in absolute time --
`m3h` queue position depends on each account's own priority/fairshare,
not submission order or job count, so do not assume a later submitter
will simply queue behind an earlier one from a different account.

**A real config mismatch cost Qwen3-Coder-30B-A3B-Instruct 117 real
instances on its first completed production job.** Job 59573903
(instruct, temperature 0.8) finished cleanly, exit code 0, in a real
16 hour run, but only wrote 1093 of 1210 instances. 261 real
`BadRequestError`s in the log, all the same real cause: this model's
`MODEL_LIMITS` entry (added in testgeneval#41's fix) says 262144
tokens, its real published native context, but `m3_run_inference.slurm`
launches vLLM with `--max-model-len "$MAX_MODEL_LEN"`, which defaults
to 32768 and was never overridden on any of this project's real
Qwen3-Coder-30B submissions. So `run_api.py`'s own client-side prompt
truncation logic thought it had room it never actually had, sent
oversized prompts straight to the server, and the server rejected them
outright, `process_instance` discards a whole instance on any
exception, so these 117 instances got nothing written at all, not
partial data.

The `MAX_MODEL_LEN=32768` default itself is not a bug, it is a real,
deliberate choice, the script's own comment explains it exists because
Qwen3-4B-Instruct-2507 genuinely needed 36 GiB at its native 262144
context when only 31.62 GiB was available, a real earlier OOM. The bug
is that `MODEL_LIMITS`'s per-model claims and this script's own safety
default were never checked against each other. Confirmed real via
direct log checks, not assumed: Qwen3-Coder-30B is affected (261 real
errors), Qwen3-4B-Instruct-2507, Meta-Llama-3.1-8B-Instruct, and
gpt-oss-20B all show zero hits of this error across real, substantial
samples, even though the dataset's own real max prompt (about 43000
tokens for `kg_only`, measured earlier) exceeds 32768 for all of them
too, likely because different models' own tokenizers count the same
text differently, not because Qwen3-Coder-30B's real needs are uniquely
large. gpt-oss-120B and Llama-4-Scout have not run at real scale yet,
so absence of the error in their logs so far is not yet evidence either
way for them. Full writeup in testgeneval#44.

Real GPU memory is tight on this model's 2x L40S setup: vLLM's own
startup log shows about 28.8 GiB of each 44.39 GiB GPU already used for
weights, leaving roughly 11.5 to 14.4 GiB for KV cache at the current
`MAX_MODEL_LEN=32768`/`MAX_NUM_SEQS=16`. A rough proportional estimate
puts the real KV cache need at `MAX_MODEL_LEN=65536` around 23.6 GiB,
over that real headroom at the current concurrency, a genuine risk of
reintroducing the same class of OOM the current default protects
against, just for this model. Real calibration test running (job
59590979, `MAX_MODEL_LEN=65536` with `MAX_NUM_SEQS` lowered to 8 to
compensate for the larger real per-sequence KV cache need) before
trusting a higher value for production. The already-completed job's 117
lost instances still need a real decision, regenerate them specifically
once the fix is confirmed, or accept the loss as a disclosed gap in
that run.

## Why context-overflow losses happen

The `instruct` arm's prompt shows the model the whole source file as flat
text. Smaller-context models can't fit `instruct`'s largest files (some
source files in this dataset are large) within their real max context,
even with a small requested output budget. This is a genuine, deterministic
model limitation, not a transient failure: `run_api.py`'s `@retry` burns
its retry budget against the same guaranteed-to-fail request every time,
then logs `Failed, skipping...` and the instance is silently absent from
the output file (no `id` written, no error record kept elsewhere).

Confirmed real via `grep -c "Failed, skipping" slurm-<jobid>.out` per shard,
summed across shards, matching each model's exact (1210 - completed) gap.

Whether `kg_only`'s more compact per-function context view hits the same
ceiling as often for these models is an open, real comparison worth making
once `kg_only` runs exist -- may itself be a meaningful finding (KG framing
could structurally avoid this failure mode more often than a flat-file dump
does for the same context-limited models).

## Two different fairness questions, two different denominators

This dataset actually supports two genuinely different comparisons, and
they need different handling of skipped or missing instances. Conflating
them risks either an unfair cross-model comparison or quietly burying a
real result.

**"Which model is inherently more capable at test generation?"** Here,
context-window ceiling is part of the capability being measured. A
smaller-context model failing on large files is a real limitation, the
same one it would have in actual use. Pass rate computed over each
model's own completed subset is legitimate under this framing, but the
denominator (completed out of 1210) needs to be reported alongside the
rate every time. "StarCoder2 passed 70% of the 832 instances it could
attempt" is not the same claim as "StarCoder2 passed 70% of the
benchmark," and a bare percentage collapses that distinction.

**"Does KG-augmented context outperform flat-file context for test
generation, controlling for model?"** This is the actual research
question (see docs/EXPERIMENT_PLAN.md), and fairness is
stricter here. What matters is comparing instruct vs kg_only on the same
instance set, per model. If instruct's flat-file prompt overflows a
model's context on instance X but kg_only's more compact per-function
view doesn't, that isn't noise to average away. It's potentially the
actual finding: KG framing structurally avoiding context-overflow
failures more often than a flat-file dump does, for the same
context-limited model. Silently dropping mismatched instances from both
arms just to force equal denominators would bury a result worth
reporting.

**Where it genuinely gets unfair** is comparing models using each one's
own denominator as if it were the same test. Reporting Model A
(1210/1210) and Model B (832/1210) as bare "pass@1 = X%" without the
denominator implicitly compares performance on different, non-random
instance subsets. Model B's subset skews toward shorter, simpler files,
which can inflate its apparent rate relative to what it would score on
the full set. The fix isn't excluding Model B from the comparison, it's
never reporting a bare percentage without its denominator, and also
reporting pass rate over the intersection of instances every compared
model actually completed, as a second, strictly apples-to-apples number
alongside the per-model number that includes its own context limits.

## "PostprocessingError" can mask a real generation failure

`swebench_docker/swebench_utils.py`'s `classify_error()` is the function
that turns raw test output into the short label written to `test_error`
in results JSON. It has a catch-all branch: if the captured output has no
recognizable error text and the literal word "test" doesn't appear
anywhere in it, it labels the instance `PostprocessingError`, implying
something went wrong parsing the model's output rather than in the
generated test itself.

Diagnosed for real on `django__django-11179-15747` under
Qwen2.5-Coder-7B-Instruct (`instruct` arm): the generated test file
imports `django.test.TestCase` but the class itself subclasses plain
`unittest.TestCase`. Django's test database only gets created and wired
up automatically for `django.test.TestCase`/`TransactionTestCase`; plain
`unittest.TestCase` skips that setup entirely. The generated test then
calls `self.MyModel.objects.create(...)` against a database connection
that was never configured, and the real underlying exception is
`django.core.exceptions.ImproperlyConfigured: settings.DATABASES is
improperly configured. Please supply the NAME value.` That exception
doesn't match any of `classify_error()`'s known patterns, so it falls
through to the generic label.

Confirmed by reproducing directly in the real `kdjain/swe-bench-django_
django-testbed:3.0` container: running the gold test module with the
same `coverage run ./tests/runtests.py` command works cleanly (41/41
passed), so this isn't a testbed or harness bug. It's a real model
output defect (wrong base class for a Django ORM test), not noise to
filter out. Across the one results file checked (Qwen2.5-Coder-7B-
Instruct, `instruct` arm, 134 instances), 46 hit `PostprocessingError`,
21 in astropy and 25 in django, concentrated in those two repos only.
The astropy side hasn't been diagnosed against a matching model+instance
log yet, so it may or may not share this same root cause.

Not fixing `classify_error()` right now since evaluation runs are
actively in flight elsewhere (M3 inference plus at least one teammate
running `run_evaluation.py` locally); changing the classification
mid-run would mean the same underlying error gets labelled differently
depending on when an instance happened to be evaluated, which is its own
consistency problem when comparing results later. Worth fixing between
runs: surface the real exception name (e.g. `ImproperlyConfigured`)
instead of the generic label, so this kind of failure doesn't need
re-diagnosing from scratch next time.

## Evaluation can now run directly on M3, not just locally

Confirmed 2026-08-24: a real evaluation run through the new Apptainer
backend (`run_evaluation.py --backend apptainer`, see testgeneval#2)
produces identical results to the same instance run through Docker.
Same instance (`astropy__astropy-13579-15669`), same numbers: All Tests
Passed, `CoverageLOG` 35.978%, `FunctionCoverageLOG` 4.762%,
`MutationLOG` 12.37%, `FunctionMutationLOG` 0.0%. Not just "it ran
without crashing", the actual scientific output matches, including
`FunctionMutationLOG`, the metric behind the cosmic-ray subprocess fix
in testgeneval#27/#28.

`FunctionCoverageLOG`/`FunctionMutationLOG` are this fork's own addition
on top of the original TestGenEval benchmark (see
`docs/EXPERIMENT_PLAN.md`'s Stage 5), scoped to just the lines of the
patch's target function rather than the whole file. They're the intended
primary comparison metric for RQ2/RQ3, since `kg_only` can structurally
only generate tests for the function it was shown while `instruct` sees
the whole file and could pick up incidental coverage or mutation kills
elsewhere. Whole-file `CoverageLOG`/`MutationLOG` are the original
benchmark's metrics, kept as secondary/contextual numbers. Worth noting
for this one instance: the function-scoped numbers are much lower than
the whole-file ones (4.76% vs 35.98% coverage, 0% vs 12.37% mutation
score), which is structurally plausible on a single instance (the
generated test may exercise the file broadly without deeply exercising
the target function's own lines) but shouldn't be read as a trend from
one data point.

Two real gotchas hit getting this working, both worth knowing before
trying it yourself:
- `load_dataset`'s cache defaults to `$HOME`, not scratch space, and
  `$HOME` has its own small, easy to exhaust quota that `lfs quota`
  can't even inspect (it's not a Lustre filesystem). Set
  `HF_DATASETS_CACHE` to somewhere under scratch, same idea as the
  existing `HF_HOME` fix for model weights.
- A shared git checkout with commits from more than one account can end
  up with `.git` objects owned by different people, breaking `git pull`
  for everyone but each object's original owner, and it's not fixable
  with `setfacl` (would need per-object grants from each owner). Clone
  fresh into your own directory instead of continuing to share one
  checkout across accounts.

Only `astropy_astropy_5.0.sif` exists on M3 so far. The remaining ~142
testbed `.sif` files (one per repo/version combination) still need
building and transferring before this can replace the local Docker
evaluation path at full scale, see testgeneval#2 for the build process
(has to happen off M3, M3 needs `sudo` for `apptainer build`/`pull`
from Docker Hub, which regular accounts don't have).
