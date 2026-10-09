# Pre-registration (FROZEN v1, 2026-10-04)

Status: **frozen** on 2026-10-04, after pilots P1, P1b, P2, P2 v1b, the full teacher run and the SFT smoke test, and before any main-run SFT or GRPO job. The md5 of this file and a read-only copy (`PREREG_frozen_2026-10-04.md`) are recorded in README. Later changes go only into the dated Addenda at the end; this text stays unchanged.

## 1. Question


ActObs (arXiv 2609.20715) supervises environment-observation tokens during SFT. After GRPO this gives higher pass@k, more retained entropy and less policy movement. The authors attribute the gain to learning action consequences, i.e. a world model.

Their loss (Eq. 1) is one pooled per-token mean over action and observation tokens. At λ=1 this shrinks the action term's share of every update to |A|/(|A|+|O|), about 0.55 in their corpus. That is a smaller, "less over-trained" move of the action policy. SFT-to-RL plasticity work (2606.09932) predicts that alone could help RL.

The paper never matched action-update magnitude, and none of its controls (wild logs, shuffled, salient) were taken through RL.

**Question:** in multi-turn text-to-SQL, does observation supervision help GRPO through world-model learning, or through diluting action learning?

## 2. Setting (fixed for every arm)

**Protocol v1b** (chosen in the pilots; see the Pilot log). It applies to teacher, SFT data, GRPO and evaluation.
- SkyRL-SQL `text2sql` in recipe mode: a single assistant message, with observations appended as raw text. SkyRL is pinned at 99824f0 and unmodified.
- `enable_thinking=false` everywhere.
- The `verify` system prompt (data/v1/*, system sha256 prefix de61669595464fbc).
- `text2sql` is bound to `agenticrl.sql_lenient.LenientSQLEnv`, which differs from SkyRL's env only by dropping its two `<think>` format rules.
- Max 6 turns, max 1024 generated tokens per turn, max input 8192 tokens.
- Node-local DB path `$LOCAL/omnisql/data`; `PYTHONHASHSEED=0`.

**Models**
- Student: Qwen/Qwen3-4B, the hybrid checkpoint (snapshot 1cfa9a72).
- Teacher: Qwen/Qwen3-32B (snapshot 9216db57), sampled at T=0.6, top-p 0.95, top-k 20.

**SFT data** (teacher job 1505395; built by scripts/build_sft_data.py)
- 6,294 Spider-train questions × 4 samples gave 25,176 trajectories.
- 24,043 were kept: unfinished (`length`) ones (44) and format failures (1,089) were dropped. Of those kept, 12,171 are correct.
- Tokens: action 7,189,000; observation payload 2,451,405; observation wrapper 774,680. The observation share is 0.310, so ActObs's λ=1 action-share factor is about 0.69.
- `data/sft/train.jsonl`, sha256 040a1f17f6b60fd4…

**Held-out trajectories** (job 1505397; used for measurements and arm C's selection only)
- 592 questions on 12 held-out DBs × 2 samples gave 1,149 kept trajectories.
- `data/sft/heldout.jsonl`, sha256 ad976d5fce8a1dee…

**Dilution corpus** (arm D): HotpotQA Wikipedia paragraphs, 30.0M tokens. `data/sft/dilution.jsonl`, sha256 dae919ab9f51bca1…

**SFT recipe** (scripts/sft_train.py, 4-GPU DDP)
- Full fine-tuning with fp32 master weights and bf16 autocast; AdamW (0.9, 0.95), weight decay 0.
- LR 1e-5, cosine decay with 3% warmup, batch 64 trajectories, 1 epoch (375 steps), max length 12288.
- Identical example order in every arm (seed 0). Per-example loss as in Eq. 1, averaged over the batch.

**GRPO** (scripts/grpo_args.sh via agenticrl/main_grpo.py; the same in every arm)
- Tasks: data/v1/rl_train_nonempty.parquet, i.e. the 363 SkyRL-SQL-653 tasks whose gold result is non-empty (sha256 prefix 1496171565ce26bf).
- Optimization: lr 1e-6 constant, no KL, clip 0.2, token-mean loss, grad-norm 0.5, token-level TIS with cap 2.0.
- Batch 32 tasks × 16 samples, 135 steps, 4 GPUs, flash-attn with padding removal.
- Rollouts at T=0.6, top-p 0.95. Seeds {1, 2, 3}.

**Evaluation** (scripts/rollout.py through scripts/jobs/eval.sge, then scripts/passk.py)
- Spider-dev and BIRD-dev (data/v1), 16 samples per question, T=0.6, top-p 0.95, top-k 20.
- Unbiased pass@k for k ∈ {1, 2, 4, 8, 16}.
- The **primary eval set** drops questions whose gold returns no rows or fails to execute (results/gold_flags). That removes 51 Spider-dev and 2 BIRD-dev questions, leaving 983 + 1,532.

## 3. Arms (SFT starting points; only the loss differs unless stated)

| Arm | SFT loss | Purpose |
|---|---|---|
| A | λ=0 (ActionSFT) | Baseline |
| B | λ=1, pooled Eq. 1, O = payload + wrapper (ActObs) | Replication |
| C | λ=0, learning rate chosen so that held-out action-token KL(π_C ‖ π_0) matches B (rule below) | Matched action movement, no observation learning |
| D | λ=1, with O replaced by |O|-matched unrelated Wikipedia tokens in a separate sequence, pooled like Eq. 1 | Dilution without environment prediction |
| B′ (deferred) | Action mean + observation mean (`--norm separate`) | Observation learning without action dilution |

**B′ is deferred** (decided 2026-10-04). It is not part of the first-round confirmatory analysis. If it is run later, it uses the same recipe, its result is added as a pre-specified secondary arm, and the date it was run is noted.

**Rule for C.**
1. Train ActionSFT at LR = 1e-5 × r for r ∈ {0.85, 0.70, 0.55, 0.40}.
2. Compute the token-weighted mean action-token KL to the base model on the held-out trajectories (scripts/measure.py, first 300 trajectories in seed-0 order).
3. Pick the r whose KL is closest to B's.
4. If none is within ±25%, add one more point at 0.73 × the smallest r, then take the closest.
5. Report every candidate.

**First round.**
- SFT: A, B, D, and the 4 C candidates (7 SFT runs).
- Measurement, then selection of C.
- GRPO: A, B, C, D × seeds {1, 2, 3} (12 runs).
- Evaluation of the base model, the 4 SFT arms and the 12 GRPO runs.

## 4. Outcomes and hypotheses

**Primary outcome**
- Post-GRPO pass@16, pooled over the primary eval set (Spider-dev and BIRD-dev questions together).
- Averaged over seeds, and also reported per seed.
- pass@1 is reported alongside it with the same procedure.
- Paired differences between arms use a 95% bootstrap CI over questions (10k resamples, questions resampled jointly).

**Hypotheses**
- **H-rep:** pass@16(B) − pass@16(A) > 0. The pooled CI must exclude 0, with the same sign in at least 2 of 3 seeds.
- **H-WM (world model):**
  - B − C > 0 and B − D > 0, with both CIs excluding 0.
  - Across arms × seeds, the post-RL gain over A increases with the SFT-stage improvement in held-out observation-payload NLL. Report Spearman ρ.
- **H-DIL (dilution / plasticity):** C (or D) recovers at least two thirds of B's gain. The recovery ratio is R_X = (X − A)/(B − A), reported with a bootstrap CI.
- **Mixed outcome:** R_C and R_D are reported regardless of significance. Partial recovery is a valid result.

**If H-rep fails:**
- Report "the ActObs RL benefit does not replicate on text-to-SQL", with the CI.
- C and D are still run, because they test whether dilution alone changes RL behaviour (entropy, movement).

## 5. Secondary measurements (all arms, descriptive)

**SFT stage**
- pass@k before RL.
- Held-out teacher-forced CE per class (action, payload, wrapper).
- Action-token KL to the base model, action entropy, and top-1 − top-2 logit margin.
- Gradient cosine and norm ratio between action and observation losses, summed over tokens, on 256 held-out trajectories in 4 partitions.

**During and after GRPO**
- Reward curve and response length.
- Policy entropy during GRPO.
- Policy movement: action-token KL(π_RL ‖ π_SFT) on 200 fixed held-out traces.
- Fraction of groups with zero reward variance.

**Data descriptives**
- Observation-token share of the SFT corpus: 0.310.
- Fraction of tasks whose gold SQL returns an empty result: 44% of the original SkyRL-SQL-653, which is why the RL tasks are filtered.

## 6. Fixed compute settings (from the pilots)

| Stage | Settings | Pilot evidence |
|---|---|---|
| GRPO | 32 × 16 × 135 steps on 4 GPUs | P2 v1b: 423–433 s per step on 2 GPUs; about 4 min per step on 4 GPUs is extrapolated, not measured |
| SFT | 1 epoch on 4 GPUs | Smoke test, arm D, 2 GPUs: 3 steps without error |

If node 004 offers only 2 GPUs, a run may use 2 GPUs with identical global batch sizes; the GPU count is logged. Arms of the same seed should preferably use the same GPU count.

## 7. Analysis rules

- No exclusion of questions, seeds or runs based on outcomes.
- A failed run is rerun with the same seed once, and the failure is logged.
- Every arm is reported, including nulls.
- Arm C's LR selection uses held-out KL only, never pass@k.
- Deviations go into a dated Addendum below.
- Analyses not listed here are allowed but labelled exploratory.

## Pilot log (before freezing)

**P1 (2026-10-03, job 1501377).** Teacher Qwen3-32B, thinking on, original SkyRL-SQL prompt, 50 pilot questions × 2 samples.

Results:
- Accuracy 0.44; format failures (reward −1) 0.19; mean 1.37 turns.
- **All 81 well-formed trajectories are single-turn**: a direct `<solution>` with no `<sql>` call. The observation-token share of the kept data is therefore 0.0%.
- Every trajectory that used the tool failed the format check. After an observation, the thinking-mode model continues with `</think>` (it believes it is still inside its reasoning block) or writes no tool tag, so the env rejects the action.
- Base Qwen3-4B in the GRPO smoke (P2, thinking on) shows the same pattern: mean raw reward −0.70 on step 1.

**Consequence.** With thinking on and the original prompt, SFT data contain no observations and the study is void.

**P1b** (jobs 1501386/1501387) tests the allowed adjustment (ii), `enable_thinking=false`, under two prompts:
- `orig`, unchanged;
- `verify`, the original plus one Format line: "Before giving the final solution, you MUST execute at least one SQL query with the SQL tool (for example, your candidate query) and check its result in the observation."

The chosen protocol (thinking setting × prompt) will apply identically to teacher, SFT data, GRPO (`generator.chat_template_kwargs`) and evaluation. It is chosen on format validity and observation share only, never on accuracy differences between arms; no arm exists yet.

**P1b results.** Thinking off, 100 trajectories per prompt.

Under SkyRL's strict format rule, both prompts fail 100%: the model never writes `<think>` tags (0/100). It nevertheless uses the tool normally. Re-scored offline (scripts/rescore.py) with only the `<think>` rules removed:

| prompt | lenient accuracy | format fail | multi-turn among kept | observation-token share (kept) | mean action / observation tokens per trajectory |
|---|---|---|---|---|---|
| orig (thinking on, P1, for reference) | 0.52 | 0.00 | 19/100 | 0.014 | 1094 / 15 |
| orig | 0.52 | 0.06 | 81/94 | 0.216 | 305 / 84 |
| verify | 0.49 | 0.03 | 96/97 | 0.269 | 299 / 110 |

**Decision: protocol v1.** It applies to every stage: teacher, SFT data, GRPO and evaluation.
- Thinking off (`enable_thinking=false`).
- The `verify` prompt (data/v1/*, system sha256 prefix de61669595464fbc).
- The `text2sql` id bound to `agenticrl.sql_lenient.LenientSQLEnv`. SkyRL's env is unchanged except that format rules (3)–(4) about `<think>` are dropped; observation text, turn limits and the execution-match reward are identical.
- Chosen on format validity and observation share. The accuracy difference (0.52 vs 0.49, n = 100) is within noise and played no role.

**Consequences for the design.**
- The expected observation share is about 0.27, so ActObs's λ=1 dilution factor is about 0.73 here (paper: about 0.55).
- The candidate grid for arm C becomes r ∈ {0.85, 0.70, 0.55, 0.40}; the selection rule in §3 is unchanged.

**GRPO additions** (identical across arms):
- Token-level TIS with cap 2.0. Step 1 of P2 logged a maximum rollout-vs-trainer log-prob difference of 60.5.
- `generator.chat_template_kwargs={enable_thinking: false}`. The first syntax tried, without a space, parsed to the wrong key; the CPU dry run caught it.

**P2 (thinking on, before the switch).** Step 1 took 1234 s:
- generate 150 s;
- forward 159 s;
- train about 925 s.

Mean trajectory length was 2467 tokens, and the batch was padded to 11,402 tokens. flash-attn is unavailable on these Blackwell GPUs, so padding cannot be removed. The job was stopped after step 1, and P2 is repeated under protocol v1.

**Gold-result audit (CPU, scripts/gold_stats.py, before any arm exists).**

The reward is frozenset equality of execution results. A gold query that returns no rows therefore gives reward 1 to any well-formed query that also returns no rows.

| Set | Questions | Gold returns no rows | Note |
|---|---|---|---|
| SkyRL-SQL-653 | 653 | 290 (44.4%) | SynSQL rows 283/540 (only 18 use 'now'/current-date functions; the rest are sparse synthetic DBs); Spider rows 7/113 |
| Spider-dev | 1,034 | 49 (4.7%) | 2 more gold queries fail |
| BIRD-dev | 1,534 | see results/gold_stats.json | |

BIRD-dev result: 0 gold queries return no rows; 2 fail (results/gold_stats.json, results/gold_flags/).

**P2 v1 (job 1501470).** Protocol v1, base Qwen3-4B, 32 tasks × 16 samples, 2 steps. It ran end to end (RC 0) and exported the HF model.

| Measure | Step 1 | Step 2 |
|---|---|---|
| Step time | 3354 s | 3168 s |
| Generate | — | 188 s |
| Forward | — | 419 s |
| Policy train | — | 2545 s |
| Mean raw reward | −0.13 | −0.13 |
| mean_positive_reward | 0.26 | 0.28 |
| End-of-run eval, Spider-dev pass@1 | 0.55 | |
| Mean rollout–train log-prob difference | 0.013–0.019 | |

- Mean response length was 1220 tokens, but the batch was padded to 20,350.
- A few degenerate trajectories hit 3000 tokens per turn over several turns, which set the batch maximum.
- At this rate one 135-step run would take about 5 days.

**v1b changes (adopted 2026-10-04).** They apply to every stage: teacher, GRPO and evaluation.
- `trainer.flash_attn=true` and `trainer.remove_microbatch_padding=true`. The SkyRL uv environment ships flash-attn 2.8.3; an earlier SkyRL setup had disabled it for an undocumented reason, so it is re-tested here on Blackwell.
- `max_generate_length` 3000 → 1024 per turn.
- `max_input_length` 29000 → 8192.
- Batch stays at 32 tasks × 16 samples.

Normal turns in P1b had a median of about 100 tokens; only repetitive degenerate turns reached the cap.

**P2 v1b (job 1503767, 2 GPUs).** Node 004 had only 2 free GPUs because another user's QLOGIN held the other two. Settings were v1b, base Qwen3-4B, 32 × 16, 2 steps. The run finished with RC 0 and exported the model. flash-attn works on these Blackwell GPUs.

| Measure | v1b, 2 GPUs (step 1 / step 2) | v1, 4 GPUs (step 2) |
|---|---|---|
| Step time | 423 s / 433 s | 3168 s |
| Generate | 112 s / 113 s | 188 s |
| Forward | 63 s / 63 s | 419 s |
| Policy train | 234 s / 237 s | 2545 s |
| Batch maximum length | 9,019 / 9,007 | 20,350 |
| Mean response length | 1,075 / 1,115 | — |
| Mean rollout–train log-prob difference | 0.017 | 0.015 |
| Maximum log-prob difference | 57 / 62 | 55 |
| TIS clip ratio | 0 | — |
| Spider-dev pass@1 at end | 0.551 | 0.550 |

- Behaviour is unchanged by flash-attn and the tighter limits; the run is about 7× faster.
- Extrapolation to 4 GPUs (not yet measured): about 4 min per step, so about 9 h for 135 steps.

**Decisions.**
- RL tasks = the 363 SkyRL-SQL-653 tasks with a non-empty gold result: 257 SynSQL and 106 Spider (data/v1/rl_train_nonempty.parquet, sha256 prefix 1496171565ce26bf). Otherwise about 44% of tasks would reward "return nothing" regardless of the question.
- Primary evaluation counts only questions whose gold result is non-empty and executes. Full-set numbers are secondary.

## Addenda

### Addendum 1 (2026-10-05): round-1 SFT failed; memory-only fix to the SFT implementation

**What happened.** None of the 7 round-1 SFT runs (jobs 1506498–1506510) finished.
- B and D ran out of GPU memory at step 11; A, C085, C070 and C055 ran out at step 218. Every arm uses the same example order, so each failed on the same long trajectory.
- The cause was the DDP memory footprint: fp32 parameters, gradients and AdamW states take about 64 GB per GPU. On top of that, the fp32 `[n_targets, 151936]` logits of an 8k-token trajectory need about 10 GB.
- The SFT smoke test (3 steps) never reached a long example. This was my oversight.
- C040 stopped at step 15 at 01:56. Its log and the logs of the 7 measurement jobs and the notifier are truncated mid-write at the same moment, which points to /groups being briefly unwritable. The cause is unconfirmed; the Qwen3-32B teacher snapshot was deleted from the shared HF cache around then, which suggests a space clean-up.
- No result from round 1 was looked at beyond these logs. No pass@k or KL was computed.

**Fix** (scripts/sft_train.py, new sha256 prefix 9b3cfcffb3b82147). Implementation only; the objective (Eq. 1 and the arm definitions) and the AdamW hyper-parameters are unchanged.
1. The LM head is computed in chunks of 2048 target positions, and the logits are recomputed in the backward pass with `torch.utils.checkpoint`. Loss and gradients equal the unchunked computation (tests/test_sft_loss.py `test_chunked_lm_head_matches_unchunked`).
2. AdamW states are sharded over ranks with `torch.distributed.optim.ZeroRedundancyOptimizer` (ZeRO-1). This is the same update as AdamW.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in scripts/jobs/sft.sge.
4. Peak GPU memory is logged on every step.

**Before rerunning.** The 64 longest trajectories are put into a single step (data/sft/worstcase64.jsonl, lengths 5,932–8,308) for arms B and D (jobs 1507456/1507457).

**Rerun.** Round 1 is rerun unchanged otherwise: same data, same order, the same 7 arms and the same C rule. The Qwen3-32B deletion does not affect the design, because the teacher data are complete and hashed.


### Addendum 2 (2026-10-05): SFT rerun finished; measurement-script fix

**SFT rerun** (jobs 1507462–1507474). All 7 arms finished: 375 steps each, models saved.

**Measurement failures.** The held-out measurements for C085, C070, C055 and C040 succeeded. Those for A, B and D, the three arms that also compute the gradient cosine, ran out of memory.
- `grad_vectors` concatenated all parameter gradients into two 4B-element fp32 vectors on the GPU (16 GB each).
- The forward statistics (KL, CE) were written only at the end, so they were lost as well, including B's KL, which the C rule needs.

**Fix** (scripts/measure.py, new sha256 prefix ec974cf73f4f3797). Same metric definitions.
- The forward statistics are written before the gradient pass.
- The action gradient is parked on the CPU parameter by parameter, and dot products and norms are accumulated per parameter.
- The LM head is chunked and recomputed, as in sft_train.py.
- Equality with the naive full-vector computation is checked in tests/test_measure_passk.py (`test_grad_stats_match_full_vectors`).

**Rerun.** The A, B and D measurements were rerun as jobs 1508921, 1508922 and 1508924. No outcome-based change was made: the C candidates' KL values were produced by the same code path, unchanged.

### Addendum 3 (2026-10-05): evaluation-fidelity check (validation only; no change to the plan)

**Concern.** Our evaluator (scripts/rollout.py) gave the base model Spider-dev pass@1 0.482 with 31% format failures. SkyRL's built-in eval in the P2 v1b smoke gave 0.551 with 21% format failures. The question was whether the evaluator diverges from SkyRL's generator.

**Check** (jobs 1509114/1509115). Spider-dev, n=4, SkyRL's sampling settings (top-k −1).

| model | pass@1 | format fail |
|---|---|---|
| base | 0.492 | 30.3% |
| smoke model after 2 GRPO steps (the model SkyRL evaluated) | 0.540 | 23.0% |
| SkyRL built-in eval, same model, n=1 | 0.551 | 21.4% |

**Conclusion.** The gap came from the two GRPO steps, which quickly reduce format failures, and not from the evaluator, which reproduces SkyRL. top-k 20 vs −1 makes no material difference. The primary evaluation settings stay as frozen.

**Pre-RL (SFT-stage) results.** pass@1 / pass@16 on all questions, 16 samples, frozen settings. The four arms are indistinguishable before RL, as in ActObs.

| SFT arm | Spider-dev | BIRD-dev |
|---|---|---|
| A | 71.4 / 84.4 | 49.7 / 73.1 |
| B | 71.7 / 84.0 | 50.1 / 73.3 |
| C (C085) | 71.3 / 84.5 | 50.2 / 73.7 |
| D | 71.9 / 84.4 | 50.3 / 73.0 |

For reference, the base model scores 48.2 / 82.7 on Spider-dev and 32.9 / 66.0 on BIRD-dev.

### Addendum 4 (2026-10-06): 11 of 12 GRPO runs failed because node 004's local disk filled; rerun with identical training settings

**What happened.** grpo_A_s1 (job 1508971) finished normally at 05:40, and its eval and policy-movement measurement completed. grpo_B_s1 failed after about 4 h, and the other 10 runs failed at start-up. The failures were "No space left on device" on node 004's 1.8 TB `/tmp`, which had reached 100%. Two things filled it:
1. **SkyRL checkpoints.** SkyRL saves a full FSDP checkpoint (with optimizer state) and an HF export at every epoch end whenever `ckpt_interval` or `hf_save_interval` is greater than 0 (trainer.py:457–462). One epoch is about 11 steps, so a 135-step run saved about 12 times; A_s1's run directory reached 856 GB.
2. **uv environments.** `uv run --isolated` left an ephemeral environment of about 18 GB per job, about 158 of them, in the uv cache's builds-v0.

The dependent eval and measurement jobs failed because their model was missing. No result of the failed runs was looked at.

**Fixes** (operational; the training objective, data, hyper-parameters and seeds are unchanged).
- `trainer.ckpt_interval=-1`: no intermediate checkpoints, so a failed run is rerun from scratch. The final HF export is kept.
- The node-local run directory is deleted when the job exits, after copying the final model and infra logs to /groups.
- GRPO jobs abort before starting if `/tmp` has less than 300 GB free.
- One persistent uv environment per node (`UV_PROJECT_ENVIRONMENT`) replaces `--isolated`.
- Eval and measurement jobs exit quietly when their model is missing, so one failure no longer produces a cascade of mails.
- Node 004's `/tmp` was cleaned to 24% used. Only this project's run directories (final models already on /groups), local SFT copies and the stale uv environments were removed; other projects' data were left untouched.

**Rerun.** The 11 runs B_s1 … D_s3 are rerun with the same seeds (PREREG §7: a failed run is rerun once with the same seed). A_s1 is kept.

### Addendum 5 (2026-10-06): GRPO and eval jobs limited to 2 GPUs on node 004, run as one serial chain

**Change.** To share node 004 with other jobs, this project uses at most 2 of its 4 GPUs.
- The rerun of Addendum 4 was cancelled twice before any result existed. The first attempt ran B_s1 on 4 GPUs; it was stopped after a few minutes. The second attempt started B_s1 and C_s1 at the same time, each on 2 GPUs, so 4 GPUs together; C_s1 was cancelled immediately.
- All remaining work now runs as one chain (scripts/submit_round1_grpo_chain.sh), in this order: B_s1 (job 1510918, 2 GPUs) → its eval → its movement measurement → C_s1 → … → D_s3.
- GRPO uses 2 FSDP ranks and 2 vLLM engines (NGPU=2). Evals use 2 shards.
- The global batch (32 tasks × 16 samples), steps, learning rate and seeds are unchanged; the GPU count only changes how the same batch is split across devices.

**Deviation.** PREREG §6 prefers that arms of the same seed use the same GPU count. A_s1 ran on 4 GPUs; every other run uses 2. The GPU count is recorded per run in results/round1_grpo_jobs.json. A_s1 vs the other seed-1 arms is therefore flagged as a possible (small) numerical difference; a rerun of A_s1 on 2 GPUs can be added if the A-vs-B contrast turns out sensitive to seed 1.

**Timing.** About 7 min per step on 2 GPUs, i.e. about 16 h per run. The remaining 11 runs plus evals take about 8 days.

### Addendum 6 (2026-10-07): re-prioritised after seed 1; new SFT-stage arms on why observation supervision leaves the action policy unchanged

**What was seen before this addendum.** Seed 1 of A and B after GRPO, primary set (results/eval/primary_pairs_AB_s1.json): B − A pass@1 is −0.07 [−0.66, 0.52] on Spider-dev and −0.94 [−1.66, −0.26] on BIRD-dev; pass@16 is +0.10 and −0.20, both CIs covering 0. The CIs resample questions only, not training seeds. RL movement (KL to the own SFT start): B 0.107, A 0.067. Behaviour after error and empty observations is the same for A and B, both before and after GRPO. C_s1 finished training and is being evaluated as planned; its result was not seen when this addendum was written.

**Decision (2026-10-07).** The C and D arms exist to explain B's gain over A. With no gain in seed 1 they lose their purpose, so the following were put on hold (`qhold`, not deleted):
- D_s1;
- C and D seeds 2–3;
- A and B seed 3.

A_s2 and B_s2 still run as pre-registered, to test whether B ≈ A holds in a second seed. This decision was made **after** seeing seed-1 results. The A_s2/B_s2 comparison itself is unchanged from §3/§4 and is unaffected. Only the C/D runs are deferred.

**New question.** Why does observation supervision not change the action policy, and what kind of observation learning would? Prior work already reports two facts we do not claim: SFT checkpoints are matched before RL, and the action and observation gradients become orthogonal within 10–20 steps (ActObs §5.1 and its "wild logs" control). The new arms below are SFT-only, use the same data order, seed and hyper-parameters as A/B, and run on 2 GPUs. The global batch of 64 is unchanged, and each step uses the same 64 examples as before.

**Diagnostics on existing checkpoints** (scripts/jobs/diag.sge, scripts/obs_breakdown.py):
- **D1:** action-token KL between arms on the held-out set (B, C085, D vs A).
- **D2:** observation CE split by outcome type (error / empty / rows / invalid) and by part (wrapper, first 3 payload tokens, rest of the first line, rest). Measured for base, A, B and D on held-out DBs and on training trajectories.
- **gradtrain:** gradient cosine and norm ratio on *training* trajectories for A and B.

**New arms.**
- **D_obs.** Control D, but the |O|-matched extra text is observation tokens from *other* training trajectories, seen out of context (data/sft/dilution_obs.jsonl). It is SQL-observation text that cannot be predicted from the current trajectory.
- **E1, auxiliary outcome prediction.** After each `</sql>` call, insert one line stating the call's outcome, derived from the real observation: "Expected result: N rows, M columns." / "no rows" / "error (<message>)" / "timeout". Supervise only this line, as an observation-class target with λ=1. The real observation tokens stay in context but are not supervised. The line is never generated at test time.
- **E2, predict-then-act.** The same line is inserted *before* the `<sql>` call. It becomes part of the action, so the query that follows conditions on it. Training is action-only SFT. E1 and E2 supervise identical lines with identical weights in the pooled loss (data/sft/variants_report.json); the only difference is whether the prediction sits in the action's computation path.

**Predictions, written before any of these results exist.**

*Mechanism (Adam normalisation and memorisation):*
1. D1: KL(B‖A) is no larger than KL(C085‖A); B and A are as close as an LR change of 15%.
2. D2:
   - On held-out DBs, B's advantage over A is concentrated in the wrapper and in the outcome-type tokens. On row values B is not much better than A.
   - On training trajectories, B is much better than A on row values as well (memorisation).
3. gradtrain: the observation/action gradient-norm ratio for B is much smaller on training trajectories than on held-out ones (5.49).
4. D_obs: its dilution text stays harder to predict than B's in-context observations, and its action KL to base falls below B's (0.363), towards D's (0.352).

*Design (E1 vs E2):*
- E1: action KL to base ≈ A's (the auxiliary target does not reach the action policy).
- E2: actions differ from A's more than B's and E1's do, in the shared e2 context.
- On unseen DBs, E1 and E2 predict the outcome line better than A and B.
- Pre-RL pass@k is reported for E1 and E2 without a directional prediction.
- The one with the larger action change (if any) will be trained with GRPO (seed 1) and compared with A_s1/A_s2.

**Chain** (results/followup_jobs.json): diagnostics → sft_Dobs → sft_E1 → sft_E2 → pre-RL eval E1, E2 → follow-up measurements → A_s2 → B_s2. The E-arm GRPO run is added after A_s2. All jobs run on node 004 with at most 2 GPUs at once.

**Results of the diagnostics and C_s1 (added 2026-10-07 13:00, after the above was written).**
- **Prediction 1 (D1): refuted as stated.** Held-out action KL to A: B 0.0049, C085 0.0010, D 0.0027. All are under 1.5% of the SFT movement from base (about 0.36). Still, B is about 5× further from A than a 15% LR change is, and about 2× further than Wikipedia dilution.
  - Follow-up added **after seeing this**: sft_Aseed1 (job 1514021), action-only SFT with data-order seed 1, as a noise floor for KL(·‖A). It goes after sft_E2, and its KL is computed in diag followup_a (resubmitted as 1514022).
- **Prediction 2 (D2): supported.** Observation CE on held-out DBs (results/measure/diag_obs_breakdown.json):
  - B is far better than A on the wrapper (0.03 vs 2.79), the start of the result header (1.02 vs 5.64), the rest of the header (0.30 vs 2.22) and error messages (0.23 vs 3.93).
  - B is not better on row values (1.92 vs 1.88).
  - On training trajectories, B's row-value CE is 0.70 vs A's 1.60 (memorisation).
  - Caveat: the "rows" category also contains the invalid-action notices, which the breakdown script does not separate.
- **Prediction 3 (gradtrain): supported.** The observation/action gradient-norm ratio on training trajectories is 1.09 for B (5.49 on held-out) and 5.87 for A (6.15 on held-out).
- **C_s1** (all questions, pass@1/pass@16): Spider-dev 80.3/88.5, BIRD-dev 58.6/77.3. RL movement (KL to own SFT start) is 0.118, close to B's 0.107; A_s1's 0.067 is the outlier. A_s1 is the only run on 4 GPUs (Addendum 5), so the A-vs-B movement difference reported above is **not** attributed to the arm until A_s2 (2 GPUs) is in.

**Follow-up results (added 2026-10-07 22:00).**
- **Noise floor.** The data-order seed alone gives KL(A_seed1‖A) = 0.0070. B's 0.0049 is below it, so the D1 "refutation" above does not stand: B's action difference from A is within data-order noise.
- **D_obs.** The out-of-context observation text stays harder than B's in-context observations: its CE falls 3.20 → 1.09, while B's falls to 0.25. Its action KL to base is 0.358, between B (0.363) and D (0.352), so it dilutes partially. This is in the direction of prediction 4, but KL to base is an insensitive metric (C040–A spans only 0.334–0.364).
- **E1 / E2, outcome-line CE on unseen DBs.** E1 0.20 and E2 0.26, vs about 4 for A and B. Both learn the prediction.
- **Action change in the shared e2 context (KL to A).** B 0.0058, E1 0.0043, E2 0.0162. Only E2 exceeds the data-order noise floor, by about 2.3×. The floor was measured on the plain held-out set, not in the e2 context.
- **Pre-RL pass@1/pass@16.**
  - E1: Spider-dev 71.1/84.4, BIRD-dev 49.7/73.5.
  - E2: Spider-dev 71.2/84.2, BIRD-dev 49.9/73.7.
  - Both are the same as A.
- **E2 predictions in its own rollouts.**
  - It writes a prediction before 99.99% of SQL calls.
  - Outcome-type accuracy is 0.88 (Spider-dev) and 0.78 (BIRD-dev), but this mostly reflects the "rows" base rate.
  - It predicts an error for only 9 of 8,321 calls that actually error, and "no rows" for 408 of 4,459 empty results.
  - When it predicts rows, row and column counts are exactly right 59–67% of the time.
  - The teacher data contain few failures (2.8% errors) and the labels are written in hindsight, so E2 learned a world model that is blind to failures.
- **Decision by the rule above:** E2 has the larger action change and goes to GRPO (seed 1, job 1516002), after A_s2 and before B_s2.

### Addendum 7 (2026-10-08): E3 "answer contract" arm, chosen after a literature review and a $0 kill test; queue re-ordered

**What was seen before this addendum.**
- *Addendum 6 follow-ups.* E2 changes actions beyond the noise floor, but its pre-RL pass@k is unchanged and its predictions are blind to failures.
- *Two verified literature reviews (about 60 papers; key ones re-checked).* They show:
  - observation losses at SFT do not improve policies, and auxiliary-loss gains during RL match content-free placebos (2607.21273);
  - predict-then-act helps only with labels grounded in the agent's own interaction and after RL (2510.15047, 2510.08558), and rewarding prediction accuracy in GRPO collapses training (2607.21273);
  - when execution is free, learning to predict one's own execution results does not improve generation (2510.02387, 2503.05703; PreAct on SQL +0);
  - what helps is knowledge the executor cannot give: a specification of the correct answer (2207.10397, 2502.01619, 2604.19825, 2304.05128), used by a policy that learns on-policy to act on mismatches (2410.02089, 2409.12917).
  - No prior work trains a model to predict the gold result's shape and act on mismatches.
- *Kill test (scripts/shape_killtest.py, results/killtest_shape_v3.json).* Oracle selection among the 16 samples of GRPO-A_s1 by the gold result's coarse shape (empty / 1 row / several rows, plus the column count), against a random pick:
  - Spider-dev: 80.5 → 82.6;
  - BIRD-dev: 57.9 → 63.6;
  - the free rule "not empty, no error" alone reaches 81.1 and 60.5.
  - Wrong answers whose exact shape differs from gold: 44% (Spider-dev) and 52% (BIRD-dev).

**Decision (2026-10-08).** The queue after A_s2 becomes: sft_E3 → pre-RL eval E3 → GRPO E3 seed 1 (+ eval, movement) → sft_E3p → pre-RL eval E3p → B_s2 → GRPO E2 seed 1.
- This deviates from Addendum 6, which put E2's GRPO next.
- E2's GRPO is moved last. The reason is the literature above (hindsight labels; one's own query outcome is redundant when execution is free), not any E2 GRPO result, since none exists.

**Arms** (scripts/build_e3.py; action-only SFT, same data order, hyper-parameters and 2 GPUs as A):
- **E3.** Each trajectory starts with "Expected answer: <coarse shape of the gold result>." After every SQL observation, the next action starts with "Check: got <observed coarse shape>; this matches / does not match the expected answer."
  - The check is computable from the model's own contract and the observation, so it is a skill, not a hindsight label.
  - The GRPO reward is unchanged (execution match only); contract accuracy is never rewarded.
- **E3p (placebo).** Same format, but each question's contract comes from another question (a seeded derangement), with checks computed against that contract.
  - It tests whether gains come from the contract's content or from the format and extra tokens.
  - E3p gets SFT and a pre-RL eval only (no GRPO by Oct 10). That is a stated limitation.

**Predictions, written before any E3/E3p result.**
1. *Contract knowledge.* On Spider-dev and BIRD-dev questions (unseen DBs), the contract E3 writes in its own rollouts matches the gold coarse shape more often than (a) always writing the most frequent dev-set shape and (b) E3p's contracts.
2. *SFT behaviour.* In E3's pre-RL rollouts, the agent submits a solution right after a "does not match" check less often than after a "matches" check.
3. *Pre-RL pass@1.* E3 ≈ A (within ±1 point on both sets). SFT alone is not expected to teach acting on mismatches.
4. *Zero-training readout.* Among E3's 16 pre-RL samples per question, picking one whose final result matches its own contract raises BIRD-dev accuracy over a random pick by more than the free rule (non-empty, no error) does.
5. *Main.* After GRPO (seed 1), E3's primary-set pass@1 on BIRD-dev exceeds the mean of A_s1 and A_s2 by at least 1 point. The effect on Spider-dev is expected to be smaller (the kill-test bound is +2.1). Anything else is reported as a null.
6. *RL behaviour.* After GRPO, E3 submits right after a "does not match" check less often than before RL.

**Results so far (added 2026-10-08 16:10).**
- **A_s2 (2 GPUs)** on the primary set, pass@1/pass@16: Spider-dev 80.3/88.4, BIRD-dev 58.2/76.8.
  - Seed-to-seed A difference: pass@1 −0.23 (Spider-dev) and +0.04 (BIRD-dev); all CIs cover 0.
  - B_s1 − A_s2 on BIRD-dev pass@1: −0.98 [−1.69, −0.24], the same as against A_s1. One B seed only; B_s2 is pending.
  - RL movement of A_s2 is 0.082, against 0.067 for A_s1, 0.107 for B and 0.118 for C.
- **E3 before RL** (results/e3_readouts_pre_E3.json; primary set; Spider-dev / BIRD-dev):
  - *P1(a) supported.* Contract accuracy on unseen DBs is 0.891 / 0.824, vs 0.417 / 0.599 for always writing the majority shape. P1(b) waits for E3p.
  - *Checks are computed correctly* in 99.97% / 99.74% of cases.
  - *P2 supported.* The agent submits right after "matches" in 0.899 / 0.739 of cases, and right after "does not match" in 0.554 / 0.313.
  - *P3 supported.* Pass@1 is 71.8 / 49.5, vs A's 71.4 / 49.7.
  - *P4 supported.* Picking among the 16 samples by the model's own contract gives 76.95 / 56.80, vs 71.81 / 49.49 for a random pick and 72.71 / 53.77 for the free rule. The oracle coarse shape gives 78.2 / 60.3.

**Addendum 7 results (added 2026-10-09 09:50; all runs finished as queued, eval_E3_s1 re-run on 1 GPU because node 004 had one free GPU).**

*Noise floors.*
- In the e2 context, KL(A_seed1‖A) = 0.0070 (as on the plain held-out set), so Outcome-in-Action's 0.0162 is about 2.3× the noise floor.
- A_seed1's KL to base is 0.361, against A's 0.364, so KL to base varies by about 0.003 across data orders.
- A_seed1 pre-RL pass@1 (all questions) is 70.1 on Spider and 49.1 on BIRD, against A's 71.4 and 49.7.

*Pooled primary outcome* (results/paper_analyses.json; 2,498 primary questions; 10k bootstrap):
- B − mean(A): pass@16 +0.02 [−0.72, 0.78]; pass@1 −0.57 [−1.02, −0.12].
- C − B: pass@1 +1.04 [0.59, 1.50].
- E3 − mean(A): pass@16 −0.02 [−0.74, 0.70]; pass@1 −0.22 [−0.57, 0.13].

*E3 predictions (Spider-dev / BIRD-dev):*
1. **Supported.**
   - Contract accuracy is 0.891 / 0.824 (modal over 16 samples: 0.897 / 0.840). For comparison: always the majority shape 0.417 / 0.599; a question-only naive-Bayes classifier 0.711 / 0.491.
   - E3p (placebo) reaches only 0.299 / 0.349.
2. **Supported as stated, but not attributable to the contract.**
   - In its own checks, E3 submits right after a mismatch 0.554 / 0.313 of the time, against 0.899 / 0.739 after a match.
   - The behaviour baseline relabels each observation against the *gold* shape. ActionSFT shows the same asymmetry: P(submit | match) 0.900 / 0.755 vs P(submit | mismatch) 0.475 / 0.291. E3 relabelled the same way gives 0.904 / 0.753 vs 0.491 / 0.282.
   - So the contract does not change submit-or-revise behaviour; it makes an existing, teacher-inherited tendency explicit.
3. **Supported.** Pre-RL pass@1 is 71.8 / 49.5, the same as A.
4. **Supported.**
   - Selection by E3's own contract gives 76.95 / 56.80, vs 72.71 / 53.77 for the free rule. E3p's own contract gives only 73.22 / 52.70.
   - E3's modal contract also transfers to other models' samples: on SFT-A it gives 76.86 / 56.85.
   - Before RL, majority vote is a stronger baseline: 75.82 / 58.70 on SFT-A. Contract + vote gives 78.37 / 60.15 (+2.6 / +1.5 over vote).
5. **Refuted.** After GRPO, E3 pass@1 is 80.77 / 57.52. Against A_s1 and A_s2 on BIRD, the differences are −0.60 [−1.17, −0.03] and −0.65 [−1.22, −0.07]. On Spider they are +0.29 and +0.52.
6. **Refuted.**
   - After GRPO, E3 submits right after a mismatch *more* often: 0.786 / 0.461 by its own checks.
   - Against gold shapes all arms move the same way after RL. On BIRD, A goes 0.291 → 0.40 and E3 goes 0.282 → 0.410.
   - After RL, contract selection no longer beats the free rule: own contract 81.34 / 59.63 vs 81.20 / 59.75. Contract + vote does not beat vote on GRPO-A1 (82.55 / 63.51 vs 82.60 / 64.18).

*Summary.* The contract is learned and transfers to unseen DBs. Before RL it works as an external selector. It does not change the policy's own decisions, and after GRPO it adds nothing to single-attempt accuracy or to selection beyond majority vote.

**Addendum 8 (added 2026-10-09 11:30): primary-set correction for re-execution analyses.** Nothing about the design, arms or hypotheses changes.

*What happened.* `scripts/paper_analyses.py` re-executed every gold query on each run with a 30 s limit. On the 2026-10-08 run, 17 BIRD gold queries timed out on a loaded node and were silently treated as failing, so the pooled set had 2,498 questions instead of the defined primary set (983 Spider + 1,532 BIRD = 2,515). The 2026-10-09 rerun completed all 1,532. The pooled outcome, selection, wrong-answer and decision analyses therefore used a load-dependent subset. Table 1 and the per-dataset pairs (collect_results.py) were not affected.

*Fix.* Gold results are now executed once and frozen (`results/gold_shapes_dev.json`, copy `gold_shapes_dev.frozen_2026-10-09.json`); every analysis uses the defined primary sets. The previous outputs are kept as `results/decision_evidence.pre2515.json`.

*Corrected numbers (2,515 questions; 10k bootstrap):*
- Pooled primary outcome: B − mean(A) pass@16 −0.06 [−0.82, 0.68] (was +0.02 [−0.72, 0.78]); pass@1 −0.57 [−1.04, −0.11]. Conclusion unchanged: no gain.
- E3 − mean(A): pass@16 −0.06 [−0.78, 0.66]; pass@1 −0.22 [−0.58, 0.12]. C − B pass@1 +1.06 [0.60, 1.52].
- Decisions (BIRD): max |ΔS| over SFT arms 0.0095 vs data-order reference 0.020; GRPO RL-seed reference 0.019; A submits after a gold-shape mismatch 0.29 → 0.41 after RL (was 0.40). Spider unchanged.
- Selection, computed in one pipeline on the same set (`selection_own_contract` added): E3's own contract 77.0 / 57.0 vs free rule 72.7 / 53.6 (+4.3 / +3.4). **E3p's own contract 73.7 / 54.5 vs 72.5 / 53.4, i.e. about +1 point**, not "no gain" as in Addendum 7 (73.22 / 52.70 came from e3_readouts.py on a different question set). On SFT-A: contract 76.9 / 56.5, contract + vote 78.4 / 59.7 vs vote 75.8 / 58.3. After GRPO: contract + vote on A1 82.5 / 63.3 vs vote 82.6 / 63.9; E3-GRPO own contract 81.3 / 59.4 vs free rule 81.2 / 59.6.

### Addendum 9 (2026-10-09 18:30): B seed 2 results

B_s2 (job 1510964, same SFT checkpoint as B_s1, RL seed 2) finished; eval 1510965, movement 1510966. Primary sets (983 Spider + 1,532 BIRD; results/eval/primary_pairs_B2.json, results/paper_analyses.json).

- B_s2 pass@1 / pass@16: Spider 81.39 / 88.91, BIRD 57.62 / 77.48.
- **Primary outcome, averaged over seeds as pre-registered (§4):** mean(B) − mean(A) pooled pass@16 = +0.26 [−0.38, 0.91]; pass@1 −0.24 [−0.65, 0.16]. Per seed vs mean(A): B_s1 −0.06 [−0.82, 0.68], B_s2 +0.58 [−0.22, 1.35]. **H-rep not supported:** no CI excludes 0 and the two seeds run have opposite signs (seed 3 on hold, Addendum 6).
- RL-seed variance is larger than the A seeds suggested: B_s2 − B_s1 (same checkpoint) is +0.97 [0.50, 1.47] on Spider pass@1 (A_s2 − A_s1: −0.23). B_s2 vs A on BIRD pass@1: −0.50 [−1.15, 0.16] and −0.55 [−1.24, 0.13], so B_s1's significant BIRD gap (−0.94/−0.98) does not replicate.
- Decision sensitivity after GRPO: ΔS(B_s2 − A_s1) = −0.025 (Spider) / −0.031 (BIRD); the B seeds differ by 0.022 / 0.029, comparable to the A-seed reference (0.052 / 0.019).
- RL movement: KL to its SFT init 0.100, action entropy 0.472 (B_s1: 0.107 / 0.474).
