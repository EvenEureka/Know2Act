### Addendum 19 (2026-10-13, written 14:45–15:05 EDT, before any new data was built or any statistic computed): ALF-CF revised design after the stage 6 review; the dose fatal test

*Status of this document.* This is a **hash-committed protocol**: its md5 is recorded when it is written, and the coordinating session pushes it to the private tracker repository as a timestamp. It is not a public pre-registration, and the paper must not call it "pre-registered" unless a public timestamp is added.

*Why this addendum.*
- The stage 6 review (`research-issue-tracker/ideas/know2act_alfworld/stage6_review_2026-10-13.md`) found that the Addendum 18 dose arms do not identify "note path vs weight path".
- It also found that whole-game mixing makes the dose depend on position, that the pooled S_flip is confounded by each arm's own search length, and that removal and placebo are null by construction.
- It made changes R1–R7 blocking. This addendum implements them. The author has approved small-scale work only; the main study still needs the author's approval and is not started here.

*What had been seen when this was written.*
- Everything up to the Addendum 18b results: the val pilot of N100 seed 0.
- The reviewer's recomputations, quoted in the review: the realised first-decision dose of the old whole-game arms (N50 0.81, N75 0.93) and the breakdown of the pilot's later positions.
- No data for any arm below had been built. No statistic about them had been computed. No new model had been trained.

**R1. Estimands.**
- **Primary estimand: S_flip@first.** It is defined at the first fallback decision point of a game (position p = 0), learnable classes, with the model's own flip note:
  - S_flip@first = P(next `go to` type = foil top | flip) − P(same | intact).
  - The unit is the game (one first decision point per game). The arm-level value is the mean over games.
- **This estimand was chosen after the val pilot** (Addendum 18b results: 0.84 at p = 0 vs about 0.15 later). It is confirmatory only on the sealed test split, in a main study registered separately.
- **Main-study primary population.** Test games from held-out floor plans plus valid_unseen. valid_seen (train floor plans) is reported separately and not pooled.
- **Later positions** (p ≥ 1) are reported by position (1, 2, 3+) and, for learnable classes, by whether the convention class has already been visited (*refuted*: visited and the target not found) or not. They are never pooled across positions and never compared across arms as a pooled number.
- **Registered secondary measure, resistance:** Res@first = S_flip@first(unlearnable) − S_flip@first(learnable), at the same position p = 0.
- **Knowledge strength, reported per arm:**
  - note top-1 hit at p = 0;
  - P(convention class | remove) at p = 0 on learnable moved games;
  - P(intact top | intact).

  S_flip@first is also reported conditional on these.

**R2. Action-side dose, mixed per decision point (arm SA50).**
- *Teacher.* A new teacher pass on the train split uses the same games and note draws as N100 (construction draw 0). At every fallback decision point whose note draw is *real*, an independent seeded coin with s = 0.5 decides whether the action is the T⁺ choice (top of p_cf over the remaining types) or the T⁻ choice (top of p_std). The trajectory then continues from the action actually taken.
- *Fixed across arms.* At no-note and placebo decision points the action is always the T⁺ choice. Notes are always the correct p_cf ranking. The note-draw sequence is the same as in N100 (keyed by game and decision index).
- *Dose.* The realised T⁺-choice share and S_data (including coincidences) are reported by position and by class.
- *Matching.* SA50 trajectories are longer, so SA50 examples are subsampled (seeded) until the total number of fallback decision points equals N100's (within ±1%). Example counts, first-decision counts, supervised tokens and optimiser steps are reported for both arms.

**R3. Note-side dose (arm NQ50): the one-difference contrast with N100.**
- NQ50 uses N100 construction-draw-0 trajectories and actions, unchanged.
- At each real-note fallback decision point that has a stored flip (another learnable class of the same scene type whose top remaining type differs, already drawn at random by the teacher pass), an independent seeded coin with q = 0.5 replaces the note line with that foil note. The foil has the same template, so it is a real-note form.
- The foil note's tokens are **loss-masked** (token class 0). Action tokens stay supervised.
- Then S_data = 1 − q_realised. Action sequences are identical to N100, and lengths differ only by the note text.
- q = 0.25 is registered as an optional main-study arm.

**R4. Removal and placebo.**
- *Arm N100M* (main study; built now, not trained in this addendum). N100 construction-draw-0 data with the action loss masked at no-note fallback decision points (about 30%). It doubles as the analogue of Liu et al. 2026's action-loss-masking lever. Their k = 0.3 sample-level variant is optional.
- *New evaluation condition* **random**: an untrained random-order note. The template is filled with a seeded random permutation of the remaining types, top-3 listed, conditioned on the top differing from both the intact top and the alphabetical first type. Read-out: S_random = P(random top | random) − P(random top | intact).
- The alphabetical **placebo** is kept only as a manipulation check, because it was trained (7% of decision points) and never followed.

**R5. Flip protocol (evaluation v2, `alfcf_pilot_eval_v2.py`).**
- **Own flip.** At every decision point, the model's own greedy note is generated for *every* learnable same-scene class o′ ≠ target: rename the target to o′ in the task line and in earlier notes, prefill `<prediction>The o′ is most likely in a`, then rename back.
  - o′ is drawn uniformly (seeded) among the *valid* candidates: own top still unvisited and different from the intact top.
  - Recorded: the number of valid candidates, and the foil's *plausibility*, i.e. the normalised rank of the foil top in the true class's p_cf ranking over the remaining types.
  - Decision points with no valid own flip are excluded from S_flip and counted. Template flips do not enter any S_flip estimate.
- **Template check (p = 0).** A template flip with the same o′ (the p_cf ranking of o′) is a separate condition. It serves as the val equivalence check, model-own vs template, at the first decision point.
- **No whole-history flips.**
- **Later positions (p ≥ 1).** A 2×2 design: {earlier notes kept, earlier notes removed (every `<prediction>` line in the history deleted, which is in distribution)} × {current note intact, own flip}.
- **Episode level: counterfactual rollouts.** For each val game with a valid first-decision foil o′, two greedy rollouts (≤ 50 steps) run with forced notes:
  - cf_foil: the template ranking of o′ over the remaining types, written at every note slot (target never seen, and the last observation is neither "is closed" nor "Nothing happens");
  - cf_true: the true class's template ranking, as the control.

  Actions are the model's own. Read-outs: P(first visited type = o′'s top), fallback visits until the target is seen, and success. Each is compared with cf_true and with the free rollout.
- **Other conditions.** All conditions use K = 4 samples at temperature 0.7, with common random numbers keyed by (decision point, k). The conditions are: intact, flip, remove, placebo, random; flip_template at p = 0; earlier-removed × {intact, flip} at p ≥ 1.

**R6. The dose fatal test (val only; test stays sealed).**
- *Runs.* Qwen3-4B with the unchanged SFT hyperparameters (lr 1e-5, 1 epoch, global batch 64, max length 12,288, λ = 0, 2 GPUs):

  | Run | Arm | Construction draw | Data-order seed |
  |---|---|---|---|
  | (a) | N100 | 1 (new teacher pass: new note draws, new within-type instance order) | 1 |
  | (b) | NQ50 | 0 | 0 |
  | (c) | SA50 | 0 | 0 |
  | (d) | existing N100 seed-0 model (job 1537563), evaluation v2 only | 0 | 0 |

  All four use evaluation v2 on the 413 val games. That is three SFTs plus four evaluations, about 5–6 GPU-hours, on node 004 (or 003 if 2 GPUs are free), serial per node.
- *Decision rule.*
  - Let S̄ = mean(S_flip@first of N100 s0, N100 s1), and Δ_X = S_flip@first(X) − S̄ for X ∈ {NQ50, SA50}.
  - **If max(|Δ_NQ50|, |Δ_SA50|) < 0.10, the data-lever half of the claim is dropped.** The study becomes "unnecessary but controlling, fading with evidence". No dose arm goes into the main study.
  - Otherwise the lever is retained. The main-study primary contrast is fixed now:
    - N100 vs NQ50 if |Δ_NQ50| ≥ 0.10;
    - otherwise N100 vs SA50.
  - **Inconclusive:** if |S0 − S1| ≥ 0.10, the seed noise is as large as the threshold. A third N100 seed (construction draw 2) is run before any decision.
- *Seed rule for the main study.*
  - σ̂ = max(|S0 − S1| / √2, 0.03).
  - n per primary arm = max(5, ⌈2 (1.96 + 0.84)² σ̂² / 0.10²⌉).
  - Secondary arms with a claim attached get at least 3 seeds; otherwise they are labelled descriptive.
- *Predictions (val, learnable, first decision point unless stated).*

  | Read-out | Prediction |
  |---|---|
  | N100 s0, evaluation v2 | 0.78–0.92 |
  | N100 s1 | within ±0.06 of s0 |
  | NQ50 | 0.05–0.45 (central 0.20), with Res@first ≥ +0.20: the learnable convention now resists a disagreeing note, the unlearnable classes resist less |
  | SA50 | 0.30–0.65 (central 0.45); P(intact top \| intact) 0.55–0.80 |
  | N100 Res@first | within ±0.05 |
  | N100 S_random | 0.30–0.80 (central 0.60), below S_flip |
  | N100 template vs own flip at p = 0 | within ±0.05 |
  | N100 later positions: earlier-notes-removed minus kept, under flip | ≥ +0.05 (probability 0.6) |
  | N100 counterfactual rollouts: P(first visit = foil top) | 0.75–0.95 |
  | N100 counterfactual rollouts: fallback visits to target, cf_foil minus cf_true | ≥ +1.0 |
  | N100 counterfactual rollouts: success | ≥ 0.95 |
  | At least one arm with \|Δ\| ≥ 0.10 | probability 0.85 |

- *Data-construction checks, before any training.* If a check fails, the construction is fixed and the fix logged before any SFT.
  - (d1) N100 draw 1: both teachers win ≥ 98%; S_data = 1.000; note presence 0.70 ± 0.02; placebo share 0.10 ± 0.02.
  - (d2) NQ50: per example, the action token sequences equal N100 draw 0's; no foil token is supervised; realised q among real-note decision points is 0.50 ± 0.03 overall and 0.50 ± 0.05 at p = 0 and at p ≥ 1, for learnable and for unlearnable classes.
  - (d3) SA50: the realised T⁺-choice share at real-note decision points is 0.50 ± 0.05 at p = 0 and at p ≥ 1, for learnable and for unlearnable classes; no-note and placebo decision points are 100% T⁺; after subsampling, total fallback decision points are within ±1% of N100's; tokens, examples and steps reported.
  - (d4) N100M: the masked share of fallback actions is 0.30 ± 0.02; nothing else changes.

**R7. Registration hygiene.**
- **One primary contrast** in the main study, fixed by R6. All secondaries are Holm-corrected:
  - SA50 or NQ50 vs N100, whichever is not primary;
  - Res@first per arm and its difference between the primary arms;
  - N100M vs N100 on S_flip@first and on P(convention | remove);
  - the counterfactual-rollout P(first visit = foil top) between the primary arms;
  - the 2×2 removal effect at p ≥ 1 in N100.
- **Seed** means the pair (data-construction draw, data-order seed). Every new seed of an arm changes both.
- valid_seen is reported separately.
- Seed-level analysis: Welch t on seed means, plus a game-cluster bootstrap within seed.
- The main study (arms, seeds, test split) is registered in its own addendum after this fatal test and after the author's approval.

*Main-track items, registered later and not part of this addendum:* Llama-3.1-8B on the primary arms (≥ 3 seeds), one visit-penalised GRPO stage, an ALF-STD replicate of S_flip@first, and a paraphrased-note arm. Dropped from the main line: the IWM baseline, the ScienceWorld appendix, and the A arm unless a registered claim needs it.
