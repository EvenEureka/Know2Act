### Addendum 20 (2026-10-13, written 17:25–17:50 EDT, before any new data was built or any model trained): ALF-CF main study with three design fixes (author-approved)

*Status.* A **hash-committed protocol**: its md5/sha256 go to `research-issue-tracker/ideas/know2act_alfworld/addendum20_hash.txt`, and the coordinating session pushes them to the private tracker. The paper must not call it "pre-registered" without a public timestamp.

*Approval.* The author approved the main study, relayed by the coordinating session on 2026-10-13 at about 17:20 EDT, together with the three fixes below.

*What had been seen when this was written.*
- Everything through the Addendum 19 results, which are val only.
- The test split is still sealed. No model has been run on a test game; the test games entered only the environment statistics of Addendum 18 step 2.

**Fix 1. Primary note-side arm NQ50op: the foil note's opener is supervised.**
- *Why.* NQ50 (Addendum 19) masked the whole foil note. That cut supervised note writing to about 39% of fallback decision points (against 30% supervised direct actions), and the model then wrote notes at only 0.5% of decision points. So its S_flip of −0.001 confounds "notes are unreliable" with "the model does not write notes, so an inserted note is foreign text".
- *Construction.*
  - N100 trajectories and actions of construction draw d are kept unchanged.
  - At each real-note fallback decision point that has a stored foil, a coin (seed `stable_seed("foil", id, k, d)`) with q = 0.5 writes the foil note in the same template.
  - **Every token of the foil note is supervised except those overlapping the receptacle type names**, which are masked by character spans with offsets. The opener `<prediction>The O is most likely in a`, the connectives and `</prediction>` stay supervised.
  - So the decision to write a note, and its form, are supervised at exactly the same decision points as in N100. Only the content of the foil is never learnt.
- *Arms.* NQ50op is the **primary** note-side arm. NQ25op (q = 0.25, same coin) is a secondary arm. The old fully masked NQ50 stays as a secondary decomposition arm.
- *Mandatory covariate.* For every model: the note-writing rate at fallback decision points in its own greedy rollouts.
- *Reading of S_flip@first(NQ50op), registered.*
  - Below 0.15: **reliability drives the drop**. Unreliable notes remove control even when notes are written.
  - Between 0.15 and S̄(N100) − 0.15: **mixed**. Part of NQ50's drop came from not writing notes. The decomposition S(NQ50op) − S(NQ50) estimates that part.
  - Above S̄(N100) − 0.15: **not writing drove it**. The note-side reliability lever is withdrawn.

**Fix 2. Action-side arm rebuilt (SA50v2), and a presence-only baseline.**
- *(a) Construction.* A new teacher, `mixall`, flips the s = 0.5 coin at **every** fallback decision point whatever its note draw (real, placebo or none): the T⁺ choice if the coin says so, the T⁻ choice otherwise. Notes are always the correct p_cf ranking. So the presence of a note no longer predicts deviation.
  - Matching: subsampled (seeded) to N100's total fallback decision points within ±1%.
  - The Addendum 19 SA50, whose no-note points were always T⁺, is **dropped from the main study**. Its val result stays as decomposition evidence, and its model is deleted at the end.
- *(b) Presence-only condition, added to evaluation for every arm.* Condition `presence` writes the same template with every type slot filled by the generic word "receptacle" ("The O is most likely in a receptacle, then a receptacle, then a receptacle."). It is content-free, untrained, and cannot be copied into a valid `go to` type. Read-outs:
  - presence effect PE = P(intact top | remove) − P(intact top | presence);
  - S_flip net of presence = P(foil top | flip) − P(foil top | presence);
  - follow-normalised S_flip = S_flip / P(intact top | intact).

  Raw S_flip@first stays the primary estimand.

**Fix 3. N100's copying is reported as a finding.** The claim is framed as "training data set how much control a written prediction has over decisions", never as "the model reasons with its knowledge". These registered descriptive secondaries are reported for every arm:
- S_random@first: an untrained random-order note;
- the plausibility slope: S_flip@first in the most-plausible tercile minus the least-plausible tercile;
- the removal null: P(intact top | intact) − P(intact top | remove), and P(convention | remove);
- whole-episode foil effects: P(first visit = foil top | cf_foil), and fallback visits under cf_foil minus cf_true.

**Main study.**
- *Model and training.* Qwen3-4B with the SFT hyperparameters unchanged (lr 1e-5, 1 epoch, global batch 64, max length 12,288, λ = 0, 2 GPUs).
- *Seeds.* A seed is the pair (construction draw d, data-order seed d).
- *Arms.*

  | Arm | Role | Seeds (d) | Existing models reused |
  |---|---|---|---|
  | N100 | primary | 0–7 | d0 = 1537563, d1 = 1537801 |
  | NQ50op | primary | 0–7 | — |
  | NQ50 (fully masked) | secondary | 0–2 | d0 = 1537799 |
  | SA50v2 | secondary | 0–2 | — |
  | N100M (action loss masked at no-note points) | secondary | 0–2 | — |
  | NQ25op | secondary | 0–2 | — |

  - That is 25 new SFTs.
  - Liu et al.'s k = 0.3 variant is **not run**. N100M is the registered analogue, and the extra 3 SFTs and 3 test evaluations would push completion past the next morning.
- *Evaluation v3* = evaluation v2 plus the `presence` condition plus a `--split` option. No other change.
- *Val sanity checks* (exploratory; no decision depends on them): evaluation v3 on val for NQ50op d0, NQ25op d0, SA50v2 d0 and N100M d0.
- *Test.*
  - The test evaluation of all 28 models is submitted held on all 25 SFT jobs.
  - The test analysis script `alfcf_main_analysis.py` and the per-model read-out script are frozen. Their md5s are recorded in a freeze note before the first test job can start.
  - The test split has 1,396 games, all solvable by both teachers.
  - **Primary population:** held-out-floor-plan games plus valid_unseen (1,116 games). valid_seen (280) is reported separately.
- *Primary contrast.*
  - S_flip@first, learnable classes, one value per model (the mean over games).
  - N100 (8 seeds) vs NQ50op (8 seeds), Welch t on seed values, two-sided, α = 0.05.
  - Reported: the difference, the Welch 95% CI, and game-bootstrap CIs within each seed.
  - Prediction: N100 − NQ50op ≥ 0.40 with p < 0.05.
  - For the Fix 1 reading: P(below 0.15) = 0.6, P(mixed) = 0.3, P(near N100) = 0.1.
- *Secondary contrasts.* Holm over H1–H8, family α = 0.05.

  | # | Test | Prediction |
  |---|---|---|
  | H1 | NQ50op vs NQ50 (decomposition) | — |
  | H2 | SA50v2 vs N100 | SA50v2 in 0.15–0.45 |
  | H3 | N100M vs N100, S_flip@first | N100M in 0.80–0.95 |
  | H4 | N100M vs N100, P(convention \| remove) at p = 0, learnable moved | N100M ≥ 0.05 lower (probability 0.6) |
  | H5 | NQ25op vs N100 | NQ25op in 0.10–0.60 |
  | H6 | Res@first in N100 ≠ 0 (one-sample t over 8 seeds) | +0.03 to +0.12 |
  | H7 | P(first visit = foil top \| cf_foil), N100 vs NQ50op | — |
  | H8 | Later-position 2×2 removal effect in N100 ≠ 0 (one-sample t) | +0.05 to +0.15 |

  Further predictions:
  - NQ50 lies in −0.05 to 0.05.
  - SA50v2: |PE| < 0.05 and P(intact top \| intact) 0.5–0.75.
  - N100: S_flip@first mean 0.75–0.88 with seed SD ≤ 0.08; S_random ≥ 0.85; |plausibility slope| < 0.10; |removal null| < 0.05; cf visits difference ≥ +5.
  - Note-writing rate: N100 ≥ 0.95, NQ50op ≥ 0.90, NQ50 ≤ 0.10.
- *Covariates reported per model:*
  - note-writing rate;
  - note top-1 hit;
  - P(intact top | intact);
  - P(convention | remove);
  - presence effect;
  - success;
  - S_flip@first by learnable / unlearnable class.

**Data-construction checks before any SFT on new data.** The check step exits with code 100 on failure, which keeps the SFTs held.
- (e1) NQ50op and NQ25op, per draw:
  - action sequences identical to N100 draw d;
  - supervised note-start share at fallback decision points identical to N100 draw d;
  - masked tokens appear only inside foil type names (every masked token overlaps a type-name span);
  - q realised within ±0.03 of the target, overall and at p = 0.
- (e2) SA50v2, per draw:
  - P(T⁻ action | real note), P(T⁻ action | no note) and P(T⁻ action | placebo) each within 0.50 ± 0.05, and within ±0.03 of each other, at p = 0 and at p ≥ 1, for learnable classes. The p = 0 unlearnable cell is reported.
  - Decision-point count within ±1% of N100 draw d.
- (e3) N100 draws 2–7: teacher wins ≥ 98%; S_data = 1; note presence 0.70 ± 0.02.
- (e4) N100M: masked share 0.30 ± 0.02.

**Scheduling and concurrency.**
- Three 2-GPU lanes: node 003 (2 free GPUs while SARRA uses 2) and node 004 (two lanes). Each lane is a serial `-hold_jid` chain.
- Know2Act jobs share a node only because every temporary path is scoped by job id (SFT: `$LOCAL/sft/sft_<arm>_<jobid>`; evaluation writes nothing node-local). The earlier rsync race came from a shared `$LOCAL/omnisql` copy in the SQL pipeline, which these jobs do not use.
- CPU teacher passes (N100 draws 2–7, mixall draws 0–2) run as one CPU job.

**Compute.** About 25 × 0.6 (SFT) + 4 × 0.6 (val) + 28 × 1.5 (test) ≈ 60 GPU-hours, roughly 12–14 hours of wall clock on three lanes.

**Disk.** Superseded models (Addendum 19 SA50, 1537803) and smoke outputs are deleted at the end. The 28 final models (about 7.6 GB each, roughly 213 GB) are kept unless the author decides otherwise.
