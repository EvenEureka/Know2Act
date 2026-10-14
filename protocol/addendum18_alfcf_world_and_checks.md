### Addendum 18 (2026-10-13, written 12:45–13:10 EDT, before any FT1-CF or teacher statistic was computed): ALFWorld counterfactual-placement world ALF-CF v1, frozen generation rules, FT1 rerun and teacher-data checks, go/no-go for a kill-check pilot

*Context.* On 2026-10-13 the author replaced the text-to-SQL second scenario with ALFWorld. The stage 0–5 files of the lab protocol (`research-issue-tracker/PROCESS_idea_evaluation.md`) are in `research-issue-tracker/ideas/know2act_alfworld/`. FT1 on standard ALFWorld was REDESIGN (object-specific knowledge saves 0.57 visits against a bar of 1.0), so the counterfactual-placement world ALF-CF carries identification. The author approved these defaults on 2026-10-13: π = 0.8; 25% of the object classes unlearnable, as an in-ALFWorld negative control; three dose arms with S_data = 0.50 / 0.75 / 1.00, where S_data is the share of decisions that follow the written location note; and a placebo note in the same format with no information content. Addendum 17 is the paused database-convention draft (`PREREG_conv_DRAFT_2026-10-13.md`); it was never entered here and never run.

*What had been seen when this was written.*
- FT1–FT5 and the exploratory simulation `k2a_alfworld/results/cf_design_sim.json` (π = 0.8: Δ_M 3.7, CDF_A 0.86, 2.3 decision points per game; that convention avoided only the data prior).
- The output of the rule script (`k2a_alfworld/alfcf/rules_v1.json`): the class assignment, the relaxation level per class and the per-class prior statistics of the chosen classes.
- The generation counts per split (moved / not moved / no slot).
- Debug runs of the teacher code on 8 train variants and 15 val variants, to fix two bugs (co-located receptacles answer "Nothing happens." to `go to`; the teacher now examines them). I saw win/loss and the step counts of those episodes; no aggregate was computed.
- No FT1-CF statistic, no teacher-data statistic and no S_data value had been computed. No model had been trained or run on ALF-CF.

*Frozen artifacts (md5).*

| File | md5 |
|---|---|
| `k2a_alfworld/alfcf/rules_v1.json` | 42ccfee301653e361d401d53b037e399 |
| `k2a_alfworld/scripts/alfcf_rules.py` | c4e77dffa1d0ca937beb772b327bac6e |
| `k2a_alfworld/scripts/alfcf_common.py` | bfc5020589de9d0430cbf02a8815891a |
| `k2a_alfworld/scripts/alfcf_generate.py` | b93d25f14986718439d672d0993c1a6a |
| `k2a_alfworld/scripts/alfcf_teachers.py` | 519261ad5581dfbc793a691c0bab9b47 |
| `k2a_alfworld/scripts/ft1cf_analysis.py` | d41341cbb13aa0536f5b65218acbd086 |
| `k2a_alfworld/scripts/alfcf_build_sft.py` | c09a97a2e840b77d27c4308407e2dde0 |
| `k2a_alfworld/scripts/ft1_prior.py` (FT1 definitions, unchanged) | 0b634dfab41290b4cd16a25e72754317 |
| `data/alfcf_v1/index.jsonl` (generated games) | 2157f172b8d6e8879292f950fcfb8952 |

If a script crashes, I may fix the crash. Any fix that changes a definition below is logged with a reason in the results section.

**Step 1: the ALF-CF v1 rules (frozen).**
- *Data.* The ALFWorld json_2.1.1 games used by FT1 (train 3,553; valid_seen 140; valid_unseen 134).
- *Splits.* For each scene type, the 27 train floor plans are shuffled with a seed and split 21 train / 3 val / 3 test. valid_seen and valid_unseen go to test. valid_seen rooms are train floor plans, so test is reported by subset. Base games: train 2,716, val 413, test 698 (424 held-out-floor-plan, 140 valid_seen, 134 valid_unseen).
- *Rule variants.* Each base game yields independent per-game draws: 3 variants in train, 1 in val, 2 in test. All variants of a base game stay in one split. The result is 8,148 / 413 / 1,396 games.
- *Unlearnable classes.* 12 of the 49 target classes (24.5%; 20.3% of train games): apple, butterknife, cd, laptop, peppershaker, remotecontrol, soapbottle, spoon, spraybottle, tissuebox, vase, wateringcan. They were drawn with a seed and accepted only if their game share lies in [0.20, 0.30] and every scene type has one with at least 20 train games. The first draw was accepted.
- *Learnable conventions.* Each learnable (scene type, class) gets a primary class r1 and a secondary r2. They are chosen greedily, classes in descending game count, at the first level of a relaxation ladder that has a candidate. The ladder is in the rules file:
  - presence in ≥ 60% → ≥ 40% of the scene type's floor plans;
  - normalised rank in the class's ALFRED type-share ranking ≥ 0.5 → 0.2;
  - the lower of the Qwen3-4B and Qwen3-8B mean normalised zero-shot ranks (FT4 responses) ≥ 0.4 → 0.1.

  Three conditions hold at every level: never one of the class's top-3 ALFRED types; destination share < 0.10; and a usage cap, so that no r1 class holds more than 30% of the scene type's learnable train games, which defeats object-agnostic heuristics. The kitchen (20 classes) and bathroom (6 classes) are all at level 0. Living (13) and bedroom (12) need levels up to 2 and 4, because few receptacle classes there are both common and off-prior.
- *Placement.* With probability π = 0.8, every instance of the target class moves into one uniformly drawn instance of r1, if r1 is in the room and is not the task's destination class; otherwise into r2 under the same conditions; otherwise the original placement is kept (no_slot). With probability 0.2 the original placement is kept.
- *Unlearnable classes.* With probability 0.8, all instances move into one instance of a class drawn uniformly per game from the classes in the room, destination excluded.
- *Editing.* Only `:init` is edited, and each game keeps its own embedded domain and grammar (FT2 fixes).
- *Generation counts.*
  - train: learnable moved r1 4,837, r2 384, no_slot 40, not moved 1,297; unlearnable moved 1,290, not moved 300.
  - val: 244 / 33 / 1 / 64; 60 / 11.
  - test: 772 / 42 / 0 / 226; 280 / 76.

**Step 2: FT1 rerun on the generated games (`ft1cf_analysis.py`).**
- *Definitions.* The definitions are FT1's (`ft1_prior.py`), applied to ALF-CF:
  - per-instance priors p(o, t) and the object-agnostic marginal q(t) are fitted on the ALF-CF train split;
  - the target location is where the generator put it.
- *Policies.*
  - U: uniform.
  - V: ALFRED class constraints.
  - M_cf: object-agnostic.
  - C_cf: object-conditional.
  - C_std: the standard prior of the same train games before the move, i.e. the ALFRED data prior and T⁻.
  - Q4b / Q8b: the FT4 zero-shot rankings, on the 274 valid base games, both variants.
- *Primary population.* Test split, learnable classes.
- *Measures.*
  - **Δ_M** = E[cost | M_cf] − E[cost | C_cf] (visits saved by object-specific knowledge; bar 1.0);
  - **CDF_A**, the share of decision points along the C_cf path where knowledge is decisive (C_cf top type ≠ M_cf top type), with its strong version;
  - **decisions per game**: visits to the first hit along C_cf, and decision points with at least 2 remaining types;
  - **flippable decision points**: decision points with at least one learnable foil class of the same scene type whose top remaining type differs, summed over the test split, plus CDF_flip.
- *Power check.* The flippable count is compared with the per-arm requirements in `power_and_scale.md`: 327 for one contrast at 0.10, 437 for three contrasts at 0.10, and 1,309 for one contrast at 0.05 (variance 0.15, ICC 0.3).
- *Predictions (written before computing).*
  - Test, learnable classes:

    | Quantity | Predicted |
    |---|---|
    | E[U] | 10–13 |
    | E[M_cf] | 4–7 |
    | E[C_cf] | 1.8–3.0 |
    | E[C_std] | 9–14 |
    | **Δ_M** | **2.0–4.0** |
    | Δ_std | ≥ 6 |
    | CDF_A | 0.60–0.90 |
    | Decisions per game | 1.8–2.8 |
    | CDF_flip | 0.6–0.9 |
    | Flippable decision points | 1,400–2,600 |

  - Unlearnable classes: Δ_M between −0.5 and 1.0.
  - Q4b / Q8b on the learnable valid variants: E[cost] at least E[C_cf] + 4, and top-1 hit ≤ 0.15. In words, the zero-shot models do not know the convention.
  - Val, learnable: 450–900 flippable decision points.
  - I give GO about 80%.
- *Decision rule (test, learnable; coded in the script).*
  - **NO-GO** if Δ_M < 0.3 or CDF_A < 0.20.
  - Otherwise **REDESIGN** if any of these holds: Δ_M < 1.0; CDF_A < 0.40; flippable decision points < 437; decisions per game < 1.5; or the negative control fails, i.e. Δ_M(unlearnable) > Δ_M(learnable) − 1.0.
  - Otherwise **GO**.
  - The Q4b/Q8b prediction and Δ_std are reported against their predictions without a decision attached.

**Step 3: scripted-teacher data (`alfcf_teachers.py`, `alfcf_build_sft.py`).**
- *Teachers.* T⁺ and T⁻ are non-privileged scripted searchers with object memory.
  - They visit receptacle types by a type-level prior over the types that still have unvisited instances, one random unvisited instance at a time, then finish the task by script.
  - T⁺ uses the ALF-CF train prior p_cf; T⁻ uses the standard prior p_std of the same games.
- *Decision points.* Every `go to` taken while the target has never been seen is a fallback decision point.
- *Notes.* Each decision point carries a note draw that is shared by both teachers at the same decision index: real 0.63, placebo 0.07, none 0.30. So notes are present 70% of the time, and 10% of the notes are placebos.
  - The real note is the top-3 of p_cf over the remaining types, written before the action as `<prediction>The O is most likely in a T1, then a T2, then a T3.</prediction>`.
  - The placebo uses the same format with the alphabetical order, which is also the order of the room description, so it carries no information about the target.
  - Each decision point also stores one flip: the p_cf ranking of another learnable class of the same scene type whose top remaining type differs. This is the template pool of model-like flip strings.
- *Arms.*
  - N100: all games T⁺.
  - N75 and N50: a seeded share of games T⁺, the rest T⁻, with the share chosen so that the realised S_data is closest to the target.
  - A (ActionSFT, main study only): the N50 games without note lines.
  - Games where either teacher fails are dropped from every arm.
- *Measures from the data alone.*
  - S_data: share of real-note decision points whose `go to` targets the note's top type.
  - The perfect imitator's flip effect S_flip_imit. The imitator follows the context's note in T⁺ games and acts as T⁻ regardless of the note in T⁻ games, so S_flip_imit equals the share of real-note decision points from T⁺ games.
  - The coincidence P(action = flip top | intact).
  - Learnable-only versions of all of these.
  - Note frequency and placebo share.
  - Flip coverage.
  - Teacher win rates.
  - Decision points and characters per example, per arm.
- *Predictions.*
  - T⁺ wins ≥ 99% and T⁻ ≥ 97% (150-step cap).
  - Realised S_data is within 0.01 of each target.
  - S_flip_imit is 0.40–0.48 (N50), 0.70–0.74 (N75) and 1.00 (N100).
  - The learnable S_data of N50 and N75 is below the overall value, because T⁺ and T⁻ nearly agree on unlearnable classes.
  - Note presence is 0.70 ± 0.02 and placebo share 0.10 ± 0.02.
  - Flip coverage is ≥ 0.90.
  - N50 examples carry 1.5–2.5× the decision points of N100.
- *Criteria (all must hold).*
  - (c1) Both teachers win ≥ 98% of train and of val games.
  - (c2) Realised S_data is within ±0.03 of 0.50 / 0.75 / 1.00.
  - (c3) S_flip_imit(N100) ≥ 0.95, and S_flip_imit rises N50 < N75 < N100.
  - (c4) Note presence is in [0.68, 0.72] and placebo share in [0.08, 0.12].
  - (c5) Flip coverage of real-note decision points is ≥ 0.90.
  - (c6) In every won T⁺ game, the target is found in a receptacle of a type the generator placed it in (generation fidelity).
  - The decision-point and token imbalance between arms is descriptive here. Matching for the main study is designed after seeing it and registered before that study.

**Go / no-go.**
- If step 2 is GO and (c1)–(c6) hold, a short protocol for the kill-check pilot is frozen in a dated Addendum 18b and submitted.
  - *Run.* N100, one seed, Qwen3-4B, SFT with the unchanged hyperparameters (lr 1e-5, 1 epoch, global batch 64, max length 12,288, 2 GPUs, λ = 0).
  - *Evaluation.* Validation split only, with the decision-point interventions intact / flip / remove / placebo.
  - *Kill-check thresholds* (power_and_scale.md): format compliance ≥ 95%; note top-1 type hit on learnable val classes ≥ 0.6; at least 1.5 fallback decision points per game.
- The test split stays sealed until the main study's protocol is frozen.
- Step 2 REDESIGN or NO-GO, or a failed criterion, stops the pilot. The failure is reported, with the redesign it implies: for example more test variants if only the power check fails, or a revised ladder if Δ_M fails.
