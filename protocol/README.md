# Protocol files

The ALF-CF study was run under four protocol addenda. Each was written before the step it governs and committed by hash to a private, timestamped repository. The files here are **byte-identical** copies of the hashed texts. You can check them with `sha256sum -c SHA256SUMS` or `md5sum -c MD5SUMS` in this folder. The checksums show that these files are the hashed texts. They do not show when the texts were written, because the repository that holds the hashes is private.

| File | Defines | Written |
|---|---|---|
| `addendum18_alfcf_world_and_checks.md` | The counterfactual world ALF-CF: its generation rules, the decision-value checks on the generated games, and the scripted-teacher data checks | 2026-10-13, before any ALF-CF statistic |
| `addendum19_revised_design.md` | The estimand (flip effect at the first fallback decision point), the note-side and action-side training arms, the evaluation conditions (flip, remove, placebo, random order), and counterfactual rollouts | 2026-10-13, before the data for these arms existed |
| `addendum20_main_study.md` | **The main study**: the primary contrast N100 vs NQ50op, its decision rule, secondary contrasts H1–H8 with Holm correction, eight seeds per primary arm, and the held-out test population | 2026-10-13, before any main-study data was built or model trained |
| `addendum21_imitator_reference.md` | The perfect-imitator reference S_flip* and the copy bias (analysis only) | 2026-10-13, before any test read-out existed |

The addenda are working documents. In a few places they name internal job identifiers, compute nodes, and sibling projects of the lab. These names are kept, because editing them would break the hashes.

## What "confirmatory" means here

A result is confirmatory only if three things held:

- the protocol was committed by hash before the test split was opened;
- the study-level analysis script was frozen before any test evaluation ran;
- the test evaluations started only after all 28 models had been trained.

The frozen analysis script is `scripts/alfcf_main_analysis.py`. Its md5, `47e6a69bb26d97dcc772663374112bf0`, was recorded before the first test job.

Everything evaluated on the validation split is exploratory.

## Released code vs. the hashed versions

The following released scripts are byte-identical to the versions recorded in the protocol:

| Script | md5 |
|---|---|
| `alfcf_main_analysis.py` | 47e6a69b… |
| `alfcf_eval_v3_analysis.py` | 452dcf37… |
| `alfcf_teachers.py` | a29d7ea4… |

The following scripts differ only in machine-specific path constants. Absolute paths were replaced by environment variables (`ALFWORLD_ROOT`, `ALFWORLD_JSON`, `ALFCF_DIR`, `ALFWORLD_PYTHON`) or by repository-relative paths, and docstrings were edited to match, including the removal of internal file references:

- `alfcf_common.py`
- `ft1_prior.py`
- `ft1cf_analysis.py`
- `alfcf_build_v3.py`
- `alfcf_imitator_reference.py`
- `alfcf_pilot_eval_v3.py`
- `alfcf_tokenize.py`
- `alfcf_envworker.py`
- `ft4_build_probes.py`
- `ft4_probe.py`

Teacher-rollout files also follow one naming scheme now: `teachers/train_plus_draw{d}.jsonl` and `teachers/train_mixall050_draw{d}.jsonl`.

No logic changed.

## Deviations during the main study

Three deviations were logged during the main study. None of them touched a model output or the analysis.

1. Two training jobs that shared a machine collided on the default `torchrun` rendezvous port. They were rerun with a job-specific port.
2. A job-submission list was truncated by the scheduler's comma separator. The affected data-building jobs were rerun before any training used their output.
3. One data-construction check for SA50v2 was too strict for small cells: it required every pair of deviation rates to agree within 0.03, and the placebo cells have n ≈ 360. It was replaced by a note-present vs. no-note criterion plus a χ² independence test. The replacement was made before any SA50v2 model was trained. All draws pass both versions except for that one pairwise condition.
