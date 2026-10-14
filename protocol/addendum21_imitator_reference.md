### Addendum 21 (registered 2026-10-13 18:58 EDT, analysis only, before any test read-out existed): perfect-imitator reference S_flip* and copy bias

*Status.* A hash-committed protocol, analysis only. It changes nothing in Addendum 20: not the primary contrast, not the H1–H8 predictions, not the decision rules, and not the frozen `alfcf_main_analysis.py`. When it was written, the test-output directory contained 0 entries, and no test evaluation had started. Training was still running.

*Why.* Stage 2 of the lab protocol asks whether a perfect imitator of the training data would give the same result.
- In NQ50op, NQ25op and NQ50, the actions are identical to N100's. Given the observation, the action is therefore independent of the note's content: a foil note is always followed by the true convention action.
- An imitator would ignore a flipped note. S_flip ≈ 0 in these arms is then implied by how the data were built and is not, by itself, a finding about the model.
- The informative quantities are:
  - (a) N100, whose training data never contain a flipped note, so its S_flip is the model's inductive choice;
  - (b) each arm's departure from its own imitator reference.

*Definition.* S_flip* is the S_flip of a policy that reproduces exactly its training data's P(action type | observation, note).
- It is evaluated at the first fallback decision point (training index k = 0), for learnable classes.
- The flip is another learnable class's convention ranking, drawn uniformly among the valid classes, as in the evaluation's own-flip condition.
- Assumptions:
  - the model's own flip note at evaluation has the same content as the template ranking of that class (the val check own vs template at p = 0 gave −0.003 and −0.089 for the two N100 seeds);
  - the imitator conditions on the full observation, so it can tell a foil from the true convention;
  - the within-type instance choice does not matter (S_flip is defined on types).

Computed by `k2a_alfworld/scripts/alfcf_imitator_reference.py data` from the training-data generators (teacher records, foil coins, and the subsampled SA50v2 game sets). Output: `k2a_alfworld/results/imitator_reference.json`.

| Arm (draws) | S_flip* | Data basis |
|---|---|---|
| NQ50op (d0–d7) | **0 (point)** | P(foil top \| foil note) = 0 and P(stored-foil top \| real note) = 0 in every draw (2,050–2,119 foil decision points at k = 0 per draw) |
| NQ25op (d0–d2) | **0 (point)** | same (1,017–1,063 foil decision points) |
| NQ50 (d0–d2) | **0 (point)** | same; the foil note is loss-masked, but the action that follows it is supervised and never goes to the foil top |
| SA50v2 (d0–d2) | **undefined; bounds [0, s]** with s = **0.501 / 0.491 / 0.499** | No foil ever occurs in training (notes are the correct ranking or the placebo). A context-attributing imitator gives 0. A note-attributing imitator follows the written top in the T⁺ half and acts as T⁻ whatever the note in the other half, which gives exactly the realised T⁺ share s at k = 0 real-note points (the T⁻ coincidence with the foil top, about 0.09–0.10, cancels) |
| N100 (d0–d7), N100M (d0–d2) | **undefined; bounds [0, 1]** | No foil and S_data = 1: 0 if the tie goes to the convention, 1 if it goes to the note |

*Copy bias (secondary, descriptive).* Copy bias = S_flip − S_flip*. It uses the per-model S_flip@first (learnable, primary test population) from the Addendum 20 read-outs, with the same within-model game-bootstrap interval shifted by S_flip*.
- For bounded arms it is the interval [S_flip − upper, S_flip − lower].
- Per arm: the mean over seeds with a t interval.

*One registered comparison* (secondary; outside the Holm family H1–H8; two-sided Welch t on seed values, α = 0.05):
- copy bias of NQ50op (S_flip − 0) vs copy bias of SA50v2 against its **upper** reference (S_flip − s), i.e. the smallest copy bias attributable to SA50v2;
- the version against the lower reference (S_flip − 0) is reported alongside.

*Command after the test read-outs exist.* `python3 k2a_alfworld/scripts/alfcf_imitator_reference.py test k2a_alfworld/results/imitator_reference.json k2a_alfworld/results/main_manifest_test.tsv k2a_alfworld/results/imitator_copybias_test.json`

| File | md5 |
|---|---|
| `alfcf_imitator_reference.py` | 909e7c08b455331902294190220666d0 |
| `imitator_reference.json` | 3375d5a5e333cde90263df79190c77c4 |
