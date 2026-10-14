"""PREREG Addendum 21: perfect-imitator reference S_flip* at the first fallback decision point, and copy bias.
Analysis only; does not change Addendum 20 (primary contrast, H1-H8, decision rules, alfcf_main_analysis.py).

S_flip* = the S_flip a policy shows if it reproduces exactly its training data's P(action type | observation, note),
evaluated at the first fallback decision point (training k = 0), learnable classes, with a flip = another learnable
class's convention ranking (the stored foil, drawn uniformly among valid classes, as the evaluation's own flip).
  NQ50op / NQ25op / NQ50 : foil notes occur in training with the T+ action, whose type is the true convention top and
      never the foil top, so P(foil top | foil note) and P(foil top | real note) are both read off the data
      -> S_flip* = P_data(a = foil top | foil) - P_data(a = stored-foil top | real)  (0 by construction; computed).
  SA50v2 : notes are always the correct ranking (or placebo); a foil never occurs, so P(a | obs, foil) is undefined.
      Bounds: context-attributing imitator 0; note-attributing imitator (follows the written top in the T+ half,
      acts as T- regardless of the note in the other half) = realised T+ share s at k = 0 real-note dps.
  N100, N100M : no foil and S_data = 1 -> undefined; bounds 0 (ties to the convention) and 1 (ties to the note).
Copy bias = S_flip - S_flip* (interval for bounded arms), per model; per arm mean over seeds with a t interval;
registered comparison: copy bias of NQ50op vs SA50v2 (Welch t on seed values; SA50v2 against its upper reference s,
i.e. the smallest copy bias attributable to it; the lower-reference version is reported alongside).
usage: python alfcf_imitator_reference.py data <out_reference.json>
       python alfcf_imitator_reference.py test <reference.json> <manifest_test.tsv> <out.json>
"""
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alfcf_text as T  # noqa: E402

D = os.environ.get("ALFCF_DIR", "data/alfcf_v1")  # generated ALF-CF games, teacher rollouts, built arms
QS = {"NQ50op": 0.5, "NQ25op": 0.25, "NQ50": 0.5}
DRAWS = {"N100": range(8), "NQ50op": range(8), "NQ50": range(3), "SA50v2": range(3), "N100M": range(3), "NQ25op": range(3)}


def plus_path(d):
    return f"{D}/teachers/train_plus_draw{d}.jsonl"


def first_dps(path, teacher):
    for l in open(path):
        r = json.loads(l)
        if r["teacher"] != teacher or not r["won"] or not r["learnable"]:
            continue
        for t in r["turns"]:
            dp = t["dp"]
            if dp is not None and dp["k"] == 0:
                yield r, dp
                break


def data_mode(out):
    ref = {}
    for arm, draws in DRAWS.items():
        for d in draws:
            key = f"{arm}|{d}"
            if arm in QS:
                c = Counter()
                for r, dp in first_dps(plus_path(d), "plus"):
                    if dp["draw"] != "real" or not dp["flip_rank"] or dp["n_types"] < 2:
                        continue
                    foil = random.Random(T.stable_seed("foil", r["id"], dp["k"], d)).random() < QS[arm]
                    ft = dp["flip_rank"][0]
                    c["foil" if foil else "real"] += 1
                    c["foil_hit" if foil else "real_hit"] += dp["chosen"] == ft
                    c["real_follow"] += (not foil) and dp["chosen"] == dp["note_rank"][0]
                pf = c["foil_hit"] / max(1, c["foil"])
                pr = c["real_hit"] / max(1, c["real"])
                ref[key] = dict(kind="point", S_star=pf - pr, P_foiltop_given_foil=pf, P_foiltop_given_real=pr,
                                n_foil=c["foil"], n_real=c["real"], P_follow_real=c["real_follow"] / max(1, c["real"]))
            elif arm == "SA50v2":
                c = Counter()
                kept = {json.loads(l)["game"] for l in open(f"{D}/sft_v3/text_SA50v2_d{d}.jsonl")}  # subsampled set
                for r, dp in first_dps(f"{D}/teachers/train_mixall050_draw{d}.jsonl", "mixall"):
                    if r["id"] not in kept or dp["draw"] != "real" or not dp["flip_rank"] or dp["n_types"] < 2:
                        continue
                    c["n"] += 1
                    c["plus"] += dp["chosen_by"] == "plus"
                    c["minus"] += dp["chosen_by"] == "minus"
                    c["minus_coinc_foil"] += dp["chosen_by"] == "minus" and dp["minus_choice"] == dp["flip_rank"][0]
                s = c["plus"] / max(1, c["n"])
                ref[key] = dict(kind="bounds", S_star_low=0.0, S_star_high=s, realised_plus_share=s, n=c["n"],
                                minus_coincides_with_foil_top=c["minus_coinc_foil"] / max(1, c["minus"]))
            else:
                ref[key] = dict(kind="bounds", S_star_low=0.0, S_star_high=1.0,
                                note="no foil in training and S_data = 1: P(a | obs, foil) undefined")
    json.dump(ref, open(out, "w"), indent=1)
    agg = {}
    for key, v in ref.items():
        arm = key.split("|")[0]
        agg.setdefault(arm, []).append(v)
    for arm, vs in agg.items():
        if vs[0]["kind"] == "point":
            print(arm, "S_star per draw", [round(v["S_star"], 4) for v in vs], "n_foil", [v["n_foil"] for v in vs])
        else:
            print(arm, "bounds per draw", [(v["S_star_low"], round(v["S_star_high"], 4)) for v in vs])


def test_mode(ref_path, manifest, out):
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_analysis", os.path.join(HERE, "alfcf_main_analysis.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    ref = json.load(open(ref_path))
    rows = [l.rstrip("\n").split("\t") for l in open(manifest) if l.strip() and not l.startswith("arm\t")]
    per = {}
    for arm, d, path in rows:
        b = json.load(open(path))["primary"]["first_L"]["S_flip"]
        r = ref[f"{arm}|{d}"]
        lo = r["S_star"] if r["kind"] == "point" else r["S_star_low"]
        hi = r["S_star"] if r["kind"] == "point" else r["S_star_high"]
        per.setdefault(arm, []).append(dict(draw=int(d), S=b[0], S_ci=b[1:3], S_star_low=lo, S_star_high=hi,
                                            copy_bias_vs_low=b[0] - lo, copy_bias_vs_high=b[0] - hi,
                                            copy_bias_ci_vs_high=[b[1] - hi, b[2] - hi]))
    res = {"per_model": per, "per_arm": {}}
    for arm, v in per.items():
        for k in ["copy_bias_vs_low", "copy_bias_vs_high"]:
            x = [m[k] for m in v]
            res["per_arm"].setdefault(arm, {})[k] = M.one_sample(x) if len(x) > 1 else {"mean": x[0]}
    cb_nq = [m["copy_bias_vs_high"] for m in per["NQ50op"]]
    res["registered_comparison"] = {
        "NQ50op_vs_SA50v2_upper_reference": M.welch(cb_nq, [m["copy_bias_vs_high"] for m in per["SA50v2"]]),
        "NQ50op_vs_SA50v2_lower_reference": M.welch(cb_nq, [m["copy_bias_vs_low"] for m in per["SA50v2"]])}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["registered_comparison"], indent=1))
    for arm, v in res["per_arm"].items():
        print(arm, {k: round(x["mean"], 4) for k, x in v.items()})


if __name__ == "__main__":
    if sys.argv[1] == "data":
        data_mode(sys.argv[2])
    else:
        test_mode(*sys.argv[2:5])
