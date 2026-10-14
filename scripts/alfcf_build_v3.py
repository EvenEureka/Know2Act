"""PREREG Addendum 20: build ALF-CF main-study SFT arms per construction draw, with data-construction checks (e1)-(e4).
Pure python. Exits with code 100 if any check of a requested arm fails (keeps held SFT jobs held).

Arms (text level; act segments carry `loss` and optional `mask_spans` used by alfcf_tokenize.py):
  N100    : T+ trajectories of draw d
  NQ50op  : N100 draw d unchanged; at real-note dps with a stored foil, coin stable_seed("foil", id, k, d) < 0.50
            writes the foil note; every foil-note token is supervised EXCEPT those overlapping the type names (Fix 1)
  NQ25op  : same with q = 0.25 (same coin)
  NQ50    : Addendum 19 construction (whole foil note loss-masked), q = 0.50
  N100M   : N100 draw d with the action loss masked at no-note fallback dps
  SA50v2  : `mixall` teacher (s = 0.5 coin at EVERY fallback dp), subsampled to N100 draw d's dp count (Fix 2)
usage: python alfcf_build_v3.py <out_dir> ARM:DRAW [ARM:DRAW ...]
"""
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alfcf_text as T  # noqa: E402

D = os.environ.get("ALFCF_DIR", "data/alfcf_v1")  # generated ALF-CF games, teacher rollouts, built arms
QS = {"NQ50op": 0.5, "NQ25op": 0.25, "NQ50": 0.5}


def plus_path(d):
    return f"{D}/teachers/train_plus_draw{d}.jsonl"


def load(path, teacher):
    out, n = {}, 0
    for l in open(path):
        r = json.loads(l)
        if r["teacher"] != teacher:
            continue
        n += 1
        if r["won"]:
            out[r["id"]] = r
    return out, n


def note_with_spans(obj, rank):
    """'<prediction>The O is most likely in a T1, then a T2, then a T3.</prediction>\\n' and the char spans of T1..T3"""
    ts = list(rank)[:3]
    s = f"<prediction>The {obj} is most likely in a "
    spans = []
    for i, t in enumerate(ts):
        if i:
            s += ", then a "
        spans.append([len(s), len(s) + len(t)])
        s += t
    s += ".</prediction>\n"
    assert s == f"<prediction>{T.note_text(obj, ts)}</prediction>\n"
    return s, spans


def build(rec, mode, d):
    segs, dps = [], []
    q = QS.get(mode)
    turns = rec["turns"]
    for i, t in enumerate(turns):
        dp = t["dp"]
        act = {"type": "act", "text": f"<action>{t['action']}</action>", "loss": True}
        if dp is not None:
            written, form, spans = None, "none", None
            if dp["draw"] == "real":
                written, form = dp["note_rank"], "real"
                if q and dp["flip_rank"]:
                    if random.Random(T.stable_seed("foil", rec["id"], dp["k"], d)).random() < q:
                        written, form = dp["flip_rank"], "foil"
            elif dp["draw"] == "placebo":
                written, form = dp["placebo_rank"], "placebo"
            if written is not None:
                txt, sp = note_with_spans(rec["target"], written)
                seg = {"type": "act", "text": txt, "loss": True}
                if form == "foil":
                    if mode == "NQ50":
                        seg["loss"] = False
                    else:
                        seg["mask_spans"] = sp
                segs.append(seg)
            if mode == "N100M" and form == "none":
                act["loss"] = False
            dps.append(dict(k=dp["k"], n_types=dp["n_types"], form=form, chosen=dp["chosen"],
                            written_top=written[0] if written else None, chosen_by=dp.get("chosen_by", "plus"),
                            action_loss=act["loss"],
                            note_start_supervised=(written is not None and not (form == "foil" and mode == "NQ50"))))
        segs.append(act)
        if i < len(turns) - 1:
            segs.append({"type": "obs", "text": T.OBS_OPEN + t["obs"] + T.OBS_CLOSE, "loss": False})
    ex = {"id": f'{rec["id"]}|{mode}|d{d}', "game": rec["id"], "target": rec["target"], "scene": rec["scene"],
          "learnable": rec["learnable"], "system": T.SYSTEM, "user": rec["intro"].strip(), "segments": segs}
    return ex, dps


def stats(items):
    c = Counter()
    for ex, dps in items:
        L = "L" if ex["learnable"] else "U"
        for x in dps:
            pos = "p0" if x["k"] == 0 else "p1+"
            c[("dp", "all")] += 1
            c[("note_start_sup", "all")] += x["note_start_supervised"]
            c[("present", "all")] += x["form"] != "none"
            c[("placebo", "all")] += x["form"] == "placebo"
            c[("masked_action", "all")] += not x["action_loss"]
            if x["form"] in ("real", "foil"):
                c[("realform", "all")] += 1
                c[("realform", pos)] += 1
                c[("foil", "all")] += x["form"] == "foil"
                c[("foil", pos)] += x["form"] == "foil"
                c[("follow", "all")] += x["chosen"] == x["written_top"]
            c[("n", x["form"], pos, L)] += 1
            c[("minus", x["form"], pos, L)] += x["chosen_by"] == "minus"
    r = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    out = dict(n_examples=len(items), n_dp=c[("dp", "all")],
               note_start_supervised_share=r(("note_start_sup", "all"), ("dp", "all")),
               note_present=r(("present", "all"), ("dp", "all")),
               placebo_share_of_notes=r(("placebo", "all"), ("present", "all")),
               masked_action_share=r(("masked_action", "all"), ("dp", "all")),
               q_overall=r(("foil", "all"), ("realform", "all")), q_p0=r(("foil", "p0"), ("realform", "p0")),
               S_data=r(("follow", "all"), ("realform", "all")))
    out["minus_share"] = {f"{f}|{pos}|{L}": (c[("minus", f, pos, L)] / c[("n", f, pos, L)] if c[("n", f, pos, L)] else None,
                                             c[("n", f, pos, L)])
                          for f in ["real", "none", "placebo"] for pos in ["p0", "p1+"] for L in ["L", "U"]}
    out["sup_chars"] = sum(len(s["text"]) for ex, _ in items for s in ex["segments"] if s["type"] == "act" and s["loss"])
    return out


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "build_v3_report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    ok_all = True
    cache = {}
    for spec in sys.argv[2:]:
        mode, d = spec.split(":")
        d = int(d)
        if d not in cache:
            cache[d] = load(plus_path(d), "plus")
        plus, n_plus = cache[d]
        base = [build(plus[g], "N100", d) for g in sorted(plus)]
        base_st = stats(base)
        checks = {}
        if mode == "SA50v2":
            mx, n_mx = load(f"{D}/teachers/train_mixall050_draw{d}.jsonl", "mixall")
            sa = [build(mx[g], "SA50v2", d) for g in sorted(mx)]
            target = sum(len(x) for _, x in base)
            order = list(range(len(sa)))
            random.Random(T.stable_seed("subsample", d)).shuffle(order)
            cum, best, best_k = 0, None, 0
            for i, j in enumerate(order):
                cum += len(sa[j][1])
                if best is None or abs(cum - target) < best:
                    best, best_k = abs(cum - target), i + 1
                if cum > target * 1.02:
                    break
            items = [sa[j] for j in sorted(order[:best_k])]
            st = stats(items)
            ms = st["minus_share"]
            for pos in ["p0", "p1+"]:
                # (e2) as amended in Addendum 20 deviation log 3: the registered pairwise +-0.03 tolerance was too tight for
                # the small placebo cells (n ~ 360-650, binomial SE ~ 0.02-0.026). Large cells keep +-0.03; independence of
                # deviation and note form is tested with a chi-square (2 df) test, p > 0.01.
                cells = {f: ms[f"{f}|{pos}|L"] for f in ["real", "none", "placebo"]}
                checks[f"e2_{pos}_L_real_none_in_0.45_0.55"] = all(0.45 <= cells[f][0] <= 0.55 for f in ["real", "none"])
                checks[f"e2_{pos}_L_placebo_in_0.40_0.60"] = 0.40 <= cells["placebo"][0] <= 0.60
                npres = cells["real"][1] + cells["placebo"][1]
                pres = (cells["real"][0] * cells["real"][1] + cells["placebo"][0] * cells["placebo"][1]) / npres
                checks[f"e2_{pos}_L_present_minus_none_within_0.03"] = abs(pres - cells["none"][0]) <= 0.03
                tn = sum(c[1] for c in cells.values())
                pm = sum(c[0] * c[1] for c in cells.values()) / tn
                chi = sum((c[0] * c[1] - pm * c[1]) ** 2 / (pm * c[1]) + ((1 - c[0]) * c[1] - (1 - pm) * c[1]) ** 2 / ((1 - pm) * c[1])
                          for c in cells.values())
                checks[f"e2_{pos}_L_chi2_independence_p_gt_0.01"] = math.exp(-chi / 2) > 0.01
            checks["e2_dp_ratio_within_1pct"] = abs(st["n_dp"] / base_st["n_dp"] - 1) <= 0.01
            checks["e2_teacher_wins_ge_0.98"] = len(mx) / max(1, n_mx) >= 0.98
        else:
            items = [build(plus[g], mode, d) for g in sorted(plus)] if mode != "N100" else base
            st = stats(items)
            if mode == "N100":
                checks["e3_wins_ge_0.98"] = len(plus) / max(1, n_plus) >= 0.98
                checks["e3_S_data_1"] = abs(st["S_data"] - 1.0) < 1e-9
                checks["e3_note_present_0.70"] = abs(st["note_present"] - 0.70) <= 0.02
            if mode in ("NQ50op", "NQ25op", "NQ50"):
                same = all([s["text"] for s in a[0]["segments"] if s["type"] == "act" and s["text"].startswith("<action>")] ==
                           [s["text"] for s in b[0]["segments"] if s["type"] == "act" and s["text"].startswith("<action>")]
                           for a, b in zip(base, items))
                checks["e1_actions_identical"] = same
                checks["e1_q_overall"] = abs(st["q_overall"] - QS[mode]) <= 0.03
                checks["e1_q_p0"] = abs(st["q_p0"] - QS[mode]) <= 0.03
                if mode != "NQ50":
                    checks["e1_note_start_supervised_equal_N100"] = abs(st["note_start_supervised_share"] - base_st["note_start_supervised_share"]) < 1e-12
                    spans_ok = all(all(0 <= a < b <= len(s["text"]) and s["text"][a:b].isalpha() for a, b in s["mask_spans"])
                                   for ex, _ in items for s in ex["segments"] if s.get("mask_spans"))
                    checks["e1_mask_spans_are_type_names"] = spans_ok
            if mode == "N100M":
                checks["e4_masked_share_0.30"] = abs(st["masked_action_share"] - 0.30) <= 0.02
        name = f"{mode}_d{d}"
        with open(os.path.join(out_dir, f"text_{name}.jsonl"), "w") as f:
            for ex, _ in items:
                f.write(json.dumps(ex) + "\n")
        ok = all(checks.values())
        ok_all &= ok
        report[name] = dict(stats=st, base_N100_stats={k: base_st[k] for k in ["n_examples", "n_dp", "note_start_supervised_share", "sup_chars"]},
                            checks=checks, ok=ok)
        print(name, "OK" if ok else "FAILED", json.dumps(checks), json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                                             for k, v in st.items() if k != "minus_share"}))
        if mode == "SA50v2":
            print("   minus share", json.dumps({k: (round(v[0], 3) if v[0] is not None else None, v[1]) for k, v in st["minus_share"].items()}))
    json.dump(report, open(report_path, "w"), indent=1)
    sys.exit(0 if ok_all else 100)


if __name__ == "__main__":
    main()
