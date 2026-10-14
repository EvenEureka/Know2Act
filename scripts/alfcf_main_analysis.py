"""PREREG Addendum 20: study-level analysis of the ALF-CF main study on the TEST split. Pure python (no scipy).
Input: a manifest TSV (arm, draw, readouts.json of eval v3 on test, produced by alfcf_eval_v3_analysis.py).
Unit of inference = seed (construction draw + data order). Per model: S = S_flip@first, learnable, primary population
(held-out floor plans + valid_unseen); valid_seen reported separately.
Primary: Welch t, N100 vs NQ50op, two-sided, alpha 0.05; difference, Welch 95% CI; Fix 1 reading band:
  S(NQ50op) < 0.15 -> "reliability"; S(NQ50op) > Sbar(N100) - 0.15 -> "not-writing"; else "mixed".
Secondaries H1-H8, Holm over the family (alpha 0.05):
  H1 NQ50op vs NQ50; H2 SA50v2 vs N100; H3 N100M vs N100 (S); H4 N100M vs N100 (P(conv|remove), first dp, learnable moved);
  H5 NQ25op vs N100; H6 Res@first in N100 vs 0 (one-sample); H7 cf P(first visit = foil top) N100 vs NQ50op;
  H8 later-position 2x2 removal effect in N100 vs 0 (one-sample).
usage: python alfcf_main_analysis.py <manifest.tsv> <out.json>
"""
import json
import math
import sys
from collections import defaultdict


def betacf(a, b, x):
    m, eps = 300, 3e-14
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for k in range(1, m + 1):
        m2 = 2 * k
        aa = k * (b - k) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / (c if abs(c) > 1e-300 else 1e-300)
        h *= d * c
        aa = -(a + k) * (qab + k) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / (c if abs(c) > 1e-300 else 1e-300)
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * betacf(a, b, x) / a
    return 1 - math.exp(lb) * betacf(b, a, 1 - x) / b


def p_two_sided(t, df):
    return ibeta(df / 2, 0.5, df / (df + t * t))


def t_quantile(df, p=0.05):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if p_two_sided(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean(v):
    return sum(v) / len(v)


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")


def welch(a, b):
    """a - b"""
    va, vb, na, nb = var(a), var(b), len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    diff = mean(a) - mean(b)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se > 0 else float("inf")
    t = diff / se if se > 0 else float("inf")
    p = p_two_sided(abs(t), df) if se > 0 else (1.0 if diff == 0 else 0.0)
    q = t_quantile(df) if se > 0 else 0.0
    return dict(diff=diff, ci95=[diff - q * se, diff + q * se], t=t, df=df, p=p, n=[na, nb], means=[mean(a), mean(b)],
                sds=[math.sqrt(va), math.sqrt(vb)])


def one_sample(a):
    n, m, s = len(a), mean(a), math.sqrt(var(a))
    se = s / math.sqrt(n)
    t = m / se if se > 0 else float("inf")
    df = n - 1
    q = t_quantile(df)
    p = p_two_sided(abs(t), df) if se > 0 else (1.0 if m == 0 else 0.0)
    return dict(mean=m, ci95=[m - q * se, m + q * se], t=t, df=df, p=p, n=n, sd=s)


def main():
    rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip() and not l.startswith("arm\t")]
    R = defaultdict(dict)
    for arm, draw, path in rows:
        R[arm][int(draw)] = json.load(open(path))

    def vals(arm, fn, pop="primary"):
        return [fn(r[pop]) for d, r in sorted(R[arm].items())]

    S = lambda b: b["first_L"]["S_flip"][0]
    out = {"n_seeds": {a: sorted(R[a]) for a in R}}
    sN, sO = vals("N100", S), vals("NQ50op", S)
    prim = welch(sN, sO)
    sbar = mean(sN)
    so = mean(sO)
    band = "reliability" if so < 0.15 else ("not-writing" if so > sbar - 0.15 else "mixed")
    out["primary"] = dict(contrast="N100 - NQ50op, S_flip@first learnable, primary population", **prim,
                          significant=prim["p"] < 0.05, fix1_reading=band)
    H = {}
    if "NQ50" in R:
        H["H1_NQ50op_vs_NQ50"] = welch(sO, vals("NQ50", S))
    if "SA50v2" in R:
        H["H2_SA50v2_vs_N100"] = welch(vals("SA50v2", S), sN)
    if "N100M" in R:
        H["H3_N100M_vs_N100_S"] = welch(vals("N100M", S), sN)
        pc = lambda b: b["first_L"]["P_conv"]["remove"]
        H["H4_N100M_vs_N100_Pconv_remove"] = welch(vals("N100M", pc), vals("N100", pc))
    if "NQ25op" in R:
        H["H5_NQ25op_vs_N100"] = welch(vals("NQ25op", S), sN)
    H["H6_Res_first_N100"] = one_sample(vals("N100", lambda b: b["Res_first_U_minus_L"][0]))
    cfv = lambda b: b["counterfactual_rollouts"]["cf_foil|L"]["P_first_visit_eq_foil_top"]
    H["H7_cf_first_visit_foil_N100_vs_NQ50op"] = welch(vals("N100", cfv), vals("NQ50op", cfv))
    H["H8_2x2_removal_effect_N100"] = one_sample(vals("N100", lambda b: b["later_2x2"]["L"]["removed_minus_kept"][0]))
    order = sorted(H, key=lambda k: H[k]["p"])
    m, stop = len(order), False
    for i, k in enumerate(order):
        thr = 0.05 / (m - i)
        H[k]["holm_threshold"] = thr
        H[k]["holm_reject"] = (not stop) and H[k]["p"] <= thr
        if not H[k]["holm_reject"]:
            stop = True
    out["secondary"] = H
    cov = {}
    keys = {"S_flip_first_L": S,
            "S_flip_first_U": lambda b: b["first_U"]["S_flip"][0],
            "Res_first": lambda b: b["Res_first_U_minus_L"][0],
            "note_writing_rate": lambda b: b["note_writing_rate"],
            "note_top1_hit": lambda b: b["first_L"]["note_top1_hit"],
            "follow_intact": lambda b: b["first_L"]["follow_intact"],
            "P_conv_remove": lambda b: b["first_L"]["P_conv"]["remove"],
            "presence_effect": lambda b: b["first_L"]["presence_effect"][0],
            "S_flip_vs_presence": lambda b: b["first_L"]["S_flip_vs_presence"][0],
            "S_flip_follow_normalised": lambda b: b["first_L"]["S_flip_follow_normalised"],
            "S_random": lambda b: b["first_L"]["S_random"][0],
            "plausibility_slope": lambda b: b["first_L"]["plausibility"]["slope_most_minus_least"],
            "removal_null": lambda b: b["first_L"]["removal_null"][0],
            "cf_first_visit_foil": cfv,
            "cf_visits_foil_minus_true": lambda b: b["counterfactual_rollouts"]["visits_foil_minus_true|L"],
            "two_by_two_removal_effect": lambda b: b["later_2x2"]["L"]["removed_minus_kept"][0],
            "success": lambda b: b["success"]}
    for pop in ["primary", "valid_seen"]:
        cov[pop] = {}
        for arm in R:
            cov[pop][arm] = {}
            for k, fn in keys.items():
                v = []
                for d, r in sorted(R[arm].items()):
                    try:
                        x = fn(r[pop])
                        v.append(x if x == x else None)
                    except (KeyError, TypeError, ZeroDivisionError):
                        v.append(None)
                vv = [x for x in v if x is not None]
                cov[pop][arm][k] = dict(per_seed=v, mean=mean(vv) if vv else None,
                                        sd=math.sqrt(var(vv)) if len(vv) > 1 else None)
    out["per_arm"] = cov
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out["primary"], indent=1))
    for k in order:
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in H[k].items() if a in ("diff", "mean", "p", "holm_reject", "ci95")})


if __name__ == "__main__":
    main()
