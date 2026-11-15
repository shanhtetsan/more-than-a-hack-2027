"""Score the v1 rules, unchanged, on SkillCorner open-data matches with the official metric.

    uv run --frozen python -m external.evaluate                 # v1 as configured
    uv run --frozen python -m external.evaluate --set MIN_CONFIRMED_FRAMES=12 CONTROL_RADIUS_M=2.0

--set overrides config values for this run only (for a sweep on external data). The competition
data is never read here.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

import baseline  # noqa: E402
import possession_metrics as pm  # noqa: E402

from possession import config, core  # noqa: E402

from . import skillcorner  # noqa: E402

KEYS = ("spell_f1_mean", "macro_f1")


def make_cfg(length, width, overrides):
    cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    cfg.PITCH_LENGTH_M, cfg.PITCH_WIDTH_M = length, width
    for key, value in overrides.items():
        setattr(cfg, key, type(getattr(config, key))(value))
    return cfg


def baseline_labels(m, window):
    """The organisers' baseline on this match: nearest player to the seen ball, then a +/- window
    majority vote (starter/baseline.smooth). X outside the periods."""
    d = np.hypot(m["players"][..., 0] - m["ball"][:, None, 0],
                 m["players"][..., 1] - m["ball"][:, None, 1])
    seen = ~np.isnan(m["ball"]).any(axis=1)
    near = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
    codes = np.where(seen, m["team"][near], -1)
    labels = np.full(m["n_frames"], "X", dtype="<U1")
    for _, first, last in m["blocks"]:
        labels[first:last + 1] = baseline.smooth(codes[first:last + 1], window)
    return labels


def score_match(mid, overrides, window=None, cache={}):  # noqa: B006  (per-process cache)
    if mid not in cache:
        cache[mid] = skillcorner.load_match(mid)
    m = cache[mid]
    if window is not None:
        labels = baseline_labels(m, window)
    else:
        cfg = make_cfg(m["length"], m["width"], overrides)
        labels, _ = core.run_match(m["ball"], m["players"], m["team"], m["restarts"],
                                   m["n_frames"], m["blocks"], cfg, m["goal_restarts"])
    gt = pm.label_codes(m["gt"])
    res = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
    return {k: float(res[k]) for k in KEYS} | {"poss": res["n_pred_spells"],
                                                "gt_poss": res["n_gt_spells"]}


def score_arrays(m, overrides):
    """Official metrics for one loaded match dict (skillcorner or metrica_sample)."""
    cfg = make_cfg(m["length"], m["width"], overrides)
    labels, _ = core.run_match(m["ball"], m["players"], m["team"], m["restarts"], m["n_frames"],
                               m["blocks"], cfg, m["goal_restarts"])
    res = pm.evaluate(pm.label_codes(m["gt"]), pm.label_codes(labels), bootstrap_n=0)
    graded = m["gt"] != "X"
    return {"spell_f1_mean": float(res["spell_f1_mean"]), "macro_f1": float(res["macro_f1"]),
            "poss": res["n_pred_spells"], "gt_poss": res["n_gt_spells"],
            "dead": float(np.mean(labels[graded] == "D")),
            "gt_dead": float(np.mean(m["gt"][graded] == "D"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", type=int, nargs="*", help="SkillCorner match ids (default: all)")
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE config overrides")
    ap.add_argument("--baseline", type=int, help="score the organisers' baseline with this window")
    args = ap.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    mids = args.matches or skillcorner.match_ids()

    rows = []
    for mid in mids:
        r = score_match(mid, overrides, args.baseline)
        rows.append(r)
        print(f"{mid}  spellF1 {r['spell_f1_mean']:.4f}  macroF1 {r['macro_f1']:.4f}  "
              f"poss {r['poss']}  gt {r['gt_poss']}", flush=True)
    for k in KEYS:
        vals = np.array([r[k] for r in rows])
        print(f"mean {k}: {vals.mean():.4f}  (min {vals.min():.4f}, max {vals.max():.4f}, "
              f"n={len(vals)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
