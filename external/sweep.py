"""The parameter study behind v2, on SkillCorner open data only (never the competition match).

    uv run --frozen python -m external.sweep

1. Split the 20 matches into development and held-out halves (alternating by sorted id).
2. Score a 24-setting grid on the development matches.
3. Score v1 and v2 once on the held-out matches, with a paired bootstrap on the per-match gain.
"""

import itertools
import sys

import numpy as np

from . import evaluate, skillcorner

GRID = {"MIN_CONFIRMED_FRAMES": (4, 8, 12, 16),
        "CONTROL_RADIUS_M": (1.0, 1.5, 2.0),
        "MIN_TEAM_RUN_FRAMES": (25, 50)}
V1 = {"MIN_CONFIRMED_FRAMES": "8", "CONTROL_RADIUS_M": "1.5", "MIN_TEAM_RUN_FRAMES": "25"}
V2 = {"MIN_CONFIRMED_FRAMES": "12", "CONTROL_RADIUS_M": "2.0", "MIN_TEAM_RUN_FRAMES": "50"}


def spell(mids, overrides):
    return np.array([evaluate.score_match(m, overrides)["spell_f1_mean"] for m in mids])


def main():
    ids = sorted(skillcorner.match_ids())
    dev, held = ids[0::2], ids[1::2]
    print("development:", dev)
    print("held-out:   ", held)

    print("\ngrid on development matches (mean spellF1):")
    rows = []
    for combo in itertools.product(*GRID.values()):
        overrides = {k: str(v) for k, v in zip(GRID, combo, strict=True)}
        rows.append((spell(dev, overrides).mean(), combo))
        print(f"  {rows[-1][0]:.4f}  {dict(zip(GRID, combo, strict=True))}", flush=True)
    print("best five:", [(round(float(s), 4), c) for s, c in sorted(rows, reverse=True)[:5]])

    v1, v2 = spell(held, V1), spell(held, V2)
    gain = v2 - v1
    rng = np.random.default_rng(0)
    boot = [rng.choice(gain, len(gain)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nheld-out: v1 {v1.mean():.4f}  v2 {v2.mean():.4f}  "
          f"v2 better on {int((gain > 0).sum())}/{len(gain)}  "
          f"mean gain {gain.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
