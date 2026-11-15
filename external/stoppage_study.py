"""The v2 -> v3 study: a stoppage detector from collective player speed, on external data only.

    uv run --frozen python -m external.stoppage_study

Development: 10 SkillCorner matches and Metrica sample game 1 (clean, and with the ball hidden on
40% of frames). Held-out: the other 10 SkillCorner matches and Metrica sample game 2.

Note on the Metrica sample games: their ball has no position while out of play, so the ball-out
rule can never fire there and v2 has almost no way to say D. They measure how well the detector
finds dead ball on same-provider tracking; the v2 -> v3 gap on them overstates the gain.
"""

import itertools
import sys

import numpy as np

from . import evaluate, metrica_sample, skillcorner

OFF = {"STOPPAGE_SPEED_MPS": "0"}
GRID = {"STOPPAGE_SPEED_MPS": (1.2, 1.35, 1.5),
        "STOPPAGE_BALL_MPS": (0.0, 2.0, 3.0),
        "STOPPAGE_MIN_FRAMES": (75, 125)}
DROPOUT = 0.4


def skill(mids, overrides, key="spell_f1_mean"):
    return np.array([evaluate.score_match(m, overrides)[key] for m in mids])


def line(name, r):
    return (f"{name}: spellF1 {r['spell_f1_mean']:.3f}  macroF1 {r['macro_f1']:.3f}  "
            f"possessions {r['poss']}/{r['gt_poss']}  dead share {r['dead']:.2f} "
            f"(truth {r['gt_dead']:.2f})")


def main():
    ids = sorted(skillcorner.match_ids())
    dev, held = ids[0::2], ids[1::2]
    game1 = {d: metrica_sample.load_game(1, d) for d in (0.0, DROPOUT)}

    print("development, detector off (v2):")
    print(f"  SkillCorner dev mean spellF1 {skill(dev, OFF).mean():.4f}")
    for d, m in game1.items():
        print("  " + line(f"Metrica game 1, ball dropout {d}", evaluate.score_arrays(m, OFF)))

    print("\ngrid on development data (spellF1: SkillCorner dev | game 1 clean | game 1 dropout):")
    for combo in itertools.product(*GRID.values()):
        o = {k: str(v) for k, v in zip(GRID, combo, strict=True)}
        scores = [skill(dev, o).mean()] + [evaluate.score_arrays(m, o)["spell_f1_mean"]
                                           for m in game1.values()]
        cells = "  ".join(f"{s:.3f}" for s in scores)
        print(f"  {cells}   {dict(zip(GRID, combo, strict=True))}", flush=True)

    print("\nheld-out, v2 (detector off) against v3 (config.py as committed):")
    v2, v3 = skill(held, OFF), skill(held, {})
    gain = v3 - v2
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([rng.choice(gain, len(gain)).mean() for _ in range(10000)],
                           [2.5, 97.5])
    print(f"  SkillCorner held-out: v2 {v2.mean():.4f}  v3 {v3.mean():.4f}  "
          f"v3 better on {int((gain > 0).sum())}/{len(gain)}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"macroF1 {skill(held, OFF, 'macro_f1').mean():.3f} -> "
          f"{skill(held, {}, 'macro_f1').mean():.3f}")
    for d in (0.0, DROPOUT):
        m = metrica_sample.load_game(2, d)
        print("  " + line(f"Metrica game 2, ball dropout {d}, v2", evaluate.score_arrays(m, OFF)))
        print("  " + line(f"Metrica game 2, ball dropout {d}, v3", evaluate.score_arrays(m, {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
