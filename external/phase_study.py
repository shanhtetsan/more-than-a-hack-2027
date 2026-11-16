"""The v3 -> v4 study: Smart Tagging phases as possession evidence.

    uv run --frozen python -m external.phase_study

No external dataset ships Metrica's Smart Tagging, so this cannot be a held-out test of the real
tags. It answers a narrower question on external games: IF automatic phase tags of a given
quality exist, does using them the way v4 does help or hurt, and how good must they be?

Simulated tags are cut from each external game's own possession labels and then degraded:
only `COVER` of possessions get a phase, its ends are jittered by about a second, half the phases
are "defending" codes that name the team WITHOUT the ball, and a share `error` name the wrong
team. v4 must work out by itself which codes mean what (core.phase_polarity).
"""

import sys

import numpy as np

from possession.core import AWAY, HOME, run_lengths

from . import evaluate, metrica_sample, skillcorner

COVER, JITTER_FRAMES, MIN_SPELL = 0.6, 25, 75
ERRORS = (0.0, 0.1, 0.2, 0.3, 0.4)
MIN_FRAMES = (50, 75, 125)
OFF = {"PHASE_MIN_FRAMES": "0"}


def simulated_phases(gt, error, seed):
    rng = np.random.default_rng(seed)
    starts, lengths, values = run_lengths(np.asarray(gt, dtype="<U1"))
    out = []
    for start, length, value in zip(starts, lengths, values, strict=True):
        if value not in "HA" or length < MIN_SPELL or rng.random() > COVER:
            continue
        holder = HOME if value == "H" else AWAY
        if rng.random() < error:
            holder = AWAY if holder == HOME else HOME
        first = int(start + rng.normal(0, JITTER_FRAMES))
        end = int(start + length + rng.normal(0, JITTER_FRAMES))
        if end <= first:
            continue
        if rng.random() < 0.5:
            out.append(("attacking phase", first, end, holder))
        else:
            out.append(("defending phase", first, end, AWAY if holder == HOME else HOME))
    return out


def mean_score(matches, overrides, error=None):
    scores = []
    for k, m in enumerate(matches):
        phases = None if error is None else simulated_phases(m["gt"], error, seed=k)
        scores.append(evaluate.score_arrays(m, overrides, phases)["spell_f1_mean"])
    return float(np.mean(scores))


def report(name, matches, settings):
    print(f"\n{name}: v3 (no phases) {mean_score(matches, OFF):.4f}")
    print("  tag error   " + "   ".join(f"min {m:>3} fr" for m in settings))
    for error in ERRORS:
        cells = [mean_score(matches, {"PHASE_MIN_FRAMES": str(m)}, error) for m in settings]
        print(f"  {error:>8.0%}   " + "   ".join(f"{c:>10.4f}" for c in cells), flush=True)


def main():
    ids = sorted(skillcorner.match_ids())
    dev = [evaluate.loaded(m) for m in ids[0::2]]
    held = [evaluate.loaded(m) for m in ids[1::2]]
    game = {g: metrica_sample.load_game(g, 0.4) for g in metrica_sample.GAMES}

    print("development (choose the minimum phase run):")
    report("SkillCorner dev (10)", dev, MIN_FRAMES)
    report("Metrica sample game 1, ball hidden 40%", [game[1]], MIN_FRAMES)

    print("\nheld-out, with config.py as committed:")
    from possession import config
    committed = (config.PHASE_MIN_FRAMES,) if config.PHASE_MIN_FRAMES else (75,)
    report("SkillCorner held-out (10)", held, committed)
    report("Metrica sample game 2, ball hidden 40%", [game[2]], committed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
