"""Figures for the README and technical document, from EXTERNAL games only.

    uv run --frozen python -m external.figures      # writes docs/figures/*.png

Nothing here reads the competition data, so the images are safe to publish.
  1. timeline.png  four minutes of Metrica sample game 2 (held-out): provider labels, v2, v3
  2. versions.png  mean spellF1 on the 10 held-out SkillCorner matches, baseline -> v3
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from possession import core  # noqa: E402

from . import evaluate, metrica_sample, skillcorner  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
COLOUR = {"H": "#2a78d6", "A": "#eb6834", "D": "#d9d8d2", "X": SURFACE}
NAME = {"H": "Home in possession", "A": "Away in possession", "D": "Dead ball"}
OFF = {"STOPPAGE_SPEED_MPS": "0"}
V1 = OFF | {"MIN_CONFIRMED_FRAMES": "8", "CONTROL_RADIUS_M": "1.5", "MIN_TEAM_RUN_FRAMES": "25"}
FPS = 25


def labels_for(match, overrides):
    cfg = evaluate.make_cfg(match["length"], match["width"], overrides)
    return core.run_match(match["ball"], match["players"], match["team"], match["restarts"],
                          match["n_frames"], match["blocks"], cfg, match["goal_restarts"])[0]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0, labelsize=10)


def timeline(start_min=20, minutes=4):
    match = metrica_sample.load_game(2)
    first = match["blocks"][0][1] + start_min * 60 * FPS
    window = slice(first, first + minutes * 60 * FPS)
    rows = [("Provider labels", match["gt"][window]),
            ("v2 (ball out only)", labels_for(match, OFF)[window]),
            ("v3 (+ stoppages)", labels_for(match, {})[window])]

    fig, ax = plt.subplots(figsize=(11, 2.9), dpi=160, facecolor=SURFACE)
    style(ax)
    for y, (_, letters) in enumerate(reversed(rows)):
        starts, lengths, values = core.run_lengths(np.asarray(letters, dtype="<U1"))
        for s, n, v in zip(starts, lengths, values, strict=True):
            ax.barh(y, n / FPS, left=s / FPS, height=0.62, color=COLOUR[v],
                    edgecolor=SURFACE, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [name for name, _ in reversed(rows)], color=INK, fontsize=11)
    ax.set_xlim(0, minutes * 60)
    ax.set_xticks(range(0, minutes * 60 + 1, 30),
                  [f"{start_min + t // 60}:{t % 60:02d}" for t in range(0, minutes * 60 + 1, 30)])
    ax.set_xlabel("match clock (min:sec)", color=MUTED, fontsize=10)
    ax.set_title("Four minutes of a held-out Metrica sample game: what each version labels",
                 loc="left", color=INK, fontsize=12.5, pad=28)
    ax.legend(handles=[Patch(color=COLOUR[k], label=NAME[k]) for k in "HAD"], ncols=3,
              loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10,
              labelcolor=MUTED, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "timeline.png", facecolor=SURFACE)
    plt.close(fig)


def versions():
    held = sorted(skillcorner.match_ids())[1::2]

    def mean(overrides, window=None):
        return float(np.mean([evaluate.score_match(m, overrides, window)["spell_f1_mean"]
                              for m in held]))

    bars = [("Organisers' baseline\n(smooth50)", mean({}, 50)),
            ("v1\nliterature defaults", mean(V1)),
            ("v2\nparameters set on\nexternal data", mean(OFF)),
            ("v3 (submitted)\n+ stoppage detector", mean({}))]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=160, facecolor=SURFACE)
    style(ax)
    x = np.arange(len(bars))
    values = [v for _, v in bars]
    ax.bar(x, values, width=0.5, color=["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6"])
    for xi, v in zip(x, values, strict=True):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(x, [name for name, _ in bars], color=INK, fontsize=9.5)
    ax.set_ylim(0, max(values) + 0.07)
    ax.set_yticks(np.arange(0, max(values) + 0.07, 0.1))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("mean spellF1", color=MUTED, fontsize=10)
    ax.set_title("Held-out external matches (10 SkillCorner A-League games)",
                 loc="left", color=INK, fontsize=12.5, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "versions.png", facecolor=SURFACE)
    plt.close(fig)
    return bars


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timeline()
    for name, value in versions():
        print(f"{value:.4f}  {name.splitlines()[0]}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
