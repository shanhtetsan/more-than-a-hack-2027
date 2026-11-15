"""Metrica Sports public sample games 1 and 2 as an external test set for the stoppage detector.

Same provider and frame rate (25 fps) as the competition tracking, different (anonymised) games.
Source: https://github.com/metrica-sports/sample-data (pinned revision below). The repository
asks users to acknowledge the source; it carries no licence file.

Ground truth for "dead ball" here is the tracking itself: in these files the ball has no
position while it is out of play. The detector under test sees player positions only, so the
label does not leak into it.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from possession.core import AWAY, HOME

REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"
RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/{rev}/data/Sample_Game_{g}/"
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("MTAH_EXTERNAL_DIR", ROOT / "external_data")) / "metrica_sample"
PITCH = (105.0, 68.0)  # the sample data is normalised 0-1; Metrica documents 105 x 68 m
GAMES = (1, 2)


def _fetch(game, name):
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(RAW.format(rev=REVISION, g=game) + name, tmp)
        tmp.rename(path)
    return path


def _csv(game, side):
    path = _fetch(game, f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv")
    return pd.read_csv(path, skiprows=2).iloc[:, 3:].to_numpy(float)


ON_BALL = ("PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST")  # events that name the holder
GOAL_END = ("KICK OFF", "CORNER KICK", "GOAL KICK")


def load_game(game, ball_dropout=0.0, seed=0):
    """Everything run_match needs for one sample game, plus ground truth.

    Ground truth: D where the ball has no position (the provider's own "out of play"); otherwise
    the team of the latest on-ball event (pass, shot, recovery, set piece, ball lost), which is
    the provider's human-collected event data for this other game.
    Restarts (stand-in for Smart Tagging): SET PIECE events and goal kicks, with their team.
    `ball_dropout` hides the ball on that share of frames, in bursts, to mimic automatic
    tracking that loses the ball (the competition data misses it on about 41% of frames).
    """
    home, away = _csv(game, "Home"), _csv(game, "Away")
    period = pd.read_csv(_fetch(game, f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"),
                         skiprows=2, usecols=[0]).iloc[:, 0].to_numpy()
    n = min(len(home), len(away))
    hp = home[:n, :-2].reshape(n, -1, 2) * PITCH
    ap = away[:n, :-2].reshape(n, -1, 2) * PITCH
    team = np.r_[np.full(hp.shape[1], HOME), np.full(ap.shape[1], AWAY)].astype(np.int8)
    ball = home[:n, -2:] * PITCH
    dead = np.isnan(ball).any(axis=1)

    events = pd.read_csv(_fetch(game, f"Sample_Game_{game}_RawEventsData.csv"))
    events = events.sort_values("Start Frame", kind="stable")
    holder = np.full(n, -1, dtype=np.int8)
    restarts, goal_restarts = {}, {}
    for ev in events.itertuples():
        frame = min(max(int(ev._5) - 1, 0), n - 1)  # "Start Frame" is 1-based
        side = HOME if ev.Team == "Home" else AWAY
        subtype = ev.Subtype if isinstance(ev.Subtype, str) else ""
        if ev.Type in ON_BALL:
            holder[frame] = side
        if ev.Type == "SET PIECE" or "GOAL KICK" in subtype:
            restarts[frame] = side
            if any(k in subtype for k in GOAL_END):
                goal_restarts[frame] = side
    idx = np.where(holder >= 0, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    filled = np.where(idx >= 0, holder[np.maximum(idx, 0)], -1)
    first = np.argmax(holder >= 0)
    filled[:first] = holder[first]

    blocks = []
    for p in np.unique(period[:n]):
        rows = np.flatnonzero(period[:n] == p)
        blocks.append((int(p), int(rows[0]), int(rows[-1])))
        restarts.setdefault(int(rows[0]), int(filled[rows[0]]))
    gt = np.where(dead, "D", np.where(filled == HOME, "H", "A")).astype(object)

    if ball_dropout:
        rng = np.random.default_rng(seed)
        ball = ball.copy()
        hidden = np.zeros(n, dtype=bool)
        while hidden.mean() < ball_dropout:  # bursts of 0.2 s to 6 s
            start = rng.integers(0, n)
            hidden[start:start + rng.integers(5, 150)] = True
        ball[hidden] = np.nan
    return dict(ball=ball, players=np.concatenate([hp, ap], axis=1).astype(np.float32),
                team=team, dead=dead, gt=gt, restarts=restarts, goal_restarts=goal_restarts,
                blocks=blocks, n_frames=n, length=PITCH[0], width=PITCH[1])
