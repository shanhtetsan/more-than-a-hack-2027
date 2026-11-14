"""SkillCorner open data (A-League 2024/25, MIT licence) as an external test set for v1.

Nothing here touches the competition data. It turns one SkillCorner match into exactly the
inputs `possession.core.run_match` takes, plus ground-truth labels, so the unchanged v1 rules can
be scored with the organisers' own metric on matches we never looked at while building them.

Source: https://github.com/SkillCorner/opendata (pinned revision below), MIT licence,
"Copyright (c) 2020 SkillCorner". Data is downloaded into external_data/ (gitignored).

How each input is built
- Tracking: 10 fps broadcast tracking, metres, origin at the centre spot. Shifted so the pitch
  runs 0..length, 0..width like the competition data, then linearly resampled to 25 fps so every
  frame-count parameter in config.py means the same duration. The ball is kept only where
  SkillCorner flags it `is_detected` (an off-screen ball is extrapolated, which our data never is).
- Labels (ground truth): SkillCorner "phases of play". A phase exists only while the ball is in
  play and names the team in possession, so frames inside a phase are that team's H/A and frames
  of a period outside every phase are D. Frames outside the periods are X (not graded).
- Restarts (stand-in for Smart Tagging): SkillCorner dynamic events of type player_possession
  whose start_type is a set-piece reception or interception (throw-in, free kick, goal kick,
  corner) give a restart frame and the team that receives it. A period's first frame is a
  kickoff with no team (the first confirmed control decides). The first possession after a goal
  is the kickoff after the goal. These are automatic events, like Smart Tagging, but anchored at
  the reception rather than a window around the restart.
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from possession.core import AWAY, HOME, NONE

REVISION = "4340d274572876239c154c90bc507a9b3250a656"
RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{REVISION}/data"
MEDIA = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{REVISION}/data"
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("MTAH_EXTERNAL_DIR", ROOT / "external_data" / "skillcorner"))

SRC_FPS, FPS = 10, 25
SET_PIECE_STARTS = ("throw_in", "free_kick", "goal_kick", "corner")
GOAL_END_STARTS = ("goal_kick", "corner")  # restarts that can follow a shot over the bar


def match_ids():
    return [int(m["id"]) for m in json.loads(_fetch(f"{RAW}/matches.json", "matches.json"))]


def _fetch(url, name):
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path.read_bytes()


def _files(mid):
    meta = json.loads(_fetch(f"{RAW}/matches/{mid}/{mid}_match.json", f"{mid}/match.json"))
    _fetch(f"{MEDIA}/matches/{mid}/{mid}_tracking_extrapolated.jsonl", f"{mid}/tracking.jsonl")
    _fetch(f"{RAW}/matches/{mid}/{mid}_dynamic_events.csv", f"{mid}/dynamic_events.csv")
    _fetch(f"{RAW}/matches/{mid}/{mid}_phases_of_play.csv", f"{mid}/phases_of_play.csv")
    return meta, CACHE / str(mid)


def to_target(src_frame):
    """10 fps source frame -> 25 fps target frame (nearest)."""
    return int(round(src_frame * FPS / SRC_FPS))


def resample(values):
    """(n_src, ...) at 10 fps -> (n_tgt, ...) at 25 fps by linear interpolation. NaN spreads to
    every target frame that leans on a missing source frame."""
    n_src = len(values)
    n_tgt = (n_src - 1) * FPS // SRC_FPS + 1
    pos = np.arange(n_tgt) * SRC_FPS / FPS
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_src - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    mixed = values[lo] * (1 - w) + values[hi] * w
    return np.where(w == 0, values[lo], mixed)  # exact source frame: no dependence on the next


def load_match(mid):
    """Everything run_match needs for one SkillCorner match, plus gt letters.

    Returns dict: ball (n,2), players (n,P,2), team (P,), restarts, goal_restarts, blocks,
    n_frames, gt (n,) letters, length, width.
    """
    meta, folder = _files(mid)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    side = {p["id"]: (HOME if p["team_id"] == home_id else AWAY) for p in meta["players"]}
    pids = sorted(side)
    col = {pid: k for k, pid in enumerate(pids)}

    rows = [json.loads(line) for line in open(folder / "tracking.jsonl")]
    n_src = max(r["frame"] for r in rows) + 1
    ball = np.full((n_src, 2), np.nan)
    players = np.full((n_src, len(pids), 2), np.nan)
    for r in rows:
        b = r["ball_data"]
        if b and b.get("is_detected") and b.get("x") is not None:
            ball[r["frame"]] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in r["player_data"] or ():
            k = col.get(p["player_id"])
            if k is not None and p.get("x") is not None:
                players[r["frame"], k] = (p["x"] + length / 2, p["y"] + width / 2)

    ball, players = resample(ball), resample(players)
    n = len(ball)

    blocks = [(int(p["period"]), to_target(p["start_frame"]), min(to_target(p["end_frame"]), n - 1))
              for p in meta["match_periods"]]

    gt = np.full(n, "X", dtype=object)
    for _, first, last in blocks:
        gt[first:last + 1] = "D"
    phases = pd.read_csv(folder / "phases_of_play.csv")
    for ph in phases.itertuples():
        letter = "H" if ph.team_in_possession_id == home_id else "A"
        gt[to_target(ph.frame_start):to_target(ph.frame_end) + 1] = letter

    restarts, goal_restarts = {first: NONE for _, first, _ in blocks}, {}
    events = pd.read_csv(folder / "dynamic_events.csv", low_memory=False)
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    after_goal = pp["game_interruption_before"].fillna("").str.startswith("goal")
    for ev, goal in zip(pp.itertuples(), after_goal, strict=True):
        kind = ev.start_type if isinstance(ev.start_type, str) else ""
        team = HOME if ev.team_id == home_id else AWAY
        frame = to_target(ev.frame_start)
        if goal:  # kickoff after a goal
            restarts[frame] = goal_restarts[frame] = team
        elif kind.startswith(SET_PIECE_STARTS):
            restarts[frame] = team
            if kind.startswith(GOAL_END_STARTS):
                goal_restarts[frame] = team
    return dict(ball=ball, players=players, team=np.array([side[p] for p in pids], np.int8),
                restarts=restarts, goal_restarts=goal_restarts, blocks=blocks, n_frames=n,
                gt=gt, length=length, width=width)
