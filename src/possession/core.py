"""The v1 possession rules, as pure functions on arrays. No file I/O, so all of it is unit-tested
on synthetic data (tests/test_possession.py).

Internal codes: HOME=0, AWAY=1, DEAD=2, NOX=3 (abstain), NONE=-1 (no answer yet / no control).
"""

from collections import Counter

import numpy as np

HOME, AWAY, DEAD, NOX, NONE = 0, 1, 2, 3, -1
LETTERS = np.array(["H", "A", "D", "X"])
SIDE_CODE = {"H": HOME, "A": AWAY}


# --- (1) units ------------------------------------------------------------------------------

def to_metres(xy, length, width):
    """Normalised 0-1 coordinates (..., 2) -> metres. NaN stays NaN."""
    xy = np.asarray(xy, dtype=float)
    return xy * np.array([length, width])


# --- (2) ball gaps --------------------------------------------------------------------------

def interpolate_ball(ball, max_gap):
    """Linearly fill interior NaN gaps of at most `max_gap` frames in an (n, 2) ball track.

    A gap is filled only when it has an observed frame on both sides, so the start and end of
    the track are never extrapolated. Longer gaps stay NaN in full (not partially filled).
    """
    ball = np.array(ball, dtype=float, copy=True)
    missing = np.isnan(ball).any(axis=1)
    ball[missing] = np.nan
    if max_gap <= 0 or not missing.any() or missing.all():
        return ball

    seen = np.flatnonzero(~missing)
    # Consecutive observed frames with a hole between them.
    left, right = seen[:-1], seen[1:]
    gap = right - left - 1
    fill = (gap > 0) & (gap <= max_gap)
    for lo, hi in zip(left[fill], right[fill], strict=True):
        t = (np.arange(lo + 1, hi) - lo) / (hi - lo)
        ball[lo + 1:hi] = ball[lo] + t[:, None] * (ball[hi] - ball[lo])
    return ball


# --- (3) per-frame control ------------------------------------------------------------------

def frame_control(ball, players, team, control_radius, duel_radius):
    """Team in control on each frame, or NONE.

    ball (n, 2) and players (n, P, 2) in metres, NaN where unseen; team (P,) of HOME/AWAY.
    A team controls the frame when its nearest player is within `control_radius` of the ball and
    no opponent is within `duel_radius`. If both teams qualify (both nearest players beyond the
    duel radius but inside the control radius), the nearer team wins.
    """
    ball = np.asarray(ball, dtype=float)
    players = np.asarray(players, dtype=float)
    team = np.asarray(team)
    dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)

    def nearest(side):
        cols = team == side
        if not cols.any():
            return np.full(len(ball), np.inf)
        return dist[:, cols].min(axis=1)

    d_home, d_away = nearest(HOME), nearest(AWAY)
    home = (d_home <= control_radius) & (d_away > duel_radius)
    away = (d_away <= control_radius) & (d_home > duel_radius)

    out = np.full(len(ball), NONE, dtype=np.int8)
    out[home] = HOME
    out[away] = AWAY
    both = home & away
    out[both] = np.where(d_home[both] <= d_away[both], HOME, AWAY)
    return out


# --- (4) confirmed control ------------------------------------------------------------------

def run_lengths(codes):
    """(starts, lengths, values) of maximal runs of equal values."""
    codes = np.asarray(codes)
    if len(codes) == 0:
        return np.array([], int), np.array([], int), codes[:0]
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(codes)])))
    return starts, lengths, codes[starts]


def confirmed_starts(control, min_frames):
    """Per frame: the team whose confirmed control run *starts* on that frame, else NONE.

    A run is confirmed when the same team controls at least `min_frames` consecutive frames.
    Marking the first frame of the run means possession switches from where control began, not
    from the frame it was confirmed.
    """
    out = np.full(len(control), NONE, dtype=np.int8)
    starts, lengths, values = run_lengths(control)
    keep = (values != NONE) & (lengths >= min_frames)
    out[starts[keep]] = values[keep]
    return out


# --- (5) dead ball --------------------------------------------------------------------------

def ball_out(ball, length, width, margin=0.0):
    """True where the ball is seen strictly outside the pitch (+ margin). False where unseen."""
    ball = np.asarray(ball, dtype=float)
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return ((x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin))


def ball_in(ball, length, width, margin=0.0):
    """True where the ball is seen on the pitch. False where unseen."""
    ball = np.asarray(ball, dtype=float)
    seen = ~np.isnan(ball).any(axis=1)
    return seen & ~ball_out(ball, length, width, margin)


def goal_mouth_exit(ball, length, width, out_margin, half_width):
    """True where the ball is beyond a goal line (by more than `out_margin`) and within
    `half_width` of the goal's centre line across the pitch: a possible goal.

    The tracking has no ball height, so a shot over the crossbar looks the same as a goal.
    """
    ball = np.asarray(ball, dtype=float)
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        beyond = (x < -out_margin) | (x > length + out_margin)
        return beyond & (np.abs(y - width / 2) <= half_width)


def sustained_out_starts(out, min_frames):
    """True on the first frame of every run of >= `min_frames` consecutive out frames.

    A missing-ball frame is not an out frame, so it breaks the run. Marking the first frame
    means D starts where the ball left the pitch, not where the run was confirmed.
    """
    out = np.asarray(out, dtype=bool)
    mark = np.zeros(len(out), dtype=bool)
    starts, lengths, values = run_lengths(out)
    mark[starts[values & (lengths >= min_frames)]] = True
    return mark


def anchor_frame(event, anchor, blocks=None):
    """The frame a restart tag fires on: its "start", "mid" or "end" frame, except that a tag
    whose window [start_frame, end_frame] overlaps the first graded frame of a period fires on
    that frame. Otherwise a kickoff tag whose window opens before the first graded frame would
    anchor outside the period and never end its period-start D.
    """
    start, end = int(event["start_frame"]), int(event["end_frame"])
    for _, first, _ in blocks or ():
        if start <= first <= end:
            return first
    return {"start": start, "end": end, "mid": (start + end) // 2}[anchor]


def restart_events(events, code, anchor, team_to_side, types=None, blocks=None):
    """{frame: HOME/AWAY/NONE} for every Smart Tagging restart.

    `events` are dicts with code, start_frame, end_frame and tags (starter/load_smart_tagging.py).
    Codes, Type values and team names are matched case-insensitively. A team tag that maps to no
    side ("N/A") gives NONE: the possessor is then decided by the next confirmed control.
    `types`, if given, keeps only restarts whose Type tag is one of them (e.g. the restarts
    that can follow a possible goal). `blocks`, if given, are the graded periods used by
    anchor_frame.
    """
    team_to_side = {k.strip().lower(): v for k, v in team_to_side.items()}
    wanted = None if types is None else {t.strip().lower() for t in types}
    out = {}
    for event in sorted(events, key=lambda e: e["start_frame"]):
        if (event.get("code") or "").strip().lower() != code.strip().lower():
            continue
        kind = ((event.get("tags") or {}).get("Type") or "").strip().lower()
        if wanted is not None and kind not in wanted:
            continue
        frame = anchor_frame(event, anchor, blocks)
        tag = ((event.get("tags") or {}).get("Team") or "").strip().lower()
        out[frame] = SIDE_CODE.get(team_to_side.get(tag), NONE)
    return out


def restart_types(events, code, anchor, blocks=None):
    """{frame: Type tag (lower case, "" if none)} for every restart, keyed like restart_events."""
    out = {}
    for event in sorted(events, key=lambda e: e["start_frame"]):
        if (event.get("code") or "").strip().lower() != code.strip().lower():
            continue
        frame = anchor_frame(event, anchor, blocks)
        out[frame] = ((event.get("tags") or {}).get("Type") or "").strip().lower()
    return out


# --- state machine: (4) + (5) together ------------------------------------------------------

def new_stats():
    """Counters assign_states fills in when given one. Counts and tag frames only."""
    return {
        # (cause, ending) -> number of D segments, before step 6 merging. cause is
        # "period start" / "ball out" / "possible goal"; ending is "restart tag" /
        # "control exit" / "goal restart tag" (kickoff, goal kick, corner) / "never ended".
        "d_segments": Counter(),
        "used_restarts": set(),      # restart-tag frames that ended a D
        "control_switches": 0,       # possession changes by confirmed control while live
        "period_start_frames": [],   # per period: frames of D before its first restart
    }


def assign_states(confirmed, out_start, inside, restarts, first, last,
                  goal_mouth=None, goal_restarts=None, stats=None):
    """Labels for frames first..last (inclusive) of one graded period, as codes.

    `out_start` marks the first frame of each sustained exit from the pitch
    (sustained_out_starts); `inside` marks frames where the ball is seen within the lines.

    The period starts dead (before its kickoff) and that D ends only at a restart tag. Then:
      - live -> dead on an out_start frame. The trigger only re-arms once the ball has been seen
        inside again after a restart, so a throw-in taken from outside the line does not end
        the restart it belongs to.
      - a stoppage caused by the ball going out ends at whichever comes first:
          * a restart tag: possession goes to the tagged team. An untagged restart (NONE)
            leaves the possessor pending until the next confirmed control, which is then
            backfilled to the restart;
          * the ball seen back inside the lines, followed (same frame or later) by the start of
            a confirmed control run: possession goes to that team from that frame.
      - except a possible goal: when `goal_mouth` is True on the out_start frame, the D ends
        only at the next frame in `goal_restarts` (the kickoff, goal-kick and corner subset
        of `restarts`). No other restart tag and no early exit ends it.
      - while live, only the *other* team's confirmed control changes possession; frames with
        no control, contested frames and missing-ball frames hold the last state.
    A pending possessor never resolved by a control falls back to the possessor before the
    stoppage, or NOX if there was none.

    `stats`, if given (new_stats()), is filled with counts of what happened; labels are the
    same with or without it.
    """
    n = last - first + 1
    labels = np.full(n, NONE, dtype=np.int8)
    live, armed = False, False
    after_out, back_in = False, False  # dead because the ball went out / seen inside since
    possible_goal = False
    goal_restarts = {} if goal_restarts is None else goal_restarts
    possessor, previous = NONE, NONE
    pending_from = None
    cause = "period start"

    def ended(how, frame=None):
        if stats is None:
            return
        stats["d_segments"][(cause, how)] += 1
        if frame is not None:
            stats["used_restarts"].add(frame)
        if cause == "period start":
            stats["period_start_frames"].append(i)

    def resolve(upto, team):
        nonlocal pending_from
        if pending_from is not None:
            labels[pending_from:upto] = team if team != NONE else NOX
            pending_from = None

    for i in range(n):
        f = first + i
        if not live and possible_goal:
            if f in goal_restarts:
                ended("goal restart tag", f)
                live, armed, possible_goal = True, False, False
                possessor = goal_restarts[f]
                if possessor == NONE:
                    pending_from = i
        elif not live:
            if f in restarts:
                ended("restart tag", f)
                live, armed = True, False
                possessor = restarts[f]
                if possessor == NONE:
                    pending_from = i
            elif after_out:
                back_in = back_in or bool(inside[f])
                if back_in and confirmed[f] != NONE:
                    # play has visibly resumed without a tagged restart
                    ended("control exit")
                    live, possessor = True, confirmed[f]
        if live:
            if inside[f]:
                armed = True
            if armed and out_start[f]:
                resolve(i, previous)
                if possessor != NONE:
                    previous = possessor
                live, possessor = False, NONE
                after_out, back_in = True, False
                possible_goal = goal_mouth is not None and bool(goal_mouth[f])
                cause = "possible goal" if possible_goal else "ball out"
            elif confirmed[f] != NONE and confirmed[f] != possessor:
                if stats is not None and possessor != NONE:
                    stats["control_switches"] += 1
                resolve(i, confirmed[f])
                possessor = confirmed[f]
        if not live:
            labels[i] = DEAD
        elif possessor != NONE:
            labels[i] = possessor
            previous = possessor
    resolve(n, previous)
    if not live:
        i = n
        ended("never ended")
    return labels


# --- (6) short-run merging ------------------------------------------------------------------

def merge_short_runs(labels, min_team, min_dead):
    """Absorb H/A runs shorter than `min_team` and D runs shorter than `min_dead` into the run
    before them (into the one after, for a short first run). NOX runs are never absorbed.

    One left-to-right pass: a run absorbed into its predecessor makes that run longer before the
    next run is looked at, and neighbours that end up with the same label join.
    """
    labels = np.asarray(labels)
    starts, lengths, values = run_lengths(labels)
    if len(values) <= 1:
        return labels.copy()

    def short(value, length):
        if value in (HOME, AWAY):
            return length < min_team
        if value == DEAD:
            return length < min_dead
        return False

    merged = []  # [value, length]
    carry = 0  # frames of short leading runs waiting for the first kept run
    for value, length in zip(values.tolist(), lengths.tolist(), strict=True):
        if merged and (short(value, length) or merged[-1][0] == value):
            merged[-1][1] += length
        elif not merged and short(value, length):
            carry += length
        else:
            merged.append([value, length + carry])
            carry = 0
    if not merged:  # every run was short: keep them as they were
        return labels.copy()
    return np.repeat([v for v, _ in merged], [k for _, k in merged]).astype(labels.dtype)


# --- (7) whole match ------------------------------------------------------------------------

def label_match(ball, players, team, restarts, n_frames, blocks, cfg,
                goal_restarts=None):
    """One letter per frame; see run_match."""
    return run_match(ball, players, team, restarts, n_frames, blocks, cfg, goal_restarts)[0]


def run_match(ball, players, team, restarts, n_frames, blocks, cfg, goal_restarts=None):
    """Everything from metres-coordinates to (one letter per frame, stats).

    ball (n, 2) and players (n, P, 2) in metres with NaN gaps; team (P,); restarts as returned
    by restart_events, and goal_restarts the subset that can end a possible goal
    (restart_events with types=cfg.POSSIBLE_GOAL_END_TYPES); blocks [(half, first, last)] of
    graded frames; cfg the config.
    Frames outside every block are X. stats is new_stats() filled in, plus "ball",
    "confirmed_runs" and "final" counts (see match_counts).
    """
    raw_seen = ~np.isnan(np.asarray(ball, dtype=float)).any(axis=1)
    ball = interpolate_ball(ball, cfg.MAX_BALL_GAP_FRAMES)
    control = frame_control(ball, players, team, cfg.CONTROL_RADIUS_M, cfg.DUEL_RADIUS_M)
    confirmed = confirmed_starts(control, cfg.MIN_CONFIRMED_FRAMES)
    out = ball_out(ball, cfg.PITCH_LENGTH_M, cfg.PITCH_WIDTH_M, cfg.OUT_MARGIN_M)
    out_start = sustained_out_starts(out, cfg.MIN_OUT_FRAMES)
    # "back inside" means within the lines themselves, with no noise margin
    inside = ball_in(ball, cfg.PITCH_LENGTH_M, cfg.PITCH_WIDTH_M, 0.0)
    goal_mouth = goal_mouth_exit(ball, cfg.PITCH_LENGTH_M, cfg.PITCH_WIDTH_M, cfg.OUT_MARGIN_M,
                                 cfg.GOAL_HALF_WIDTH_M + cfg.GOAL_MOUTH_MARGIN_M)

    stats = new_stats()
    codes = np.full(n_frames, NOX, dtype=np.int8)
    for _, first, last in blocks:
        states = assign_states(confirmed, out_start, inside, restarts, first, last,
                               goal_mouth, goal_restarts, stats)
        states[states == NONE] = NOX
        codes[first:last + 1] = merge_short_runs(
            states, cfg.MIN_TEAM_RUN_FRAMES, cfg.MIN_DEAD_RUN_FRAMES)
    stats.update(match_counts(codes, raw_seen, ~np.isnan(ball).any(axis=1), confirmed, blocks))
    return LETTERS[codes], stats


def match_counts(codes, raw_seen, filled_seen, confirmed, blocks):
    """Counts over the final codes and the ball track, for the diagnostic summary."""
    graded = np.zeros(len(codes), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    def ball_counts(mask):
        return {"observed": int(np.sum(raw_seen & mask)),
                "interpolated": int(np.sum(filled_seen & ~raw_seen & mask)),
                "missing": int(np.sum(~filled_seen & mask))}

    final = Counter()
    for _, first, last in blocks:
        _, lengths, values = run_lengths(codes[first:last + 1])
        teams = values[(values == HOME) | (values == AWAY)]
        final["possessions"] += len(teams)  # maximal H/A runs, as spellF1 counts them
        final["team changes"] += int(np.sum(teams[1:] != teams[:-1]))  # H..A, D between or not
        final["direct H<->A"] += int(np.sum(
            np.isin(values[:-1], (HOME, AWAY)) & np.isin(values[1:], (HOME, AWAY))))
        final["D segments"] += int(np.sum(values == DEAD))
    return {
        "ball": {"all frames": ball_counts(np.ones(len(codes), dtype=bool)),
                 "graded frames": ball_counts(graded)},
        "confirmed_runs": {"H": int(np.sum((confirmed == HOME) & graded)),
                           "A": int(np.sum((confirmed == AWAY) & graded))},
        "final": dict(final),
        "letters": {letter: int(np.sum(codes == k)) for k, letter in enumerate("HADX")},
        "graded_letters": {letter: int(np.sum((codes == k) & graded))
                           for k, letter in enumerate("HADX")},
    }
