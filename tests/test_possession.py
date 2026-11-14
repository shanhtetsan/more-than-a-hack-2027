"""Unit tests for the v1 possession pipeline. Synthetic data only: nothing here opens data/."""

import types

import numpy as np
import pytest

from possession import config, core, diagnostics, loaders
from possession.core import AWAY, HOME, NONE

L, W = config.PITCH_LENGTH_M, config.PITCH_WIDTH_M
NAN = np.nan


def codes(text):
    """'HHAD' -> array of internal codes, for readable expectations."""
    return np.array(["HADX".index(c) for c in text], dtype=np.int8)


# --- config ---------------------------------------------------------------------------------

def test_config_values_match_the_brief():
    assert (config.PITCH_LENGTH_M, config.PITCH_WIDTH_M) == (105.3, 68.0)
    assert (config.OUT_MARGIN_M, config.MIN_OUT_FRAMES) == (0.5, 5)
    assert (config.GOAL_HALF_WIDTH_M, config.GOAL_MOUTH_MARGIN_M) == (3.66, 0.5)
    assert config.MAX_BALL_GAP_FRAMES == 25
    assert (config.CONTROL_RADIUS_M, config.DUEL_RADIUS_M) == (2.0, 1.0)
    assert config.MIN_CONFIRMED_FRAMES == 12
    assert (config.MIN_TEAM_RUN_FRAMES, config.MIN_DEAD_RUN_FRAMES) == (50, 50)
    assert config.GRADED_BLOCKS == [(1, 11185, 90651), (2, 110808, 190197)]
    assert config.N_FRAMES == 199319


# --- (1) units ------------------------------------------------------------------------------

def test_to_metres_scales_each_axis_and_keeps_nan():
    out = core.to_metres([[0.5, 0.5], [1.0, 0.0], [NAN, NAN]], L, W)
    assert np.allclose(out[:2], [[L / 2, W / 2], [L, 0.0]])
    assert np.isnan(out[2]).all()


# --- (2) interpolation ----------------------------------------------------------------------

def test_interpolate_fills_short_interior_gap_linearly():
    ball = np.array([[0, 0], [NAN, NAN], [NAN, NAN], [3, 6]], dtype=float)
    out = core.interpolate_ball(ball, max_gap=25)
    assert np.allclose(out, [[0, 0], [1, 2], [2, 4], [3, 6]])


def test_interpolate_boundary_gap_of_exactly_max_is_filled_and_longer_is_not():
    for gap, filled in ((25, True), (26, False)):
        ball = np.vstack([[0, 0], np.full((gap, 2), NAN), [10, 10]])
        out = core.interpolate_ball(ball, max_gap=25)
        assert (not np.isnan(out).any()) == filled
        if not filled:
            assert np.isnan(out[1:-1]).all()  # not partially filled


def test_interpolate_never_extrapolates_track_ends():
    ball = np.array([[NAN, NAN], [1, 1], [NAN, NAN]], dtype=float)
    out = core.interpolate_ball(ball, max_gap=25)
    assert np.isnan(out[0]).all() and np.isnan(out[2]).all()


def test_interpolate_treats_half_missing_point_as_missing():
    ball = np.array([[0, 0], [5, NAN], [2, 2]], dtype=float)
    assert np.allclose(core.interpolate_ball(ball, 25)[1], [1, 1])


# --- (3) per-frame control ------------------------------------------------------------------

TEAM = np.array([HOME, AWAY], dtype=np.int8)


def control_of(home_d, away_d):
    """Control code for one frame with the home player home_d m and away player away_d m away."""
    ball = np.array([[50.0, 30.0]])
    players = np.array([[[50.0 + home_d, 30.0], [50.0, 30.0 + away_d]]])
    return core.frame_control(ball, players, TEAM, 1.5, 1.0)[0]


@pytest.mark.parametrize("home_d, away_d, expected", [
    (0.5, 5.0, HOME),     # clear home control
    (1.5, 5.0, HOME),     # exactly on the control radius counts
    (1.51, 5.0, NONE),    # just outside it
    (0.5, 1.0, NONE),     # opponent exactly at the duel radius blocks it
    (0.5, 1.01, HOME),    # opponent just beyond it does not
    (5.0, 0.3, AWAY),     # symmetric for away
    (1.2, 1.4, HOME),     # both inside 1.5 but beyond 1.0: nearer team wins
    (1.4, 1.2, AWAY),
])
def test_frame_control_thresholds(home_d, away_d, expected):
    assert control_of(home_d, away_d) == expected


def test_frame_control_missing_ball_or_players():
    ball = np.array([[NAN, NAN], [50.0, 30.0]])
    players = np.array([[[50.0, 30.0], [NAN, NAN]], [[NAN, NAN], [NAN, NAN]]])
    assert list(core.frame_control(ball, players, TEAM, 1.5, 1.0)) == [NONE, NONE]


def test_frame_control_uses_nearest_player_of_each_team():
    ball = np.array([[50.0, 30.0]])
    team = np.array([HOME, HOME, AWAY], dtype=np.int8)
    players = np.array([[[70.0, 30.0], [50.5, 30.0], [55.0, 30.0]]])
    assert core.frame_control(ball, players, team, 1.5, 1.0)[0] == HOME


# --- (4) confirmed control ------------------------------------------------------------------

def test_confirmed_needs_eight_consecutive_frames_and_marks_run_start():
    control = np.array([NONE] * 2 + [HOME] * 8 + [NONE] + [AWAY] * 7 + [NONE], dtype=np.int8)
    out = core.confirmed_starts(control, 8)
    assert np.flatnonzero(out != NONE).tolist() == [2]
    assert out[2] == HOME  # the 7-frame away run is not confirmed


def test_confirmed_run_is_broken_by_a_gap():
    control = np.array([HOME] * 4 + [NONE] + [HOME] * 4, dtype=np.int8)
    assert (core.confirmed_starts(control, 8) == NONE).all()


# --- (5) dead ball and restarts -------------------------------------------------------------

def test_ball_out_is_strictly_outside_and_false_when_unseen():
    ball = np.array([[0, 0], [L, W], [-0.01, 10], [10, W + 0.01], [NAN, NAN]], dtype=float)
    assert core.ball_out(ball, L, W).tolist() == [False, False, True, True, False]
    assert core.ball_in(ball, L, W).tolist() == [True, True, False, False, False]


def test_ball_out_margin_ignores_noise_near_the_lines():
    ball = np.array([[10, -0.4], [10, -0.5], [10, -0.6], [L + 0.6, 10], [10, W + 0.6]],
                    dtype=float)
    assert core.ball_out(ball, L, W, margin=0.5).tolist() == [False, False, True, True, True]


@pytest.mark.parametrize("run, expected", [(4, []), (5, [3]), (9, [3])])
def test_sustained_out_needs_five_frames_and_marks_the_first(run, expected):
    out = np.zeros(20, bool)
    out[3:3 + run] = True
    assert np.flatnonzero(core.sustained_out_starts(out, 5)).tolist() == expected


def test_sustained_out_run_is_broken_by_a_missing_ball_frame():
    out = np.array([True] * 3 + [False] + [True] * 3)  # ball_out is False where unseen
    assert not core.sustained_out_starts(out, 5).any()


def test_goal_mouth_exit_is_over_a_goal_line_between_the_posts_plus_margin():
    mid, half = W / 2, 3.66 + 0.5
    ball = np.array([
        [L + 1.0, mid],                 # straight through the goal
        [-1.0, mid + half],             # exactly on the widened post, other end
        [L + 1.0, mid + half + 0.01],   # just wide
        [L + 0.4, mid],                 # not beyond the goal line by the out margin
        [50.0, -2.0],                   # touchline exit
        [NAN, NAN],
    ])
    assert core.goal_mouth_exit(ball, L, W, 0.5, half).tolist() == [
        True, True, False, False, False, False]


def test_restart_events_type_filter_keeps_possible_goal_restarts():
    def tag(frame, team, kind):
        return {"code": "SET PIECES", "start_frame": frame, "end_frame": frame + 10,
                "tags": {"Team": team, "Type": kind}}

    events = [
        tag(10, "Barcelona", "Kick Off (Start)"),
        tag(30, "Barcelona", "Corner Kick"),
        tag(50, "Atletic Club", "KICK OFF (AFTER GOAL)"),
        tag(70, "Atletic Club", "goal kick"),
        tag(90, "N/A", "Throw-in"),
        tag(110, "Barcelona", "Free Kick"),
    ]
    kept = core.restart_events(events, "set pieces", "start", config.TEAM_TAG_TO_SIDE,
                               types=config.POSSIBLE_GOAL_END_TYPES)
    assert kept == {10: HOME, 30: HOME, 50: AWAY, 70: AWAY}  # no throw-in, no free kick
    assert len(core.restart_events(events, "set pieces", "start", config.TEAM_TAG_TO_SIDE)) == 6


def test_restart_events_map_teams_and_anchor():
    events = [
        {"code": "SET PIECES", "start_frame": 100, "end_frame": 200, "tags": {"Team": "Barcelona"}},
        {"code": "Set Pieces", "start_frame": 300, "end_frame": 400, 
         "tags": {"Team": "Atletic Club"}},
        {"code": "SET PIECES", "start_frame": 500, "end_frame": 600, "tags": {"Team": "N/A"}},
        {"code": "BUILD UP", "start_frame": 700, "end_frame": 800, "tags": {"Team": "Barcelona"}},
    ]
    sides = config.TEAM_TAG_TO_SIDE
    assert core.restart_events(events, "set pieces", "start", sides) == {
        100: HOME, 300: AWAY, 500: NONE}
    assert list(core.restart_events(events, "set pieces", "mid", sides)) == [150, 350, 550]
    assert list(core.restart_events(events, "set pieces", "end", sides)) == [200, 400, 600]


# --- state machine --------------------------------------------------------------------------

def run_states(n, restarts, confirmed=(), out=(), inside=None):
    """assign_states over frames 0..n-1 with sparse confirmed {frame: team}.

    `out` lists the frames where a sustained exit from the pitch *starts*; `inside` defaults to
    "inside on every other frame".
    """
    conf = np.full(n, NONE, dtype=np.int8)
    for frame, team in dict(confirmed).items():
        conf[frame] = team
    out_mask = np.zeros(n, bool)
    out_mask[list(out)] = True
    inside_mask = ~out_mask if inside is None else inside
    return core.assign_states(conf, out_mask, inside_mask, restarts, 0, n - 1)


def test_period_is_dead_until_kickoff_then_tagged_team():
    assert np.array_equal(run_states(6, {2: HOME}), codes("DDHHHH"))


def test_switch_only_on_other_teams_confirmed_control():
    states = run_states(8, {0: HOME}, confirmed={3: HOME, 5: AWAY})
    assert np.array_equal(states, codes("HHHHHAAA"))


def test_out_of_pitch_is_dead_until_next_restart_then_tagged_team():
    states = run_states(10, {0: HOME, 4: HOME, 7: AWAY}, out=[3])
    # out at 3, restart at 4 ends the stoppage; the restart at 7 arrives while live: ignored
    assert np.array_equal(states, codes("HHHDHHHHHH"))
    states = run_states(10, {0: HOME, 7: AWAY}, out=[3])
    assert np.array_equal(states, codes("HHHDDDDAAA"))


def outside(n, frames):
    """inside mask that is False on `frames` (ball off the pitch, or unseen, there)."""
    return ~np.isin(np.arange(n), list(frames))


def test_controls_while_ball_is_still_outside_are_ignored():
    # e.g. the thrower holds the ball off the pitch (2-5): that does not restart play
    states = run_states(8, {0: HOME, 6: HOME}, confirmed={4: AWAY}, out=[2],
                        inside=outside(8, range(2, 6)))
    assert np.array_equal(states, codes("HHDDDDHH"))


def test_dead_ends_early_when_ball_back_inside_and_team_confirms_control():
    # out at 2, back inside from 4, away confirms at 5; the restart tag at 8 comes too late
    states = run_states(10, {0: HOME, 8: HOME}, confirmed={5: AWAY}, out=[2],
                        inside=outside(10, [2, 3]))
    assert np.array_equal(states, codes("HHDDDAAAAA"))


def test_early_exit_on_the_frame_the_ball_comes_back_in():
    states = run_states(8, {0: HOME}, confirmed={4: HOME}, out=[2], inside=outside(8, [2, 3]))
    assert np.array_equal(states, codes("HHDDHHHH"))


def test_early_exit_needs_the_control_run_to_start_after_ball_is_back_in():
    # the control run starts at 3 while the ball is still out (inside from 4): no early exit
    states = run_states(10, {0: HOME, 8: AWAY}, confirmed={3: AWAY}, out=[2],
                        inside=outside(10, [2, 3]))
    assert np.array_equal(states, codes("HHDDDDDDAA"))


def test_restart_tag_wins_when_it_comes_first():
    states = run_states(10, {0: HOME, 5: HOME}, confirmed={7: AWAY}, out=[2],
                        inside=outside(10, [2, 3]))
    assert np.array_equal(states, codes("HHDDDHHAAA"))


def run_goal(n, restarts, goal_restarts, confirmed=(), out=(), inside=None, goal=()):
    """run_states, plus a goal_mouth mask True on `goal` frames and the subset of restarts
    (kickoff / goal kick / corner) that can end a possible goal."""
    conf = np.full(n, NONE, dtype=np.int8)
    for frame, team in dict(confirmed).items():
        conf[frame] = team
    out_mask = np.isin(np.arange(n), list(out))
    inside_mask = ~out_mask if inside is None else inside
    return core.assign_states(conf, out_mask, inside_mask, restarts, 0, n - 1,
                              goal_mouth=np.isin(np.arange(n), list(goal)),
                              goal_restarts=goal_restarts)


def test_possible_goal_ignores_early_exit_and_other_restarts_until_kickoff():
    # out through the goal at 2; ball carried back in and controlled at 4; a throw-in or
    # free-kick tag at 5 (in restarts, not in goal_restarts); kickoff for the away side at 8
    states = run_goal(10, {0: HOME, 5: HOME, 8: AWAY}, goal_restarts={0: HOME, 8: AWAY},
                      confirmed={4: HOME}, out=[2], inside=outside(10, [2, 3]), goal=[2])
    assert np.array_equal(states, codes("HHDDDDDDAA"))


@pytest.mark.parametrize("kind", ["goal kick", "corner"])
def test_possible_goal_ends_at_goal_kick_or_corner_if_it_comes_first(kind):
    # shot over the bar at 2: the goal kick / corner tag (home) at 5 beats the kickoff at 8
    states = run_goal(10, {0: AWAY, 5: HOME, 8: AWAY}, goal_restarts={0: AWAY, 5: HOME, 8: AWAY},
                      out=[2], inside=outside(10, [2, 3]), goal=[2])
    assert np.array_equal(states, codes("AADDDHHHHH"))


def test_wide_of_the_posts_keeps_the_early_exit():
    states = run_goal(10, {0: HOME, 8: AWAY}, goal_restarts={0: HOME, 8: AWAY},
                      confirmed={4: HOME}, out=[2], inside=outside(10, [2, 3]), goal=[])
    assert np.array_equal(states, codes("HHDDHHHHHH"))


def test_goal_mouth_only_counts_on_the_frame_the_exit_starts():
    # goal_mouth True later in the stoppage (ball rolls behind the goal) changes nothing
    states = run_goal(10, {0: HOME, 6: AWAY}, goal_restarts={0: HOME}, out=[2],
                      inside=outside(10, [2, 3, 4, 5]), goal=[4])
    assert np.array_equal(states, codes("HHDDDDAAAA"))


def test_possible_goal_without_a_later_goal_restart_stays_dead():
    states = run_goal(8, {0: HOME, 5: AWAY}, goal_restarts={0: HOME}, confirmed={4: AWAY},
                      out=[2], inside=outside(8, [2, 3]), goal=[2])
    assert np.array_equal(states, codes("HHDDDDDD"))


def test_untagged_kickoff_after_possible_goal_is_backfilled_by_control():
    states = run_goal(10, {0: HOME, 5: NONE}, goal_restarts={0: HOME, 5: NONE},
                      confirmed={7: AWAY}, out=[2], inside=outside(10, [2, 3]), goal=[2])
    assert np.array_equal(states, codes("HHDDDAAAAA"))


def test_pre_kickoff_control_does_not_start_the_period():
    # players stand on the ball at the centre spot: only the kickoff tag ends period-start D
    states = run_states(8, {5: AWAY}, confirmed={1: HOME})
    assert np.array_equal(states, codes("DDDDDAAA"))


def test_untagged_restart_is_backfilled_by_next_confirmed_control():
    states = run_states(10, {0: HOME, 4: NONE}, confirmed={7: AWAY}, out=[2])
    assert np.array_equal(states, codes("HHDDAAAAAA"))


def test_untagged_restart_without_control_falls_back_to_previous_possessor():
    states = run_states(10, {0: AWAY, 4: NONE}, out=[2, 7])
    assert np.array_equal(states, codes("AADDAAADDD"))


def test_untagged_kickoff_with_nothing_to_go_on_is_x():
    assert np.array_equal(run_states(4, {1: NONE}), codes("DXXX"))


def test_throw_in_from_outside_does_not_retrigger_dead_ball():
    # out from 2; restart at 4 with the ball still outside (4-5, a sustained run starting at
    # 4 too), back in from 6
    states = run_states(10, {0: HOME, 4: AWAY}, out=[2, 4], inside=outside(10, range(2, 6)))
    assert np.array_equal(states, codes("HHDDAAAAAA"))
    # and once back inside, a later exit triggers again
    states = run_states(10, {0: HOME, 4: AWAY}, out=[2, 4, 8],
                        inside=outside(10, [2, 3, 4, 5, 8, 9]))
    assert np.array_equal(states, codes("HHDDAAAADD"))


def test_missing_ball_holds_state():
    # missing ball = not out, not inside, no control: nothing changes
    states = run_states(6, {0: AWAY}, inside=np.zeros(6, bool))
    assert np.array_equal(states, codes("AAAAAA"))


# --- (6) merging ----------------------------------------------------------------------------

def test_short_team_run_absorbed_into_previous():
    labels = codes("H" * 30 + "A" * 24 + "H" * 30)
    assert np.array_equal(core.merge_short_runs(labels, 25, 50), codes("H" * 84))


def test_team_run_of_exactly_min_length_is_kept():
    labels = codes("H" * 30 + "A" * 25 + "H" * 30)
    assert np.array_equal(core.merge_short_runs(labels, 25, 50), labels)


def test_short_dead_run_absorbed_but_long_one_kept():
    short = codes("H" * 30 + "D" * 49 + "A" * 30)
    assert np.array_equal(core.merge_short_runs(short, 25, 50), codes("H" * 79 + "A" * 30))
    long = codes("H" * 30 + "D" * 50 + "A" * 30)
    assert np.array_equal(core.merge_short_runs(long, 25, 50), long)


def test_short_leading_run_goes_to_the_next_run():
    labels = codes("A" * 10 + "D" * 60)
    assert np.array_equal(core.merge_short_runs(labels, 25, 50), codes("D" * 70))


def test_x_runs_are_never_absorbed():
    labels = codes("H" * 30 + "X" * 3 + "H" * 30)
    assert np.array_equal(core.merge_short_runs(labels, 25, 50), labels)


# --- (7) whole pipeline on a synthetic match ------------------------------------------------

def synthetic_match():
    """400 frames, one graded block 20..379, one home and one away player.

    kickoff tag (H) at 30 -> home dribbles, with two out-of-pitch noise blips that must not
    become D -> away wins it at 150 -> ball out at 250 -> untagged restart at 300 (throw-in,
    from outside) -> ball back in from 305 -> away confirmed at 320.
    """
    n = 400
    ball = np.tile([50.0, 30.0], (n, 1))
    home = np.tile([80.0, 10.0], (n, 1))
    away = np.tile([20.0, 60.0], (n, 1))
    ball[60:80] = [50.0, -0.4]                       # hugging the line, within the margin
    ball[100:104] = [50.0, -2.0]                     # clearly out, but only 4 frames
    home[30:150] = ball[30:150] + [0.5, 0]          # home on the ball
    away[150:250] = ball[150:250] + [0, 0.5]        # away on the ball
    ball[200:210] = NAN                              # short gap: interpolated, holds A
    ball[250:300] = [50.0, -2.0]                     # out over the touchline
    ball[300:305] = [50.0, -0.5]                     # throw-in taken from outside
    away[320:] = ball[320:] + [0, 0.5]               # away confirmed from 320
    players = np.stack([home, away], axis=1)
    team = np.array([HOME, AWAY], dtype=np.int8)
    restarts = {30: HOME, 300: NONE}
    return ball, players, team, restarts, n


def test_label_match_end_to_end():
    ball, players, team, restarts, n = synthetic_match()
    labels = core.label_match(ball, players, team, restarts, n, [(1, 20, 379)], config)
    # the 10 pre-kickoff D frames are < MIN_DEAD_RUN_FRAMES, so step 6 merges them into H
    expected = "X" * 20 + "H" * 130 + "A" * 100 + "D" * 50 + "A" * 80 + "X" * 20
    assert "".join(labels) == expected
    assert len(labels) == n


def test_label_match_is_deterministic():
    ball, players, team, restarts, n = synthetic_match()
    first = core.label_match(ball, players, team, restarts, n, [(1, 20, 379)], config)
    second = core.label_match(ball, players, team, restarts, n, [(1, 20, 379)], config)
    assert np.array_equal(first, second)


def test_label_match_early_exit_without_a_restart_tag():
    ball, players, team, restarts, n = synthetic_match()
    del restarts[300]  # the tagger missed the throw-in: play resumes on control alone
    labels = "".join(core.label_match(ball, players, team, restarts, n, [(1, 20, 379)], config))
    # back inside from 305, away confirmed from 320 -> D 250..319, then A
    assert labels[250:320] == "D" * 70
    assert labels[320:380] == "A" * 60


def test_label_match_reads_parameters_from_cfg():
    ball, players, team, restarts, n = synthetic_match()
    cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    cfg.MIN_CONFIRMED_FRAMES = 200  # no run that long: possession never changes hands
    labels = "".join(core.label_match(ball, players, team, restarts, n, [(1, 20, 379)], cfg))
    assert labels[150:250] == "H" * 100


# --- loaders, without files -----------------------------------------------------------------

def test_rows_to_arrays_indexes_by_frame_id_minus_one():
    rows = [
        {"frame_id": 1, "ball_x": 0.5, "ball_y": 0.25, "player_p1_x": 0.1, "player_p1_y": 0.2},
        {"frame_id": 3, "ball_x": NAN, "ball_y": NAN, "player_p2_x": 0.9, "player_p2_y": 0.8},
        {"frame_id": 99, "ball_x": 0.0, "ball_y": 0.0},  # beyond n_frames: dropped
    ]
    ball, players = loaders.rows_to_arrays(rows, ["p1", "p2"], n_frames=3)
    assert np.allclose(ball[0], [0.5, 0.25])
    assert np.isnan(ball[1]).all() and np.isnan(ball[2]).all()
    assert np.allclose(players[0, 0], [0.1, 0.2]) and np.isnan(players[0, 1]).all()
    assert np.allclose(players[2, 1], [0.9, 0.8])


def test_kloppy_epts_internals_still_exist():
    # load_tracking leans on these kloppy 3.18 internals; fail loudly here if a bump moves them.
    from kloppy.domain import Ground  # noqa: F401
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata  # noqa: F401
    from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data  # noqa: F401


def goal_match(goal_y, tags=None):
    """400 frames, block 20..379: kickoff (H) at 30, home on the ball to 249, exit over the
    right goal line at height goal_y for 250..299, then the ball back at the centre with an
    away player on it from 300. `tags` is {frame: (team, is_possible_goal_restart)}; by
    default a free-kick tag (H) at 320 and the away kickoff at 340.
    """
    tags = {320: (HOME, False), 340: (AWAY, True)} if tags is None else tags
    n, mid = 400, W / 2
    ball = np.tile([50.0, mid], (n, 1))
    home = np.tile([80.0, 10.0], (n, 1))
    away = np.tile([20.0, 60.0], (n, 1))
    home[30:250] = ball[30:250] + [0.5, 0]
    ball[250:300] = [L + 2.0, goal_y]
    away[300:] = ball[300:] + [0, 0.5]
    players = np.stack([home, away], axis=1)
    team = np.array([HOME, AWAY], dtype=np.int8)
    restarts = {30: HOME, **{f: t for f, (t, _) in tags.items()}}
    goal_restarts = {30: HOME, **{f: t for f, (t, ok) in tags.items() if ok}}
    # synthetic scenes are short: keep the v1 1 s possession floor so a 40-frame run survives
    cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    cfg.MIN_TEAM_RUN_FRAMES = 25
    labels = core.label_match(ball, players, team, restarts, n, [(1, 20, 379)], cfg,
                              goal_restarts=goal_restarts)
    return "".join(labels)


def test_label_match_goal_waits_for_kickoff():
    # the 10 pre-kickoff D frames merge into H (step 6); the free kick at 320 is ignored
    assert goal_match(W / 2) == "X" * 20 + "H" * 230 + "D" * 90 + "A" * 40 + "X" * 20


def test_label_match_shot_over_the_bar_ends_at_the_goal_kick():
    # goal kick for the away side at 320 comes before any kickoff (none tagged here)
    labels = goal_match(W / 2, tags={320: (AWAY, True)})
    assert labels == "X" * 20 + "H" * 230 + "D" * 70 + "A" * 60 + "X" * 20


def test_label_match_exit_wide_of_the_posts_resumes_on_control():
    # back inside at 300 with away control: live from 300; the later tags arrive while live
    labels = goal_match(W / 2 + 3.66 + 0.5 + 1.0)
    assert labels == "X" * 20 + "H" * 230 + "D" * 50 + "A" * 80 + "X" * 20


# --- diagnostics ----------------------------------------------------------------------------

def test_run_match_stats_count_what_happened():
    ball, players, team, restarts, n = synthetic_match()
    labels, stats = core.run_match(ball, players, team, restarts, n, [(1, 20, 379)], config)
    assert np.array_equal(labels, core.label_match(ball, players, team, restarts, n,
                                                   [(1, 20, 379)], config))
    assert stats["d_segments"] == {("period start", "restart tag"): 1,
                                   ("ball out", "restart tag"): 1}
    assert stats["used_restarts"] == {30, 300}
    assert stats["period_start_frames"] == [10]
    assert stats["control_switches"] == 1          # H -> A at 150; 320 only fills a pending
    assert stats["confirmed_runs"] == {"H": 1, "A": 2}
    assert stats["final"] == {"possessions": 3, "team changes": 1, "direct H<->A": 1,
                              "D segments": 1}
    assert stats["ball"]["all frames"] == {"observed": 390, "interpolated": 10, "missing": 0}
    assert stats["graded_letters"] == {"H": 130, "A": 180, "D": 50, "X": 0}
    assert stats["letters"]["X"] == 40


def test_stats_record_control_exit_goal_restart_and_never_ended():
    ball, players, team, restarts, n = synthetic_match()
    del restarts[300]  # throw-in untagged: the stoppage ends by control exit
    _, stats = core.run_match(ball, players, team, restarts, n, [(1, 20, 379)], config)
    assert stats["d_segments"][("ball out", "control exit")] == 1

    states = run_goal(8, {0: HOME, 5: AWAY}, goal_restarts={0: HOME}, out=[2],
                      inside=outside(8, [2, 3]), goal=[2])
    assert np.array_equal(states, codes("HHDDDDDD"))
    stats = core.new_stats()
    conf = np.full(10, NONE, dtype=np.int8)
    out = np.isin(np.arange(10), [2])
    core.assign_states(conf, out, outside(10, [2, 3]), {0: HOME, 5: AWAY, 7: HOME}, 0, 9,
                       goal_mouth=out, goal_restarts={0: HOME, 7: HOME}, stats=stats)
    assert stats["d_segments"] == {("period start", "restart tag"): 1,
                                   ("possible goal", "goal restart tag"): 1}
    stats = core.new_stats()
    core.assign_states(conf, out, outside(10, [2, 3]), {0: HOME}, 0, 9, stats=stats)
    assert stats["d_segments"][("ball out", "never ended")] == 1


def test_stats_do_not_change_labels():
    conf = np.full(10, NONE, dtype=np.int8)
    conf[5] = AWAY
    out = np.isin(np.arange(10), [2])
    args = (conf, out, outside(10, [2, 3]), {0: HOME, 8: HOME}, 0, 9)
    assert np.array_equal(core.assign_states(*args),
                          core.assign_states(*args, stats=core.new_stats()))


def test_restart_types_follow_the_anchor():
    events = [{"code": "SET PIECES", "start_frame": 10, "end_frame": 30,
               "tags": {"Type": "Throw-in"}},
              {"code": "BUILD UP", "start_frame": 40, "end_frame": 50, "tags": {}}]
    assert core.restart_types(events, "set pieces", "start") == {10: "throw-in"}
    assert core.restart_types(events, "set pieces", "mid") == {20: "throw-in"}


def test_summary_reports_a_kickoff_tag_anchored_before_the_period():
    # the kickoff tag window opens before the first graded frame, so its "start" anchor
    # falls outside the period and cannot end the period-start D: the summary must show it
    ball, players, team, _, n = synthetic_match()
    restarts = {15: HOME, 300: NONE}
    _, stats = core.run_match(ball, players, team, restarts, n, [(1, 20, 379)], config)
    types = {15: "kick off (start)", 300: "throw-in"}
    tags = diagnostics.restart_tag_counts(types, stats["used_restarts"], [(1, 20, 379)])
    assert tags["kick off (start)"] == {"parsed": 1, "in graded": 0, "ended a D": 0}
    assert tags["throw-in"] == {"parsed": 1, "in graded": 1, "ended a D": 1}
    assert stats["period_start_frames"] == [280]  # D until the throw-in tag at 300

    lines = diagnostics.summary_lines(stats, types, [(1, 20, 379)])
    text = "\n".join(lines)
    assert "period 1: 280 frames of D before its first restart" in text
    assert "kick off (start)" in text and "possessions" in text
    assert len(lines) < 40  # a summary, not a frame dump


# --- restart tag overlapping the start of a period ------------------------------------------

def kickoff_tag(start, end, team="Barcelona", kind="Kick Off (Start)"):
    return {"code": "SET PIECES", "start_frame": start, "end_frame": end,
            "tags": {"Team": team, "Type": kind}}


@pytest.mark.parametrize("anchor", ["start", "mid", "end"])
def test_tag_overlapping_first_graded_frame_fires_on_it(anchor):
    blocks = [(1, 11185, 90651), (2, 110808, 190197)]
    events = [kickoff_tag(11100, 11225),                    # contains 11185
              kickoff_tag(113700, 113825, "Atletic Club"),  # 2nd-half kickoff, inside block 2
              kickoff_tag(11185, 11300)]                    # opens exactly on 11185 too
    got = core.restart_events(events, "set pieces", anchor, config.TEAM_TAG_TO_SIDE,
                              blocks=blocks)
    plain = core.anchor_frame(events[1], anchor)
    assert got == {11185: HOME, plain: AWAY}
    assert core.restart_types(events, "set pieces", anchor, blocks) == {
        11185: "kick off (start)", plain: "kick off (start)"}


def test_tag_window_edges_against_first_graded_frame():
    blocks = [(1, 100, 500)]
    assert core.anchor_frame(kickoff_tag(50, 100), "start", blocks) == 100   # ends on it
    assert core.anchor_frame(kickoff_tag(50, 99), "start", blocks) == 50     # ends just before
    assert core.anchor_frame(kickoff_tag(101, 150), "start", blocks) == 101  # opens just after
    assert core.anchor_frame(kickoff_tag(50, 99), "start") == 50             # no blocks: plain


def test_kickoff_tag_opening_before_the_period_starts_play_on_its_first_frame():
    # synthetic match graded from 20, home on the ball from 20; the kickoff tag runs 10..40
    ball, players, team, _, n = synthetic_match()
    players[20:150, 0] = ball[20:150] + [0.5, 0]
    blocks = [(1, 20, 379)]
    events = [kickoff_tag(10, 40),
              {"code": "SET PIECES", "start_frame": 300, "end_frame": 340,
               "tags": {"Team": "N/A", "Type": "Throw-in"}}]
    restarts = core.restart_events(events, "set pieces", "start", config.TEAM_TAG_TO_SIDE,
                                   blocks=blocks)
    labels, stats = core.run_match(ball, players, team, restarts, n, blocks, config)
    assert stats["period_start_frames"] == [0]
    assert "".join(labels[20:150]) == "H" * 130

    # without the blocks the anchor (10) is outside the period: D until the throw-in at 300
    restarts = core.restart_events(events, "set pieces", "start", config.TEAM_TAG_TO_SIDE)
    _, stats = core.run_match(ball, players, team, restarts, n, blocks, config)
    assert stats["period_start_frames"] == [280]


# --- external harness ------------------------------------------------------------------------

def test_external_resample_10_to_25_fps_is_linear_and_spreads_nan():
    from external import skillcorner
    x = np.array([[0.0], [10.0], [20.0], [np.nan], [40.0]])
    out = skillcorner.resample(x)[:, 0]
    assert len(out) == 11                      # (5 - 1) * 25 // 10 + 1
    assert np.allclose(out[:6], [0.0, 4.0, 8.0, 12.0, 16.0, 20.0])  # frame 5 = source 2 exactly
    assert np.isnan(out[6:10]).all() and out[10] == 40.0
    assert skillcorner.to_target(10) == 25
