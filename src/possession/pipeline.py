"""Build the submission file: data in, frame,label CSV out.

v1 rule-based pipeline (every number is in config.py, every rule in core.py):
  1. tactical ATD via kloppy, scaled to metres with the games.py pitch
  2. ball gaps of <= 25 frames linearly interpolated
  3. per-frame control: nearest player <= 1.5 m from the ball, no opponent <= 1.0 m
  4. confirmed control = same team >= 8 consecutive frames; possession switches only on the
     other team's confirmed control, otherwise the last state holds
  5. ball > 0.5 m outside the pitch for >= 5 frames -> D, until the next Smart Tagging
     set-piece restart (possession to the tagged team) or, if sooner, the ball back inside
     with a team's confirmed control (possession to that team). A possible goal (exit over a
     goal line between the posts, +0.5 m) ends only at the next kickoff, goal-kick or
     corner tag
  6. H/A runs < 25 frames and D runs < 50 frames merged into their neighbour
  7. X outside the graded halves
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from . import config, core, diagnostics, loaders

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import baseline  # noqa: E402  (starter/baseline.py, for write_submission)

import games  # noqa: E402


def predict():
    """(labels, summary): one 'H'/'A'/'D'/'X' per tactical-video frame, and the diagnostic
    summary lines (counts only)."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw", "smart_tagging"])[:2]
    print("reading tracking via kloppy (one streaming pass, takes a few minutes) ...")
    ball, players, team = loaders.load_tracking(meta_path, raw_path)
    events = loaders.load_smart_tagging()
    blocks = config.GRADED_BLOCKS
    restarts = core.restart_events(events, config.RESTART_CODE, config.RESTART_ANCHOR,
                                   config.TEAM_TAG_TO_SIDE, blocks=blocks)
    goal_restarts = core.restart_events(events, config.RESTART_CODE, config.RESTART_ANCHOR,
                                        config.TEAM_TAG_TO_SIDE,
                                        types=config.POSSIBLE_GOAL_END_TYPES, blocks=blocks)
    print(f"  {len(team)} track slots, {len(restarts)} set-piece restarts "
          f"({len(goal_restarts)} kickoff/goal-kick/corner)")
    labels, stats = core.run_match(ball, players, team, restarts, config.N_FRAMES,
                                   config.GRADED_BLOCKS, config, goal_restarts)
    types = core.restart_types(events, config.RESTART_CODE, config.RESTART_ANCHOR, blocks)
    return labels, diagnostics.summary_lines(stats, types, config.GRADED_BLOCKS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file is <team>.csv")
    parser.add_argument("--out", default=str(ROOT / "submissions"), help="output directory")
    args = parser.parse_args(argv)

    labels, summary = predict()
    path = baseline.write_submission(os.path.join(args.out, f"{args.team}.csv"), labels)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(f"wrote {os.path.relpath(path, ROOT)}  sha256 {digest}")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
