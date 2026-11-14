"""Reading the match data into arrays. The only module here that opens files under data/.

ATD goes through kloppy's Metrica EPTS reader. We use its metadata parser and streaming row
reader (the same two pieces metrica.load_tracking_epts is built from) rather than the full
TrackingDataset, because 199,319 Frame objects x 46 track slots costs minutes and gigabytes
(starter/load_atd.py says the same). The coordinates are the raw normalised 0-1 values, which we
scale to metres with the games.py pitch.
"""

import os
import sys

import numpy as np

from . import config
from .core import AWAY, HOME, to_metres

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def rows_to_arrays(rows, player_ids, n_frames):
    """kloppy EPTS rows -> (ball (n, 2), players (n, P, 2)) in normalised units, NaN if unseen.

    Each row is a dict with frame_id (1-based), ball_x, ball_y and player_<id>_x/_y. Row i of
    the output is video frame i = frame_id - 1 (DATA.md). Rows outside 0..n-1 are dropped.
    """
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, len(player_ids), 2), np.nan, dtype=np.float32)
    keys = [(f"player_{pid}_x", f"player_{pid}_y") for pid in player_ids]
    for row in rows:
        frame = int(row["frame_id"]) - 1
        if not 0 <= frame < n_frames:
            continue
        ball[frame] = (row.get("ball_x", np.nan), row.get("ball_y", np.nan))
        players[frame] = [(row.get(kx, np.nan), row.get(ky, np.nan)) for kx, ky in keys]
    return ball, players


def load_tracking(meta_path, raw_path, n_frames=config.N_FRAMES,
                  length=config.PITCH_LENGTH_M, width=config.PITCH_WIDTH_M):
    """(ball (n, 2), players (n, P, 2), team (P,)) in metres; team is HOME/AWAY per slot."""
    from kloppy.domain import Ground
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
    from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data

    with open(meta_path, "rb") as handle:
        metadata = load_metadata(handle)

    player_ids, team = [], []
    for side in metadata.teams:
        code = HOME if side.ground == Ground.HOME else AWAY
        for player in side.players:
            player_ids.append(player.player_id)
            team.append(code)

    with open(raw_path, "rb") as handle:
        rows = read_raw_data(handle, metadata, sensor_ids=["position"])
        ball, players = rows_to_arrays(rows, player_ids, n_frames)

    return (to_metres(ball, length, width),
            to_metres(players, length, width).astype(np.float32),
            np.array(team, dtype=np.int8))


def load_smart_tagging():
    """Smart Tagging instances as dicts with code, start_frame, end_frame and tags.

    starter/load_smart_tagging.load: standard library, time base verified against both kickoffs.
    """
    import load_smart_tagging

    return load_smart_tagging.load()
