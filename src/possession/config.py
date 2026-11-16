"""Every tunable number in the v1 pipeline, in one place, each with where it came from.

Sources, as checked against the papers/code (not paraphrased from memory):

- Vidal-Codina et al. 2022, "Automatic event detection in football using tracking data",
  Sports Engineering 25:18 (arXiv 2202.00804). Defines a possession zone (radius R_pz around
  each player) and a duel zone (radius R_dz >= R_pz around the ball; two opponents inside it
  = duel, and a duel overrides possession). Results section: R_pz = 0.5 m or 1 m depending on
  provider, R_dz = 1 m for all providers; the discussion suggests "augmenting the possession
  zone radius" for lower-quality data such as a single tactical camera -- our case.
- databallpy (Alek050/databallpy, main): TrackingData.add_individual_player_possession(
  pz_radius=1.5, bv_threshold=5.0, ba_threshold=10.0, min_frames_pz=0), documented as
  implementing Vidal-Codina et al. 2022. Its tracking quality check
  (_check_missing_ball_data) warns on ball gaps of >= 1 s, i.e. 1 * framerate frames.
- Link & Hoernig 2017, "Individual ball possession in soccer", PLoS ONE 12(7):e0179953. Their
  distance thresholds are trained, not published as numbers. What they do state: 25 Hz data,
  and "Very short sections (< 1 s) are excluded from further processing."

Where none of the three gives the number, the comment says so. Those are v1 choices to be
revisited, not literature values.

v2 (external validation): three values were then set on EXTERNAL data only -- SkillCorner open
data, 20 A-League 2024/25 matches, MIT licence (external/). A 24-setting grid was run on 10
development matches; the chosen setting was then checked once on the other 10 held-out matches:
spellF1 0.437 (v1) -> 0.470 (v2), better on 9 of 10, paired bootstrap 95% CI of the gain
[+0.020, +0.046]. The competition data, its ground truth and its video played no part.
Reproduce: `make external` (v2) and docs/TECHNICAL.md section 5.

v3 adds a stoppage detector (section 5b below), also set on external data only.
v4 adds Smart Tagging phases as possession evidence (section 4b at the end of this file).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import games  # noqa: E402

GAME = games.GAME

# --- Frame rate and geometry ----------------------------------------------------------------

# 25 fps tactical video (games.py). Also the rate Link & Hoernig 2017 work at ("25 Hz").
FPS = games.GAMES[GAME]["fps"]

# Pitch in metres, from games.py (measured from the corrected tracking). The ATD metadata says
# 105 x 68; DATA.md notes the 30 cm only matters for distances in metres, which is what we compute.
PITCH_LENGTH_M, PITCH_WIDTH_M = games.GAMES[GAME]["pitch"]

# --- (2) Ball gap interpolation -------------------------------------------------------------

# Linearly interpolate ball gaps of at most this many missing frames (1 s at 25 fps).
# databallpy's tracking quality check flags ball gaps of >= 1 s (1 * framerate) as a problem,
# so gaps under that are treated as fillable. Linear (not Savitzky-Golay) interpolation is a v1
# choice.
MAX_BALL_GAP_FRAMES = 25

# --- (3) Per-frame control ------------------------------------------------------------------

# A team controls the ball on a frame when its nearest player is within this radius of it.
# v1 used databallpy's default pz_radius = 1.5 m (Vidal-Codina et al. 2022 possession zone,
# enlarged from the paper's 0.5-1 m, as the paper suggests for lower-quality tracking).
# v2: 2.0 m, chosen on external data (see module docstring); the five best of the 24 settings on
# the development matches all used 2.0 m, consistent with the paper's advice for noisy tracking.
CONTROL_RADIUS_M = 2.0

# ...and no opponent is within this radius of the ball. Vidal-Codina et al. 2022 duel zone,
# R_dz = 1 m for all providers; a duel overrides possession, so the frame has no controller.
DUEL_RADIUS_M = 1.0

# --- (4) Confirmed control ------------------------------------------------------------------

# Control by the same team must hold for this many consecutive frames (0.32 s) to count, and
# possession then switches from the first frame of that run (Vidal-Codina et al. 2022 also put
# the gain at the start of the control interval, f0).
# No literature source: databallpy's min_frames_pz defaults to 0. v1 used 8 (a design choice).
# v2: 12 (0.48 s), chosen on external data; 8, 12 and 16 were within 0.008 of each other at
# 2.0 m, so the middle of that flat region was taken rather than the single best point.
MIN_CONFIRMED_FRAMES = 12

# --- (5) Dead ball and restarts -------------------------------------------------------------

# Entering D. v1 DESIGN CHOICES, fixed before any scoring (football logic, not tuned; none of
# the three sources gives these numbers):
# The ball counts as out only when it is more than this far outside the lines, so tracking noise
# around a touchline or goal line is not read as the ball leaving the pitch.
OUT_MARGIN_M = 0.5
# ...and only when it stays out by that margin for at least this many consecutive frames
# (0.2 s at 25 fps). D then starts on the first of those frames.
MIN_OUT_FRAMES = 5

# Leaving D (v1 design choice, fixed before any scoring; not a number, so documented here):
# a stoppage caused by the ball going out ends at the next set-piece restart tag, OR as soon as
# the ball has been seen back inside the lines and either team then gets confirmed control,
# whichever comes first. The period-start D (before kickoff) ends only at the kickoff tag, since
# players stand on the ball at the centre spot before play starts.

# Possible goal (v1 DESIGN CHOICE, fixed before any scoring): a sustained exit over a goal line
# with the ball within this distance of the goal's centre line is treated as a possible goal.
# That D ends only at the next kickoff, goal-kick or corner tag, whichever comes first: a goal
# is restarted by a kickoff, a shot over the crossbar (indistinguishable here, the tracking has
# no ball height) by a goal kick or a corner. The early "back inside + control" exit and every
# other set-piece tag (throw-in, free kick) are disabled for it, since players carrying the
# ball back to the centre spot after a goal would otherwise look like play resuming.
# Goal half-width 3.66 m: IFAB Laws of the Game, Law 1 (7.32 m between the posts).
GOAL_HALF_WIDTH_M = 3.66
# Tolerance on that half-width for tracking noise (the same 0.5 m as OUT_MARGIN_M). v1 choice.
GOAL_MOUTH_MARGIN_M = 0.5
# Smart Tagging Type values that end a possible-goal D (DATA.md spellings), matched
# case-insensitively. v1 design choice, fixed before any scoring.
POSSIBLE_GOAL_END_TYPES = ("kick off (start)", "kick off (after goal)", "goal kick",
                           "corner kick")

# Smart Tagging code that marks a restart (DATA.md / starter/load_smart_tagging.py). Matched
# case-insensitively. Every Type of this code is a restart (throw-in, corner, goal kick, free
# kick, kick-offs), so all of them end a dead-ball stretch.
RESTART_CODE = "set pieces"

# Which frame of a tagged set-piece interval counts as the restart: "start", "mid" or "end".
# starter/load_smart_tagging.py notes the tag marks a ~5 s window *around* the restart, not the
# instant. "start" is the literal reading of the brief. No literature source; v1 choice.
# Exception (core.anchor_frame): a tag whose window overlaps the first graded frame of a period
# fires on that frame. Fix after v1 diagnostics showed the first-half kickoff tag anchored before
# frame 11185, leaving 6348 frames of D before the first restart.
RESTART_ANCHOR = "start"

# Smart Tagging Team values -> side (DATA.md: home is Barcelona, away is Athletic, spelled
# "Atletic Club" in the file). Anything else, notably "N/A" (every throw-in), means "unknown":
# the possessor is then taken from the first confirmed control after the restart.
TEAM_TAG_TO_SIDE = {"barcelona": "H", "atletic club": "A"}

# --- (6) Short-run merging ------------------------------------------------------------------

# H/A runs shorter than this are absorbed into the run before them. v1: 25 frames, from Link &
# Hoernig 2017 ("Very short sections (< 1 s) are excluded from further processing").
# v2: 50 frames (2 s), chosen on external data; 50 beat 25 in 11 of the 12 radius x confirm
# settings on the development matches.
MIN_TEAM_RUN_FRAMES = 50

# D runs shorter than this (2 s) are absorbed the same way.
# No literature source; v1 choice from the brief.
MIN_DEAD_RUN_FRAMES = 50

# --- (7) Graded periods ---------------------------------------------------------------------

# Frames outside these video-frame blocks get X: [(half, first, last)], inclusive. From games.py,
# the same table DATA.md publishes and the starter code uses.
GRADED_BLOCKS = games.half_video_blocks(GAME)

# Submission length: one row per tactical-video frame.
N_FRAMES = games.source_frames("tactical", GAME)


# --- (5b) stoppages from collective movement (v3) -----------------------------------------
# v2 only saw a dead ball when the ball left the pitch, so fouls, injuries, substitutions and the
# wait before a restart all read as live play. On external data v2 called 15-22% of graded frames
# dead where the providers' labels have 39-46%.
#
# v3: play is also dead while the players are collectively slow -- the idea of Mueller et al.
# style game-interruption detection from player movement alone, in its simplest form. It needs no
# ball and no player identity, which suits tracking that loses the ball on ~41% of frames.
#
# All values below were set on EXTERNAL data only (never the competition match):
#   development: Metrica public sample game 1 (same provider, 25 fps; "dead" = the ball has no
#                position) and 10 SkillCorner matches;
#   held-out:    Metrica sample game 2 and the other 10 SkillCorner matches.
# Held-out result, v2 -> v3: SkillCorner spellF1 0.470 -> 0.487 (better on 9 of 10, paired
# bootstrap 95% CI [+0.010, +0.026]); Metrica game 2 spellF1 0.365 -> 0.467, and 0.239 -> 0.377
# with the ball hidden on 40% of frames. Reproduce: `make external-stoppage`.

# Mean player speed (m/s) below which play counts as stopped. On both providers the mean speed
# sits around 1.0 m/s when the ball is dead and 2.0 m/s when it is live. 1.2-1.5 scored alike on
# the development data; 1.35 is the middle of that range. 0 switches the detector off (= v2).
STOPPAGE_SPEED_MPS = 1.35

# The speed is averaged over a centred window this long, and a stoppage must last at least this
# long (3 s each). Shorter lulls are slow play, not stoppages.
STOPPAGE_WINDOW_FRAMES = 75
STOPPAGE_MIN_FRAMES = 75

# A slot's speed is its displacement over this many frames (1 s), which averages out tracking
# jitter. A slot implying more than 12 m/s (faster than any sprint) is a track swap and is
# ignored. With fewer than 6 usable slots the frame has no speed and is never a stoppage.
STOPPAGE_SPAN_FRAMES = 25
STOPPAGE_MAX_PLAYER_MPS = 12.0
STOPPAGE_MIN_PLAYERS = 6

# Veto: where the ball IS seen and travels at this smoothed speed or more, slow players mean slow
# build-up, not a dead ball. 2-3 m/s scored alike on the development data. An unseen ball vetoes
# nothing, so the detector still works through ball gaps.
STOPPAGE_BALL_MPS = 3.0


# --- (4b) Smart Tagging phases as possession evidence (v4) ---------------------------------
# v3 used only the 59 set-piece tags. The other ~580 Smart Tagging instances are tactical phases
# (Build Up, Progression, Defending in Middle Third, ...), each tagged with a team. DATA.md calls
# them "a strong prior, for free" and warns that they are automatic, are phases rather than
# possessions, and can be wrong. v4 uses them only where the geometry has no answer: when no team
# shows control (ball missing, loose or in flight) for a sustained run and the phases keep naming
# one team, that run counts as that team's confirmed control. A frame where the geometry does
# show control is never overridden.
#
# Which codes name the team WITH the ball and which the team WITHOUT it is not assumed. It is
# learned on the match itself from the pipeline's own confirmed, ball-in-view control frames
# (algorithmic pseudo-labels, no ground truth, no human decision -- RULES.md "learning without
# human labels"). A code whose meaning is unclear is not used, so tags that turn out unreliable
# switch the feature off and the output is v3.
#
# No external dataset ships Smart Tagging, so the values below were set on SIMULATED tags cut
# from external games' own labels and degraded (60% coverage, about 1 s jitter, half the phases
# naming the defending team, 0-40% naming the wrong team): external/phase_study.py,
# `make external-phases`. Held-out result against v3: better while at most about 20% of tags are
# wrong (SkillCorner 0.487 -> 0.511 / 0.490 / 0.488 at 0 / 10 / 20% wrong; Metrica game 2 with
# the ball hidden on 40% of frames 0.377 -> 0.460 / 0.451 / 0.399), identical to v3 beyond that
# because the agreement gate rejects the codes.

# Codes that are not phases of open play.
PHASE_SKIP_CODES = ("SET PIECES", "GOALS")

# A phase-only run must last this long (2 s) to count as confirmed control. The tags are accurate
# to about a second; 50 was the best of 50 / 75 / 125 on the development games.
PHASE_MIN_FRAMES = 50

# A code is used only if the tagged team matches our confirmed-control team on at least this
# share of anchor frames (it names the team with the ball) or at most 1 - this share (it names
# the team without it), with at least PHASE_MIN_ANCHOR_FRAMES anchor frames (20 s) to judge by.
# 0.85 rather than 0.80: at 0.80 a tag set with 20% errors was still let through and cost a
# little on the development games.
PHASE_MIN_AGREEMENT = 0.85
PHASE_MIN_ANCHOR_FRAMES = 500
