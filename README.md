# DX Lab Team 3 — More Than a Hack 2027, Phase 1: ball possession

A rule-based system that labels every frame of FC Barcelona vs Athletic Club
(LaLiga 2024-25, MD2) as **H** (Barça in possession), **A** (Athletic in possession),
**D** (dead ball) or **X** (not graded), from automatic tracking data and Smart Tagging.

No training, and no tuning on the competition match. Three parameters were chosen on
**external** data (20 SkillCorner open-data matches, held-out tested); the rest come from the
literature, the Laws of the Game, or design choices fixed before any scoring. Each one is cited in
[`src/possession/config.py`](src/possession/config.py).

**Technical document:** [`docs/TECHNICAL.md`](docs/TECHNICAL.md)

## Results

**External validation, 10 held-out SkillCorner matches (official metric code):**

| | mean spellF1 |
|---|---|
| Organisers' `baseline_smooth50` (all 20 matches) | 0.350 |
| v1 (literature defaults) | 0.437 |
| **v2, submitted** | **0.470** (better than v1 on 9 of 10) |

**Competition match, public 5-minute window** (17 possessions, a correctness check only, never
used to choose anything): v1 scored spellF1 0.388 against 0.326 for the published floor
`baseline_smooth50`; v2 scored 0.455.

## How it works

```mermaid
flowchart LR
    A[Tracking<br/>ball + 46 track slots] --> B[Fill ball gaps<br/>up to 1 s]
    B --> C[Per-frame control<br/>≤2 m, no opponent ≤1 m]
    C --> D[Confirmed control<br/>≥12 frames]
    E[Smart Tagging<br/>set-piece restarts] --> F
    B --> G[Ball out of play<br/>&gt;0.5 m out, ≥5 frames]
    D --> F[Possession state machine]
    G --> F
    F --> H[Merge short runs<br/>&lt;2 s]
    H --> I[frame,label CSV]
```

Possession only changes hands when the **other** team gets confirmed control. While the ball is
missing, contested or in the air, the last state holds. Full rules: [technical document](docs/TECHNICAL.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and the match data bundle (not included; the data is
not redistributable).

```bash
uv sync --frozen                    # Python 3.11, exact pins from uv.lock
# unpack the bundle into ./data (layout in data/README.md), or set MTAH_DATA_DIR
make submission TEAM=DX_Lab_Team_3  # build -> validate -> score on the public window
make reproduce  TEAM=DX_Lab_Team_3  # rebuild twice, check the file is byte-identical
make external                       # external validation on 20 SkillCorner matches
make external-sweep                 # the parameter study behind v2 (dev grid + held-out check)
make test                           # unit tests on synthetic data, no match data needed
```

Output: `submissions/DX_Lab_Team_3.csv`. The pipeline is deterministic (no randomness).

## Layout

| Path | What |
|---|---|
| `src/possession/config.py` | every parameter, with its source |
| `src/possession/core.py` | the rules, as pure numpy functions |
| `src/possession/loaders.py` | the only module that reads match data |
| `src/possession/pipeline.py` | entry point: data in, CSV out |
| `src/possession/diagnostics.py` | counts-only run summary |
| `external/` | external validation harness (SkillCorner open data) |
| `tests/` | unit tests on synthetic matches |
| `evaluation/`, `starter/`, `games.py` | organisers' code, unmodified (checked by `evaluation.sha256`) |

The organisers' original README is kept as [`ORGANISERS_README.md`](ORGANISERS_README.md).
