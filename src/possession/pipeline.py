"""Build the submission file: data in, frame,label CSV out.

v0 is the published floor (nearest player to the ball, majority vote over +/-50 frames), reusing
the organisers' starter/baseline.py so we start from a known score. Replace `predict` with the
real method; keep the rest so `make submission` stays a single reproducible command.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import baseline  # noqa: E402  (starter/baseline.py)

WINDOW = 50


def predict():
    """One label per tactical-video frame, as a numpy array of 'H'/'A'/'D'/'X'."""
    codes = baseline.build()
    return baseline.smooth(codes, WINDOW)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file is <team>.csv")
    parser.add_argument("--out", default=str(ROOT / "submissions"), help="output directory")
    args = parser.parse_args(argv)

    labels = predict()
    path = baseline.write_submission(os.path.join(args.out, f"{args.team}.csv"), labels)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(f"wrote {os.path.relpath(path, ROOT)}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
