"""Smoke tests needing no match data: the scorer runs and the baselines score as published."""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Published public-5-minute numbers from README.md.
EXPECTED = {
    "baseline_smooth125": (0.5812, 0.4242),
    "baseline_smooth50": (0.6207, 0.3256),
    "baseline_smooth25": (0.6200, 0.2868),
    "baseline_raw": (0.5972, 0.0378),
}


def test_scorer_unmodified():
    subprocess.run(["shasum", "-a", "256", "-c", "evaluation.sha256"], cwd=ROOT, check=True)


def test_baselines_validate():
    for name in EXPECTED:
        subprocess.run([sys.executable, "evaluation/validate_submission.py",
                        f"evaluation/results/{name}.csv"], cwd=ROOT, check=True,
                       capture_output=True)


def test_baselines_score_as_published(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in EXPECTED:
        (results / f"{name}.csv").write_bytes(
            (ROOT / "evaluation" / "results" / f"{name}.csv").read_bytes())
    subprocess.run([sys.executable, "evaluation/run_evaluation.py", "--results", str(results),
                    "--out", str(tmp_path), "--no-detail"], cwd=ROOT, check=True,
                   capture_output=True)
    with open(tmp_path / "scoreboard.csv") as handle:
        rows = {row["submission"]: row for row in csv.DictReader(handle)}
    for name, (macro, spell) in EXPECTED.items():
        row = rows[name]
        assert round(float(row["macro_f1"]), 4) == macro, name
        assert round(float(row["spell_f1_mean"]), 4) == spell, name
