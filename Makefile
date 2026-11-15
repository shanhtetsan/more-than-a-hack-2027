# One entry point for everything. `make submission TEAM=<name>` is the command that regenerates
# the file we upload. Recipes need real tabs.

TEAM    ?= DX_Lab_Team_3
PY      := uv run --frozen python
SUB     := submissions/$(TEAM).csv

.PHONY: help setup verify-scorer check-data baselines submission validate score score-all \
        reproduce external external-sweep test lint format clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup:           ## create .venv from the lockfile
	uv sync --frozen

verify-scorer:   ## fail if the official scorer or public GT was edited
	@shasum -a 256 -c --quiet evaluation.sha256 && echo "official scorer: unmodified"

check-data:      ## fail with a readable message if the match data is missing
	@$(PY) -c "import games; games.require(['atd_meta','atd_raw','smart_tagging','homography']); print('data: ok ->', games.DATA_DIR)"

baselines: check-data  ## regenerate the 4 published baselines from the ATD
	$(PY) starter/baseline.py

submission: verify-scorer check-data  ## build, validate and score submissions/$(TEAM).csv
	$(PY) -m possession.pipeline --team $(TEAM)
	$(PY) evaluation/validate_submission.py $(SUB)
	cp $(SUB) evaluation/results/$(TEAM).csv
	$(PY) evaluation/run_evaluation.py --no-detail

validate:        ## validate submissions/$(TEAM).csv (or FILE=path)
	$(PY) evaluation/validate_submission.py $(or $(FILE),$(SUB))

score: verify-scorer  ## score everything in evaluation/results/ on the public 5 minutes
	$(PY) evaluation/run_evaluation.py --no-detail

score-all: verify-scorer  ## same, with per-submission detail and bootstrap intervals
	$(PY) evaluation/run_evaluation.py --bootstrap 1000

reproduce: check-data  ## rebuild twice and check the file is byte-identical
	$(PY) -m possession.pipeline --team $(TEAM) --out .repro/a >/dev/null
	$(PY) -m possession.pipeline --team $(TEAM) --out .repro/b >/dev/null
	cmp .repro/a/$(TEAM).csv .repro/b/$(TEAM).csv && echo "reproducible: identical"
	rm -rf .repro

external:        ## score the pipeline on 20 SkillCorner open-data matches (external validation)
	$(PY) -m external.evaluate
	$(PY) -m external.evaluate --baseline 50

external-sweep:  ## the v1 -> v2 parameter study on SkillCorner (dev grid, held-out check)
	$(PY) -m external.sweep

test:            ## smoke tests (no match data needed)
	uv run --frozen pytest -q

lint:            ## ruff check
	uv run --frozen ruff check .

format:          ## ruff format our code
	uv run --frozen ruff format src tests

clean:           ## remove generated reports and caches
	rm -rf evaluation/reports evaluation/scoreboard.csv .repro .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
