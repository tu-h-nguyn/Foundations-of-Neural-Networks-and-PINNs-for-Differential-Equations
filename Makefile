PYTHON ?= python3

.PHONY: help install test lint format verify verify-rerun experiments thesis clean

help:
	@echo "install       install the package with its development dependencies"
	@echo "test          run the unit test suite"
	@echo "lint          run ruff over the sources"
	@echo "format        apply ruff's automatic fixes"
	@echo "verify        re-derive every published number from code/results/*.json"
	@echo "verify-rerun  the same, plus re-running experiment 1 from scratch (~1 min)"
	@echo "experiments   re-run all six experiments (slow; rewrites code/results/)"
	@echo "thesis        build main.pdf with latexmk (needs a TeX distribution)"
	@echo "clean         remove build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .

verify:
	cd code && $(PYTHON) verify_results.py

verify-rerun:
	cd code && $(PYTHON) verify_results.py --chay-lai-tn1

experiments:
	cd code && $(PYTHON) -m pinns.run_all

thesis:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf build dist .pytest_cache .ruff_cache *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	latexmk -C || true
