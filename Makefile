PYTHON ?= /home/jarvis/.venvs/adasigboost/bin/python
PIP ?= /home/jarvis/.venvs/adasigboost/bin/pip
PYTEST ?= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest
RUFF ?= /home/jarvis/.venvs/adasigboost/bin/ruff
MYPY ?= /home/jarvis/.venvs/adasigboost/bin/mypy

.PHONY: help build test lint typecheck theory clean

help:
	@echo "Targets: build, test, lint, typecheck, theory, clean"

build:
	$(PIP) install --no-build-isolation --no-deps -e .

test:
	$(PYTEST)

lint:
	$(RUFF) check adasigboost tests benchmarks

typecheck:
	$(MYPY) adasigboost

theory:
	cd docs/theory && pdflatex -interaction=nonstopmode AdaSigBoost_Theory_Note.tex && pdflatex -interaction=nonstopmode AdaSigBoost_Theory_Note.tex

clean:
	rm -rf build _skbuild dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
