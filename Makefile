.PHONY: sync format lint typecheck test check hooks

sync:
	uv sync

## auto-fix lint findings + reformat
format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

## check only, no writes (what the pre-commit hook and CI run)
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test:
	uv run pytest

typecheck:
	uv run mypy
	uv run pyright

## the full gate: lint + types + tests
check: lint typecheck test

## install the git pre-commit hook (once per clone)
hooks:
	uv run pre-commit install
