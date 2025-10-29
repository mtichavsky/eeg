
.PHONY: venv lock format typecheck

format:
	ruff format .
	ruff check --fix .

typecheck:
	poetry run mypy thesis/ *.py

venv:
	poetry install --with dev

lock:
	poetry lock
