
.PHONY: venv lock format typecheck

format:
	poetry run ruff format .
	poetry run ruff check --fix .

typecheck:
	poetry run mypy thesis/ *.py

venv:
	poetry install --with dev

lock:
	poetry lock
