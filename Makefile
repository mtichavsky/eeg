
.PHONY: venv lock format

format:
	ruff format .
	ruff check --fix .

venv:
	poetry install --with dev

lock: 
	poetry lock
