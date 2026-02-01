.PHONY: venv lock format typecheck test container-reqs

test:
	poetry run pytest .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

typecheck:
	poetry run mypy thesis/ *.py

venv:
	poetry install --with dev

lock:
	poetry lock

container-reqs: .venv-pip-tools/bin/pip-compile
	.venv-pip-tools/bin/pip-compile requirements.in -o requirements.txt --strip-extras

.venv-pip-tools/bin/pip-compile:
	python3.12 -m venv .venv-pip-tools
	.venv-pip-tools/bin/pip install --quiet pip-tools

