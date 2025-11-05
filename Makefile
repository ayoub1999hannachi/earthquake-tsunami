.PHONY: env install test lint run-notebook

env:
	python -m venv .venv
	. .venv/bin/activate; pip install -r requirements.txt

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	black --check src tests

run-notebook:
	jupyter notebook notebooks/earthquake-tsunami.ipynb
