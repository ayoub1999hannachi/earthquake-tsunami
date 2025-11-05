.PHONY: env install test lint run-notebook

env:
	python -m venv .venv ;
	 .\.venv\Scripts\Activate.ps1   ; 

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	black --check src tests

run-notebook:
	jupyter notebook notebooks/earthquake-tsunami.ipynb
