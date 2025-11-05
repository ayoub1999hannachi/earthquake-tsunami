
.PHONY: setup data eda train test clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt


data:
	python -m src.generate_data

eda:
	python -m src.run_eda

train:
	python -m src.train_model

test:
	@echo ">> Running tests with src in PYTHONPATH"
	@set PYTHONPATH=. && pytest -v



clean:
	rm -rf models/* reports/*
