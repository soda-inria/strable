quality:

	isort . -c

	flake8

	mypy

	black --check .


style:

	isort .

	black .


run-benchmark:

	python script_evaluate.py

