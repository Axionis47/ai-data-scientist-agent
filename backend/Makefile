PY=python
PIP=pip

.PHONY: setup lint test run docker-build
setup:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install .

lint:
	ruff check app tests
	black --check app tests
	mypy app

test:
	pytest -q --maxfail=1 --disable-warnings

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t ai-ds-backend:local .



bench:
	$(PY) bench/run_bench.py --out bench/bench.csv

bench-report:
	$(PY) bench/report.py --csv bench/bench.csv --out bench/bench.html --thresholds bench/thresholds.yaml
	@echo "Open backend/bench/bench.html in your browser"
