PY=python
PIP=pip

.PHONY: setup lint test run docker-build openai-smoke docker-openai-smoke
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

openai-smoke:
	uvicorn app.main:app --host 127.0.0.1 --port 8090 & \
	sleep 1; \
	curl -s -S -f http://127.0.0.1:8090/openai-smoke | jq -r .ok,.sdk_installed,.has_key; \
	curl -s -S -f "http://127.0.0.1:8090/openai-smoke?live=true" | jq -r .ok,.reply || true; \
	kill %1 2>/dev/null || true

docker-openai-smoke:
	docker compose up -d backend; \
	sleep 1; \
	curl -s -S -f http://localhost:8000/openai-smoke | jq -r .ok,.sdk_installed,.has_key; \
	curl -s -S -f "http://localhost:8000/openai-smoke?live=true" | jq -r .ok,.reply || true; \
	docker compose down

docker-build:
	docker build -t ai-ds-backend:local .



bench:
	$(PY) bench/run_bench.py --out bench/bench.csv

bench-report:
	$(PY) bench/report.py --csv bench/bench.csv --out bench/bench.html --thresholds bench/thresholds.yaml
	@echo "Open backend/bench/bench.html in your browser"
