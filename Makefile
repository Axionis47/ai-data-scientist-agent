# Root Makefile for developer ergonomics

.PHONY: up down logs backend-tests frontend-build bench bench-report smoke compose-smoke openai-smoke docker-openai-smoke

# Local dev (non-Docker): run backend and frontend in two terminals
# (kept as comments for reference)
# dev:
# 	( cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000 ) & \
# 	( cd frontend && npm run dev )

up:
	docker compose up --build -d

logs:
	docker compose logs -f --tail=200

down:
	docker compose down

backend-tests:
	cd backend && pytest -q

frontend-build:
	cd frontend && npm run build

bench:
	cd backend && $(MAKE) bench

bench-report:
	cd backend && $(MAKE) bench-report

smoke:
	bash scripts/smoke.sh

compose-smoke:
	docker compose up --build -d
	@echo "Waiting for backend health..."
	@i=0; until curl -s -S -m 2 -f http://localhost:8000/health >/dev/null; do \
	  i=$$((i+1)); \
	  if [ $$i -gt 60 ]; then echo "Timeout waiting for backend health"; docker compose logs backend; docker compose down; exit 1; fi; \
	  sleep 1; \
	done
	bash scripts/smoke.sh || (docker compose logs backend; docker compose down; exit 1)
	docker compose down

openai-smoke:
	cd backend && $(MAKE) openai-smoke

docker-openai-smoke:
	cd backend && $(MAKE) docker-openai-smoke

compose-rebuild-frontend:
	docker compose up -d --build frontend
