# Run with Docker Compose

## Prereqs
- Docker Desktop (or Docker Engine)

## Start
- docker compose up --build
- Frontend: http://localhost:3000
- Backend: http://localhost:8000 (health at /health)

## Env
- To enable LLM-based reporting, pass OPENAI_API_KEY to the backend service in docker-compose.yml

## Data persistence
- backend/data is volume-mounted into the backend container at /app/data so jobs persist between restarts.

## Notes
- The tiny sample flow is available via POST /sample
- If you encounter modeling errors on the tiny sample (edge cases), uploads of normal CSVs should model fine.

