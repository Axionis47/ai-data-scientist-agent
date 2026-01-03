# Deployment Guide

## Branch Strategy

| Branch | Purpose | Auto-Deploy |
|--------|---------|-------------|
| `main` | Production-ready code | No (manual) |
| `staging` | Pre-production testing | Yes (to staging) |
| `develop` | Integration branch | No |
| `feature/*` | Feature development | No |

## Deployment Workflow

### 1. Feature Development
```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-feature

# Make changes, commit, push
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

### 2. Merge to Develop
- Create PR: `feature/my-feature` → `develop`
- CI runs: ruff, pytest, docker build
- Merge after review

### 3. Deploy to Staging
```bash
# Merge develop to staging
git checkout staging
git pull origin staging
git merge develop
git push origin staging
```
- Triggers `deploy_staging.yml` workflow
- Builds and pushes Docker image to GHCR
- Image tagged as `staging` and `staging-<sha>`

### 4. Deploy to Production
```bash
# After staging validation, merge to main
git checkout main
git pull origin main
git merge staging
git push origin main
```
- Manual deployment from `main` branch

## Docker Image

### Build Locally
```bash
docker build -t sdlc-api:local .
docker run -d -p 8080:8080 sdlc-api:local
```

### Pull from Registry
```bash
# Staging image
docker pull ghcr.io/<owner>/sdlc-api:staging

# Specific commit
docker pull ghcr.io/<owner>/sdlc-api:staging-abc1234
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings/LLM | Yes |
| `STORAGE_DIR` | Path to document storage | No (default: `/app/storage`) |
| `DATASETS_DIR` | Path to dataset storage | No (default: `/app/datasets`) |

## Health Check

```bash
curl http://localhost:8080/health
# Expected: {"status": "healthy"}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload_context_doc` | POST | Upload context document (.docx) |
| `/upload_dataset` | POST | Upload dataset (.csv) |
| `/ask` | POST | Ask a question |

## Smoke Test

```bash
# 1. Upload context document
curl -X POST http://localhost:8080/upload_context_doc \
  -F "file=@context.docx"

# 2. Upload dataset
curl -X POST http://localhost:8080/upload_dataset \
  -F "file=@data.csv"

# 3. Ask a question
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Give me an overview of the dataset",
    "doc_id": "<doc_id_from_step_1>",
    "dataset_id": "<dataset_id_from_step_2>"
  }'
```

## Troubleshooting

### Container won't start
- Check logs: `docker logs <container_id>`
- Verify OPENAI_API_KEY is set
- Ensure ports are not in use

### Tests failing
```bash
# Run tests locally
python -m pytest services/api/tests/ -v

# Run linter
python -m ruff check .
```

### Docker build fails
- Ensure all dependencies are in requirements.txt
- Check Dockerfile syntax
- Verify base image is accessible

