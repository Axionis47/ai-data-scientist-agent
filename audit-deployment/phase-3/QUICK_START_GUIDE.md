# 🚀 Quick Start Guide - Using Your Deployed Infrastructure

**Last Updated**: 2025-10-17
**Status**: Infrastructure deployed and ready

---

## 📋 What's Deployed

- ✅ **GCP Infrastructure** - 30 resources live
- ✅ **Secret Manager** - 3 encrypted secrets
- ✅ **Artifact Registry** - Docker repository ready
- ✅ **Workload Identity** - Keyless GitHub authentication
- ✅ **CI/CD Pipeline** - GitHub Actions workflow

---

## 🎯 Quick Commands

### View Your Secrets

```bash
# List all secrets
gcloud secrets list --project=plotpointe

# View secret metadata (not the actual value)
gcloud secrets describe botds-openai-api-key-dev --project=plotpointe

# Access secret value (requires permissions)
gcloud secrets versions access latest --secret=botds-openai-api-key-dev --project=plotpointe
```

### View Artifact Registry

```bash
# List repositories
gcloud artifacts repositories list --project=plotpointe

# List Docker images (after you push some)
gcloud artifacts docker images list us-central1-docker.pkg.dev/plotpointe/botds
```

### View Service Accounts

```bash
# List all service accounts
gcloud iam service-accounts list --project=plotpointe

# View specific service account details
gcloud iam service-accounts describe github-actions-sa@plotpointe.iam.gserviceaccount.com
```

### View Terraform State

```bash
cd audit-deployment/phase-3/terraform

# View all outputs
terraform output

# View specific output
terraform output workload_identity_provider
terraform output artifact_registry_repository
```

---

## 🔄 Using the CI/CD Pipeline

### Automatic Deployment Flow

1. **Push to `develop` branch** → Deploys to Dev environment
2. **Push to `main` branch** → Deploys to Production environment
3. **Pull Request** → Runs tests only (no deployment)

### Example Workflow

```bash
# Make changes to your code
git checkout -b feature/my-new-feature

# Commit changes
git add .
git commit -m "Add new feature"

# Push to GitHub
git push origin feature/my-new-feature

# Create Pull Request on GitHub
# → Tests run automatically

# After PR is approved and merged to develop
# → Automatically deploys to Dev environment

# After testing in Dev, merge to main
# → Automatically deploys to Production
```

### Monitor Pipeline

1. Go to your GitHub repository
2. Click **"Actions"** tab
3. See all workflow runs
4. Click on a run to see details

---

## 🐳 Manual Docker Operations

### Build and Push Docker Image Manually

```bash
# Build Docker image
docker build -t botds:latest .

# Tag for Artifact Registry
docker tag botds:latest us-central1-docker.pkg.dev/plotpointe/botds/botds:latest

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/plotpointe/botds/botds:latest

# Verify image was pushed
gcloud artifacts docker images list us-central1-docker.pkg.dev/plotpointe/botds
```

### Pull and Run Docker Image

```bash
# Pull image from Artifact Registry
docker pull us-central1-docker.pkg.dev/plotpointe/botds/botds:latest

# Run locally with OpenAI API key
docker run --rm \
  -e OPENAI_API_KEY="your-key-here" \
  us-central1-docker.pkg.dev/plotpointe/botds/botds:latest \
  python -m cli.run --config configs/iris.yaml
```

---

## ☁️ Deploy to Cloud Run (Manual)

### Deploy to Dev Environment

```bash
gcloud run deploy botds-dev \
  --image=us-central1-docker.pkg.dev/plotpointe/botds/botds:latest \
  --platform=managed \
  --region=us-central1 \
  --service-account=botds-cloud-run-dev@plotpointe.iam.gserviceaccount.com \
  --set-secrets=OPENAI_API_KEY=botds-openai-api-key-dev:latest \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0
```

### Deploy to Production Environment

```bash
gcloud run deploy botds-prod \
  --image=us-central1-docker.pkg.dev/plotpointe/botds/botds:latest \
  --platform=managed \
  --region=us-central1 \
  --service-account=botds-cloud-run-prod@plotpointe.iam.gserviceaccount.com \
  --set-secrets=OPENAI_API_KEY=botds-openai-api-key-prod:latest \
  --no-allow-unauthenticated \
  --memory=4Gi \
  --cpu=4 \
  --timeout=600 \
  --max-instances=20 \
  --min-instances=1
```

### View Deployed Services

```bash
# List all Cloud Run services
gcloud run services list --project=plotpointe --region=us-central1

# Get service URL
gcloud run services describe botds-dev --region=us-central1 --format='value(status.url)'

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=botds-dev" \
  --limit=50 \
  --project=plotpointe
```

---

## 🔐 Managing Secrets

### Update a Secret

```bash
# Update dev secret
echo -n "new-api-key-here" | gcloud secrets versions add botds-openai-api-key-dev \
  --data-file=- \
  --project=plotpointe

# Verify new version was created
gcloud secrets versions list botds-openai-api-key-dev --project=plotpointe
```

### Rotate Secrets

```bash
# 1. Create new OpenAI API key at https://platform.openai.com/api-keys

# 2. Add new version to Secret Manager
echo -n "sk-proj-NEW-KEY-HERE" | gcloud secrets versions add botds-openai-api-key-prod \
  --data-file=- \
  --project=plotpointe

# 3. Cloud Run will automatically use the latest version on next deployment

# 4. Disable old version (optional)
gcloud secrets versions disable 1 --secret=botds-openai-api-key-prod --project=plotpointe
```

---

## 📊 Monitoring and Logs

### View Logs

```bash
# View all logs for the project
gcloud logging read --project=plotpointe --limit=50

# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" \
  --limit=50 \
  --project=plotpointe

# View Secret Manager access logs
gcloud logging read "protoPayload.serviceName=secretmanager.googleapis.com" \
  --limit=50 \
  --project=plotpointe

# Follow logs in real-time
gcloud logging tail --project=plotpointe
```

### View Metrics

```bash
# Open Cloud Console Monitoring
gcloud monitoring dashboards list --project=plotpointe

# View in browser
open "https://console.cloud.google.com/monitoring?project=plotpointe"
```

---

## 💰 Cost Management

### View Current Costs

```bash
# Open billing dashboard
open "https://console.cloud.google.com/billing?project=plotpointe"

# View cost breakdown
gcloud billing accounts list
```

### Set Up Budget Alerts

```bash
# Create budget (via Console)
open "https://console.cloud.google.com/billing/budgets?project=plotpointe"

# Recommended budgets:
# - Alert at $5/month (50%)
# - Alert at $10/month (100%)
# - Alert at $25/month (250%)
```

---

## 🛠️ Troubleshooting

### CI/CD Pipeline Fails

```bash
# Check GitHub Actions logs
# Go to: https://github.com/Axionis47/ai-data-scientist-agent/actions

# Common issues:
# 1. Tests failing → Check test output in Actions tab
# 2. Docker build failing → Check Dockerfile syntax
# 3. Permission denied → Check Workload Identity configuration
```

### Can't Access Secrets

```bash
# Verify you have permission
gcloud secrets get-iam-policy botds-openai-api-key-dev --project=plotpointe

# Grant yourself access (if needed)
gcloud secrets add-iam-policy-binding botds-openai-api-key-dev \
  --member="user:your-email@example.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=plotpointe
```

### Docker Push Fails

```bash
# Re-authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Verify you have permission
gcloud artifacts repositories get-iam-policy botds \
  --location=us-central1 \
  --project=plotpointe
```

---

## 📚 Useful Links

### GCP Console
- **Project Dashboard**: https://console.cloud.google.com/home/dashboard?project=plotpointe
- **Secret Manager**: https://console.cloud.google.com/security/secret-manager?project=plotpointe
- **Artifact Registry**: https://console.cloud.google.com/artifacts?project=plotpointe
- **Cloud Run**: https://console.cloud.google.com/run?project=plotpointe
- **IAM**: https://console.cloud.google.com/iam-admin/iam?project=plotpointe
- **Logs**: https://console.cloud.google.com/logs?project=plotpointe

### GitHub
- **Repository**: https://github.com/Axionis47/ai-data-scientist-agent
- **Actions**: https://github.com/Axionis47/ai-data-scientist-agent/actions

### Documentation
- **Phase 3 Complete**: `audit-deployment/phase-3/PHASE_3_COMPLETE.md`
- **GCP Secret Guide**: `audit-deployment/phase-3/GCP_SECRET_MANAGEMENT_GUIDE.md`
- **Terraform README**: `audit-deployment/phase-3/terraform/README.md`

---

## 🎯 Next Steps

1. **Push to GitHub** to trigger CI/CD pipeline
2. **Monitor GitHub Actions** to see pipeline run
3. **Deploy to Cloud Run** (automatic or manual)
4. **Set up monitoring** and alerts
5. **Configure budget alerts** to track costs

---

**Status**: ✅ Infrastructure deployed and ready to use

**Support**: See troubleshooting section or check documentation

**Cost**: Currently ~$0.28/month (infrastructure only)

