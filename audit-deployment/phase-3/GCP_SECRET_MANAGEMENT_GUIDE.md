# 🔐 GCP Secret Management Guide - OpenAI API Key

## Overview

This guide explains how to securely manage the OpenAI API key for the Bot Data Scientist project using **GCP Secret Manager** with **Workload Identity Federation** for keyless authentication from GitHub Actions.

---

## 🎯 Security Goals

1. **No secrets in code** - Never commit API keys to Git
2. **No service account keys** - Use Workload Identity instead
3. **Environment separation** - Different secrets for dev/staging/prod
4. **Audit logging** - Track all secret access
5. **Least privilege** - Minimal IAM permissions

---

## 📋 Prerequisites

### GCP Setup
- GCP project created
- Billing enabled
- `gcloud` CLI installed and authenticated
- Required APIs enabled:
  - Secret Manager API
  - IAM API
  - Cloud Resource Manager API

### GitHub Setup
- Repository with GitHub Actions enabled
- Admin access to repository settings

---

## 🚀 Step-by-Step Setup

### Step 1: Enable Required GCP APIs

```bash
# Set your GCP project ID
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable secretmanager.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable iamcredentials.googleapis.com
gcloud services enable sts.googleapis.com
```

---

### Step 2: Create Secrets in Secret Manager

```bash
# Create secret for development environment
echo -n "sk-your-dev-openai-api-key" | \
  gcloud secrets create botds-openai-api-key-dev \
  --data-file=- \
  --replication-policy="automatic"

# Create secret for staging environment
echo -n "sk-your-staging-openai-api-key" | \
  gcloud secrets create botds-openai-api-key-staging \
  --data-file=- \
  --replication-policy="automatic"

# Create secret for production environment
echo -n "sk-your-prod-openai-api-key" | \
  gcloud secrets create botds-openai-api-key-prod \
  --data-file=- \
  --replication-policy="automatic"

# Verify secrets were created
gcloud secrets list
```

**Important**: Replace `sk-your-*-openai-api-key` with actual OpenAI API keys.

---

### Step 3: Set Up Workload Identity Federation

This allows GitHub Actions to authenticate to GCP **without service account keys**.

```bash
# Set variables
export PROJECT_ID="your-gcp-project-id"
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
export POOL_NAME="github-actions-pool"
export PROVIDER_NAME="github-actions-provider"
export SERVICE_ACCOUNT_NAME="github-actions-sa"
export GITHUB_REPO="your-github-username/your-repo-name"

# Create Workload Identity Pool
gcloud iam workload-identity-pools create $POOL_NAME \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create Workload Identity Provider (GitHub OIDC)
gcloud iam workload-identity-pools providers create-oidc $PROVIDER_NAME \
  --location="global" \
  --workload-identity-pool=$POOL_NAME \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$GITHUB_REPO'"

# Create Service Account for GitHub Actions
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
  --display-name="GitHub Actions Service Account"

# Grant Secret Manager access to the service account
gcloud secrets add-iam-policy-binding botds-openai-api-key-dev \
  --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding botds-openai-api-key-staging \
  --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding botds-openai-api-key-prod \
  --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Allow GitHub Actions to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
  $SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_NAME/attribute.repository/$GITHUB_REPO"
```

---

### Step 4: Get Workload Identity Provider Resource Name

```bash
# Get the full resource name for GitHub Actions configuration
gcloud iam workload-identity-pools providers describe $PROVIDER_NAME \
  --location="global" \
  --workload-identity-pool=$POOL_NAME \
  --format="value(name)"
```

**Save this output** - you'll need it for GitHub Actions configuration.

Example output:
```
projects/123456789/locations/global/workloadIdentityPools/github-actions-pool/providers/github-actions-provider
```

---

### Step 5: Configure GitHub Repository Secrets

Add these secrets to your GitHub repository:

1. Go to: `Settings` → `Secrets and variables` → `Actions`
2. Add the following repository secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `GCP_PROJECT_ID` | `your-gcp-project-id` | Your GCP project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/.../providers/...` | From Step 4 |
| `GCP_SERVICE_ACCOUNT` | `github-actions-sa@PROJECT_ID.iam.gserviceaccount.com` | Service account email |

**Note**: Do NOT add the OpenAI API key to GitHub secrets - it will be fetched from GCP Secret Manager.

---

## 🔄 GitHub Actions Workflow

### Example Workflow with Secret Access

Create `.github/workflows/test-with-secrets.yml`:

```yaml
name: Test with GCP Secrets

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      id-token: write  # Required for OIDC authentication
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      
      - name: Access OpenAI API Key from Secret Manager
        id: secrets
        run: |
          # Fetch secret from GCP Secret Manager
          OPENAI_API_KEY=$(gcloud secrets versions access latest \
            --secret="botds-openai-api-key-dev" \
            --project="${{ secrets.GCP_PROJECT_ID }}")
          
          # Set as environment variable for subsequent steps
          echo "::add-mask::$OPENAI_API_KEY"  # Mask in logs
          echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> $GITHUB_ENV
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests with real API key
        env:
          OPENAI_API_KEY: ${{ env.OPENAI_API_KEY }}
        run: |
          python -m pytest tests/unit/ -v --cov=botds --cov-report=term
      
      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ env.OPENAI_API_KEY }}
        run: |
          python test_system.py
```

---

## 🐳 Docker Container with Secret Access

### Dockerfile with Secret Support

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY botds/ ./botds/
COPY cli/ ./cli/
COPY configs/ ./configs/
COPY schemas/ ./schemas/

# Create directories
RUN mkdir -p /app/cache /app/artifacts

# Set environment variables (secrets will be injected at runtime)
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import botds; print('OK')" || exit 1

# Run the application
ENTRYPOINT ["python", "-m", "cli.run"]
```

### Cloud Run Deployment with Secrets

```bash
# Deploy to Cloud Run with secret from Secret Manager
gcloud run deploy botds \
  --image=gcr.io/$PROJECT_ID/botds:latest \
  --platform=managed \
  --region=us-central1 \
  --set-secrets=OPENAI_API_KEY=botds-openai-api-key-prod:latest \
  --service-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com \
  --no-allow-unauthenticated
```

---

## 🔍 Accessing Secrets Locally (Development)

### Option 1: Using gcloud CLI

```bash
# Fetch secret and set as environment variable
export OPENAI_API_KEY=$(gcloud secrets versions access latest \
  --secret="botds-openai-api-key-dev")

# Run application
python -m cli.run --config configs/iris.yaml
```

### Option 2: Using Python Client Library

```python
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Fetch secret from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
openai_api_key = get_secret("your-project-id", "botds-openai-api-key-dev")
```

---

## 🔄 Secret Rotation

### Rotate OpenAI API Key

```bash
# Create new version of secret
echo -n "sk-new-openai-api-key" | \
  gcloud secrets versions add botds-openai-api-key-prod \
  --data-file=-

# Verify new version
gcloud secrets versions list botds-openai-api-key-prod

# Disable old version (after testing)
gcloud secrets versions disable 1 --secret=botds-openai-api-key-prod

# Destroy old version (after grace period)
gcloud secrets versions destroy 1 --secret=botds-openai-api-key-prod
```

---

## 📊 Audit Logging

### View Secret Access Logs

```bash
# View who accessed secrets in the last 7 days
gcloud logging read \
  'resource.type="secretmanager.googleapis.com/Secret"
   AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"' \
  --limit=50 \
  --format=json \
  --freshness=7d
```

---

## ✅ Security Checklist

- [ ] Secrets created in Secret Manager for all environments
- [ ] Workload Identity Federation configured
- [ ] Service account has minimal permissions (secretAccessor only)
- [ ] GitHub Actions authenticate without service account keys
- [ ] Secrets are masked in GitHub Actions logs
- [ ] Audit logging enabled
- [ ] `.env` file in `.gitignore`
- [ ] No API keys in code or GitHub
- [ ] Secret rotation process documented
- [ ] Access logs monitored

---

## 🚨 Troubleshooting

### Issue: "Permission denied" when accessing secret

**Solution**: Verify IAM permissions:
```bash
gcloud secrets get-iam-policy botds-openai-api-key-dev
```

### Issue: GitHub Actions can't authenticate to GCP

**Solution**: Verify Workload Identity setup:
```bash
gcloud iam workload-identity-pools providers describe $PROVIDER_NAME \
  --location="global" \
  --workload-identity-pool=$POOL_NAME
```

### Issue: Secret not found

**Solution**: List all secrets:
```bash
gcloud secrets list
```

---

## 📚 Additional Resources

- [GCP Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [GitHub Actions + GCP](https://github.com/google-github-actions/auth)
- [OpenAI API Key Security](https://platform.openai.com/docs/guides/safety-best-practices)

---

**Status**: 📋 **GUIDE COMPLETE - READY FOR IMPLEMENTATION**
**Next**: Follow steps 1-5 to set up GCP Secret Manager

