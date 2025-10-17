# Terraform Configuration for Bot Data Scientist GCP Infrastructure

This directory contains Terraform configuration to provision GCP infrastructure for the Bot Data Scientist project.

## 📋 What This Creates

### Secret Manager
- `botds-openai-api-key-dev` - Development OpenAI API key
- `botds-openai-api-key-staging` - Staging OpenAI API key
- `botds-openai-api-key-prod` - Production OpenAI API key

### Workload Identity Federation
- Workload Identity Pool for GitHub Actions
- OIDC Provider for GitHub
- Service account for GitHub Actions with secret access

### Artifact Registry
- Docker repository for container images

### Service Accounts
- `github-actions-sa` - For GitHub Actions workflows
- `botds-cloud-run-dev` - For Cloud Run dev environment
- `botds-cloud-run-staging` - For Cloud Run staging environment
- `botds-cloud-run-prod` - For Cloud Run production environment

### IAM Permissions
- Secret Manager access for all service accounts
- Artifact Registry write access for GitHub Actions
- Workload Identity bindings

---

## 🚀 Prerequisites

1. **GCP Project**: Create a GCP project
2. **Billing**: Enable billing on the project
3. **gcloud CLI**: Install and authenticate
   ```bash
   gcloud auth application-default login
   ```
4. **Terraform**: Install Terraform >= 1.0
   ```bash
   brew install terraform  # macOS
   ```

---

## 📝 Setup Instructions

### Step 1: Configure Variables

```bash
# Copy the example file
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
# IMPORTANT: DO NOT commit terraform.tfvars to Git!
```

Edit `terraform.tfvars`:
```hcl
project_id = "your-gcp-project-id"
region     = "us-central1"
github_repo = "your-username/your-repo"
openai_api_key_dev     = "sk-..."
openai_api_key_staging = "sk-..."
openai_api_key_prod    = "sk-..."
```

### Step 2: Initialize Terraform

```bash
terraform init
```

### Step 3: Review the Plan

```bash
terraform plan
```

Review the output to ensure everything looks correct.

### Step 4: Apply the Configuration

```bash
terraform apply
```

Type `yes` when prompted.

### Step 5: Save Outputs

```bash
# Save outputs to a file for reference
terraform output > outputs.txt

# Get specific outputs
terraform output workload_identity_provider
terraform output github_actions_service_account
```

---

## 📊 Outputs

After applying, Terraform will output:

- `workload_identity_provider` - Use this in GitHub Actions
- `github_actions_service_account` - Use this in GitHub Actions
- `artifact_registry_repository` - Docker image repository URL
- `cloud_run_service_accounts` - Service accounts for Cloud Run
- `secret_names` - Secret Manager secret names

---

## 🔐 Security Best Practices

### 1. Protect terraform.tfvars

Add to `.gitignore`:
```
terraform.tfvars
*.tfstate
*.tfstate.backup
.terraform/
```

### 2. Use Remote State (Production)

For production, use GCS backend:

```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "botds/state"
  }
}
```

### 3. Separate Environments

For production, consider separate Terraform workspaces or directories:
```
terraform/
├── dev/
├── staging/
└── prod/
```

---

## 🔄 Common Operations

### Update a Secret

```bash
# Update the variable in terraform.tfvars
# Then apply
terraform apply -target=google_secret_manager_secret_version.openai_api_key_dev
```

### View Current State

```bash
terraform show
```

### List Resources

```bash
terraform state list
```

### Destroy Everything (CAREFUL!)

```bash
terraform destroy
```

---

## 🔍 Troubleshooting

### Error: API not enabled

**Solution**: Enable the API manually:
```bash
gcloud services enable secretmanager.googleapis.com
```

### Error: Permission denied

**Solution**: Ensure you have the required roles:
- `roles/owner` or
- `roles/editor` + `roles/secretmanager.admin` + `roles/iam.securityAdmin`

### Error: Workload Identity Pool already exists

**Solution**: Import existing resource:
```bash
terraform import google_iam_workload_identity_pool.github_actions \
  projects/PROJECT_ID/locations/global/workloadIdentityPools/github-actions-pool
```

---

## 📚 Next Steps

After applying this Terraform configuration:

1. **Configure GitHub Secrets**:
   - Add `GCP_PROJECT_ID`
   - Add `GCP_WORKLOAD_IDENTITY_PROVIDER` (from Terraform output)
   - Add `GCP_SERVICE_ACCOUNT` (from Terraform output)

2. **Update GitHub Actions Workflow**:
   - Use the new Workload Identity provider
   - Access secrets from Secret Manager

3. **Test the Setup**:
   - Run a GitHub Actions workflow
   - Verify secret access works

---

## 🔗 Resources

- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GCP Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)

---

**Status**: ✅ Ready to use
**Last Updated**: 2025-10-17

