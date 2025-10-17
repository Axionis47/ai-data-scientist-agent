# Bot Data Scientist - GCP Infrastructure
# Terraform configuration for Secret Manager, Workload Identity, and Cloud Run

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  # Backend configuration for state storage
  # Uncomment and configure for production use
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "botds/state"
  # }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "GitHub repository in format 'owner/repo'"
  type        = string
}

variable "openai_api_key_dev" {
  description = "OpenAI API Key for development"
  type        = string
  sensitive   = true
}

variable "openai_api_key_staging" {
  description = "OpenAI API Key for staging"
  type        = string
  sensitive   = true
}

variable "openai_api_key_prod" {
  description = "OpenAI API Key for production"
  type        = string
  sensitive   = true
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Data source for project number
data "google_project" "project" {
  project_id = var.project_id
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ])
  
  service            = each.value
  disable_on_destroy = false
}

# ============================================================================
# SECRET MANAGER
# ============================================================================

# Development environment secret
resource "google_secret_manager_secret" "openai_api_key_dev" {
  secret_id = "botds-openai-api-key-dev"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_version" "openai_api_key_dev" {
  secret      = google_secret_manager_secret.openai_api_key_dev.id
  secret_data = var.openai_api_key_dev
}

# Staging environment secret
resource "google_secret_manager_secret" "openai_api_key_staging" {
  secret_id = "botds-openai-api-key-staging"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_version" "openai_api_key_staging" {
  secret      = google_secret_manager_secret.openai_api_key_staging.id
  secret_data = var.openai_api_key_staging
}

# Production environment secret
resource "google_secret_manager_secret" "openai_api_key_prod" {
  secret_id = "botds-openai-api-key-prod"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_version" "openai_api_key_prod" {
  secret      = google_secret_manager_secret.openai_api_key_prod.id
  secret_data = var.openai_api_key_prod
}

# ============================================================================
# WORKLOAD IDENTITY FEDERATION (GitHub Actions)
# ============================================================================

# Workload Identity Pool
resource "google_iam_workload_identity_pool" "github_actions" {
  workload_identity_pool_id = "github-actions-pool"
  display_name              = "GitHub Actions Pool"
  description               = "Workload Identity Pool for GitHub Actions"
  
  depends_on = [google_project_service.required_apis]
}

# Workload Identity Provider (GitHub OIDC)
resource "google_iam_workload_identity_pool_provider" "github_actions" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_actions.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions-provider"
  display_name                       = "GitHub Actions Provider"
  description                        = "OIDC provider for GitHub Actions"
  
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }
  
  attribute_condition = "assertion.repository == '${var.github_repo}'"
  
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Service Account for GitHub Actions
resource "google_service_account" "github_actions" {
  account_id   = "github-actions-sa"
  display_name = "GitHub Actions Service Account"
  description  = "Service account for GitHub Actions workflows"
}

# Allow GitHub Actions to impersonate the service account
resource "google_service_account_iam_member" "github_actions_workload_identity" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_actions.name}/attribute.repository/${var.github_repo}"
}

# Grant Secret Manager access to GitHub Actions service account
resource "google_secret_manager_secret_iam_member" "github_actions_dev" {
  secret_id = google_secret_manager_secret.openai_api_key_dev.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.github_actions.email}"
}

resource "google_secret_manager_secret_iam_member" "github_actions_staging" {
  secret_id = google_secret_manager_secret.openai_api_key_staging.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.github_actions.email}"
}

resource "google_secret_manager_secret_iam_member" "github_actions_prod" {
  secret_id = google_secret_manager_secret.openai_api_key_prod.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.github_actions.email}"
}

# ============================================================================
# ARTIFACT REGISTRY
# ============================================================================

resource "google_artifact_registry_repository" "botds" {
  location      = var.region
  repository_id = "botds"
  description   = "Docker repository for Bot Data Scientist"
  format        = "DOCKER"
  
  depends_on = [google_project_service.required_apis]
}

# Grant GitHub Actions permission to push images
resource "google_artifact_registry_repository_iam_member" "github_actions_writer" {
  location   = google_artifact_registry_repository.botds.location
  repository = google_artifact_registry_repository.botds.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_actions.email}"
}

# ============================================================================
# CLOUD RUN SERVICE ACCOUNTS
# ============================================================================

# Service account for Cloud Run (dev)
resource "google_service_account" "cloud_run_dev" {
  account_id   = "botds-cloud-run-dev"
  display_name = "Bot DS Cloud Run Dev"
  description  = "Service account for Cloud Run (dev environment)"
}

# Grant secret access to Cloud Run dev
resource "google_secret_manager_secret_iam_member" "cloud_run_dev" {
  secret_id = google_secret_manager_secret.openai_api_key_dev.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_dev.email}"
}

# Service account for Cloud Run (staging)
resource "google_service_account" "cloud_run_staging" {
  account_id   = "botds-cloud-run-staging"
  display_name = "Bot DS Cloud Run Staging"
  description  = "Service account for Cloud Run (staging environment)"
}

# Grant secret access to Cloud Run staging
resource "google_secret_manager_secret_iam_member" "cloud_run_staging" {
  secret_id = google_secret_manager_secret.openai_api_key_staging.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_staging.email}"
}

# Service account for Cloud Run (prod)
resource "google_service_account" "cloud_run_prod" {
  account_id   = "botds-cloud-run-prod"
  display_name = "Bot DS Cloud Run Prod"
  description  = "Service account for Cloud Run (production environment)"
}

# Grant secret access to Cloud Run prod
resource "google_secret_manager_secret_iam_member" "cloud_run_prod" {
  secret_id = google_secret_manager_secret.openai_api_key_prod.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_prod.email}"
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "project_number" {
  description = "GCP Project Number"
  value       = data.google_project.project.number
}

output "workload_identity_provider" {
  description = "Workload Identity Provider resource name for GitHub Actions"
  value       = google_iam_workload_identity_pool_provider.github_actions.name
}

output "github_actions_service_account" {
  description = "GitHub Actions service account email"
  value       = google_service_account.github_actions.email
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.botds.repository_id}"
}

output "cloud_run_service_accounts" {
  description = "Cloud Run service account emails"
  value = {
    dev     = google_service_account.cloud_run_dev.email
    staging = google_service_account.cloud_run_staging.email
    prod    = google_service_account.cloud_run_prod.email
  }
}

output "secret_names" {
  description = "Secret Manager secret names"
  value = {
    dev     = google_secret_manager_secret.openai_api_key_dev.secret_id
    staging = google_secret_manager_secret.openai_api_key_staging.secret_id
    prod    = google_secret_manager_secret.openai_api_key_prod.secret_id
  }
}

