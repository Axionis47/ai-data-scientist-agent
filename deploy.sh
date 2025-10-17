#!/bin/bash
# Comprehensive deployment script for AI Data Scientist Agent to GCP Cloud Run
# This script handles the complete deployment pipeline

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from terraform.tfvars
PROJECT_ID="plotpointe"
REGION="us-central1"
GITHUB_REPO="Axionis47/ai-data-scientist-agent"
ARTIFACT_REGISTRY="us-central1-docker.pkg.dev/plotpointe/botds"

# Service names
BACKEND_SERVICE="botds-backend"
FRONTEND_SERVICE="botds-frontend"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_header() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install it first."
        exit 1
    fi
    print_status "gcloud CLI found"
    
    # Check terraform
    if ! command -v terraform &> /dev/null; then
        print_error "terraform not found. Please install it first."
        exit 1
    fi
    print_status "terraform found"
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        print_error "docker not found. Please install it first."
        exit 1
    fi
    print_status "docker found"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    fi
    print_status "Docker daemon is running"
    
    # Check current GCP project
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
        print_warning "Current GCP project is $CURRENT_PROJECT, switching to $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    print_status "GCP project set to $PROJECT_ID"
    
    # Authenticate with GCP
    print_info "Checking GCP authentication..."
    if ! gcloud auth print-access-token &> /dev/null; then
        print_warning "Not authenticated with GCP. Running gcloud auth login..."
        gcloud auth login
    fi
    print_status "Authenticated with GCP"
    
    # Configure Docker for Artifact Registry
    print_info "Configuring Docker for Artifact Registry..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
    print_status "Docker configured for Artifact Registry"
}

# Apply Terraform configuration
apply_terraform() {
    print_header "Applying Terraform Configuration"
    
    cd audit-deployment/phase-3/terraform
    
    # Initialize Terraform
    print_info "Initializing Terraform..."
    terraform init
    print_status "Terraform initialized"
    
    # Validate configuration
    print_info "Validating Terraform configuration..."
    terraform validate
    print_status "Terraform configuration valid"
    
    # Plan
    print_info "Creating Terraform plan..."
    terraform plan -out=tfplan
    print_status "Terraform plan created"
    
    # Apply
    print_info "Applying Terraform configuration..."
    terraform apply tfplan
    print_status "Terraform applied successfully"
    
    cd ../../..
}

# Build and push Docker images
build_and_push_images() {
    print_header "Building and Pushing Docker Images"
    
    # Build backend image
    print_info "Building backend Docker image..."
    docker build --platform linux/amd64 -t ${ARTIFACT_REGISTRY}/backend:latest ./backend
    print_status "Backend image built"

    # Build frontend image
    print_info "Building frontend Docker image..."
    docker build --platform linux/amd64 -t ${ARTIFACT_REGISTRY}/frontend:latest ./frontend
    print_status "Frontend image built"
    
    # Push backend image
    print_info "Pushing backend image to Artifact Registry..."
    docker push ${ARTIFACT_REGISTRY}/backend:latest
    print_status "Backend image pushed"
    
    # Push frontend image
    print_info "Pushing frontend image to Artifact Registry..."
    docker push ${ARTIFACT_REGISTRY}/frontend:latest
    print_status "Frontend image pushed"
}

# Deploy backend to Cloud Run
deploy_backend() {
    print_header "Deploying Backend to Cloud Run"

    # Get frontend URL if it exists for CORS
    CORS_ORIGINS="http://localhost:3000"
    if [ -f .frontend_url ]; then
        FRONTEND_URL=$(cat .frontend_url)
        CORS_ORIGINS="${CORS_ORIGINS},${FRONTEND_URL}"
    fi

    # Create env vars file
    cat > backend-env.yaml << EOF
MAX_CONCURRENT_JOBS: "2"
EDA_TIMEOUT_S: "300"
MODEL_TIMEOUT_S: "600"
REPORT_TIMEOUT_S: "300"
REPORT_JSON_FIRST: "true"
ALLOWED_ORIGINS: "${CORS_ORIGINS}"
EOF

    print_info "Deploying backend service with CORS origins: $CORS_ORIGINS"
    gcloud run deploy ${BACKEND_SERVICE} \
        --image ${ARTIFACT_REGISTRY}/backend:latest \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --service-account botds-cloud-run-prod@${PROJECT_ID}.iam.gserviceaccount.com \
        --env-vars-file backend-env.yaml \
        --set-secrets "OPENAI_API_KEY=botds-openai-api-key-prod:latest" \
        --memory 2Gi \
        --cpu 2 \
        --timeout 900 \
        --max-instances 10 \
        --min-instances 0 \
        --port 8000

    # Get backend URL
    BACKEND_URL=$(gcloud run services describe ${BACKEND_SERVICE} --region ${REGION} --format 'value(status.url)')
    print_status "Backend deployed at: $BACKEND_URL"

    echo "$BACKEND_URL" > .backend_url
}

# Deploy frontend to Cloud Run
deploy_frontend() {
    print_header "Deploying Frontend to Cloud Run"
    
    # Read backend URL
    if [ ! -f .backend_url ]; then
        print_error "Backend URL not found. Please deploy backend first."
        exit 1
    fi
    BACKEND_URL=$(cat .backend_url)
    
    print_info "Deploying frontend service with backend URL: $BACKEND_URL"
    gcloud run deploy ${FRONTEND_SERVICE} \
        --image ${ARTIFACT_REGISTRY}/frontend:latest \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --service-account botds-cloud-run-prod@${PROJECT_ID}.iam.gserviceaccount.com \
        --set-env-vars "NEXT_PUBLIC_API_URL=${BACKEND_URL}" \
        --memory 512Mi \
        --cpu 1 \
        --timeout 300 \
        --max-instances 10 \
        --min-instances 0 \
        --port 3000
    
    # Get frontend URL
    FRONTEND_URL=$(gcloud run services describe ${FRONTEND_SERVICE} --region ${REGION} --format 'value(status.url)')
    print_status "Frontend deployed at: $FRONTEND_URL"
    
    echo "$FRONTEND_URL" > .frontend_url
}

# Update backend CORS to allow frontend URL
update_backend_cors() {
    print_header "Updating Backend CORS Configuration"

    if [ ! -f .frontend_url ] || [ ! -f .backend_url ]; then
        print_warning "Frontend or Backend URL not found. Skipping CORS update."
        return
    fi

    FRONTEND_URL=$(cat .frontend_url)
    BACKEND_URL=$(cat .backend_url)
    CORS_ORIGINS="http://localhost:3000,${FRONTEND_URL}"

    # Create env vars file
    cat > backend-env.yaml << EOF
MAX_CONCURRENT_JOBS: "2"
EDA_TIMEOUT_S: "300"
MODEL_TIMEOUT_S: "600"
REPORT_TIMEOUT_S: "300"
REPORT_JSON_FIRST: "true"
ALLOWED_ORIGINS: "${CORS_ORIGINS}"
EOF

    print_info "Redeploying backend with updated CORS origins: $CORS_ORIGINS"
    gcloud run deploy ${BACKEND_SERVICE} \
        --image ${ARTIFACT_REGISTRY}/backend:latest \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --service-account botds-cloud-run-prod@${PROJECT_ID}.iam.gserviceaccount.com \
        --env-vars-file backend-env.yaml \
        --set-secrets "OPENAI_API_KEY=botds-openai-api-key-prod:latest" \
        --memory 2Gi \
        --cpu 2 \
        --timeout 900 \
        --max-instances 10 \
        --min-instances 0 \
        --port 8000

    print_status "Backend CORS updated to allow frontend URL"
}

# Verify deployment
verify_deployment() {
    print_header "Verifying Deployment"
    
    if [ ! -f .backend_url ] || [ ! -f .frontend_url ]; then
        print_error "Deployment URLs not found. Please complete deployment first."
        exit 1
    fi
    
    BACKEND_URL=$(cat .backend_url)
    FRONTEND_URL=$(cat .frontend_url)
    
    # Test backend health
    print_info "Testing backend health endpoint..."
    if curl -s -f "${BACKEND_URL}/health" > /dev/null; then
        print_status "Backend health check passed"
    else
        print_error "Backend health check failed"
        exit 1
    fi
    
    # Test frontend
    print_info "Testing frontend..."
    if curl -s -f "${FRONTEND_URL}" > /dev/null; then
        print_status "Frontend is accessible"
    else
        print_error "Frontend is not accessible"
        exit 1
    fi
    
    print_header "Deployment Complete!"
    echo ""
    print_status "Backend URL: $BACKEND_URL"
    print_status "Frontend URL: $FRONTEND_URL"
    echo ""
    print_info "You can now access your AI Data Scientist at:"
    echo -e "${GREEN}${FRONTEND_URL}${NC}"
    echo ""
}

# Main deployment flow
main() {
    print_header "AI Data Scientist - GCP Cloud Run Deployment"
    
    # Parse command line arguments
    SKIP_TERRAFORM=false
    SKIP_BUILD=false
    SKIP_DEPLOY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-terraform)
                SKIP_TERRAFORM=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-deploy)
                SKIP_DEPLOY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Usage: $0 [--skip-terraform] [--skip-build] [--skip-deploy]"
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    
    if [ "$SKIP_TERRAFORM" = false ]; then
        apply_terraform
    else
        print_warning "Skipping Terraform apply"
    fi
    
    if [ "$SKIP_BUILD" = false ]; then
        build_and_push_images
    else
        print_warning "Skipping Docker build and push"
    fi
    
    if [ "$SKIP_DEPLOY" = false ]; then
        deploy_backend
        deploy_frontend
        update_backend_cors
        verify_deployment
    else
        print_warning "Skipping Cloud Run deployment"
    fi
}

# Run main function
main "$@"

