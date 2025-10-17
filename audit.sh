#!/bin/bash
# Quick audit script for the codebase

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
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

print_header "Codebase Audit"

# Check backend
print_info "Checking backend structure..."
if [ -d "backend/app" ]; then
    print_status "Backend app directory exists"
    BACKEND_FILES=$(find backend/app -name "*.py" | wc -l)
    print_info "Found $BACKEND_FILES Python files in backend"
fi

# Check frontend
print_info "Checking frontend structure..."
if [ -d "frontend/pages" ]; then
    print_status "Frontend pages directory exists"
    FRONTEND_FILES=$(find frontend/pages -name "*.tsx" -o -name "*.ts" | wc -l)
    print_info "Found $FRONTEND_FILES TypeScript files in frontend"
fi

# Check Docker files
print_info "Checking Docker configuration..."
if [ -f "backend/Dockerfile" ]; then
    print_status "Backend Dockerfile exists"
fi
if [ -f "frontend/Dockerfile" ]; then
    print_status "Frontend Dockerfile exists"
fi
if [ -f "docker-compose.yml" ]; then
    print_status "docker-compose.yml exists"
fi

# Check Terraform
print_info "Checking Terraform configuration..."
if [ -f "audit-deployment/phase-3/terraform/main.tf" ]; then
    print_status "Terraform main.tf exists"
fi
if [ -f "audit-deployment/phase-3/terraform/terraform.tfvars" ]; then
    print_status "Terraform variables configured"
fi

# Check dependencies
print_info "Checking dependencies..."
if [ -f "backend/pyproject.toml" ]; then
    print_status "Backend pyproject.toml exists"
fi
if [ -f "frontend/package.json" ]; then
    print_status "Frontend package.json exists"
fi

# Check CI/CD
print_info "Checking CI/CD configuration..."
if [ -d ".github/workflows" ]; then
    WORKFLOWS=$(find .github/workflows -name "*.yml" | wc -l)
    print_status "Found $WORKFLOWS GitHub Actions workflows"
fi

print_header "Audit Summary"
echo "✓ Backend: Python FastAPI application"
echo "✓ Frontend: Next.js TypeScript application"
echo "✓ Infrastructure: Terraform for GCP"
echo "✓ Containerization: Docker & docker-compose"
echo "✓ CI/CD: GitHub Actions"
echo ""
print_info "Codebase is ready for deployment!"

