#!/bin/bash
# Demo script for staging environment
# Usage: ./scripts/demo_staging.sh
# Override base URL: STAGING_BASE_URL=https://... ./scripts/demo_staging.sh

set -e

# Default staging URL
STAGING_BASE_URL="${STAGING_BASE_URL:-https://sdlc-api-staging-du6qod3mja-uc.a.run.app}"

echo "=== SDLC API Staging Demo ==="
echo "Base URL: ${STAGING_BASE_URL}"
echo ""

# Step 1: Health check
echo ">>> Step 1: Health check"
HEALTH=$(curl -sf "${STAGING_BASE_URL}/health")
echo "Response: ${HEALTH}"
echo ""

# Step 2: Version check
echo ">>> Step 2: Version check"
VERSION=$(curl -sf "${STAGING_BASE_URL}/version")
echo "Response: ${VERSION}"
GIT_SHA=$(echo "$VERSION" | jq -r '.git_sha')
APP_ENV=$(echo "$VERSION" | jq -r '.app_env')
echo "Git SHA: ${GIT_SHA}"
echo "App Env: ${APP_ENV}"
echo ""

# Step 3: Upload context document
echo ">>> Step 3: Upload context document"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCX_PATH="${PROJECT_ROOT}/docs/samples/context_template.docx"

if [ ! -f "$DOCX_PATH" ]; then
    echo "ERROR: Context document not found at ${DOCX_PATH}"
    exit 1
fi

UPLOAD_DOC=$(curl -sf -X POST "${STAGING_BASE_URL}/upload_context_doc" \
    -F "file=@${DOCX_PATH}")
echo "Response: ${UPLOAD_DOC}"
DOC_ID=$(echo "$UPLOAD_DOC" | jq -r '.doc_id')
echo "Doc ID: ${DOC_ID}"
echo ""

# Step 4: Upload dataset
echo ">>> Step 4: Upload dataset"
CSV_PATH="${PROJECT_ROOT}/services/api/tests/fixtures/causal_binary_treatment.csv"

if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: Dataset not found at ${CSV_PATH}"
    exit 1
fi

UPLOAD_DS=$(curl -sf -X POST "${STAGING_BASE_URL}/upload_dataset" \
    -F "file=@${CSV_PATH}")
echo "Response: ${UPLOAD_DS}"
DATASET_ID=$(echo "$UPLOAD_DS" | jq -r '.dataset_id')
echo "Dataset ID: ${DATASET_ID}"
echo ""

# Step 5a: Causal question WITHOUT confirmations
echo ">>> Step 5a: Causal question WITHOUT confirmations (expect NEEDS_CLARIFICATION or diagnostics)"
ASK_NO_CONFIRM=$(curl -sf -X POST "${STAGING_BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d "{
        \"question\": \"What is the causal effect of treatment on outcome?\",
        \"doc_id\": \"${DOC_ID}\",
        \"dataset_id\": \"${DATASET_ID}\",
        \"causal_spec_override\": {
            \"treatment\": \"treatment\",
            \"outcome\": \"outcome\",
            \"confounders\": [\"age\", \"income\", \"prior_purchases\"]
        }
    }")
echo "Response (trimmed):"
echo "$ASK_NO_CONFIRM" | jq '{route: .router_decision.route, trace_id: .trace_id, artifact_types: [.artifacts[].type]}'
TRACE_ID_1=$(echo "$ASK_NO_CONFIRM" | jq -r '.trace_id')
echo "Trace ID: ${TRACE_ID_1}"
echo ""

# Step 5b: Causal question WITH confirmations
echo ">>> Step 5b: Causal question WITH confirmations (expect CausalEstimateArtifact)"
ASK_WITH_CONFIRM=$(curl -sf -X POST "${STAGING_BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d "{
        \"question\": \"What is the causal effect of treatment on outcome?\",
        \"doc_id\": \"${DOC_ID}\",
        \"dataset_id\": \"${DATASET_ID}\",
        \"causal_spec_override\": {
            \"treatment\": \"treatment\",
            \"outcome\": \"outcome\",
            \"confounders\": [\"age\", \"income\", \"prior_purchases\"]
        },
        \"causal_confirmations\": {
            \"assignment_mechanism\": \"randomized\",
            \"interference\": \"no_interference\",
            \"missing_data_policy\": \"listwise_delete\",
            \"ok_to_estimate\": true
        }
    }")
echo "Response (trimmed):"
echo "$ASK_WITH_CONFIRM" | jq '{route: .router_decision.route, trace_id: .trace_id, artifact_types: [.artifacts[].type]}'
TRACE_ID_2=$(echo "$ASK_WITH_CONFIRM" | jq -r '.trace_id')
echo "Trace ID: ${TRACE_ID_2}"

# Check for causal estimate
ESTIMATE=$(echo "$ASK_WITH_CONFIRM" | jq '.artifacts[] | select(.type == "causal_estimate")')
if [ -n "$ESTIMATE" ]; then
    echo ""
    echo ">>> Causal Estimate Found:"
    echo "$ESTIMATE" | jq '{method, estimate, ci_low, ci_high, n_used}'
else
    echo ""
    echo ">>> No causal estimate in response (may need dataset with proper treatment)"
fi

echo ""
echo "=== Demo Complete ==="
echo "Trace IDs:"
echo "  - Without confirmations: ${TRACE_ID_1}"
echo "  - With confirmations: ${TRACE_ID_2}"

