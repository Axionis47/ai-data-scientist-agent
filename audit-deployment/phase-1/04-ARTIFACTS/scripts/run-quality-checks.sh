#!/bin/bash
# Phase 1: Code Quality Analysis Script
# This script runs all code quality checks and generates reports

set -e  # Exit on error

echo "=================================================="
echo "Phase 1: Code Quality Analysis"
echo "=================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_section() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

# Output directory
OUTPUT_DIR="audit-deployment/phase-1/04-ARTIFACTS"
mkdir -p "$OUTPUT_DIR"

# Start timestamp
START_TIME=$(date +%s)

# 1. Type Checking with mypy
print_section "1. Running Type Checking (mypy)"
mypy botds/ cli/ --html-report "$OUTPUT_DIR/mypy-report" --txt-report "$OUTPUT_DIR" --no-error-summary 2>&1 | tee "$OUTPUT_DIR/mypy-output.txt" || true
print_status "Type checking complete"

# 2. Linting with pylint
print_section "2. Running Linting (pylint)"
pylint botds/ cli/ --output-format=json > "$OUTPUT_DIR/pylint-report.json" 2>&1 || true
pylint botds/ cli/ --output-format=text > "$OUTPUT_DIR/pylint-report.txt" 2>&1 || true
PYLINT_SCORE=$(grep "Your code has been rated" "$OUTPUT_DIR/pylint-report.txt" | awk '{print $7}' || echo "N/A")
print_status "Linting complete - Score: $PYLINT_SCORE"

# 3. Code Formatting Check with black
print_section "3. Checking Code Formatting (black)"
black --check --diff botds/ cli/ tests/ > "$OUTPUT_DIR/black-report.txt" 2>&1 || true
BLACK_STATUS=$?
if [ $BLACK_STATUS -eq 0 ]; then
    print_status "Code formatting: PASS"
else
    echo "Code formatting: NEEDS FORMATTING (see black-report.txt)"
fi

# 4. Import Sorting Check with isort
print_section "4. Checking Import Sorting (isort)"
isort --check-only --diff botds/ cli/ tests/ > "$OUTPUT_DIR/isort-report.txt" 2>&1 || true
ISORT_STATUS=$?
if [ $ISORT_STATUS -eq 0 ]; then
    print_status "Import sorting: PASS"
else
    echo "Import sorting: NEEDS SORTING (see isort-report.txt)"
fi

# 5. Complexity Analysis with radon
print_section "5. Running Complexity Analysis (radon)"

# Cyclomatic complexity
radon cc botds/ cli/ -a -s -j > "$OUTPUT_DIR/complexity.json" 2>&1 || true
radon cc botds/ cli/ -a -s > "$OUTPUT_DIR/complexity.txt" 2>&1 || true
print_status "Cyclomatic complexity analysis complete"

# Maintainability index
radon mi botds/ cli/ -s -j > "$OUTPUT_DIR/maintainability.json" 2>&1 || true
radon mi botds/ cli/ -s > "$OUTPUT_DIR/maintainability.txt" 2>&1 || true
print_status "Maintainability index analysis complete"

# Raw metrics
radon raw botds/ cli/ -s -j > "$OUTPUT_DIR/raw-metrics.json" 2>&1 || true
radon raw botds/ cli/ -s > "$OUTPUT_DIR/raw-metrics.txt" 2>&1 || true
print_status "Raw metrics analysis complete"

# 6. Style Check with flake8
print_section "6. Running Style Check (flake8)"
flake8 botds/ cli/ tests/ --statistics --output-file="$OUTPUT_DIR/flake8-report.txt" 2>&1 || true
flake8 botds/ cli/ tests/ --format=json --output-file="$OUTPUT_DIR/flake8-report.json" 2>&1 || true
FLAKE8_ERRORS=$(grep -c "^" "$OUTPUT_DIR/flake8-report.txt" 2>/dev/null || echo "0")
print_status "Style check complete - Issues found: $FLAKE8_ERRORS"

# 7. Count lines of code
print_section "7. Counting Lines of Code"
echo "Lines of Code Analysis" > "$OUTPUT_DIR/loc-report.txt"
echo "======================" >> "$OUTPUT_DIR/loc-report.txt"
echo "" >> "$OUTPUT_DIR/loc-report.txt"

# Total LOC
TOTAL_LOC=$(find botds/ cli/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
echo "Total Lines of Code: $TOTAL_LOC" >> "$OUTPUT_DIR/loc-report.txt"

# Per directory
echo "" >> "$OUTPUT_DIR/loc-report.txt"
echo "Lines per directory:" >> "$OUTPUT_DIR/loc-report.txt"
find botds/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print "  botds/: " $1}' >> "$OUTPUT_DIR/loc-report.txt"
find cli/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print "  cli/: " $1}' >> "$OUTPUT_DIR/loc-report.txt"

# Test LOC
TEST_LOC=$(find tests/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
echo "  tests/: $TEST_LOC" >> "$OUTPUT_DIR/loc-report.txt"

print_status "Lines of code: $TOTAL_LOC (excluding tests)"

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Summary
print_section "Quality Check Summary"
echo "Duration: ${DURATION}s"
echo ""
echo "Reports generated in: $OUTPUT_DIR"
echo "  - mypy-report/index.html (type checking)"
echo "  - pylint-report.txt (linting - Score: $PYLINT_SCORE)"
echo "  - black-report.txt (formatting)"
echo "  - isort-report.txt (import sorting)"
echo "  - complexity.txt (cyclomatic complexity)"
echo "  - maintainability.txt (maintainability index)"
echo "  - flake8-report.txt (style violations)"
echo "  - loc-report.txt (lines of code)"
echo ""

# Generate summary JSON
cat > "$OUTPUT_DIR/quality-summary.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $DURATION,
  "metrics": {
    "total_loc": $TOTAL_LOC,
    "test_loc": $TEST_LOC,
    "pylint_score": "$PYLINT_SCORE",
    "flake8_issues": $FLAKE8_ERRORS,
    "black_compliant": $([ $BLACK_STATUS -eq 0 ] && echo "true" || echo "false"),
    "isort_compliant": $([ $ISORT_STATUS -eq 0 ] && echo "true" || echo "false")
  }
}
EOF

print_status "Summary saved to quality-summary.json"

echo ""
echo "=================================================="
echo "Code Quality Analysis Complete!"
echo "=================================================="
echo ""

