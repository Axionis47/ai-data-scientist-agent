#!/bin/bash
# Phase 1: Test Coverage Analysis Script
# This script runs test coverage analysis and generates reports

set -e  # Exit on error

echo "=================================================="
echo "Phase 1: Test Coverage Analysis"
echo "=================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_section() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

# Output directory
OUTPUT_DIR="audit-deployment/phase-1/04-ARTIFACTS/coverage-report"
mkdir -p "$OUTPUT_DIR"

# Start timestamp
START_TIME=$(date +%s)

# 1. Run tests with coverage
print_section "1. Running Tests with Coverage"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    print_warning "OPENAI_API_KEY not set - some tests may be skipped"
    echo "Set with: export OPENAI_API_KEY=sk-your-key-here"
    echo ""
fi

# Run pytest with coverage
pytest tests/ \
    --cov=botds \
    --cov=cli \
    --cov-report=html:"$OUTPUT_DIR/html" \
    --cov-report=json:"$OUTPUT_DIR/coverage.json" \
    --cov-report=term \
    --cov-report=term-missing \
    -v \
    2>&1 | tee "$OUTPUT_DIR/pytest-output.txt" || true

print_status "Test execution complete"

# 2. Extract coverage metrics
print_section "2. Extracting Coverage Metrics"

# Parse coverage from JSON
if [ -f "$OUTPUT_DIR/coverage.json" ]; then
    TOTAL_COVERAGE=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/coverage.json')); print(f\"{data['totals']['percent_covered']:.2f}\")" 2>/dev/null || echo "N/A")
    TOTAL_STATEMENTS=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/coverage.json')); print(data['totals']['num_statements'])" 2>/dev/null || echo "N/A")
    COVERED_STATEMENTS=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/coverage.json')); print(data['totals']['covered_lines'])" 2>/dev/null || echo "N/A")
    MISSING_STATEMENTS=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/coverage.json')); print(data['totals']['missing_lines'])" 2>/dev/null || echo "N/A")
    
    echo "Overall Coverage: $TOTAL_COVERAGE%"
    echo "Total Statements: $TOTAL_STATEMENTS"
    echo "Covered: $COVERED_STATEMENTS"
    echo "Missing: $MISSING_STATEMENTS"
    
    if (( $(echo "$TOTAL_COVERAGE >= 80" | bc -l 2>/dev/null || echo "0") )); then
        print_status "Coverage is above 80% threshold"
    elif (( $(echo "$TOTAL_COVERAGE >= 60" | bc -l 2>/dev/null || echo "0") )); then
        print_warning "Coverage is between 60-80% - improvement recommended"
    else
        print_error "Coverage is below 60% - significant improvement needed"
    fi
else
    print_error "Coverage JSON not found - test execution may have failed"
    TOTAL_COVERAGE="N/A"
fi

# 3. Analyze test results
print_section "3. Analyzing Test Results"

# Count tests
TOTAL_TESTS=$(grep -c "PASSED\|FAILED\|SKIPPED" "$OUTPUT_DIR/pytest-output.txt" 2>/dev/null || echo "0")
PASSED_TESTS=$(grep -c "PASSED" "$OUTPUT_DIR/pytest-output.txt" 2>/dev/null || echo "0")
FAILED_TESTS=$(grep -c "FAILED" "$OUTPUT_DIR/pytest-output.txt" 2>/dev/null || echo "0")
SKIPPED_TESTS=$(grep -c "SKIPPED" "$OUTPUT_DIR/pytest-output.txt" 2>/dev/null || echo "0")

echo "Test Results:"
echo "  Total:   $TOTAL_TESTS"
echo "  Passed:  $PASSED_TESTS"
echo "  Failed:  $FAILED_TESTS"
echo "  Skipped: $SKIPPED_TESTS"

if [ "$FAILED_TESTS" -gt 0 ]; then
    print_error "$FAILED_TESTS tests failed"
else
    print_status "All tests passed"
fi

# 4. Identify uncovered files
print_section "4. Identifying Uncovered Code"

# Create uncovered files report
echo "Files with Low Coverage (<80%)" > "$OUTPUT_DIR/low-coverage-files.txt"
echo "================================" >> "$OUTPUT_DIR/low-coverage-files.txt"
echo "" >> "$OUTPUT_DIR/low-coverage-files.txt"

if [ -f "$OUTPUT_DIR/coverage.json" ]; then
    python3 << 'PYTHON_SCRIPT' > "$OUTPUT_DIR/low-coverage-files.txt"
import json
import sys

try:
    with open('audit-deployment/phase-1/04-ARTIFACTS/coverage-report/coverage.json') as f:
        data = json.load(f)
    
    print("Files with Coverage < 80%")
    print("=" * 60)
    print(f"{'File':<50} {'Coverage':>8}")
    print("-" * 60)
    
    low_coverage_files = []
    for file_path, file_data in data['files'].items():
        coverage = file_data['summary']['percent_covered']
        if coverage < 80:
            low_coverage_files.append((file_path, coverage))
    
    # Sort by coverage (lowest first)
    low_coverage_files.sort(key=lambda x: x[1])
    
    for file_path, coverage in low_coverage_files:
        # Shorten path for display
        display_path = file_path.replace('audit-deployment/phase-1/04-ARTIFACTS/coverage-report/', '')
        if len(display_path) > 48:
            display_path = '...' + display_path[-45:]
        print(f"{display_path:<50} {coverage:>7.2f}%")
    
    if not low_coverage_files:
        print("No files with coverage < 80%")
    
    print()
    print(f"Total files with low coverage: {len(low_coverage_files)}")
    
except Exception as e:
    print(f"Error analyzing coverage: {e}", file=sys.stderr)
PYTHON_SCRIPT

    print_status "Low coverage files identified"
else
    echo "Coverage data not available" >> "$OUTPUT_DIR/low-coverage-files.txt"
fi

# 5. Generate coverage badge
print_section "5. Generating Coverage Badge"

# Create a simple coverage badge SVG
if [ "$TOTAL_COVERAGE" != "N/A" ]; then
    # Determine color based on coverage
    if (( $(echo "$TOTAL_COVERAGE >= 80" | bc -l 2>/dev/null || echo "0") )); then
        COLOR="brightgreen"
    elif (( $(echo "$TOTAL_COVERAGE >= 60" | bc -l 2>/dev/null || echo "0") )); then
        COLOR="yellow"
    else
        COLOR="red"
    fi
    
    cat > "$OUTPUT_DIR/coverage-badge.svg" << EOF
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="120" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h63v20H0z"/>
    <path fill="$COLOR" d="M63 0h57v20H63z"/>
    <path fill="url(#b)" d="M0 0h120v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="31.5" y="14">coverage</text>
    <text x="90.5" y="15" fill="#010101" fill-opacity=".3">$TOTAL_COVERAGE%</text>
    <text x="90.5" y="14">$TOTAL_COVERAGE%</text>
  </g>
</svg>
EOF
    print_status "Coverage badge generated"
fi

# 6. Test execution time analysis
print_section "6. Analyzing Test Execution Time"

# Extract test durations
grep "passed in\|failed in" "$OUTPUT_DIR/pytest-output.txt" | tail -1 > "$OUTPUT_DIR/test-duration.txt" || echo "N/A" > "$OUTPUT_DIR/test-duration.txt"
TEST_DURATION=$(cat "$OUTPUT_DIR/test-duration.txt")
echo "Test execution time: $TEST_DURATION"

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Generate summary
print_section "Coverage Analysis Summary"
echo "Duration: ${DURATION}s"
echo ""
echo "Coverage Metrics:"
echo "  Overall Coverage:    $TOTAL_COVERAGE%"
echo "  Total Statements:    $TOTAL_STATEMENTS"
echo "  Covered Statements:  $COVERED_STATEMENTS"
echo "  Missing Statements:  $MISSING_STATEMENTS"
echo ""
echo "Test Results:"
echo "  Total Tests:   $TOTAL_TESTS"
echo "  Passed:        $PASSED_TESTS"
echo "  Failed:        $FAILED_TESTS"
echo "  Skipped:       $SKIPPED_TESTS"
echo ""
echo "Reports generated in: $OUTPUT_DIR"
echo "  - html/index.html (interactive coverage report)"
echo "  - coverage.json (machine-readable coverage data)"
echo "  - low-coverage-files.txt (files needing improvement)"
echo "  - coverage-badge.svg (coverage badge)"
echo ""

# Generate summary JSON
cat > "$OUTPUT_DIR/coverage-summary.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $DURATION,
  "coverage": {
    "percent": "$TOTAL_COVERAGE",
    "total_statements": "$TOTAL_STATEMENTS",
    "covered_statements": "$COVERED_STATEMENTS",
    "missing_statements": "$MISSING_STATEMENTS"
  },
  "tests": {
    "total": $TOTAL_TESTS,
    "passed": $PASSED_TESTS,
    "failed": $FAILED_TESTS,
    "skipped": $SKIPPED_TESTS
  },
  "test_duration": "$TEST_DURATION"
}
EOF

print_status "Summary saved to coverage-summary.json"

# Recommendations
echo ""
print_section "Recommendations"

if [ "$TOTAL_COVERAGE" != "N/A" ]; then
    if (( $(echo "$TOTAL_COVERAGE < 80" | bc -l 2>/dev/null || echo "1") )); then
        echo "1. Increase test coverage to at least 80%"
        echo "2. Focus on files listed in low-coverage-files.txt"
        echo "3. Add integration tests for critical paths"
        echo "4. Add edge case tests"
    else
        echo "✓ Coverage is good! Maintain current level."
    fi
fi

if [ "$FAILED_TESTS" -gt 0 ]; then
    echo "⚠ Fix failing tests before proceeding to Phase 2"
fi

if [ "$SKIPPED_TESTS" -gt 0 ]; then
    echo "⚠ Review skipped tests - may indicate missing dependencies or configuration"
fi

echo ""
echo "=================================================="
echo "Coverage Analysis Complete!"
echo "=================================================="
echo ""
echo "View detailed report: open $OUTPUT_DIR/html/index.html"
echo ""

