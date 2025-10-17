#!/bin/bash
# Phase 1: Security Vulnerability Scanning Script
# This script runs all security scans and generates reports

set -e  # Exit on error

echo "=================================================="
echo "Phase 1: Security Vulnerability Scanning"
echo "=================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
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
OUTPUT_DIR="audit-deployment/phase-1/04-ARTIFACTS/security-scan-results"
mkdir -p "$OUTPUT_DIR"

# Start timestamp
START_TIME=$(date +%s)

# 1. Bandit Security Scan
print_section "1. Running Bandit Security Scan"
bandit -r botds/ cli/ -f json -o "$OUTPUT_DIR/bandit.json" 2>&1 || true
bandit -r botds/ cli/ -f txt -o "$OUTPUT_DIR/bandit.txt" 2>&1 || true

# Count issues by severity
BANDIT_HIGH=$(grep -c '"issue_severity": "HIGH"' "$OUTPUT_DIR/bandit.json" 2>/dev/null || echo "0")
BANDIT_MEDIUM=$(grep -c '"issue_severity": "MEDIUM"' "$OUTPUT_DIR/bandit.json" 2>/dev/null || echo "0")
BANDIT_LOW=$(grep -c '"issue_severity": "LOW"' "$OUTPUT_DIR/bandit.json" 2>/dev/null || echo "0")

if [ "$BANDIT_HIGH" -gt 0 ]; then
    print_error "Bandit: $BANDIT_HIGH HIGH, $BANDIT_MEDIUM MEDIUM, $BANDIT_LOW LOW severity issues"
else
    print_status "Bandit: $BANDIT_HIGH HIGH, $BANDIT_MEDIUM MEDIUM, $BANDIT_LOW LOW severity issues"
fi

# 2. Safety Dependency Check
print_section "2. Running Safety Dependency Check"
safety check --json > "$OUTPUT_DIR/safety.json" 2>&1 || true
safety check --full-report > "$OUTPUT_DIR/safety.txt" 2>&1 || true

# Count vulnerabilities
SAFETY_VULNS=$(grep -c '"vulnerability"' "$OUTPUT_DIR/safety.json" 2>/dev/null || echo "0")
if [ "$SAFETY_VULNS" -gt 0 ]; then
    print_warning "Safety: $SAFETY_VULNS vulnerabilities found"
else
    print_status "Safety: No known vulnerabilities"
fi

# 3. pip-audit Vulnerability Scan
print_section "3. Running pip-audit Vulnerability Scan"
pip-audit --format json > "$OUTPUT_DIR/pip-audit.json" 2>&1 || true
pip-audit --format markdown > "$OUTPUT_DIR/pip-audit.md" 2>&1 || true

# Count vulnerabilities
PIP_AUDIT_VULNS=$(grep -c '"vulnerabilities"' "$OUTPUT_DIR/pip-audit.json" 2>/dev/null || echo "0")
if [ "$PIP_AUDIT_VULNS" -gt 0 ]; then
    print_warning "pip-audit: $PIP_AUDIT_VULNS packages with vulnerabilities"
else
    print_status "pip-audit: No vulnerabilities found"
fi

# 4. Secrets Detection
print_section "4. Running Secrets Detection"
detect-secrets scan --all-files --exclude-files '\.git/.*|\.venv/.*|venv/.*|env/.*|cache/.*|artifacts/.*' > "$OUTPUT_DIR/secrets-baseline.json" 2>&1 || true

# Count potential secrets
SECRETS_COUNT=$(grep -c '"type"' "$OUTPUT_DIR/secrets-baseline.json" 2>/dev/null || echo "0")
if [ "$SECRETS_COUNT" -gt 0 ]; then
    print_warning "detect-secrets: $SECRETS_COUNT potential secrets detected (review for false positives)"
else
    print_status "detect-secrets: No secrets detected"
fi

# 5. Check for .env file exposure
print_section "5. Checking Environment File Security"
if [ -f ".env" ]; then
    print_warning ".env file exists - ensure it's in .gitignore"
    if grep -q "^\.env$" .gitignore 2>/dev/null; then
        print_status ".env is in .gitignore"
    else
        print_error ".env is NOT in .gitignore - SECURITY RISK!"
    fi
else
    print_status "No .env file in repository root"
fi

# Check .env.example
if [ -f ".env.example" ]; then
    if grep -q "sk-" .env.example 2>/dev/null; then
        print_error ".env.example contains actual API keys - SECURITY RISK!"
    else
        print_status ".env.example is safe (no actual keys)"
    fi
fi

# 6. Check for hardcoded secrets in code
print_section "6. Checking for Hardcoded Secrets"
echo "Searching for potential hardcoded secrets..." > "$OUTPUT_DIR/hardcoded-secrets.txt"

# Search for common patterns
grep -r -n -i "api[_-]key\s*=\s*['\"]sk-" botds/ cli/ >> "$OUTPUT_DIR/hardcoded-secrets.txt" 2>&1 || true
grep -r -n -i "password\s*=\s*['\"][^'\"]\+" botds/ cli/ >> "$OUTPUT_DIR/hardcoded-secrets.txt" 2>&1 || true
grep -r -n -i "secret\s*=\s*['\"][^'\"]\+" botds/ cli/ >> "$OUTPUT_DIR/hardcoded-secrets.txt" 2>&1 || true
grep -r -n -i "token\s*=\s*['\"][^'\"]\+" botds/ cli/ >> "$OUTPUT_DIR/hardcoded-secrets.txt" 2>&1 || true

HARDCODED_COUNT=$(grep -c ":" "$OUTPUT_DIR/hardcoded-secrets.txt" 2>/dev/null || echo "0")
if [ "$HARDCODED_COUNT" -gt 1 ]; then  # >1 because of header line
    print_warning "Found $((HARDCODED_COUNT-1)) potential hardcoded secrets (review manually)"
else
    print_status "No obvious hardcoded secrets found"
fi

# 7. Check dependency licenses
print_section "7. Checking Dependency Licenses"
pip-licenses --format=json --with-urls > "$OUTPUT_DIR/licenses.json" 2>&1 || true
pip-licenses --format=markdown --with-urls > "$OUTPUT_DIR/licenses.md" 2>&1 || true

# Check for problematic licenses
PROBLEMATIC_LICENSES=$(grep -i "GPL\|AGPL\|LGPL" "$OUTPUT_DIR/licenses.md" 2>/dev/null || echo "")
if [ -n "$PROBLEMATIC_LICENSES" ]; then
    print_warning "Found potentially problematic licenses (GPL family) - review required"
else
    print_status "No problematic licenses detected"
fi

# 8. Check for outdated packages with known vulnerabilities
print_section "8. Checking for Outdated Packages"
pip list --outdated --format=json > "$OUTPUT_DIR/outdated-packages.json" 2>&1 || true

OUTDATED_COUNT=$(grep -c '"name"' "$OUTPUT_DIR/outdated-packages.json" 2>/dev/null || echo "0")
if [ "$OUTDATED_COUNT" -gt 0 ]; then
    print_warning "$OUTDATED_COUNT packages are outdated"
else
    print_status "All packages are up to date"
fi

# 9. Semgrep SAST (if available)
print_section "9. Running Semgrep SAST (if available)"
if command -v semgrep &> /dev/null; then
    semgrep --config=auto --json --output="$OUTPUT_DIR/semgrep.json" botds/ cli/ 2>&1 || true
    semgrep --config=auto --output="$OUTPUT_DIR/semgrep.txt" botds/ cli/ 2>&1 || true
    
    SEMGREP_FINDINGS=$(grep -c '"check_id"' "$OUTPUT_DIR/semgrep.json" 2>/dev/null || echo "0")
    if [ "$SEMGREP_FINDINGS" -gt 0 ]; then
        print_warning "Semgrep: $SEMGREP_FINDINGS findings"
    else
        print_status "Semgrep: No issues found"
    fi
else
    print_warning "Semgrep not installed - skipping SAST scan"
    echo "Install with: brew install semgrep (macOS) or pip install semgrep"
fi

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Generate security summary
print_section "Security Scan Summary"
echo "Duration: ${DURATION}s"
echo ""
echo "Vulnerability Summary:"
echo "  Bandit:        $BANDIT_HIGH HIGH, $BANDIT_MEDIUM MEDIUM, $BANDIT_LOW LOW"
echo "  Safety:        $SAFETY_VULNS known vulnerabilities"
echo "  pip-audit:     $PIP_AUDIT_VULNS vulnerable packages"
echo "  Secrets:       $SECRETS_COUNT potential secrets"
echo "  Outdated:      $OUTDATED_COUNT outdated packages"
echo ""
echo "Reports generated in: $OUTPUT_DIR"
echo "  - bandit.txt (security issues)"
echo "  - safety.txt (dependency vulnerabilities)"
echo "  - pip-audit.md (PyPI vulnerabilities)"
echo "  - secrets-baseline.json (secret detection)"
echo "  - licenses.md (license compliance)"
echo "  - outdated-packages.json (update recommendations)"
echo ""

# Generate summary JSON
cat > "$OUTPUT_DIR/security-summary.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $DURATION,
  "findings": {
    "bandit": {
      "high": $BANDIT_HIGH,
      "medium": $BANDIT_MEDIUM,
      "low": $BANDIT_LOW
    },
    "safety_vulnerabilities": $SAFETY_VULNS,
    "pip_audit_vulnerabilities": $PIP_AUDIT_VULNS,
    "potential_secrets": $SECRETS_COUNT,
    "outdated_packages": $OUTDATED_COUNT
  },
  "risk_level": "$([ $BANDIT_HIGH -gt 0 ] && echo "HIGH" || [ $SAFETY_VULNS -gt 0 ] && echo "MEDIUM" || echo "LOW")"
}
EOF

print_status "Summary saved to security-summary.json"

# Overall risk assessment
echo ""
if [ "$BANDIT_HIGH" -gt 0 ] || [ "$SAFETY_VULNS" -gt 5 ]; then
    print_error "OVERALL RISK: HIGH - Immediate action required"
elif [ "$BANDIT_MEDIUM" -gt 5 ] || [ "$SAFETY_VULNS" -gt 0 ]; then
    print_warning "OVERALL RISK: MEDIUM - Review and remediate soon"
else
    print_status "OVERALL RISK: LOW - Continue monitoring"
fi

echo ""
echo "=================================================="
echo "Security Scanning Complete!"
echo "=================================================="
echo ""

