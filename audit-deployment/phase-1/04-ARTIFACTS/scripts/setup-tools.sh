#!/bin/bash
# Phase 1: Tool Installation Script
# This script installs all required tools for the foundation audit

set -e  # Exit on error

echo "=================================================="
echo "Phase 1: Foundation Audit - Tool Setup"
echo "=================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.12"

if [[ "$PYTHON_VERSION" < "$REQUIRED_VERSION" ]]; then
    print_error "Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
else
    print_status "Python version: $PYTHON_VERSION"
fi

echo ""
echo "Installing Code Quality Tools..."
echo "--------------------------------"

# Code quality tools
pip install -q mypy pylint black isort radon flake8 mccabe
print_status "mypy - Type checking"
print_status "pylint - Linting and code quality"
print_status "black - Code formatting"
print_status "isort - Import sorting"
print_status "radon - Code complexity metrics"
print_status "flake8 - Style guide enforcement"

echo ""
echo "Installing Security Tools..."
echo "----------------------------"

# Security tools
pip install -q bandit safety pip-audit
print_status "bandit - Security vulnerability scanner"
print_status "safety - Dependency vulnerability checker"
print_status "pip-audit - PyPI vulnerability scanner"

# detect-secrets
pip install -q detect-secrets
print_status "detect-secrets - Secret detection"

# semgrep (optional, requires separate installation)
if command -v semgrep &> /dev/null; then
    print_status "semgrep - SAST (already installed)"
else
    print_warning "semgrep not found. Install with: brew install semgrep (macOS)"
fi

echo ""
echo "Installing Testing Tools..."
echo "---------------------------"

# Testing tools
pip install -q pytest-cov coverage pytest-benchmark
print_status "pytest-cov - Test coverage measurement"
print_status "coverage - Coverage reporting"
print_status "pytest-benchmark - Performance benchmarking"

echo ""
echo "Installing Dependency Tools..."
echo "------------------------------"

# Dependency tools
pip install -q pipdeptree pip-licenses
print_status "pipdeptree - Dependency tree visualization"
print_status "pip-licenses - License checker"

echo ""
echo "Installing Reporting Tools..."
echo "-----------------------------"

# Reporting tools
pip install -q jinja2 markdown
print_status "jinja2 - Template engine for reports"
print_status "markdown - Markdown processing"

echo ""
echo "Creating Configuration Files..."
echo "-------------------------------"

# Create .pylintrc if it doesn't exist
if [ ! -f ".pylintrc" ]; then
    cat > .pylintrc << 'EOF'
[MASTER]
ignore=CVS,.git,__pycache__,venv,env

[MESSAGES CONTROL]
disable=C0111,  # missing-docstring
        C0103,  # invalid-name
        R0903,  # too-few-public-methods
        R0913,  # too-many-arguments
        W0212   # protected-access

[FORMAT]
max-line-length=120
indent-string='    '

[DESIGN]
max-args=7
max-locals=20
max-returns=6
max-branches=15
max-statements=60
EOF
    print_status "Created .pylintrc"
else
    print_warning ".pylintrc already exists, skipping"
fi

# Create mypy.ini if it doesn't exist
if [ ! -f "mypy.ini" ]; then
    cat > mypy.ini << 'EOF'
[mypy]
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
strict_equality = True

[mypy-tests.*]
ignore_errors = True

[mypy-setuptools.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-tqdm.*]
ignore_missing_imports = True

[mypy-joblib.*]
ignore_missing_imports = True
EOF
    print_status "Created mypy.ini"
else
    print_warning "mypy.ini already exists, skipping"
fi

# Create bandit.yaml if it doesn't exist
if [ ! -f ".bandit" ]; then
    cat > .bandit << 'EOF'
exclude_dirs:
  - /tests/
  - /venv/
  - /env/
  - /.venv/

skips:
  - B101  # assert_used
  - B601  # paramiko_calls

tests:
  - B201  # flask_debug_true
  - B301  # pickle
  - B302  # marshal
  - B303  # md5
  - B304  # ciphers
  - B305  # cipher_modes
  - B306  # mktemp_q
  - B307  # eval
  - B308  # mark_safe
  - B309  # httpsconnection
  - B310  # urllib_urlopen
  - B311  # random
  - B312  # telnetlib
  - B313  # xml_bad_cElementTree
  - B314  # xml_bad_ElementTree
  - B315  # xml_bad_expatreader
  - B316  # xml_bad_expatbuilder
  - B317  # xml_bad_sax
  - B318  # xml_bad_minidom
  - B319  # xml_bad_pulldom
  - B320  # xml_bad_etree
  - B321  # ftplib
  - B323  # unverified_context
  - B324  # hashlib_new_insecure_functions
  - B325  # tempnam
  - B401  # import_telnetlib
  - B402  # import_ftplib
  - B403  # import_pickle
  - B404  # import_subprocess
  - B405  # import_xml_etree
  - B406  # import_xml_sax
  - B407  # import_xml_expat
  - B408  # import_xml_minidom
  - B409  # import_xml_pulldom
  - B410  # import_lxml
  - B411  # import_xmlrpclib
  - B412  # import_httpoxy
  - B413  # import_pycrypto
  - B501  # request_with_no_cert_validation
  - B502  # ssl_with_bad_version
  - B503  # ssl_with_bad_defaults
  - B504  # ssl_with_no_version
  - B505  # weak_cryptographic_key
  - B506  # yaml_load
  - B507  # ssh_no_host_key_verification
  - B601  # paramiko_calls
  - B602  # subprocess_popen_with_shell_equals_true
  - B603  # subprocess_without_shell_equals_true
  - B604  # any_other_function_with_shell_equals_true
  - B605  # start_process_with_a_shell
  - B606  # start_process_with_no_shell
  - B607  # start_process_with_partial_path
  - B608  # hardcoded_sql_expressions
  - B609  # linux_commands_wildcard_injection
  - B610  # django_extra_used
  - B611  # django_rawsql_used
  - B701  # jinja2_autoescape_false
  - B702  # use_of_mako_templates
  - B703  # django_mark_safe
EOF
    print_status "Created .bandit"
else
    print_warning ".bandit already exists, skipping"
fi

# Create .flake8 if it doesn't exist
if [ ! -f ".flake8" ]; then
    cat > .flake8 << 'EOF'
[flake8]
max-line-length = 120
exclude = 
    .git,
    __pycache__,
    venv,
    env,
    .venv,
    build,
    dist,
    *.egg-info
ignore = 
    E203,  # whitespace before ':'
    E501,  # line too long (handled by black)
    W503,  # line break before binary operator
    W504   # line break after binary operator
EOF
    print_status "Created .flake8"
else
    print_warning ".flake8 already exists, skipping"
fi

echo ""
echo "Creating Output Directories..."
echo "------------------------------"

# Create output directories
mkdir -p audit-deployment/phase-1/03-FINDINGS
mkdir -p audit-deployment/phase-1/04-ARTIFACTS/coverage-report
mkdir -p audit-deployment/phase-1/04-ARTIFACTS/security-scan-results
mkdir -p audit-deployment/phase-1/04-ARTIFACTS/mypy-report
mkdir -p audit-deployment/phase-1/04-ARTIFACTS/reports

print_status "Created output directories"

echo ""
echo "=================================================="
echo "Tool Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run code quality checks: ./audit-deployment/phase-1/04-ARTIFACTS/scripts/run-quality-checks.sh"
echo "  2. Run security scans: ./audit-deployment/phase-1/04-ARTIFACTS/scripts/run-security-scans.sh"
echo "  3. Run coverage analysis: ./audit-deployment/phase-1/04-ARTIFACTS/scripts/run-coverage-analysis.sh"
echo ""

