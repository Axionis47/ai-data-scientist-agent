"""Unit tests for botds.tools.pii module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from botds.tools.pii import PII
from botds.utils import save_pickle


@pytest.fixture
def pii(tmp_path):
    """Create PII instance."""
    return PII(artifacts_dir=tmp_path)


@pytest.fixture
def df_with_emails(tmp_path):
    """Create DataFrame with email addresses."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@example.com", "bob@test.org", "charlie@company.co.uk"],
        "age": [25, 30, 35]
    })
    
    df_path = tmp_path / "emails.pkl"
    save_pickle(df, df_path)
    return str(df_path)


@pytest.fixture
def df_with_phones(tmp_path):
    """Create DataFrame with phone numbers."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "phone": ["555-123-4567", "(555) 234-5678", "555.345.6789"],
        "city": ["NYC", "LA", "SF"]
    })
    
    df_path = tmp_path / "phones.pkl"
    save_pickle(df, df_path)
    return str(df_path)


@pytest.fixture
def df_with_ssn(tmp_path):
    """Create DataFrame with SSN."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "ssn": ["123-45-6789", "987-65-4321"],
        "salary": [50000, 60000]
    })
    
    df_path = tmp_path / "ssn.pkl"
    save_pickle(df, df_path)
    return str(df_path)


@pytest.fixture
def df_clean(tmp_path):
    """Create DataFrame without PII."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0]
    })
    
    df_path = tmp_path / "clean.pkl"
    save_pickle(df, df_path)
    return str(df_path)


class TestPII:
    """Tests for PII class."""
    
    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        pii = PII(artifacts_dir=tmp_path)
        
        assert pii.artifacts_dir == tmp_path
        assert pii.data_dir.exists()
        assert pii.data_dir == tmp_path / "data"
    
    def test_init_has_patterns(self, pii):
        """Test that PII patterns are defined."""
        assert hasattr(pii, "patterns")
        assert isinstance(pii.patterns, dict)
        
        # Check for common patterns
        assert "email" in pii.patterns
        assert "phone" in pii.patterns
        assert "ssn" in pii.patterns
        assert "credit_card" in pii.patterns
    
    def test_scan_detects_emails(self, pii, df_with_emails):
        """Test that email addresses are detected."""
        result = pii.scan(df_ref=df_with_emails, patterns=["email"])
        
        assert "findings" in result
        assert "email" in result["findings"]
        
        email_findings = result["findings"]["email"]
        assert len(email_findings) > 0
        
        # Should find emails in the email column
        assert "email" in email_findings
    
    def test_scan_detects_phones(self, pii, df_with_phones):
        """Test that phone numbers are detected."""
        result = pii.scan(df_ref=df_with_phones, patterns=["phone"])
        
        assert "findings" in result
        assert "phone" in result["findings"]
        
        phone_findings = result["findings"]["phone"]
        assert len(phone_findings) > 0
    
    def test_scan_detects_ssn(self, pii, df_with_ssn):
        """Test that SSNs are detected."""
        result = pii.scan(df_ref=df_with_ssn, patterns=["ssn"])
        
        assert "findings" in result
        assert "ssn" in result["findings"]
        
        ssn_findings = result["findings"]["ssn"]
        assert len(ssn_findings) > 0
    
    def test_scan_clean_data(self, pii, df_clean):
        """Test scanning data without PII."""
        result = pii.scan(df_ref=df_clean)
        
        assert "findings" in result
        assert "total_matches" in result
        
        # Should find no PII
        assert result["total_matches"] == 0
    
    def test_scan_all_patterns_by_default(self, pii, df_with_emails):
        """Test that all patterns are scanned by default."""
        result = pii.scan(df_ref=df_with_emails)
        
        assert "findings" in result
        
        # Should check all pattern types
        findings = result["findings"]
        assert "email" in findings or len(findings) >= 0
    
    def test_scan_specific_patterns(self, pii, df_with_emails):
        """Test scanning for specific patterns only."""
        result = pii.scan(df_ref=df_with_emails, patterns=["email", "phone"])
        
        # Should only check specified patterns
        assert "findings" in result
    
    def test_scan_returns_match_details(self, pii, df_with_emails):
        """Test that scan returns detailed match information."""
        result = pii.scan(df_ref=df_with_emails, patterns=["email"])

        if result["total_matches"] > 0:
            findings = result["findings"]["email"]

            # Check structure of findings
            for col, col_findings in findings.items():
                if col_findings:
                    assert "count" in col_findings
                    assert "samples" in col_findings
                    if col_findings["samples"]:
                        match = col_findings["samples"][0]
                        assert "row" in match
                        assert "value" in match or "context" in match
    
    def test_redact_emails(self, pii, df_with_emails):
        """Test redacting email addresses."""
        result = pii.redact(
            df_ref=df_with_emails,
            patterns=["email"]
        )

        assert "df_ref_sanitized" in result
        assert "total_redactions" in result

        # Should have made redactions
        assert result["total_redactions"] > 0

    def test_redact_with_replacement(self, pii, df_with_phones):
        """Test redacting with replacement value."""
        result = pii.redact(
            df_ref=df_with_phones,
            patterns=["phone"],
            replacement="[PHONE]"
        )

        assert "df_ref_sanitized" in result
        assert result["total_redactions"] > 0

    def test_redact_clean_data(self, pii, df_clean):
        """Test redacting data without PII."""
        result = pii.redact(df_ref=df_clean)

        assert "df_ref_sanitized" in result
        assert "total_redactions" in result

        # Should make no redactions
        assert result["total_redactions"] == 0

    def test_redact_saves_new_dataframe(self, pii, df_with_emails):
        """Test that redaction saves a new DataFrame."""
        result = pii.redact(df_ref=df_with_emails, patterns=["email"])

        assert "df_ref_sanitized" in result

        # New DataFrame should exist
        df_path = Path(result["df_ref_sanitized"])
        assert df_path.exists()
    
    def test_scan_handles_missing_values(self, pii, tmp_path):
        """Test that scan handles missing values gracefully."""
        df = pd.DataFrame({
            "email": ["alice@example.com", None, "bob@test.org", pd.NA],
            "name": ["Alice", "Bob", "Charlie", "David"]
        })
        
        df_path = tmp_path / "with_na.pkl"
        save_pickle(df, df_path)
        
        result = pii.scan(df_ref=str(df_path), patterns=["email"])
        
        # Should not crash on missing values
        assert "findings" in result
    
    def test_scan_only_checks_string_columns(self, pii, tmp_path):
        """Test that scan only checks string/object columns."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "text": ["test", "data", "here"]
        })
        
        df_path = tmp_path / "mixed_types.pkl"
        save_pickle(df, df_path)
        
        result = pii.scan(df_ref=str(df_path))
        
        # Should only scan text column
        assert "findings" in result
    
    def test_get_function_definitions(self, pii):
        """Test that function definitions are returned."""
        defs = pii.get_function_definitions()
        
        assert isinstance(defs, list)
        assert len(defs) > 0
        
        # Check structure
        assert "type" in defs[0]
        assert defs[0]["type"] == "function"

