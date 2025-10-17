"""Reporting Service Facade

Stable entry point for report generation.
"""

from ..reporting.report import reporting_expert  # re-export

__all__ = ["reporting_expert"]
