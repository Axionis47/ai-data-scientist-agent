"""Modeling Service Facade

Stable entry point for modeling. Delegates to app.modeling.pipeline.
"""
from typing import Any, Dict

from ..modeling.pipeline import run_modeling  # re-export

__all__ = ["run_modeling"]

