"""Critique Service Facade

Stable entry points for post-model critique (Mixture-of-Experts).
"""
from ..critique.moe import CRITIQUE_POST_MODEL, critique_post_model  # re-export

__all__ = ["CRITIQUE_POST_MODEL", "critique_post_model"]

