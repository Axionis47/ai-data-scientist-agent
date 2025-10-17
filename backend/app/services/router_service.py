"""Router Service Facade

Stable entry points for planning and decision routing.
"""

from ..agent.router import build_context_pack, plan_with_router  # re-export

__all__ = ["build_context_pack", "plan_with_router"]
