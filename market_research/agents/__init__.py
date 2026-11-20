"""The Researcher, Analyst and Critic roles. Each takes the shared state and returns an update."""
from .analyst import analyst
from .critic import critic
from .researcher import researcher

__all__ = ["analyst", "critic", "researcher"]
