"""
Agent Zoo: Modular building blocks for LLM agent architectures.

A comprehensive library providing reusable components for building
sophisticated LLM-powered agents, from simple reactive patterns to
complex multi-agent systems with self-improvement capabilities.
"""

__version__ = "0.1.0"

from agent_zoo.core.base import AgentConfig, BaseAgent
from agent_zoo.interfaces.messages import Message, MessageRole
from agent_zoo.interfaces.state import AgentState, AgentStatus

__all__ = [
    "__version__",
    "BaseAgent",
    "AgentConfig",
    "Message",
    "MessageRole",
    "AgentState",
    "AgentStatus",
]
