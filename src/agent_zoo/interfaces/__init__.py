"""
Agent interfaces for messages, state, and configuration.
"""

from agent_zoo.interfaces.messages import (
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    Thought,
    Action,
    Observation,
    Conversation,
    MessageBatch,
    create_message,
    format_conversation,
)

from agent_zoo.interfaces.state import (
    AgentStatus,
    AgentState,
    StateType,
    StateSnapshot,
    StateManager,
)

__all__ = [
    # Messages
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "Thought",
    "Action",
    "Observation",
    "Conversation",
    "MessageBatch",
    "create_message",
    "format_conversation",
    # State
    "AgentStatus",
    "AgentState",
    "StateType",
    "StateSnapshot",
    "StateManager",
]