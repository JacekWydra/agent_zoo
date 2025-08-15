"""
Message protocols and types for agent communication.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for messages in agent communication."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """
    Base message class for agent communication.

    This provides a unified interface for all messages
    exchanged between agents, users, and tools.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole  # Who sent the message
    content: str
    name: str | None = None  # For tools: the tool name
    tool_calls: list["ToolCall"] | None = None  # For assistant messages calling tools
    tool_call_id: str | None = None  # For tool responses
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None  # For message threading
    agent_id: str | None = None  # ID of the agent that created this message

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def __str__(self) -> str:
        """String representation of the message."""
        return f"[{self.role.value}] {self.content[:100]}..."
    
    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return self.model_dump()


class ToolCall(BaseModel):
    """Represents a tool/function call."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ToolResult(BaseModel):
    """Represents the result of a tool/function call."""

    tool_call_id: str
    content: Any
    success: bool = True
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: float | None = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Thought(BaseModel):
    """Represents an agent's thought/reasoning step."""

    content: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning_type: str | None = None  # 'deductive', 'inductive', 'abductive', etc.
    supporting_evidence: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Action(BaseModel):
    """Represents an agent's action."""

    type: str  # 'tool_use', 'response', 'delegate', etc.
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Observation(BaseModel):
    """Represents an observation from the environment or tool."""

    content: Any
    source: str | None = None  # 'tool', 'environment', 'user', etc.
    relevance: float = Field(ge=0.0, le=1.0, default=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Conversation(BaseModel):
    """Represents a conversation thread."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """Get all messages with a specific role."""
        return [msg for msg in self.messages if msg.role == role]
    
    def get_messages(self) -> list[Message]:
        """Get all messages in the conversation."""
        return self.messages
    
    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        messages = [Message(**msg) for msg in data.get("messages", [])]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            messages=messages,
            metadata=data.get("metadata", {}),
        )
    
    def __len__(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages)

    def get_last_message(self) -> Message | None:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def get_context(self, max_messages: int = 10) -> list[Message]:
        """Get recent context from the conversation."""
        return self.messages[-max_messages:] if self.messages else []

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages = []
        self.updated_at = datetime.now()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MessageBatch(BaseModel):
    """Represents a batch of messages for parallel processing."""

    messages: list[Message]
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


def create_message(content: str, role: MessageRole = MessageRole.USER, **kwargs) -> Message:
    """
    Helper function to create a message.

    Args:
        content: Message content
        role: Role of message sender
        **kwargs: Additional message fields

    Returns:
        Created message
    """
    return Message(role=role, content=content, **kwargs)


def format_conversation(conversation: Conversation, include_metadata: bool = False) -> str:
    """
    Format a conversation as a readable string.

    Args:
        conversation: Conversation to format
        include_metadata: Whether to include metadata

    Returns:
        Formatted conversation string
    """
    lines = []
    for msg in conversation.messages:
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        role_str = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
        lines.append(f"[{timestamp}] {role_str}: {msg.content}")

        if include_metadata and msg.metadata:
            for key, value in msg.metadata.items():
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)
