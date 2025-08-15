"""
State management interfaces for agents.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Status of an agent."""
    
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class StateType(str, Enum):
    """Types of agent states."""

    MEMORY = "memory"
    CONTEXT = "context"
    GOALS = "goals"
    PLAN = "plan"
    KNOWLEDGE = "knowledge"
    SKILLS = "skills"
    HISTORY = "history"


class AgentState(BaseModel):
    """
    Comprehensive state representation for agents.

    This class maintains all stateful information needed for
    agent operation, including memory, context, goals, and history.
    """
    
    # Agent runtime status
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current agent status")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

    # Current execution state
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    current_task: str | None = None
    current_step: int = 0
    max_steps: int = 10

    # Goals and planning
    goals: list[str] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    completed_goals: list[str] = Field(default_factory=list)

    # Tools and capabilities
    available_tools: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    iteration_count: int = 0
    token_count: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Flags
    is_complete: bool = False
    has_error: bool = False
    error_message: str | None = None
    requires_user_input: bool = False

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the state."""
        self.messages.append(message)
        self.token_count += len(str(message.get("content", "")))

    def add_goal(self, goal: str) -> None:
        """Add a goal to pursue."""
        if goal not in self.goals:
            self.goals.append(goal)

    def complete_goal(self, goal: str) -> None:
        """Mark a goal as completed."""
        if goal in self.goals:
            self.goals.remove(goal)
            self.completed_goals.append(goal)

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Store a tool execution result."""
        self.tool_results[tool_name] = result
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)

    def increment_step(self) -> bool:
        """
        Increment the current step counter.

        Returns:
            True if still within max_steps, False otherwise
        """
        self.current_step += 1
        self.iteration_count += 1
        return self.current_step <= self.max_steps

    def set_error(self, error_message: str) -> None:
        """Set error state."""
        self.has_error = True
        self.error_message = error_message
        self.is_complete = True

    def reset(self) -> None:
        """Reset state to initial values."""
        self.status = AgentStatus.IDLE
        self.metrics = {}
        self.messages = []
        self.context = {}
        self.current_task = None
        self.current_step = 0
        self.goals = []
        self.completed_goals = []
        self.tools_used = []
        self.tool_results = {}
        self.iteration_count = 0
        self.token_count = 0
        self.start_time = None
        self.end_time = None
        self.is_complete = False
        self.has_error = False
        self.error_message = None
        self.requires_user_input = False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert state to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "AgentState":
        """Create state from JSON string."""
        return cls.model_validate_json(json_str)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the current state."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "current_task": self.current_task,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "goals_pending": len(self.goals),
            "goals_completed": len(self.completed_goals),
            "messages_count": len(self.messages),
            "tools_used": self.tools_used,
            "token_count": self.token_count,
            "iteration_count": self.iteration_count,
            "duration_seconds": duration,
            "is_complete": self.is_complete,
            "has_error": self.has_error,
            "error_message": self.error_message,
        }


class StateSnapshot(BaseModel):
    """
    A snapshot of agent state at a point in time.

    Used for checkpointing, rollback, and history tracking.
    """

    id: str
    timestamp: datetime
    state: AgentState
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class StateManager:
    """
    Manages agent state with history and rollback capabilities.
    """

    def __init__(self, max_history: int = 10):
        """
        Initialize state manager.

        Args:
            max_history: Maximum number of state snapshots to keep
        """
        self.current_state = AgentState()
        self.history: list[StateSnapshot] = []
        self.max_history = max_history

    def get_state(self) -> AgentState:
        """Get current state."""
        return self.current_state

    def set_state(self, state: AgentState) -> None:
        """Set current state."""
        self.current_state = state

    def save_snapshot(self, description: str | None = None) -> str:
        """
        Save current state as a snapshot.

        Args:
            description: Optional description of the snapshot

        Returns:
            Snapshot ID
        """
        snapshot_id = f"snapshot_{datetime.now().timestamp()}"
        snapshot = StateSnapshot(
            id=snapshot_id,
            timestamp=datetime.now(),
            state=self.current_state.model_copy(deep=True),
            description=description,
        )

        self.history.append(snapshot)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore state from a snapshot.

        Args:
            snapshot_id: ID of snapshot to restore

        Returns:
            True if successful, False otherwise
        """
        for snapshot in self.history:
            if snapshot.id == snapshot_id:
                self.current_state = snapshot.state.model_copy(deep=True)
                return True
        return False

    def get_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Get a specific snapshot."""
        for snapshot in self.history:
            if snapshot.id == snapshot_id:
                return snapshot
        return None

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all available snapshots."""
        return [
            {
                "id": s.id,
                "timestamp": s.timestamp.isoformat(),
                "description": s.description,
            }
            for s in self.history
        ]

    def clear_history(self) -> None:
        """Clear all snapshots."""
        self.history = []

    def export_state(self, filepath: str) -> None:
        """Export current state to file."""
        with open(filepath, "w") as f:
            json.dump(self.current_state.to_dict(), f, indent=2, default=str)

    def import_state(self, filepath: str) -> None:
        """Import state from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.current_state = AgentState.model_validate(data)
