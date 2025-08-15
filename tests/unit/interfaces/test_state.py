"""
Unit tests for state management interfaces.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from agent_zoo.interfaces.state import (
    AgentState,
    AgentStatus,
    StateManager,
    StateSnapshot,
    StateType,
)


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        """Test all status values are accessible."""
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.THINKING == "thinking"
        assert AgentStatus.ACTING == "acting"
        assert AgentStatus.WAITING == "waiting"
        assert AgentStatus.ERROR == "error"
        assert AgentStatus.COMPLETED == "completed"

    def test_status_string_conversion(self):
        """Test status string conversion."""
        status = AgentStatus.IDLE
        assert status.value == "idle"


class TestStateType:
    """Tests for StateType enum."""

    def test_state_type_values(self):
        """Test all state type values."""
        assert StateType.MEMORY == "memory"
        assert StateType.CONTEXT == "context"
        assert StateType.GOALS == "goals"
        assert StateType.PLAN == "plan"
        assert StateType.KNOWLEDGE == "knowledge"
        assert StateType.SKILLS == "skills"
        assert StateType.HISTORY == "history"


class TestAgentState:
    """Tests for AgentState."""

    def test_default_state(self):
        """Test default state initialization."""
        state = AgentState()

        assert state.status == AgentStatus.IDLE
        assert state.metrics == {}
        assert state.messages == []
        assert state.context == {}
        assert state.current_task is None
        assert state.plan == []
        assert state.goals == []
        assert state.completed_goals == []
        assert state.available_tools == []
        assert state.tools_used == []
        assert state.iteration_count == 0
        assert state.token_count == 0
        assert state.has_error is False
        assert state.error_message is None
        assert state.start_time is None
        assert state.end_time is None

    def test_custom_state(self):
        """Test custom state initialization."""
        now = datetime.now()
        state = AgentState(
            status=AgentStatus.THINKING,
            metrics={"accuracy": 0.95},
            messages=[{"role": "user", "content": "Hello"}],
            context={"session": "test123"},
            current_task="Solve problem",
            goals=["Goal 1", "Goal 2"],
            plan=[{"step": 1, "action": "Step 1"}, {"step": 2, "action": "Step 2"}],
            completed_goals=["Goal 0"],
            available_tools=["tool1", "tool2"],
            tools_used=["tool1"],
            tool_results={"tool1": "result"},
            iteration_count=5,
            token_count=100,
            has_error=True,
            error_message="Test error",
            start_time=now,
            end_time=now,
        )

        assert state.status == AgentStatus.THINKING
        assert state.metrics == {"accuracy": 0.95}
        assert state.messages == [{"role": "user", "content": "Hello"}]
        assert state.context == {"session": "test123"}
        assert state.current_task == "Solve problem"
        assert state.goals == ["Goal 1", "Goal 2"]
        assert state.plan == [{"step": 1, "action": "Step 1"}, {"step": 2, "action": "Step 2"}]
        assert state.completed_goals == ["Goal 0"]
        assert state.available_tools == ["tool1", "tool2"]
        assert state.tools_used == ["tool1"]
        assert state.tool_results == {"tool1": "result"}
        assert state.iteration_count == 5
        assert state.token_count == 100
        assert state.has_error is True
        assert state.error_message == "Test error"
        assert state.start_time == now
        assert state.end_time == now

    def test_state_serialization(self):
        """Test state serialization to dict."""
        state = AgentState(
            status=AgentStatus.ACTING,
            current_task="Test task",
            iteration_count=3,
        )

        state_dict = state.model_dump()

        assert state_dict["status"] == "acting"
        assert state_dict["current_task"] == "Test task"
        assert state_dict["iteration_count"] == 3
        assert "start_time" in state_dict
        assert "end_time" in state_dict

    def test_state_deserialization(self):
        """Test state deserialization from dict."""
        state_dict = {
            "status": "thinking",
            "current_task": "Solve problem",
            "iteration_count": 2,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        state = AgentState(**state_dict)

        assert state.status == AgentStatus.THINKING
        assert state.current_task == "Solve problem"
        assert state.iteration_count == 2
        assert state.messages == [{"role": "user", "content": "Hi"}]

    def test_state_json_serialization(self):
        """Test JSON serialization."""
        state = AgentState(
            status=AgentStatus.WAITING,
            metrics={"score": 0.8},
        )

        json_str = state.model_dump_json()
        loaded_dict = json.loads(json_str)

        assert loaded_dict["status"] == "waiting"
        assert loaded_dict["metrics"]["score"] == 0.8

        # Deserialize back
        state2 = AgentState.model_validate_json(json_str)
        assert state2.status == AgentStatus.WAITING
        assert state2.metrics == {"score": 0.8}

    def test_state_update(self):
        """Test updating state fields."""
        state = AgentState()

        # Update fields
        state.status = AgentStatus.COMPLETED
        state.iteration_count += 1
        state.messages.append({"role": "assistant", "content": "Done"})

        assert state.status == AgentStatus.COMPLETED
        assert state.iteration_count == 1
        assert len(state.messages) == 1

    def test_state_copy(self):
        """Test creating a copy of state."""
        state1 = AgentState(
            status=AgentStatus.THINKING,
            current_task="Original task",
        )

        state2 = state1.model_copy()
        state2.current_task = "New task"

        assert state1.current_task == "Original task"
        assert state2.current_task == "New task"

    def test_state_validation(self):
        """Test state field validation."""
        # Invalid status should raise error
        with pytest.raises(ValidationError):
            AgentState(status="invalid_status")

        # Negative iteration count should be allowed since there's no validation
        state = AgentState(iteration_count=-1)
        assert state.iteration_count == -1

        # Negative token count should be allowed since there's no validation
        state = AgentState(token_count=-10)
        assert state.token_count == -10


class TestStateSnapshot:
    """Tests for StateSnapshot."""

    def test_snapshot_creation(self, agent_state):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            id="snapshot-1",
            state=agent_state,
            timestamp=datetime.now(),
            description="Test snapshot",
        )

        assert snapshot.state == agent_state
        assert isinstance(snapshot.timestamp, datetime)
        assert snapshot.description == "Test snapshot"

    def test_snapshot_without_description(self, agent_state):
        """Test snapshot without description."""
        snapshot = StateSnapshot(
            id="snapshot-2",
            state=agent_state,
            timestamp=datetime.now(),
        )

        assert snapshot.state == agent_state
        assert snapshot.description is None


class TestStateManager:
    """Tests for StateManager."""

    @pytest.fixture
    def state_manager(self):
        """Create a state manager instance."""
        return StateManager(max_history=5)

    def test_state_manager_initialization(self, state_manager):
        """Test state manager initialization."""
        assert state_manager.max_history == 5
        assert isinstance(state_manager.current_state, AgentState)
        assert len(state_manager.history) == 0

    def test_update_state(self, state_manager):
        """Test updating state."""
        new_state = AgentState(
            status=AgentStatus.THINKING,
            current_task="New task",
        )

        state_manager.set_state(new_state)

        assert state_manager.current_state == new_state
        assert state_manager.current_state.status == AgentStatus.THINKING

    def test_save_snapshot(self, state_manager):
        """Test saving state snapshots."""
        # Save initial snapshot
        state_manager.save_snapshot("Initial state")
        assert len(state_manager.history) == 1

        # Update and save another
        state_manager.current_state.status = AgentStatus.ACTING
        state_manager.save_snapshot("Acting state")
        assert len(state_manager.history) == 2

        # Check snapshots
        snapshots = state_manager.history
        assert len(snapshots) == 2
        assert snapshots[0].description == "Initial state"
        assert snapshots[1].description == "Acting state"

    def test_max_history_limit(self, state_manager):
        """Test that history respects max_history limit."""
        # Save more snapshots than max_history
        for i in range(7):
            state_manager.current_state.iteration_count = i
            state_manager.save_snapshot(f"Snapshot {i}")

        # Should only keep last 5
        assert len(state_manager.history) == 5
        snapshots = state_manager.history
        assert snapshots[0].description == "Snapshot 2"
        assert snapshots[-1].description == "Snapshot 6"

    def test_restore_snapshot(self, state_manager):
        """Test restoring from snapshot."""
        # Save snapshots with different states
        snapshot_id1 = state_manager.save_snapshot("State 1")

        state_manager.current_state.status = AgentStatus.THINKING
        state_manager.current_state.current_task = "Task 2"
        snapshot_id2 = state_manager.save_snapshot("State 2")

        state_manager.current_state.status = AgentStatus.ACTING
        state_manager.current_state.current_task = "Task 3"
        snapshot_id3 = state_manager.save_snapshot("State 3")

        # Restore to first snapshot
        restored = state_manager.restore_snapshot(snapshot_id1)
        assert restored is True
        assert state_manager.current_state.status == AgentStatus.IDLE
        assert state_manager.current_state.current_task is None

        # Restore to second snapshot
        restored = state_manager.restore_snapshot(snapshot_id2)
        assert restored is True
        assert state_manager.current_state.status == AgentStatus.THINKING
        assert state_manager.current_state.current_task == "Task 2"

    def test_restore_invalid_index(self, state_manager):
        """Test restoring with invalid snapshot id."""
        state_manager.save_snapshot("Test")

        # Invalid snapshot id
        restored = state_manager.restore_snapshot("invalid_snapshot_id")
        assert restored is False

        # Empty string
        restored = state_manager.restore_snapshot("")
        assert restored is False

    def test_clear_history(self, state_manager):
        """Test clearing history."""
        # Add some snapshots
        for i in range(3):
            state_manager.save_snapshot(f"Snapshot {i}")

        assert len(state_manager.history) == 3

        # Clear history
        state_manager.clear_history()
        assert len(state_manager.history) == 0
        assert state_manager.history == []

    def test_get_state_by_type(self, state_manager):
        """Test getting state data."""
        # Set up state with different types of data
        state_manager.current_state.context = {"session": "123"}
        state_manager.current_state.plan = [
            {"step": 1, "action": "step1"},
            {"step": 2, "action": "step2"},
        ]
        state_manager.current_state.metadata = {"version": "1.0"}

        # Access state data directly
        assert state_manager.current_state.context == {"session": "123"}
        assert state_manager.current_state.plan == [
            {"step": 1, "action": "step1"},
            {"step": 2, "action": "step2"},
        ]
        assert state_manager.current_state.metadata == {"version": "1.0"}

    def test_update_state_by_type(self, state_manager):
        """Test updating state data."""
        # Update context directly
        state_manager.current_state.context = {"updated": "context"}
        assert state_manager.current_state.context == {"updated": "context"}

        # Update goals directly
        state_manager.current_state.goals = ["New goal"]
        assert state_manager.current_state.goals == ["New goal"]

        # Update metadata directly
        state_manager.current_state.metadata = {"new": "metadata"}
        assert state_manager.current_state.metadata == {"new": "metadata"}

    def test_state_manager_copy(self, state_manager):
        """Test that state manager creates copies of states."""
        state_manager.save_snapshot("Before change")

        # Modify current state
        state_manager.current_state.status = AgentStatus.COMPLETED

        # Check that snapshot has independent copy
        snapshot = state_manager.history[0]
        assert snapshot.state.status == AgentStatus.IDLE
        assert state_manager.current_state.status == AgentStatus.COMPLETED
