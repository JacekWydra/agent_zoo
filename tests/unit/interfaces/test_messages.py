"""
Unit tests for message interfaces.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from agent_zoo.interfaces.messages import (
    Action,
    Conversation,
    Message,
    MessageRole,
    Observation,
    Thought,
    ToolCall,
    ToolResult,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_role_values(self):
        """Test all role values."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL == "tool"


class TestMessage:
    """Tests for Message class."""

    def test_basic_message(self):
        """Test basic message creation."""
        msg = Message(role=MessageRole.USER, content="Hello, world!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_assistant_message_with_tool_calls(self):
        """Test assistant message with tool calls."""
        tool_calls = [
            ToolCall(
                id="call_123",
                name="calculator",
                arguments={"operation": "add", "a": 2, "b": 3},
            )
        ]

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="I'll calculate that for you.",
            tool_calls=tool_calls,
        )

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "I'll calculate that for you."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_123"
        assert msg.tool_calls[0].name == "calculator"

    def test_tool_message(self):
        """Test tool result message."""
        msg = Message(
            role=MessageRole.TOOL,
            content="5",
            tool_call_id="call_123",
            name="calculator",
        )

        assert msg.role == MessageRole.TOOL
        assert msg.content == "5"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "calculator"

    def test_message_with_metadata(self):
        """Test message with metadata."""
        metadata = {"source": "api", "version": "1.0"}
        msg = Message(
            role=MessageRole.SYSTEM,
            content="System message",
            metadata=metadata,
        )

        assert msg.metadata == metadata
        assert msg.metadata["source"] == "api"

    def test_message_validation(self):
        """Test message validation."""
        # Invalid role - test with string that's not a valid enum value
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")

        # Empty content
        msg = Message(role=MessageRole.USER, content="")
        assert msg.content == ""  # Empty content is allowed

    def test_message_serialization(self):
        """Test message serialization."""
        msg = Message(
            role=MessageRole.USER,
            content="Test",
            metadata={"key": "value"},
        )

        msg_dict = msg.model_dump()
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test"
        assert msg_dict["metadata"]["key"] == "value"
        assert "timestamp" in msg_dict

    def test_message_to_dict(self):
        """Test to_dict method."""
        msg = Message(role=MessageRole.ASSISTANT, content="Response")
        msg_dict = msg.to_dict()

        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Response"
        assert "timestamp" in msg_dict


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            id="call_456",
            name="web_search",
            arguments={"query": "Python tutorials"},
        )

        assert tool_call.id == "call_456"
        assert tool_call.name == "web_search"
        assert tool_call.arguments == {"query": "Python tutorials"}

    def test_tool_call_serialization(self):
        """Test tool call serialization."""
        tool_call = ToolCall(
            id="call_789",
            name="calculator",
            arguments={"operation": "multiply", "a": 4, "b": 5},
        )

        tc_dict = tool_call.model_dump()
        assert tc_dict["id"] == "call_789"
        assert tc_dict["name"] == "calculator"
        assert tc_dict["arguments"]["operation"] == "multiply"


class TestToolResult:
    """Tests for ToolResult class."""

    def test_tool_result_success(self):
        """Test successful tool result."""
        result = ToolResult(
            tool_call_id="call_123",
            content="Result: 42",
            success=True,
        )

        assert result.tool_call_id == "call_123"
        assert result.content == "Result: 42"
        assert result.success is True

    def test_tool_result_error(self):
        """Test error tool result."""
        result = ToolResult(
            tool_call_id="call_456",
            content="Error: Division by zero",
            success=False,
            error="Division by zero",
        )

        assert result.tool_call_id == "call_456"
        assert result.content == "Error: Division by zero"
        assert result.success is False
        assert result.error == "Division by zero"

    def test_tool_result_default(self):
        """Test tool result with defaults."""
        result = ToolResult(
            tool_call_id="call_789",
            content="Success",
        )

        assert result.success is True  # Default value


class TestThought:
    """Tests for Thought class."""

    def test_thought_creation(self):
        """Test creating a thought."""
        thought = Thought(
            content="I need to break this problem down into steps.",
            reasoning_type="analytical",
        )

        assert thought.content == "I need to break this problem down into steps."
        assert thought.reasoning_type == "analytical"

    def test_thought_without_type(self):
        """Test thought without reasoning type."""
        thought = Thought(
            content="This seems complex.",
        )

        assert thought.content == "This seems complex."
        assert thought.reasoning_type is None

    def test_thought_with_confidence(self):
        """Test thought with confidence."""
        thought = Thought(
            content="Analyzing the problem",
            reasoning_type="problem_solving",
            confidence=0.8,
        )

        assert thought.confidence == 0.8


class TestAction:
    """Tests for Action class."""

    def test_action_creation(self):
        """Test creating an action."""
        action = Action(
            type="search",
            parameters={"query": "weather today"},
            description="Search for current weather",
        )

        assert action.type == "search"
        assert action.parameters == {"query": "weather today"}
        assert action.description == "Search for current weather"

    def test_action_minimal(self):
        """Test action with minimal fields."""
        action = Action(
            type="calculate",
            description="Calculate expression",
            parameters={"expression": "2+2"},
        )

        assert action.type == "calculate"
        assert action.parameters == {"expression": "2+2"}
        assert action.description == "Calculate expression"

    def test_action_with_expected_outcome(self):
        """Test action with expected outcome."""
        action = Action(
            type="api_call",
            description="Call user API",
            parameters={"endpoint": "/users"},
            expected_outcome="List of users",
        )

        assert action.expected_outcome == "List of users"


class TestObservation:
    """Tests for Observation class."""

    def test_observation_creation(self):
        """Test creating an observation."""
        obs = Observation(
            content="The API returned a 200 status code with data.",
            source="api_response",
        )

        assert obs.content == "The API returned a 200 status code with data."
        assert obs.source == "api_response"

    def test_observation_without_source(self):
        """Test observation without source."""
        obs = Observation(
            content="Observed behavior",
        )

        assert obs.content == "Observed behavior"
        assert obs.source is None

    def test_observation_with_relevance(self):
        """Test observation with relevance."""
        obs = Observation(
            content="Data received",
            source="sensor",
            relevance=0.95,
        )

        assert obs.relevance == 0.95


class TestConversation:
    """Tests for Conversation class."""

    @pytest.fixture
    def conversation(self):
        """Create a conversation instance."""
        return Conversation()

    def test_empty_conversation(self, conversation):
        """Test empty conversation."""
        assert len(conversation.messages) == 0
        assert conversation.get_messages() == []

    def test_add_message(self, conversation):
        """Test adding messages."""
        msg1 = Message(role=MessageRole.USER, content="Hello")
        msg2 = Message(role=MessageRole.ASSISTANT, content="Hi there!")

        conversation.add_message(msg1)
        conversation.add_message(msg2)

        assert len(conversation.messages) == 2
        assert conversation.messages[0] == msg1
        assert conversation.messages[1] == msg2

    def test_get_messages_by_role(self, conversation):
        """Test filtering messages by role."""
        conversation.add_message(Message(role=MessageRole.USER, content="Question 1"))
        conversation.add_message(Message(role=MessageRole.ASSISTANT, content="Answer 1"))
        conversation.add_message(Message(role=MessageRole.USER, content="Question 2"))
        conversation.add_message(Message(role=MessageRole.SYSTEM, content="System note"))

        user_msgs = conversation.get_messages_by_role(MessageRole.USER)
        assert len(user_msgs) == 2
        assert all(msg.role == MessageRole.USER for msg in user_msgs)

        assistant_msgs = conversation.get_messages_by_role(MessageRole.ASSISTANT)
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "Answer 1"

    def test_get_last_message(self, conversation):
        """Test getting the last message."""
        assert conversation.get_last_message() is None

        msg1 = Message(role=MessageRole.USER, content="First")
        msg2 = Message(role=MessageRole.ASSISTANT, content="Second")

        conversation.add_message(msg1)
        conversation.add_message(msg2)

        last_msg = conversation.get_last_message()
        assert last_msg == msg2
        assert last_msg.content == "Second"

    def test_clear_conversation(self, conversation):
        """Test clearing conversation."""
        conversation.add_message(Message(role=MessageRole.USER, content="Test"))
        conversation.add_message(Message(role=MessageRole.ASSISTANT, content="Response"))

        assert len(conversation.messages) == 2

        conversation.clear()
        assert len(conversation.messages) == 0
        assert conversation.get_last_message() is None

    def test_to_dict(self, conversation):
        """Test converting conversation to dict."""
        conversation.add_message(Message(role=MessageRole.USER, content="Hi"))
        conversation.add_message(Message(role=MessageRole.ASSISTANT, content="Hello"))

        conv_dict = conversation.to_dict()

        assert "messages" in conv_dict
        assert len(conv_dict["messages"]) == 2
        assert conv_dict["messages"][0]["role"] == "user"
        assert conv_dict["messages"][1]["content"] == "Hello"

    def test_from_dict(self):
        """Test creating conversation from dict."""
        conv_dict = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ]
        }

        conversation = Conversation.from_dict(conv_dict)

        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == MessageRole.USER
        assert conversation.messages[1].content == "Test answer"

    def test_conversation_with_tool_calls(self, conversation):
        """Test conversation with tool calls."""
        # User message
        conversation.add_message(Message(role=MessageRole.USER, content="What's 2+2?"))

        # Assistant with tool call
        tool_call = ToolCall(
            id="calc_1",
            name="calculator",
            arguments={"operation": "add", "a": 2, "b": 2},
        )
        conversation.add_message(
            Message(
                role=MessageRole.ASSISTANT,
                content="Let me calculate that.",
                tool_calls=[tool_call],
            )
        )

        # Tool result
        conversation.add_message(
            Message(
                role=MessageRole.TOOL,
                content="4",
                tool_call_id="calc_1",
                name="calculator",
            )
        )

        # Final response
        conversation.add_message(
            Message(role=MessageRole.ASSISTANT, content="The answer is 4.")
        )

        assert len(conversation.messages) == 4

        # Check tool messages
        tool_msgs = conversation.get_messages_by_role(MessageRole.TOOL)
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "calc_1"

    def test_conversation_metadata(self):
        """Test conversation with metadata."""
        metadata = {"session_id": "abc123", "user_id": "user_456"}
        conversation = Conversation(metadata=metadata)

        assert conversation.metadata == metadata
        assert conversation.metadata["session_id"] == "abc123"

    def test_conversation_length(self, conversation):
        """Test conversation length."""
        assert len(conversation) == 0

        for i in range(5):
            conversation.add_message(
                Message(role=MessageRole.USER, content=f"Message {i}")
            )

        assert len(conversation) == 5