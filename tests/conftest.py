"""
Shared pytest fixtures for agent_zoo tests.
"""

import asyncio

import pytest

from agent_zoo.core.base import AgentConfig
from agent_zoo.interfaces.messages import Message
from agent_zoo.interfaces.state import AgentState, AgentStatus
from agent_zoo.tools.rate_limit import CallRateLimit, TokenRateLimit
from agent_zoo.tools.schema import (
    ParameterProperty,
    ParameterSchema,
    ToolSchema,
)
from agent_zoo.tools.tool import Tool


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def agent_config():
    """Create a sample agent configuration."""
    return AgentConfig(
        name="test_agent",
        description="Test agent for unit tests",
        max_iterations=5,
        timeout_seconds=30.0,
        enable_monitoring=True,
        enable_caching=False,
        retry_attempts=2,
    )


@pytest.fixture
def agent_state():
    """Create a sample agent state."""
    return AgentState(
        status=AgentStatus.IDLE,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        context={"session_id": "test123"},
        current_goal="Test goal",
        plan_steps=["Step 1", "Step 2"],
        iteration_count=0,
        total_tokens_used=100,
    )


@pytest.fixture
def simple_tool_schema():
    """Create a simple tool schema for testing."""
    return ToolSchema(
        name="calculator",
        description="Performs basic arithmetic operations",
        parameters=ParameterSchema(
            properties={
                "operation": ParameterProperty(
                    type="string",
                    description="The operation to perform",
                    enum=["add", "subtract", "multiply", "divide"],
                ),
                "a": ParameterProperty(
                    type="number",
                    description="First operand",
                ),
                "b": ParameterProperty(
                    type="number",
                    description="Second operand",
                ),
            },
            required=["operation", "a", "b"],
        ),
        tags=["math", "calculation"],
        timeout_seconds=10.0,
    )


@pytest.fixture
def rate_limited_tool_schema():
    """Create a tool schema with rate limits."""
    return ToolSchema(
        name="api_caller",
        description="Calls an external API",
        parameters=ParameterSchema(
            properties={
                "endpoint": ParameterProperty(
                    type="string",
                    description="API endpoint to call",
                ),
            },
            required=["endpoint"],
        ),
        rate_limits=[
            CallRateLimit(max_calls=10, window_seconds=60),
            TokenRateLimit(max_tokens=1000, window_seconds=60),
        ],
        tags=["api", "network"],
    )


@pytest.fixture
def calculator_function():
    """Create a calculator function for tool testing."""

    def calculate(operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return calculate


@pytest.fixture
def async_calculator_function():
    """Create an async calculator function for tool testing."""

    async def calculate(operation: str, a: float, b: float) -> float:
        await asyncio.sleep(0.01)  # Simulate async operation
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return calculate


@pytest.fixture
def simple_tool(simple_tool_schema, calculator_function):
    """Create a simple tool instance."""
    return Tool(schema=simple_tool_schema, function=calculator_function)


@pytest.fixture
def async_tool(simple_tool_schema, async_calculator_function):
    """Create an async tool instance."""
    return Tool(schema=simple_tool_schema, function=async_calculator_function)


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="Let me calculate that for you."),
        Message(
            role="tool",
            content="4",
            tool_call_id="calc_123",
            name="calculator",
        ),
        Message(role="assistant", content="The answer is 4."),
    ]


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll help you with that calculation.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def tool_registry():
    """Create an empty tool registry for testing."""
    from agent_zoo.tools.registry import ToolRegistry

    return ToolRegistry()


@pytest.fixture
def populated_tool_registry(tool_registry, simple_tool):
    """Create a tool registry with some tools."""
    tool_registry.register(simple_tool)

    # Add more tools with different tags
    search_schema = ToolSchema(
        name="web_search",
        description="Search the web",
        parameters=ParameterSchema(
            properties={
                "query": ParameterProperty(
                    type="string",
                    description="Search query",
                ),
            },
            required=["query"],
        ),
        tags=["search", "web"],
    )
    search_tool = Tool(
        schema=search_schema, function=lambda query: f"Results for: {query}"
    )
    tool_registry.register(search_tool)

    return tool_registry