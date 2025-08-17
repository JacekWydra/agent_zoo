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


# Memory System Fixtures
@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns deterministic routing decisions."""
    import json
    
    class MockLLMClient:
        async def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
            # Extract the actual content being classified from the prompt
            # Look for "Content to classify: " line
            content_line = ""
            for line in prompt.split('\n'):
                if "content to classify:" in line.lower():
                    content_line = line.lower()
                    break
            
            # For query routing (different prompt format)
            if "query:" in prompt.lower() and "which memory types" in prompt.lower():
                return json.dumps({
                    "types": ["WORKING", "SEMANTIC"],
                    "reasoning": "General query"
                })
            
            # Check patterns in the actual content, not the full prompt
            # Check for procedural content
            if "how to" in content_line or "steps:" in content_line or "procedure:" in content_line or "instructions:" in content_line:
                return json.dumps({
                    "type": "PROCEDURAL",
                    "confidence": 0.85,
                    "reasoning": "Contains procedural knowledge"
                })
            # Check for episodic content
            elif "remember" in content_line or "yesterday" in content_line or "happened" in content_line or "experienced" in content_line:
                return json.dumps({
                    "type": "EPISODIC",
                    "confidence": 0.8,
                    "reasoning": "Contains experiential information"
                })
            # Check for semantic/factual content  
            elif "capital of" in content_line or "fact" in content_line or "definition" in content_line:
                return json.dumps({
                    "type": "SEMANTIC",
                    "confidence": 0.9,
                    "reasoning": "Contains factual information"
                })
            else:
                return json.dumps({
                    "type": "WORKING",
                    "confidence": 0.7,
                    "reasoning": "Default to working memory"
                })
    
    return MockLLMClient()


@pytest.fixture
def memory_manager_config():
    """Create MemoryManagerConfig with test settings."""
    from agent_zoo.core.memory.manager import MemoryManagerConfig
    
    return MemoryManagerConfig(
        persist_directory=None,  # In-memory for tests
        use_llm_router=True,
        cache_routing_decisions=True,
        max_context_tokens=1000,
        default_search_results=5,
        auto_capture=True,
        capture_messages=True,
        auto_consolidate=False,  # Disable for predictable tests
        consolidation_interval_seconds=300,
    )


@pytest.fixture
def sample_working_item():
    """Create a WorkingMemoryItem for testing."""
    from agent_zoo.core.memory.items import WorkingMemoryItem
    from datetime import datetime, timedelta
    
    return WorkingMemoryItem(
        content="Current task: analyzing data",
        importance=7.0,
        is_active=True,
        priority=8.0,
        task_id="task_123",
        expires_at=datetime.now() + timedelta(hours=1),
        token_count=10,
        source="user_input",
        related_items=["item_1", "item_2"],
    )


@pytest.fixture
def sample_semantic_item():
    """Create a SemanticMemoryItem for testing."""
    from agent_zoo.core.memory.items import SemanticMemoryItem
    
    return SemanticMemoryItem(
        content="Paris is the capital of France",
        importance=9.0,
        concepts=["Paris", "France", "capital"],
        confidence=1.0,
        verified=True,
        source="encyclopedia",
        source_reliability=0.95,
        domain="geography",
        category="capitals",
    )


@pytest.fixture
def sample_episodic_item():
    """Create an EpisodicMemoryItem for testing."""
    from agent_zoo.core.memory.items import EpisodicMemoryItem
    from datetime import datetime
    
    return EpisodicMemoryItem(
        content="User asked about weather and I provided forecast",
        importance=5.0,
        event_type="conversation",
        event_time=datetime.now(),
        duration_seconds=30.5,
        sequence_number=1,
        participants=["user", "assistant"],
        location="chat_session",
        emotional_valence=0.2,
        significance=6.0,
        outcomes=["weather_info_provided"],
        lessons_learned=["user_prefers_detailed_forecasts"],
    )


@pytest.fixture
def sample_procedural_item():
    """Create a ProceduralMemoryItem for testing."""
    from agent_zoo.core.memory.items import ProceduralMemoryItem
    
    return ProceduralMemoryItem(
        content="How to make coffee",
        importance=6.0,
        procedure_name="Make Coffee",
        steps=[
            {"step": 1, "action": "Boil water"},
            {"step": 2, "action": "Add coffee grounds"},
            {"step": 3, "action": "Pour hot water"},
            {"step": 4, "action": "Wait 4 minutes"},
            {"step": 5, "action": "Serve"},
        ],
        prerequisites=["coffee_beans", "hot_water"],
        required_tools=["coffee_maker", "mug"],
        complexity=3.0,
        estimated_duration=300.0,
        success_rate=0.95,
        execution_count=20,
        domain="culinary",
        skill_type="beverage_preparation",
        variations=["french_press", "espresso"],
        common_errors=["water_too_hot", "over_extraction"],
    )


@pytest.fixture
def mock_embedding_function():
    """Mock embedding function that returns fixed vectors."""
    class MockEmbeddingFunction:
        def __call__(self, texts: list[str]) -> list[list[float]]:
            # Return fixed-size embeddings based on text hash for consistency
            embeddings = []
            for text in texts:
                # Create a deterministic embedding based on text
                hash_val = hash(text) % 1000
                embedding = [hash_val / 1000.0] * 384  # Match all-MiniLM-L6-v2 dimension
                embeddings.append(embedding)
            return embeddings
    
    return MockEmbeddingFunction()