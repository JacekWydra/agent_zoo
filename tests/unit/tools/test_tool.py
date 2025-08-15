"""
Unit tests for tool execution.
"""

import asyncio
from datetime import datetime

import pytest

from agent_zoo.tools.rate_limit import CallRateLimit, TokenRateLimit
from agent_zoo.tools.schema import ParameterProperty, ParameterSchema, ToolSchema
from agent_zoo.tools.tool import Tool, ToolExecutionError, ToolMetrics


class TestToolMetrics:
    """Tests for ToolMetrics."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ToolMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.total_duration_seconds == 0.0
        assert metrics.average_duration_seconds == 0.0
        assert metrics.tokens_used == 0
        assert metrics.last_called is None
        assert metrics.error_rate == 0.0

    def test_update_metrics(self):
        """Test updating metrics."""
        metrics = ToolMetrics()

        # First successful call
        metrics.update(success=True, duration=1.5, tokens=10)

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.total_duration_seconds == 1.5
        assert metrics.average_duration_seconds == 1.5
        assert metrics.tokens_used == 10
        assert metrics.error_rate == 0.0
        assert isinstance(metrics.last_called, datetime)

        # Second failed call
        metrics.update(success=False, duration=0.5, tokens=5)

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.total_duration_seconds == 2.0
        assert metrics.average_duration_seconds == 1.0
        assert metrics.tokens_used == 15
        assert metrics.error_rate == 0.5

    def test_reset_metrics(self):
        """Test resetting metrics."""
        metrics = ToolMetrics()

        # Add some data
        metrics.update(success=True, duration=1.0, tokens=20)
        metrics.update(success=False, duration=2.0, tokens=30)

        # Reset
        metrics.reset()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.total_duration_seconds == 0.0
        assert metrics.tokens_used == 0
        assert metrics.last_called is None

    def test_to_dict(self):
        """Test converting metrics to dict."""
        metrics = ToolMetrics()
        metrics.update(success=True, duration=1.0, tokens=10)

        metrics_dict = metrics.to_dict()

        assert metrics_dict["total_calls"] == 1
        assert metrics_dict["successful_calls"] == 1
        assert metrics_dict["error_rate"] == 0.0
        assert "last_called" in metrics_dict


class TestTool:
    """Tests for Tool class."""

    @pytest.fixture
    def simple_schema(self):
        """Create a simple tool schema."""
        return ToolSchema(
            name="calculator",
            description="Performs calculations",
            parameters=ParameterSchema(
                properties={
                    "operation": ParameterProperty(
                        type="string",
                        description="Operation to perform",
                        enum=["add", "subtract", "multiply", "divide"],
                    ),
                    "a": ParameterProperty(type="number", description="First number"),
                    "b": ParameterProperty(type="number", description="Second number"),
                },
                required=["operation", "a", "b"],
            ),
        )

    @pytest.fixture
    def calc_function(self):
        """Create a calculator function."""

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
    def async_calc_function(self):
        """Create an async calculator function."""

        async def calculate(operation: str, a: float, b: float) -> float:
            await asyncio.sleep(0.01)  # Simulate async work
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

    def test_tool_initialization(self, simple_schema, calc_function):
        """Test tool initialization."""
        tool = Tool(schema=simple_schema, function=calc_function)

        assert tool.schema == simple_schema
        assert tool.function == calc_function
        assert isinstance(tool.metrics, ToolMetrics)
        assert tool.rate_limits == []

    def test_tool_with_rate_limits(self, simple_schema, calc_function):
        """Test tool with rate limits."""
        rate_limit = CallRateLimit(max_calls=10, window_seconds=60)
        simple_schema.rate_limits = [rate_limit]

        tool = Tool(schema=simple_schema, function=calc_function)

        assert len(tool.rate_limits) == 1
        assert tool.rate_limits[0] == rate_limit

    def test_sync_execution(self, simple_schema, calc_function):
        """Test synchronous tool execution."""
        tool = Tool(schema=simple_schema, function=calc_function)

        result = tool.execute_sync(operation="add", a=5, b=3)

        assert result == 8
        assert tool.metrics.total_calls == 1
        assert tool.metrics.successful_calls == 1

    def test_sync_execution_with_error(self, simple_schema, calc_function):
        """Test synchronous execution with error."""
        tool = Tool(schema=simple_schema, function=calc_function)

        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute_sync(operation="divide", a=5, b=0)

        assert "Division by zero" in str(exc_info.value)
        assert tool.metrics.total_calls == 1
        assert tool.metrics.failed_calls == 1

    def test_sync_execution_with_invalid_args(self, simple_schema, calc_function):
        """Test execution with invalid arguments."""
        tool = Tool(schema=simple_schema, function=calc_function)

        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute_sync(operation="add", a=5)  # Missing 'b'

        assert "Invalid arguments" in str(exc_info.value)
        # Metrics shouldn't be updated for validation failures
        assert tool.metrics.total_calls == 0

    @pytest.mark.asyncio
    async def test_async_execution(self, simple_schema, async_calc_function):
        """Test asynchronous tool execution."""
        tool = Tool(schema=simple_schema, function=async_calc_function)

        result = await tool.execute(operation="multiply", a=4, b=7)

        assert result == 28
        assert tool.metrics.total_calls == 1
        assert tool.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_async_execution_with_sync_function(self, simple_schema, calc_function):
        """Test async execution with sync function (should work via wrapper)."""
        tool = Tool(schema=simple_schema, function=calc_function)

        result = await tool.execute(operation="subtract", a=10, b=3)

        assert result == 7
        assert tool.metrics.total_calls == 1

    @pytest.mark.asyncio
    async def test_rate_limiting(self, simple_schema, async_calc_function):
        """Test rate limiting enforcement."""
        rate_limit = CallRateLimit(max_calls=2, window_seconds=1)
        simple_schema.rate_limits = [rate_limit]

        tool = Tool(schema=simple_schema, function=async_calc_function)

        # First two calls should succeed
        result1 = await tool.execute(operation="add", a=1, b=1)
        result2 = await tool.execute(operation="add", a=2, b=2)

        assert result1 == 2
        assert result2 == 4

        # Third call should fail due to rate limit
        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(operation="add", a=3, b=3)

        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout(self, simple_schema):
        """Test execution timeout."""
        simple_schema.timeout_seconds = 0.1

        async def slow_function(**kwargs):
            await asyncio.sleep(1.0)
            return "too slow"

        tool = Tool(schema=simple_schema, function=slow_function)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(operation="add", a=1, b=1)

        assert "timed out" in str(exc_info.value)
        assert tool.metrics.failed_calls == 1

    def test_get_info(self, simple_schema, calc_function):
        """Test getting tool information."""
        tool = Tool(schema=simple_schema, function=calc_function)

        # Execute a few times to generate metrics
        tool.execute_sync(operation="add", a=1, b=1)
        tool.execute_sync(operation="multiply", a=2, b=3)

        info = tool.get_info()

        assert info["name"] == "calculator"
        assert info["description"] == "Performs calculations"
        assert info["metrics"]["total_calls"] == 2
        assert info["metrics"]["successful_calls"] == 2
        assert "parameters" in info

    def test_reset_metrics(self, simple_schema, calc_function):
        """Test resetting tool metrics."""
        tool = Tool(schema=simple_schema, function=calc_function)

        # Execute and generate metrics
        tool.execute_sync(operation="add", a=1, b=1)
        assert tool.metrics.total_calls == 1

        # Reset
        tool.reset_metrics()
        assert tool.metrics.total_calls == 0

    @pytest.mark.asyncio
    async def test_token_rate_limiting(self, simple_schema, async_calc_function):
        """Test token-based rate limiting."""
        # Create a rate limit that allows 10 tokens
        rate_limit = TokenRateLimit(max_tokens=10, window_seconds=60)
        simple_schema.rate_limits = [rate_limit]

        tool = Tool(schema=simple_schema, function=async_calc_function)

        # First call with 5 tokens should succeed
        result = await tool.execute(operation="add", a=1, b=1, _token_count=5)
        assert result == 2

        # Second call with 4 tokens should succeed (total 9)
        result = await tool.execute(operation="add", a=2, b=2, _token_count=4)
        assert result == 4

        # Third call with 5 tokens should fail (would be 14 total)
        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(operation="add", a=3, b=3, _token_count=5)

        assert "Rate limit exceeded" in str(exc_info.value)

    def test_tool_with_preprocessing(self, simple_schema, calc_function):
        """Test tool with preprocessing function."""

        def preprocess(args: dict) -> dict:
            # Convert string numbers to floats
            if "a" in args and isinstance(args["a"], str):
                args["a"] = float(args["a"])
            if "b" in args and isinstance(args["b"], str):
                args["b"] = float(args["b"])
            return args

        tool = Tool(
            schema=simple_schema,
            function=calc_function,
            preprocess=preprocess,
        )

        # Execute with string arguments
        result = tool.execute_sync(operation="add", a="5", b="3")
        assert result == 8

    def test_tool_with_postprocessing(self, simple_schema, calc_function):
        """Test tool with postprocessing function."""

        def postprocess(result: float) -> dict:
            return {"result": result, "formatted": f"The answer is {result}"}

        tool = Tool(
            schema=simple_schema,
            function=calc_function,
            postprocess=postprocess,
        )

        result = tool.execute_sync(operation="multiply", a=3, b=4)

        assert result["result"] == 12
        assert result["formatted"] == "The answer is 12"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, simple_schema, async_calc_function):
        """Test concurrent execution of tool."""
        tool = Tool(schema=simple_schema, function=async_calc_function)

        # Execute multiple operations concurrently
        tasks = [tool.execute(operation="add", a=i, b=i) for i in range(5)]

        results = await asyncio.gather(*tasks)

        assert results == [0, 2, 4, 6, 8]
        assert tool.metrics.total_calls == 5
        assert tool.metrics.successful_calls == 5

    def test_tool_representation(self, simple_schema, calc_function):
        """Test tool string representation."""
        tool = Tool(schema=simple_schema, function=calc_function)

        repr_str = repr(tool)
        assert "Tool" in repr_str
        assert "calculator" in repr_str

    @pytest.mark.asyncio
    async def test_execution_with_side_effects(self, simple_schema):
        """Test tool execution with side effects."""
        side_effects = []

        async def function_with_side_effects(operation: str, a: float, b: float):
            side_effects.append((operation, a, b))
            return a + b if operation == "add" else a - b

        tool = Tool(schema=simple_schema, function=function_with_side_effects)

        result = await tool.execute(operation="add", a=5, b=3)

        assert result == 8
        assert side_effects == [("add", 5, 3)]

    def test_tool_with_custom_validator(self, simple_schema, calc_function):
        """Test tool with custom argument validator."""

        def custom_validator(args: dict) -> tuple[bool, list[str]]:
            errors = []
            if "a" in args and "b" in args:
                if args["a"] == args["b"]:
                    errors.append("Numbers must be different")
            return len(errors) == 0, errors

        tool = Tool(
            schema=simple_schema,
            function=calc_function,
            validator=custom_validator,
        )

        # Should fail custom validation
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute_sync(operation="add", a=5, b=5)

        assert "Numbers must be different" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_streaming_support(self, simple_schema):
        """Test tool with streaming support."""
        simple_schema.supports_streaming = True

        async def streaming_function(**kwargs):
            for i in range(3):
                yield f"Part {i}"
                await asyncio.sleep(0.01)

        tool = Tool(schema=simple_schema, function=streaming_function)

        # Collect streamed results
        results = []
        async for part in tool.execute_stream({"operation": "add", "a": 1, "b": 1}):
            results.append(part)

        assert results == ["Part 0", "Part 1", "Part 2"]
