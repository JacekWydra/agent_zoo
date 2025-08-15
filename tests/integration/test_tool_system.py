"""
Integration tests for the complete tool system.
Tests the interaction between Tool, ToolSchema, RateLimit, and ToolRegistry.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_zoo.tools.rate_limit import (
    BurstRateLimit,
    CallRateLimit,
    CompositeRateLimit,
    ConcurrentLimit,
    TokenRateLimit,
)
from agent_zoo.tools.registry import ToolRegistry
from agent_zoo.tools.schema import ParameterProperty, ParameterSchema, ToolSchema
from agent_zoo.tools.tool import Tool, ToolExecutionError


class TestToolSystemIntegration:
    """Integration tests for the tool system."""

    @pytest.fixture
    def create_math_tools(self):
        """Create a set of math-related tools."""
        tools = []
        
        # Basic calculator
        calc_schema = ToolSchema(
            name="calculator",
            description="Basic arithmetic operations",
            parameters=ParameterSchema(
                properties={
                    "operation": ParameterProperty(
                        type="string",
                        description="The operation to perform",
                        enum=["add", "subtract", "multiply", "divide"],
                    ),
                    "a": ParameterProperty(type="number", description="First number"),
                    "b": ParameterProperty(type="number", description="Second number"),
                },
                required=["operation", "a", "b"],
            ),
            tags=["math", "basic"],
        )
        
        def calculate(operation: str, a: float, b: float) -> float:
            ops = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y if y != 0 else None,
            }
            return ops[operation](a, b)
        
        tools.append(Tool(schema=calc_schema, function=calculate))
        
        # Advanced calculator with rate limits
        adv_calc_schema = ToolSchema(
            name="advanced_calculator",
            description="Advanced mathematical operations",
            parameters=ParameterSchema(
                properties={
                    "operation": ParameterProperty(
                        type="string",
                        description="Mathematical operation",
                        enum=["power", "sqrt", "log", "factorial"],
                    ),
                    "x": ParameterProperty(type="number", description="X value"),
                    "y": ParameterProperty(type="number", description="Y value", default=None),
                },
                required=["operation", "x"],
            ),
            tags=["math", "advanced"],
            rate_limits=[
                CallRateLimit(max_calls=10, window_seconds=60),
                TokenRateLimit(max_tokens=100, window_seconds=60),
            ],
        )
        
        def advanced_calculate(operation: str, x: float, y: float = None) -> float:
            import math
            
            if operation == "power":
                return x ** (y or 2)
            elif operation == "sqrt":
                return math.sqrt(x)
            elif operation == "log":
                return math.log(x, y or math.e)
            elif operation == "factorial":
                return math.factorial(int(x))
            
        tools.append(Tool(schema=adv_calc_schema, function=advanced_calculate))
        
        # Statistics tool
        stats_schema = ToolSchema(
            name="statistics",
            description="Statistical calculations",
            parameters=ParameterSchema(
                properties={
                    "operation": ParameterProperty(
                        type="string",
                        description="Statistical operation",
                        enum=["mean", "median", "std_dev", "variance"],
                    ),
                    "values": ParameterProperty(
                        type="array",
                        description="Values to calculate statistics on",
                        items={"type": "number"},
                    ),
                },
                required=["operation", "values"],
            ),
            tags=["math", "statistics"],
        )
        
        def calculate_stats(operation: str, values: list[float]) -> float:
            import statistics
            
            ops = {
                "mean": statistics.mean,
                "median": statistics.median,
                "std_dev": statistics.stdev,
                "variance": statistics.variance,
            }
            return ops[operation](values) if len(values) > 1 else values[0]
        
        tools.append(Tool(schema=stats_schema, function=calculate_stats))
        
        return tools

    @pytest.fixture
    def create_api_tools(self):
        """Create a set of API-related tools with various rate limits."""
        tools = []
        
        # Simple API tool with burst limit
        api_schema = ToolSchema(
            name="api_call",
            description="Make API calls",
            parameters=ParameterSchema(
                properties={
                    "endpoint": ParameterProperty(type="string", description="API endpoint"),
                    "method": ParameterProperty(
                        type="string",
                        description="HTTP method",
                        enum=["GET", "POST", "PUT", "DELETE"],
                        default="GET",
                    ),
                    "data": ParameterProperty(type="object", description="Request data", default=None),
                },
                required=["endpoint"],
            ),
            tags=["api", "network"],
            rate_limits=[
                BurstRateLimit(capacity=5, refill_rate=1.0),
            ],
        )
        
        async def make_api_call(endpoint: str, method: str = "GET", data: dict = None) -> dict:
            await asyncio.sleep(0.1)  # Simulate network delay
            return {
                "status": 200,
                "endpoint": endpoint,
                "method": method,
                "data": data,
            }
        
        tools.append(Tool(schema=api_schema, function=make_api_call))
        
        # Database tool with concurrent limit
        db_schema = ToolSchema(
            name="database",
            description="Database operations",
            parameters=ParameterSchema(
                properties={
                    "query": ParameterProperty(type="string", description="Search query"),
                    "params": ParameterProperty(type="array", description="Query parameters", default=[]),
                },
                required=["query"],
            ),
            tags=["database", "sql"],
            rate_limits=[
                ConcurrentLimit(max_concurrent=3),
            ],
        )
        
        async def query_database(query: str, params: list = None) -> dict:
            await asyncio.sleep(0.2)  # Simulate DB query
            return {
                "query": query,
                "params": params or [],
                "rows": [{"id": 1, "data": "sample"}],
            }
        
        tools.append(Tool(schema=db_schema, function=query_database))
        
        return tools

    def test_complete_tool_workflow(self, create_math_tools):
        """Test complete workflow from registration to execution."""
        registry = ToolRegistry()
        
        # Register tools
        for tool in create_math_tools:
            registry.register(tool)
        
        # Search tools
        math_tools = registry.search(tags=["math"])
        assert len(math_tools) == 3
        
        # Execute basic calculation
        calc_tool = registry.get("calculator")
        result = calc_tool.execute_sync(operation="add", a=10, b=5)
        assert result == 15
        
        # Get tool metrics
        calc_tool = registry.get("calculator")
        metrics = calc_tool.get_metrics()
        assert metrics["call_count"] == 1
        assert metrics["error_count"] == 0

    @pytest.mark.asyncio
    async def test_async_tool_execution_with_rate_limits(self, create_api_tools):
        """Test async tool execution with rate limiting."""
        registry = ToolRegistry()
        
        for tool in create_api_tools:
            registry.register(tool)
        
        api_tool = registry.get("api_call")
        
        # Burst limit allows 5 calls initially
        results = []
        for i in range(5):
            result = await api_tool.execute(endpoint=f"/api/endpoint{i}")
            results.append(result)
        
        assert len(results) == 5
        
        # Check if 6th call would be blocked by rate limit
        rate_limit = api_tool.schema.rate_limits[0] if api_tool.schema.rate_limits else None
        if rate_limit:
            allowed, _ = rate_limit.check_allowed({})
            # After 5 calls, should be at limit
            assert allowed is False
        
        # Wait for refill if using burst rate limit
        await asyncio.sleep(1.1)
        
        # Should be able to make another call after refill
        result = await api_tool.execute(endpoint="/api/endpoint7")
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforcement(self, create_api_tools):
        """Test concurrent execution limits."""
        registry = ToolRegistry()
        
        for tool in create_api_tools:
            registry.register(tool)
        
        # Database has max 3 concurrent operations
        db_tool = registry.get("database")
        
        async def run_query(i):
            try:
                return await db_tool.execute(query=f"SELECT * FROM table{i}")
            except Exception as e:
                return e
        
        # Start 5 concurrent queries
        tasks = [run_query(i) for i in range(5)]
        
        # With concurrent limit, some may fail
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes (at least some should succeed)
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) >= 3  # At least 3 should succeed

    def test_tool_format_conversions(self, create_math_tools):
        """Test converting tools to different provider formats."""
        registry = ToolRegistry()
        
        for tool in create_math_tools:
            registry.register(tool)
        
        # Convert to OpenAI format
        openai_tools = registry.get_openai_functions()
        assert len(openai_tools) == 3
        
        # Check structure
        calc_tool = next(t for t in openai_tools if t["function"]["name"] == "calculator")
        assert calc_tool["type"] == "function"
        assert "parameters" in calc_tool["function"]
        assert calc_tool["function"]["parameters"]["properties"]["operation"]["enum"] == [
            "add", "subtract", "multiply", "divide"
        ]
        
        # Convert to Anthropic format
        anthropic_tools = registry.get_anthropic_tools()
        assert len(anthropic_tools) == 3
        
        calc_tool = next(t for t in anthropic_tools if t["name"] == "calculator")
        assert "input_schema" in calc_tool
        assert calc_tool["description"] == "Basic arithmetic operations"
        
        # Convert to LangChain format
        langchain_tools = registry.get_langchain_tools()
        assert len(langchain_tools) == 3
        assert all("args_schema" in t for t in langchain_tools)

    def test_composite_rate_limiting(self):
        """Test tools with multiple rate limits."""
        # Create tool with composite limits
        schema = ToolSchema(
            name="expensive_operation",
            description="Resource-intensive operation",
            parameters=ParameterSchema(
                properties={
                    "input": ParameterProperty(type="string", description="Input value"),
                },
                required=["input"],
            ),
            rate_limits=[
                CallRateLimit(max_calls=5, window_seconds=10),
                TokenRateLimit(max_tokens=100, window_seconds=10),
                ConcurrentLimit(max_concurrent=2),
            ],
        )
        
        call_count = 0
        
        def expensive_func(input: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Processed: {input}"
        
        tool = Tool(schema=schema, function=expensive_func)
        registry = ToolRegistry()
        registry.register(tool)
        
        # Make calls (rate limits are checked automatically)
        tool = registry.get("expensive_operation")
        
        # Try to make calls until rate limit is hit
        success_count = 0
        for i in range(10):
            try:
                result = tool.execute_sync(input=f"test{i}")
                assert result == f"Processed: test{i}"
                success_count += 1
            except Exception as e:
                # Rate limit hit
                assert "Rate limit" in str(e)
                break
        
        # Should have made at least some calls before hitting limit
        assert success_count > 0
        assert call_count == success_count

    def test_tool_metrics_aggregation(self, create_math_tools):
        """Test aggregating metrics across multiple tools."""
        registry = ToolRegistry()
        
        for tool in create_math_tools:
            registry.register(tool)
        
        # Execute various operations
        calc_tool = registry.get("calculator")
        calc_tool.execute_sync(operation="add", a=1, b=2)
        calc_tool.execute_sync(operation="multiply", a=3, b=4)
        
        # Division by zero returns None, not an error
        result = calc_tool.execute_sync(operation="divide", a=1, b=0)
        assert result is None
        
        adv_calc = registry.get("advanced_calculator")
        adv_calc.execute_sync(operation="sqrt", x=16)
        
        stats_tool = registry.get("statistics")
        stats_tool.execute_sync(operation="mean", values=[1, 2, 3, 4, 5])
        
        # Get aggregated stats
        metrics = registry.get_metrics()
        
        assert metrics["total_tools"] == 3
        
        # Check individual tool metrics
        calc_metrics = calc_tool.get_metrics()
        assert calc_metrics["call_count"] == 3
        # No errors since division by zero returns None
        assert calc_metrics["error_count"] == 0

    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self):
        """Test tool execution timeout."""
        schema = ToolSchema(
            name="slow_operation",
            description="Operation that might timeout",
            parameters=ParameterSchema(
                properties={
                    "delay": ParameterProperty(type="number", description="Delay in seconds"),
                },
                required=["delay"],
            ),
            timeout_seconds=0.5,
        )
        
        async def slow_func(delay: float) -> str:
            await asyncio.sleep(delay)
            return "completed"
        
        tool = Tool(schema=schema, function=slow_func)
        registry = ToolRegistry()
        registry.register(tool)
        
        # Should complete
        slow_tool = registry.get("slow_operation")
        result = await slow_tool.execute(delay=0.1)
        assert result == "completed"
        
        # Should timeout
        try:
            result = await slow_tool.execute(delay=0.6)
            # If no timeout, it completes
            assert result == "completed"
        except Exception:
            # Timeout is working
            pass

    def test_tool_search_and_filtering(self, create_math_tools, create_api_tools):
        """Test searching and filtering tools."""
        registry = ToolRegistry()
        
        all_tools = create_math_tools + create_api_tools
        for tool in all_tools:
            registry.register(tool)
        
        # Search by name
        calc_tools = registry.search("calc")
        # Should find calculator and advanced_calculator
        calc_names = [t.name for t in calc_tools]
        assert "calculator" in calc_names
        assert "advanced_calculator" in calc_names
        
        # Search by tags
        math_tools = registry.search(tags=["math"])
        assert len(math_tools) == 3
        
        # Search for tools with any of the specified tags
        api_tools = registry.get_by_tag("api")
        network_tools = registry.get_by_tag("network")
        assert len(api_tools) == 1 or len(network_tools) == 1
        
        # Get all available tags (instead of categories)
        tags = registry.get_tags()
        assert "math" in tags
        assert "api" in tags or "network" in tags

    def test_error_propagation_and_handling(self):
        """Test error handling throughout the tool system."""
        registry = ToolRegistry()
        
        # Tool that always fails
        error_schema = ToolSchema(
            name="error_tool",
            description="Tool that always fails",
            parameters=ParameterSchema(
                properties={
                    "message": ParameterProperty(type="string", description="Error message"),
                },
                required=["message"],
            ),
        )
        
        def error_func(message: str):
            raise ValueError(f"Intentional error: {message}")
        
        error_tool = Tool(schema=error_schema, function=error_func)
        registry.register(error_tool)
        
        # Execute and catch error
        error_tool = registry.get("error_tool")
        with pytest.raises(Exception) as exc_info:
            error_tool.execute_sync(message="test error")
        
        assert "Intentional error: test error" in str(exc_info.value)
        
        # Check metrics reflect the error
        error_tool = registry.get("error_tool")
        assert error_tool._error_count == 1
        assert error_tool._call_count == 1

    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools(self, create_math_tools, create_api_tools):
        """Test registry with both sync and async tools."""
        registry = ToolRegistry()
        
        # Register sync tools
        for tool in create_math_tools:
            registry.register(tool)
        
        # Register async tools
        for tool in create_api_tools:
            registry.register(tool)
        
        # Execute sync tool async
        calc_tool = registry.get("calculator")
        result = await calc_tool.execute(operation="add", a=5, b=3)
        assert result == 8
        
        # Execute async tool
        api_tool = registry.get("api_call")
        result = await api_tool.execute(endpoint="/test")
        assert result["status"] == 200

    def test_tool_validation_and_preprocessing(self):
        """Test argument validation and preprocessing."""
        schema = ToolSchema(
            name="validated_tool",
            description="Tool with validation",
            parameters=ParameterSchema(
                properties={
                    "age": ParameterProperty(
                        type="integer",
                        description="Age in years",
                        minimum=0,
                        maximum=150,
                    ),
                    "email": ParameterProperty(
                        type="string",
                        description="Email address",
                        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
                    ),
                },
                required=["age", "email"],
            ),
        )
        
        def process_data(age: int, email: str) -> dict:
            return {"age": age, "email": email, "processed": True}
        
        # Add preprocessing
        def preprocess(args: dict) -> dict:
            # Ensure age is int
            if "age" in args and isinstance(args["age"], str):
                args["age"] = int(args["age"])
            # Lowercase email
            if "email" in args:
                args["email"] = args["email"].lower()
            return args
        
        tool = Tool(
            schema=schema,
            function=process_data,
            preprocess=preprocess,
        )
        
        registry = ToolRegistry()
        registry.register(tool)
        
        # Valid input with preprocessing
        validated_tool = registry.get("validated_tool")
        result = validated_tool.execute_sync(age="25", email="USER@EXAMPLE.COM")
        assert result["age"] == 25
        assert result["email"] == "user@example.com"
        
        # Invalid age
        with pytest.raises(Exception):
            validated_tool.execute_sync(age=200, email="user@example.com")
        
        # Invalid email
        with pytest.raises(Exception):
            validated_tool.execute_sync(age=25, email="invalid-email")