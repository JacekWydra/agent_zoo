"""
Tool implementation class for executing functions with defined schemas.
"""

import asyncio
import inspect
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable

import structlog

from agent_zoo.tools.rate_limit import ConcurrentLimit
from agent_zoo.tools.schema import ToolSchema

logger = structlog.get_logger()


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""

    pass


class ToolMetrics:
    """Metrics tracking for tool execution."""

    def __init__(self):
        """Initialize metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_duration_seconds = 0.0
        self.average_duration_seconds = 0.0
        self.tokens_used = 0
        self.last_called: datetime | None = None
        self.error_rate = 0.0

    def update(self, success: bool, duration: float, tokens: int = 0) -> None:
        """Update metrics with execution results."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        self.total_duration_seconds += duration
        self.average_duration_seconds = self.total_duration_seconds / self.total_calls
        self.tokens_used += tokens
        self.last_called = datetime.now()
        self.error_rate = self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_duration_seconds = 0.0
        self.average_duration_seconds = 0.0
        self.tokens_used = 0
        self.last_called = None
        self.error_rate = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_duration_seconds": self.total_duration_seconds,
            "average_duration_seconds": self.average_duration_seconds,
            "tokens_used": self.tokens_used,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "error_rate": self.error_rate,
        }


class Tool:
    """
    Executable tool with a defined schema.

    Similar to how BaseAgent uses AgentConfig, Tool uses ToolSchema
    to define its interface while handling the actual execution.
    """

    def __init__(
        self,
        schema: ToolSchema,
        function: Callable[..., Any] | Callable[..., Awaitable[Any]],
        preprocess: Callable[[dict], dict] | None = None,
        postprocess: Callable[[Any], Any] | None = None,
        validator: Callable[[dict], tuple[bool, list[str]]] | None = None,
    ):
        """
        Initialize a tool with schema and function.

        Args:
            schema: Tool schema defining the interface
            function: Callable that implements the tool
            preprocess: Optional preprocessing function for arguments
            postprocess: Optional postprocessing function for results
            validator: Optional custom validator for arguments
        """
        self.schema = schema
        self.function = function
        self._function = function
        self._is_async = inspect.iscoroutinefunction(function)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.validator = validator
        self.rate_limits = schema.rate_limits if schema.rate_limits else []
        
        # Execution metrics
        self.metrics = ToolMetrics()
        self._call_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time: datetime | None = None
        self._error_count = 0

        logger.info(
            f"Tool initialized: {schema.name}",
            is_async=self._is_async,
            tags=schema.tags,
            rate_limits=len(schema.rate_limits),
        )

    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Arguments to pass to the function

        Returns:
            Result from the function execution

        Raises:
            ValueError: If arguments are invalid
            RuntimeError: If execution fails or rate limit exceeded
        """
        # Extract special parameters
        token_count = kwargs.pop("_token_count", 0)
        
        # Apply preprocessing if provided
        if self.preprocess:
            kwargs = self.preprocess(kwargs)
        
        # Validate arguments
        is_valid, errors = self.schema.validate_arguments(kwargs)
        if not is_valid:
            raise ToolExecutionError(f"Invalid arguments: {'; '.join(errors)}")
        
        # Apply custom validator if provided
        if self.validator:
            is_valid, custom_errors = self.validator(kwargs)
            if not is_valid:
                raise ToolExecutionError(f"Validation failed: {'; '.join(custom_errors)}")

        # Generate execution ID for tracking
        execution_id = str(uuid.uuid4())

        # Prepare context for rate limiting
        context = {
            "execution_id": execution_id,
            "tool_name": self.schema.name,
            "timestamp": datetime.now(),
            "token_count": token_count,
        }

        # Check all rate limits
        for rate_limit in self.schema.rate_limits:
            allowed, error = rate_limit.check_allowed(context)
            if not allowed:
                raise ToolExecutionError(f"Rate limit exceeded: {error}")

        # Record usage in all rate limits (before execution)
        for rate_limit in self.schema.rate_limits:
            rate_limit.record_usage(context)

        # Handle concurrent limits specially
        concurrent_limits = [
            rl for rl in self.schema.rate_limits if isinstance(rl, ConcurrentLimit)
        ]

        # Record execution start
        start_time = datetime.now()
        self._call_count += 1

        try:
            # Execute function with timeout
            if self.schema.timeout_seconds > 0:
                if self._is_async:
                    result = await asyncio.wait_for(
                        self._function(**kwargs), timeout=self.schema.timeout_seconds
                    )
                else:
                    # Run sync function in thread pool with timeout
                    loop = asyncio.get_event_loop()
                    import functools
                    func = functools.partial(self._function, **kwargs)
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func),
                        timeout=self.schema.timeout_seconds,
                    )
            else:
                # No timeout
                if self._is_async:
                    result = await self._function(**kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    import functools
                    func = functools.partial(self._function, **kwargs)
                    result = await loop.run_in_executor(None, func)

            # Record success metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._total_execution_time += execution_time
            self._last_execution_time = datetime.now()
            
            # Update metrics object
            self.metrics.update(success=True, duration=execution_time, tokens=token_count)

            logger.debug(
                f"Tool executed successfully: {self.schema.name}",
                execution_time=execution_time,
                call_count=self._call_count,
            )

            # Apply postprocessing if provided
            if self.postprocess:
                result = self.postprocess(result)
            
            return result

        except asyncio.TimeoutError:
            self._error_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update(success=False, duration=execution_time)
            logger.error(
                f"Tool execution timed out: {self.schema.name}", timeout=self.schema.timeout_seconds
            )
            raise ToolExecutionError(
                f"Tool '{self.schema.name}' timed out after {self.schema.timeout_seconds} seconds"
            )

        except Exception as e:
            self._error_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update(success=False, duration=execution_time)
            logger.error(
                f"Tool execution failed: {self.schema.name}",
                error=str(e),
                error_count=self._error_count,
            )
            raise ToolExecutionError(f"Tool execution failed: {e}") from e

        finally:
            # Release concurrent execution slots
            for limit in concurrent_limits:
                limit.release(execution_id)

    def execute_sync(self, **kwargs) -> Any:
        """
        Synchronous execution wrapper.

        Args:
            **kwargs: Arguments to pass to the function

        Returns:
            Result from the function execution

        Raises:
            RuntimeError: If the tool is async or rate limited
        """
        if self._is_async:
            raise RuntimeError(
                f"Tool '{self.schema.name}' is async. Use execute() or __call__() instead."
            )

        # Extract special parameters
        token_count = kwargs.pop("_token_count", 0)
        
        # Apply preprocessing if provided
        if self.preprocess:
            kwargs = self.preprocess(kwargs)
        
        # Validate arguments
        is_valid, errors = self.schema.validate_arguments(kwargs)
        if not is_valid:
            raise ToolExecutionError(f"Invalid arguments: {'; '.join(errors)}")
        
        # Apply custom validator if provided
        if self.validator:
            is_valid, custom_errors = self.validator(kwargs)
            if not is_valid:
                raise ToolExecutionError(f"Validation failed: {'; '.join(custom_errors)}")

        # For sync execution, we can't properly handle all rate limit types
        # Only check simple rate limits
        context = {
            "tool_name": self.schema.name,
            "timestamp": datetime.now(),
            "token_count": kwargs.get("_token_count", 0),
        }

        for rate_limit in self.schema.rate_limits:
            if not isinstance(rate_limit, ConcurrentLimit):
                allowed, error = rate_limit.check_allowed(context)
                if not allowed:
                    raise ToolExecutionError(f"Rate limit exceeded: {error}")

        # Record usage
        for rate_limit in self.schema.rate_limits:
            if not isinstance(rate_limit, ConcurrentLimit):
                rate_limit.record_usage(context)

        # Execute using asyncio to run the async execute method
        import asyncio
        return asyncio.run(self.execute(**kwargs))

    async def __call__(self, **kwargs) -> Any:
        """
        Make tool callable directly.

        Args:
            **kwargs: Arguments to pass to the function

        Returns:
            Result from the function execution
        """
        return await self.execute(**kwargs)
    
    async def execute_stream(self, kwargs_dict):
        """
        Execute tool with streaming support.
        
        Args:
            kwargs_dict: Dictionary of arguments
            
        Yields:
            Streamed results from the function
        """
        if not self.schema.supports_streaming:
            raise RuntimeError(f"Tool '{self.schema.name}' does not support streaming")
        
        # Apply preprocessing if provided
        if self.preprocess:
            kwargs_dict = self.preprocess(kwargs_dict)
        
        # Validate arguments
        is_valid, errors = self.schema.validate_arguments(kwargs_dict)
        if not is_valid:
            raise ToolExecutionError(f"Invalid arguments: {'; '.join(errors)}")
        
        # Apply custom validator if provided
        if self.validator:
            is_valid, custom_errors = self.validator(kwargs_dict)
            if not is_valid:
                raise ToolExecutionError(f"Validation failed: {'; '.join(custom_errors)}")
        
        # Execute streaming function
        async for chunk in self._function(**kwargs_dict):
            # Apply postprocessing if provided
            if self.postprocess:
                chunk = self.postprocess(chunk)
            yield chunk

    def get_info(self) -> dict[str, Any]:
        """
        Get tool information including schema and metrics.

        Returns:
            Dictionary with tool information
        """
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "parameters": self.schema.parameters,
            "tags": self.schema.tags,
            "metrics": self.metrics.to_dict(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """
        Get execution metrics for the tool.

        Returns:
            Dictionary of metrics
        """
        avg_time = self._total_execution_time / self._call_count if self._call_count > 0 else 0.0

        return {
            "name": self.schema.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time,
            "last_execution_time": (
                self._last_execution_time.isoformat() if self._last_execution_time else None
            ),
            "error_rate": (self._error_count / self._call_count if self._call_count > 0 else 0.0),
        }

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self.metrics.reset()
        self._call_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = None
        self._error_count = 0

        # Also reset rate limiters
        for rate_limit in self.schema.rate_limits:
            rate_limit.reset()

    def get_rate_limit_status(self) -> list[dict[str, Any]]:
        """
        Get current status of all rate limits.

        Returns:
            List of rate limit descriptions and statuses
        """
        status = []
        for rate_limit in self.schema.rate_limits:
            # Check if limit would allow a call now
            context = {"tool_name": self.schema.name}
            allowed, error = rate_limit.check_allowed(context)

            status.append(
                {
                    "description": rate_limit.describe(),
                    "type": type(rate_limit).__name__,
                    "allowed": allowed,
                    "message": error if not allowed else "OK",
                }
            )

        return status

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.schema.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.schema.description

    @property
    def is_async(self) -> bool:
        """Check if tool is async."""
        return self._is_async

    def __repr__(self) -> str:
        """String representation."""
        return f"Tool(name={self.schema.name}, async={self._is_async}, calls={self._call_count})"
