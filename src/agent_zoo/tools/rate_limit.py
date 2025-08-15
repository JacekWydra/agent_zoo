"""
Rate limiting strategies for tool execution.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RateLimit(BaseModel, ABC):
    """
    Base class for rate limiting strategies.

    Different rate limiting strategies can be implemented
    by subclassing this base class.
    """

    @abstractmethod
    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Check if an action is allowed under this rate limit.

        Args:
            context: Context information for the check

        Returns:
            Tuple of (is_allowed, error_message)
        """
        pass

    @abstractmethod
    def record_usage(self, context: dict[str, Any]) -> None:
        """
        Record usage for rate limiting.

        Args:
            context: Context information about the usage
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter state."""
        pass

    @abstractmethod
    def describe(self) -> str:
        """Get human-readable description of the limit."""
        pass


class CallRateLimit(RateLimit):
    """
    Rate limit based on number of calls within a time window.

    This is the most common type of rate limiting, restricting
    the number of function calls within a sliding time window.
    """

    max_calls: int = Field(description="Maximum number of calls allowed")
    window_seconds: float = Field(
        default=60.0,
        description="Time window in seconds (1=per second, 60=per minute, 3600=per hour)",
    )

    # Runtime state (not part of schema)
    _call_times: list[datetime] = []

    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if a call is allowed."""
        now = datetime.now()

        # Remove expired calls from the window
        self._call_times = [
            t for t in self._call_times if (now - t).total_seconds() < self.window_seconds
        ]

        # Check if limit is exceeded
        if len(self._call_times) >= self.max_calls:
            if self._call_times:
                oldest_call = self._call_times[0]
                wait_time = self.window_seconds - (now - oldest_call).total_seconds()
                return False, f"Rate limit exceeded. Wait {wait_time:.1f} seconds."
            else:
                return False, "Rate limit exceeded. Wait 60.0 seconds."

        return True, None

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record a call."""
        self._call_times.append(datetime.now())

    def reset(self) -> None:
        """Reset call history."""
        self._call_times = []

    def describe(self) -> str:
        """Get human-readable description."""
        if self.window_seconds == 1:
            return f"{self.max_calls} calls per second"
        elif self.window_seconds == 60:
            return f"{self.max_calls} calls per minute"
        elif self.window_seconds == 3600:
            return f"{self.max_calls} calls per hour"
        elif self.window_seconds == 86400:
            return f"{self.max_calls} calls per day"
        else:
            return f"{self.max_calls} calls per {self.window_seconds} seconds"

    class Config:
        # Allow mutable default for _call_times
        arbitrary_types_allowed = True


class TokenRateLimit(RateLimit):
    """
    Rate limit based on token usage within a time window.

    Useful for LLM-based tools where token consumption
    needs to be controlled.
    """

    max_tokens: int = Field(description="Maximum number of tokens allowed")
    window_seconds: float = Field(default=60.0, description="Time window in seconds")

    # Runtime state
    _token_usage: list[tuple[datetime, int]] = []

    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if token usage is allowed."""
        now = datetime.now()
        requested_tokens = context.get("token_count", 0)

        # Remove expired usage from the window
        self._token_usage = [
            (t, count)
            for t, count in self._token_usage
            if (now - t).total_seconds() < self.window_seconds
        ]

        # Calculate current usage
        current_usage = sum(count for _, count in self._token_usage)

        # Check if adding these tokens would exceed limit
        if current_usage + requested_tokens > self.max_tokens:
            available = self.max_tokens - current_usage
            return False, f"Token limit exceeded. Only {available} tokens available."

        return True, None

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record token usage."""
        token_count = context.get("token_count", 0)
        if token_count > 0:
            self._token_usage.append((datetime.now(), token_count))

    def reset(self) -> None:
        """Reset token usage history."""
        self._token_usage = []

    def describe(self) -> str:
        """Get human-readable description."""
        if self.window_seconds == 60:
            return f"{self.max_tokens} tokens per minute"
        elif self.window_seconds == 3600:
            return f"{self.max_tokens} tokens per hour"
        else:
            return f"{self.max_tokens} tokens per {self.window_seconds} seconds"

    class Config:
        arbitrary_types_allowed = True


class ConcurrentLimit(RateLimit):
    """
    Rate limit based on number of concurrent executions.

    Limits how many instances of the tool can be running
    simultaneously.
    """

    max_concurrent: int = Field(description="Maximum concurrent executions")

    # Runtime state
    _active_count: int = 0
    _active_ids: set[str] = set()

    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if concurrent execution is allowed."""
        if self._active_count >= self.max_concurrent:
            return False, f"Maximum {self.max_concurrent} concurrent executions allowed."
        return True, None

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record start of execution."""
        execution_id = context.get("execution_id", "")
        if execution_id and execution_id not in self._active_ids:
            self._active_ids.add(execution_id)
            self._active_count += 1

    def release(self, execution_id: str) -> None:
        """
        Release a concurrent execution slot.

        Args:
            execution_id: ID of the execution to release
        """
        if execution_id in self._active_ids:
            self._active_ids.remove(execution_id)
            self._active_count -= 1

    def reset(self) -> None:
        """Reset concurrent execution tracking."""
        self._active_count = 0
        self._active_ids = set()

    def describe(self) -> str:
        """Get human-readable description."""
        return f"Maximum {self.max_concurrent} concurrent executions"

    class Config:
        arbitrary_types_allowed = True


class CostLimit(RateLimit):
    """
    Rate limit based on monetary cost within a time window.

    Useful for expensive API calls where cost needs to be controlled.
    """

    max_cost: float = Field(description="Maximum cost allowed (in currency units)")
    window_seconds: float = Field(
        default=3600.0, description="Time window in seconds (default 1 hour)"
    )
    cost_per_call: float = Field(default=0.0, description="Fixed cost per call")
    cost_per_token: float = Field(default=0.0, description="Cost per token (for LLM tools)")

    # Runtime state
    _cost_usage: list[tuple[datetime, float]] = []

    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if cost limit allows execution."""
        now = datetime.now()

        # Calculate cost for this call
        call_cost = self.cost_per_call
        if self.cost_per_token > 0:
            token_count = context.get("token_count", 0)
            call_cost += token_count * self.cost_per_token

        # Remove expired costs from the window
        self._cost_usage = [
            (t, cost)
            for t, cost in self._cost_usage
            if (now - t).total_seconds() < self.window_seconds
        ]

        # Calculate current cost
        current_cost = sum(cost for _, cost in self._cost_usage)

        # Check if adding this cost would exceed limit
        if current_cost + call_cost > self.max_cost:
            available = self.max_cost - current_cost
            return False, f"Cost limit exceeded. ${available:.4f} remaining."

        return True, None

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record cost usage."""
        call_cost = self.cost_per_call
        if self.cost_per_token > 0:
            token_count = context.get("token_count", 0)
            call_cost += token_count * self.cost_per_token

        if call_cost > 0:
            self._cost_usage.append((datetime.now(), call_cost))

    def reset(self) -> None:
        """Reset cost usage history."""
        self._cost_usage = []

    def describe(self) -> str:
        """Get human-readable description."""
        time_desc = "hour" if self.window_seconds == 3600 else f"{self.window_seconds} seconds"
        return f"Maximum ${self.max_cost:.2f} per {time_desc}"

    class Config:
        arbitrary_types_allowed = True


class BurstRateLimit(RateLimit):
    """
    Token bucket rate limit allowing burst capacity.

    Allows temporary bursts above the steady rate, useful for
    handling traffic spikes while maintaining long-term limits.
    """

    capacity: int = Field(description="Maximum tokens in bucket (burst capacity)")
    refill_rate: float = Field(description="Tokens refilled per second")
    tokens_per_call: int = Field(default=1, description="Tokens consumed per call")

    # Runtime state
    _current_tokens: float = 0.0
    _last_refill: datetime | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._current_tokens = float(self.capacity)
        self._last_refill = datetime.now()

    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if call is allowed."""
        self._refill_tokens()

        tokens_needed = context.get("tokens_needed", self.tokens_per_call)
        if self._current_tokens >= tokens_needed:
            return True, None

        # Calculate wait time
        tokens_short = tokens_needed - self._current_tokens
        wait_time = tokens_short / self.refill_rate
        return False, f"Rate limit exceeded. Wait {wait_time:.1f} seconds."

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record token consumption."""
        tokens_used = context.get("tokens_needed", self.tokens_per_call)
        self._current_tokens = max(0, self._current_tokens - tokens_used)

    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed."""
        now = datetime.now()
        if self._last_refill:
            elapsed = (now - self._last_refill).total_seconds()
            tokens_to_add = elapsed * self.refill_rate
            self._current_tokens = min(self.capacity, self._current_tokens + tokens_to_add)
        self._last_refill = now

    def reset(self) -> None:
        """Reset to full capacity."""
        self._current_tokens = float(self.capacity)
        self._last_refill = datetime.now()

    def describe(self) -> str:
        """Get human-readable description."""
        return f"Burst capacity: {self.capacity}, refill rate: {self.refill_rate}/second"

    class Config:
        arbitrary_types_allowed = True


class CompositeRateLimit(RateLimit):
    """
    Composite rate limit that combines multiple limits.

    All limits must pass for the action to be allowed.
    """

    limits: list[RateLimit] = Field(
        default_factory=list, description="List of rate limits to enforce"
    )

    def check_allowed(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if all limits allow the action."""
        for limit in self.limits:
            allowed, error = limit.check_allowed(context)
            if not allowed:
                return False, error
        return True, None

    def record_usage(self, context: dict[str, Any]) -> None:
        """Record usage in all limits."""
        for limit in self.limits:
            limit.record_usage(context)

    def reset(self) -> None:
        """Reset all limits."""
        for limit in self.limits:
            limit.reset()

    def describe(self) -> str:
        """Get human-readable description."""
        descriptions = [limit.describe() for limit in self.limits]
        return " AND ".join(descriptions)

    class Config:
        arbitrary_types_allowed = True
