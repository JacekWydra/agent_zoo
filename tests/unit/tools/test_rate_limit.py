"""
Unit tests for rate limiting components.
"""

import time

import pytest

from agent_zoo.tools.rate_limit import (
    BurstRateLimit,
    CallRateLimit,
    CompositeRateLimit,
    ConcurrentLimit,
    CostLimit,
    TokenRateLimit,
)


class TestCallRateLimit:
    """Tests for CallRateLimit."""

    def test_initialization(self):
        """Test rate limit initialization."""
        limit = CallRateLimit(max_calls=10, window_seconds=60)

        assert limit.max_calls == 10
        assert limit.window_seconds == 60
        assert len(limit._call_times) == 0

    def test_allow_calls_within_limit(self):
        """Test allowing calls within the limit."""
        limit = CallRateLimit(max_calls=3, window_seconds=60)

        # First three calls should be allowed
        for i in range(3):
            allowed, message = limit.check_allowed({})
            assert allowed is True
            assert message is None
            limit.record_usage({})

    def test_block_calls_exceeding_limit(self):
        """Test blocking calls that exceed the limit."""
        limit = CallRateLimit(max_calls=2, window_seconds=60)

        # First two calls allowed
        allowed1, _ = limit.check_allowed({})
        assert allowed1 is True
        limit.record_usage({})

        allowed2, _ = limit.check_allowed({})
        assert allowed2 is True
        limit.record_usage({})

        # Third call should be blocked
        allowed, message = limit.check_allowed({})
        assert allowed is False
        assert "Rate limit exceeded" in message

    def test_window_sliding(self):
        """Test that old calls are removed from the window."""
        limit = CallRateLimit(max_calls=2, window_seconds=0.1)

        # Make two calls
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Should be blocked
        allowed, _ = limit.check_allowed({})
        assert allowed is False

        # Wait for window to slide
        time.sleep(0.15)

        # Should be allowed again
        allowed, _ = limit.check_allowed({})
        assert allowed is True

    def test_reset(self):
        """Test resetting the rate limit."""
        limit = CallRateLimit(max_calls=2, window_seconds=60)

        # Make calls
        limit.check_allowed({})
        limit.record_usage({})
        limit.check_allowed({})
        limit.record_usage({})

        assert len(limit._call_times) == 2

        # Reset
        limit.reset()
        assert len(limit._call_times) == 0

        # Can make calls again
        allowed, _ = limit.check_allowed({})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_async_check_allowed(self):
        """Test async version of check_allowed."""
        limit = CallRateLimit(max_calls=2, window_seconds=60)

        # Sync calls (async not implemented for RateLimit base)
        allowed1, _ = limit.check_allowed({})
        assert allowed1 is True
        limit.record_usage({})

        allowed2, _ = limit.check_allowed({})
        assert allowed2 is True
        limit.record_usage({})

        allowed3, _ = limit.check_allowed({})
        assert allowed3 is False


class TestTokenRateLimit:
    """Tests for TokenRateLimit."""

    def test_initialization(self):
        """Test token rate limit initialization."""
        limit = TokenRateLimit(max_tokens=1000, window_seconds=60)

        assert limit.max_tokens == 1000
        assert limit.window_seconds == 60
        assert len(limit._token_usage) == 0

    def test_token_tracking(self):
        """Test tracking token usage."""
        limit = TokenRateLimit(max_tokens=100, window_seconds=60)

        # Use some tokens
        allowed, message = limit.check_allowed({"token_count": 30})
        assert allowed is True
        limit.record_usage({"token_count": 30})

        allowed, message = limit.check_allowed({"token_count": 50})
        assert allowed is True
        limit.record_usage({"token_count": 50})

        # Should exceed limit
        allowed, message = limit.check_allowed({"token_count": 30})
        assert allowed is False
        assert "Token limit exceeded" in message

    def test_token_window_sliding(self):
        """Test that old token usage is removed."""
        limit = TokenRateLimit(max_tokens=100, window_seconds=0.1)

        # Use tokens
        limit.check_allowed({"token_count": 80})
        limit.record_usage({"token_count": 80})

        # Should be blocked for more tokens
        allowed, _ = limit.check_allowed({"token_count": 30})
        assert allowed is False

        # Wait for window to slide
        time.sleep(0.15)

        # Should be allowed again
        allowed, _ = limit.check_allowed({"token_count": 30})
        assert allowed is True

    def test_default_token_count(self):
        """Test default token count when not specified."""
        limit = TokenRateLimit(max_tokens=100, window_seconds=60)

        # Should use default of 0 tokens
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Check total usage
        total = sum(usage[1] for usage in limit._token_usage)
        assert total == 0


class TestConcurrentLimit:
    """Tests for ConcurrentLimit."""

    def test_initialization(self):
        """Test concurrent limit initialization."""
        limit = ConcurrentLimit(max_concurrent=3)

        assert limit.max_concurrent == 3
        assert limit._active_count == 0

    def test_acquire_and_release(self):
        """Test acquiring and releasing slots."""
        limit = ConcurrentLimit(max_concurrent=2)

        # Acquire first slot
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({"execution_id": "exec1"})
        assert limit._active_count == 1

        # Acquire second slot
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({"execution_id": "exec2"})
        assert limit._active_count == 2

        # Third should be blocked
        allowed, message = limit.check_allowed({})
        assert allowed is False
        assert "Maximum 2 concurrent executions allowed" in message

        # Release one slot
        limit.release("exec1")
        assert limit._active_count == 1

        # Can acquire again
        allowed, _ = limit.check_allowed({})
        assert allowed is True

    def test_release_when_empty(self):
        """Test releasing when no slots are in use."""
        limit = ConcurrentLimit(max_concurrent=2)

        # Release without acquiring
        limit.release("nonexistent")
        assert limit._active_count == 0  # Should not go negative

    def test_reset(self):
        """Test resetting concurrent limit."""
        limit = ConcurrentLimit(max_concurrent=2)

        # Acquire slots
        limit.check_allowed({})
        limit.record_usage({"execution_id": "exec1"})
        limit.check_allowed({})
        limit.record_usage({"execution_id": "exec2"})
        assert limit._active_count == 2

        # Reset
        limit.reset()
        assert limit._active_count == 0

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async acquire and release."""
        limit = ConcurrentLimit(max_concurrent=2)

        # Sync operations (async not implemented)
        allowed1, _ = limit.check_allowed({})
        assert allowed1 is True
        limit.record_usage({"execution_id": "exec1"})

        allowed2, _ = limit.check_allowed({})
        assert allowed2 is True
        limit.record_usage({"execution_id": "exec2"})

        allowed3, _ = limit.check_allowed({})
        assert allowed3 is False

        # Release
        limit.release("exec1")
        assert limit._active_count == 1


class TestCostLimit:
    """Tests for CostLimit."""

    def test_initialization(self):
        """Test cost limit initialization."""
        limit = CostLimit(max_cost=10.0, window_seconds=60, cost_per_call=0.5)

        assert limit.max_cost == 10.0
        assert limit.window_seconds == 60
        assert limit.cost_per_call == 0.5

    def test_cost_tracking(self):
        """Test tracking costs."""
        limit = CostLimit(max_cost=5.0, window_seconds=60, cost_per_call=1.5)

        # First call: $1.50
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Second call: $3.00 total
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Third call would be $4.50 total
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Fourth call would exceed $5.00
        allowed, message = limit.check_allowed({})
        assert allowed is False
        assert "Cost limit exceeded" in message

    def test_custom_cost_per_operation(self):
        """Test custom cost per operation."""
        limit = CostLimit(max_cost=10.0, window_seconds=60, cost_per_call=3.0)

        # First call: $3
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Second call: $6 total
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Third call: $9 total
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Fourth call would exceed $10
        allowed, _ = limit.check_allowed({})
        assert allowed is False

    def test_cost_window_sliding(self):
        """Test that old costs are removed."""
        limit = CostLimit(max_cost=5.0, window_seconds=0.1, cost_per_call=2.0)

        # Use up most of the budget
        limit.check_allowed({})  # $2
        limit.record_usage({})
        limit.check_allowed({})  # $4
        limit.record_usage({})

        # Should be blocked
        allowed, _ = limit.check_allowed({})
        assert allowed is False

        # Wait for window to slide
        time.sleep(0.15)

        # Should be allowed again
        allowed, _ = limit.check_allowed({})
        assert allowed is True


class TestBurstRateLimit:
    """Tests for BurstRateLimit."""

    def test_initialization(self):
        """Test burst rate limit initialization."""
        limit = BurstRateLimit(capacity=10, refill_rate=2.0, tokens_per_call=1)

        assert limit.capacity == 10
        assert limit.refill_rate == 2.0
        assert limit.tokens_per_call == 1
        assert limit._current_tokens == 10

    def test_token_consumption(self):
        """Test consuming burst tokens."""
        limit = BurstRateLimit(capacity=5, refill_rate=1.0)

        # Consume tokens
        for i in range(5):
            allowed, _ = limit.check_allowed({})
            assert allowed is True
            limit.record_usage({})

        # Should be out of tokens
        allowed, message = limit.check_allowed({})
        assert allowed is False
        assert "Rate limit exceeded" in message

    def test_token_refill(self):
        """Test token refill over time."""
        limit = BurstRateLimit(
            capacity=5,
            refill_rate=10.0,  # 10 tokens per second
        )

        # Consume all tokens
        for _ in range(5):
            limit.check_allowed({})
            limit.record_usage({})

        # Should be blocked
        allowed, _ = limit.check_allowed({})
        assert allowed is False

        # Wait for refill (0.1 seconds = 1 token)
        time.sleep(0.12)

        # Should have 1+ tokens now
        allowed, _ = limit.check_allowed({})
        assert allowed is True
        limit.record_usage({})

        # Should be blocked again
        allowed, _ = limit.check_allowed({})
        assert allowed is False

    def test_partial_refill(self):
        """Test partial token refill."""
        limit = BurstRateLimit(capacity=10, refill_rate=5.0)

        # Consume some tokens
        for _ in range(7):
            limit.check_allowed({})
            limit.record_usage({})

        assert 2.9 < limit._current_tokens < 3.1  # Approximately 3

        # Wait for partial refill
        time.sleep(1.1)

        # Should have refilled 5+ tokens
        limit._refill_tokens()
        assert limit._current_tokens >= 8  # 3 + 5+

    def test_max_burst_cap(self):
        """Test that tokens don't exceed burst size."""
        limit = BurstRateLimit(capacity=10, refill_rate=20.0)

        # Wait for potential overfill
        time.sleep(2)
        limit._refill_tokens()

        # Should still be capped at capacity
        assert limit._current_tokens <= 10

    def test_reset(self):
        """Test resetting burst limit."""
        limit = BurstRateLimit(capacity=5, refill_rate=1.0)

        # Consume tokens
        for _ in range(3):
            limit.check_allowed({})
            limit.record_usage({})

        assert 1.9 < limit._current_tokens < 2.1  # Approximately 2

        # Reset
        limit.reset()
        assert limit._current_tokens == 5  # Back to capacity


class TestCompositeRateLimit:
    """Tests for CompositeRateLimit."""

    def test_initialization(self):
        """Test composite rate limit initialization."""
        call_limit = CallRateLimit(max_calls=10, window_seconds=60)
        token_limit = TokenRateLimit(max_tokens=1000, window_seconds=60)

        composite = CompositeRateLimit(limits=[call_limit, token_limit])

        assert len(composite.limits) == 2
        assert call_limit in composite.limits
        assert token_limit in composite.limits

    def test_all_limits_must_pass(self):
        """Test that all limits must pass for composite to pass."""
        call_limit = CallRateLimit(max_calls=2, window_seconds=60)
        token_limit = TokenRateLimit(max_tokens=100, window_seconds=60)

        composite = CompositeRateLimit(limits=[call_limit, token_limit])

        # First call passes both
        allowed, _ = composite.check_allowed({"token_count": 30})
        assert allowed is True
        composite.record_usage({"token_count": 30})

        # Second call passes both
        allowed, _ = composite.check_allowed({"token_count": 30})
        assert allowed is True
        composite.record_usage({"token_count": 30})

        # Third call fails call limit (even though tokens are OK)
        allowed, message = composite.check_allowed({"token_count": 30})
        assert allowed is False
        assert "Rate limit exceeded" in message

    def test_first_failure_stops_checking(self):
        """Test that checking stops at first failure."""
        call_limit = CallRateLimit(max_calls=1, window_seconds=60)
        token_limit = TokenRateLimit(max_tokens=100, window_seconds=60)

        composite = CompositeRateLimit(limits=[call_limit, token_limit])

        # First call OK
        composite.check_allowed({"token_count": 50})
        composite.record_usage({"token_count": 50})

        # Second call should fail on call limit
        # Token limit shouldn't be updated
        allowed, _ = composite.check_allowed({"token_count": 60})
        assert allowed is False

        # Check that token limit wasn't updated
        # (only 50 tokens should be used, not 110)
        total_tokens = sum(usage[1] for usage in token_limit._token_usage)
        assert total_tokens == 50

    def test_multiple_failure_messages(self):
        """Test getting appropriate failure message."""
        call_limit = CallRateLimit(max_calls=0, window_seconds=60)  # Always fails
        token_limit = TokenRateLimit(max_tokens=0, window_seconds=60)  # Always fails

        composite = CompositeRateLimit(limits=[call_limit, token_limit])

        # Should get the first failure message
        allowed, message = composite.check_allowed({"token_count": 10})
        assert allowed is False
        assert "Rate limit exceeded" in message  # First limit's message

    def test_reset_all_limits(self):
        """Test resetting all limits in composite."""
        call_limit = CallRateLimit(max_calls=2, window_seconds=60)
        concurrent_limit = ConcurrentLimit(max_concurrent=1)

        composite = CompositeRateLimit(limits=[call_limit, concurrent_limit])

        # Use up limits
        composite.check_allowed({})
        composite.record_usage({"execution_id": "exec1"})

        # Reset all
        composite.reset()

        # Check individual limits are reset
        assert len(call_limit._call_times) == 0
        assert concurrent_limit._active_count == 0

    def test_empty_composite(self):
        """Test composite with no limits."""
        composite = CompositeRateLimit(limits=[])

        # Should always allow if no limits
        allowed, message = composite.check_allowed({})
        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_async_composite(self):
        """Test async operations on composite limit."""
        call_limit = CallRateLimit(max_calls=2, window_seconds=60)
        token_limit = TokenRateLimit(max_tokens=100, window_seconds=60)

        composite = CompositeRateLimit(limits=[call_limit, token_limit])

        # Sync checks (async not implemented)
        allowed1, _ = composite.check_allowed({"token_count": 30})
        assert allowed1 is True
        composite.record_usage({"token_count": 30})

        allowed2, _ = composite.check_allowed({"token_count": 30})
        assert allowed2 is True
        composite.record_usage({"token_count": 30})

        allowed3, _ = composite.check_allowed({"token_count": 30})
        assert allowed3 is False  # Exceeds call limit

    def test_mixed_limit_types(self):
        """Test composite with different limit types."""
        call_limit = CallRateLimit(max_calls=10, window_seconds=60)
        concurrent_limit = ConcurrentLimit(max_concurrent=2)
        cost_limit = CostLimit(max_cost=10.0, window_seconds=60, cost_per_call=3.0)
        burst_limit = BurstRateLimit(capacity=5, refill_rate=1.0)

        composite = CompositeRateLimit(
            limits=[
                call_limit,
                concurrent_limit,
                cost_limit,
                burst_limit,
            ]
        )

        # First two calls should work
        allowed, _ = composite.check_allowed({"execution_id": "exec1"})
        assert allowed is True
        composite.record_usage({"execution_id": "exec1"})

        allowed, _ = composite.check_allowed({"execution_id": "exec2"})
        assert allowed is True
        composite.record_usage({"execution_id": "exec2"})

        # Third should fail on concurrent limit
        allowed, message = composite.check_allowed({"execution_id": "exec3"})
        assert allowed is False
        assert "Maximum 2 concurrent executions allowed" in message

        # Release concurrent slot
        concurrent_limit.release("exec1")

        # Should work again
        allowed, _ = composite.check_allowed({"execution_id": "exec3"})
        assert allowed is True
