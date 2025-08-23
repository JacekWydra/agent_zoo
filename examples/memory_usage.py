"""
Example demonstrating how the memory system works in practice.

This shows:
1. How is_active works and when items become inactive
2. Token counting and management
3. Sliding window concept
4. Complete usage in an agent context
"""

import asyncio
from datetime import datetime, timedelta

from agent_zoo.core.memory.working import (
    WorkingMemory,
    WorkingMemoryConfig,
    WorkingMemoryItem,
    WorkingMemoryQuery,
    EvictionStrategy,
)
from agent_zoo.interfaces.messages import Message, MessageRole


async def main():
    """Demonstrate memory system usage."""
    
    # ============================================================
    # 1. BASIC SETUP AND TOKEN COUNTING
    # ============================================================
    print("=" * 60)
    print("1. MEMORY SETUP AND TOKEN COUNTING")
    print("=" * 60)
    
    # Create working memory with specific configuration
    config = WorkingMemoryConfig(
        max_tokens=2000,  # Small limit for demonstration
        target_tokens=1500,
        eviction_strategy=EvictionStrategy.HYBRID,
        inactivity_threshold_seconds=60,  # Items inactive after 1 minute
        auto_deactivate=True,
    )
    
    memory = WorkingMemory(config=config)
    await memory.initialize()
    
    print(f"Memory initialized with max_tokens={config.max_tokens}")
    print()
    
    # ============================================================
    # 2. ADDING MEMORIES AND TOKEN MANAGEMENT
    # ============================================================
    print("=" * 60)
    print("2. ADDING MEMORIES - TOKEN COUNTING IN ACTION")
    print("=" * 60)
    
    # Simulate agent receiving a user query
    user_query = WorkingMemoryItem(
        content="What is the weather like in Paris today?",
        source="user_input",
        priority=10.0,  # User input has high priority
        importance=8.0,
        # Token count estimated automatically if not provided
    )
    
    await memory.store(user_query)
    print(f"Stored user query: {user_query.content[:50]}")
    print(f"  Token count: {user_query.token_count} (auto-estimated)")
    print(f"  Is active: {user_query.is_active}")
    print(f"  Total tokens in memory: {memory.total_tokens}")
    print()
    
    # Agent fetches weather data
    weather_data = WorkingMemoryItem(
        content={
            "city": "Paris",
            "temperature": 22,
            "conditions": "Partly cloudy",
            "humidity": 65,
            "forecast": "Chance of rain in the evening"
        },
        source="weather_api",
        priority=7.0,
        related_items=[user_query.id],  # Link to user query
    )
    
    await memory.store(weather_data)
    print(f"Stored weather data for Paris")
    print(f"  Token count: {weather_data.token_count}")
    print(f"  Total tokens in memory: {memory.total_tokens}")
    print()
    
    # ============================================================
    # 3. THE is_active FIELD - LIFECYCLE DEMONSTRATION
    # ============================================================
    print("=" * 60)
    print("3. UNDERSTANDING is_active - MEMORY LIFECYCLE")
    print("=" * 60)
    
    # Add more context as the agent works
    historical_weather = WorkingMemoryItem(
        content="Yesterday in Paris: 20°C, sunny",
        source="weather_history",
        priority=3.0,  # Lower priority - background info
        importance=4.0,
    )
    await memory.store(historical_weather)
    
    print("All memories are initially active:")
    for item_id, item in memory.items.items():
        print(f"  {item.source}: is_active={item.is_active}")
    print()
    
    # Simulate time passing - some memories become less relevant
    print("After completing weather query, marking historical data as inactive...")
    historical_weather.deactivate()
    
    print(f"Historical weather after deactivation:")
    print(f"  is_active: {historical_weather.is_active}")
    print(f"  relevance_score: {historical_weather.relevance_score:.2f} (reduced)")
    print(f"  deactivated_at: {historical_weather.deactivated_at}")
    print()
    
    # But it can be reactivated if needed again
    print("User asks about weather trends - reactivating historical data...")
    historical_weather.reactivate()
    print(f"After reactivation:")
    print(f"  is_active: {historical_weather.is_active}")
    print(f"  relevance_score: {historical_weather.relevance_score:.2f} (increased)")
    print(f"  activation_count: {historical_weather.activation_count}")
    print()
    
    # ============================================================
    # 4. SLIDING WINDOW CONCEPT
    # ============================================================
    print("=" * 60)
    print("4. SLIDING WINDOW - MANAGING CONTEXT")
    print("=" * 60)
    
    print("Adding more memories to demonstrate sliding window...")
    
    # Simulate a conversation that grows
    messages = [
        "User: What about London?",
        "Assistant: Let me check London weather...",
        "Tool: London - 18°C, rainy",
        "Assistant: London is 18°C and rainy today.",
        "User: Compare with New York",
        "Tool: New York - 25°C, sunny",
        "Assistant: New York is warmer at 25°C and sunny.",
        "User: Which city has the best weather?",
    ]
    
    for i, msg_content in enumerate(messages):
        msg_memory = WorkingMemoryItem(
            content=msg_content,
            source="conversation",
            priority=5.0,
            importance=5.0 - (i * 0.5),  # Older messages less important
        )
        await memory.store(msg_memory)
        print(f"  Added: {msg_content[:30]}... (tokens: {msg_memory.token_count})")
    
    print(f"\nTotal tokens now: {memory.total_tokens}/{memory.config.max_tokens}")
    
    # ============================================================
    # 5. AUTOMATIC EVICTION WHEN TOKEN LIMIT EXCEEDED
    # ============================================================
    print("\n" + "=" * 60)
    print("5. AUTOMATIC EVICTION - TOKEN LIMIT MANAGEMENT")
    print("=" * 60)
    
    # Add a large memory that triggers eviction
    large_content = "This is a very detailed weather report. " * 50
    large_memory = WorkingMemoryItem(
        content=large_content,
        source="detailed_report",
        priority=6.0,
        token_count=500,  # Explicitly set for demonstration
    )
    
    print(f"Adding large memory ({large_memory.token_count} tokens)...")
    print(f"This will exceed our limit of {memory.config.max_tokens} tokens")
    
    initial_count = len(memory.items)
    await memory.store(large_memory)
    final_count = len(memory.items)
    
    print(f"\nEviction occurred:")
    print(f"  Items before: {initial_count}")
    print(f"  Items after: {final_count}")
    print(f"  Items evicted: {initial_count - final_count + 1}")  # +1 for the new item
    print(f"  Total tokens: {memory.total_tokens} (target: {memory.config.target_tokens})")
    print()
    
    # ============================================================
    # 6. QUERYING MEMORY - RETRIEVAL PATTERNS
    # ============================================================
    print("=" * 60)
    print("6. MEMORY RETRIEVAL - DIFFERENT QUERY PATTERNS")
    print("=" * 60)
    
    # Query 1: Get only active memories
    print("Query 1: Active memories only")
    query = WorkingMemoryQuery(
        only_active=True,
        limit=5
    )
    result = await memory.retrieve(query)
    print(f"  Found {len(result.items)} active items (total: {result.total_count})")
    for item in result.items[:3]:
        print(f"    - {item.source}: {str(item.content)[:40]}...")
    print()
    
    # Query 2: Get memories within token budget
    print("Query 2: Memories within 500 token budget")
    query = WorkingMemoryQuery(
        max_tokens=500,
        only_active=True
    )
    result = await memory.retrieve(query)
    total_tokens = sum(item.token_count for item in result.items)
    print(f"  Found {len(result.items)} items using {total_tokens} tokens")
    print()
    
    # Query 3: Get high-relevance memories
    print("Query 3: High relevance memories (>0.7)")
    query = WorkingMemoryQuery(
        min_relevance=0.7,
        only_active=False  # Check all items, not just active
    )
    result = await memory.retrieve(query)
    print(f"  Found {len(result.items)} high-relevance items")
    for item in result.items:
        print(f"    - {item.source}: relevance={item.relevance_score:.2f}")
    print()
    
    # ============================================================
    # 7. SLIDING WINDOW IN ACTION - CONTEXT MANAGEMENT
    # ============================================================
    print("=" * 60)
    print("7. SLIDING WINDOW PATTERN - BUILDING CONTEXT")
    print("=" * 60)
    
    # This is how an agent would build context for an LLM call
    context_budget = 1000  # tokens available for context
    
    # Get active context within budget
    active_context = await memory.get_active_context(max_tokens=context_budget)
    
    print(f"Building context for LLM (budget: {context_budget} tokens):")
    print(f"Selected {len(active_context)} memories:")
    
    context_tokens = 0
    for item in active_context:
        print(f"  [{item.source}] {str(item.content)[:40]}... ({item.token_count} tokens)")
        context_tokens += item.token_count
    
    print(f"\nTotal context size: {context_tokens} tokens")
    print("This context would be formatted and sent to the LLM")
    print()
    
    # ============================================================
    # 8. MEMORY CONSOLIDATION
    # ============================================================
    print("=" * 60)
    print("8. MEMORY CONSOLIDATION - CLEANUP")
    print("=" * 60)
    
    print("Running consolidation...")
    stats = await memory.consolidate()
    
    print("Consolidation results:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # ============================================================
    # 9. PRACTICAL AGENT USAGE EXAMPLE
    # ============================================================
    print("=" * 60)
    print("9. COMPLETE AGENT WORKFLOW EXAMPLE")
    print("=" * 60)
    
    async def agent_task_with_memory():
        """Simulate an agent performing a task with memory."""
        
        # Fresh memory for this example
        agent_memory = WorkingMemory(
            WorkingMemoryConfig(
                max_tokens=4000,
                eviction_strategy=EvictionStrategy.HYBRID,
            )
        )
        await agent_memory.initialize()
        
        print("Agent receives task: 'Summarize news about AI'")
        print()
        
        # Step 1: Store the task
        task = WorkingMemoryItem(
            content="Summarize recent news about AI developments",
            source="user_task",
            priority=10.0,
            importance=10.0,
        )
        await agent_memory.store(task)
        print("✓ Task stored in working memory")
        
        # Step 2: Agent searches for news (simulated)
        news_items = [
            "OpenAI announces GPT-5 with improved reasoning",
            "Google's Gemini shows multimodal capabilities",
            "Meta releases open-source Llama 3",
            "Anthropic introduces Claude 3 with better safety",
        ]
        
        for news in news_items:
            news_memory = WorkingMemoryItem(
                content=news,
                source="news_search",
                priority=7.0,
                related_items=[task.id],
            )
            await agent_memory.store(news_memory)
        print(f"✓ Stored {len(news_items)} news items")
        
        # Step 3: Agent creates summary
        summary = WorkingMemoryItem(
            content="Recent AI developments include major releases from OpenAI (GPT-5), "
                   "Google (Gemini), Meta (Llama 3), and Anthropic (Claude 3), "
                   "focusing on improved reasoning, multimodal capabilities, and safety.",
            source="agent_summary",
            priority=9.0,
            related_items=[task.id],
        )
        await agent_memory.store(summary)
        print("✓ Generated and stored summary")
        
        # Step 4: Mark intermediate items as inactive (keep summary active)
        for item_id, item in agent_memory.items.items():
            if item.source == "news_search":
                item.deactivate()  # No longer needed
        print("✓ Deactivated intermediate search results")
        
        # Step 5: Retrieve final context for response
        final_context = WorkingMemoryQuery(
            only_active=True,
            source_filter="agent_summary"
        )
        result = await agent_memory.retrieve(final_context)
        
        print("\nFinal active context for response:")
        for item in result.items:
            print(f"  {item.content}")
        
        print(f"\nMemory stats after task:")
        stats = agent_memory.get_stats()
        print(f"  Active items: {stats['active_items']}")
        print(f"  Inactive items: {stats['inactive_items']}")
        print(f"  Token usage: {stats['token_usage_percent']:.1f}%")
        
        return summary.content
    
    # Run the agent task
    result = await agent_task_with_memory()
    print(f"\nAgent result: {result}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("KEY CONCEPTS DEMONSTRATED:")
    print("=" * 60)
    print("""
1. is_active Field:
   - Memories start active when created
   - Become inactive when no longer needed (manually or auto)
   - Can be reactivated if they become relevant again
   - Inactive memories are prioritized for eviction

2. Token Counting:
   - Each memory tracks its token usage
   - Total tokens tracked across all memories
   - Automatic eviction when limit exceeded
   - Token budget queries for context building

3. Sliding Window:
   - Not a fixed window, but dynamic context management
   - Recent/relevant items stay in memory
   - Old/irrelevant items deactivated or evicted
   - Context built by selecting items within token budget

4. Practical Usage:
   - Store task and intermediate results
   - Link related memories
   - Deactivate temporary information
   - Build context for LLM calls within token limits
   - Automatic cleanup via consolidation
""")


if __name__ == "__main__":
    asyncio.run(main())