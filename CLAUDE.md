# Agent Zoo - Project Guidelines and Status

## Project Overview
Agent Zoo is a modular Python repository providing building blocks for LLM agent architectures. The project implements various agent patterns from simple reactive agents to complex multi-agent systems, with a focus on reusability, type safety, and production readiness.

## Current Implementation Status

### ✅ Completed Components

#### Core Infrastructure
- **Base Agent System** (`src/agent_zoo/core/base.py`)
  - `BaseAgent` abstract class with generic type support
  - `AgentConfig` for configuration management
  - Full async/sync support
  - State management integration

#### Interfaces (`src/agent_zoo/interfaces/`)
- **State Management** (`state.py`)
  - `AgentState` with comprehensive state fields
  - `StateManager` for state history and snapshots
  - `AgentStatus` enum for tracking agent states
  
- **Message System** (`messages.py`)
  - `Message`, `ToolCall`, `ToolResult` classes
  - `Thought`, `Action`, `Observation` for reasoning traces
  - `Conversation` manager with filtering capabilities

#### Tool System (`src/agent_zoo/tools/`)
- **Schema Definition** (`schema.py`)
  - `ToolSchema` with parameter validation
  - `ParameterSchema` and `ParameterProperty` for structured validation
  - Multi-provider format conversion (OpenAI, Anthropic, LangChain)
  
- **Tool Execution** (`tool.py`)
  - `Tool` class with async/sync execution
  - Metrics tracking (`ToolMetrics`)
  - Preprocessing and postprocessing support
  - Timeout handling
  
- **Rate Limiting** (`rate_limit.py`)
  - 6 rate limiting strategies:
    - `CallRateLimit` - limit calls per time window
    - `TokenRateLimit` - limit tokens consumed
    - `ConcurrentLimit` - limit concurrent executions
    - `CostLimit` - limit monetary cost
    - `BurstRateLimit` - token bucket algorithm
    - `CompositeRateLimit` - combine multiple limits
  
- **Tool Registry** (`registry.py`)
  - Tool registration and discovery
  - Search by name, tags, category
  - Format conversion for all providers
  - Centralized execution with validation

#### Testing Framework (`tests/`)
- **Comprehensive Test Suite**
  - Unit tests for all components
  - Integration tests for tool system
  - Shared fixtures in `conftest.py`
  - Granular folder structure for future expansion

### 🚧 In Progress
- Core utilities (async helpers, monitoring, caching)

### ⏳ Pending
- Memory systems (working, semantic, episodic, procedural)
- Agent implementations (ReAct, CoT, ToT, GoT)
- Multi-agent coordination
- Example implementations
- Documentation

## Coding Rules

### Type Hints
- Use modern typing: `list`, `dict` instead of `List`, `Dict`
- Use `something | None` instead of `Optional[something]`
- Always specify return types, including `-> None` for procedures
- Use generics where appropriate (e.g., `BaseAgent[T]`)

### Code Style
- Follow PEP 8 with 100-character line limit
- Use descriptive variable names
- Prefer composition over inheritance
- Keep functions focused and single-purpose
- Use async/await for I/O operations

### Pydantic Models
- Use Pydantic v2 syntax
- Always provide field descriptions with `Field(description="...")`
- Use `model_dump()` instead of `dict()`
- Implement `model_validate()` for deserialization

### Testing
- Write tests alongside implementation
- Use pytest fixtures for reusable test data
- Test both success and failure cases
- Include integration tests for component interactions
- Aim for >90% code coverage

### Documentation
- Include docstrings for all public methods
- Use Google-style docstrings
- Provide usage examples in docstrings
- Keep comments minimal and meaningful

## Architecture Decisions

### Tool System Design
The tool system was designed with flexibility in mind:
- **Schema-first**: Tool schemas are separate from implementation
- **Multi-provider support**: Easy conversion between different LLM providers
- **Rate limiting**: Comprehensive rate limiting for production use
- **Metrics**: Built-in tracking for debugging and optimization

### State Management
- Centralized state in `AgentState` class
- State snapshots for debugging and rollback
- Clear separation between configuration and runtime state

### Async-First Design
- All I/O operations are async by default
- Sync wrappers provided for compatibility
- Proper timeout and cancellation support

## Development Workflow

### Adding New Components
1. Define interfaces in appropriate module
2. Implement with full type hints
3. Write comprehensive unit tests
4. Add integration tests if needed
5. Update this document

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agent_zoo --cov-report=html

# Run specific test file
pytest tests/unit/tools/test_schema.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Key Design Patterns

### Factory Pattern
Used for creating agents and tools dynamically based on configuration.

### Registry Pattern
Central registries for tools, agents, and other components for easy discovery and management.

### Strategy Pattern
Rate limiting strategies can be swapped and combined without changing client code.

### Observer Pattern
Event-driven communication between components (planned for multi-agent systems).

## Future Enhancements

### Priority 1 (Next Sprint)
- Implement ReAct agent
- Add working memory system
- Create basic examples

### Priority 2
- Chain-of-Thought variants
- Tree of Thoughts implementation
- Semantic memory with vector DB

### Priority 3
- Multi-agent coordination patterns
- Self-improvement mechanisms
- Production deployment guides