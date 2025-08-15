# Agent Zoo

A modular Python repository providing building blocks for LLM agent architectures.

## Purpose

This repository was created with two primary goals:

### 1. Learning LLM Agent Architectures
To gain hands-on experience and deep understanding of:
- How LLM agents are structured and organized
- The key components that make up agent systems (tools, state management, message passing)
- How different architectural patterns interact and work together
- Real-world implementation challenges and solutions

### 2. Mastering Vibe-Coding with Claude Code
To explore and find the optimal balance in AI-assisted development:
- Learning when to give Claude Code creative freedom vs. providing specific guidance
- Understanding the rhythm of effective human-AI collaboration
- Discovering the sweet spot between automation and control
- Building intuition for productive pair programming with AI

## What's Inside

Agent Zoo implements various agent design patterns from simple reactive agents to complex multi-agent systems. The framework focuses on reusability, type safety, and production readiness.

### Features

- 🎯 **Modular Design**: Reusable components for building custom agents
- 🔧 **Tool System**: Comprehensive tool management with rate limiting
- 📝 **Type Safe**: Full type hints with modern Python typing
- ⚡ **Async First**: Built for concurrent operations
- 🔄 **Multi-Provider**: Support for OpenAI, Anthropic, and LangChain
- ✅ **100% Test Coverage**: Complete test suite demonstrating proper implementation

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from agent_zoo.tools.schema import ToolSchema, ParameterSchema, ParameterProperty
from agent_zoo.tools.tool import Tool
from agent_zoo.tools.registry import ToolRegistry

# Define a tool schema
schema = ToolSchema(
    name="calculator",
    description="Performs calculations",
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
)

# Create tool with implementation
def calculate(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    else:
        return a / b

tool = Tool(schema=schema, function=calculate)

# Register and use
registry = ToolRegistry()
registry.register(tool)

# Get tool and execute
calc_tool = registry.get("calculator")
result = calc_tool.execute_sync(operation="add", a=5, b=3)
print(result)  # 8
```

## Components

### Core
- `BaseAgent`: Abstract base class for all agents
- `AgentConfig`: Configuration management
- `AgentState`: State management with history

### Tools
- `ToolSchema`: Define tool interfaces
- `Tool`: Execute functions with validation
- `ToolRegistry`: Manage and discover tools
- Rate limiting strategies (Call, Token, Concurrent, Cost, Burst, Composite)

### Interfaces
- Message protocols (Message, ToolCall, ToolResult)
- State management (StateManager, StateSnapshot)
- Conversation handling

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agent_zoo --cov-report=html

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Project Structure

```
agent_zoo/
├── src/agent_zoo/
│   ├── core/           # Core agent abstractions
│   ├── interfaces/     # Messages and state management
│   └── tools/          # Tool system implementation
└── tests/
    ├── unit/           # Unit tests
    └── integration/    # Integration tests
```

## Learning Journey

This repository represents a learning journey in:
- Understanding agent architectures through implementation
- Developing effective collaboration patterns with AI coding assistants
- Building production-ready code through iterative refinement
- Balancing automated code generation with thoughtful design

## Status

This project is under active development. See the repository_plan.md for implementation status and roadmap.

## License

MIT