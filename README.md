# Agent Zoo

A modular Python repository providing building blocks for LLM agent architectures.

## Features

- 🎯 **Modular Design**: Reusable components for building custom agents
- 🔧 **Tool System**: Comprehensive tool management with rate limiting
- 📝 **Type Safe**: Full type hints with modern Python typing
- ⚡ **Async First**: Built for concurrent operations
- 🔄 **Multi-Provider**: Support for OpenAI, Anthropic, and LangChain

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
                enum=["add", "subtract", "multiply", "divide"],
            ),
            "a": ParameterProperty(type="number"),
            "b": ParameterProperty(type="number"),
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

result = registry.execute("calculator", {
    "operation": "add",
    "a": 5,
    "b": 3,
})
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
```

## License

MIT

## Status

This project is under active development. See the repository_plan.md for implementation status and roadmap.