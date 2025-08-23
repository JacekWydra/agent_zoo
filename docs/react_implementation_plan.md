# ReAct Implementation Plan for Agent Zoo

## Overview

This document outlines the detailed implementation plan for adding ReAct architecture to Agent Zoo. The implementation will leverage existing infrastructure (tools, memory, messages, state) while adding ReAct-specific components.

## Implementation Phases

### Phase 1: Core ReAct Components (Week 1)

#### 1.1 Base ReAct Structure
**File**: `src/agent_zoo/agents/react/base.py`

```python
class ReActConfig(AgentConfig):
    """Configuration specific to ReAct agents."""
    max_thought_length: int = 500
    max_action_retries: int = 3
    enable_self_reflection: bool = False
    trajectory_format: str = "structured"  # or "natural"
    
class ReActAgent(BaseAgent[Message]):
    """Base ReAct agent implementation."""
    async def _process(self, message: Message, context: list[Any]) -> Message
    async def think(self, context: dict[str, Any]) -> dict[str, Any]
    async def act(self, thought: dict[str, Any]) -> dict[str, Any]
    async def observe(self, action: dict[str, Any]) -> dict[str, Any]
```

#### 1.2 Trajectory Management
**File**: `src/agent_zoo/agents/react/trajectory.py`

```python
class TrajectoryItem(BaseModel):
    """Single item in ReAct trajectory."""
    type: Literal["thought", "action", "observation"]
    content: Any
    timestamp: datetime
    metadata: dict[str, Any]

class Trajectory(BaseModel):
    """Manages ReAct execution trajectory."""
    items: list[TrajectoryItem]
    task: str
    status: str
    
    def add_thought(self, thought: Thought) -> None
    def add_action(self, action: Action) -> None
    def add_observation(self, observation: Observation) -> None
    def get_formatted(self, format: str = "structured") -> str
    def get_recent(self, n: int = 5) -> list[TrajectoryItem]
```

#### 1.3 Prompt Management
**File**: `src/agent_zoo/agents/react/prompts.py`

```python
class ReActPromptBuilder:
    """Builds prompts for ReAct agent."""
    
    def build_initial_prompt(self, task: str, tools: list[Tool]) -> str
    def build_thought_prompt(self, trajectory: Trajectory) -> str
    def build_action_prompt(self, thought: str, available_tools: list[str]) -> str
    def build_reflection_prompt(self, trajectory: Trajectory) -> str
```

### Phase 2: Tool Integration (Week 1-2)

#### 2.1 Tool Adapter
**File**: `src/agent_zoo/agents/react/tool_adapter.py`

```python
class ReActToolAdapter:
    """Adapts Agent Zoo tools for ReAct usage."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    async def execute_action(self, action_str: str) -> ToolResult
    def parse_action_string(self, action_str: str) -> tuple[str, dict]
    def format_tool_description(self, tool: Tool) -> str
    def get_tool_usage_examples(self, tool: Tool) -> list[str]
```

#### 2.2 Action Parser
**File**: `src/agent_zoo/agents/react/parser.py`

```python
class ActionParser:
    """Parses natural language actions into tool calls."""
    
    def parse(self, action_text: str) -> ParsedAction
    def validate_arguments(self, tool: Tool, args: dict) -> tuple[bool, list[str]]
    def suggest_corrections(self, error: str) -> list[str]
    
class ParsedAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    confidence: float
    alternative_interpretations: list[dict]
```

### Phase 3: Memory Integration (Week 2)

#### 3.1 Memory-Augmented ReAct
**File**: `src/agent_zoo/agents/react/memory_react.py`

```python
class MemoryAugmentedReAct(ReActAgent):
    """ReAct agent with memory capabilities."""
    
    def __init__(self, config: ReActConfig, memory: MemoryManager):
        super().__init__(config)
        self.memory = memory
        
    async def think_with_memory(self, context: dict) -> Thought
    async def recall_similar_tasks(self, task: str) -> list[EpisodicMemoryItem]
    async def store_successful_trajectory(self, trajectory: Trajectory) -> None
    async def learn_from_failure(self, trajectory: Trajectory, error: str) -> None
```

#### 3.2 Memory Strategies
**File**: `src/agent_zoo/agents/react/memory_strategies.py`

```python
class ReActMemoryStrategy:
    """Strategies for memory usage in ReAct."""
    
    async def select_relevant_memories(self, task: str, memory: MemoryManager) -> list
    async def consolidate_trajectory(self, trajectory: Trajectory) -> ProceduralMemoryItem
    async def extract_patterns(self, trajectories: list[Trajectory]) -> list[dict]
```

### Phase 4: Advanced Features (Week 2-3)

#### 4.1 Self-Reflection
**File**: `src/agent_zoo/agents/react/reflection.py`

```python
class ReflectiveReAct(ReActAgent):
    """ReAct with self-reflection capabilities."""
    
    async def reflect_on_trajectory(self, trajectory: Trajectory) -> Reflection
    async def identify_mistakes(self, trajectory: Trajectory) -> list[str]
    async def generate_alternative_approaches(self, task: str, failed_trajectory: Trajectory) -> list[str]
    
class Reflection(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    improvements: list[str]
    confidence: float
```

#### 4.2 Multi-Strategy ReAct
**File**: `src/agent_zoo/agents/react/strategies.py`

```python
class StrategyType(Enum):
    BREADTH_FIRST = "explore multiple options"
    DEPTH_FIRST = "pursue one path deeply"
    HYPOTHESIS_DRIVEN = "test specific hypotheses"
    ITERATIVE_REFINEMENT = "gradually improve solution"

class MultiStrategyReAct(ReActAgent):
    """ReAct that can switch between strategies."""
    
    def select_strategy(self, task: str) -> StrategyType
    async def execute_with_strategy(self, task: str, strategy: StrategyType) -> Any
```

### Phase 5: Testing Infrastructure (Week 3)

#### 5.1 Unit Tests
**File**: `tests/unit/agents/react/test_base.py`

```python
class TestReActAgent:
    def test_thought_generation()
    def test_action_parsing()
    def test_observation_processing()
    def test_trajectory_management()
    def test_completion_detection()
```

#### 5.2 Integration Tests
**File**: `tests/integration/agents/test_react_integration.py`

```python
class TestReActIntegration:
    def test_tool_execution()
    def test_memory_integration()
    def test_end_to_end_task()
    def test_error_recovery()
    def test_max_iterations()
```

#### 5.3 Benchmark Tests
**File**: `tests/benchmarks/test_react_performance.py`

```python
class ReActBenchmark:
    def benchmark_token_usage()
    def benchmark_execution_time()
    def benchmark_success_rate()
    def compare_with_baselines()
```

### Phase 6: Examples and Documentation (Week 3-4)

#### 6.1 Basic Examples
**File**: `examples/react/basic_react.py`

```python
# Simple Q&A with search
async def question_answering_example():
    agent = ReActAgent(config, llm_client, tools)
    result = await agent.process("What is the capital of France?")

# Multi-step reasoning
async def multi_step_example():
    agent = ReActAgent(config, llm_client, tools)
    result = await agent.process("Compare the populations of Tokyo and New York")
```

#### 6.2 Advanced Examples
**File**: `examples/react/advanced_react.py`

```python
# With memory
async def memory_augmented_example():
    memory = MemoryManager(memory_config, llm_client)
    agent = MemoryAugmentedReAct(config, memory)
    result = await agent.process("Solve this similar to the last problem")

# With reflection
async def reflective_example():
    agent = ReflectiveReAct(config, llm_client, tools)
    result = await agent.process_with_reflection("Complex task...")
```

## Technical Specifications

### Dependencies

```python
# External dependencies (already in Agent Zoo)
pydantic >= 2.0
structlog
asyncio

# Internal dependencies
from agent_zoo.core.base import BaseAgent, AgentConfig
from agent_zoo.interfaces.messages import Message, Thought, Action, Observation
from agent_zoo.interfaces.state import AgentState, AgentStatus
from agent_zoo.tools import Tool, ToolRegistry
from agent_zoo.core.memory import MemoryManager, MemoryType
```

### Performance Requirements

- **Response Time**: < 5 seconds for simple tasks, < 30 seconds for complex tasks
- **Token Usage**: < 2000 tokens for simple tasks, < 10000 for complex tasks
- **Success Rate**: > 70% on standard benchmarks
- **Memory Usage**: < 500MB per agent instance

### Error Handling

```python
class ReActError(Exception):
    """Base exception for ReAct errors."""

class ThoughtGenerationError(ReActError):
    """Failed to generate coherent thought."""

class ActionParsingError(ReActError):
    """Failed to parse action from text."""

class ToolExecutionError(ReActError):
    """Tool execution failed."""

class MaxIterationsError(ReActError):
    """Exceeded maximum iterations."""

class TrajectoryOverflowError(ReActError):
    """Trajectory exceeded context window."""
```

## Development Timeline

### Week 1: Foundation
- [ ] Implement base ReAct agent class
- [ ] Create trajectory management system
- [ ] Build prompt templates
- [ ] Write initial unit tests

### Week 2: Integration
- [ ] Integrate with tool system
- [ ] Add memory support
- [ ] Implement action parsing
- [ ] Create integration tests

### Week 3: Advanced Features
- [ ] Add self-reflection
- [ ] Implement multiple strategies
- [ ] Create performance optimizations
- [ ] Write benchmark suite

### Week 4: Polish and Documentation
- [ ] Create comprehensive examples
- [ ] Write user documentation
- [ ] Performance tuning
- [ ] Code review and cleanup

## Testing Strategy

### Unit Testing Coverage Goals
- Core functionality: 100%
- Tool integration: 95%
- Memory integration: 95%
- Error handling: 100%
- Edge cases: 90%

### Integration Test Scenarios
1. **Simple Tool Use**: Search → Process → Answer
2. **Multi-hop Reasoning**: Search → Calculate → Search → Conclude
3. **Error Recovery**: Failed action → Retry with different approach
4. **Memory Utilization**: Recall → Apply → Store
5. **Complex Task**: Decompose → Parallel execution → Synthesis

### Performance Benchmarks
1. **HotpotQA Subset**: Multi-hop question answering
2. **Tool Use Tasks**: API calls, calculations, data processing
3. **Recovery Tasks**: Intentional failures to test resilience
4. **Token Efficiency**: Measure tokens per successful task

## Configuration Examples

### Basic Configuration
```python
config = ReActConfig(
    name="basic_react",
    max_iterations=10,
    max_thought_length=500,
    timeout_seconds=30.0,
    enable_monitoring=True
)
```

### Advanced Configuration
```python
config = ReActConfig(
    name="advanced_react",
    max_iterations=15,
    max_thought_length=800,
    max_action_retries=3,
    enable_self_reflection=True,
    trajectory_format="structured",
    memory_config=MemoryManagerConfig(
        use_llm_router=True,
        auto_consolidate=True
    ),
    timeout_seconds=60.0,
    enable_monitoring=True,
    enable_caching=True
)
```

## API Design

### Core API
```python
# Create agent
agent = ReActAgent(
    config=config,
    llm_client=llm_client,
    tools=tool_registry
)

# Process task
result = await agent.process("What is the weather in Paris?")

# Access trajectory
trajectory = agent.get_trajectory()

# Get metrics
metrics = agent.get_metrics()
```

### Advanced API
```python
# With memory
agent = MemoryAugmentedReAct(
    config=config,
    memory=memory_manager
)

# Process with context
result = await agent.process(
    task="Solve this problem",
    context={"previous_attempts": [...]}
)

# Batch processing
results = await agent.process_batch([task1, task2, task3])

# Streaming
async for update in agent.process_stream(task):
    print(update)  # Real-time trajectory updates
```

## Migration Path

For users wanting to migrate from basic agents to ReAct:

1. **Minimal Change**:
```python
# Before
agent = BaseAgent(config)

# After
agent = ReActAgent(config, llm_client, tools)
```

2. **With Existing Memory**:
```python
# Reuse existing memory manager
agent = MemoryAugmentedReAct(config, existing_memory_manager)
```

3. **Progressive Enhancement**:
```python
# Start simple
agent = ReActAgent(basic_config)

# Add features incrementally
agent.enable_reflection()
agent.add_memory(memory_manager)
agent.set_strategy(StrategyType.HYPOTHESIS_DRIVEN)
```

## Success Metrics

### Functionality Metrics
- [ ] All unit tests passing (100%)
- [ ] Integration tests passing (100%)
- [ ] Benchmark performance meets targets
- [ ] Documentation complete

### Quality Metrics
- [ ] Code coverage > 90%
- [ ] No critical security issues
- [ ] Performance within requirements
- [ ] Memory usage optimized

### User Experience Metrics
- [ ] Examples run successfully
- [ ] API intuitive and consistent
- [ ] Error messages helpful
- [ ] Migration path smooth

## Risk Mitigation

### Technical Risks
1. **Context Window Overflow**
   - Mitigation: Implement trajectory pruning and summarization
   
2. **Infinite Loops**
   - Mitigation: Loop detection and maximum iteration limits
   
3. **Tool Failures**
   - Mitigation: Retry logic and fallback strategies

### Performance Risks
1. **High Token Usage**
   - Mitigation: Efficient prompting and caching
   
2. **Slow Response Times**
   - Mitigation: Parallel tool execution where possible
   
3. **Memory Leaks**
   - Mitigation: Proper cleanup and resource management

## Next Steps

1. **Review and Approval**: Get team feedback on implementation plan
2. **Environment Setup**: Prepare development environment
3. **Begin Implementation**: Start with Phase 1 core components
4. **Weekly Reviews**: Track progress and adjust plan as needed
5. **Testing**: Continuous testing throughout development
6. **Documentation**: Update docs as features are completed
7. **Release Planning**: Prepare for integration into main branch

## Conclusion

This implementation plan provides a structured approach to adding ReAct architecture to Agent Zoo. By leveraging existing infrastructure and following a phased approach, we can deliver a robust, well-tested, and documented ReAct implementation that integrates seamlessly with the current codebase.

The modular design allows for incremental development and testing, reducing risk and ensuring quality at each phase. The plan emphasizes reusability, maintainability, and performance, aligning with Agent Zoo's architectural principles.