# Memory Manager Design Document

## Problem Statement

Currently, agents must manually:
- Store every piece of information in memory
- Query memory for relevant context
- Update memory items (activate/deactivate)
- Manage token budgets
- Handle consolidation
- Coordinate between different memory types

This creates significant overhead and complexity for agent implementations.

## Design Goals

1. **Automatic Capture**: Automatically store relevant information without explicit calls
2. **Smart Retrieval**: Intelligently retrieve context based on current needs
3. **Lifecycle Management**: Handle activation/deactivation automatically
4. **Transparent Operation**: Work behind the scenes with minimal agent awareness
5. **Flexible Control**: Allow manual override when needed
6. **Multi-Memory Coordination**: Manage working, semantic, episodic, and procedural memory together

## Key Design Questions

### 1. Automatic Storage Triggers
AUTHOR: I LIKE THE PATTERN PRESENTED BELOW WHERE THE MEMORY MANAGER HAS METHODS LIKE OBSERVE AND GET CONTEXT

**Option A: Message Interceptor**
- Intercept all messages/observations
- Automatically store based on rules
- Pro: Truly automatic
- Con: May store unnecessary data

**Option B: Decorator Pattern**
```python
@memory_managed
async def process_user_input(self, message: str):
    # Automatically stored in memory
    return response
```
- Pro: Explicit but low-overhead
- Con: Requires decoration

**Option C: Context Manager**
```python
async with memory_context() as mem:
    # Everything in this block is tracked
    response = await agent.think(input)
```
- Pro: Clear boundaries
- Con: Still somewhat manual

### 2. Automatic Retrieval Strategy

**When to retrieve?**
- Before each agent action?
- On-demand when needed?
- Continuously maintain a "hot" context?

**What to retrieve?**
AUTHOR: I DO NOT UNDERSTAND ONE THING. WHILE CREATING A MEMORY ITEM WE ASSIGN A RELEVANCE SCORE TO IT. HOWEVER, HOW DO WE KNOW THE RELEVANCE OF AN ITEM? IT IS RELATIVE TO EACH QUERY IN MY UNDERSTANDING. 
- How to determine relevance without explicit query?
AUTHOR: THAT IS MY POINT THAT WE CANNOT ASSIGN RELEVANCE TO AN ITEM WITHOUT QUERY. LETS SAY THAT WE HAVE A LONG TERM MEMORY AND WE INITIALY ASSIGN RELEVANCE TO AN ITEM. HOW CAN WE KNOW WHAT TO INPUT IF WE DONT KNOW THE FUTURE USAGE.
- Should we use embedding similarity?
AUTHOR: I THINK THAT THIS IS A GREAT IDEA TO USE EMBEDDINGS AND STORE MEMORIES IN SOME KIND OF VECTOR DATABASE. WHAT DO YOU THINK? IS THIS HOW IT IS DONE IN OTHER LIBRARIES OR THIS IS AN OVERKILL?
- Role of recency vs. relevance?

### 3. Memory Type Routing

How does the manager decide which memory type to use?
AUTHOR: I THINK THAT MEMORYMANAGER SHOULD USE LLM AS WELL TO DECIDE ON WHERE TO PLACE THE MEMORIES AND WHAT KIND OF MEMORIES TO RETRIVE. IT SHOULD GET THE MESSAGE AND DECIDE WHAT TO DO WITH IT, WHAT KIND OF MEMORY IT REPRESENTS AND WHICH MEMORY TYPES TO QUERY FOR COTNTEXT. WHAT DO YOU THINK ABOUT THAT?

```python
# Automatic routing based on content type?
manager.store(item)  # Goes to working, semantic, or episodic?

# Or explicit but simplified?
manager.store_fact(fact)  # → Semantic
manager.store_experience(exp)  # → Episodic
manager.store_current(item)  # → Working
```

### 4. Context Window Management

**Automatic Context Building**
```python
# Instead of manual context building:
context = await memory.get_active_context(max_tokens=2000)

# Manager provides:
context = await manager.get_context_for_llm()  # Automatically optimized
```

**Questions:**
- How to prioritize what goes into limited context?
AUTHOR: AS I SAID BEFORE, HOW ABOUT USING LLM AS AN AUTOMATION?
- Should manager track LLM model to know token limits?
WHAT DO YOU MEAN? IT SHOULD BE PROVIDED IN A CONFIG. AM I RIGHT?
- How to handle context overflow?

### 5. Memory Consolidation Strategy

AUTHOR: ALL THOSE STRATEGIES SEEM GREAT AND ARE NOT EXCLUDING MUTUALLY I THINK THAT THEY SHOULD BE CHOSEN IN CONFIG AND USER COULD WANT TO USE MORE THAN ONE
**Option A: Time-based**
- Consolidate every N seconds/minutes
- Simple but may miss important moments

**Option B: Event-based**
- Consolidate after task completion
- Consolidate when switching topics
- More intelligent but needs event detection

**Option C: Pressure-based**
- Consolidate when memory is nearly full
- Reactive rather than proactive

### 6. Integration with Agent Base Class

Should MemoryManager be:
- A mixin that agents inherit?
- A component that agents compose?
- A middleware layer?
- Built into BaseAgent?

AUTHOR: I THINK THAT IT COULD BE A PART OF A BASEAGENT CLASS BUT THAT MEANS THAT THE ABSTRACT METHOD LIKE PROCESS SHOULD BE IMPLEMENTED AT THIS LEVEL. DERIVED CLASSES SHOULD JUST IMPLEMENT _process() METHOD THAT WOULD BE INVOKED IN THE MIDDLE OF THE PROCESS METHOD

## Proposed Architecture

### Core Components

```python
class MemoryManager:
    """
    Orchestrates all memory operations for an agent.
    """
    
    def __init__(self, config: MemoryManagerConfig):
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()  # Future
        self.episodic = EpisodicMemory()  # Future
        self.procedural = ProceduralMemory()  # Future
        
        # Automatic capture settings
        self.auto_capture = config.auto_capture
        self.capture_rules = config.capture_rules
        
        # Context management
        self.model_context_limit = config.model_context_limit
        self.context_reserve = config.context_reserve  # Reserve for prompts
        
    async def observe(self, observation: Any) -> None:
        """
        Process an observation (message, tool result, etc).
        Automatically determines storage.
        """
        
    async def get_context(self) -> list[Any]:
        """
        Get relevant context for current situation.
        Automatically determines what's relevant.
        """
        
    async def complete_task(self, task_id: str) -> None:
        """
        Signal task completion for memory management.
        Handles deactivation and consolidation.
        """
```

### Integration Patterns

**Pattern 1: Transparent Integration**
AUTHOR: I LIKE THIS PATTERN AND I THINK WE SHOULD STICK WITH IT
HOWEVER I THINK THAT GET_CONTEXT SHOULD ALSO TAKE MESSAGE AS AN INPUT TO KNOW
CONTEXT OF WHAT THE MEMORY SHOULD GENERATE.


```python
class SmartAgent(BaseAgent):
    def __init__(self):
        self.memory = MemoryManager(auto_capture=True)
    
    async def process(self, message: Message):
        # Memory automatically captures the message
        await self.memory.observe(message)
        
        # Get relevant context automatically
        context = await self.memory.get_context()
        
        # Use context for reasoning
        response = await self.llm.generate(context + [message])
        
        # Memory automatically captures response
        await self.memory.observe(response)
        
        return response
```

**Pattern 2: Hook-based Integration**
```python
class HookedAgent(BaseAgent):
    def __init__(self):
        self.memory = MemoryManager()
        
        # Register hooks
        self.on_message_received = self.memory.capture_message
        self.on_tool_called = self.memory.capture_tool_use
        self.on_response_generated = self.memory.capture_response
```

### Capture Rules System

```python
@dataclass
class CaptureRule:
    """Defines when and how to capture information."""
    
    trigger: CaptureWhen  # MESSAGE, TOOL_RESULT, THOUGHT, etc.
    condition: Callable[[Any], bool]  # Should capture?
    processor: Callable[[Any], MemoryItem]  # How to process
    destination: MemoryType  # Which memory system
    priority: float  # Storage priority
```

### Relevance Detection

```python
class RelevanceDetector:
    """Determines what memories are relevant to current context."""
    
    def __init__(self):
        self.embedder = EmbeddingModel()  # For semantic similarity
        self.patterns = []  # Learned access patterns
    
    async def find_relevant(
        self,
        current_context: Any,
        memory_pool: list[MemoryItem],
        max_items: int
    ) -> list[MemoryItem]:
        """Find most relevant memories for current situation."""
```

## Implementation Priorities

### Phase 1: Basic Automation (MVP)
- Auto-capture messages and responses
- Auto-retrieve recent context
- Auto-deactivate after task completion
- Simple rule-based routing

### Phase 2: Smart Context
- Embedding-based relevance
- Dynamic context sizing
- Multi-memory coordination
- Event-based consolidation

### Phase 3: Learning and Adaptation
- Learn access patterns
- Adaptive capture rules
- Predictive pre-fetching
- Memory importance learning

## Open Questions for Discussion

1. **How much magic is too much?**
   - Should everything be automatic or should agents have explicit control?
   - How to balance convenience vs. predictability?

2. **Performance implications**
   - Will automatic capture slow down the agent?
   - How to make relevance detection fast?

3. **Memory overhead**
   - Risk of storing too much automatically?
   - How aggressive should auto-eviction be?

4. **Debugging and observability**
   - How to debug automatic memory decisions?
   - Should we log all automatic actions?

5. **Customization needs**
   - How to handle domain-specific memory patterns?
   - Should capture rules be learned or configured?

## Example Use Cases to Consider

### Use Case 1: Conversational Agent
```python
# Should automatically:
# - Store conversation history in working memory
# - Extract facts to semantic memory
# - Store important exchanges in episodic memory
# - Learn conversation patterns in procedural memory
```

### Use Case 2: Task-Oriented Agent
```python
# Should automatically:
# - Track task state and progress
# - Store intermediate results
# - Deactivate completed subtasks
# - Maintain task context across interruptions
```

### Use Case 3: Research Agent
```python
# Should automatically:
# - Store search results temporarily
# - Extract and permanent facts
# - Track information sources
# - Build knowledge graph
```

## Decision Points Needed

1. **Default behavior**: Should MemoryManager be opt-in or opt-out for automatic features?

2. **Integration level**: Should this be part of BaseAgent or a separate component?

3. **Configuration complexity**: Simple presets vs. detailed configuration?

4. **Memory type abstraction**: Should users need to know about different memory types?

5. **Context strategy**: Push (manager provides) vs. Pull (agent requests)?

## Next Steps

1. Decide on core architecture approach
2. Define the minimal viable automation
3. Create prototype implementation
4. Test with different agent patterns
5. Iterate based on usage friction

## Questions for You

1. **What level of automation do you envision?**
   - Fully automatic with minimal agent awareness?
   - Semi-automatic with explicit triggers?
   - Manual with convenient helpers?

2. **How should agents interact with memory?**
   - Through MemoryManager only?
   - Direct access to memory types when needed?
   - Mixed approach?

3. **What's the primary pain point to solve first?**
   - Automatic capture?
   - Intelligent retrieval?
   - Context management?
   - Memory coordination?

4. **Should MemoryManager be opinionated?**
   - Enforce best practices?
   - Flexible to any pattern?
   - Configurable strictness?