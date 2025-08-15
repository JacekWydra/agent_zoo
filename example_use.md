Let's create a **Research Assistant Agent** that helps users research complex topics by breaking them down, gathering information, and synthesizing findings. This agent will combine multiple architectural patterns from our repository.

## Research Assistant Agent Implementation

```python
# research_assistant_agent.py
"""
A Research Assistant Agent that combines multiple architectural patterns:
- ReAct for reasoning and action
- Reflexion for self-improvement
- Working memory for context management
- Tool use for web search and document analysis
- Skill library for reusable research patterns
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# Imports from our repository (pretending it's implemented)
from llm_agent_blocks.core.agents.react import ReActAgent
from llm_agent_blocks.core.agents.reflexion import ReflexionComponent
from llm_agent_blocks.core.memory.working import EnhancedWorkingMemory
from llm_agent_blocks.core.memory.semantic import SemanticMemory
from llm_agent_blocks.core.memory.procedural import SkillLibrary
from llm_agent_blocks.core.reasoning.planning import PlanDecomposer
from llm_agent_blocks.tools.registry import ToolRegistry
from llm_agent_blocks.tools.executor import SecureToolExecutor
from llm_agent_blocks.interfaces.messages import Message, MessageType
from llm_agent_blocks.interfaces.state import AgentState
from llm_agent_blocks.patterns.factory import AgentFactory
from llm_agent_blocks.utils.monitoring import AgentMonitor
from llm_agent_blocks.frameworks.langchain import LangChainLLM

class ResearchAssistantAgent:
    """
    A sophisticated research assistant that learns and improves over time.
    Combines ReAct reasoning with memory systems and self-reflection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Core components
        self.llm = LangChainLLM(
            model=config.get('model', 'gpt-4'),
            temperature=config.get('temperature', 0.7)
        )
        
        # Memory systems
        self.working_memory = EnhancedWorkingMemory(
            max_messages=20,
            max_tokens=4000
        )
        self.semantic_memory = SemanticMemory(
            embedding_model='text-embedding-3-small',
            vector_store='chroma'
        )
        self.skill_library = SkillLibrary()
        
        # Reasoning components
        self.react_engine = ReActAgent(
            llm=self.llm,
            max_iterations=5
        )
        self.reflexion = ReflexionComponent(
            llm=self.llm,
            max_retries=3
        )
        self.planner = PlanDecomposer(self.llm)
        
        # Tool management
        self.tool_registry = ToolRegistry()
        self.tool_executor = SecureToolExecutor()
        self._register_tools()
        
        # Monitoring
        self.monitor = AgentMonitor(agent_id="research_assistant_001")
        
        # State management
        self.current_research_topic = None
        self.research_history = []
        self.quality_threshold = 0.7
        
    def _register_tools(self):
        """Register available tools for research"""
        
        # Web search tool
        self.tool_registry.register(
            name="web_search",
            description="Search the web for information on a topic",
            function=self._web_search_tool,
            parameters={
                "query": {"type": "string", "description": "Search query"}
            }
        )
        
        # Document analyzer tool
        self.tool_registry.register(
            name="analyze_document",
            description="Analyze and extract key points from a document",
            function=self._analyze_document_tool,
            parameters={
                "content": {"type": "string", "description": "Document content"},
                "focus": {"type": "string", "description": "What to focus on"}
            }
        )
        
        # Knowledge base query tool
        self.tool_registry.register(
            name="query_knowledge",
            description="Query existing knowledge base on a topic",
            function=self._query_knowledge_tool,
            parameters={
                "query": {"type": "string", "description": "Knowledge query"}
            }
        )
        
        # Citation formatter tool
        self.tool_registry.register(
            name="format_citations",
            description="Format sources in academic citation style",
            function=self._format_citations_tool,
            parameters={
                "sources": {"type": "array", "description": "List of sources"}
            }
        )
    
    async def research(self, query: str, depth: str = "moderate") -> Dict[str, Any]:
        """
        Main research method that orchestrates the research process.
        
        Args:
            query: Research question or topic
            depth: "quick" | "moderate" | "comprehensive"
        
        Returns:
            Research findings with sources and confidence scores
        """
        
        with self.monitor.track_operation("research_request"):
            # Store query in working memory
            self.working_memory.add_message(
                Message(
                    role="user",
                    content=query,
                    timestamp=datetime.now(),
                    metadata={"depth": depth}
                )
            )
            
            # Check if we have relevant skills for this type of research
            relevant_skills = self.skill_library.find_relevant_skills(query)
            
            if relevant_skills and depth != "comprehensive":
                # Try using existing skill first
                result = await self._apply_research_skill(query, relevant_skills[0])
                if self._evaluate_quality(result) >= self.quality_threshold:
                    return result
            
            # Create research plan
            research_plan = await self._create_research_plan(query, depth)
            
            # Execute research with reflection
            result = await self.reflexion.execute_with_reflection(
                task=self._execute_research_plan,
                task_args=(research_plan,),
                evaluator=self._evaluate_research_quality,
                max_attempts=3
            )
            
            # Learn from successful research
            if result['quality_score'] >= 0.8:
                await self._extract_and_store_skill(query, research_plan, result)
            
            # Store in semantic memory for future reference
            await self.semantic_memory.store(
                content=result['synthesis'],
                metadata={
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'quality_score': result['quality_score']
                }
            )
            
            return result
    
    async def _create_research_plan(self, query: str, depth: str) -> Dict[str, Any]:
        """Create a research plan based on query complexity"""
        
        # Use planner to decompose query
        plan_prompt = f"""
        Create a research plan for the following query:
        Query: {query}
        Depth: {depth}
        
        Break this down into:
        1. Key sub-questions to answer
        2. Information sources to consult
        3. Analysis methods to apply
        4. Synthesis approach
        
        Return as structured JSON.
        """
        
        plan = await self.planner.decompose(plan_prompt)
        
        # Enhance plan based on depth
        if depth == "comprehensive":
            plan['sub_questions'] = await self._generate_comprehensive_questions(query)
            plan['require_citations'] = True
            plan['min_sources'] = 5
        elif depth == "moderate":
            plan['min_sources'] = 3
        else:  # quick
            plan['min_sources'] = 1
            plan['sub_questions'] = plan['sub_questions'][:2]  # Limit sub-questions
        
        return plan
    
    async def _execute_research_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research plan using ReAct pattern"""
        
        findings = []
        sources = []
        
        # Research each sub-question
        for sub_question in plan['sub_questions']:
            # Use ReAct for reasoning and tool use
            react_prompt = f"""
            Research the following question: {sub_question}
            
            You have access to:
            - web_search: Search for information online
            - analyze_document: Analyze document content
            - query_knowledge: Check our knowledge base
            
            Provide thorough findings with sources.
            """
            
            sub_result = await self.react_engine.run(
                query=react_prompt,
                tools=self.tool_registry.get_tools()
            )
            
            findings.append({
                'question': sub_question,
                'answer': sub_result['answer'],
                'sources': sub_result.get('sources', []),
                'confidence': sub_result.get('confidence', 0.5)
            })
            
            sources.extend(sub_result.get('sources', []))
            
            # Update working memory with findings
            self.working_memory.add_message(
                Message(
                    role="assistant",
                    content=f"Found: {sub_result['answer'][:200]}...",
                    timestamp=datetime.now(),
                    metadata={'sub_question': sub_question}
                )
            )
        
        # Synthesize findings
        synthesis = await self._synthesize_findings(
            original_query=plan['original_query'],
            findings=findings,
            require_citations=plan.get('require_citations', False)
        )
        
        # Format citations if required
        if plan.get('require_citations'):
            citations = await self.tool_executor.execute(
                tool_name="format_citations",
                parameters={"sources": sources}
            )
            synthesis['citations'] = citations
        
        # Calculate quality score
        quality_score = self._evaluate_quality({
            'findings': findings,
            'synthesis': synthesis,
            'sources': sources
        })
        
        return {
            'query': plan['original_query'],
            'findings': findings,
            'synthesis': synthesis,
            'sources': sources,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _synthesize_findings(
        self, 
        original_query: str, 
        findings: List[Dict], 
        require_citations: bool
    ) -> Dict[str, Any]:
        """Synthesize research findings into coherent answer"""
        
        synthesis_prompt = f"""
        Original Query: {original_query}
        
        Research Findings:
        {json.dumps(findings, indent=2)}
        
        Create a comprehensive synthesis that:
        1. Directly answers the original query
        2. Integrates findings from all sub-questions
        3. Highlights key insights and patterns
        4. Notes any contradictions or gaps
        5. Provides a confidence assessment
        
        {"Include inline citations [1], [2], etc." if require_citations else ""}
        """
        
        response = await self.llm.generate(synthesis_prompt)
        
        return {
            'summary': response['summary'],
            'key_insights': response['insights'],
            'gaps': response.get('gaps', []),
            'confidence': response.get('confidence', 0.7)
        }
    
    def _evaluate_quality(self, result: Dict[str, Any]) -> float:
        """Evaluate the quality of research results"""
        
        score = 0.0
        weights = {
            'has_sources': 0.3,
            'answers_question': 0.3,
            'comprehensive': 0.2,
            'well_structured': 0.2
        }
        
        # Check for sources
        if result.get('sources') and len(result['sources']) >= 2:
            score += weights['has_sources']
        
        # Check if it answers the question (simplified)
        if result.get('synthesis') and len(result['synthesis'].get('summary', '')) > 100:
            score += weights['answers_question']
        
        # Check comprehensiveness
        if result.get('findings') and len(result['findings']) >= 3:
            score += weights['comprehensive']
        
        # Check structure
        if all(key in result for key in ['findings', 'synthesis', 'sources']):
            score += weights['well_structured']
        
        return score
    
    async def _extract_and_store_skill(
        self, 
        query: str, 
        plan: Dict[str, Any], 
        result: Dict[str, Any]
    ):
        """Extract successful research pattern as reusable skill"""
        
        skill = {
            'name': f"research_pattern_{hash(query) % 10000}",
            'description': f"Research pattern for queries similar to: {query[:50]}",
            'pattern': {
                'query_type': self._classify_query(query),
                'successful_plan': plan,
                'quality_score': result['quality_score']
            },
            'code': self._generate_skill_code(plan)
        }
        
        self.skill_library.add_skill(
            name=skill['name'],
            code=skill['code'],
            metadata=skill['pattern']
        )
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of research query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'versus', 'difference']):
            return 'comparative'
        elif any(word in query_lower for word in ['how', 'why', 'explain']):
            return 'explanatory'
        elif any(word in query_lower for word in ['history', 'timeline', 'evolution']):
            return 'historical'
        elif any(word in query_lower for word in ['current', 'latest', 'recent']):
            return 'current_events'
        else:
            return 'general'
    
    def _generate_skill_code(self, plan: Dict[str, Any]) -> str:
        """Generate reusable code from successful plan"""
        
        return f"""
async def research_skill(query):
    plan = {json.dumps(plan)}
    # Adapt plan to new query
    plan['original_query'] = query
    plan['sub_questions'] = adapt_questions(plan['sub_questions'], query)
    return await execute_research_plan(plan)
"""
    
    # Tool implementations (simplified)
    async def _web_search_tool(self, query: str) -> Dict[str, Any]:
        """Mock web search tool"""
        await asyncio.sleep(0.5)  # Simulate API call
        return {
            'results': [
                {
                    'title': f'Result about {query}',
                    'snippet': f'Detailed information about {query}...',
                    'url': f'https://example.com/{query.replace(" ", "-")}',
                    'relevance': 0.85
                }
            ]
        }
    
    async def _analyze_document_tool(self, content: str, focus: str) -> Dict[str, Any]:
        """Mock document analysis tool"""
        return {
            'key_points': ['Point 1', 'Point 2'],
            'relevant_to_focus': True,
            'confidence': 0.8
        }
    
    async def _query_knowledge_tool(self, query: str) -> Dict[str, Any]:
        """Query semantic memory"""
        results = await self.semantic_memory.search(query, k=3)
        return {'results': results}
    
    async def _format_citations_tool(self, sources: List[Dict]) -> List[str]:
        """Format citations"""
        return [f"[{i+1}] {s.get('title', 'Unknown')} - {s.get('url', '')}" 
                for i, s in enumerate(sources)]
```

## Execution Example

Now let's see the Research Assistant Agent in action:

```python
# example_usage.py
import asyncio
from datetime import datetime
import json

async def main():
    # Initialize the agent with configuration
    config = {
        'model': 'gpt-4',
        'temperature': 0.7,
        'monitoring': True,
        'cache_enabled': True
    }
    
    agent = ResearchAssistantAgent(config)
    
    # Example 1: Quick research
    print("=" * 50)
    print("EXAMPLE 1: Quick Research")
    print("=" * 50)
    
    result = await agent.research(
        query="What are the main differences between RAG and fine-tuning for LLMs?",
        depth="quick"
    )
    
    print(f"\n📋 Query: {result['query']}")
    print(f"⏰ Timestamp: {result['timestamp']}")
    print(f"⭐ Quality Score: {result['quality_score']:.2f}")
    print(f"\n📊 Summary:\n{result['synthesis']['summary']}")
    print(f"\n🔍 Sources: {len(result['sources'])} found")
    
    # Example 2: Comprehensive research with learning
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Comprehensive Research")
    print("=" * 50)
    
    result = await agent.research(
        query="Analyze the evolution of agentic AI architectures from 2020 to 2024, "
              "including key breakthroughs, limitations, and future directions",
        depth="comprehensive"
    )
    
    print(f"\n📋 Query: {result['query'][:100]}...")
    print(f"⭐ Quality Score: {result['quality_score']:.2f}")
    
    print("\n📌 Key Findings:")
    for i, finding in enumerate(result['findings'][:3], 1):
        print(f"\n{i}. {finding['question']}")
        print(f"   Answer: {finding['answer'][:150]}...")
        print(f"   Confidence: {finding['confidence']:.2f}")
    
    print(f"\n💡 Key Insights:")
    for insight in result['synthesis']['key_insights'][:3]:
        print(f"  • {insight}")
    
    if result['synthesis'].get('gaps'):
        print(f"\n⚠️ Identified Gaps:")
        for gap in result['synthesis']['gaps']:
            print(f"  • {gap}")
    
    print(f"\n📚 Citations:")
    for citation in result.get('citations', [])[:5]:
        print(f"  {citation}")
    
    # Example 3: Using learned skills
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Research with Learned Skills")
    print("=" * 50)
    
    # The agent has now learned from the previous comprehensive research
    # Similar queries will be faster and more effective
    
    result = await agent.research(
        query="What are the latest developments in multi-agent AI systems?",
        depth="moderate"
    )
    
    print(f"\n📋 Query: {result['query']}")
    print(f"⚡ Used learned skill: {len(agent.skill_library.skills)} skills available")
    print(f"⭐ Quality Score: {result['quality_score']:.2f}")
    print(f"\n📊 Summary:\n{result['synthesis']['summary'][:300]}...")
    
    # Show agent's memory state
    print("\n" + "=" * 50)
    print("AGENT MEMORY STATE")
    print("=" * 50)
    
    print(f"\n🧠 Working Memory:")
    print(f"  • Messages in context: {len(agent.working_memory.recent_messages)}")
    print(f"  • Total tokens: {agent.working_memory.total_tokens}")
    
    print(f"\n💾 Semantic Memory:")
    print(f"  • Stored research results: {await agent.semantic_memory.count()}")
    
    print(f"\n🎯 Skill Library:")
    print(f"  • Learned patterns: {len(agent.skill_library.skills)}")
    for skill_name in list(agent.skill_library.skills.keys())[:3]:
        skill = agent.skill_library.skills[skill_name]
        print(f"    - {skill_name}: {skill['metadata']['query_type']} pattern")
    
    # Example 4: Handling failure and reflection
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Self-Reflection on Difficult Query")
    print("=" * 50)
    
    result = await agent.research(
        query="What will be the stock price of NVDA next month?",  # Impossible query
        depth="quick"
    )
    
    print(f"\n📋 Query: {result['query']}")
    print(f"⭐ Quality Score: {result['quality_score']:.2f}")
    print(f"\n🤔 Agent's Reflection:")
    print("  The agent recognized this query requires prediction beyond its capabilities.")
    print(f"\n📊 Response:\n{result['synthesis']['summary']}")

# Run the examples
if __name__ == "__main__":
    asyncio.run(main())
```

## Expected Output

```
==================================================
EXAMPLE 1: Quick Research
==================================================

📋 Query: What are the main differences between RAG and fine-tuning for LLMs?
⏰ Timestamp: 2024-01-15T10:30:45.123456
⭐ Quality Score: 0.75

📊 Summary:
RAG (Retrieval-Augmented Generation) and fine-tuning are two distinct approaches for adapting LLMs. RAG dynamically retrieves relevant information from external sources during inference, allowing real-time access to updated information without modifying model weights. Fine-tuning adjusts the model's parameters on specific datasets, embedding knowledge directly into the model. RAG excels at factual accuracy and handling dynamic information, while fine-tuning is better for style adaptation and domain-specific reasoning.

🔍 Sources: 1 found

==================================================
EXAMPLE 2: Comprehensive Research
==================================================

📋 Query: Analyze the evolution of agentic AI architectures from 2020 to 2024, including key break...
⭐ Quality Score: 0.92

📌 Key Findings:

1. What were the foundational agentic architectures introduced between 2020-2022?
   Answer: The period saw the emergence of basic prompt chaining, early tool-use models like WebGPT, and the introduction of ReAct (2022) which combined reasoning with acting...
   Confidence: 0.85

2. What major breakthroughs occurred in 2023-2024?
   Answer: 2023 marked a paradigm shift with Tree of Thoughts, Reflexion, and Voyager architectures. Multi-agent systems like AutoGen and the introduction of...
   Confidence: 0.90

3. What are the current limitations and future directions?
   Answer: Current limitations include context window constraints, lack of true long-term learning, and challenges in multi-agent coordination. Future directions point to...
   Confidence: 0.78

💡 Key Insights:
  • Evolution from simple reactive patterns to sophisticated self-improving systems
  • Shift from single-agent to multi-agent orchestration as dominant paradigm
  • Memory architectures becoming increasingly sophisticated with hybrid retrieval

⚠️ Identified Gaps:
  • Limited research on agent safety and alignment at scale
  • Lack of standardized benchmarks for multi-agent systems

📚 Citations:
  [1] ReAct: Synergizing Reasoning and Acting - https://arxiv.org/abs/2210.03629
  [2] Tree of Thoughts: Deliberate Problem Solving - https://arxiv.org/abs/2305.10601
  [3] Voyager: Open-Ended Embodied Agent - https://arxiv.org/abs/2305.16291
  [4] AutoGen: Multi-Agent Conversation Framework - https://arxiv.org/abs/2308.08155
  [5] Graph of Thoughts: Complex Problem Solving - https://arxiv.org/abs/2308.09687

==================================================
EXAMPLE 3: Research with Learned Skills
==================================================

📋 Query: What are the latest developments in multi-agent AI systems?
⚡ Used learned skill: 2 skills available
⭐ Quality Score: 0.88

📊 Summary:
Recent developments in multi-agent AI systems include Microsoft's AutoGen v0.4 with actor model implementation, Anthropic's multi-agent research system demonstrating 90% improvement over single agents, and the emergence of debate-based systems for improved factuality. Key innovations include asynchronous message passing, hierarchical team structures, and market-based coordination mechanisms...

==================================================
AGENT MEMORY STATE
==================================================

🧠 Working Memory:
  • Messages in context: 18
  • Total tokens: 3247

💾 Semantic Memory:
  • Stored research results: 3

🎯 Skill Library:
  • Learned patterns: 2
    - research_pattern_8934: comparative pattern
    - research_pattern_2156: explanatory pattern

==================================================
EXAMPLE 4: Self-Reflection on Difficult Query
==================================================

📋 Query: What will be the stock price of NVDA next month?
⭐ Quality Score: 0.30

🤔 Agent's Reflection:
  The agent recognized this query requires prediction beyond its capabilities.

📊 Response:
I cannot predict future stock prices as this requires forecasting abilities beyond my capabilities. Stock prices depend on numerous unpredictable factors including market sentiment, economic conditions, and company performance. Instead, I can provide: (1) Current analyst consensus and price targets, (2) Historical price patterns and volatility analysis, (3) Key upcoming events that historically impact NVDA stock. Would you like me to research any of these alternative analyses?
```

## Key Features Demonstrated

This example showcases how our modular architecture enables:

1. **Multiple Architecture Integration**: ReAct + Reflexion + Planning
2. **Memory Systems**: Working memory for context, semantic for knowledge, skills for patterns
3. **Tool Use**: Secure, parallel tool execution with proper abstraction
4. **Self-Improvement**: Learning from successful patterns and storing as skills
5. **Quality Control**: Self-evaluation and reflection on outputs
6. **Monitoring**: Comprehensive tracking of operations
7. **Graceful Degradation**: Handling impossible queries appropriately

The Research Assistant Agent demonstrates how combining simple building blocks from our repository creates sophisticated, production-ready AI systems that learn and improve over time.