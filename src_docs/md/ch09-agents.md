# Chapter 9: Agents — when LLMs take the wheel

*From question-answering to autonomous action.*

> "The question of whether machines can think is about as relevant as whether submarines can swim."
> — Edsger Dijkstra

## The agentic leap

An LLM that only answers questions is a fancy search engine. An agent decides what to do next.

The difference matters. Ask GPT-5 "What's the weather?" and it tells you it can't check. Give GPT-5 access to a weather API as a tool, and it calls the API, parses the result, and answers with live data. Same model. Different architecture.

Agents combine three capabilities:

1. **Reasoning**: Breaking complex tasks into steps
2. **Tool use**: Calling external functions
3. **Memory**: Tracking state across interactions

This chapter covers the patterns, frameworks, and hard-won lessons from building agents that actually work.

## Imagine...

Imagine you're training a new employee. At first, you give them explicit instructions: "Go to the filing cabinet. Pull folder A-23. Find the invoice. Calculate the total."

Eventually, you trust them with goals instead of steps: "Handle the Johnson account billing." They figure out what needs to happen. They ask questions when stuck. They use judgment. They might solve it differently than you would, but they get the result.

That's the jump from LLM to agent. An LLM follows instructions. An agent pursues goals. You say "fix the authentication bug" and the agent reads logs, forms hypotheses, writes test cases, edits code, runs tests, and iterates until tests pass.

The danger is obvious: autonomous systems make autonomous mistakes. A bad instruction from you affects one task. A bad judgment from an agent cascades into a dozen wrong actions before you notice.

Agents are powerful precisely because they're dangerous. This chapter teaches you to harness that power safely.

---

## The ReAct pattern

ReAct (Reasoning + Acting) is the foundation of modern agent design. The model alternates between thinking and doing.

```
Thought: I need to find the font file first.
Action: search_files("*.ttf")
Observation: Found 3 files: Arial.ttf, Times.ttf, Garamond.ttf
Thought: Now I should analyze each font's metrics.
Action: analyze_font("Arial.ttf")
Observation: {family: "Arial", weight: 400, ...}
Thought: I have the data I need to compare them.
Action: [final answer]
```

The key insight: **explicit reasoning before action improves reliability**. Without it, models jump to conclusions and make irreversible mistakes.

### Implementing ReAct

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class AgentState(BaseModel):
    thoughts: list[str] = []
    actions: list[str] = []
    observations: list[str] = []

agent = Agent(
    model="claude-sonnet-4",
    system_prompt="""You are a research assistant. Before each action:
    1. State your thought process
    2. Choose one action
    3. Wait for the observation
    4. Repeat until you have enough information to answer."""
)

@agent.tool
async def search_files(pattern: str) -> str:
    """Search for files matching the pattern."""
    import glob
    matches = glob.glob(f"**/{pattern}", recursive=True)
    return f"Found {len(matches)} files: {', '.join(matches[:10])}"

@agent.tool
async def read_file(path: str) -> str:
    """Read contents of a file."""
    with open(path) as f:
        return f.read()[:1000]  # Truncate for safety
```

## Agent frameworks compared

The ecosystem has fragmented into competing approaches. Here's what actually works.

### PydanticAI (Recommended Starting Point)

Built by the Pydantic team. Type-safe, minimal magic.

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class FontAnalysis(BaseModel):
    family: str
    classification: str
    recommended_uses: list[str]

agent = Agent(
    model="claude-sonnet-4",
    result_type=FontAnalysis,
    system_prompt="Analyze fonts and classify them."
)

@agent.tool
async def get_font_metrics(path: str) -> dict:
    """Extract font metrics from a file."""
    from fontTools.ttLib import TTFont
    font = TTFont(path)
    return {
        "family": font["name"].getName(1, 3, 1, 1033).toUnicode(),
        "glyph_count": font["maxp"].numGlyphs,
    }

result = await agent.run("Analyze Garamond.ttf for body text suitability")
print(result.output)  # FontAnalysis object with type safety
```

**Why PydanticAI wins:**
- Structured outputs via Pydantic models
- Dependency injection for tools
- Streaming support
- Multi-model support (OpenAI, Anthropic, Gemini, Ollama)
- No hidden prompts or abstractions

### AutoGen (Microsoft's Multi-Agent Framework)

For complex workflows with multiple specialized agents.

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents with different roles
researcher = AssistantAgent(
    name="researcher",
    system_message="You research font history and typographic trends.",
    llm_config={"model": "gpt-5"}
)

analyst = AssistantAgent(
    name="analyst",
    system_message="You analyze font metrics and technical specifications.",
    llm_config={"model": "gpt-5"}
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "workspace"}
)

# Agents can converse with each other
user_proxy.initiate_chat(
    researcher,
    message="Research the history of Garamond typeface"
)
```

**Use AutoGen when:**
- You need multiple agents collaborating
- Complex workflows with handoffs
- Research or analysis pipelines

**Avoid when:**
- Simple single-agent tasks
- Cost sensitivity (multi-agent = multiple API calls)

### CrewAI (Role-Based Teams)

Assigns agents explicit roles in a "crew."

```python
from crewai import Agent, Task, Crew

# Define specialized agents
type_historian = Agent(
    role="Typography Historian",
    goal="Research typeface origins and evolution",
    backstory="Expert in typographic history from Gutenberg to digital.",
    llm="gpt-5"
)

font_analyst = Agent(
    role="Font Analyst",
    goal="Analyze font files for technical characteristics",
    backstory="Specializes in OpenType features and font engineering.",
    llm="gpt-5"
)

# Define tasks
research_task = Task(
    description="Research the origins of {font_name}",
    agent=type_historian,
    expected_output="Historical summary with key dates and designers"
)

analysis_task = Task(
    description="Analyze the technical specifications of {font_name}",
    agent=font_analyst,
    expected_output="Technical report with metrics and features"
)

# Run the crew
crew = Crew(
    agents=[type_historian, font_analyst],
    tasks=[research_task, analysis_task],
    verbose=True
)

result = crew.kickoff(inputs={"font_name": "Garamond"})
```

**CrewAI strengths:**
- Clear role definitions
- Task dependencies
- Built-in collaboration patterns

**Weaknesses:**
- Verbose setup for simple tasks
- Role definitions can feel arbitrary

### smolagents (Hugging Face)

Minimalist, code-first approach.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model="Qwen/Qwen2.5-72B-Instruct"
)

# Agent writes and executes Python code to solve tasks
result = agent.run("Find the most popular open-source fonts on GitHub")
```

**smolagents philosophy:**
- Agent writes Python code, not structured tool calls
- Execution happens in sandboxed environment
- Minimal abstraction

**Best for:**
- Code-centric tasks
- Research and data gathering
- Prototyping agent ideas

### LangGraph (State Machines)

When you need precise control flow.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    research: str
    analysis: str
    final_answer: str

def research_node(state: AgentState) -> AgentState:
    # Research step
    result = call_llm(f"Research: {state['query']}")
    return {"research": result}

def analyze_node(state: AgentState) -> AgentState:
    # Analysis step
    result = call_llm(f"Analyze: {state['research']}")
    return {"analysis": result}

def synthesize_node(state: AgentState) -> AgentState:
    # Final synthesis
    result = call_llm(f"Synthesize: {state['analysis']}")
    return {"final_answer": result}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("synthesize", synthesize_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()
result = app.invoke({"query": "Compare serif vs sans-serif for body text"})
```

**LangGraph excels at:**
- Complex conditional workflows
- Explicit state management
- Debugging and observability

**Overkill for:**
- Simple Q&A agents
- Straightforward tool use

### Framework Selection Guide

| Need | Best Choice |
|------|-------------|
| Type-safe, simple agent | PydanticAI |
| Multi-agent collaboration | AutoGen |
| Role-based teams | CrewAI |
| Code-writing agents | smolagents |
| Complex state machines | LangGraph |
| Maximum control | Build your own |

**Default recommendation: Start with PydanticAI.** Upgrade to others when you hit specific limits.

## Code execution sandboxing

Agents that write code need sandboxes. Otherwise, one bad generation deletes your filesystem.

### The Risk

```python
# Agent generates this "helpful" code:
import shutil
shutil.rmtree("/")  # Goodbye, computer
```

Never execute LLM-generated code in your main environment.

### E2B (Cloud Sandboxes)

Spin up isolated containers per request.

```python
from e2b_code_interpreter import Sandbox

with Sandbox() as sandbox:
    # Execute untrusted code safely
    execution = sandbox.run_code("""
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('chart.png')
""")

    # Get results
    for result in execution.results:
        if result.png:
            display(result.png)
```

**E2B features:**
- 200 free hours/month
- Persistent filesystems
- Custom sandbox templates
- Multiple language support

### Modal (Serverless Containers)

Better for production workloads.

```python
import modal

app = modal.App("code-sandbox")

@app.function()
def run_untrusted_code(code: str) -> str:
    """Execute code in isolated container."""
    import subprocess
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True,
        timeout=30
    )
    return result.stdout.decode()

# Call from anywhere
with app.run():
    output = run_untrusted_code.remote("print(2 + 2)")
```

### Docker (Self-Hosted)

Roll your own with resource limits.

```python
import docker

client = docker.from_env()

def run_sandboxed(code: str, timeout: int = 30) -> str:
    container = client.containers.run(
        "python:3.12-slim",
        command=["python", "-c", code],
        mem_limit="256m",
        cpu_quota=50000,  # 50% of one CPU
        network_disabled=True,
        remove=True,
        detach=False,
        stdout=True,
        stderr=True
    )
    return container.decode()
```

### Sandbox Selection

| Priority | Best Choice |
|----------|-------------|
| Ease of use | E2B |
| Production scale | Modal |
| Full control | Docker |
| No infrastructure | Replit or CodeSandbox APIs |

## Multi-agent orchestration

Single agents hit capability ceilings. Multi-agent systems can exceed them—at the cost of complexity.

### When to Use Multiple Agents

**Use single agent when:**
- Task is well-defined
- One skill set suffices
- Latency matters
- Cost matters

**Use multi-agent when:**
- Task requires different expertise
- Subtasks can run in parallel
- Error checking benefits from multiple perspectives
- Complex workflows with handoffs

### Orchestration Patterns

#### Sequential Pipeline

Agents process in order, each building on previous output.

```python
# Font analysis pipeline
raw_data = extraction_agent.run(font_file)
metrics = analysis_agent.run(raw_data)
recommendations = recommendation_agent.run(metrics)
```

**Pros**: Simple, predictable
**Cons**: Slow, single point of failure

#### Parallel Fan-Out

Multiple agents work simultaneously, results combined.

```python
import asyncio

async def parallel_analysis(font_file: str):
    # Run all analyses in parallel
    results = await asyncio.gather(
        metric_agent.run(font_file),
        history_agent.run(font_file),
        comparison_agent.run(font_file)
    )

    # Combine results
    return synthesis_agent.run(results)
```

**Pros**: Fast, resilient
**Cons**: Harder to coordinate, may produce conflicting results

#### Debate Pattern

Agents argue opposing positions, third agent judges.

```python
# Two agents debate, one judges
position_a = advocate_agent.run(
    "Argue why Garamond is better for body text"
)
position_b = critic_agent.run(
    f"Counter this argument: {position_a}"
)
verdict = judge_agent.run(
    f"Evaluate these positions:\n\nPro: {position_a}\n\nCon: {position_b}"
)
```

**Pros**: Better reasoning, catches errors
**Cons**: 3x the API calls, overkill for simple tasks

#### Hierarchical Delegation

Manager agent assigns tasks to worker agents.

```python
class ManagerAgent:
    def __init__(self):
        self.workers = {
            "research": ResearchAgent(),
            "code": CodeAgent(),
            "writing": WritingAgent()
        }

    async def handle(self, task: str):
        # Manager decides which worker to use
        plan = await self.plan_task(task)

        results = []
        for step in plan.steps:
            worker = self.workers[step.worker_type]
            result = await worker.run(step.instruction)
            results.append(result)

        return await self.synthesize(results)
```

**Pros**: Flexible, scalable
**Cons**: Manager becomes bottleneck, complex to debug

## Memory and state

Agents without memory repeat mistakes. Memory systems range from simple to sophisticated.

### Conversation History

The simplest form—just keep the messages.

```python
class SimpleMemory:
    def __init__(self, max_messages: int = 50):
        self.messages = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self) -> list[dict]:
        return self.messages.copy()
```

### Semantic Memory (Vector Store)

Long-term storage with retrieval.

```python
class SemanticMemory:
    def __init__(self):
        self.index = []  # Vector database

    def store(self, text: str, metadata: dict):
        embedding = embed(text)
        self.index.append({
            "embedding": embedding,
            "text": text,
            "metadata": metadata
        })

    def recall(self, query: str, k: int = 5) -> list[str]:
        query_embedding = embed(query)
        # Find most similar stored memories
        results = vector_search(self.index, query_embedding, k)
        return [r["text"] for r in results]
```

### Working Memory

Short-term task-specific state.

```python
from pydantic import BaseModel

class WorkingMemory(BaseModel):
    current_goal: str
    completed_steps: list[str] = []
    pending_actions: list[str] = []
    findings: dict = {}

    def update_progress(self, step: str, finding: str = None):
        self.completed_steps.append(step)
        if self.pending_actions and step in self.pending_actions:
            self.pending_actions.remove(step)
        if finding:
            self.findings[step] = finding
```

### Memory Architecture

For production agents:

```
┌─────────────────────────────────────────────┐
│                Working Memory               │
│  (Current task, recent actions, goals)      │
├─────────────────────────────────────────────┤
│              Episodic Memory                │
│  (Conversation history, past sessions)      │
├─────────────────────────────────────────────┤
│              Semantic Memory                │
│  (Facts, knowledge, embeddings)             │
└─────────────────────────────────────────────┘
```

## Common failure modes

Agents fail in predictable ways. Anticipate these:

### 1. Infinite Loops

Agent keeps calling the same tool without progress.

**Fix**: Add loop detection and maximum iterations.

```python
MAX_ITERATIONS = 10
seen_actions = set()

for i in range(MAX_ITERATIONS):
    action = agent.decide()
    action_key = f"{action.tool}:{hash(str(action.args))}"

    if action_key in seen_actions:
        break  # Stuck in loop
    seen_actions.add(action_key)
```

### 2. Tool Misuse

Agent calls tools with wrong arguments or in wrong order.

**Fix**: Better tool descriptions, few-shot examples.

```python
@agent.tool
async def analyze_font(path: str) -> str:
    """Analyze a font file and return its metrics.

    Args:
        path: Full path to a .ttf or .otf file (e.g., "/fonts/Arial.ttf")

    Returns:
        JSON with font metrics including family, weight, glyph count.

    Example:
        analyze_font("/Library/Fonts/Garamond.ttf")
        → {"family": "Garamond", "weight": 400, "glyphs": 247}
    """
```

### 3. Premature Termination

Agent declares task complete when it isn't.

**Fix**: Verification step before final answer.

```python
# Force agent to verify before completing
system_prompt = """Before giving your final answer:
1. Review what was asked
2. Check if you've addressed all parts
3. Verify your answer is complete and accurate
Only then provide your final response."""
```

### 4. Hallucinated Tools

Agent tries to use tools that don't exist.

**Fix**: Explicit tool listing in system prompt.

```python
tools_description = "\n".join([
    f"- {tool.name}: {tool.description}"
    for tool in available_tools
])

system_prompt = f"""You have access to these tools ONLY:
{tools_description}

Do NOT attempt to use any other tools."""
```

### 5. Context Window Overflow

Long conversations exceed model limits.

**Fix**: Summarization and memory management.

```python
def manage_context(messages: list[dict], max_tokens: int = 8000) -> list[dict]:
    total_tokens = count_tokens(messages)

    if total_tokens > max_tokens:
        # Summarize older messages
        old_messages = messages[:-10]
        recent_messages = messages[-10:]

        summary = summarize(old_messages)
        return [{"role": "system", "content": f"Previous context: {summary}"}] + recent_messages

    return messages
```

## Debugging agents in practice

When agents misbehave, you need visibility. Here's how to instrument them.

### Logging every step

```python
import logging
from pydantic_ai import Agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agent")

agent = Agent("claude-sonnet-4", system_prompt="You analyze fonts.")

@agent.tool
async def analyze_font(path: str) -> str:
    """Analyze a font file."""
    logger.info(f"Tool called: analyze_font({path})")
    try:
        result = do_analysis(path)
        logger.info(f"Tool result: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Tool failed: {e}")
        raise

async def run_with_logging(query: str):
    logger.info(f"Starting: {query}")
    result = await agent.run(query)
    logger.info(f"Final output: {result.output}")
    return result
```

### Inspecting message history

PydanticAI exposes the full conversation:

```python
result = await agent.run("Analyze Garamond.ttf")

# See every message exchanged
for msg in result.all_messages():
    print(f"{msg.kind}: {msg}")

# Check which tools were called
for msg in result.all_messages():
    if hasattr(msg, 'parts'):
        for part in msg.parts:
            if hasattr(part, 'tool_name'):
                print(f"Called: {part.tool_name}({part.args})")
```

### Cost tracking

Every run returns usage statistics:

```python
result = await agent.run("Compare Arial and Helvetica")

usage = result.usage()
print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Requests: {usage.requests}")

# Estimate cost (Claude Sonnet pricing)
input_cost = usage.input_tokens * 0.003 / 1000
output_cost = usage.output_tokens * 0.015 / 1000
print(f"Estimated cost: ${input_cost + output_cost:.4f}")
```

### Replay and reproduce

Save message history to reproduce issues:

```python
import json

# After a problematic run
result = await agent.run("Problematic query here")

# Save for debugging
with open("debug_messages.json", "w") as f:
    messages = [msg.model_dump() for msg in result.all_messages()]
    json.dump(messages, f, indent=2, default=str)

# Later, inspect what happened
with open("debug_messages.json") as f:
    messages = json.load(f)
    for msg in messages:
        print(f"{msg['kind']}: {msg.get('content', msg.get('parts', ''))[:200]}")
```

### The debugging checklist

When an agent fails:

1. **Check the prompt** — Did the system prompt set clear expectations?
2. **Check tool descriptions** — Are they unambiguous? Do they include examples?
3. **Check tool outputs** — Did tools return what the agent expected?
4. **Check iteration count** — Did it hit the loop limit?
5. **Check token usage** — Did context overflow?
6. **Check the model** — Try a more capable model to see if the task is too hard

Most agent bugs come from unclear tool descriptions or insufficient context. Fix those first.

## The minimal agent template

Here's a production-ready starting point:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Any
import asyncio

class AgentResult(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

class MinimalAgent:
    def __init__(self, model: str = "claude-sonnet-4"):
        self.agent = Agent(
            model=model,
            result_type=AgentResult,
            system_prompt=self._build_system_prompt()
        )
        self.memory = []
        self.max_iterations = 10

    def _build_system_prompt(self) -> str:
        return """You are a helpful assistant with access to tools.

Before each action, explain your reasoning.
After receiving results, evaluate if you have enough information.
Only provide a final answer when confident.

If stuck, try a different approach rather than repeating failed actions."""

    @property
    def tools(self):
        return self.agent.tools

    def add_tool(self, func):
        """Decorator to add tools."""
        return self.agent.tool(func)

    async def run(self, query: str) -> AgentResult:
        """Execute agent with query."""
        self.memory.append({"role": "user", "content": query})

        try:
            result = await self.agent.run(query)
            self.memory.append({"role": "assistant", "content": result.output.answer})
            return result.output
        except Exception as e:
            return AgentResult(
                answer=f"Error: {str(e)}",
                confidence=0.0,
                sources=[]
            )

# Usage
agent = MinimalAgent()

@agent.add_tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    pass

@agent.add_tool
async def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path) as f:
        return f.read()

result = asyncio.run(agent.run("What fonts work best for dyslexic readers?"))
```

---

> **Integration sidebar: Agents as orchestrators**
>
> Agents shine when they coordinate other capabilities from this book:
>
> - **Agents + RAG (Ch 7)**: The agentic RAG pattern lets agents decide when to search, what queries to run, and whether results are sufficient. The agent iterates until it finds what it needs.
> - **Agents + MCP (Ch 6)**: MCP servers become agent tools automatically. A font analysis agent can call your custom FontLab MCP server to read glyph data, calculate metrics, or generate specimens.
> - **Agents + CLI (Ch 5)**: Claude Code is itself an agent with shell access. Chain it with other CLI tools—pipe outputs through `jq`, use `git` for version control, run tests automatically.
> - **Agents + Multi-Agent (Ch 9B)**: When single agents hit limits, orchestrate teams. A research agent, a coding agent, and a review agent can collaborate on complex projects.
>
> The pattern: agents add autonomy to any capability. Instead of you running RAG queries, the agent runs them. Instead of you calling tools, the agent decides which tools to call. The tradeoff is control for capability.

---

## Getting started: two approaches

**Simple approach**: One agent, one tool, one task.

```python
from pydantic_ai import Agent

agent = Agent(
    model="claude-sonnet-4-20250514",
    system_prompt="You analyze fonts and suggest alternatives."
)

@agent.tool
async def search_fonts(query: str) -> str:
    """Search for fonts matching a description."""
    # In reality, this calls Google Fonts API or a database
    return f"Found fonts matching '{query}': Garamond, Georgia, Palatino"

result = await agent.run("Find serif fonts good for book body text")
print(result.output)
```

Single agent, single tool, immediate result. This handles most real-world use cases.

**Production approach**: Multi-tool agent with structured output and error recovery.

```python
# typography_agent.py - Production agent for font analysis
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from fontTools.ttLib import TTFont
from pathlib import Path
import asyncio

class FontRecommendation(BaseModel):
    """Structured font recommendation with rationale."""
    primary_font: str = Field(description="Recommended font family")
    alternatives: list[str] = Field(description="Fallback options")
    rationale: str = Field(description="Why this font fits the use case")
    warnings: list[str] = Field(default_factory=list, description="Potential issues")

class FontMetrics(BaseModel):
    """Extracted font technical data."""
    family: str
    weight: int
    glyph_count: int
    has_kerning: bool
    x_height_ratio: float

class TypographyAgent:
    """Production agent for typography analysis and recommendations."""

    def __init__(self, font_dir: str = "/Library/Fonts"):
        self.font_dir = Path(font_dir)
        self.agent = Agent(
            model="claude-sonnet-4-20250514",
            result_type=FontRecommendation,
            system_prompt=self._system_prompt()
        )
        self._register_tools()

    def _system_prompt(self) -> str:
        return """You are a typography expert helping designers choose fonts.

Before recommending:
1. Analyze the use case requirements
2. Search available fonts matching criteria
3. Compare metrics of candidates
4. Consider readability, aesthetics, and practical constraints

Provide structured recommendations with clear rationale."""

    def _register_tools(self):
        @self.agent.tool
        async def list_fonts(pattern: str = "*.[ot]tf") -> str:
            """List available font files."""
            fonts = list(self.font_dir.glob(pattern))
            return f"Found {len(fonts)} fonts: {[f.stem for f in fonts[:20]]}"

        @self.agent.tool
        async def analyze_font(font_name: str) -> str:
            """Extract metrics from a font file."""
            matches = list(self.font_dir.glob(f"{font_name}*.[ot]tf"))
            if not matches:
                return f"Font not found: {font_name}"

            font = TTFont(matches[0])
            metrics = FontMetrics(
                family=font_name,
                weight=font.get('OS/2', {}).usWeightClass if 'OS/2' in font else 400,
                glyph_count=font['maxp'].numGlyphs,
                has_kerning='kern' in font or 'GPOS' in font,
                x_height_ratio=self._calc_x_height(font)
            )
            return metrics.model_dump_json()

        @self.agent.tool
        async def compare_fonts(font_a: str, font_b: str) -> str:
            """Compare two fonts side by side."""
            analysis_a = await analyze_font(font_a)
            analysis_b = await analyze_font(font_b)
            return f"Comparison:\n{font_a}: {analysis_a}\n{font_b}: {analysis_b}"

    def _calc_x_height(self, font: TTFont) -> float:
        """Calculate x-height to cap-height ratio."""
        try:
            os2 = font['OS/2']
            if os2.sxHeight and os2.sCapHeight:
                return os2.sxHeight / os2.sCapHeight
        except:
            pass
        return 0.5  # Default assumption

    async def recommend(self, use_case: str) -> FontRecommendation:
        """Get font recommendation for a use case."""
        result = await self.agent.run(
            f"Recommend fonts for: {use_case}"
        )
        return result.data

# Usage
async def main():
    agent = TypographyAgent()

    # Get recommendation with full agent reasoning
    rec = await agent.recommend(
        "Body text for a literary fiction novel, approximately 80,000 words, "
        "targeting readers who prefer traditional book typography"
    )

    print(f"Primary: {rec.primary_font}")
    print(f"Alternatives: {', '.join(rec.alternatives)}")
    print(f"Rationale: {rec.rationale}")
    if rec.warnings:
        print(f"Warnings: {rec.warnings}")

asyncio.run(main())
```

The simple approach answers questions. The production approach solves problems—it gathers data, reasons over evidence, handles edge cases, and returns structured results your code can act on. Start simple; add tools and structure as your use case demands it.

## The takeaway

Agents are powerful but not magic. They're LLMs with tools and loops.

**Start simple:**
1. Single agent with 2-3 tools
2. PydanticAI for type safety
3. ReAct pattern for reasoning
4. Explicit iteration limits

**Scale up when needed:**
- Multi-agent for complex workflows
- Sandboxing for code execution
- Memory systems for long tasks

**Watch for:**
- Infinite loops
- Tool misuse
- Context overflow
- Premature completion

The best agents are boring. They do one thing well, handle errors gracefully, and stop when they should. Save the complex orchestration for when simple agents prove insufficient.

![A robotic arm holding a magnifying glass over a specimen of type, with thought bubbles showing reasoning steps, technical illustration style](https://pixy.vexy.art/)

---

*Next: Part IV begins with case studies—real projects from concept to deployment.*
