# Chapter 4B: Package integration cookbook

*When one library isn't enough.*

> "The whole is greater than the sum of its parts."
> — Aristotle

## The integration problem

Every package in Chapter 4 solves one problem well. But real projects need multiple solutions working together. LiteLLM handles provider switching. Instructor handles structured output. PydanticAI handles agents. Qdrant handles vectors.

The question isn't which to use—it's how to combine them.

This chapter provides tested recipes for common integration patterns. Copy, adapt, deploy.

## Imagine...

Imagine you're building a house. You've got excellent plumbing fixtures (LiteLLM), beautiful tiles (Instructor), smart thermostats (PydanticAI), and a solid foundation (Qdrant). Each component is top-quality. But someone still needs to connect the pipes, lay the tiles, wire the thermostat, and pour the concrete in the right order.

That's integration work. Not glamorous, but without it you've got a pile of parts, not a house.

---

## Recipe 1: LiteLLM + Instructor

**Use case**: Unified provider API with guaranteed structured output.

**The problem**: You want to switch between GPT-4, Claude, and Gemini without code changes. You also need responses in a specific JSON schema, every time.

**The solution**: LiteLLM handles the provider abstraction. Instructor patches the client for structured output.

```python
from litellm import completion
import instructor
from pydantic import BaseModel

# Patch LiteLLM with Instructor
client = instructor.from_litellm(completion)

class FontAnalysis(BaseModel):
    family_name: str
    classification: str  # serif, sans-serif, etc.
    weight: int
    recommended_uses: list[str]
    similar_fonts: list[str]

def analyze_font(description: str, model: str = "gpt-5") -> FontAnalysis:
    """Analyze a font description and return structured data."""
    return client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Analyze this font: {description}"
        }],
        response_model=FontAnalysis
    )

# Same code works across providers
result_openai = analyze_font("Helvetica Neue Light", model="gpt-5")
result_claude = analyze_font("Helvetica Neue Light", model="claude-sonnet-4-5")
result_gemini = analyze_font("Helvetica Neue Light", model="gemini/gemini-3-pro")

# All return FontAnalysis objects with identical structure
print(result_openai.classification)  # "sans-serif"
```

**Why this works**: Instructor's patching mechanism is provider-agnostic. LiteLLM normalizes the API. The combination gives you structured output from any model.

**Gotchas**:
- Some models handle schema constraints better than others. Claude excels; Gemini sometimes struggles with nested objects.
- Cost varies wildly. Same prompt, same output, 10x price difference.

---

## Recipe 2: PydanticAI + LangGraph

**Use case**: Type-safe agents with complex state machines.

**The problem**: PydanticAI gives you clean, typed agents. But some workflows need conditional branching, loops, and explicit state management that simple ReAct can't handle.

**The solution**: Use PydanticAI for individual agent steps. Use LangGraph to orchestrate the flow.

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# Define state
class WorkflowState(TypedDict):
    query: str
    research: str | None
    analysis: str | None
    needs_more_research: bool
    final_answer: str | None

# Create specialized agents
researcher = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You research topics thoroughly. Return key findings."
)

analyst = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You analyze research and draw conclusions."
)

reviewer = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You review analysis quality. Return 'sufficient' or 'needs_more'."
)

# Define nodes
async def research_node(state: WorkflowState) -> WorkflowState:
    result = await researcher.run(state["query"])
    return {"research": result.output}

async def analyze_node(state: WorkflowState) -> WorkflowState:
    prompt = f"Analyze this research:\n{state['research']}"
    result = await analyst.run(prompt)
    return {"analysis": result.output}

async def review_node(state: WorkflowState) -> WorkflowState:
    prompt = f"Is this analysis sufficient?\n{state['analysis']}"
    result = await reviewer.run(prompt)
    needs_more = "needs_more" in result.output.lower()
    return {"needs_more_research": needs_more}

async def synthesize_node(state: WorkflowState) -> WorkflowState:
    return {"final_answer": state["analysis"]}

# Define routing
def should_continue(state: WorkflowState) -> Literal["research", "synthesize"]:
    if state.get("needs_more_research"):
        return "research"
    return "synthesize"

# Build graph
workflow = StateGraph(WorkflowState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("review", review_node)
workflow.add_node("synthesize", synthesize_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "review")
workflow.add_conditional_edges("review", should_continue)
workflow.add_edge("synthesize", END)

app = workflow.compile()

# Run
result = await app.ainvoke({"query": "What fonts work best for dyslexic readers?"})
print(result["final_answer"])
```

**Why this works**: PydanticAI handles the LLM calls with type safety. LangGraph handles the workflow logic with explicit state. Each does what it's good at.

**Gotchas**:
- LangGraph adds complexity. Don't use it for linear workflows.
- Debug by printing state at each node—LangGraph's error messages are cryptic.

---

## Recipe 3: CrewAI + E2B

**Use case**: Multi-agent teams with sandboxed code execution.

**The problem**: CrewAI agents can write code, but executing it on your machine is dangerous. One bad `import os; os.system('rm -rf /')` and you're done.

**The solution**: Route all code execution through E2B sandboxes.

```python
from crewai import Agent, Task, Crew
from crewai_tools import tool
from e2b_code_interpreter import Sandbox

# Create sandboxed code execution tool
@tool("execute_python")
def execute_python(code: str) -> str:
    """Execute Python code in a secure sandbox. Returns output or error."""
    with Sandbox() as sandbox:
        execution = sandbox.run_code(code)

        if execution.error:
            return f"Error: {execution.error}"

        output = []
        for result in execution.results:
            if result.text:
                output.append(result.text)
            if result.png:
                output.append("[Generated image]")

        return "\n".join(output) if output else "Code executed successfully (no output)"

# Create agents
data_scientist = Agent(
    role="Data Scientist",
    goal="Analyze data and create visualizations",
    backstory="Expert in Python, pandas, and matplotlib.",
    tools=[execute_python],
    llm="gpt-5"
)

code_reviewer = Agent(
    role="Code Reviewer",
    goal="Review code for correctness and security",
    backstory="Senior engineer focused on code quality.",
    llm="gpt-5"
)

# Define tasks
analysis_task = Task(
    description="""Analyze font usage data and create a visualization.

    Data: {font_data}

    Requirements:
    1. Load data into pandas
    2. Calculate usage statistics
    3. Create a bar chart of top 10 fonts
    4. Return summary statistics""",
    agent=data_scientist,
    expected_output="Analysis summary with statistics"
)

review_task = Task(
    description="Review the data scientist's code for correctness",
    agent=code_reviewer,
    expected_output="Code review feedback"
)

# Run crew
crew = Crew(
    agents=[data_scientist, code_reviewer],
    tasks=[analysis_task, review_task],
    verbose=True
)

result = crew.kickoff(inputs={
    "font_data": '{"fonts": ["Arial", "Helvetica", "Times"], "usage": [45, 30, 25]}'
})
```

**Why this works**: E2B sandboxes are isolated containers. Code can't escape. Even `os.system('rm -rf /')` only destroys the sandbox, not your machine.

**Gotchas**:
- E2B has a free tier (200 hours/month) but sandbox startup adds ~2s latency.
- File persistence requires explicit handling—sandboxes are ephemeral.

---

## Recipe 4: RAGatouille + Qdrant

**Use case**: Hybrid search combining ColBERT (token-level) with Qdrant (dense vectors).

**The problem**: Dense embeddings (Qdrant) are good for semantic similarity but miss exact keywords. Sparse methods (BM25) find keywords but miss synonyms. ColBERT (RAGatouille) offers a middle ground with late interaction.

**The solution**: Query both systems, merge results.

```python
from ragatouille import RAGPretrainedModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import google.generativeai as genai

# Initialize systems
colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
qdrant = QdrantClient(":memory:")
genai.configure(api_key="your-key")

# Sample documents
documents = [
    "Garamond is a classic serif typeface designed in the 16th century.",
    "Helvetica is a neo-grotesque sans-serif developed in 1957.",
    "Comic Sans was designed by Vincent Connare in 1994.",
    "Fira Code is a monospace font with programming ligatures.",
]

# Index in ColBERT
colbert.index(
    collection="fonts",
    documents=documents,
    index_name="font_index"
)

# Index in Qdrant (dense embeddings)
qdrant.create_collection(
    collection_name="fonts",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

def embed(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

for i, doc in enumerate(documents):
    qdrant.upsert(
        collection_name="fonts",
        points=[PointStruct(id=i, vector=embed(doc), payload={"text": doc})]
    )

def hybrid_search(query: str, k: int = 5) -> list[str]:
    """Combine ColBERT and Qdrant results."""

    # ColBERT search (token-level matching)
    colbert_results = colbert.search(query=query, k=k)
    colbert_docs = {r["content"]: r["score"] for r in colbert_results}

    # Qdrant search (dense vector)
    qdrant_results = qdrant.search(
        collection_name="fonts",
        query_vector=embed(query),
        limit=k
    )
    qdrant_docs = {r.payload["text"]: r.score for r in qdrant_results}

    # Reciprocal Rank Fusion
    all_docs = set(colbert_docs.keys()) | set(qdrant_docs.keys())
    fused_scores = {}

    for doc in all_docs:
        score = 0
        if doc in colbert_docs:
            rank = list(colbert_docs.keys()).index(doc) + 1
            score += 1 / (60 + rank)  # RRF constant = 60
        if doc in qdrant_docs:
            rank = list(qdrant_docs.keys()).index(doc) + 1
            score += 1 / (60 + rank)
        fused_scores[doc] = score

    # Sort by fused score
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]

# Query
results = hybrid_search("programming font with ligatures")
# Returns: ["Fira Code is a monospace font with programming ligatures.", ...]
```

**Why this works**: ColBERT's token-level matching catches exact terms ("ligatures"). Qdrant's dense vectors catch semantic similarity ("programming" ≈ "code"). RRF combines rankings without needing score normalization.

**Gotchas**:
- ColBERT indexing is slow (minutes for large collections).
- Memory usage is higher than dense-only approaches.

---

## Recipe 5: LiteLLM + Promptfoo

**Use case**: Testing prompts across multiple providers automatically.

**The problem**: You've written a prompt that works great on GPT-5. Does it work on Claude? Gemini? Llama? Testing manually is tedious.

**The solution**: Use LiteLLM as the model backend, Promptfoo as the test harness.

```yaml
# promptfoo.yaml
providers:
  - id: litellm:gpt-5
  - id: litellm:claude-sonnet-4-5
  - id: litellm:gemini/gemini-3-pro
  - id: litellm:groq/llama-3.1-70b-versatile

prompts:
  - |
    Analyze this font and return JSON with these fields:
    - name: string
    - classification: "serif" | "sans-serif" | "monospace" | "display"
    - era: string (decade, e.g., "1950s")

    Font: {{font_description}}

tests:
  - vars:
      font_description: "Helvetica, a clean sans-serif from Switzerland"
    assert:
      - type: is-json
      - type: javascript
        value: output.classification === "sans-serif"
      - type: javascript
        value: output.era.includes("1950") || output.era.includes("1957")

  - vars:
      font_description: "Garamond, elegant French typeface from the Renaissance"
    assert:
      - type: is-json
      - type: javascript
        value: output.classification === "serif"

  - vars:
      font_description: "Fira Code, modern coding font with ligatures"
    assert:
      - type: is-json
      - type: javascript
        value: output.classification === "monospace"
```

```bash
# Run tests
promptfoo eval

# Output:
# ┌─────────────────────────────┬────────┬────────┬────────┬────────┐
# │ Test                        │ GPT-5  │ Claude │ Gemini │ Llama  │
# ├─────────────────────────────┼────────┼────────┼────────┼────────┤
# │ Helvetica classification    │ PASS   │ PASS   │ PASS   │ PASS   │
# │ Garamond classification     │ PASS   │ PASS   │ PASS   │ FAIL   │
# │ Fira Code classification    │ PASS   │ PASS   │ FAIL   │ PASS   │
# └─────────────────────────────┴────────┴────────┴────────┴────────┘
```

**Why this works**: Promptfoo handles test orchestration. LiteLLM handles provider switching. You get a matrix of results showing which models pass which tests.

**Gotchas**:
- API costs add up fast when testing across 4+ providers.
- Flaky tests happen—LLMs aren't deterministic. Run multiple times.

---

## Recipe 6: mem0 + PydanticAI

**Use case**: Agents with persistent memory across sessions.

**The problem**: PydanticAI agents forget everything when the conversation ends. Users have to repeat context every time.

**The solution**: mem0 provides a memory layer that persists facts across sessions.

```python
from mem0 import Memory
from pydantic_ai import Agent

# Initialize memory
memory = Memory()

# Create agent with memory context
agent = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="""You are a typography assistant.
    You remember user preferences and past conversations.

    Relevant memories:
    {memories}"""
)

async def chat_with_memory(user_id: str, message: str) -> str:
    # Retrieve relevant memories
    relevant = memory.search(message, user_id=user_id)
    memory_context = "\n".join([m["memory"] for m in relevant]) if relevant else "No memories yet."

    # Run agent with memory context
    result = await agent.run(
        message,
        deps={"memories": memory_context}
    )

    # Store new memories from conversation
    memory.add(
        f"User said: {message}\nAssistant replied: {result.output}",
        user_id=user_id
    )

    return result.output

# Session 1
await chat_with_memory("user123", "I prefer sans-serif fonts for body text")
# "I'll remember that you prefer sans-serif fonts for body text..."

# Session 2 (days later)
await chat_with_memory("user123", "What font should I use for my resume?")
# "Based on your preference for sans-serif fonts, I'd recommend..."
```

**Why this works**: mem0 extracts and stores facts automatically. When you search, it retrieves relevant memories by semantic similarity. The agent gets context without explicit session management.

**Gotchas**:
- Memory quality depends on extraction. Sometimes mem0 stores irrelevant details.
- Storage grows unbounded. Implement cleanup for long-running applications.

---

## Integration decision matrix

| Combination | Best For | Complexity | Cost Impact |
|-------------|----------|------------|-------------|
| LiteLLM + Instructor | Multi-provider structured output | Low | Neutral |
| PydanticAI + LangGraph | Complex agent workflows | High | +20% (more calls) |
| CrewAI + E2B | Safe code execution | Medium | +$0.01/sandbox |
| RAGatouille + Qdrant | Hybrid search precision | High | +50% indexing |
| LiteLLM + Promptfoo | Prompt testing | Low | +300% testing |
| mem0 + PydanticAI | Persistent memory | Medium | +storage costs |

## The integration philosophy

Integration isn't about using more packages—it's about using the right packages together. Every integration adds:

- **Complexity**: More moving parts, more failure modes
- **Dependencies**: More packages to update, more breaking changes
- **Cognitive load**: More APIs to remember

Only integrate when:

1. **Single packages can't solve the problem** - Don't add LangGraph for linear workflows
2. **The combination is well-tested** - Prefer documented integrations over custom glue
3. **You understand both pieces** - Debugging integration bugs requires deep knowledge

The best integration is the one you don't need. Start with one package. Add another only when you hit a wall.

![A chef combining ingredients from different jars labeled with package names, creating a dish on a cutting board shaped like a code editor](https://pixy.vexy.art/)

---

*Next: Chapter 5B covers advanced CLI workflows for power users.*
