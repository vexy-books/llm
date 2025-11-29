# Chapter 9B: Multi-agent architectures

*When one agent isn't enough.*

> "The whole is greater than the sum of its parts."
> — Aristotle

## The limitations of single agents

Chapter 9 built agents that reason, use tools, and maintain memory. They work well for bounded tasks. But some problems demand more.

A single agent analyzing a complex codebase hits context limits. A single agent researching a topic misses perspectives. A single agent making decisions lacks checks on its reasoning.

Multi-agent systems address these limits by dividing work, parallelizing effort, and introducing accountability through multiple viewpoints.

This chapter covers architectures that scale beyond single-agent capabilities: hierarchical delegation, debate protocols, shared memory, and production debugging strategies.

## Imagine...

Imagine a law firm. One brilliant lawyer could handle any case, but they'd burn out. Instead, you have specialists: a researcher who finds precedents, a writer who drafts briefs, a strategist who plans arguments, a partner who reviews everything.

Each specialist excels in their domain. The researcher doesn't write briefs; the writer doesn't argue in court. Information flows between them—research informs writing, writing enables argument. The partner provides quality control, catching errors before they reach the judge.

Multi-agent systems work the same way. Each agent has a role: researcher, coder, reviewer, orchestrator. They communicate through structured protocols. No single agent needs to be brilliant at everything—the system's intelligence emerges from their collaboration.

The challenge isn't building individual agents. It's designing how they work together: who talks to whom, what information passes between them, and who decides when the work is done.

---

## Hierarchical delegation

The most common pattern: a manager assigns tasks to specialist workers.

### Architecture

```
                    ┌─────────────────┐
                    │   Orchestrator   │
                    │   (Manager)      │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Researcher  │   │   Coder      │   │  Reviewer   │
    │  Agent       │   │   Agent      │   │  Agent      │
    └─────────────┘   └─────────────┘   └─────────────┘
```

### Implementation

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Literal
import asyncio

class TaskPlan(BaseModel):
    steps: list[dict]
    dependencies: dict[str, list[str]]  # step_id → required prior steps

class WorkerResult(BaseModel):
    step_id: str
    status: Literal["success", "failure", "needs_revision"]
    output: str
    confidence: float

class OrchestratorSystem:
    def __init__(self):
        self.orchestrator = Agent(
            model="claude-sonnet-4",
            result_type=TaskPlan,
            system_prompt="""You are a project manager. Given a complex task:
1. Break it into discrete steps
2. Identify dependencies between steps
3. Assign each step to the appropriate worker type:
   - researcher: Information gathering, fact-finding
   - coder: Writing and debugging code
   - reviewer: Quality checks, error detection

Output a plan with clear steps and dependencies."""
        )

        self.workers = {
            "researcher": self._create_researcher(),
            "coder": self._create_coder(),
            "reviewer": self._create_reviewer(),
        }

    def _create_researcher(self) -> Agent:
        return Agent(
            model="claude-sonnet-4",
            result_type=WorkerResult,
            system_prompt="""You are a research specialist.
Gather information, find sources, and synthesize findings.
Be thorough but concise. Cite your reasoning."""
        )

    def _create_coder(self) -> Agent:
        return Agent(
            model="claude-sonnet-4",
            result_type=WorkerResult,
            system_prompt="""You are a coding specialist.
Write clean, tested code. Handle edge cases.
Explain your implementation choices."""
        )

    def _create_reviewer(self) -> Agent:
        return Agent(
            model="claude-sonnet-4",
            result_type=WorkerResult,
            system_prompt="""You are a code reviewer and quality checker.
Find bugs, security issues, and logic errors.
Be specific about problems and suggested fixes."""
        )

    async def execute(self, task: str) -> dict:
        # Step 1: Create plan
        plan_result = await self.orchestrator.run(
            f"Create a plan to accomplish: {task}"
        )
        plan = plan_result.data

        # Step 2: Execute steps respecting dependencies
        results = {}
        completed = set()

        while len(completed) < len(plan.steps):
            # Find steps ready to run (dependencies satisfied)
            ready_steps = [
                step for step in plan.steps
                if step["id"] not in completed
                and all(dep in completed for dep in plan.dependencies.get(step["id"], []))
            ]

            # Execute ready steps in parallel
            tasks = [
                self._execute_step(step, results)
                for step in ready_steps
            ]
            step_results = await asyncio.gather(*tasks)

            for step, result in zip(ready_steps, step_results):
                results[step["id"]] = result
                if result.status == "success":
                    completed.add(step["id"])
                elif result.status == "failure":
                    # Handle failure - could retry, escalate, or abort
                    raise Exception(f"Step {step['id']} failed: {result.output}")

        return results

    async def _execute_step(self, step: dict, context: dict) -> WorkerResult:
        worker_type = step["worker"]
        worker = self.workers[worker_type]

        # Include relevant prior results in context
        prior_context = "\n".join([
            f"Previous result ({k}): {v.output}"
            for k, v in context.items()
            if k in step.get("context_from", [])
        ])

        prompt = f"""Task: {step["instruction"]}

Prior context:
{prior_context if prior_context else "None"}

Complete this task and report your results."""

        result = await worker.run(prompt)
        return result.data

# Usage
system = OrchestratorSystem()
results = asyncio.run(system.execute(
    "Build a CLI tool that analyzes font files and generates specimen PDFs"
))
```

### When hierarchies fail

Hierarchical systems have weaknesses:

1. **Orchestrator bottleneck**: Everything flows through the manager. If the manager misunderstands the task, all workers fail.

2. **Lost context**: Workers only see what the orchestrator passes them. Important context gets lost in translation.

3. **Rigid structure**: Predefined roles can't adapt to unexpected task requirements.

**Mitigation**: Give workers the ability to request clarification from the orchestrator, and allow the orchestrator to dynamically create new worker types.

## Debate and consensus

Instead of hierarchical authority, agents argue toward truth.

### The debate pattern

Two or more agents take positions. A judge evaluates arguments.

```python
class DebateSystem:
    def __init__(self):
        self.advocate = Agent(
            model="claude-sonnet-4",
            system_prompt="""You argue FOR the given position.
Build the strongest possible case. Use evidence and logic.
Acknowledge counterarguments but explain why your position prevails."""
        )

        self.critic = Agent(
            model="claude-sonnet-4",
            system_prompt="""You argue AGAINST the given position.
Find weaknesses, counterexamples, and flaws in reasoning.
Be rigorous but fair."""
        )

        self.judge = Agent(
            model="claude-sonnet-4",
            result_type=JudgmentResult,
            system_prompt="""You evaluate debates impartially.
Consider argument strength, evidence quality, and logical validity.
Render a judgment with clear reasoning."""
        )

    async def debate(self, question: str, rounds: int = 2) -> JudgmentResult:
        advocate_args = []
        critic_args = []

        # Initial positions
        advocate_response = await self.advocate.run(
            f"Argue that the answer to '{question}' is YES."
        )
        advocate_args.append(advocate_response.data)

        critic_response = await self.critic.run(
            f"Argue that the answer to '{question}' is NO."
        )
        critic_args.append(critic_response.data)

        # Rebuttals
        for round_num in range(rounds):
            advocate_rebuttal = await self.advocate.run(
                f"Respond to this criticism: {critic_args[-1]}"
            )
            advocate_args.append(advocate_rebuttal.data)

            critic_rebuttal = await self.critic.run(
                f"Respond to this defense: {advocate_args[-1]}"
            )
            critic_args.append(critic_rebuttal.data)

        # Final judgment
        debate_transcript = self._format_transcript(advocate_args, critic_args)
        judgment = await self.judge.run(
            f"Judge this debate:\n\n{debate_transcript}"
        )

        return judgment.data

    def _format_transcript(self, advocate_args: list, critic_args: list) -> str:
        transcript = []
        for i, (adv, crit) in enumerate(zip(advocate_args, critic_args)):
            transcript.append(f"=== Round {i+1} ===")
            transcript.append(f"ADVOCATE: {adv}")
            transcript.append(f"CRITIC: {crit}")
        return "\n\n".join(transcript)

class JudgmentResult(BaseModel):
    verdict: Literal["advocate_wins", "critic_wins", "undecided"]
    confidence: float
    key_arguments: list[str]
    reasoning: str
```

### Consensus protocols

When you need agreement, not argument:

```python
class ConsensusSystem:
    def __init__(self, num_agents: int = 3):
        self.agents = [
            Agent(
                model="claude-sonnet-4",
                result_type=Opinion,
                system_prompt=f"""You are Agent {i+1} in a consensus group.
Give your honest opinion. Explain your reasoning.
You may disagree with others but must be willing to update your view."""
            )
            for i in range(num_agents)
        ]

    async def reach_consensus(
        self,
        question: str,
        max_rounds: int = 5,
        threshold: float = 0.8
    ) -> ConsensusResult:

        opinions = []

        for round_num in range(max_rounds):
            # Gather opinions
            if round_num == 0:
                # Initial opinions without others' views
                tasks = [
                    agent.run(f"What is your opinion on: {question}")
                    for agent in self.agents
                ]
            else:
                # Update opinions based on others' views
                prior_summary = self._summarize_opinions(opinions[-1])
                tasks = [
                    agent.run(
                        f"Question: {question}\n\n"
                        f"Other agents said: {prior_summary}\n\n"
                        f"Update your opinion if warranted. Explain any changes."
                    )
                    for agent in self.agents
                ]

            round_opinions = await asyncio.gather(*tasks)
            opinions.append([r.data for r in round_opinions])

            # Check for consensus
            agreement = self._measure_agreement(opinions[-1])
            if agreement >= threshold:
                return ConsensusResult(
                    achieved=True,
                    rounds=round_num + 1,
                    final_opinions=opinions[-1],
                    agreement_level=agreement
                )

        # No consensus reached
        return ConsensusResult(
            achieved=False,
            rounds=max_rounds,
            final_opinions=opinions[-1],
            agreement_level=self._measure_agreement(opinions[-1])
        )

    def _measure_agreement(self, opinions: list[Opinion]) -> float:
        # Simple approach: embed opinions and measure average similarity
        embeddings = [embed(op.position) for op in opinions]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        return sum(similarities) / len(similarities) if similarities else 0.0

class Opinion(BaseModel):
    position: str
    confidence: float
    key_reasons: list[str]
    changed_from_prior: bool = False

class ConsensusResult(BaseModel):
    achieved: bool
    rounds: int
    final_opinions: list[Opinion]
    agreement_level: float
```

### When to use debate vs consensus

| Scenario | Best Pattern |
|----------|--------------|
| Binary decision | Debate |
| Open-ended analysis | Consensus |
| Error checking | Debate |
| Creative brainstorming | Consensus |
| Risk assessment | Both (debate risks, consensus on response) |

## Shared memory architectures

Agents need shared state. How they share it matters.

### Blackboard pattern

A shared workspace that all agents read and write.

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
import asyncio

class BlackboardEntry(BaseModel):
    author: str
    timestamp: datetime
    entry_type: str
    content: Any
    confidence: float

class Blackboard:
    def __init__(self):
        self.entries: list[BlackboardEntry] = []
        self.lock = asyncio.Lock()

    async def write(
        self,
        author: str,
        entry_type: str,
        content: Any,
        confidence: float = 1.0
    ):
        async with self.lock:
            self.entries.append(BlackboardEntry(
                author=author,
                timestamp=datetime.now(),
                entry_type=entry_type,
                content=content,
                confidence=confidence
            ))

    def read(
        self,
        entry_types: list[str] = None,
        authors: list[str] = None,
        min_confidence: float = 0.0
    ) -> list[BlackboardEntry]:
        results = self.entries

        if entry_types:
            results = [e for e in results if e.entry_type in entry_types]
        if authors:
            results = [e for e in results if e.author in authors]
        if min_confidence > 0:
            results = [e for e in results if e.confidence >= min_confidence]

        return results

    def get_latest(self, entry_type: str) -> BlackboardEntry | None:
        matching = [e for e in self.entries if e.entry_type == entry_type]
        return matching[-1] if matching else None

class BlackboardAgent:
    def __init__(self, name: str, blackboard: Blackboard, agent: Agent):
        self.name = name
        self.blackboard = blackboard
        self.agent = agent

    async def work(self, task_prompt: str):
        # Read current state from blackboard
        context = self._build_context()

        # Execute task
        result = await self.agent.run(
            f"{task_prompt}\n\nCurrent shared context:\n{context}"
        )

        # Write results back
        await self.blackboard.write(
            author=self.name,
            entry_type=result.data.entry_type,
            content=result.data.content,
            confidence=result.data.confidence
        )

        return result.data

    def _build_context(self) -> str:
        recent = self.blackboard.entries[-20:]  # Last 20 entries
        return "\n".join([
            f"[{e.author}] {e.entry_type}: {e.content}"
            for e in recent
        ])

# Usage
blackboard = Blackboard()

researcher = BlackboardAgent(
    name="researcher",
    blackboard=blackboard,
    agent=Agent(model="claude-sonnet-4", ...)
)

coder = BlackboardAgent(
    name="coder",
    blackboard=blackboard,
    agent=Agent(model="claude-sonnet-4", ...)
)

# Agents work concurrently, sharing state via blackboard
await asyncio.gather(
    researcher.work("Research best practices for font loading"),
    coder.work("Write font loading code based on shared research")
)
```

### Event-driven communication

Agents react to events rather than polling shared state.

```python
from asyncio import Queue
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class Event:
    type: str
    source: str
    payload: Any
    timestamp: datetime = Field(default_factory=datetime.now)

class EventBus:
    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = {}
        self.event_log: list[Event] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], Awaitable]):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event: Event):
        self.event_log.append(event)

        handlers = self.subscribers.get(event.type, [])
        handlers.extend(self.subscribers.get("*", []))  # Wildcard subscribers

        await asyncio.gather(*[
            handler(event) for handler in handlers
        ])

class EventDrivenAgent:
    def __init__(self, name: str, bus: EventBus, agent: Agent):
        self.name = name
        self.bus = bus
        self.agent = agent
        self.inbox = Queue()

    def listen_to(self, event_types: list[str]):
        for event_type in event_types:
            self.bus.subscribe(event_type, self._handle_event)

    async def _handle_event(self, event: Event):
        await self.inbox.put(event)

    async def run(self):
        while True:
            event = await self.inbox.get()

            # Let agent decide how to respond
            response = await self.agent.run(
                f"Event received: {event.type} from {event.source}\n"
                f"Payload: {event.payload}\n\n"
                f"How should you respond?"
            )

            # Publish response if agent produces one
            if response.data.should_publish:
                await self.bus.publish(Event(
                    type=response.data.event_type,
                    source=self.name,
                    payload=response.data.payload
                ))

# Usage
bus = EventBus()

researcher = EventDrivenAgent("researcher", bus, ...)
coder = EventDrivenAgent("coder", bus, ...)
reviewer = EventDrivenAgent("reviewer", bus, ...)

# Set up subscriptions
researcher.listen_to(["task_assigned", "clarification_needed"])
coder.listen_to(["research_complete", "review_feedback"])
reviewer.listen_to(["code_ready", "revision_complete"])

# Start agents
await asyncio.gather(
    researcher.run(),
    coder.run(),
    reviewer.run()
)

# Kick off workflow
await bus.publish(Event(
    type="task_assigned",
    source="orchestrator",
    payload={"task": "Build font analyzer CLI"}
))
```

### Memory sharing patterns

| Pattern | Best For | Drawbacks |
|---------|----------|-----------|
| Blackboard | Collaborative writing, research | Polling overhead, stale reads |
| Event bus | Reactive workflows, loose coupling | Complexity, debugging difficulty |
| Shared embedding store | Semantic memory, RAG | Storage costs, embedding drift |
| Replicated state | Fault tolerance, consistency | Synchronization complexity |

## Production debugging

Multi-agent systems fail in ways single agents don't. Debugging requires special approaches.

### Tracing and observability

Track every agent interaction:

```python
import uuid
from datetime import datetime
from typing import Optional
import json

class Trace:
    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.spans: list[Span] = []

    def start_span(
        self,
        name: str,
        agent: str,
        parent_id: Optional[str] = None
    ) -> "Span":
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=self.trace_id,
            parent_id=parent_id,
            name=name,
            agent=agent,
            start_time=datetime.now()
        )
        self.spans.append(span)
        return span

    def to_json(self) -> str:
        return json.dumps({
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans]
        }, indent=2, default=str)

class Span:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.events: list[dict] = []
        self.end_time: Optional[datetime] = None

    def log(self, event_type: str, data: dict):
        self.events.append({
            "type": event_type,
            "timestamp": datetime.now(),
            "data": data
        })

    def end(self, status: str = "ok", result: Any = None):
        self.end_time = datetime.now()
        self.status = status
        self.result = result

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "agent": self.agent,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time).total_seconds() * 1000 if self.end_time else None,
            "events": self.events,
            "status": getattr(self, "status", None),
            "result": getattr(self, "result", None)
        }

class TracedAgent:
    def __init__(self, name: str, agent: Agent, trace: Trace):
        self.name = name
        self.agent = agent
        self.trace = trace
        self.current_span: Optional[Span] = None

    async def run(self, prompt: str, parent_span_id: str = None) -> Any:
        span = self.trace.start_span(
            name=f"{self.name}.run",
            agent=self.name,
            parent_id=parent_span_id
        )

        span.log("input", {"prompt": prompt[:500]})  # Truncate for logs

        try:
            result = await self.agent.run(prompt)
            span.log("output", {"result": str(result.data)[:500]})
            span.end(status="ok", result=result.data)
            return result.data
        except Exception as e:
            span.log("error", {"exception": str(e)})
            span.end(status="error")
            raise
```

### Debugging strategies

**1. Replay failed traces**

Store traces and replay them with modified agents:

```python
class TraceReplayer:
    def __init__(self, trace: Trace):
        self.trace = trace
        self.replay_index = 0

    async def replay_with_modifications(
        self,
        agents: dict[str, Agent],
        modifications: dict[str, str]  # agent_name → new system prompt
    ):
        # Apply modifications
        for agent_name, new_prompt in modifications.items():
            if agent_name in agents:
                agents[agent_name].system_prompt = new_prompt

        # Replay spans in order
        results = []
        for span in self.trace.spans:
            agent = agents.get(span.agent)
            if not agent:
                continue

            # Find original input
            input_event = next(
                (e for e in span.events if e["type"] == "input"),
                None
            )
            if input_event:
                result = await agent.run(input_event["data"]["prompt"])
                results.append({
                    "span_id": span.span_id,
                    "original_result": span.result,
                    "replay_result": result
                })

        return results
```

**2. Bisect failures**

When a complex workflow fails, find the failing step:

```python
async def bisect_failure(
    workflow: list[Callable],
    input_data: Any
) -> int:
    """Find the first failing step using binary search."""

    left, right = 0, len(workflow) - 1

    while left < right:
        mid = (left + right) // 2

        # Run workflow up to mid point
        try:
            result = input_data
            for i in range(mid + 1):
                result = await workflow[i](result)
            left = mid + 1
        except Exception:
            right = mid

    return left
```

**3. Inject test doubles**

Replace agents with deterministic mocks:

```python
class MockAgent:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.call_log = []

    async def run(self, prompt: str):
        self.call_log.append(prompt)

        # Find matching response
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return MockResult(response)

        return MockResult("No matching mock response")

# Use in tests
mock_researcher = MockAgent({
    "research font": "Found 3 relevant sources...",
    "best practices": "Key practices: 1) Use woff2..."
})

system = MultiAgentSystem()
system.workers["researcher"] = mock_researcher
await system.execute("Build font analyzer")

# Verify interactions
assert len(mock_researcher.call_log) == 2
```

### Common multi-agent bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Deadlock | Agents waiting for each other | Add timeouts, detect cycles |
| Lost messages | Work never completes | Add acknowledgments, retry logic |
| State corruption | Inconsistent results | Use locks, atomic operations |
| Cascade failure | One agent error kills system | Add circuit breakers, fallbacks |
| Context drift | Later agents miss early context | Include summaries, refresh context |

## Cost management

Multi-agent systems multiply API costs.

### Cost tracking

```python
class CostTracker:
    PRICING = {
        "claude-sonnet-4": {"input": 3.0, "output": 15.0},  # per 1M tokens
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self):
        self.costs: dict[str, float] = {}

    def log(self, agent_name: str, model: str, input_tokens: int, output_tokens: int):
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (
            input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000
        )

        if agent_name not in self.costs:
            self.costs[agent_name] = 0
        self.costs[agent_name] += cost

    def report(self) -> dict:
        return {
            "by_agent": self.costs,
            "total": sum(self.costs.values())
        }
```

### Cost optimization strategies

1. **Use cheap models for simple agents**: Researchers and formatters don't need Claude Sonnet.

2. **Cache agent results**: Same input → same output. Don't recompute.

3. **Limit debate rounds**: Diminishing returns after 2-3 rounds.

4. **Prune early**: If an agent signals low confidence, abort before wasting tokens on downstream agents.

```python
class CostAwareOrchestrator:
    MAX_BUDGET = 0.50  # $0.50 per task

    async def execute(self, task: str, budget: float = None):
        budget = budget or self.MAX_BUDGET
        tracker = CostTracker()

        for step in self.plan(task):
            # Check budget before each step
            if tracker.report()["total"] >= budget:
                return PartialResult(
                    completed_steps=tracker.costs.keys(),
                    reason="budget_exceeded"
                )

            result = await self.run_step(step, tracker)

            # Early exit on low confidence
            if result.confidence < 0.5:
                return PartialResult(
                    completed_steps=tracker.costs.keys(),
                    reason="low_confidence"
                )

        return FullResult(tracker.report())
```

## The takeaway

Multi-agent systems trade simplicity for capability. Use them when:

- Single agents hit context limits
- Tasks require multiple specialties
- Error checking benefits from multiple perspectives
- Parallelization provides real speedup

Avoid them when:

- A single agent can handle the task
- Latency matters more than thoroughness
- Cost constraints are tight
- Debugging complexity is unacceptable

**Start with hierarchical delegation**—it's the most intuitive. Add debate for critical decisions. Use shared memory when agents need to collaborate closely.

And always: trace everything, test with mocks, and watch your costs. Multi-agent systems amplify both capabilities and failure modes. Build observability from day one.

![Multiple robot arms working together on a complex clockwork mechanism, each arm specializing in different components, technical blueprint style](https://pixy.vexy.art/)

---

*Next: Part IV's case studies show these patterns in action—real projects from concept to deployment.*
