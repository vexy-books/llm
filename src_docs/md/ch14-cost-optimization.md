# Chapter 14: Cost optimization

*Getting the same results for less.*

> "Beware of little expenses; a small leak will sink a great ship."
> — Benjamin Franklin

## The cost problem

LLM costs scale with usage. A proof-of-concept that costs $5/month can become $5,000/month in production. Most teams discover this too late.

This chapter covers practical strategies to reduce costs without sacrificing quality. We'll cover model selection, caching, prompt optimization, and architectural patterns that keep bills manageable.

> **2025 Update**: API pricing continues to evolve. Claude 3.5 Haiku runs at $0.80/$4.00 per 1M tokens (input/output), while Claude 4.5 Sonnet is $3/$15. GPT-4o Mini at $0.60/$2.40 per 1M remains excellent value. Gemini 2.5 Flash is now cheapest at $0.15/$3.50 per 1M tokens. Batch processing discounts (50% off at Anthropic and OpenAI) make non-urgent workloads much cheaper. The optimization strategies below remain valid.

## Imagine...

Imagine a restaurant that charges by the ingredient. Every tomato, every pinch of salt, every second of stove time—metered and billed. The chef who used to cook freely now watches the ticker: $0.02 for that extra garlic, $0.15 for the cream reduction, $3.50 for the wagyu.

You'd change how you cook. Simple dishes would stay simple. You'd save the expensive ingredients for where they matter. You'd prep efficiently, batch similar tasks, and reuse stock instead of making it fresh every time.

LLM costs work exactly like this. Every token is an ingredient. The expensive model is the wagyu—reserve it for the dish that demands it. The cheap model is the house red—perfectly good for most occasions. And caching? That's making a big batch of sauce on Sunday to use all week.

The goal isn't to spend less; it's to spend smart. A $3 dish that delights beats a $30 dish that disappoints. Same with AI: the right model at the right time beats the "best" model every time.

---

## Understanding your costs

Before optimizing, know where money goes.

### API pricing breakdown

| Provider | Model | Input (per 1M) | Output (per 1M) |
|----------|-------|----------------|-----------------|
| OpenAI | GPT-4o | $5.00 | $20.00 |
| OpenAI | GPT-4o Mini | $0.60 | $2.40 |
| Anthropic | Claude 4.5 Sonnet | $3.00 | $15.00 |
| Anthropic | Claude 3.5 Haiku | $0.80 | $4.00 |
| Google | Gemini 3 Pro | $2.00 | $12.00 |
| Google | Gemini 2.5 Flash | $0.15 | $3.50 |

**Key insight**: Output costs 3-5x more than input. Verbose responses drain budgets.

### Cost anatomy

A typical request:

```
System prompt:     500 tokens  (input)
User message:      200 tokens  (input)
Context/history:   2000 tokens (input)
Model response:    800 tokens  (output)
--------------------------------
Total input:       2700 tokens
Total output:      800 tokens

GPT-4o cost: ($2.50 × 0.0027) + ($10 × 0.0008) = $0.0147
GPT-4o-mini: ($0.15 × 0.0027) + ($0.60 × 0.0008) = $0.0009

Difference: 16x
```

### Measuring costs

```python
import tiktoken
from functools import lru_cache

@lru_cache()
def get_encoding(model: str):
    return tiktoken.encoding_for_model(model)

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o"
) -> float:
    """Estimate API cost in dollars."""
    pricing = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-sonnet-4-5": (3.00, 15.00),
        "claude-3-5-haiku": (0.80, 4.00),
    }

    input_rate, output_rate = pricing.get(model, (0, 0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

# Track costs per request
class CostTracker:
    def __init__(self):
        self.total = 0.0
        self.requests = []

    def log(self, input_tokens: int, output_tokens: int, model: str):
        cost = estimate_cost(input_tokens, output_tokens, model)
        self.total += cost
        self.requests.append({
            "input": input_tokens,
            "output": output_tokens,
            "model": model,
            "cost": cost
        })
        return cost

tracker = CostTracker()
```

## Strategy 1: model selection

The biggest lever. Right-size your model.

### The routing pattern

```python
def select_model(task_type: str, complexity: str) -> str:
    """Select appropriate model based on task."""

    # Simple tasks → cheap models
    if task_type in ["classification", "extraction", "formatting"]:
        return "gpt-4o-mini"

    # Complex reasoning → capable models
    if complexity == "high" or task_type in ["analysis", "code_generation"]:
        return "gpt-4o"

    # Default to mid-tier
    return "gpt-4o-mini"
```

### Intelligent router

Let a cheap model decide if the expensive one is needed:

```python
from openai import OpenAI

client = OpenAI()

def route_query(query: str) -> str:
    """Use a cheap model to route to the right model."""

    routing_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": """Classify this query's complexity:
            - SIMPLE: Factual, formatting, classification
            - COMPLEX: Reasoning, analysis, creative writing

            Respond with only SIMPLE or COMPLEX."""
        }, {
            "role": "user",
            "content": query
        }],
        max_tokens=10
    )

    complexity = routing_response.choices[0].message.content.strip()

    if complexity == "COMPLEX":
        return "gpt-4o"
    return "gpt-4o-mini"

# Usage
model = route_query("What's the capital of France?")  # → gpt-4o-mini
model = route_query("Analyze the themes in Hamlet")   # → gpt-4o
```

### Cost impact

| Scenario | GPT-4o Only | With Routing | Savings |
|----------|-------------|--------------|---------|
| 1000 queries/day | $44/day | $12/day | 73% |
| 80% simple, 20% complex | - | - | - |

## Strategy 2: caching

Don't pay twice for the same answer.

### Semantic caching

Cache based on meaning, not exact match:

```python
import hashlib
import json
from pathlib import Path
import numpy as np

class SemanticCache:
    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings = {}  # query_hash → embedding
        self.responses = {}   # query_hash → response

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _embed(self, text: str) -> np.ndarray:
        # Use a fast embedding model
        # (Implementation from Chapter 8)
        return embed_google([text])[0]

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str, threshold: float = 0.95) -> str | None:
        """Find semantically similar cached response."""
        query_embedding = self._embed(query)

        for cached_hash, cached_embedding in self.embeddings.items():
            similarity = self._similarity(query_embedding, cached_embedding)
            if similarity >= threshold:
                return self.responses[cached_hash]

        return None

    def set(self, query: str, response: str):
        """Cache a query-response pair."""
        query_hash = self._hash(query)
        self.embeddings[query_hash] = self._embed(query)
        self.responses[query_hash] = response

        # Persist to disk
        cache_file = self.cache_dir / f"{query_hash}.json"
        cache_file.write_text(json.dumps({
            "query": query,
            "response": response
        }))

# Usage
cache = SemanticCache()

def query_with_cache(query: str) -> str:
    # Check cache first
    cached = cache.get(query)
    if cached:
        print("Cache hit!")
        return cached

    # Cache miss - call API
    response = call_llm(query)
    cache.set(query, response)
    return response
```

### Time-based invalidation

Some responses expire:

```python
from datetime import datetime, timedelta

class ExpiringCache(SemanticCache):
    def __init__(self, ttl_hours: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.ttl = timedelta(hours=ttl_hours)
        self.timestamps = {}

    def get(self, query: str, threshold: float = 0.95) -> str | None:
        result = super().get(query, threshold)
        if result:
            query_hash = self._hash(query)
            if datetime.now() - self.timestamps.get(query_hash, datetime.min) < self.ttl:
                return result
            # Expired - remove from cache
            del self.responses[query_hash]
            del self.embeddings[query_hash]
        return None

    def set(self, query: str, response: str):
        super().set(query, response)
        self.timestamps[self._hash(query)] = datetime.now()
```

### Cache hit rates

Typical patterns:

| Use Case | Expected Hit Rate | Monthly Savings |
|----------|-------------------|-----------------|
| FAQ bot | 60-80% | $500+ |
| Code assistance | 20-40% | $200+ |
| Creative writing | 5-15% | $50+ |

## Strategy 3: prompt optimization

Shorter prompts cost less. Better prompts need fewer retries.

### Compress system prompts

```python
# Before: 500 tokens
system_prompt_verbose = """
You are a helpful AI assistant designed to help users with their questions.
You should always be polite, professional, and thorough in your responses.
When answering questions, make sure to consider multiple perspectives and
provide balanced information. If you don't know something, say so honestly.
Always cite your sources when providing factual information...
[continues for another 400 tokens]
"""

# After: 80 tokens
system_prompt_compact = """
You're a helpful assistant. Be concise, accurate, and honest.
Cite sources for facts. Say "I don't know" when uncertain.
"""

# Savings: 420 tokens × $2.50/1M × 1000 requests/day = $1.05/day
```

### Reduce context window

Don't send the whole conversation:

```python
def trim_conversation(messages: list[dict], max_tokens: int = 4000) -> list[dict]:
    """Keep only recent messages that fit in budget."""
    total = 0
    trimmed = []

    # Always keep system prompt
    if messages and messages[0]["role"] == "system":
        system_tokens = count_tokens(messages[0]["content"])
        trimmed.append(messages[0])
        total += system_tokens
        messages = messages[1:]

    # Add messages from most recent, backwards
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg["content"])
        if total + msg_tokens > max_tokens:
            break
        trimmed.insert(1 if trimmed else 0, msg)
        total += msg_tokens

    return trimmed
```

### Summarize history

```python
async def summarize_conversation(messages: list[dict]) -> str:
    """Summarize old messages to reduce tokens."""
    conversation_text = "\n".join([
        f"{m['role']}: {m['content']}" for m in messages
    ])

    summary = await client.chat.completions.create(
        model="gpt-4o-mini",  # Cheap model for summarization
        messages=[{
            "role": "system",
            "content": "Summarize this conversation in 2-3 sentences, preserving key facts and decisions."
        }, {
            "role": "user",
            "content": conversation_text
        }],
        max_tokens=200
    )

    return summary.choices[0].message.content

# Usage
if len(messages) > 20:
    old_messages = messages[1:-10]  # Keep system + last 10
    summary = await summarize_conversation(old_messages)
    messages = [
        messages[0],  # System prompt
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *messages[-10:]  # Recent messages
    ]
```

## Strategy 4: batching

Process multiple requests efficiently.

### Batch embeddings

```python
# Expensive: One API call per text
texts = ["text1", "text2", "text3", ...]  # 100 texts
for text in texts:
    embedding = embed(text)  # 100 API calls

# Cheap: One API call for all
embeddings = embed_batch(texts)  # 1 API call
```

### Batch LLM requests

OpenAI's Batch API offers 50% discount:

```python
from openai import OpenAI

client = OpenAI()

def create_batch_job(requests: list[dict]) -> str:
    """Submit batch job for 50% cost reduction."""
    # Create JSONL file
    batch_file = "batch_input.jsonl"
    with open(batch_file, "w") as f:
        for i, req in enumerate(requests):
            f.write(json.dumps({
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": req
            }) + "\n")

    # Upload file
    with open(batch_file, "rb") as f:
        file = client.files.create(file=f, purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch.id

# Retrieve results later
def get_batch_results(batch_id: str) -> list[dict]:
    batch = client.batches.retrieve(batch_id)
    if batch.status == "completed":
        results_file = client.files.content(batch.output_file_id)
        return [json.loads(line) for line in results_file.text.split("\n") if line]
    return []
```

## Strategy 5: provider arbitrage

Different providers, same capability, different prices.

### LiteLLM for provider switching

```python
from litellm import completion

def get_cheapest_provider(task: str) -> str:
    """Select cheapest provider for task type."""
    # Prices as of late 2025 (per 1M tokens, input/output average)
    providers = {
        "groq/llama-3.1-70b": 0.80,      # Fast and cheap
        "together/meta-llama/llama-3.1-70b": 0.90,
        "gpt-5-mini": 0.375,
        "gemini/gemini-2.5-flash": 0.19,
        "claude-3-5-haiku": 0.75,
    }

    # For simple tasks, cheapest wins
    if task in ["classification", "extraction"]:
        return min(providers, key=providers.get)

    # For quality tasks, use known-good models
    return "gpt-4o-mini"

def query_cheapest(prompt: str, task: str = "general") -> str:
    model = get_cheapest_provider(task)
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Free tier maximization

Exhaust free tiers before paying:

```python
FREE_TIER_LIMITS = {
    "gemini": {"rpm": 15, "tpd": 1_000_000},
    "groq": {"rpm": 30, "tpd": 6000},
    "mistral": {"rpm": 2, "messages": 1000},
}

class FreeTierRouter:
    def __init__(self):
        self.usage = {provider: 0 for provider in FREE_TIER_LIMITS}

    def get_available_free_provider(self) -> str | None:
        for provider, limits in FREE_TIER_LIMITS.items():
            if self.usage[provider] < limits.get("tpd", float("inf")):
                return provider
        return None

    def route(self, prompt: str) -> str:
        free_provider = self.get_available_free_provider()
        if free_provider:
            result = query_provider(free_provider, prompt)
            self.usage[free_provider] += count_tokens(prompt + result)
            return result

        # Fall back to paid
        return query_provider("gpt-4o-mini", prompt)
```

## Strategy 6: output control

Limit response length to limit costs.

### Max tokens setting

```python
# Expensive: No limit
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    # No max_tokens - could return 4000+ tokens
)

# Controlled: Explicit limit
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing in 2-3 sentences"}],
    max_tokens=100  # Hard limit
)
```

### Structured output for brevity

```python
from pydantic import BaseModel

class BriefAnswer(BaseModel):
    answer: str  # Max 50 words
    confidence: float

# Forces concise response
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Is Python good for ML?"}],
    response_format=BriefAnswer
)
```

## Cost optimization checklist

### Immediate wins (day 1)

- [ ] Add token counting to all requests
- [ ] Set max_tokens on every API call
- [ ] Use GPT-4o-mini/Haiku for simple tasks

### Short-term (week 1)

- [ ] Implement response caching
- [ ] Compress system prompts
- [ ] Trim conversation history

### Medium-term (month 1)

- [ ] Build model routing logic
- [ ] Implement semantic caching
- [ ] Use batch API for non-urgent requests
- [ ] Set up free tier rotation

### Long-term (quarter 1)

- [ ] Self-host embeddings (if volume justifies)
- [ ] Fine-tune small models for specific tasks
- [ ] Build cost dashboards and alerts

## Real-world savings

A production chatbot optimization:

| Optimization | Before | After | Savings |
|--------------|--------|-------|---------|
| Model routing | $3,000/mo | $1,200/mo | 60% |
| Response caching | $1,200/mo | $800/mo | 33% |
| Prompt compression | $800/mo | $650/mo | 19% |
| Batch processing | $650/mo | $500/mo | 23% |
| **Total** | $3,000/mo | $500/mo | **83%** |

## Getting started: two approaches

**Simple approach**: Just use the cheap model. That's it.

```python
from openai import OpenAI

client = OpenAI()

def query_simple(prompt: str) -> str:
    """Default to the cheapest capable model."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 16x cheaper than gpt-4o
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500  # Control output length
    )
    return response.choices[0].message.content

# That's it. You just cut costs by 90%.
# gpt-4o-mini handles most tasks just fine.
```

For most use cases, switching from GPT-4o to GPT-4o-mini (or Claude Sonnet to Claude Haiku) cuts costs by 80-90% with minimal quality loss. Try the cheap model first; upgrade only when you see actual quality problems.

**Production approach**: Multi-layer optimization with routing, caching, and batching.

```python
# cost_optimizer.py - Production cost management
from litellm import completion
from functools import lru_cache
import hashlib
import json
from pathlib import Path

class CostOptimizedLLM:
    """LLM client with automatic cost optimization."""

    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.stats = {"cache_hits": 0, "cheap_model": 0, "expensive_model": 0}

    def _cache_key(self, prompt: str, model: str) -> str:
        return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()

    def _get_cached(self, prompt: str, model: str) -> str | None:
        key = self._cache_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            self.stats["cache_hits"] += 1
            return json.loads(cache_file.read_text())["response"]
        return None

    def _set_cached(self, prompt: str, model: str, response: str):
        key = self._cache_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps({"prompt": prompt, "response": response}))

    def _route_model(self, prompt: str, task: str) -> str:
        """Select cheapest model that can handle the task."""
        # Simple tasks → cheapest model
        simple_tasks = {"classify", "extract", "format", "summarize"}
        if task in simple_tasks or len(prompt) < 200:
            self.stats["cheap_model"] += 1
            return "gpt-4o-mini"

        # Complex tasks → capable model
        self.stats["expensive_model"] += 1
        return "gpt-4o"

    def query(
        self,
        prompt: str,
        task: str = "general",
        use_cache: bool = True,
        max_tokens: int = 500
    ) -> str:
        """Query with automatic optimization."""
        model = self._route_model(prompt, task)

        # Check cache
        if use_cache:
            cached = self._get_cached(prompt, model)
            if cached:
                return cached

        # Make request
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        result = response.choices[0].message.content

        # Cache result
        if use_cache:
            self._set_cached(prompt, model, result)

        return result

    def report(self) -> dict:
        """Return usage statistics."""
        total = self.stats["cheap_model"] + self.stats["expensive_model"]
        return {
            **self.stats,
            "cache_rate": self.stats["cache_hits"] / max(total, 1),
            "cheap_rate": self.stats["cheap_model"] / max(total, 1)
        }

# Usage
llm = CostOptimizedLLM()

# Automatic routing and caching
answer = llm.query("What is 2+2?", task="classify")  # Uses cheap model, caches
answer = llm.query("What is 2+2?", task="classify")  # Cache hit, free!
answer = llm.query("Analyze the themes in Hamlet", task="analysis")  # Uses gpt-4o

print(llm.report())
# {'cache_hits': 1, 'cheap_model': 1, 'expensive_model': 1, 'cache_rate': 0.33, 'cheap_rate': 0.5}
```

The simple approach saves 90% by just picking the right model. The production approach layers caching, routing, and output control to squeeze out another 50-80%. Start simple—add complexity only when your bill justifies the engineering time.

## The takeaway

Cost optimization isn't about being cheap—it's about being smart.

1. **Measure first**: You can't optimize what you don't track
2. **Right-size models**: GPT-4o-mini handles 80% of tasks
3. **Cache aggressively**: Same question = same answer = free
4. **Control output**: Shorter responses cost less
5. **Use free tiers**: They exist; use them

The goal isn't minimum cost—it's maximum value per dollar. Sometimes GPT-4o is worth the premium. Usually, it's not.

![A factory production line where tokens flow through optimization machines that compress, cache, and route them efficiently, technical diagram style](https://pixy.vexy.art/)

---

*Next: Chapter 15 looks ahead—what's coming in 2026 and beyond.*
