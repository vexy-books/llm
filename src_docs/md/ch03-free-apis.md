# Chapter 3: Free APIs — the budget detective's guide

*You don't need to go broke to go smart.*

> "The best things in life are free. The second best are very expensive."
> — Coco Chanel

## The zero-dollar workflow

The kerning project started as a weekend experiment. I had an idea: could an LLM identify font substitution issues by analyzing glyph shapes? The technical answer was "probably." The financial answer was "not if it costs $50 in API calls to find out."

Free tiers exist for a reason. Providers want you hooked before you pay. That's fine—we can use their marketing strategy to bootstrap real projects.

This chapter maps every meaningful free tier as of October 2025. Spoiler: you can build production-quality prototypes without spending a cent.

## Imagine...

Imagine a city with five competing power companies. Each offers unlimited electricity for your first month—free. They're betting you'll like the lights on and keep paying.

Now imagine you're clever. You don't need unlimited power forever. You need enough to build a prototype, test an idea, prove a concept. So you wire your workshop to draw from Company A during peak hours, Company B overnight, Company C for heavy machinery. When one meter maxes out, you switch.

That's free tier arbitrage. The providers are giving away samples hoping you'll subscribe. You're accepting every sample, combining them strategically, and building real things on $0/month.

It's not cheating. It's using the system exactly as designed—just more creatively than they expected.

---

## The free tier landscape

### Tier 1: genuinely useful (daily driver material)

**Google AI Studio (Gemini)**

| Model | Daily Limit | Context | Commercial Use |
|-------|-------------|---------|----------------|
| Gemini 2.5 Flash | 250 RPD | 1M tokens | Yes |
| Gemini 2.5 Pro | 3M TPD | 1M tokens | Yes |

This is the best free tier in the industry. Three million tokens per day of Pro-level capability with million-token context. Commercial use allowed. Data used for training by default (disable in settings).

The Gemini CLI provides 1,000 requests/day with web grounding included. For most developers, this is enough to never pay.

**Groq**

| Model | Limit | Speed |
|-------|-------|-------|
| Llama 3.1 70B | 14,400 RPD | ~500 tok/s |
| Mixtral 8x7B | 14,400 RPD | ~800 tok/s |

Groq's free tier emphasizes speed. Their LPU hardware delivers the fastest inference available. Rate limits are per-model, so you can effectively multiply capacity by using multiple models.

**Cerebras**

| Model | Limit | Speed |
|-------|-------|-------|
| Llama 3.3 70B | 14,400 RPD, 1M TPD | 2,000 tok/s |

Similar to Groq but with even faster inference. The combination of high daily limits and blazing speed makes this excellent for interactive development.

**OpenRouter (Free Models)**

| Model | Limit |
|-------|-------|
| Various free models | 50 RPD |
| With $10 credit | 1,000 RPD |

OpenRouter aggregates 300+ models. Their free tier is limited, but a $10 deposit unlocks 1,000 requests/day across any model. Good for experimentation.

### Tier 2: useful with limitations

**Mistral AI**

| Model | Limit |
|-------|-------|
| Codestral | 2,000 RPD |
| Mistral Small | Limited |

Codestral is optimized for code generation. The free tier is generous enough for development work.

**Hugging Face**

| Access | Limit |
|--------|-------|
| Serverless Inference | 300 req/hr (registered) |
| Pro ($9/mo) | 1,000 req/hr |

Access to thousands of open models including embedding models. Rate limits are per-model, so parallel requests to different models work well.

**Pollinations.AI**

| Feature | Limit |
|---------|-------|
| Text & Image Generation | Unlimited |

No signup required. No API key needed. Truly unlimited for both text and image generation. Quality varies, but for prototyping, it's unbeatable.

### Tier 3: vector databases (the hidden essential)

Free LLM access is worthless without somewhere to store embeddings. These providers offer permanent free tiers:

**Pinecone**
- 2GB storage
- 2M writes/month
- Serverless deployment
- Best for: Production RAG with moderate scale

**Qdrant Cloud**
- 1GB cluster
- Suspends after 7 days idle (restarts on request)
- Best for: Development and testing

**Chroma**
- Unlimited local storage
- In-memory or persistent
- Best for: Prototyping and local development

**LanceDB**
- Unlimited local storage
- Embedded database (no server)
- Best for: Local-first applications

**Zilliz Cloud**
- 5GB storage
- 2.5M vector compute units/month
- Best for: Larger prototypes

### Tier 4: image generation

**Google Gemini**
- 100 images/day through Imagen
- High quality, commercial use allowed

**Pollinations.AI**
- Unlimited images
- No signup required
- Variable quality

**Leonardo.ai**
- 150 tokens/month
- Good quality but very limited

### What's NOT free

**Video generation** has no viable free API tier. The compute costs are too high. Web tools like Kling and Runway offer limited credits, but nothing usable for programmatic access.

**OpenAI** has no meaningful free tier. New accounts get $5 credit that expires, but there's no ongoing free access.

**Anthropic** has no API free tier. The Claude.ai chat interface has limited free messages, but API access requires payment.

## The gotchas

Free tiers come with catches. Know them before you build.

### 1. Commercial use restrictions

Most free tiers prohibit commercial use. The exceptions:

- **Google AI Studio**: Commercial allowed
- **Groq**: Commercial allowed
- **Pollinations**: No restrictions (no terms of service)

If you're building a product, verify the terms.

### 2. Data training defaults

Free tier data often trains future models. Providers with opt-out:

- **Google**: Default on, can disable
- **OpenAI**: API data not used for training
- **Anthropic**: Opt-out available

If privacy matters, read the fine print.

### 3. Nested rate limits

Daily limits aren't the only constraint. Watch for:

- Requests per minute (RPM)
- Tokens per minute (TPM)
- Concurrent request limits
- Per-model vs. account-wide limits

A 14,400 RPD limit means nothing if RPM is 10.

### 4. Volatility

Free tiers disappear. Together AI eliminated theirs in August 2025. AWS Bedrock's free tier lasted 6 months. Build with fallbacks.

### 5. Peak hour degradation

Free tiers get deprioritized under load. Expect 5x latency during peak hours (US business hours, typically).

---

### Using free tiers: two approaches

**Approach 1: Single provider (simple)**

Pick Google's Gemini and stick with it. One API key, one provider, no complexity.

```bash
# Get your key from aistudio.google.com (takes 2 minutes)
export GOOGLE_API_KEY="AIza..."

# Install the client
pip install google-generativeai

# Use it
python -c "
import google.generativeai as genai
genai.configure()
model = genai.GenerativeModel('gemini-2.5-pro')
print(model.generate_content('What makes Helvetica distinctive?').text)
"
```

You get 3 million tokens per day. For personal projects and learning, this is plenty. If you hit limits, just wait until tomorrow.

**Approach 2: Multi-provider orchestration (resilient)**

Build a fault-tolerant system that falls back across providers, maximizes daily capacity, and routes based on task requirements.

```python
import os
from dataclasses import dataclass
from litellm import completion, embedding
from typing import Literal

@dataclass
class FreeQuota:
    provider: str
    daily_tokens: int
    strength: Literal["speed", "reasoning", "embedding"]

FREE_PROVIDERS = [
    FreeQuota("gemini/gemini-2.5-pro", 3_000_000, "reasoning"),
    FreeQuota("groq/llama-3.1-70b-versatile", 500_000, "speed"),
    FreeQuota("cerebras/llama-3.3-70b", 1_000_000, "speed"),
]

class FreeTierOrchestrator:
    """Maximize free tier usage across providers."""

    def __init__(self):
        self.usage = {p.provider: 0 for p in FREE_PROVIDERS}

    def best_provider(self, need: str = "reasoning") -> str:
        """Get provider with most remaining quota for given need."""
        candidates = [p for p in FREE_PROVIDERS if p.strength == need]
        return min(
            candidates,
            key=lambda p: self.usage[p.provider] / p.daily_tokens
        ).provider

    def query(self, prompt: str, need: str = "reasoning") -> str:
        provider = self.best_provider(need)
        response = completion(
            model=provider,
            messages=[{"role": "user", "content": prompt}],
            fallbacks=[p.provider for p in FREE_PROVIDERS if p.provider != provider]
        )
        # Track usage (estimate 4 chars per token)
        tokens = (len(prompt) + len(response.choices[0].message.content)) // 4
        self.usage[provider] += tokens
        return response.choices[0].message.content

# Usage
orchestrator = FreeTierOrchestrator()
result = orchestrator.query("Analyze the x-height ratio of Garamond", need="reasoning")
```

This gives you 4.5M+ tokens per day across providers, with automatic failover. Overkill for personal use, essential for any shared tool.

---

## Building a free-tier stack

Here's a production-quality stack that costs $0:

```
┌─────────────────────────────────────────────┐
│              Application Layer               │
├─────────────────────────────────────────────┤
│ LiteLLM (provider abstraction + fallback)   │
├─────────────────────────────────────────────┤
│ Primary: Google Gemini (3M tokens/day)      │
│ Fallback: Groq (14,400 req/day)             │
│ Speed: Cerebras (14,400 req/day)            │
├─────────────────────────────────────────────┤
│ Embeddings: Google (unlimited free)          │
├─────────────────────────────────────────────┤
│ Vector DB: Pinecone (2GB free)              │
└─────────────────────────────────────────────┘
```

**Daily capacity**: 3M+ tokens of LLM inference, unlimited embeddings, 2GB vector storage.

**Monthly cost**: $0.

### Implementation with LiteLLM

```python
from litellm import completion

# Configure fallback chain
response = completion(
    model="gemini/gemini-2.5-pro",
    messages=[{"role": "user", "content": prompt}],
    fallbacks=["groq/llama-3.1-70b", "cerebras/llama-3.3-70b"],
    timeout=30
)
```

When Gemini hits rate limits, LiteLLM automatically falls back to Groq, then Cerebras. Your application keeps running.

### Embedding strategy

Google's `gemini-embedding-001` is free and ranks #1 on MTEB benchmarks. Use it:

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def embed(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )
    return result['embedding']
```

### Vector storage

For prototyping, Chroma runs locally with no setup:

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("fonts")

collection.add(
    documents=["Garamond has moderate contrast..."],
    embeddings=[embed("Garamond has moderate contrast...")],
    ids=["garamond-001"]
)
```

For production, Pinecone's free tier handles real traffic:

```python
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("fonts")

index.upsert(vectors=[{
    "id": "garamond-001",
    "values": embed("Garamond has moderate contrast..."),
    "metadata": {"family": "Garamond"}
}])
```

## The "good enough" threshold

Free tiers have limits. Understanding where they stop being useful helps you plan:

| Use Case | Free Tier Sufficient? | When to Upgrade |
|----------|----------------------|-----------------|
| Learning/Experimentation | Yes | Never |
| Weekend projects | Yes | Never |
| Internal tools (small team) | Usually | >50 daily users |
| MVP/Prototype | Yes | Before public launch |
| Production (low traffic) | Maybe | >1,000 daily requests |
| Production (high traffic) | No | From day one |

The line is roughly: **if paying customers depend on it, pay for it.**

## Cost comparison: free vs. paid

What would this stack cost without free tiers?

| Component | Free Tier | Paid Equivalent |
|-----------|-----------|-----------------|
| LLM (3M tokens/day) | $0 | ~$225/mo (GPT-4o) |
| Embeddings (1M/day) | $0 | ~$4.50/mo (OpenAI) |
| Vector DB (2GB) | $0 | ~$70/mo (Pinecone starter) |
| **Total** | **$0** | **~$300/mo** |

Free tiers save real money. Use them.

## The upgrade path

When free tiers stop working, graduate strategically:

1. **First upgrade**: Embedding provider (Voyage AI at $0.02/M is cheap)
2. **Second upgrade**: Vector DB (Pinecone paid tiers scale smoothly)
3. **Third upgrade**: LLM provider (OpenAI Plus at $20/mo or Cerebras Pro at $50/mo)
4. **Final upgrade**: Dedicated infrastructure (only at serious scale)

Most projects never get past step 2.

## The takeaway

Free tiers aren't charity—they're marketing. But marketing that works both ways. The providers get potential customers; you get production-quality infrastructure for $0.

The font identification project I mentioned? It ran on free tiers for three months before I spent a dollar. By then, I knew exactly what I needed and where to spend.

Start free. Upgrade when forced to. Let your usage patterns guide your spending.

![A detective piecing together clues on a cork board, each clue labeled with provider logos and pricing, noir style](https://pixy.vexy.art/)

---

*Next: Chapter 4 surveys the Python package jungle—122+ libraries for every LLM workflow imaginable.*
