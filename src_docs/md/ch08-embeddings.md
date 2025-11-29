# Chapter 8: Embeddings — the secret sauce

*Turning text into math. More useful than it sounds.*

> "Mathematics is the language with which God has written the universe."
> — Galileo Galilei

## What embeddings actually are

Every RAG system, every similarity search, every semantic query depends on embeddings. They're the foundation nobody talks about because the concept sounds boring.

An embedding is a vector—a list of numbers—that represents meaning. "Happy" and "joyful" have similar embeddings. "Happy" and "database" don't. The magic is that mathematical operations on these vectors (distance, similarity) correspond to semantic relationships in language.

```
"happy" → [0.23, -0.14, 0.67, ..., 0.89]  (1536 numbers)
"joyful" → [0.21, -0.12, 0.65, ..., 0.91]  (similar!)
"database" → [-0.45, 0.82, -0.11, ..., 0.03]  (different)
```

**Cosine similarity** measures how similar two embeddings are. Values range from -1 (opposite) to 1 (identical). "Happy" and "joyful" might score 0.95. "Happy" and "database" might score 0.1.

This chapter covers the provider landscape, dimension trade-offs, and practical strategies for high-volume embedding workloads.

## Imagine...

Imagine a sommelier who can smell a wine and instantly place it: "Burgundy, 2018, probably from the Côte de Nuits." They're compressing the full sensory experience of the wine into a mental coordinate system—grape variety, region, vintage, oak treatment, acidity.

Embeddings do this for text. They compress the full meaning of a sentence into a list of numbers—a coordinate in semantic space. "The wine has notes of cherry and earth" lands near other Pinot Noir descriptions. "The database connection timed out" lands in a completely different region.

The magic is that distance in this space correlates with meaning. Things that mean similar things are close together. This lets you search by meaning, not keywords. A query for "fruit-forward red wine" finds results that never use those exact words but describe exactly that.

The sommelier's skill is hard-won. Embeddings are trained on billions of text pairs, learning which words and phrases belong near each other. The result is a compression algorithm for meaning itself.

---

## Provider comparison

The embedding market commoditized faster than anyone expected. Quality went up, prices crashed.

### The complete landscape (late 2025)

| Provider | Model | Price/M | MTEB Score | Dims | Free Tier |
|----------|-------|---------|------------|------|-----------|
| **Google Gemini** | gemini-embedding-001 | $0.15 | **68.37** | 3072 | Unlimited |
| **Voyage AI** | voyage-3.5-lite | $0.02 | 66.1 | varies | 200M tokens |
| **OpenAI** | text-embedding-3-small | $0.02 | 62.3 | 1536 | None |
| **OpenAI** | text-embedding-3-large | $0.13 | 64.6 | 3072 | None |
| **Jina AI** | jina-embeddings-v4 | $0.045 | High | varies | 10M tokens |
| **Cohere** | Embed v4 | $0.12 | SOTA | 1536 | 1k calls |
| **DeepInfra** | bge-base-en-v1.5 | $0.002 | 63.57 | 768 | None |
| **Fireworks AI** | nomic-embed-text | $0.008 | 62.28 | 768 | $1 credit |
| **Together AI** | BGE-Base-EN | $0.01 | 63.57 | 768 | $25 credit |
| **AWS Bedrock** | Titan Embeddings v2 | $0.11 | Medium | 1024 | None |
| **Cloudflare** | EmbeddingGemma-300m | $0.011/1k neurons | 65.0 | 768 | 10k neurons/day |

### Key Insight: Performance Is Decoupled From Price

Google Gemini ranks **#1** on MTEB benchmarks (68.37) and offers an **unlimited free tier**.

Voyage AI scores 66.1 (96% of Gemini's performance) at $0.02/M tokens.

OpenAI's most expensive option (text-embedding-3-large at $0.13/M) scores lower than the free alternative.

**Price doesn't predict quality.** Always benchmark on your data.

## Provider deep dives

### Google Gemini (Best Overall Value)

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def embed_google(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return result['embedding']
```

**Why Gemini wins:**
- Unlimited free tier through AI Studio
- Highest MTEB score (68.37)
- Configurable dimensions (768, 1536, or 3072)
- Commercial use allowed

**Gotcha**: Default data training is on. Disable in settings if sensitive.

### Voyage AI (Best Cost/Performance)

```python
import voyageai

vo = voyageai.Client()

def embed_voyage(texts: list[str]) -> list[list[float]]:
    result = vo.embed(
        texts,
        model="voyage-3.5-lite",
        input_type="document"
    )
    return result.embeddings
```

**Why Voyage matters:**
- 200M free tokens on signup
- voyage-code-3 optimized for code (best for code search)
- voyage-3 handles 32k context (long documents)
- Acquired by MongoDB (2025)—good for longevity

**Use Voyage when**: Code embeddings, long documents, or after exhausting Gemini free tier.

### OpenAI (Ecosystem Integration)

```python
from openai import OpenAI

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]
```

**Two models:**
- `text-embedding-3-small`: 1536 dims, $0.02/M, 62.3 MTEB
- `text-embedding-3-large`: 3072 dims, $0.13/M, 64.6 MTEB

**When OpenAI makes sense**: You're already paying for GPT-4o and want one vendor.

**When to avoid**: Cost matters. Better options exist for less.

### Budget Options

**DeepInfra** — Cheapest at $0.002/M
```python
import httpx

def embed_deepinfra(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        "https://api.deepinfra.com/v1/inference/BAAI/bge-base-en-v1.5",
        headers={"Authorization": f"Bearer {os.environ['DEEPINFRA_TOKEN']}"},
        json={"inputs": texts}
    )
    return response.json()["embeddings"]
```

**Fireworks AI** — $0.008/M real-time, $0.004/M batch
```python
# OpenAI-compatible API
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ["FIREWORKS_API_KEY"]
)

response = client.embeddings.create(
    model="nomic-ai/nomic-embed-text-v1.5",
    input=texts
)
```

## Dimension trade-offs

Embedding dimension affects storage, speed, and quality.

| Dimension | Storage (1M vectors) | Search Speed | Quality |
|-----------|---------------------|--------------|---------|
| 768 | ~3 GB | Fastest | Good |
| 1536 | ~6 GB | Medium | Better |
| 3072 | ~12 GB | Slowest | Best |

### When to Use Each

**768 dimensions**: Budget-constrained, high-volume, speed-critical
**1536 dimensions**: Balanced (most common choice)
**3072 dimensions**: Quality-critical, small corpus, budget available

### Dimension Reduction

Some providers support dimension reduction—generating full-size embeddings then truncating:

```python
# OpenAI text-embedding-3-large supports this
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts,
    dimensions=1536  # Reduce from 3072 to 1536
)
```

You get most of the quality at reduced storage cost.

## Batch processing strategies

Embedding millions of documents? Batch efficiently.

### Basic Batching

```python
def batch_embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embed_google(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### Async Batching

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed_batch_async(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    async def process_batch(batch):
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        return [item.embedding for item in response.data]

    results = await asyncio.gather(*[process_batch(b) for b in batches])
    return [emb for batch in results for emb in batch]
```

### Rate Limit Handling

```python
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(5))
def embed_with_retry(texts: list[str]) -> list[list[float]]:
    return embed_google(texts)
```

### Provider Fallback

```python
def embed_with_fallback(texts: list[str]) -> list[list[float]]:
    providers = [
        ("google", embed_google),
        ("voyage", embed_voyage),
        ("openai", embed_openai),
    ]

    for name, embed_fn in providers:
        try:
            return embed_fn(texts)
        except Exception as e:
            print(f"{name} failed: {e}")
            continue

    raise RuntimeError("All embedding providers failed")
```

## Self-hosting economics

When does self-hosting beat APIs?

### The Math

- **Voyage AI**: $0.02/M tokens
- **NVIDIA T4 GPU**: ~$365/month

Breakeven: **18.25 billion tokens/month**

Unless you're processing billions of tokens monthly, APIs are cheaper than self-hosting.

### When Self-Hosting Makes Sense

1. **Privacy requirements**: Data can't leave your infrastructure
2. **Extreme latency needs**: Network round-trip is unacceptable
3. **Offline operation**: No internet available
4. **Massive scale**: >20B tokens/month

### Self-Hosting Options

**Hugging Face TEI (Text Embeddings Inference)**:
```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-base-en-v1.5
```

**Cloudflare Workers AI** (serverless):
```javascript
// Runs on Cloudflare's edge
const response = await env.AI.run(
  "@cf/google/embeddinggemma-300m",
  { text: "Your text here" }
);
```

## Embedding strategies by use case

### Code Search

Use Voyage AI's `voyage-code-3`:

```python
vo = voyageai.Client()

# Index code
code_embeddings = vo.embed(
    code_snippets,
    model="voyage-code-3",
    input_type="document"
)

# Search code
query_embedding = vo.embed(
    ["function to calculate cosine similarity"],
    model="voyage-code-3",
    input_type="query"
)
```

### Long Documents

Use Voyage AI's `voyage-3` (32k context):

```python
# Can embed entire documents without chunking
long_doc_embedding = vo.embed(
    [entire_document],  # Up to 32k tokens
    model="voyage-3"
)
```

### Multimodal (Text + Images)

Use Cohere `embed-4` or Jina `v4`:

```python
import cohere

co = cohere.Client()

# Text embedding
text_emb = co.embed(
    texts=["Description of image"],
    model="embed-v4",
    input_type="search_document"
)

# Image embedding (Cohere v4 supports this)
image_emb = co.embed(
    images=["base64_encoded_image"],
    model="embed-v4",
    input_type="image"
)
```

### Privacy-Sensitive

Use `EmbeddingGemma-300m` self-hosted:

```python
# Via Hugging Face TEI
import httpx

def embed_local(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        "http://localhost:8080/embed",
        json={"inputs": texts}
    )
    return response.json()
```

## Caching embeddings

Don't re-embed unchanged content.

```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("./embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

def embed_cached(text: str) -> list[float]:
    # Create cache key from text hash
    key = hashlib.md5(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"

    # Check cache
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    # Generate embedding
    embedding = embed_google([text])[0]

    # Save to cache
    cache_file.write_text(json.dumps(embedding))

    return embedding
```

For production, use Redis or your vector database's built-in caching.

## The strategic recommendation

**For most users (under 1B tokens/month):**

1. **Start**: Google Gemini (unlimited free, best quality)
2. **Backup**: Voyage AI (200M free tokens, great for code)
3. **Scale**: DeepInfra ($0.002/M) or Fireworks ($0.004/M batch)

**Specialized needs:**

| Need | Provider | Model |
|------|----------|-------|
| Code search | Voyage AI | voyage-code-3 |
| Long docs | Voyage AI | voyage-3 |
| Multimodal | Cohere | embed-v4 |
| Privacy | Self-hosted | EmbeddingGemma-300m |
| AWS ecosystem | AWS | Titan v2 |
| Cheapest | DeepInfra | bge-base-en-v1.5 |

---

> **Integration sidebar: Embeddings everywhere**
>
> Embeddings power more than just RAG. They're the translation layer between human meaning and machine math:
>
> - **Embeddings + RAG (Ch 7)**: The core use case. Quality embeddings mean quality retrieval. Test your embedding model on your actual queries before committing.
> - **Embeddings + Agents (Ch 9)**: Give agents tools to search embedding spaces. An agent can explore related concepts, find similar examples, or cluster results—all operations that embeddings enable.
> - **Embeddings + Python packages (Ch 4)**: LangChain and LlamaIndex abstract embedding providers. PydanticAI works with any embedding function. Instructor can extract structured data then embed it for later retrieval.
> - **Embeddings + Typography (Ch 10)**: CLIP embeddings turn font specimens into searchable vectors. Combine with custom classifiers for production font identification.
>
> The pattern: embeddings convert anything (text, images, code) into a format that supports similarity search, clustering, and classification. Once embedded, everything becomes searchable.

---

## Getting started: two approaches

**Simple approach**: Use Google Gemini. Free, best quality, zero configuration.

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

def embed_text(text: str) -> list[float]:
    """Embed a single text with Gemini."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

# That's it. Find similar fonts.
font_embedding = embed_text("Garamond is a classic old-style serif typeface")
```

One API call, one embedding. The free tier handles most projects indefinitely.

**Production approach**: Multi-provider system with caching, fallbacks, and specialized models.

```python
# typography_embeddings.py - Production embedding service
from pydantic import BaseModel
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import google.generativeai as genai
import voyageai

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    provider: str = "gemini"
    model: str = "gemini-embedding-001"
    dimensions: int = 768
    cache_enabled: bool = True

class TypographyEmbedder:
    """Production embedding service for typography content."""

    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize providers
        genai.configure()
        self.voyage = voyageai.Client()

        # Track usage
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

    def _cache_key(self, text: str, config: EmbeddingConfig) -> str:
        """Generate cache key from text and config."""
        content = f"{config.provider}:{config.model}:{config.dimensions}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> list[float] | None:
        """Retrieve cached embedding."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            self.stats["hits"] += 1
            return json.loads(cache_file.read_text())
        return None

    def _set_cached(self, key: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(embedding))

    def embed(
        self,
        text: str,
        config: EmbeddingConfig | None = None
    ) -> list[float]:
        """Embed text with caching and provider fallback."""
        config = config or EmbeddingConfig()

        # Check cache first
        if config.cache_enabled:
            key = self._cache_key(text, config)
            if cached := self._get_cached(key):
                return cached

        self.stats["misses"] += 1

        # Try primary provider
        embedding = None
        try:
            if config.provider == "gemini":
                embedding = self._embed_gemini(text, config)
            elif config.provider == "voyage":
                embedding = self._embed_voyage(text, config)
        except Exception as e:
            self.stats["errors"] += 1
            # Fall back to alternate provider
            embedding = self._fallback_embed(text, config)

        if embedding and config.cache_enabled:
            self._set_cached(key, embedding)

        return embedding

    def _embed_gemini(self, text: str, config: EmbeddingConfig) -> list[float]:
        """Embed with Google Gemini."""
        result = genai.embed_content(
            model=f"models/{config.model}",
            content=text,
            task_type="retrieval_document",
            output_dimensionality=config.dimensions
        )
        return result['embedding']

    def _embed_voyage(self, text: str, config: EmbeddingConfig) -> list[float]:
        """Embed with Voyage AI - best for code and long docs."""
        result = self.voyage.embed(
            [text],
            model="voyage-3.5-lite",
            input_type="document"
        )
        return result.embeddings[0]

    def _fallback_embed(self, text: str, config: EmbeddingConfig) -> list[float]:
        """Fallback to alternate provider."""
        if config.provider == "gemini":
            return self._embed_voyage(text, config)
        return self._embed_gemini(text, config)

    def embed_batch(
        self,
        texts: list[str],
        config: EmbeddingConfig | None = None
    ) -> list[list[float]]:
        """Batch embed with parallel caching."""
        return [self.embed(text, config) for text in texts]

# Usage
embedder = TypographyEmbedder()

# Simple embedding
vec = embedder.embed("Helvetica Neue is a neo-grotesque sans-serif")

# Custom config for code
code_config = EmbeddingConfig(provider="voyage", model="voyage-code-3")
code_vec = embedder.embed("def calculate_kerning(pair):", code_config)

# Batch process font descriptions
fonts = ["Bodoni: high contrast serif", "Futura: geometric sans-serif"]
vecs = embedder.embed_batch(fonts)
```

The simple approach handles prototypes and small projects. The production approach adds caching (no re-embedding unchanged content), fallbacks (service outages don't break your app), and provider selection (specialized models for code vs. text).

## The takeaway

Embeddings are a solved problem. The market has commoditized to the point where you can get state-of-the-art quality for free (Google Gemini) or near-free (Voyage, DeepInfra).

Don't overthink it:

1. Use Google Gemini's free tier
2. Fall back to Voyage if you hit limits
3. Cache aggressively
4. Benchmark on your actual data (MTEB scores are guidelines, not guarantees)

The embedding model is rarely the bottleneck. Chunking strategy, retrieval logic, and LLM quality matter more. Spend your optimization budget there.

![Mathematical vectors in 3D space converging toward a central point labeled 'meaning', with floating words transforming into numbers, technical illustration](https://pixy.vexy.art/)

---

*Next: Chapter 9 explores agents—autonomous LLMs that can reason, plan, and act.*
