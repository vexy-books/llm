# Chapter 8B: Embedding engineering

*The dark arts of vector spaces.*

> "The universe is not only queerer than we suppose, but queerer than we can suppose."
> — J.B.S. Haldane

## Beyond off-the-shelf embeddings

Chapter 8 covered the basics: pick a provider, embed your text, search by similarity. That works for 90% of use cases.

This chapter is for the other 10%. When your domain has specialized vocabulary that generic models miss. When storage costs matter at scale. When your users speak multiple languages but your documents don't.

These are the techniques that turn good retrieval into great retrieval.

## Imagine...

Imagine you're mapping a city. Google Maps works fine for most trips—it knows where the streets are, roughly how long things take. But it doesn't know that the shortcut through the alley is only safe before dark. It doesn't know that "the taco place by the old theater" means Maria's Tacos on 5th Street.

Local knowledge changes the map. Someone who lives there knows things the satellite can't see.

That's what embedding engineering does. Off-the-shelf embeddings are Google Maps—good enough for most journeys. Fine-tuned embeddings are local knowledge—they understand the shortcuts and landmarks that generic models miss.

Your domain has its own vocabulary, its own relationships, its own "taco place by the old theater." Engineering your embeddings captures that knowledge.

---

## Technique 1: domain fine-tuning

**The problem**: Generic embeddings treat all text equally. "Kerning" and "tracking" might have similar embeddings in a general model, but a typography expert knows they're distinct concepts with different use cases.

**The solution**: Fine-tune on your domain data. Teach the model your vocabulary.

### Contrastive fine-tuning with sentence-transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Start with a good base model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Create training examples: (anchor, positive, negative)
# Positive = similar meaning, Negative = different meaning
train_examples = [
    InputExample(texts=[
        "Kerning adjusts space between letter pairs",  # anchor
        "Letter spacing for specific character combinations",  # positive
        "The overall color of a text block"  # negative
    ]),
    InputExample(texts=[
        "Tracking affects uniform letter spacing",
        "Uniform adjustment to spacing across text",
        "Adjusting x-height of characters"
    ]),
    InputExample(texts=[
        "Leading controls vertical line spacing",
        "Distance between baselines of text",
        "The width of an em dash"
    ]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model=model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./typography-embeddings"
)
```

### Generating training data from your corpus

Manual triplet creation doesn't scale. Here's how to generate training data automatically:

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class TrainingTriplet(BaseModel):
    anchor: str
    positive: str
    negative: str

def generate_triplets(documents: list[str], n_triplets: int = 1000) -> list[TrainingTriplet]:
    """Use an LLM to generate high-quality training triplets."""
    triplets = []

    for doc in documents[:n_triplets]:
        response = client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[{
                "role": "system",
                "content": """Generate a training triplet for semantic similarity.

                Given an anchor text, create:
                - positive: a paraphrase or semantically similar text
                - negative: a semantically different but plausibly related text

                Keep texts concise (1-2 sentences)."""
            }, {
                "role": "user",
                "content": f"Anchor: {doc[:500]}"
            }],
            response_format=TrainingTriplet
        )
        triplets.append(response.choices[0].message.parsed)

    return triplets

# Generate from your corpus
triplets = generate_triplets(your_documents)
train_examples = [
    InputExample(texts=[t.anchor, t.positive, t.negative])
    for t in triplets
]
```

### When fine-tuning matters

| Scenario | Improvement | Worth it? |
|----------|-------------|-----------|
| Generic domain | 0-5% | No |
| Specialized terminology | 10-20% | Yes |
| Internal jargon/acronyms | 20-40% | Definitely |
| Completely new domain | 30-50% | Required |

**Rule of thumb**: If your users frequently use terms that don't appear in Wikipedia, fine-tuning will help.

---

## Technique 2: Matryoshka embeddings

**The problem**: Full-dimension embeddings are expensive to store and search. A 1M vector corpus at 1536 dimensions consumes ~6GB. At 3072 dimensions, 12GB.

**The solution**: Matryoshka Representation Learning (MRL). These embeddings are designed so truncating to smaller dimensions still works—just with slightly reduced quality.

### How it works

Traditional embeddings: All 1536 dimensions are equally important.

Matryoshka embeddings: Early dimensions capture the most important information. You can truncate `[0:768]` and still get useful similarity scores.

```
Full embedding:    [important, important, ..., less important, least important]
                   |____________ dims 768 ___________|______ dims 768 ______|

Truncated (768):   [important, important, ...]
                   |____________ dims 768 ___________|
```

### Using OpenAI's Matryoshka-enabled models

```python
from openai import OpenAI

client = OpenAI()

def embed_matryoshka(texts: list[str], dimensions: int = 768) -> list[list[float]]:
    """
    text-embedding-3-small and text-embedding-3-large support dimension reduction.

    Dimensions options:
    - text-embedding-3-small: 512, 1024, 1536 (default)
    - text-embedding-3-large: 256, 1024, 3072 (default)
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=dimensions  # Truncate on API side
    )
    return [item.embedding for item in response.data]

# Index with 1024 dims (saves 66% storage vs 3072)
embeddings_1024 = embed_matryoshka(documents, dimensions=1024)

# Still get reasonable quality
embeddings_256 = embed_matryoshka(documents, dimensions=256)  # 91% smaller
```

### Storage/quality tradeoff

Tested on MTEB benchmarks with text-embedding-3-large:

| Dimensions | Storage (per 1M vectors) | MTEB Score | % of Full |
|------------|-------------------------|------------|-----------|
| 3072 | 12 GB | 64.6 | 100% |
| 1536 | 6 GB | 63.8 | 98.8% |
| 1024 | 4 GB | 63.2 | 97.8% |
| 512 | 2 GB | 61.9 | 95.8% |
| 256 | 1 GB | 59.7 | 92.4% |

**Insight**: You can cut storage by 75% (3072→768) and lose only ~3% quality. That's often a worthwhile trade.

### Hybrid dimension strategy

Use different dimensions for different tiers:

```python
class HybridSearch:
    """
    Two-tier search:
    1. Fast, low-dimension pre-filter
    2. Precise, full-dimension re-rank
    """

    def __init__(self, low_dim: int = 256, high_dim: int = 1536):
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.low_index = {}   # id -> low-dim embedding
        self.high_index = {}  # id -> high-dim embedding

    def index(self, doc_id: str, text: str):
        # Store both dimensions
        self.low_index[doc_id] = embed_matryoshka([text], self.low_dim)[0]
        self.high_index[doc_id] = embed_matryoshka([text], self.high_dim)[0]

    def search(self, query: str, k: int = 10, pre_filter_k: int = 100) -> list[str]:
        # Fast pre-filter with low dimensions
        query_low = embed_matryoshka([query], self.low_dim)[0]
        candidates = self._top_k(query_low, self.low_index, pre_filter_k)

        # Precise re-rank with high dimensions
        query_high = embed_matryoshka([query], self.high_dim)[0]
        reranked = self._top_k(
            query_high,
            {doc_id: self.high_index[doc_id] for doc_id in candidates},
            k
        )

        return reranked

    def _top_k(self, query: list[float], index: dict, k: int) -> list[str]:
        scores = {
            doc_id: self._cosine_sim(query, emb)
            for doc_id, emb in index.items()
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## Technique 3: cross-lingual retrieval

**The problem**: Your documents are in English. Your users might ask questions in French, Spanish, or Japanese. Traditional embeddings encode language—"hello" and "bonjour" have very different vectors.

**The solution**: Multilingual embedding models that map all languages into a shared semantic space.

### Provider options

| Model | Languages | Quality | Notes |
|-------|-----------|---------|-------|
| Cohere embed-multilingual-v3.0 | 100+ | Excellent | Commercial |
| OpenAI text-embedding-3 | ~30 | Good | Implicit, not trained for it |
| Jina jina-embeddings-v4 | 89 | Excellent | Open weights available |
| BGE-M3 | 100+ | Excellent | Open source, self-hostable |

### Implementation with Cohere

```python
import cohere

co = cohere.Client()

def embed_multilingual(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """
    input_type options:
    - "search_document": for indexing documents
    - "search_query": for search queries
    - "classification": for classification tasks
    - "clustering": for clustering tasks
    """
    response = co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type=input_type
    )
    return response.embeddings

# Index English documents
english_docs = [
    "Typography is the art of arranging type",
    "Kerning adjusts spacing between letter pairs",
    "Sans-serif fonts lack decorative strokes"
]
doc_embeddings = embed_multilingual(english_docs, "search_document")

# Search in French
french_query = "Qu'est-ce que le crénage dans la typographie?"
query_embedding = embed_multilingual([french_query], "search_query")

# Cross-lingual similarity works!
# French query retrieves English kerning document
```

### Self-hosted BGE-M3

If you need cross-lingual without API costs:

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def embed_bge_m3(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    return embeddings["dense_vecs"].tolist()

# Works across languages
japanese_query = "タイポグラフィとは何ですか"
english_doc = "Typography is the art of arranging type"

q_emb = embed_bge_m3([japanese_query])
d_emb = embed_bge_m3([english_doc])

# High similarity despite different languages
```

### When cross-lingual matters

| Use case | Need cross-lingual? |
|----------|---------------------|
| Internal docs, English org | No |
| Global product docs | Yes |
| Customer support multilingual | Yes |
| Academic paper search | Yes |
| Code documentation | Maybe (code is universal, comments vary) |

---

## Technique 4: embedding compression

**The problem**: Even with Matryoshka, embeddings at scale are expensive. A billion vectors at 768 dimensions, 4 bytes per float: 3 TB.

**The solution**: Quantization and binary embeddings.

### Scalar quantization

Convert 32-bit floats to 8-bit integers:

```python
import numpy as np

def quantize_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Quantize 32-bit float embeddings to 8-bit integers.
    Returns quantized embeddings and calibration data for dequantization.
    """
    # Find min/max per dimension
    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)

    # Scale to 0-255
    scale = 255.0 / (maxs - mins + 1e-8)
    quantized = ((embeddings - mins) * scale).astype(np.uint8)

    calibration = {"mins": mins, "scale": scale}
    return quantized, calibration

def dequantize(quantized: np.ndarray, calibration: dict) -> np.ndarray:
    """Restore approximate original embeddings."""
    return (quantized / calibration["scale"]) + calibration["mins"]

# Compress
embeddings = np.array(your_embeddings, dtype=np.float32)  # 4 bytes per value
quantized, calibration = quantize_embeddings(embeddings)  # 1 byte per value

# 75% storage reduction, ~1-2% quality loss
```

### Binary embeddings

Even more aggressive: 1 bit per dimension.

```python
def binarize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Convert to binary: positive values become 1, negative become 0.
    Each 8 dimensions pack into 1 byte.
    """
    binary = (embeddings > 0).astype(np.uint8)
    # Pack 8 bits into each byte
    packed = np.packbits(binary, axis=1)
    return packed

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count different bits between two binary embeddings."""
    xor = np.bitwise_xor(a, b)
    return np.unpackbits(xor).sum()

# Usage
binary_embeddings = binarize_embeddings(embeddings)
# 768 dims: 768 bits = 96 bytes (vs 3072 bytes for float32)
# 32x compression!
```

### Storage comparison

1 million vectors at 768 dimensions:

| Format | Size | Quality Loss |
|--------|------|--------------|
| Float32 | 3 GB | 0% (baseline) |
| Float16 | 1.5 GB | ~0% |
| Int8 quantized | 768 MB | 1-2% |
| Binary | 96 MB | 5-10% |

### Hybrid quantization strategy

Combine techniques for best results:

```python
class QuantizedIndex:
    """
    Two-tier quantized search:
    1. Binary pre-filter (32x smaller, fast Hamming distance)
    2. Int8 re-rank (4x smaller than float32, accurate)
    """

    def __init__(self):
        self.binary_index = {}
        self.int8_index = {}
        self.calibration = None

    def index(self, doc_id: str, embedding: np.ndarray):
        self.binary_index[doc_id] = binarize_embeddings(embedding.reshape(1, -1))[0]

        if self.calibration is None:
            # Initialize calibration with first embedding
            self.calibration = {
                "mins": embedding.min(),
                "scale": 255.0 / (embedding.max() - embedding.min() + 1e-8)
            }

        quantized = ((embedding - self.calibration["mins"]) * self.calibration["scale"]).astype(np.uint8)
        self.int8_index[doc_id] = quantized

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[str]:
        # Binary pre-filter
        query_binary = binarize_embeddings(query_embedding.reshape(1, -1))[0]
        candidates = sorted(
            self.binary_index.keys(),
            key=lambda doc_id: hamming_distance(query_binary, self.binary_index[doc_id])
        )[:k * 10]

        # Int8 re-rank
        query_int8 = ((query_embedding - self.calibration["mins"]) * self.calibration["scale"]).astype(np.uint8)

        def int8_similarity(doc_id):
            # Dot product works on int8 (overflow-safe with int32 accumulator)
            return np.dot(query_int8.astype(np.int32), self.int8_index[doc_id].astype(np.int32))

        return sorted(candidates, key=int8_similarity, reverse=True)[:k]
```

---

## Technique 5: late interaction (ColBERT)

**The problem**: Single-vector embeddings compress an entire document into one point. Nuances get lost.

**The solution**: Generate one vector per token, then compute similarity as the sum of maximum similarities.

### How ColBERT works

```
Traditional embedding:
Document → [single 768-dim vector]
Query    → [single 768-dim vector]
Score    = cosine(doc_vec, query_vec)

ColBERT late interaction:
Document → [[vec1], [vec2], ..., [vecN]]  (one per token)
Query    → [[q1], [q2], ..., [qM]]        (one per token)
Score    = sum of max_j(cosine(qi, docj)) for each qi
```

### Implementation with RAGatouille

```python
from ragatouille import RAGPretrainedModel

# Initialize ColBERT
colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index (slower than single-vector, stores more data)
colbert.index(
    collection="typography",
    documents=[
        "Kerning adjusts spacing between specific letter pairs like AV and To",
        "Tracking uniformly adjusts spacing across all characters in a block",
        "Leading controls the vertical space between lines of text"
    ],
    index_name="type_index"
)

# Search (finds exact term matches better than dense embeddings)
results = colbert.search("AV kerning pairs", k=3)
# ColBERT excels because it matches "AV" token directly
```

### When ColBERT beats dense embeddings

| Query type | Dense | ColBERT |
|------------|-------|---------|
| "what is kerning" | Good | Good |
| "AV pair adjustment" | Poor (rare terms) | Excellent |
| "error E_AUTH_403" | Poor | Excellent |
| "function calculate_metrics" | Poor | Excellent |

**Rule of thumb**: If your queries contain rare terms, identifiers, or exact phrases, ColBERT helps.

---

## Putting it together

A production embedding pipeline might combine several techniques:

```python
class ProductionEmbeddingPipeline:
    """
    Production-ready embedding pipeline:
    1. Domain fine-tuned model
    2. Matryoshka for dimension flexibility
    3. Quantization for storage efficiency
    4. ColBERT for exact match queries
    """

    def __init__(self, use_colbert: bool = False):
        # Fine-tuned model (hypothetical path)
        self.model = SentenceTransformer("./my-fine-tuned-model")

        # Quantization calibration
        self.calibration = None

        # Optional ColBERT
        self.colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") if use_colbert else None

    def embed(self, texts: list[str], dimensions: int = 768, quantize: bool = True) -> np.ndarray:
        # Get embeddings from fine-tuned model
        embeddings = self.model.encode(texts)

        # Matryoshka truncation
        embeddings = embeddings[:, :dimensions]

        # Quantization
        if quantize:
            embeddings, self.calibration = quantize_embeddings(embeddings)

        return embeddings

    def hybrid_search(self, query: str, dense_index, colbert_index=None, k: int = 10):
        # Dense search
        query_emb = self.embed([query])
        dense_results = dense_index.search(query_emb, k=k*2)

        # ColBERT rerank (if enabled)
        if self.colbert and colbert_index:
            colbert_results = self.colbert.search(query, k=k*2)
            # RRF fusion
            return self._rrf_fusion([dense_results, colbert_results], k)

        return dense_results[:k]
```

---

## The cost/benefit matrix

| Technique | Implementation effort | Quality gain | Storage impact |
|-----------|----------------------|--------------|----------------|
| Domain fine-tuning | High | 10-40% | None |
| Matryoshka | Low | -2-5% | -50-75% |
| Cross-lingual | Low | +multilingual | None |
| Quantization | Medium | -1-10% | -75-97% |
| ColBERT | Medium | +exact match | +10x |

## The takeaway

Off-the-shelf embeddings work for most cases. But when they don't:

1. **Domain-specific terminology?** Fine-tune.
2. **Storage constraints?** Matryoshka + quantization.
3. **Multilingual users?** Cross-lingual models.
4. **Exact term matching?** Add ColBERT.

Start simple. Measure. Add complexity only when measurements prove it helps.

The best embedding system isn't the most sophisticated—it's the one that actually improves your retrieval quality on your data for your users.

![Nested Russian dolls (matryoshka) with binary numbers visible inside each layer, mathematical vectors radiating outward, technical illustration style](https://pixy.vexy.art/)

---

*Next: Chapter 9 explores agents—LLMs that plan, execute, and iterate autonomously.*
