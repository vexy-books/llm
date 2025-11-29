# Chapter 7: RAG — when LLMs need to read

*The art of giving language models the right context.*

> "What we call knowledge is often merely the belief that we know."
> — Simone de Beauvoir

## The knowledge cutoff problem

Every LLM has a knowledge cutoff date. Ask GPT-5 about events after its training, and it'll confidently hallucinate. Ask Claude about your company's internal documentation, and it has nothing to say.

RAG (Retrieval-Augmented Generation) solves this by injecting relevant context at query time. Instead of relying solely on trained weights, the model receives external information alongside your question.

```
Traditional:
  Question → LLM → Answer (from training data only)

RAG:
  Question → Retrieve Relevant Docs → LLM + Context → Answer
```

This chapter covers RAG architecture patterns, vector database selection, and the platforms that make implementation straightforward.

## Imagine...

Imagine an exam. Closed-book exams test what's in your head—you either know the answer or you don't. Open-book exams are different. You might not remember the exact formula, but you know which chapter to flip to.

LLMs without RAG take closed-book exams. They answer from memory (training data). Sometimes they remember correctly. Sometimes they confidently make things up.

RAG gives the model an open book. When you ask about your company's authentication system, instead of guessing, the model flips to the relevant documentation. The answer still requires understanding—the model has to read and synthesize—but it's grounded in actual sources rather than vague recollection.

The trick is building a good index. A book with no table of contents is useless in an exam. RAG with poor retrieval is the same: the information exists, but you can't find it in time.

---

## RAG architecture patterns

### Pattern 1: Basic RAG

The simplest implementation. Works for most use cases.

```
Query → Embed → Vector Search → Top K Results → LLM + Context → Answer
```

```python
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI()
pc = Pinecone()
index = pc.Index("documents")

def basic_rag(query: str, k: int = 5) -> str:
    # 1. Embed the query
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # 2. Search vector database
    results = index.query(
        vector=embedding,
        top_k=k,
        include_metadata=True
    )

    # 3. Build context from results
    context = "\n\n".join([
        match.metadata["text"]
        for match in results.matches
    ])

    # 4. Generate with context
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": f"Use this context:\n{context}"},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content
```

**Pros**: Simple, fast, predictable costs
**Cons**: No reasoning about what to retrieve, limited by embedding quality

### Pattern 2: Hybrid RAG

Combines semantic search (embeddings) with keyword search (BM25). Better for domains with specific terminology.

```
Query → [Embed + Keyword Extract] → [Vector Search + BM25 Search]
      → Merge & Rerank → LLM + Context → Answer
```

```python
from sentence_transformers import CrossEncoder

def hybrid_rag(query: str, k: int = 10) -> str:
    # Semantic search
    semantic_results = vector_search(query, k=k)

    # Keyword search (BM25)
    keyword_results = bm25_search(query, k=k)

    # Merge results
    all_results = list(set(semantic_results + keyword_results))

    # Rerank with cross-encoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc.text] for doc in all_results]
    scores = reranker.predict(pairs)

    # Take top results
    ranked = sorted(zip(all_results, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:5]]

    # Generate
    return generate_with_context(query, top_docs)
```

**Pros**: Better recall, handles both conceptual and keyword queries
**Cons**: More complex, slower, requires two search systems

### Pattern 3: Agentic RAG

The LLM decides what to retrieve and when. Multi-step reasoning over documents.

```
Query → LLM (Plan Retrieval) → Search → LLM (Evaluate)
      → Need More? → [Yes: Search Again] / [No: Generate Answer]
```

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class SearchDecision(BaseModel):
    should_search: bool
    search_query: str | None
    reasoning: str

agent = Agent(
    model="claude-sonnet-4",
    system_prompt="""You are a research assistant with access to a document database.
    Before answering, decide if you need to search for information.
    You can search multiple times to gather complete information."""
)

@agent.tool
async def search_documents(query: str) -> str:
    """Search the document database."""
    results = vector_search(query, k=5)
    return format_results(results)

@agent.run
async def agentic_rag(question: str) -> str:
    # Agent autonomously decides when to search
    # Can search multiple times, refine queries
    pass
```

**Pros**: Handles complex queries, self-correcting, can gather multiple sources
**Cons**: Higher latency, unpredictable costs, requires capable model

### Pattern 4: GraphRAG

Uses knowledge graphs for relationship-aware retrieval. Microsoft's research shows 35% better performance on complex queries.

```
Query → Entity Extraction → Graph Traversal → Related Nodes
      → LLM + Graph Context → Answer
```

```python
from graphrag import GraphRAG

# Build knowledge graph from documents
graph = GraphRAG()
graph.index_documents(documents)

def graph_rag(query: str) -> str:
    # Extract entities from query
    entities = graph.extract_entities(query)

    # Traverse graph for related concepts
    context = graph.get_context(
        entities=entities,
        hops=2,  # How far to traverse
        max_nodes=50
    )

    # Generate with graph context
    return generate_with_context(query, context)
```

**Pros**: Understands relationships, better for "how does X relate to Y" queries
**Cons**: Complex setup, expensive indexing, overkill for simple retrieval

## Choosing a RAG architecture

| Query Type | Best Pattern |
|------------|--------------|
| Factual lookup | Basic RAG |
| Domain-specific terminology | Hybrid RAG |
| Multi-step reasoning | Agentic RAG |
| Relationship queries | GraphRAG |
| Simple Q&A | Basic RAG |

**Start with Basic RAG. Upgrade when you hit limits.**

## Vector databases compared

Your choice of vector database matters less than you think—LLM costs dominate. But here's the landscape:

### Pinecone

**Best for**: Production, serverless, managed

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("my-index")

# Upsert
index.upsert(vectors=[
    {"id": "doc1", "values": embedding, "metadata": {"text": "..."}}
])

# Query
results = index.query(vector=query_embedding, top_k=10)
```

- **Free tier**: 2GB storage, 2M writes/month
- **Pros**: Zero ops, scales automatically, excellent documentation
- **Cons**: Vendor lock-in, no local option

### Qdrant

**Best for**: Performance, hybrid search, self-hosted

```python
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")  # or url="http://localhost:6333"

# Create collection
client.create_collection(
    collection_name="docs",
    vectors_config={"size": 1536, "distance": "Cosine"}
)

# Search
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    limit=10
)
```

- **Free cloud tier**: 1GB (suspends after 7 days idle)
- **Pros**: Fast, supports filtering, local option, hybrid search built-in
- **Cons**: Cloud tier limitations, more setup than Pinecone

### Chroma

**Best for**: Prototyping, local development

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

# Add documents (auto-embeds)
collection.add(
    documents=["Document text..."],
    ids=["doc1"]
)

# Query
results = collection.query(query_texts=["My question"], n_results=5)
```

- **Free tier**: Unlimited (local)
- **Pros**: Zero setup, auto-embedding, great for prototypes
- **Cons**: Not production-ready for scale, limited features

### LanceDB

**Best for**: Local-first apps, embedded use

```python
import lancedb

db = lancedb.connect("./my-db")
table = db.create_table("docs", data=[
    {"text": "Document...", "vector": embedding}
])

results = table.search(query_embedding).limit(10).to_list()
```

- **Free tier**: Unlimited (embedded)
- **Pros**: No server needed, scales to billions, fast
- **Cons**: Less ecosystem support, newer project

### pgvector

**Best for**: Existing PostgreSQL users

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- Index for fast search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Query
SELECT * FROM documents
ORDER BY embedding <-> $1
LIMIT 10;
```

- **Free tier**: Whatever your Postgres costs
- **Pros**: No new infrastructure, SQL familiarity, transactions
- **Cons**: Performance ceiling, requires PostgreSQL expertise

### Decision Matrix

| Priority | Best Choice |
|----------|-------------|
| Zero ops | Pinecone |
| Performance | Qdrant |
| Prototyping | Chroma |
| Embedded/local | LanceDB |
| Existing Postgres | pgvector |
| Hybrid search | Qdrant or Pinecone |

## Low-code RAG platforms

Don't want to write code? These platforms provide RAG out of the box.

### Dify

**The standout choice.** Self-hosted or cloud. Visual workflow builder.

```yaml
# docker-compose.yml
services:
  dify:
    image: langgenius/dify
    ports:
      - "3000:3000"
    volumes:
      - ./data:/app/data
```

- **Cost**: $50-200/mo self-hosted (compute + LLM API)
- **Setup**: 1 hour to production
- **Features**: RAG, agents, workflow builder, API access
- **Best for**: Teams that need results fast

### Flowise

Open-source LangChain visual builder.

- **Cost**: $20-100/mo self-hosted
- **Setup**: 1 hour
- **Features**: Drag-and-drop flows, many integrations
- **Best for**: LangChain users who want visual building

### AnythingLLM

Complete self-hosted RAG solution.

- **Cost**: $20-50/mo self-hosted
- **Setup**: 1 hour (Docker)
- **Features**: Multi-user, document management, chat interface
- **Best for**: Teams needing a turnkey solution

### RAGFlow

Knowledge-graph enhanced RAG platform.

- **Cost**: Free (self-hosted)
- **Setup**: 4-6 hours
- **Features**: Deep document parsing, graph extraction
- **Best for**: Document-heavy use cases

## Chunking strategies

How you split documents affects retrieval quality more than most realize.

### Fixed-Size Chunking

Simple but effective. Split every N tokens with overlap.

```python
def fixed_chunks(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks
```

- **Pros**: Predictable, simple
- **Cons**: Breaks semantic units

### Semantic Chunking

Split at natural boundaries (paragraphs, sections, sentences).

```python
import re

def semantic_chunks(text: str, max_size: int = 500) -> list[str]:
    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_size:
            current_chunk += "\n\n" + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
```

- **Pros**: Preserves meaning
- **Cons**: Variable sizes, may miss cross-paragraph context

### Recursive Chunking

LangChain's default. Tries multiple separators in priority order.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document)
```

- **Pros**: Balance of semantic and size
- **Cons**: Can still break awkwardly

### The Right Chunk Size

| Content Type | Recommended Size | Overlap |
|--------------|-----------------|---------|
| Code | 1000-2000 tokens | 100-200 |
| Technical docs | 500-1000 tokens | 50-100 |
| Conversational | 200-500 tokens | 20-50 |
| Legal/dense | 300-500 tokens | 50-100 |

**When in doubt**: 500 tokens, 50 overlap.

## Cost breakdown

For a production RAG system handling 200 requests/minute:

| Component | Monthly Cost |
|-----------|--------------|
| **LLM API** | ~$3,500 (largest!) |
| Vector DB | $50-300 |
| Compute | $50-200 |
| Web search (optional) | $5-50 |
| **Total** | ~$4,000/mo |

**LLM costs dominate.** Optimizing your vector database won't matter if you're burning money on GPT-5 calls.

### Cost Optimization

1. **Route queries**: Simple questions → cheap models (Gemini Flash, Groq)
2. **Cache embeddings**: Don't re-embed unchanged documents
3. **Batch processing**: Embed in batches, not one at a time
4. **Smaller models**: text-embedding-3-small often suffices
5. **Fewer chunks**: Better chunking = fewer retrievals needed

## Production checklist

Before going live with RAG:

- [ ] **Test retrieval quality**: Sample queries, check relevance
- [ ] **Monitor latency**: Embedding + search + generation time
- [ ] **Set up caching**: Repeated queries shouldn't re-embed
- [ ] **Add fallbacks**: What happens when vector DB is down?
- [ ] **Track costs**: Per-query cost monitoring
- [ ] **Handle failures**: Graceful degradation, error messages
- [ ] **Update strategy**: How do you refresh documents?

---

> **Integration sidebar: RAG + other tools**
>
> RAG pairs naturally with other techniques from this book:
>
> - **RAG + Agents (Ch 9)**: Let agents decide when to retrieve and what queries to run. The agent observes results and refines searches automatically.
> - **RAG + MCP (Ch 6)**: Build MCP servers that expose your vector database as tools. Claude Code can then search your docs directly from the terminal.
> - **RAG + Embeddings (Ch 8)**: Custom embeddings trained on your domain outperform generic models. Chapter 8 covers when this investment pays off.
> - **RAG + CLI (Ch 5)**: Pipe documents through `llm embed` for quick indexing. Combine with `jq` for structured metadata extraction before ingestion.
>
> The pattern: RAG provides context, other tools provide capability. Start with basic RAG, then add orchestration as queries get complex.

---

## Getting started: two approaches

**Simple approach**: Use a managed platform. Zero infrastructure.

```python
# Dify provides RAG out of the box
import requests

DIFY_API = "https://api.dify.ai/v1"
API_KEY = "your-api-key"

def ask_dify(question: str) -> str:
    """Query your documents through Dify's hosted RAG."""
    response = requests.post(
        f"{DIFY_API}/chat-messages",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "inputs": {},
            "query": question,
            "response_mode": "blocking",
            "user": "user-1"
        }
    )
    return response.json()["answer"]

# Upload documents through Dify's UI, then query
answer = ask_dify("What fonts does our style guide recommend for body text?")
```

Upload your documents through the Dify web interface, point to your API, done. No vector databases, no embedding code, no chunking decisions.

**Production approach**: Build a RAG pipeline with hybrid search and reranking.

```python
# font_rag.py - Production RAG for typography documentation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from anthropic import Anthropic
import hashlib

class FontDocRAG:
    """Production RAG system for typography documentation."""

    def __init__(self, collection: str = "font_docs"):
        self.qdrant = QdrantClient(":memory:")  # or url for production
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = Anthropic()
        self.collection = collection

        # Create collection if needed
        self.qdrant.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def index_documents(self, documents: list[dict]) -> int:
        """Index documents with embeddings and metadata."""
        points = []
        for doc in documents:
            doc_id = hashlib.md5(doc["text"].encode()).hexdigest()
            embedding = self.embedder.encode(doc["text"]).tolist()
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": doc["text"], "source": doc.get("source", "")}
            ))

        self.qdrant.upsert(collection_name=self.collection, points=points)
        return len(points)

    def query(self, question: str, top_k: int = 10, final_k: int = 3) -> str:
        """Query with hybrid retrieval and reranking."""
        # Step 1: Embed query
        query_embedding = self.embedder.encode(question).tolist()

        # Step 2: Retrieve candidates
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k
        )

        if not results:
            return "No relevant documents found."

        # Step 3: Rerank with cross-encoder
        candidates = [r.payload["text"] for r in results]
        pairs = [[question, text] for text in candidates]
        scores = self.reranker.predict(pairs)

        # Step 4: Take top results after reranking
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        context = "\n\n---\n\n".join([text for text, _ in ranked[:final_k]])

        # Step 5: Generate answer with context
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Based on this documentation:

{context}

Question: {question}

Answer based only on the provided context. If the answer isn't in the context, say so."""
            }]
        )

        return response.content[0].text

# Usage
rag = FontDocRAG()
rag.index_documents([
    {"text": "Garamond is a serif typeface...", "source": "style-guide.md"},
    {"text": "For body text, use 16px minimum...", "source": "typography.md"},
])

answer = rag.query("What size should body text be?")
```

The simple approach gets you to production in an afternoon. The production approach gives you control over retrieval quality, reranking, and embedding models. Start with Dify; build custom when you need to optimize retrieval for your specific domain.

## The takeaway

RAG isn't magic—it's plumbing. You're building a pipeline from user questions to relevant documents to LLM-generated answers.

Start simple:
1. Pick a vector database (Pinecone for managed, Chroma for local)
2. Implement basic RAG
3. Measure what's actually wrong
4. Upgrade only when you hit limits

The fancy patterns (GraphRAG, Agentic RAG) solve real problems, but most applications don't have those problems yet. Build the boring version first.

![A plumber connecting pipes labeled 'Query', 'Embeddings', 'Vector DB', and 'LLM', with water (data) flowing through, technical illustration style](https://pixy.vexy.art/)

---

*Next: Chapter 8 digs into embeddings—the math that makes RAG possible.*
