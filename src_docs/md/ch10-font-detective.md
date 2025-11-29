# Chapter 10: Case study — the font detective

*Building a font identification system with RAG and embeddings.*

> "Type is a beautiful group of letters, not a group of beautiful letters."
> — Matthew Carter

## The problem

A client emails you a screenshot. "What font is this?" They've cropped a single word from a billboard, taken at an angle, slightly blurred. They need to know by tomorrow for a rebrand presentation.

You could spend an hour squinting at serifs. Or you could build a system that does it for you.

This case study walks through building a font identification tool using:

- **Embeddings** to encode font visual characteristics
- **RAG** to retrieve similar fonts from a database
- **An agent** to refine results based on context

By the end, you'll have a working system that identifies fonts from descriptions or images.

## Imagine...

Imagine you're a detective specializing in forgeries. Someone hands you a ransom note. The words don't matter—what matters is the typeface. Is that an "a" with a single story or double? Are the serifs bracketed or wedge-shaped? Is the stroke contrast high or low?

Each of these details narrows the search. Single-story "a"? Probably a geometric sans-serif. Bracketed serifs? Old-style text face. High stroke contrast? Display type, likely Didone.

This is how font identification works: accumulate constraints until only a few candidates remain. The human version requires years of visual training. The machine version requires:

1. A database of fonts with searchable characteristics
2. A way to extract characteristics from an unknown sample
3. A method to find the closest matches

That's exactly what we're building. A font detective that never forgets a letterform.

---

## Architecture overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Input     │────►│  Font Detective  │────►│  Identification │
│  (image/text)   │     │  Agent           │     │  Results        │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │ Vision  │  │ Vector  │  │ Font    │
              │ API     │  │ Search  │  │ Metadata│
              └─────────┘  └─────────┘  └─────────┘
```

**Components:**

1. **Vision API**: Extract visual characteristics from font images
2. **Vector Database**: Store embeddings of font descriptions
3. **Font Metadata**: Technical specs and classifications
4. **Agent**: Orchestrate queries and refine results

## Step 1: building the font database

First, we need a searchable database of fonts with their characteristics.

### Gathering Font Data

```python
from pathlib import Path
from fontTools.ttLib import TTFont
from pydantic import BaseModel

class FontMetadata(BaseModel):
    name: str
    family: str
    classification: str  # serif, sans-serif, monospace, etc.
    weight: int
    style: str  # regular, italic, bold, etc.
    characteristics: list[str]  # ["high x-height", "open counters", etc.]
    designer: str | None
    year: int | None
    path: str

def extract_font_metadata(font_path: Path) -> FontMetadata:
    """Extract metadata from a font file."""
    font = TTFont(font_path)
    name_table = font["name"]

    def get_name(name_id: int) -> str:
        for platform_id in [3, 1, 0]:
            record = name_table.getName(name_id, platform_id, 1, 1033)
            if record:
                return record.toUnicode()
        return ""

    # Extract basic info
    family = get_name(1)
    subfamily = get_name(2)
    full_name = get_name(4)
    designer = get_name(9) or None

    # Determine weight from subfamily
    weight_map = {
        "thin": 100, "extralight": 200, "light": 300,
        "regular": 400, "medium": 500, "semibold": 600,
        "bold": 700, "extrabold": 800, "black": 900
    }
    weight = 400
    for w_name, w_value in weight_map.items():
        if w_name in subfamily.lower():
            weight = w_value
            break

    # Determine style
    style = "italic" if "italic" in subfamily.lower() else "regular"

    # Classification from OS/2 table
    os2 = font.get("OS/2")
    classification = "unknown"
    if os2:
        family_class = os2.sFamilyClass >> 8
        class_map = {
            1: "oldstyle-serif", 2: "transitional-serif",
            3: "modern-serif", 4: "clarendon-serif",
            5: "slab-serif", 7: "freeform-serif",
            8: "sans-serif", 9: "ornamental",
            10: "script", 12: "symbolic"
        }
        classification = class_map.get(family_class, "unknown")

    return FontMetadata(
        name=full_name,
        family=family,
        classification=classification,
        weight=weight,
        style=style,
        characteristics=[],  # Will be enriched later
        designer=designer,
        year=None,  # Not in font file, need external data
        path=str(font_path)
    )
```

### Enriching with LLM-Generated Characteristics

Font files don't contain descriptions like "elegant" or "modern." We generate those.

```python
from openai import OpenAI

client = OpenAI()

def generate_characteristics(font: FontMetadata) -> list[str]:
    """Use an LLM to describe font characteristics."""

    prompt = f"""Describe the visual characteristics of the font "{font.name}"
({font.classification}, {font.weight} weight, {font.style}).

List 5-10 visual characteristics in short phrases. Focus on:
- Letterform shapes (x-height, counters, stroke contrast)
- Personality (modern, classical, playful, serious)
- Best use cases (body text, headlines, signage)

Return as a JSON array of strings."""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    import json
    result = json.loads(response.choices[0].message.content)
    return result.get("characteristics", [])
```

### Creating Embeddings

```python
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize clients
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
qdrant = QdrantClient(":memory:")  # Use persistent storage in production

# Create collection
qdrant.create_collection(
    collection_name="fonts",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

def embed_text(text: str) -> list[float]:
    """Generate embedding using Google Gemini."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

def index_font(font: FontMetadata, font_id: int):
    """Index a font in the vector database."""
    # Create searchable description
    description = f"""
    {font.name}: A {font.classification} typeface.
    Weight: {font.weight}. Style: {font.style}.
    Characteristics: {', '.join(font.characteristics)}.
    Designer: {font.designer or 'Unknown'}.
    """

    embedding = embed_text(description)

    qdrant.upsert(
        collection_name="fonts",
        points=[PointStruct(
            id=font_id,
            vector=embedding,
            payload={
                "name": font.name,
                "family": font.family,
                "classification": font.classification,
                "characteristics": font.characteristics,
                "path": font.path
            }
        )]
    )
```

### Building the Index

```python
def build_font_index(font_dirs: list[Path]) -> int:
    """Index all fonts in the given directories."""
    font_id = 0

    for font_dir in font_dirs:
        for font_path in font_dir.glob("**/*.[ot]tf"):
            try:
                # Extract metadata
                font = extract_font_metadata(font_path)

                # Generate characteristics
                font.characteristics = generate_characteristics(font)

                # Index
                index_font(font, font_id)
                font_id += 1
                print(f"Indexed: {font.name}")

            except Exception as e:
                print(f"Failed: {font_path} - {e}")

    return font_id

# Build index from system fonts
indexed = build_font_index([
    Path("/Library/Fonts"),
    Path.home() / "Library/Fonts"
])
print(f"Indexed {indexed} fonts")
```

## Step 2: the search engine

With fonts indexed, we can search by description.

### Basic Similarity Search

```python
def search_fonts(query: str, limit: int = 10) -> list[dict]:
    """Search for fonts matching the query."""
    query_embedding = embed_text(query)

    results = qdrant.search(
        collection_name="fonts",
        query_vector=query_embedding,
        limit=limit
    )

    return [
        {
            "name": hit.payload["name"],
            "family": hit.payload["family"],
            "classification": hit.payload["classification"],
            "characteristics": hit.payload["characteristics"],
            "score": hit.score
        }
        for hit in results
    ]

# Example
results = search_fonts("elegant serif with high contrast for luxury branding")
for r in results[:3]:
    print(f"{r['name']}: {r['score']:.3f}")
```

### Filtered Search

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def search_fonts_filtered(
    query: str,
    classification: str | None = None,
    limit: int = 10
) -> list[dict]:
    """Search with optional classification filter."""
    query_embedding = embed_text(query)

    # Build filter
    filter_conditions = None
    if classification:
        filter_conditions = Filter(
            must=[FieldCondition(
                key="classification",
                match=MatchValue(value=classification)
            )]
        )

    results = qdrant.search(
        collection_name="fonts",
        query_vector=query_embedding,
        query_filter=filter_conditions,
        limit=limit
    )

    return [{"name": hit.payload["name"], "score": hit.score} for hit in results]

# Search only sans-serif fonts
results = search_fonts_filtered(
    "clean modern geometric",
    classification="sans-serif"
)
```

## Step 3: vision-based identification

Text queries are useful, but users often have images. We need vision capabilities.

### Extracting Font Characteristics from Images

```python
import base64
from openai import OpenAI

client = OpenAI()

def analyze_font_image(image_path: str) -> str:
    """Extract font characteristics from an image using GPT-5 Vision."""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this font image and describe its characteristics:

1. Classification (serif, sans-serif, script, etc.)
2. Weight (thin, regular, bold, etc.)
3. Visual characteristics (x-height, stroke contrast, etc.)
4. Personality (modern, classical, playful, etc.)
5. Similar well-known fonts

Be specific and technical. This will be used to search a font database."""
                },
                {
                    "type": "image_url",
                    "url": {"url": f"data:image/png;base64,{image_data}"}
                }
            ]
        }],
        max_tokens=500
    )

    return response.choices[0].message.content
```

### Image-to-Font Pipeline

```python
def identify_font_from_image(image_path: str) -> list[dict]:
    """Identify fonts from an image."""

    # Step 1: Analyze image
    analysis = analyze_font_image(image_path)
    print(f"Analysis:\n{analysis}\n")

    # Step 2: Search based on analysis
    results = search_fonts(analysis)

    return results

# Example usage
matches = identify_font_from_image("unknown_font.png")
print("Top matches:")
for match in matches[:5]:
    print(f"  {match['name']}: {match['score']:.3f}")
```

## Step 4: the detective agent

Wrap everything in an agent for interactive use.

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class FontIdentification(BaseModel):
    identified_font: str | None
    confidence: float
    alternatives: list[str]
    reasoning: str

detective = Agent(
    model="claude-sonnet-4",
    result_type=FontIdentification,
    system_prompt="""You are the Font Detective, an expert at identifying typefaces.

When given a description or image analysis, you:
1. Search the font database for matches
2. Evaluate results based on the query
3. Consider context (where the font was seen, its purpose)
4. Provide your best identification with confidence level

Be honest about uncertainty. Font identification is difficult."""
)

@detective.tool
async def search_font_database(query: str) -> str:
    """Search the indexed font database."""
    results = search_fonts(query, limit=10)
    return "\n".join([
        f"- {r['name']} (score: {r['score']:.2f}): {', '.join(r['characteristics'][:3])}"
        for r in results
    ])

@detective.tool
async def get_font_details(font_name: str) -> str:
    """Get detailed information about a specific font."""
    # Search for exact match
    results = search_fonts(font_name, limit=1)
    if results:
        r = results[0]
        return f"""
Font: {r['name']}
Family: {r['family']}
Classification: {r['classification']}
Characteristics: {', '.join(r['characteristics'])}
"""
    return f"Font '{font_name}' not found in database."

@detective.tool
async def analyze_image(image_path: str) -> str:
    """Analyze a font image to extract characteristics."""
    return analyze_font_image(image_path)
```

### Using the Detective

```python
async def identify_font(query: str, image_path: str | None = None) -> FontIdentification:
    """Main entry point for font identification."""

    if image_path:
        query = f"Identify the font in this image: {image_path}\n\nUser context: {query}"

    result = await detective.run(query)
    return result.data

# Text-based query
result = await identify_font(
    "I saw this elegant serif font on a wine bottle label. "
    "Very high contrast, thin strokes, looks expensive."
)
print(f"Identified: {result.identified_font}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Reasoning: {result.reasoning}")

# Image-based query
result = await identify_font(
    "This was on a tech company's website",
    image_path="mystery_font.png"
)
```

## Step 5: deployment

### FastAPI Service

```python
from fastapi import FastAPI, File, UploadFile, Form
from tempfile import NamedTemporaryFile

app = FastAPI(title="Font Detective API")

@app.post("/identify")
async def identify_font_endpoint(
    description: str = Form(...),
    image: UploadFile | None = File(None)
):
    """Identify a font from description and/or image."""

    image_path = None
    if image:
        # Save uploaded image temporarily
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            image_path = tmp.name

    result = await identify_font(description, image_path)

    return {
        "identified_font": result.identified_font,
        "confidence": result.confidence,
        "alternatives": result.alternatives,
        "reasoning": result.reasoning
    }

@app.get("/search")
async def search_endpoint(query: str, limit: int = 10):
    """Search the font database."""
    results = search_fonts(query, limit)
    return {"results": results}
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build font index
RUN python build_index.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cost Analysis

For a production deployment handling 1,000 queries/day:

| Component | Monthly Cost |
|-----------|-------------|
| Embeddings (Gemini) | $0 (free tier) |
| Vision (GPT-5, 30% of queries) | ~$50 |
| Agent (Claude Sonnet) | ~$100 |
| Vector DB (Qdrant Cloud) | ~$50 |
| Compute (fly.io) | ~$20 |
| **Total** | ~$220/month |

## Lessons learned

### What Worked

1. **Hybrid search**: Combining embeddings with LLM analysis caught fonts that pure vector search missed.

2. **Generated characteristics**: LLM-written descriptions made the index more searchable than raw metadata alone.

3. **Agent orchestration**: The detective agent could ask follow-up questions and refine searches.

### What Didn't

1. **Vision accuracy**: GPT-5 sometimes misidentified fonts, especially for similar-looking alternatives. Always present multiple options.

2. **Index freshness**: New fonts required re-embedding. Solution: schedule nightly index updates.

3. **Cold starts**: First query took 3-4 seconds. Solution: pre-warm the vision model.

### Improvements for V2

- **Glyph-level comparison**: Extract specific glyphs (a, g, R) for more precise matching
- **Historical context**: Add font release dates for "this looks 1990s" queries
- **Font pairing**: Suggest complementary fonts for identified matches
- **Specimen generation**: Show sample text in identified fonts

## The complete code

```python
# font_detective.py - Complete implementation

import os
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
import base64

# Configuration
OPENAI_CLIENT = OpenAI()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
qdrant = QdrantClient(path="./font_db")  # Persistent storage

# Models
class FontMetadata(BaseModel):
    name: str
    family: str
    classification: str
    weight: int
    characteristics: list[str]
    path: str

class FontIdentification(BaseModel):
    identified_font: str | None
    confidence: float
    alternatives: list[str]
    reasoning: str

# Embedding
def embed(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

# Search
def search_fonts(query: str, limit: int = 10) -> list[dict]:
    results = qdrant.search(
        collection_name="fonts",
        query_vector=embed(query),
        limit=limit
    )
    return [hit.payload | {"score": hit.score} for hit in results]

# Vision
def analyze_image(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-5",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this font's visual characteristics."},
                {"type": "image_url", "url": {"url": f"data:image/png;base64,{data}"}}
            ]
        }]
    )
    return response.choices[0].message.content

# Agent
detective = Agent(
    model="claude-sonnet-4",
    result_type=FontIdentification,
    system_prompt="You are the Font Detective. Identify fonts accurately."
)

@detective.tool
async def search(query: str) -> str:
    results = search_fonts(query, 10)
    return "\n".join([f"- {r['name']} ({r['score']:.2f})" for r in results])

@detective.tool
async def analyze(image_path: str) -> str:
    return analyze_image(image_path)

# Main
async def identify(query: str, image: str | None = None) -> FontIdentification:
    if image:
        query = f"Image analysis: {analyze_image(image)}\n\nContext: {query}"
    result = await detective.run(query)
    return result.data
```

## Getting started: two approaches

**Simple approach**: Skip the infrastructure. Use a single API call.

```python
import base64
from openai import OpenAI

client = OpenAI()

def identify_font_simple(image_path: str) -> str:
    """Quick font identification using vision alone."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": """Identify this font. Be specific:
1. Font name (or closest match)
2. Classification (serif, sans-serif, etc.)
3. Similar alternatives
4. Confidence level (high/medium/low)"""},
                {"type": "image_url", "url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }]
    )
    return response.choices[0].message.content

# One call, immediate answer
result = identify_font_simple("mystery_font.png")
print(result)
```

No database, no embeddings, no agent. GPT-5 alone handles many cases. Costs ~$0.03 per image.

**Production approach**: The full system described in this chapter—embeddings, RAG, and an orchestrating agent—for accuracy and scale.

```python
# Production deployment summary
from font_detective import identify

async def production_identification(
    description: str,
    image_path: str | None = None
) -> dict:
    """Production font identification with full pipeline."""

    # 1. Analyze image (if provided)
    # 2. Search embedded font database
    # 3. Agent refines with follow-up queries
    # 4. Return structured result with confidence

    result = await identify(description, image_path)

    return {
        "identified": result.identified_font,
        "confidence": result.confidence,
        "alternatives": result.alternatives,
        "reasoning": result.reasoning
    }

# Handles edge cases, scales to thousands of queries
result = await production_identification(
    "Elegant serif seen on luxury wine label",
    "bottle_crop.png"
)
```

The simple approach works for occasional queries with clear images. The production approach handles ambiguous cases, scales efficiently, and improves over time as you add fonts to the database. Start simple; build infrastructure when accuracy matters.

## The takeaway

Building a font identification system taught us:

1. **Embeddings work for fonts**: Text descriptions of visual characteristics are surprisingly effective for similarity search.

2. **Vision + RAG > Vision alone**: Combining GPT-5's analysis with a curated database beats either approach alone.

3. **Agents add value for ambiguous queries**: When the user says "elegant serif," an agent can ask clarifying questions and iterate.

4. **Free tiers scale**: Google Gemini embeddings + Qdrant's free tier handle thousands of fonts without cost.

The Font Detective isn't magic—it's plumbing. Embeddings, RAG, vision, and an agent, all wired together. The same architecture works for any domain where you need to match descriptions to a catalog.

![A detective with a magnifying glass examining giant letterforms, with a wall of suspect fonts behind them, noir comic style](https://pixy.vexy.art/)

---

*Next: Chapter 11 builds an automated documentation generator using agents and RAG.*
