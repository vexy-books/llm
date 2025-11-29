# The Vexy Book of LLMs - Master Plan

## Meta-Level Planning

### Level 3: Plan the Planning of the Planning

**Question**: How do we approach understanding 1,500+ source files to write a coherent book?

**Strategy**: Tiered reading with progressive refinement

```
Layer 0: private6.txt        → Master map (done)
Layer 1: private2/ TLDRs     → Category summaries
Layer 2: private/ selected   → Deep-dive key sections
Layer 3: Synthesis           → Extract book-worthy content
```

### Level 2: Plan the Planning (Reading Strategy)

**Phase 1: Survey (Day 1)**
- [ ] Read all TLDR files in private2/ by category
- [ ] Create category-level summaries
- [ ] Identify top 20% of content (Pareto)

**Phase 2: Deep Dive (Days 2-3)**
- [ ] Read full source for high-value sections
- [ ] Extract code examples that work
- [ ] Document provider-specific gotchas

**Phase 3: Synthesis (Day 4)**
- [ ] Map content to book chapters
- [ ] Identify narrative threads
- [ ] Flag gaps needing additional research

### Level 1: Implementation Plan (Book Structure)

## Book Structure

### Part I: The Landscape (Chapters 1-2)

#### Chapter 1: The LLM Gold Rush
*Hook: "The font file opened, and the kerning was all wrong. But this time, I had an army of language models to help me figure out why."*

- What LLMs actually are (no hype)
- The December 2025 landscape
- Why this matters for developers

**Source**: llm_research/md/001-intro.md

#### Chapter 2: The Provider Wars
*The big five face off. Your wallet hangs in the balance.*

- Claude (Anthropic) - The safety-first contender
- GPT (OpenAI) - The incumbent
- Gemini (Google) - The multimodal maverick
- Grok (xAI) - The real-time rebel
- Perplexity - The search-native upstart

**Source**: 2510-free-ai-apis/, flatrate-ai/

---

### Part II: The Arsenal (Chapters 3-6)

#### Chapter 3: Free APIs - The Budget Detective's Guide
*You don't need to go broke to go smart.*

- Free tier comparison matrix
- Rate limits and how to live within them
- The "good enough" threshold

**Source**: 2510-free-ai-apis/03-best/

#### Chapter 4: The Python Package Encyclopedia
*122+ libraries. All cataloged. Your reference guide.*

Comprehensive coverage organized by category:

**4.1 Full-Stack Platforms** (10 packages)
AgentForge, AGiXT, Archon, BeeAI, Dify, Julep, OpenAGI, OpenAI Agents SDK, SuperAGI, uAgents

**4.2 General-Purpose Frameworks** (13 packages)
Adala, AdalFlow, AgentScope, agentUniverse, Agno, Atomic Agents, Griptape, LangChain, LangGraph, Langroid, PocketFlow, Semantic Kernel, smolagents

**4.3 Multi-Agent Systems** (10 packages)
AgentVerse, AutoGen, CAMEL, ChatDev, CrewAI, MetaGPT, OWL, Swarm, swarms, TinyTroupe

**4.4 Coding Specialists** (20 packages)
Agent Zero, any-agent, AutoAgent, AutoGPT, DeepAgents, DeerFlow, E2B, FastAgency, Google ADK, Kimi CLI, Microsoft Agent Framework, Open Interpreter, OpenCodeInterpreter, OpenHands, TaskWeaver, Trae Agent, Upsonic, XAgent, Youtu-Agent

**4.5 Workflow Engines** (6 packages)
Airflow, ControlFlow, Dagster, Flyte, Orchest, Temporal

**4.6 LLM Clients & Structured Output** (21 packages)
AIConfig, DSPy, Guidance, Instructor, LiteLLM, LMQL, Magentic, Marvin, Mirascope, OpenAI Python, Outlines, Promptify, prompttools, Prompty, PydanticAI, pydantic-ai-retry-fallback, Pydantic AI Scaffolding, SGLang, POML, BAML

**4.7 RAG, Memory & Knowledge** (19 packages)
Cognee, Cognita, FlashRAG, GraphBit, graphiti, GraphRAG, Haystack, LanceDB, langmem, Letta, LightRAG, Llama Stack, LlamaIndex, llmware, mem0, Parlant, RAG-Anything, txtai, Zep

**4.8 Tool & Browser Layers** (10 packages)
aiohttp, aiohttp-pydantic, aiohttp-security, aiohttp-session, browser-use, Composio, E2B, LaVague, portkey-ai

**4.9 Observability & Evaluation** (8 packages)
genai-processors, Helicone, Inspect AI, MLflow, PromptFlow, Promptfoo, R2R, Ragas

**4.10 Domain-Specific** (5 packages)
baml-examples, Motia, poml, spaCy, spaCy-LLM

**Source**: llm_packages/02-tldr/, llm_packages/03-uldr.md, llm_packages/06-packages.md

#### Chapter 5: CLI Tools for the Command-Line Cowboy
*The terminal is your friend. These tools make it friendlier.*

- Claude Code - The Anthropic way
- Codex CLI - OpenAI's Rust-powered agent
- AIChat - The versatile configuration king
- Mods - Unix pipeline integration

**Source**: llm_research/md/203-codex.md, 206-aichat.md, 207-mods.md

#### Chapter 6: The MCP Protocol
*How Claude Code actually talks to the world.*

- What MCP servers are
- Building your first server
- Existing servers worth using

**Source**: Research needed, vexy-co-model-catalog/external/reference/

---

### Part III: Building Things (Chapters 7-9)

#### Chapter 7: RAG - When LLMs Need to Read
*The art of giving language models the right context.*

- RAG architecture explained
- Vector stores compared (Qdrant, Pinecone, Chroma)
- Chunking strategies that don't suck

**Source**: rag-research/ragres-01.md through ragres-12.md

#### Chapter 8: Embeddings - The Secret Sauce
*Turning text into math. More useful than it sounds.*

- Provider comparison (Google, OpenAI, Voyage, Cohere)
- Dimension trade-offs
- Batch processing for volume

**Source**: embedding/emb-*.md

#### Chapter 9: Agents - When LLMs Get Autonomous
*Teaching your models to think in loops.*

- ReAct pattern explained
- Code execution sandboxing
- Multi-agent orchestration

**Source**: llm_packages/02-tldr/autogen.md, crewAI.md, smolagents.md

---

### Part IV: Case Studies (Chapters 10-12)

#### Chapter 10: The Font Detective
*Using LLMs to analyze typeface characteristics and find similar fonts.*

- Building a font embedding system
- Querying visual characteristics with text
- Vexy Lines integration demo

**Narrative**: A designer receives a mysterious PDF with an unidentified font. They build an LLM-powered system to identify it.

#### Chapter 11: The Documentation Generator
*From code to docs without the suffering.*

- Analyzing codebases with Claude
- Generating accurate API documentation
- Keeping docs in sync

**Narrative**: A legacy codebase with zero documentation. Three days to understand it.

#### Chapter 12: The MCP Server Heist
*Building custom integrations that feel like magic.*

- Designing a FontLab MCP server
- Connecting to design tools
- Real-time glyph manipulation

**Narrative**: Breaking into (your own) creative workflow with automation.

---

### Part V: The Smart Money (Chapters 13-15)

#### Chapter 13: Subscription Showdown
*Is $20/month worth it? Depends.*

- ChatGPT Plus vs Claude Pro vs Gemini Advanced
- API credits vs subscriptions
- When to upgrade

**Source**: flatrate-ai/report2.md

#### Chapter 14: Cost Optimization Wizardry
*Spend less, get more. The boring chapter that saves you money.*

- Prompt caching
- Batch processing
- Fallback strategies

**Source**: 2510-free-ai-apis/03-best/04.md

#### Chapter 15: What's Next
*The crystal ball chapter. Take with salt.*

- Trends we're watching
- Models on the horizon
- The open source factor

---

## Narrative Thread

**The Red Thread**: A designer/developer investigating how to automate their typography workflow. Each chapter advances both the technical content and their journey.

**Tone**: Technical accuracy with personality. Like a smart friend explaining things over coffee, occasionally cracking jokes but never at the expense of precision.

## Image Placeholders

Use the pattern: `![alt text with prompt](https://pixy.vexy.art/)`

Examples:
- `![A detective examining font specimens with a magnifying glass, noir style](https://pixy.vexy.art/)`
- `![Five LLM providers as chess pieces on a board](https://pixy.vexy.art/)`

## Source Priority Matrix

| Source | Priority | Est. Pages |
|--------|----------|------------|
| llm_packages/02-tldr/ | High | 40 |
| llm_research/md/ | High | 30 |
| 2510-free-ai-apis/ | High | 25 |
| rag-research/ | Medium | 20 |
| embedding/ | Medium | 10 |
| flatrate-ai/ | Medium | 10 |
| vexy-co-model-catalog/ | Reference | 5 |

**Total estimated**: ~140 pages of content

## Build & Deploy

**Local build**:
```bash
./build.sh  # or: zensical build --clean
```

**Preview**:
```bash
zensical serve
```

**Production**: GitHub Actions (`.github/workflows/docs.yml`) builds on push to main and deploys to GitHub Pages.

**Live site**: https://vexy.boo/llm/

# 2nd Edition Plan

## Goals
- **Double** the content (~75,000 words target)
- **Deepen** existing chapters with integration examples
- **Add** metaphoric explanations ("Imagine...")
- **Include** quotes from smart women and men
- **Alternate** between "average Joe" and "Stephen Fry" examples

---

## New Chapters (2nd Edition Additions)

### Part II Expansion: The Arsenal Extended

**Chapter 4B: Package Integration Cookbook**
*When one library isn't enough.*

- LiteLLM + Instructor (unified API with structured output)
- PydanticAI + LangGraph (type-safe agents with state machines)
- CrewAI + E2B (multi-agent with sandboxed execution)
- RAG-Anything + Qdrant (production RAG stack)

**Chapter 5B: Advanced CLI Workflows**
*Power user patterns.*

- Chaining Claude Code with MCP servers
- AIChat as local API gateway
- Building custom shell integrations
- Prompt templates and aliases

### Part III Expansion: Building Things Extended

**Chapter 7B: RAG Patterns Deep Dive**
*Beyond basic retrieval.*

- Hybrid search (BM25 + semantic)
- GraphRAG implementation with Neo4j
- Multi-vector retrieval
- Query routing and rewriting

**Chapter 8B: Embedding Engineering**
*The dark arts of vector spaces.*

- Fine-tuning embeddings for domain
- Matryoshka embeddings
- Cross-lingual retrieval
- Embedding compression

**Chapter 9B: Multi-Agent Architectures**
*When one agent isn't enough.*

- Hierarchical delegation patterns
- Debate and consensus protocols
- Agent memory sharing
- Production debugging

### Part IV Expansion: More Case Studies

**Chapter 10B: The Typography Classifier**
*Machine learning meets letterforms.*

- Training a font classifier
- Visual embedding with CLIP
- Automated specimen generation

**Chapter 12B: The Real-Time Collaborator**
*Live font editing with AI.*

- WebSocket MCP servers
- Streaming glyph modifications
- Collaborative editing protocol

### Part VI: New Section - The Philosophy

**Chapter 16: How LLMs Actually Work**
*No math required. Just understanding.*

Metaphoric explanations:
- "Imagine a library where the librarian has read every book..."
- "Think of embeddings as GPS coordinates for meaning..."
- "RAG is like giving the model an open-book test..."

**Chapter 17: The Ethics of AI-Assisted Creation**
*When the machine becomes the collaborator.*

- Attribution and authorship
- The environmental cost
- Dependency and skill atrophy

---

## Chapter Enhancement Plan

### Every Chapter Gets:

1. **Opening quote** - From literature, science, or philosophy
2. **"Imagine..." section** - Metaphoric explanation
3. **Two examples** - One simple (Joe), one sophisticated (Fry)
4. **Integration sidebar** - How this connects to other tools
5. **2025 update box** - Latest pricing/features

### Quote Sources to Mine:

**Women:**
- Ada Lovelace, Marie Curie, Grace Hopper
- Ursula K. Le Guin, Octavia Butler
- Hannah Arendt, Simone de Beauvoir
- Margaret Atwood, Zadie Smith

**Men:**
- Alan Turing, Claude Shannon
- Jorge Luis Borges, Italo Calvino
- Douglas Hofstadter, Marvin Minsky
- Neal Stephenson, Ted Chiang

**Contemporary:**
- Andrej Karpathy, Ilya Sutskever
- Timnit Gebru, Emily Bender
- Lex Fridman guests, Dwarkesh Patel interviews

---

## Metaphor Bank

**LLMs:**
- "A parrot with a photographic memory and no understanding"
- "A jazz musician who's heard every song but never lived"
- "The world's fastest librarian with a vivid imagination"

**Embeddings:**
- "GPS coordinates for meaning, not location"
- "Compressing Shakespeare to a zip code"
- "The smell of a word, converted to numbers"

**RAG:**
- "An open-book test for the AI"
- "Giving the model a cheat sheet"
- "The difference between knowing and remembering"

**Agents:**
- "A to-do list that checks itself off"
- "Delegation without the meeting"
- "Teaching a model to be impatient"

**MCP:**
- "A universal adapter for AI"
- "Teaching Claude to use tools like a human"
- "The nervous system connecting brain to hands"

---

## 2nd Edition Timeline

1. **Review 1st Edition** - Identify weak sections
2. **Research updates** - Latest pricing, features, packages
3. **Write new chapters** - 6 new chapters (~18,000 words)
4. **Enhance existing** - Add quotes, metaphors, examples
5. **Integration examples** - Code that combines packages
6. **Final polish** - Unified voice, consistent style 

Also remember to follow THE RULES OF PROSE WRITING:

## Hook and hold

- **First line sells the second line** – No throat-clearing
- **Enter late, leave early** – Start in action, end before over-explaining
- **Conflict creates interest** – What's at stake?

## Clarity above all

- **Embrace plain language** – Never use "utilize" when "use" works. Clarity is kindness to your reader
- **No corporate jargon** – Clear, concrete language only
- **Use active voice and strong verbs** – "John slammed the door" beats "The door was slammed by John"
- **Omit needless words** – Every sentence should either reveal character or advance action. If it doesn't, cut it
- **Avoid Title Case** — I hate it. Use sentence case even in headings. Remove existing Title Case. Reserve it solely for the title of the book. 

## Show, don’t tell

- **Show through action, not exposition** – Instead of "Sarah was nervous," write "Sarah picked at her cuticles until they bled"
- **Use specific details, not generic descriptions** – "A 1973 Plymouth Duster with a cracked windshield" beats "an old car"
- **Trust the reader's intelligence** – Stop explaining what you've just shown. Your reader doesn't need training wheels

## Focus your impact

- **One person, one problem** – Specific beats generic
- **Write for one reader, not everyone** – Pick your ideal reader and ignore everyone else
- **Transformation over features** – Show the change, not the tool

## Edit without mercy

- **Kill your darlings** – If it doesn't serve the reader, delete it
- **Skepticism is healthy** – Question everything, including this guide
- **Light humor allowed** – But clarity comes first

## Sell gently

- **Pain before gain** – Start with the problem they feel today, not the solution you're selling
- **Benefits trump features** – "Sleep through the night" beats "memory foam with 3-inch density"
- **Social proof early** – Third-party validation in the first third builds trust faster than any claim

## Explain clearly

- **Lead with the outcome** – Tell them what they'll accomplish before how to accomplish it
- **Progressive disclosure** – Basic usage first, advanced options later, edge cases last
- **Error messages are UX** – Write them like helpful directions, not system diagnostics

**The golden rule**: If the reader has to read it twice, you've failed once.


---

# 3rd Edition

When you finish 2nd Ed, work on 3rd, and then 4th. Each time: 

- Research facts and current numbers
- Annotate our text with citations when appropriate
- Challenge & double-check our claims
- Polish our metaphors
- Improve clarity and accuracy
- Add practical and actionable examples
- In addition to Python, explore the Rust & JS/TS/Node ecosystems (esp. Rust with Python interaction)
- Be educational
- Try to make this book a hit that readers will LOVE 

Grow the book. Re-read it. Brainstorm creatively. Add more image placeholders with useful prompts. Then plan & improve. 

