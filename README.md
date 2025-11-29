# The Vexy Book of Large Language Models

An accessible, humorous, and accurate guide to LLMs in late 2025.

## What This Is

A technical book with narrative flair. Crime thriller meets developer documentation. We investigate the LLM landscape like detectives hunting for the best solutions.

## Scope

| Topic | Coverage |
|-------|----------|
| **API Providers** | Free tiers, pricing, limits (Claude, GPT, Gemini, Grok, Perplexity) |
| **Python Libraries** | 100+ packages cataloged with battle-tested recommendations |
| **CLI Tools** | AIChat, Mods, Claude Code, Codex - command-line power |
| **RAG Systems** | 12 research reports, implementation patterns |
| **Agent Frameworks** | AutoGen, CrewAI, LangGraph, PydanticAI, Smolagents |
| **Embeddings** | Provider comparison, volume pricing, performance |
| **MCP Servers** | The new integration standard |

## Project Structure

```
llm/
├── src_docs/md/          # Book source (Markdown)
├── docs/                  # Built site (GitHub Pages)
├── private/              # Research vault
│   ├── private/          # Full source material
│   ├── private2/         # File-by-file TLDRs
│   └── private6.txt      # Master map (1,500+ paths)
├── .claude/              # Claude Code scaffolding
│   ├── agents/           # Specialized writing agents
│   └── skills/           # Reusable writing skills
├── zensical.toml         # Site generator config
├── PLAN.md               # Book planning document
└── TODO.md               # Active task list
```

## Source Material

Seven research areas feed this book:

1. **2510-free-ai-apis/** - Free API research across 6 providers
2. **embedding/** - Text embedding API deep-dive
3. **flatrate-ai/** - LLM subscription value analysis
4. **llm_packages/** - 100+ Python library catalog
5. **llm_research/** - CLI tools (200s) + Python libs (300s) + code tests
6. **rag-research/** - 12 RAG implementation reports
7. **vexy-co-model-catalog/** - Working 40+ provider catalog tool

## The Process

```
Plan the Planning of the Planning (meta-strategy)
    ↓
Plan the Planning (reading strategy)
    ↓
Plan the Implementation (book structure)
    ↓
Implement (write chapters)
    ↓
Iterate (revise, reflect, refine)
```

## Writing Rules

- **First line sells the second line** - No throat-clearing
- **Show, don't tell** - "Sarah picked at her cuticles" beats "Sarah was nervous"
- **Kill your darlings** - If it doesn't serve the reader, delete it
- **Trust the reader** - Stop explaining what you just showed
- **Light humor allowed** - But clarity comes first

## Case Studies

The book includes practical projects themed around typography and design:

1. **The Font Detective** - Using LLMs to analyze typeface characteristics
2. **The Vexy Lines Mystery** - Automating graphic design workflows
3. **The MCP Server Heist** - Building custom integrations

## Building the Book

```bash
# Install dependencies
pip install zensical

# Preview locally
zensical serve

# Build static site
zensical build --clean
```

## Deployment

GitHub Actions builds on push to main, deploying to GitHub Pages from `/docs`.

## License

Research and documentation project by Vexy Books.
