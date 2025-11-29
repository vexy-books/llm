# Work Log

## Current Status

**2nd Edition**: COMPLETE
**Live site**: https://vexy.boo/llm/

## Book Statistics

| Edition | Chapters | Words |
|---------|----------|-------|
| 1st Edition | 15 | ~37,000 |
| 2nd Edition (new) | 9 | ~28,000 |
| **Total** | **24** | **~65,000** |

## Recent Updates

- [x] Updated ch06 MCP server examples to FastMCP 2.0 style (`from fastmcp import FastMCP`, `@mcp.tool`, `mcp.run()`)
- [x] Updated ch12 low-level MCP SDK imports: `from mcp.server.lowlevel import Server`, `import mcp.server.stdio`
- [x] Fixed ch14 Gemini model names: Gemini 2.0 Pro/Flash → Gemini 3 Pro / Gemini 2.5 Flash with current pricing
- [x] Fixed ch02 Gemini 2.0 Flash-Lite → Gemini 2.5 Flash-Lite
- [x] Verified ch08 embedding model names and pricing are current (gemini-embedding-001, voyage-3.5-lite, text-embedding-3-small)
- [x] Fixed ch13/ch14 model names: claude-3-haiku → claude-3-5-haiku, gpt-4o-mini → gpt-5-mini, gemini-1.5-flash → gemini-2.5-flash
- [x] Fixed remaining PydanticAI `result.data` → `result.output` across 6 chapters (ch04, ch04b, ch09, ch09b, ch10, ch12b)
- [x] Updated GPT-4/GPT-4o references to GPT-5/GPT-5-mini across 11 chapters (ch02, ch03, ch04b, ch05, ch07b, ch08, ch08b, ch09, ch09b, ch10, ch15)
- [x] Updated ch06 FastMCP imports: `from mcp.server.fastmcp` → `from fastmcp` (FastMCP 2.0)
- [x] Updated ch06 setup: `uv add mcp` → `uv add fastmcp`
- [x] Updated ch06 decorator style: `@mcp.tool()` → `@mcp.tool` (cleaner)
- [x] Updated ch11 PydanticAI code: `result.data` → `result.output`
- [x] Fixed ch04 PydanticAI attribution (Pydantic team, not Anthropic)
- [x] Verified ch07 vector database APIs current (Pinecone, Qdrant, Chroma)
- [x] Verified ch08 embedding model info accurate (gemini-embedding-001, MTEB scores)
- [x] Updated Gemini models across all chapters: Gemini 3 Pro (flagship) + Gemini 2.5 Flash (speed)
- [x] Updated ch13 subscription model names (GPT-5, Claude 4.5, o3-Pro)
- [x] Updated ch15 trends with current models and shipped features (Operator, computer use)
- [x] Updated ch05 Codex CLI with Nov 2025 features (GPT-5.1-Codex-Max, approval modes)
- [x] Updated ch05 Claude Code models to Sonnet/Opus 4.5
- [x] Fixed ch14 pricing code: claude-3-haiku → claude-3-5-haiku with $0.80/$4.00
- [x] Updated ch02 provider info with Nov 2025 releases
- [x] Updated ch09 PydanticAI code: `result.data` → `result.output`
- [x] Added "Debugging agents in practice" section to ch09

## Next Actions

1. Test code examples more thoroughly (imports require API keys)
2. Continue improving the book per PLAN.md 3rd Edition guidance
