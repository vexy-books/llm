# Work Log

## Current Status

**2nd Edition**: COMPLETE
**Live site**: https://vexy.boo/llm/
**Build**: `./build.sh` or `zensical build --clean`

## Book Statistics

| Edition | Chapters | Words |
|---------|----------|-------|
| 1st Edition | 15 | ~37,000 |
| 2nd Edition (new) | 9 | ~28,000 |
| **Total** | **24** | **~65,000** |

## Recent Updates (Nov 2025)

Model and API updates:
- Updated all model names to Nov 2025 generation (GPT-5, Claude 4.5, Gemini 3 Pro)
- Fixed MCP imports: FastMCP 2.0 (`from fastmcp`), low-level SDK (`mcp.server.lowlevel`)
- Updated PydanticAI API: `result.data` → `result.output`
- Verified embedding model pricing (gemini-embedding-001, voyage-3.5-lite, text-embedding-3-small)
- Updated Gemini pricing: Gemini 3 Pro ($2/$12), Gemini 2.5 Flash ($0.15/$3.50)

## Next Actions

1. Test code examples (imports require API keys)
2. Continue per PLAN.md 3rd Edition guidance
