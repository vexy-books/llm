# Write Chapter Command

Write a chapter for The Vexy Book of LLMs.

## Usage

```
/write-chapter [chapter-number]
```

## Process

1. **Load Context**
   - Read PLAN.md for chapter outline
   - Identify source files from private/
   - Check existing drafts in src_docs/md/

2. **Analyze Sources**
   - Use source-analyzer agent on identified files
   - Extract key content and code examples
   - Note gaps and questions

3. **Write Draft**
   - Use chapter-writer agent
   - Follow technical-writing skill guidelines
   - Include narrative thread elements

4. **Review**
   - Use prose-reviewer agent
   - Apply suggested fixes
   - Verify code examples work

5. **Save**
   - Write to src_docs/md/[chapter-slug].md
   - Update TODO.md status

## Chapter Mapping

| Ch | Title | Key Sources |
|----|-------|-------------|
| 1 | LLM Gold Rush | llm_research/md/001-intro.md |
| 2 | Provider Wars | 2510-free-ai-apis/ |
| 3 | Free APIs | 2510-free-ai-apis/03-best/ |
| 4 | Python Packages | llm_packages/02-tldr/ |
| 5 | CLI Tools | llm_research/md/203-207 |
| 6 | MCP Protocol | vexy-co-model-catalog/external/ |
| 7 | RAG | rag-research/ragres-*.md |
| 8 | Embeddings | embedding/emb-*.md |
| 9 | Agents | llm_packages/02-tldr/autogen,crewai,smolagents |
| 10 | Font Detective | (case study - create) |
| 11 | Doc Generator | (case study - create) |
| 12 | MCP Heist | (case study - create) |
| 13 | Subscriptions | flatrate-ai/ |
| 14 | Cost Optimization | 2510-free-ai-apis/03-best/04.md |
| 15 | What's Next | (synthesis) |
