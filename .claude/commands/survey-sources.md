# Survey Sources Command

Survey source material for a specific topic.

## Usage

```
/survey-sources [topic]
```

Topics: free-apis, packages, rag, embeddings, cli-tools, subscriptions, model-catalog

## Process

1. **Identify Files**
   - Check private6.txt for relevant paths
   - List files in private2/[topic]/
   - Count total files to process

2. **Quick Scan TLDRs**
   - Read all TLDR files in private2/[topic]/
   - Create summary of key findings
   - Identify high-value sources for deep read

3. **Create Survey Report**
   - Save to WORK.md or dedicated survey file
   - Include prioritized reading list
   - Flag questions and gaps

## Topic Mapping

| Topic | private/ Path | Est. Files |
|-------|---------------|------------|
| free-apis | 2510-free-ai-apis/ | 20 |
| packages | llm_packages/ | 100+ |
| rag | rag-research/ | 35 |
| embeddings | embedding/ | 10 |
| cli-tools | llm_research/md/20x | 10 |
| subscriptions | flatrate-ai/ | 15 |
| model-catalog | vexy-co-model-catalog/ | 50 |

## Output Format

```markdown
# Survey: [Topic]

## Files Scanned: [N]

## Key Findings
1. [Finding]
2. [Finding]

## Recommended Deep Read
1. [filepath] - [reason]
2. [filepath] - [reason]

## Chapter Mapping
- Chapter [N]: [relevant content]

## Questions
- [What's unclear or missing]
```
