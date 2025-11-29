# Source Reading Skill

## When to Use

Invoke when reading files from private/ to extract content for chapters.

## Reading Strategy

### Quick Scan (30 seconds)
1. Read title and headers
2. Check date/version info
3. Identify document type (research, tutorial, reference)
4. Estimate relevance to current chapter

### Deep Read (as needed)
1. Read intro and conclusion first
2. Focus on code examples
3. Note gotchas and warnings
4. Extract specific numbers/versions

## Extraction Priorities

### High Value (always extract)
- Working code examples
- Version-specific behavior
- Provider comparisons with data
- Error handling patterns
- Performance numbers

### Medium Value (extract if relevant)
- Architecture explanations
- Best practices
- Historical context
- Alternative approaches

### Low Value (usually skip)
- Generic introductions
- Overly theoretical content
- Dated predictions
- Marketing language

## Cross-Reference Pattern

When reading about a tool/provider:

```
1. What does private6.txt say? (master TLDR)
2. What does private2/ TLDR say? (file summary)
3. What does full source say? (details)
4. Do they conflict? (flag for resolution)
```

## Output Format for Extraction

```markdown
## From: [filepath]

### Keep (quote or paraphrase)
> [exact text worth preserving]

### Code
```[lang]
[code that works]
```

### Numbers
- [Specific metrics, limits, prices]

### Warnings
- [Things that will break]

### For Chapter [N]
- [Specific relevance]
```

## File Type Handling

### Research Reports (ragres-*.md)
- Skip methodology, focus on findings
- Extract comparison tables
- Note final recommendations

### Package TLDRs (02-tldr/*.md)
- Focus on core API patterns
- Extract signature examples
- Note unique features

### Provider Analysis (provider/*.md)
- Focus on limits and pricing
- Extract authentication patterns
- Note free tier specifics
