# Technical Writing Skill

## When to Use

Invoke this skill when writing any chapter content, documentation, or technical explanation.

## Core Principles

### The Golden Rule
If the reader has to read it twice, you've failed once.

### Hook Structure

```
[Punchy opening that creates tension or curiosity]
[Context that explains why this matters]
[Promise of what they'll learn]
[Get into it - no more preamble]
```

### Code Presentation

```python
# BAD - explains what
x = get_response()  # get the response

# GOOD - explains why
x = get_response()  # cache for retry logic below
```

### Comparison Tables

Always use tables for provider/tool comparisons:

```markdown
| Tool | Strength | Weakness | Best For |
|------|----------|----------|----------|
| A    | Fast     | Limited  | Scripts  |
| B    | Flexible | Complex  | Apps     |
```

### Progressive Disclosure

1. Basic usage first
2. Common patterns next
3. Advanced options later
4. Edge cases last

### Error Messages as UX

When showing errors, show the fix:

```
BAD: "Rate limit exceeded"
GOOD: "Rate limit exceeded (429). Retry in 60s or reduce batch size."
```

## Anti-Patterns to Avoid

1. **The Dictionary Definition** - Don't start with "X is defined as..."
2. **The Apology** - Don't say "This might seem complex but..."
3. **The Hedge** - Don't say "It's worth noting that perhaps..."
4. **The Exhaustive List** - Pick the top 3-5, not all 47
5. **The Future Promise** - Don't say "We'll cover this later"

## Templates

### Introducing a Tool

```markdown
**[Tool Name]** does one thing well: [what].

```bash
[minimal working example]
```

That's the 80% use case. For the other 20%:
[advanced usage]
```

### Comparing Options

```markdown
Three paths forward:

1. **[Option A]** - [one-line description]
2. **[Option B]** - [one-line description]
3. **[Option C]** - [one-line description]

The choice depends on [key factor]. Most readers should start with [recommendation].
```

### Warning/Gotcha

```markdown
!!! warning "Rate Limits Hit Different"
    [Provider] throttles at [N] RPM on free tier.
    Batch your requests or face 429s.
```
