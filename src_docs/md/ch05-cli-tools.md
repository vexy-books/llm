# Chapter 5: CLI tools for the command-line cowboy

*The terminal is your friend. These tools make it friendlier.*

> "This is the Unix philosophy: Write programs that do one thing and do it well."
> — Doug McIlroy

## The terminal renaissance

The GUI revolution tried to kill the terminal. It failed. Developers keep returning to the command line because text interfaces are faster, more composable, and easier to automate than clicking through menus.

LLM CLI tools embrace this. They're not chatbots with a command-line skin—they're Unix-native tools that respect pipes, stdin, and the shell's philosophy of composability.

This chapter surveys the CLI landscape as of late 2025. Eight major tools, each with distinct philosophies about how humans and language models should interact through text.

## Imagine...

Imagine you're a plumber with a toolbox. You could carry one massive multi-tool that does everything poorly. Or you could carry specialized tools: a wrench for pipes, a snake for drains, a torch for soldering. Each tool does one thing well. Combine them for complex jobs.

The Unix philosophy says: build small, sharp tools that compose. Want to count words in a file? `cat file.txt | wc -w`. Want to find files containing "kerning"? `grep -r "kerning" ./`. Each command does one thing. Pipes connect them.

LLM CLI tools follow the same philosophy. They're not replacements for ChatGPT—they're new tools in the toolbox. Pipe your git diff through an LLM for a summary. Pipe a log file for anomaly detection. Chain commands together for workflows that would take hours in a GUI.

The terminal isn't old-fashioned. It's a force multiplier.

---

## The big three: ecosystem tools

Three major providers bundle CLI tools with their platforms. These aren't neutral—they're designed to lock you into ecosystems. That's fine if you're already there.

### Codex CLI (OpenAI)

**Language**: Rust
**Open Source**: Yes (npm: `@openai/codex`)
**Models**: GPT-5, GPT-5.1, GPT-5-Codex-Mini, GPT-5.1-Codex-Max
**Bundled With**: ChatGPT Plus ($20/mo), Pro ($200/mo), Business, Edu, Enterprise

OpenAI's official agentic CLI, significantly upgraded in November 2025.

**Key Features**:
- Sandbox code execution (Docker-based, 90% faster with container caching)
- **Three approval modes**: Suggest (safe), Auto Edit, Full Access
- MCP integration for external tools
- Web search and image attachment support
- Conversation resumption across sessions
- Quiet mode (`-q`) for CI/CD automation
- IDE extension for VS Code/Cursor

**Latest models** (November 2025):
- **GPT-5.1-Codex-Max**: Frontier agentic model for project-scale work
- **GPT-5-Codex-Mini**: Compact, cost-efficient option

```bash
# Basic usage
codex "Analyze these TypeScript files for unused imports"

# With file context
codex --files src/**/*.ts "Refactor to use named exports"

# Provider switching
codex --provider cerebras/qwen-3-coder "Generate unit tests"

# Quiet mode for CI/CD
CODEX_QUIET_MODE=1 codex "Fix lint errors in src/"
```

**When Codex shines**: You're in the OpenAI ecosystem, need project-scale refactoring, or want CI/CD integration.

**When Codex struggles**: You need RAG over large document sets. You want minimal dependencies.

### Claude Code (Anthropic)

**Language**: TypeScript
**Open Source**: No (proprietary)
**Models**: Claude Sonnet 4.5, Claude Opus 4.5
**Bundled With**: Claude Pro ($20/mo), Max plans ($100-200/mo)

Anthropic's agentic CLI. The most opinionated of the three.

**Key Features**:
- Automatic model switching (Sonnet for speed, Opus for complexity)
- MCP (Model Context Protocol) server integration
- Extended context handling (200K tokens)
- Multi-turn task memory
- IDE integration (VS Code extension)

**Philosophy**: Extended autonomous tasks. Claude Code is designed for workflows that take hours, not seconds. Give it a complex goal, walk away, come back to results.

```bash
# Long-running task
claude "Refactor the entire authentication system to use JWTs"

# With MCP server
claude --mcp filesystem,github "Create PR with type fixes"

# Context from files
claude @src/auth/*.ts "Document these modules"
```

**Max plan advantage**: Unlocks Opus, which excels at multi-hour autonomous work. Pro is limited to Sonnet.

**When Claude Code shines**: Complex refactoring, extended reasoning tasks, you trust autonomous agents.

**When Claude Code struggles**: Simple one-off queries. You want fine-grained control. Open-source requirement.

### Gemini CLI (Google)

**Language**: TypeScript (npm package)
**Open Source**: Yes
**Models**: Gemini 3 Pro, Gemini 2.5 Flash
**Bundled With**: Free tier (1,000 req/day), Code Assist Standard ($19/mo), Enterprise ($45/mo)

Google's dark horse. Best free tier, built-in web search, million-token context.

**Key Features**:
- Web grounding (searches web by default)
- 1M token context (entire codebases)
- Multimodal (text, images, video, audio)
- MCP support
- Provider-agnostic (can use other models)

**The free tier**: 1,000 CLI requests per day. For most developers, this is enough to never pay.

```bash
# Web search included
gemini "What's the latest TypeScript version and what changed?"

# Million-token context
gemini @docs/ "Summarize all API changes since v1.0"

# Multimodal
gemini @screenshot.png "What UI framework is this?"

# Provider override
gemini --provider openai/gpt-4o "Compare approaches"
```

**When Gemini CLI shines**: You need current information, huge context windows, multimodal input, or free tier.

**When Gemini CLI struggles**: You need the absolute best reasoning (Opus edges it out), or privacy concerns about Google.

## The generalists: universal tools

These tools aren't tied to specific providers. They work with any OpenAI-compatible API.

### AIChat

**Language**: Rust
**Open Source**: Yes
**Models**: Any OpenAI-compatible provider

The Swiss Army knife of LLM CLIs.

**Key Features**:
- **Built-in RAG**: Knowledge base management without external tools
- **Local API mode**: Serves HTTP API from command line
- **Session persistence**: Conversations saved and resumable
- **YAML configuration**: Per-project settings
- **Role system**: Define custom personas and behaviors

```bash
# RAG: Create knowledge base
aichat --rag kb-create fonts

# RAG: Add documents
aichat --rag kb-add fonts ./font-docs/**/*.md

# RAG: Query
aichat --rag kb-query fonts "Explain kerning"

# Serve API locally
aichat --serve 127.0.0.1:8080

# Session management
aichat --session typography "Continue our kerning discussion"
```

**Configuration** (`.aichat.yaml`):
```yaml
model: cerebras/llama-3.3-70b
temperature: 0.7
rag:
  chunk_size: 1000
  chunk_overlap: 200
```

**When AIChat shines**: You need RAG without vector DB setup. You want a local API. Configuration as code matters.

**When AIChat struggles**: GUI preferences. Team collaboration features.

### Mods

**Language**: Go
**Open Source**: Yes
**Models**: Any OpenAI-compatible provider
**Philosophy**: Unix pipeline integration

Mods treats LLMs as Unix tools. Embrace pipes, embrace composition.

**Key Features**:
- **Stdin/stdout**: First-class pipeline citizen
- **Template system**: Reusable prompt patterns
- **MCP support**: Tool integration
- **Format conversion**: JSON, YAML, Markdown
- **Chaining**: Pipe to other mods instances

```bash
# Pipeline: analyze git diff
git diff | mods "Summarize changes" | pbcopy

# Template system
echo "Garamond" | mods --template font-analysis

# Chain multiple steps
cat fonts.csv | \
  mods "Extract serif fonts" | \
  mods "Rank by readability" | \
  mods --format json > results.json

# MCP integration
mods --mcp filesystem "Analyze src/ for patterns"
```

**Templates** (`~/.config/mods/templates/font-analysis.md`):
```markdown
Analyze this font family:
{{stdin}}

Provide:
- Classification
- Historical context
- Use cases
```

**When Mods shines**: Pipeline workflows, text processing, automation, Unix philosophy purists.

**When Mods struggles**: Interactive sessions, complex state management, GUI needed.

### llxprt

**Language**: Go
**Open Source**: Yes
**Models**: Any OpenAI-compatible provider
**Philosophy**: Minimal and fast

The lightweight option. Does one thing well: query LLMs from terminal.

```bash
# Basic query
llxprt "Convert RGB to hex: 255,128,64"

# File context
llxprt -f styles.css "Explain this selector specificity"

# Provider config
export LLXPRT_API_KEY=sk-...
export LLXPRT_MODEL=groq/llama-3.1-70b
llxprt "What's 2^16?"
```

**When llxprt shines**: Simple queries, minimal setup, shell scripts.

**When llxprt struggles**: Complex features, RAG, sessions.

## The specialist: code editors

### Zed

**Language**: Rust
**Open Source**: Yes
**Integration**: Editor with built-in AI

Not strictly a CLI tool, but worth mentioning. Zed is a code editor with native LLM integration.

**Features**:
- Inline completions
- Chat panel with context awareness
- Multi-file refactoring
- Terminal integration

For developers who want AI in the editor itself rather than a separate CLI.

## Provider switching patterns

Most CLIs support provider switching. This is how you escape vendor lock-in:

**Environment Variables** (universal):
```bash
export OPENAI_API_BASE=https://api.cerebras.ai/v1
export OPENAI_API_KEY=sk-cerebras-...
```

**CLI Flags** (tool-specific):
```bash
# Codex
codex --provider groq/llama-3.1-70b

# Gemini
gemini --provider anthropic/claude-sonnet-4

# AIChat
aichat --model fireworks/qwen-coder
```

**Config Files** (persistent):
```yaml
# .codexrc
default_provider: cerebras
providers:
  cerebras:
    api_key: sk-...
    base_url: https://api.cerebras.ai/v1
  groq:
    api_key: gsk-...
```

## The multi-provider workflow

Smart developers don't pick one tool. They use several:

**Daily driver**: Gemini CLI (free, web search, huge context)
**Code generation**: Codex with Cerebras provider (speed)
**Complex reasoning**: Claude Code with Opus (quality)
**Pipeline work**: Mods (composability)
**Quick queries**: llxprt (minimal overhead)

Shell aliases make this seamless:

```bash
# .zshrc
alias ask="gemini"
alias code="codex --provider cerebras/qwen-3-coder"
alias think="claude --model opus"
alias pipe="mods"
alias quick="llxprt"
```

## MCP: the protocol that connects everything

Model Context Protocol (MCP) is Anthropic's standard for tool integration. Growing support across CLIs:

**MCP Servers** expose functionality:
- `filesystem`: Read/write files
- `github`: Repository operations
- `postgres`: Database queries
- `browser`: Web automation
- Custom: Build your own

**CLI Integration**:
```bash
# Claude Code (native)
claude --mcp filesystem,github "Create PR with fixes"

# Mods (via plugin)
mods --mcp postgres "Query user table, analyze patterns"

# Gemini (experimental)
gemini --enable-mcp filesystem "Analyze project structure"
```

MCP matters because it standardizes how LLMs interact with external systems. Chapter 6 covers building MCP servers.

## Configuration philosophy

Each tool has different config approaches:

**Codex**: Profiles in `~/.codex/profiles/`
**Claude**: Centralized in `~/.anthropic/config`
**Gemini**: Per-project `.gemini/config.yaml`
**AIChat**: Hierarchical `.aichat.yaml` (global + project)
**Mods**: Templates in `~/.config/mods/templates/`

**Universal pattern**: Environment variables override config files.

## The recommendation matrix

| Priority | Best Choice | Alternative |
|----------|-------------|-------------|
| **Free tier** | Gemini CLI | AIChat (BYOK) |
| **Code generation** | Codex + Cerebras | Claude Code |
| **Reasoning** | Claude Code (Opus) | Gemini CLI (Pro) |
| **Pipelines** | Mods | Bash + llxprt |
| **RAG** | AIChat | Manual vector DB + any CLI |
| **Speed** | Codex + Groq | Gemini (Flash) |
| **Minimal** | llxprt | Mods |

## Getting started: two approaches

**Simple approach**: One tool, one query. This works when you need quick answers.

```bash
# Ask a question, get an answer
gemini "What fonts pair well with Bodoni for body text?"

# Or with file context
llxprt -f specimen.txt "What font family is this?"
```

Direct. No configuration. Works today with your free tier.

**Production approach**: Chain tools for automated workflows. This is where CLI tools shine.

```bash
#!/bin/bash
# analyze-fonts.sh: Batch analyze font files and generate reports

FONT_DIR="${1:-.}"
OUTPUT_DIR="./analysis"

mkdir -p "$OUTPUT_DIR"

# Find all font files
find "$FONT_DIR" -name "*.otf" -o -name "*.ttf" | while read -r font; do
  name=$(basename "$font" | sed 's/\.[^.]*$//')

  # Extract metadata with fonttools, pipe to LLM for analysis
  fonttools ttx -o - "$font" 2>/dev/null | \
    mods --template font-analysis | \
    mods --format json > "$OUTPUT_DIR/${name}.json"

  echo "Analyzed: $name"
done

# Generate summary report
cat "$OUTPUT_DIR"/*.json | \
  mods "Summarize these font analyses. Group by classification. Note any fonts that would pair well together." \
  > "$OUTPUT_DIR/summary.md"

echo "Report: $OUTPUT_DIR/summary.md"
```

The template (`~/.config/mods/templates/font-analysis.md`):

```markdown
Analyze this font based on its OpenType tables:
{{stdin}}

Return JSON with:
- classification: serif/sans-serif/display/script/mono
- characteristics: array of visual traits
- era: design period estimate
- similar_fonts: 3-5 alternatives
- best_uses: recommended applications
```

The simple approach gets you started. The production approach automates your workflow. Start simple, graduate to pipelines when you find yourself repeating commands.

## The takeaway

CLI tools aren't worse versions of ChatGPT—they're different interfaces for different workflows. GUIs excel at exploration. CLIs excel at automation.

Pick your daily driver based on what you already pay for:

- **ChatGPT Plus?** Use Codex
- **Claude Pro/Max?** Use Claude Code
- **Nothing?** Use Gemini CLI (free tier is excellent)

Then add Mods for pipeline work and AIChat if you need RAG.

The terminal isn't going anywhere. LLMs that speak Unix are here to stay.

![A command terminal screen showing multiple CLI tools running in split panes, each displaying different LLM interactions, hacker aesthetic](https://pixy.vexy.art/)

---

*Next: Chapter 6 explores the MCP protocol in depth—how Claude Code (and others) actually talk to external tools.*
