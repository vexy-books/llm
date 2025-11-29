# Chapter 5B: Advanced CLI workflows

*Power user patterns for terminal-native AI.*

> "The programmer, like the poet, works only slightly removed from pure thought-stuff."
> — Fred Brooks, *The Mythical Man-Month*

## Beyond basic prompts

Chapter 5 covered the tools. This chapter covers the craft.

Most CLI users type a question, get an answer, move on. Power users build workflows that compound. They chain tools, cache results, template prompts, and integrate AI into their existing shell habits.

This chapter shows you how.

## Imagine...

Imagine your terminal is a workshop. The basic tools—hammers, screwdrivers—are your individual CLI commands. But a master craftsman doesn't just grab tools randomly. They build jigs, create templates, organize their workspace so the right tool is always at hand.

Advanced CLI workflows are your jigs. They turn repetitive AI interactions into muscle memory.

---

## Pattern 1: Chaining Claude Code with MCP servers

Claude Code becomes powerful when connected to external systems. MCP servers are the connectors.

### The basic chain

```bash
# Claude Code with filesystem and git MCP servers
claude --mcp-server filesystem --mcp-server git

# Now Claude can read files AND check git history
> "What changed in the last commit and why might it have broken the tests?"
```

### Custom MCP server chains

Create a shell function that loads your preferred servers:

```bash
# ~/.zshrc or ~/.bashrc
claude-dev() {
    claude \
        --mcp-server filesystem \
        --mcp-server git \
        --mcp-server postgres \
        --mcp-server browser \
        "$@"
}

claude-fonts() {
    claude \
        --mcp-server filesystem \
        --mcp-server fontlab \
        "$@"
}

# Usage
claude-dev "Analyze the database schema and suggest indexes for slow queries"
claude-fonts "Compare the kerning tables in these two font files"
```

### Dynamic server loading

Load MCP servers based on project type:

```bash
# ~/.claude/auto-mcp.sh
auto_claude() {
    local servers=""

    # Detect project type and load appropriate servers
    if [[ -f "package.json" ]]; then
        servers="$servers --mcp-server npm"
    fi

    if [[ -f "Cargo.toml" ]]; then
        servers="$servers --mcp-server cargo"
    fi

    if [[ -f "pyproject.toml" ]] || [[ -f "requirements.txt" ]]; then
        servers="$servers --mcp-server pip"
    fi

    if [[ -d ".git" ]]; then
        servers="$servers --mcp-server git"
    fi

    if [[ -f "docker-compose.yml" ]]; then
        servers="$servers --mcp-server docker"
    fi

    claude $servers "$@"
}

alias c='auto_claude'
```

Now `c "What's the project structure?"` automatically loads the right tools.

---

## Pattern 2: AIChat as local API gateway

AIChat can serve as a local API endpoint, letting you hit multiple providers through a single interface.

### Setting up the gateway

```yaml
# ~/.config/aichat/config.yaml
api:
  enabled: true
  port: 8080
  host: 127.0.0.1

clients:
  - name: openai
    type: openai
    api_key: ${OPENAI_API_KEY}

  - name: claude
    type: claude
    api_key: ${ANTHROPIC_API_KEY}

  - name: gemini
    type: gemini
    api_key: ${GOOGLE_API_KEY}

  - name: groq
    type: openai-compatible
    api_base: https://api.groq.com/openai/v1
    api_key: ${GROQ_API_KEY}
```

```bash
# Start the local server
aichat --serve

# Now any tool can hit localhost:8080 with OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Why this matters

1. **Single endpoint**: Your scripts don't need per-provider logic
2. **Local caching**: AIChat caches responses, saving API costs
3. **Offline fallback**: Configure local models as fallback
4. **Request logging**: See all AI requests in one place

### Integration with other tools

```bash
# Use with curl in scripts
ask_ai() {
    curl -s http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$1\", \"messages\": [{\"role\": \"user\", \"content\": \"$2\"}]}" \
        | jq -r '.choices[0].message.content'
}

# Quick questions to different models
ask_ai "groq" "Explain quicksort in one sentence"
ask_ai "claude" "Review this code: $(cat main.py)"
```

---

## Pattern 3: Prompt templates and aliases

Don't retype common prompts. Template them.

### Basic aliases

```bash
# ~/.zshrc
alias explain='claude "Explain this code simply:" '
alias review='claude "Review this code for bugs and improvements:" '
alias fix='claude "Fix this error:" '
alias doc='claude "Write a docstring for this function:" '

# Usage
cat broken.py | fix
explain "$(pbpaste)"
```

### Parameterized templates

```bash
# Template function with parameters
code_review() {
    local language="${1:-python}"
    local focus="${2:-bugs}"

    cat <<EOF | claude
Review the following $language code. Focus on $focus.

Code:
$(cat)

Provide specific line numbers and suggestions.
EOF
}

# Usage
cat main.py | code_review python "security vulnerabilities"
cat app.rs | code_review rust "performance"
```

### Template files

Store complex prompts in files:

```bash
# ~/.claude/templates/pr-review.md
You are reviewing a pull request. Analyze the diff and provide:

1. **Summary**: What does this PR do? (2-3 sentences)
2. **Concerns**: Any bugs, security issues, or performance problems?
3. **Style**: Does it follow project conventions?
4. **Tests**: Are the changes adequately tested?

Diff:
```

```bash
# Function to use template
pr_review() {
    local template="$HOME/.claude/templates/pr-review.md"
    local diff=$(git diff main...HEAD)

    {
        cat "$template"
        echo "$diff"
    } | claude
}
```

### Context-aware templates

Templates that adapt to the current directory:

```bash
smart_review() {
    local context=""

    # Add project-specific context
    if [[ -f "README.md" ]]; then
        context="Project README:\n$(head -50 README.md)\n\n"
    fi

    if [[ -f ".github/CONTRIBUTING.md" ]]; then
        context="${context}Contributing guidelines:\n$(cat .github/CONTRIBUTING.md)\n\n"
    fi

    echo -e "${context}Review this code:\n$(cat)" | claude
}
```

---

## Pattern 4: Unix pipeline integration

The Unix philosophy: do one thing well, compose with pipes. AI fits naturally.

### Mods for pipeline AI

```bash
# Mods excels at pipeline integration
git diff | mods "summarize these changes"
cat error.log | mods "what's causing these errors?"
ls -la | mods "which files were modified today?"

# Chain with other tools
curl -s api.example.com/users | jq '.[] | .email' | mods "categorize these email domains"
```

### Building AI pipelines

```bash
# Multi-stage pipeline
cat article.md \
    | mods "extract the main arguments" \
    | mods "find logical fallacies" \
    | mods "suggest counterarguments"

# Parallel processing with xargs
find . -name "*.py" -print0 \
    | xargs -0 -P 4 -I {} sh -c 'cat {} | mods "rate code quality 1-10" > {}.review'
```

### Filter patterns

```bash
# AI as a filter
ai_filter() {
    local criteria="$1"
    while IFS= read -r line; do
        result=$(echo "$line" | mods "Does this match: $criteria? Answer only YES or NO")
        if [[ "$result" == *"YES"* ]]; then
            echo "$line"
        fi
    done
}

# Usage: filter log lines
cat server.log | ai_filter "security-related events"
```

### Transform patterns

```bash
# AI as a transformer
ai_transform() {
    local instruction="$1"
    while IFS= read -r line; do
        echo "$line" | mods "$instruction"
    done
}

# Usage: transform data formats
cat data.csv | ai_transform "convert this CSV row to JSON"
```

---

## Pattern 5: Session management

Long conversations need state. Here's how to manage it.

### Named sessions

```bash
# Claude Code supports sessions
claude --session font-project "What were we working on?"

# Create session aliases
alias font='claude --session font-project'
alias api='claude --session api-design'
alias debug='claude --session debugging'

# Each alias maintains separate conversation history
font "Let's continue with the kerning analysis"
api "Add authentication to the endpoint we discussed"
```

### Session with context files

```bash
# Load context at session start
start_session() {
    local name="$1"
    local context_file="$HOME/.claude/contexts/$name.md"

    if [[ -f "$context_file" ]]; then
        claude --session "$name" < "$context_file"
    else
        claude --session "$name"
    fi
}

# Context file: ~/.claude/contexts/font-project.md
# Contains project background, decisions made, current status
```

### Exporting sessions

```bash
# Save conversation for later reference
claude --session important-decision --export > decisions/2025-01-auth.md

# Review past sessions
ls ~/.claude/sessions/
cat ~/.claude/sessions/font-project.json | jq '.messages[-5:]'
```

---

## Pattern 6: Cost-aware workflows

API costs add up. Build awareness into your workflow.

### Token counting before sending

```bash
# Estimate tokens before expensive calls
count_tokens() {
    local text="$1"
    # Rough estimate: 1 token ≈ 4 characters
    local chars=$(echo "$text" | wc -c)
    local tokens=$((chars / 4))
    echo "~$tokens tokens (est. \$$(echo "scale=4; $tokens * 0.00001" | bc))"
}

# Wrapper that shows cost estimate
claude_cost() {
    local input="$*"
    echo "Input: $(count_tokens "$input")"
    read -p "Proceed? [y/N] " confirm
    if [[ "$confirm" == "y" ]]; then
        claude "$input"
    fi
}
```

### Model routing by task

```bash
# Route to appropriate model based on task
smart_ask() {
    local task="$1"
    shift
    local query="$*"

    case "$task" in
        quick|simple|fast)
            # Use cheap/fast model
            aichat -m groq/llama-3.1-8b "$query"
            ;;
        code|review|complex)
            # Use capable model
            claude "$query"
            ;;
        research|thorough)
            # Use best model
            claude --model opus "$query"
            ;;
        *)
            # Default to balanced
            claude "$query"
            ;;
    esac
}

# Usage
smart_ask quick "What's 2+2?"
smart_ask code "Review this function for bugs"
smart_ask research "Analyze the trade-offs of microservices"
```

### Budget tracking

```bash
# Track daily spending
log_ai_cost() {
    local model="$1"
    local tokens="$2"
    local date=$(date +%Y-%m-%d)
    local cost_file="$HOME/.claude/costs/$date.log"

    echo "$model,$tokens,$(date +%H:%M:%S)" >> "$cost_file"
}

daily_spend() {
    local date=$(date +%Y-%m-%d)
    local cost_file="$HOME/.claude/costs/$date.log"

    if [[ -f "$cost_file" ]]; then
        awk -F',' '{sum += $2} END {printf "Today: ~%d tokens (~$%.2f)\n", sum, sum*0.00001}' "$cost_file"
    else
        echo "No API calls logged today"
    fi
}
```

---

## Pattern 7: Git integration

AI-assisted git workflows.

### Smart commits

```bash
# Generate commit message from staged changes
git_ai_commit() {
    local diff=$(git diff --cached)

    if [[ -z "$diff" ]]; then
        echo "No staged changes"
        return 1
    fi

    local message=$(echo "$diff" | mods "Write a concise commit message for these changes. Format: type(scope): description")

    echo "Suggested commit message:"
    echo "$message"
    echo
    read -p "Use this message? [y/e/n] " choice

    case "$choice" in
        y) git commit -m "$message" ;;
        e) git commit -e -m "$message" ;;
        *) echo "Aborted" ;;
    esac
}

alias gac='git add -p && git_ai_commit'
```

### PR descriptions

```bash
# Generate PR description from commits
git_ai_pr() {
    local base="${1:-main}"
    local commits=$(git log --oneline $base..HEAD)
    local diff=$(git diff $base...HEAD --stat)

    {
        echo "Commits:"
        echo "$commits"
        echo
        echo "Changes:"
        echo "$diff"
    } | mods "Write a pull request description with Summary, Changes, and Testing sections"
}
```

### Code archaeology

```bash
# Understand why code exists
git_ai_blame() {
    local file="$1"
    local line="$2"

    local commit=$(git blame -L "$line,$line" "$file" | awk '{print $1}')
    local message=$(git log -1 --format="%B" "$commit")
    local diff=$(git show "$commit" -- "$file")

    {
        echo "Commit: $commit"
        echo "Message: $message"
        echo "Diff:"
        echo "$diff"
    } | claude "Explain why this code was added and what problem it solved"
}
```

---

## Pattern 8: Project scaffolding

Use AI to bootstrap projects consistently.

### Project initialization

```bash
# Create project with AI-generated structure
init_project() {
    local name="$1"
    local type="$2"

    mkdir -p "$name"
    cd "$name"

    claude "Create a $type project structure for '$name'.
    Output as shell commands to create directories and files.
    Include: README.md, .gitignore, basic config files.
    Use best practices for $type projects." | bash

    git init
    git add .
    git commit -m "Initial project structure"
}

# Usage
init_project my-api "FastAPI REST API"
init_project font-tools "Python CLI tool"
```

### File generation

```bash
# Generate boilerplate files
gen_file() {
    local type="$1"
    local name="$2"

    case "$type" in
        component)
            claude "Generate a React TypeScript component named $name with props interface" > "src/components/$name.tsx"
            ;;
        model)
            claude "Generate a Pydantic model named $name with common fields" > "src/models/$name.py"
            ;;
        test)
            local target="${name%.py}"
            claude "Generate pytest tests for $(cat src/$name)" > "tests/test_$target.py"
            ;;
    esac
}
```

---

## The complete .zshrc setup

Here's a production-ready configuration:

```bash
# ~/.zshrc - AI CLI configuration

# === Environment ===
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# === Basic Aliases ===
alias c='claude'
alias m='mods'
alias a='aichat'

# === Quick Actions ===
alias explain='claude "Explain simply:"'
alias review='claude "Review for bugs:"'
alias fix='claude "Fix this error:"'
alias summarize='mods "Summarize in 3 bullets:"'

# === Smart Claude with Auto-MCP ===
c() {
    local servers=""
    [[ -d ".git" ]] && servers="$servers --mcp-server git"
    [[ -f "package.json" ]] && servers="$servers --mcp-server npm"
    [[ -f "pyproject.toml" ]] && servers="$servers --mcp-server pip"
    claude $servers "$@"
}

# === Session Management ===
cs() { claude --session "$1" "${@:2}"; }  # cs project-name "query"

# === Cost Tracking ===
daily_tokens() {
    local log="$HOME/.claude/costs/$(date +%Y-%m-%d).log"
    [[ -f "$log" ]] && awk -F',' '{s+=$2}END{print s" tokens"}' "$log" || echo "0 tokens"
}

# === Git Integration ===
gai() { git diff --cached | mods "commit message:" | git commit -F -; }
prai() { git log main..HEAD | mods "PR description"; }

# === Prompt Templates ===
CLAUDE_TEMPLATES="$HOME/.claude/templates"
tpl() { cat "$CLAUDE_TEMPLATES/$1.md" | claude; }

# === Pipeline Helpers ===
ai() { mods "$@"; }  # Shorthand for pipeline use
```

## The takeaway

Advanced CLI workflows aren't about memorizing commands. They're about building systems:

1. **Template** common prompts so you never retype them
2. **Chain** tools together with MCP and pipes
3. **Route** queries to appropriate models by cost/capability
4. **Track** spending before it surprises you
5. **Integrate** with existing workflows (git, project setup)

The best CLI setup is invisible. It anticipates what you need and gets out of the way.

Start with one pattern. Add another when you hit friction. In a month, you'll wonder how you worked without it.

![A craftsman's workshop with organized tools on pegboards, each tool labeled with AI model names, warm workshop lighting](https://pixy.vexy.art/)

---

*Next: Chapter 7B dives deep into advanced RAG patterns.*
