# Chapter 6: The MCP protocol

*How Claude Code actually talks to the world.*

> "Any sufficiently advanced technology is indistinguishable from magic."
> — Arthur C. Clarke

## The tool integration problem

Language models are brilliant at text manipulation. They're terrible at everything else. They can't read files, query databases, browse the web, or adjust your font kerning. They can only pretend to do these things and hope you don't notice.

Tool use changed this. When you tell Claude to "read the file at /src/app.ts", Claude doesn't read it—Claude asks you to read it and report back. The model generates a structured request, your runtime executes it, and the result feeds back into the conversation.

This works. But every provider implemented tools differently. OpenAI has function calling. Anthropic has tool use. Google has function declarations. Same concept, different formats, incompatible implementations.

MCP (Model Context Protocol) is Anthropic's attempt to standardize this mess.

## Imagine...

Imagine Claude as a brilliant consultant locked in a glass room. You can slide documents under the door. You can ask questions through an intercom. But Claude can't walk to the filing cabinet, can't look up records in your database, can't adjust the thermostat.

Now imagine you hire an assistant who *can* do these things. Claude writes notes: "Please fetch file X." The assistant retrieves it and passes it through the slot. Claude analyzes, asks for more, the assistant fetches more.

MCP formalizes this assistant role. It defines exactly how to write notes (JSON-RPC), what kinds of tasks the assistant can perform (tools), and what information is pre-loaded in the room (resources). Any assistant who follows the protocol works with any consultant who speaks it.

The glass room isn't a limitation—it's a security feature. Claude only gets what you explicitly allow through the slot.

---

## What MCP actually is

MCP is a protocol for connecting LLMs to external systems through **servers**. Each server exposes:

- **Tools**: Functions the LLM can call
- **Resources**: Data the LLM can read
- **Prompts**: Pre-defined prompt templates

The protocol uses JSON-RPC over stdio (local) or HTTP (remote). Servers are separate processes that communicate with the LLM runtime.

```
┌─────────────────┐     JSON-RPC      ┌──────────────────┐
│  Claude Code    │ ◄───────────────► │   MCP Server     │
│  (LLM Runtime)  │                   │  (filesystem)    │
└─────────────────┘                   └──────────────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  Your Files      │
                                      └──────────────────┘
```

**Why servers instead of SDKs?**

Isolation. Security. Reusability.

A filesystem MCP server runs as a separate process. If it crashes, the LLM keeps working. If it misbehaves, you can kill it. The same server works with any MCP-compatible client.

## The three primitives

### Tools

Functions that the LLM can invoke. The LLM receives the function signature (name, parameters, description), decides when to call it, and receives the result.

```python
@server.tool("read_file")
async def read_file(path: str) -> str:
    """Read contents of a file at the given path."""
    with open(path) as f:
        return f.read()
```

The LLM sees:
```json
{
  "name": "read_file",
  "description": "Read contents of a file at the given path.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    },
    "required": ["path"]
  }
}
```

When the LLM decides to read a file, it generates:
```json
{
  "tool": "read_file",
  "arguments": {"path": "/src/app.ts"}
}
```

Your server executes this, returns the file contents, and the conversation continues.

### Resources

Data that the LLM can access without explicit tool calls. Think of them as "context that's always available."

```python
@server.resource("config://app")
async def get_config() -> str:
    """Application configuration."""
    return json.dumps(load_config())
```

Resources are identified by URIs. The client can request any resource at any time.

### Prompts

Pre-defined prompt templates that the server recommends. Less commonly used but helpful for standardizing workflows.

```python
@server.prompt("analyze-font")
async def analyze_font_prompt(font_name: str) -> str:
    """Generate prompt for font analysis."""
    return f"""Analyze the font '{font_name}':
    - Classification (serif, sans-serif, etc.)
    - Visual characteristics
    - Recommended use cases"""
```

## Building your first MCP server

Let's build a minimal MCP server from scratch. This server provides one tool: checking if a font file exists.

### Setup

```bash
# Create project
mkdir font-checker-mcp && cd font-checker-mcp
uv init
uv add mcp
```

### The Server (FastMCP — recommended)

The `FastMCP` class provides a clean decorator-based API:

```python
# server.py
from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("font-checker")

@mcp.tool()
def check_font(font_name: str) -> str:
    """Check if a font file exists in common locations."""

    font_dirs = [
        Path.home() / "Library/Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    ]

    for dir in font_dirs:
        for ext in [".ttf", ".otf", ".ttc"]:
            font_path = dir / f"{font_name}{ext}"
            if font_path.exists():
                return f"Found: {font_path}"

    return f"Font '{font_name}' not found in system directories."

if __name__ == "__main__":
    mcp.run()
```

That's it—FastMCP handles all the JSON-RPC boilerplate. Tools are just functions with docstrings.

### Running the Server

MCP servers typically run through client configuration, not directly. For Claude Code:

```json
// ~/.config/claude/mcp.json
{
  "servers": {
    "font-checker": {
      "command": "uv",
      "args": ["run", "python", "/path/to/server.py"]
    }
  }
}
```

Now Claude Code can use your `check_font` tool:

```
> claude "Check if Garamond is installed on my system"

Using tool: check_font
Arguments: {"font_name": "Garamond"}

Found: /Library/Fonts/Garamond.ttf
```

## A more complete server: font analyzer

Let's build something useful. A font analysis server that extracts metadata from font files.

```python
# font_analyzer_server.py
from mcp.server.fastmcp import FastMCP
from pathlib import Path
from fontTools.ttLib import TTFont
import json

mcp = FastMCP("font-analyzer")

def extract_metadata(font_path: Path) -> dict:
    """Extract metadata from a font file."""
    font = TTFont(font_path)
    name_table = font['name']

    def get_name(name_id: int) -> str:
        record = name_table.getName(name_id, 3, 1, 1033)
        if record:
            return record.toUnicode()
        record = name_table.getName(name_id, 1, 0, 0)
        return record.toUnicode() if record else ""

    return {
        "family": get_name(1),
        "subfamily": get_name(2),
        "full_name": get_name(4),
        "version": get_name(5),
        "postscript_name": get_name(6),
        "designer": get_name(9),
        "description": get_name(10),
        "license": get_name(13),
        "glyph_count": font['maxp'].numGlyphs,
        "units_per_em": font['head'].unitsPerEm,
    }

@mcp.tool()
def analyze_font(path: str) -> str:
    """Analyze a font file and return its metadata."""
    font_path = Path(path)

    if not font_path.exists():
        return f"Error: Font file not found: {path}"

    if font_path.suffix.lower() not in ['.ttf', '.otf', '.ttc']:
        return f"Error: Unsupported format: {font_path.suffix}"

    try:
        metadata = extract_metadata(font_path)
        return json.dumps(metadata, indent=2)
    except Exception as e:
        return f"Error analyzing font: {e}"

@mcp.tool()
def compare_fonts(path1: str, path2: str) -> str:
    """Compare metadata between two font files."""
    try:
        meta1 = extract_metadata(Path(path1))
        meta2 = extract_metadata(Path(path2))

        comparison = {
            "font1": {"path": path1, **meta1},
            "font2": {"path": path2, **meta2},
            "differences": {}
        }

        for key in meta1:
            if meta1[key] != meta2[key]:
                comparison["differences"][key] = {
                    "font1": meta1[key],
                    "font2": meta2[key]
                }

        return json.dumps(comparison, indent=2)
    except Exception as e:
        return f"Error comparing fonts: {e}"

@mcp.resource("font://system/list")
def list_system_fonts() -> str:
    """List all installed system fonts."""
    fonts = []
    for dir in [
        Path.home() / "Library/Fonts",
        Path("/Library/Fonts"),
    ]:
        if dir.exists():
            for font in dir.glob("*.[to]tf"):
                fonts.append(str(font))
    return json.dumps(fonts)

if __name__ == "__main__":
    mcp.run()
```

### Using the Font Analyzer

```
> claude --mcp font-analyzer "Compare Garamond and Georgia fonts"

Using tool: compare_fonts
Arguments: {
  "path1": "/Library/Fonts/Garamond.ttf",
  "path2": "/Library/Fonts/Georgia.ttf"
}

{
  "font1": {
    "path": "/Library/Fonts/Garamond.ttf",
    "family": "Garamond",
    "glyph_count": 247,
    ...
  },
  "font2": {
    "path": "/Library/Fonts/Georgia.ttf",
    "family": "Georgia",
    "glyph_count": 683,
    ...
  },
  "differences": {
    "glyph_count": {"font1": 247, "font2": 683},
    ...
  }
}
```

## Existing MCP servers

Don't build everything from scratch. These servers already exist:

### Official Servers (Anthropic)

| Server | Purpose |
|--------|---------|
| `filesystem` | Read/write files, directory operations |
| `github` | Repository operations, issues, PRs |
| `gitlab` | GitLab API integration |
| `postgres` | Database queries |
| `sqlite` | Local database operations |
| `memory` | Persistent memory across sessions |
| `puppeteer` | Browser automation |
| `brave-search` | Web search |
| `fetch` | HTTP requests |

### Community Servers

| Server | Purpose |
|--------|---------|
| `mcp-server-docker` | Container management |
| `mcp-server-kubernetes` | K8s operations |
| `mcp-server-slack` | Slack integration |
| `mcp-server-notion` | Notion API |
| `mcp-server-linear` | Linear project management |

### Installation

Most servers install via npm:

```bash
# Official servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github

# Community servers
npm install -g mcp-server-docker
```

Then configure in your client:

```json
// ~/.config/claude/mcp.json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    }
  }
}
```

## Security considerations

MCP servers have real system access. Think carefully about what you expose.

### Sandboxing

Limit filesystem access:
```json
{
  "filesystem": {
    "command": "npx",
    "args": [
      "-y", "@modelcontextprotocol/server-filesystem",
      "/safe/directory",  // Only this directory
      "--read-only"       // No writes
    ]
  }
}
```

### Environment Variables

Never put secrets in MCP configuration files directly. Use environment variables:

```json
{
  "github": {
    "env": {
      "GITHUB_TOKEN": "${GITHUB_TOKEN}"  // From shell environment
    }
  }
}
```

### Tool Approval

Claude Code (and most clients) ask permission before using tools. Don't disable this:

```
> claude "Delete all .bak files"

Claude wants to use: filesystem.delete
Path: **/*.bak
Allow? [y/N]
```

## Client support

MCP adoption is growing but not universal:

| Client | MCP Support |
|--------|-------------|
| Claude Code | Full (native) |
| Gemini CLI | Experimental |
| Mods | Via plugins |
| Codex | Planned |
| Custom clients | SDK available |

### Using MCP in Custom Code

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/safe/path"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available: {[t.name for t in tools.tools]}")

            # Call a tool
            result = await session.call_tool(
                "read_file",
                {"path": "/safe/path/config.json"}
            )
            print(result.content)
```

## The future of MCP

MCP is young but gaining traction. Watch for:

- **More client adoption**: Codex, Cursor, and others considering support
- **Server ecosystem**: Community building servers for everything
- **Remote servers**: HTTP-based servers for cloud deployments
- **Authentication**: Standard auth patterns for multi-user scenarios
- **Discovery**: Automatic server discovery and capability negotiation

Whether MCP becomes the universal standard or just "Anthropic's thing" depends on adoption. For now, if you're using Claude Code, MCP is the way to extend its capabilities.

## Getting started: two approaches

**Simple approach**: Use existing servers. Don't build anything.

```bash
# Install official filesystem server
npm install -g @modelcontextprotocol/server-filesystem

# Configure Claude Code to use it
cat > ~/.config/claude/mcp.json << 'EOF'
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./fonts"]
    }
  }
}
EOF

# Now Claude can read your font files
claude "What font files are in the fonts directory? Summarize what each contains."
```

Five minutes from zero to working. Claude can now browse your files, read contents, and analyze them.

**Production approach**: Build a domain-specific server with typed tools and validation.

```python
# font_mcp_server.py - Production font analysis server
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, Field
from pathlib import Path
from fontTools.ttLib import TTFont
import json

class FontMetadata(BaseModel):
    """Structured font metadata with validation."""
    family: str
    subfamily: str = ""
    postscript_name: str = ""
    designer: str = ""
    glyph_count: int = Field(ge=0)
    units_per_em: int = Field(ge=16, le=16384)
    has_kerning: bool = False
    supported_scripts: list[str] = []

class FontAnalysisResult(BaseModel):
    """Analysis result with classification."""
    metadata: FontMetadata
    classification: str  # serif, sans-serif, display, etc.
    quality_score: float = Field(ge=0, le=1)
    issues: list[str] = []

server = Server("font-analyzer-pro")

def analyze_font_file(path: Path) -> FontAnalysisResult:
    """Deep analysis of a font file."""
    font = TTFont(path)
    name_table = font['name']

    def get_name(name_id: int) -> str:
        for platform_id, encoding_id, lang_id in [(3, 1, 1033), (1, 0, 0)]:
            record = name_table.getName(name_id, platform_id, encoding_id, lang_id)
            if record:
                return record.toUnicode()
        return ""

    # Extract metadata
    metadata = FontMetadata(
        family=get_name(1) or path.stem,
        subfamily=get_name(2),
        postscript_name=get_name(6),
        designer=get_name(9),
        glyph_count=font['maxp'].numGlyphs,
        units_per_em=font['head'].unitsPerEm,
        has_kerning='kern' in font or 'GPOS' in font,
        supported_scripts=list(font['cmap'].getBestCmap().keys())[:10]
    )

    # Classify based on characteristics
    classification = classify_font(font)

    # Quality checks
    issues = []
    if metadata.glyph_count < 200:
        issues.append("Limited character set")
    if not metadata.has_kerning:
        issues.append("No kerning data")

    quality_score = 1.0 - (len(issues) * 0.2)

    return FontAnalysisResult(
        metadata=metadata,
        classification=classification,
        quality_score=max(0, quality_score),
        issues=issues
    )

def classify_font(font: TTFont) -> str:
    """Classify font by analyzing its characteristics."""
    # Simplified classification logic
    os2 = font.get('OS/2')
    if os2:
        family_class = os2.sFamilyClass >> 8
        if family_class == 1:
            return "oldstyle-serif"
        elif family_class == 2:
            return "transitional-serif"
        elif family_class == 8:
            return "sans-serif"
        elif family_class == 10:
            return "script"
    return "unknown"

@server.tool("analyze_font")
async def analyze_font_tool(path: str) -> str:
    """Analyze a font file with full metadata extraction and classification."""
    font_path = Path(path).expanduser()

    if not font_path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    if font_path.suffix.lower() not in {'.ttf', '.otf', '.ttc'}:
        return json.dumps({"error": f"Unsupported format: {font_path.suffix}"})

    result = analyze_font_file(font_path)
    return result.model_dump_json(indent=2)

@server.tool("batch_analyze")
async def batch_analyze(directory: str, pattern: str = "*.[ot]tf") -> str:
    """Analyze all fonts in a directory matching pattern."""
    dir_path = Path(directory).expanduser()
    results = []

    for font_path in dir_path.glob(pattern):
        try:
            result = analyze_font_file(font_path)
            results.append({"path": str(font_path), **result.model_dump()})
        except Exception as e:
            results.append({"path": str(font_path), "error": str(e)})

    return json.dumps(results, indent=2)

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

The simple approach gets Claude reading files in minutes. The production approach gives you validated, typed responses with domain-specific analysis. Start simple; build custom servers when you hit the limits of generic tools.

## The takeaway

MCP separates what an LLM wants to do from how it's done. The LLM understands the intent ("read this file"); the server handles the implementation (filesystem access, permissions, error handling).

This separation is powerful:

- **Security**: Servers can sandbox dangerous operations
- **Reusability**: One server works with any MCP client
- **Clarity**: Tool definitions are explicit, not prompt-engineered

For the font project that threads through this book, MCP is how we'll connect Claude to FontLab. The MCP server becomes the bridge between natural language and glyph manipulation.

Build servers for your domain. Share them if they're useful. The protocol is open; the ecosystem is yours to shape.

![A network diagram showing Claude at center connected to various MCP servers (filesystem, database, browser, font tools) via glowing connection lines](https://pixy.vexy.art/)

---

*Next: Chapter 7 digs into RAG—how to give LLMs access to your own data without fine-tuning.*
