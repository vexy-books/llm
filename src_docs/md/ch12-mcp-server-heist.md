# Chapter 12: Case study — the MCP server heist

*Connecting Claude to FontLab for automated font manipulation.*

> "Every great magic trick consists of three parts: the pledge, the turn, the prestige."
> — Christopher Priest, *The Prestige*

## The problem

You manage a font library. Thousands of files. Clients request modifications constantly: "Make the lowercase a taller." "Reduce the x-height by 5%." "Add OpenType stylistic alternates."

These aren't creative decisions—they're mechanical adjustments. A skilled type designer takes 20 minutes per font. You have 200 fonts to modify.

This case study builds an MCP server that lets Claude manipulate font files directly. Natural language in, modified fonts out.

The heist: stealing hours of tedious work and giving them back to actual design.

## Imagine...

Imagine a master locksmith who can open any font file like a safe. They reach in with precise tools—measure this curve, adjust that stem weight, widen the spacing between these letters. Years of training taught them where to find each mechanism, how to modify it without breaking adjacent systems.

Now imagine giving that locksmith the ability to understand plain English. "Make the letters breathe more." They know that means increase letter-spacing by 2-3%. "The x-height feels cramped." They adjust the vertical proportions while preserving the cap height ratio.

An MCP server is the bridge between natural language and specialized tools. Claude speaks human; fontTools speaks font. The MCP server translates, allowing you to say "increase the x-height by 10%" and have the exact right bytes modified in the exact right table.

The heist isn't about breaking in—it's about having a native speaker in both domains. You describe what you want; the font changes itself.

---

## What we're building

```
┌─────────────┐     MCP Protocol      ┌─────────────────┐
│ Claude Code │ ◄──────────────────► │  FontLab MCP    │
│ (User)      │                       │  Server         │
└─────────────┘                       └────────┬────────┘
                                               │
                                      ┌────────▼────────┐
                                      │   FontLab API   │
                                      │   (fontTools)   │
                                      └────────┬────────┘
                                               │
                                      ┌────────▼────────┐
                                      │   Font Files    │
                                      │   (.otf, .ttf)  │
                                      └─────────────────┘
```

**Capabilities:**

- **Read**: Extract metrics, glyphs, OpenType features
- **Modify**: Adjust metrics, transform glyphs, change names
- **Generate**: Create font variations, export formats
- **Query**: Answer questions about font characteristics

## Step 1: the server skeleton

Start with the MCP boilerplate.

```python
# fontlab_server.py
from mcp.server.lowlevel import Server
import mcp.server.stdio
from pathlib import Path
import json

server = Server("fontlab-mcp")

# Store state
WORKSPACE = Path("./font_workspace")
WORKSPACE.mkdir(exist_ok=True)

@server.list_tools()
async def list_tools():
    """Return available tools."""
    return [
        {
            "name": "list_fonts",
            "description": "List all font files in the workspace",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_font_info",
            "description": "Get detailed information about a font file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to font file"}
                },
                "required": ["path"]
            }
        },
        # More tools defined below
    ]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Step 2: font reading tools

First, let Claude inspect fonts.

```python
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

@server.tool("list_fonts")
async def list_fonts() -> str:
    """List all font files in the workspace."""
    fonts = []
    for ext in ["*.ttf", "*.otf", "*.woff", "*.woff2"]:
        fonts.extend(WORKSPACE.glob(f"**/{ext}"))

    if not fonts:
        return "No fonts found in workspace."

    return json.dumps([str(f.relative_to(WORKSPACE)) for f in fonts], indent=2)

@server.tool("get_font_info")
async def get_font_info(path: str) -> str:
    """Get detailed information about a font file."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)

    # Extract name table entries
    name_table = font["name"]
    def get_name(name_id):
        record = name_table.getName(name_id, 3, 1, 1033)
        if record:
            return record.toUnicode()
        record = name_table.getName(name_id, 1, 0, 0)
        return record.toUnicode() if record else ""

    # Get metrics
    os2 = font.get("OS/2")
    head = font["head"]
    hhea = font["hhea"]

    info = {
        "family": get_name(1),
        "subfamily": get_name(2),
        "full_name": get_name(4),
        "version": get_name(5),
        "designer": get_name(9),
        "metrics": {
            "units_per_em": head.unitsPerEm,
            "ascender": hhea.ascent,
            "descender": hhea.descent,
            "line_gap": hhea.lineGap,
            "x_height": os2.sxHeight if os2 else None,
            "cap_height": os2.sCapHeight if os2 else None,
        },
        "glyph_count": font["maxp"].numGlyphs,
        "tables": list(font.keys()),
    }

    font.close()
    return json.dumps(info, indent=2)

@server.tool("get_glyph_info")
async def get_glyph_info(path: str, glyph_name: str) -> str:
    """Get information about a specific glyph."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)

    if glyph_name not in font.getGlyphSet():
        return f"Error: Glyph '{glyph_name}' not found"

    glyph_set = font.getGlyphSet()
    glyph = glyph_set[glyph_name]

    # Get bounds
    pen = BoundsPen(glyph_set)
    glyph.draw(pen)
    bounds = pen.bounds

    # Get advance width
    hmtx = font["hmtx"]
    width, lsb = hmtx[glyph_name]

    info = {
        "name": glyph_name,
        "width": width,
        "left_side_bearing": lsb,
        "bounds": {
            "xMin": bounds[0] if bounds else 0,
            "yMin": bounds[1] if bounds else 0,
            "xMax": bounds[2] if bounds else 0,
            "yMax": bounds[3] if bounds else 0,
        } if bounds else None,
        "unicode": [hex(u) for u in font.getBestCmap().items() if u[1] == glyph_name]
    }

    font.close()
    return json.dumps(info, indent=2)

@server.tool("list_glyphs")
async def list_glyphs(path: str) -> str:
    """List all glyph names in a font."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)
    glyphs = font.getGlyphOrder()
    font.close()

    return json.dumps(glyphs, indent=2)
```

## Step 3: font modification tools

Now the interesting part—modifying fonts.

```python
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.misc.transform import Transform

@server.tool("modify_metrics")
async def modify_metrics(
    path: str,
    ascender: int | None = None,
    descender: int | None = None,
    line_gap: int | None = None,
    x_height: int | None = None,
    cap_height: int | None = None,
    output_path: str | None = None
) -> str:
    """Modify font metrics. Only provided values are changed."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)

    # Modify hhea table
    if ascender is not None:
        font["hhea"].ascent = ascender
    if descender is not None:
        font["hhea"].descent = descender
    if line_gap is not None:
        font["hhea"].lineGap = line_gap

    # Modify OS/2 table
    os2 = font.get("OS/2")
    if os2:
        if ascender is not None:
            os2.sTypoAscender = ascender
            os2.usWinAscent = ascender
        if descender is not None:
            os2.sTypoDescender = descender
            os2.usWinDescent = abs(descender)
        if x_height is not None:
            os2.sxHeight = x_height
        if cap_height is not None:
            os2.sCapHeight = cap_height

    # Save
    out_path = WORKSPACE / (output_path or path)
    font.save(out_path)
    font.close()

    return f"Metrics modified. Saved to {out_path}"

@server.tool("scale_font")
async def scale_font(
    path: str,
    scale_factor: float,
    output_path: str | None = None
) -> str:
    """Scale all glyphs by a factor (e.g., 1.1 = 110%)."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()

    # Scale transform
    transform = Transform().scale(scale_factor)

    # Apply to all glyphs
    for glyph_name in font.getGlyphOrder():
        if glyph_name == ".notdef":
            continue

        # Record original outline
        pen = RecordingPen()
        glyph_set[glyph_name].draw(pen)

        # Scale and rewrite
        # (Implementation depends on font format - simplified here)

    # Scale metrics
    font["head"].unitsPerEm = int(font["head"].unitsPerEm * scale_factor)
    font["hhea"].ascent = int(font["hhea"].ascent * scale_factor)
    font["hhea"].descent = int(font["hhea"].descent * scale_factor)

    out_path = WORKSPACE / (output_path or f"scaled_{Path(path).name}")
    font.save(out_path)
    font.close()

    return f"Font scaled by {scale_factor}x. Saved to {out_path}"

@server.tool("modify_glyph_width")
async def modify_glyph_width(
    path: str,
    glyph_name: str,
    new_width: int | None = None,
    width_delta: int | None = None,
    output_path: str | None = None
) -> str:
    """Modify the advance width of a glyph."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)

    if glyph_name not in font.getGlyphSet():
        font.close()
        return f"Error: Glyph '{glyph_name}' not found"

    hmtx = font["hmtx"]
    current_width, lsb = hmtx[glyph_name]

    if new_width is not None:
        hmtx[glyph_name] = (new_width, lsb)
    elif width_delta is not None:
        hmtx[glyph_name] = (current_width + width_delta, lsb)
    else:
        font.close()
        return "Error: Provide either new_width or width_delta"

    out_path = WORKSPACE / (output_path or path)
    font.save(out_path)
    font.close()

    return f"Glyph '{glyph_name}' width changed from {current_width} to {hmtx[glyph_name][0]}"

@server.tool("rename_font")
async def rename_font(
    path: str,
    family_name: str | None = None,
    subfamily: str | None = None,
    full_name: str | None = None,
    output_path: str | None = None
) -> str:
    """Rename font family, subfamily, or full name."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)
    name_table = font["name"]

    def set_name(name_id: int, value: str):
        # Set for multiple platform/encoding combinations
        for platform_id, enc_id, lang_id in [(3, 1, 1033), (1, 0, 0)]:
            name_table.setName(value, name_id, platform_id, enc_id, lang_id)

    if family_name:
        set_name(1, family_name)  # Family
        set_name(16, family_name)  # Typographic Family
    if subfamily:
        set_name(2, subfamily)  # Subfamily
        set_name(17, subfamily)  # Typographic Subfamily
    if full_name:
        set_name(4, full_name)  # Full name
    elif family_name and subfamily:
        set_name(4, f"{family_name} {subfamily}")

    out_path = WORKSPACE / (output_path or path)
    font.save(out_path)
    font.close()

    return f"Font renamed. Saved to {out_path}"
```

## Step 4: generation tools

Create font variations and exports.

```python
from fontTools.varLib import instancer
from fontTools.subset import Subsetter, Options

@server.tool("create_subset")
async def create_subset(
    path: str,
    glyphs: list[str] | None = None,
    unicodes: list[str] | None = None,
    text: str | None = None,
    output_path: str | None = None
) -> str:
    """Create a subset font with only specified glyphs."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)

    options = Options()
    options.layout_features = ["*"]  # Keep all OpenType features

    subsetter = Subsetter(options=options)

    if glyphs:
        subsetter.populate(glyphs=glyphs)
    elif unicodes:
        # Convert hex strings to integers
        codes = [int(u, 16) for u in unicodes]
        subsetter.populate(unicodes=codes)
    elif text:
        subsetter.populate(text=text)
    else:
        font.close()
        return "Error: Provide glyphs, unicodes, or text to subset"

    subsetter.subset(font)

    out_path = WORKSPACE / (output_path or f"subset_{Path(path).name}")
    font.save(out_path)
    font.close()

    return f"Subset created with {len(font.getGlyphOrder())} glyphs. Saved to {out_path}"

@server.tool("convert_format")
async def convert_format(
    path: str,
    output_format: str,
    output_path: str | None = None
) -> str:
    """Convert font to different format (ttf, otf, woff, woff2)."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    if output_format not in ["ttf", "otf", "woff", "woff2"]:
        return f"Error: Unsupported format '{output_format}'"

    font = TTFont(font_path)

    # Set flavor for web fonts
    if output_format == "woff":
        font.flavor = "woff"
    elif output_format == "woff2":
        font.flavor = "woff2"
    else:
        font.flavor = None

    out_name = Path(path).stem + f".{output_format}"
    out_path = WORKSPACE / (output_path or out_name)
    font.save(out_path)
    font.close()

    return f"Converted to {output_format}. Saved to {out_path}"

@server.tool("generate_web_fonts")
async def generate_web_fonts(
    path: str,
    output_dir: str | None = None
) -> str:
    """Generate all web font formats (woff, woff2) plus CSS."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    out_dir = WORKSPACE / (output_dir or "webfonts")
    out_dir.mkdir(exist_ok=True)

    font = TTFont(font_path)
    base_name = Path(path).stem

    # Get font info for CSS
    name_table = font["name"]
    family = name_table.getName(1, 3, 1, 1033).toUnicode()

    # Generate woff
    font.flavor = "woff"
    font.save(out_dir / f"{base_name}.woff")

    # Generate woff2
    font.flavor = "woff2"
    font.save(out_dir / f"{base_name}.woff2")

    font.close()

    # Generate CSS
    css = f"""@font-face {{
    font-family: '{family}';
    src: url('{base_name}.woff2') format('woff2'),
         url('{base_name}.woff') format('woff');
    font-weight: normal;
    font-style: normal;
    font-display: swap;
}}
"""
    (out_dir / f"{base_name}.css").write_text(css)

    return f"Web fonts generated in {out_dir}: .woff, .woff2, .css"
```

## Step 5: query tools

Let Claude answer questions about fonts.

```python
@server.tool("compare_fonts")
async def compare_fonts(path1: str, path2: str) -> str:
    """Compare two fonts and highlight differences."""
    font1_path = WORKSPACE / path1
    font2_path = WORKSPACE / path2

    if not font1_path.exists():
        return f"Error: Font not found at {path1}"
    if not font2_path.exists():
        return f"Error: Font not found at {path2}"

    font1 = TTFont(font1_path)
    font2 = TTFont(font2_path)

    comparison = {
        "metrics": {},
        "glyphs": {},
        "tables": {}
    }

    # Compare metrics
    for metric in ["unitsPerEm", "ascent", "descent"]:
        val1 = getattr(font1["hhea"], metric.replace("unitsPerEm", ""), None) or getattr(font1["head"], metric, None)
        val2 = getattr(font2["hhea"], metric.replace("unitsPerEm", ""), None) or getattr(font2["head"], metric, None)
        if val1 != val2:
            comparison["metrics"][metric] = {"font1": val1, "font2": val2}

    # Compare glyph counts
    glyphs1 = set(font1.getGlyphOrder())
    glyphs2 = set(font2.getGlyphOrder())

    comparison["glyphs"]["only_in_font1"] = list(glyphs1 - glyphs2)[:20]
    comparison["glyphs"]["only_in_font2"] = list(glyphs2 - glyphs1)[:20]
    comparison["glyphs"]["font1_count"] = len(glyphs1)
    comparison["glyphs"]["font2_count"] = len(glyphs2)

    # Compare tables
    tables1 = set(font1.keys())
    tables2 = set(font2.keys())
    comparison["tables"]["only_in_font1"] = list(tables1 - tables2)
    comparison["tables"]["only_in_font2"] = list(tables2 - tables1)

    font1.close()
    font2.close()

    return json.dumps(comparison, indent=2)

@server.tool("find_similar_glyphs")
async def find_similar_glyphs(path: str, glyph_name: str) -> str:
    """Find glyphs with similar metrics to the specified glyph."""
    font_path = WORKSPACE / path
    if not font_path.exists():
        return f"Error: Font not found at {path}"

    font = TTFont(font_path)
    hmtx = font["hmtx"]

    if glyph_name not in hmtx.metrics:
        font.close()
        return f"Error: Glyph '{glyph_name}' not found"

    target_width, target_lsb = hmtx[glyph_name]

    similar = []
    for name, (width, lsb) in hmtx.metrics.items():
        if name == glyph_name:
            continue
        if abs(width - target_width) <= target_width * 0.1:  # Within 10%
            similar.append({
                "name": name,
                "width": width,
                "width_diff": width - target_width
            })

    font.close()

    similar.sort(key=lambda x: abs(x["width_diff"]))
    return json.dumps(similar[:20], indent=2)
```

## Step 6: resource endpoints

Expose font data as MCP resources for context.

```python
@server.list_resources()
async def list_resources():
    """List available resources."""
    resources = []

    for font_path in WORKSPACE.glob("**/*.[ot]tf"):
        rel_path = font_path.relative_to(WORKSPACE)
        resources.append({
            "uri": f"font://{rel_path}",
            "name": font_path.name,
            "mimeType": "application/json"
        })

    return resources

@server.get_resource("font://{path}")
async def get_font_resource(path: str) -> str:
    """Get font information as a resource."""
    return await get_font_info(path)
```

## Putting it together

### Server Configuration

```json
// ~/.config/claude/mcp.json
{
  "servers": {
    "fontlab": {
      "command": "uv",
      "args": ["run", "python", "/path/to/fontlab_server.py"],
      "env": {
        "FONT_WORKSPACE": "/path/to/fonts"
      }
    }
  }
}
```

### Using the Server

```
> claude "List all fonts in the workspace"

Using tool: list_fonts

[
  "Garamond.ttf",
  "Georgia.otf",
  "custom/MyFont-Regular.ttf"
]

> claude "Show me the metrics for Garamond.ttf"

Using tool: get_font_info
Arguments: {"path": "Garamond.ttf"}

{
  "family": "Garamond",
  "metrics": {
    "units_per_em": 1000,
    "ascender": 800,
    "descender": -200,
    "x_height": 450,
    "cap_height": 700
  },
  "glyph_count": 247
}

> claude "Increase the x-height by 10% and save as Garamond-Tall.ttf"

Using tool: modify_metrics
Arguments: {
  "path": "Garamond.ttf",
  "x_height": 495,
  "output_path": "Garamond-Tall.ttf"
}

Metrics modified. Saved to Garamond-Tall.ttf

> claude "Generate web fonts for the modified version"

Using tool: generate_web_fonts
Arguments: {"path": "Garamond-Tall.ttf"}

Web fonts generated in webfonts: .woff, .woff2, .css
```

## Batch operations

The real power: bulk modifications.

```
> claude "I need to prepare all TTF fonts in the workspace for web use:
         1. Create woff2 versions
         2. Subset to Latin characters only
         3. Generate CSS files
         Do this for all fonts."

I'll process each font systematically.

Using tool: list_fonts
...
Found 15 fonts.

Using tool: create_subset
Arguments: {"path": "font1.ttf", "unicodes": ["0020-007F", "00A0-00FF"], ...}
...
[Processing 15 fonts...]

Complete. Generated 15 woff2 files with Latin subsets and CSS.
Total size reduction: 68% (2.3MB → 740KB)
```

## Security considerations

Font operations can be risky. Implement safeguards:

```python
# Add to server initialization
ALLOWED_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_path(path: str) -> Path:
    """Validate and resolve font path."""
    resolved = (WORKSPACE / path).resolve()

    # Must be within workspace
    if not str(resolved).startswith(str(WORKSPACE.resolve())):
        raise ValueError(f"Path escapes workspace: {path}")

    # Must have valid extension
    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type: {resolved.suffix}")

    # Size check for existing files
    if resolved.exists() and resolved.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {resolved.stat().st_size} bytes")

    return resolved

# Use in all tools
@server.tool("get_font_info")
async def get_font_info(path: str) -> str:
    try:
        font_path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    # ... rest of implementation
```

## Cost analysis

Running the FontLab MCP server:

| Component | Cost |
|-----------|------|
| Server compute | $0 (local) |
| Claude API calls | ~$0.02/request |
| Storage | Local disk |

For 1,000 font operations/month: **~$20**

Compare to manual work: 1,000 operations × 20 minutes × $50/hour = **$16,667**

## Lessons learned

### What Worked

1. **Granular tools**: Small, focused tools (get_glyph_info, modify_glyph_width) worked better than monolithic ones.

2. **Stateless operations**: Each tool reads, modifies, saves. No complex state management.

3. **Output paths**: Always allowing custom output paths prevented accidental overwrites.

### What Didn't

1. **Complex transformations**: Glyph outline manipulation (rotating, skewing) is hard to expose safely. Kept it simple.

2. **Validation gaps**: Early versions let Claude create invalid fonts. Added validation after each modification.

3. **Error verbosity**: Initial error messages were too technical. Rewrote for LLM comprehension.

### V2 Ideas

- **Preview generation**: Render sample text before/after modifications
- **Undo support**: Track changes, allow rollback
- **Font validation**: Check font integrity after modifications
- **Variable font axes**: Expose axis manipulation for variable fonts
- **Kerning pairs**: Add/modify kerning tables

## The complete server

```python
# fontlab_server.py - Complete implementation
from mcp.server.lowlevel import Server
import mcp.server.stdio
from fontTools.ttLib import TTFont
from pathlib import Path
import json
import os

server = Server("fontlab-mcp")
WORKSPACE = Path(os.environ.get("FONT_WORKSPACE", "./fonts"))
WORKSPACE.mkdir(exist_ok=True)

# Tools: list_fonts, get_font_info, get_glyph_info, list_glyphs
# Tools: modify_metrics, scale_font, modify_glyph_width, rename_font
# Tools: create_subset, convert_format, generate_web_fonts
# Tools: compare_fonts, find_similar_glyphs
# (Full implementations above)

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Getting started: two approaches

**Simple approach**: Use fontTools directly from Claude Code's shell. No server needed.

```bash
# In Claude Code terminal - quick font inspection
claude "What are the metrics for this font?"

# Claude runs:
python -c "
from fontTools.ttLib import TTFont
font = TTFont('MyFont.ttf')
print(f'Family: {font[\"name\"].getName(1,3,1,1033).toUnicode()}')
print(f'Glyph count: {font[\"maxp\"].numGlyphs}')
print(f'x-height: {font[\"OS/2\"].sxHeight}')
"
```

Direct shell commands work for occasional queries. Claude can write and run fontTools scripts on the fly.

**Production approach**: The full MCP server from this chapter—persistent workspace, validated tools, batch capabilities.

```python
# Quick setup: Install and configure
# pip install mcp fonttools

# Configure Claude Code
# ~/.config/claude/mcp.json
{
  "servers": {
    "fontlab": {
      "command": "python",
      "args": ["/path/to/fontlab_server.py"],
      "env": {"FONT_WORKSPACE": "/path/to/fonts"}
    }
  }
}

# Now Claude has persistent font manipulation
# Natural language becomes font modifications:

# "List all fonts" → list_fonts tool
# "Show Garamond metrics" → get_font_info tool
# "Increase x-height by 10%" → modify_metrics tool with calculated values
# "Generate web fonts for all TTFs" → batch generate_web_fonts calls
```

```bash
# Example conversation with MCP server:
> claude "Process all fonts in workspace: subset to Latin, convert to woff2, generate CSS"

# Claude systematically calls:
# 1. list_fonts → finds 20 TTF files
# 2. For each: create_subset → Latin only
# 3. For each: convert_format → woff2
# 4. For each: write CSS @font-face
# Result: 20 web-ready fonts with 70% size reduction
```

The simple approach handles one-off queries. The production approach gives Claude a persistent, secure workspace for batch operations and complex workflows. Start with direct scripts; build the MCP server when you're doing enough font work to justify the infrastructure.

## The takeaway

MCP servers transform Claude from a text generator into a domain-specific tool. For font work:

- **Natural language → Font modifications**: "Make the x-height taller" becomes actual glyph changes
- **Bulk operations**: Process hundreds of fonts with a single conversation
- **Domain expertise**: Claude can explain what each change means typographically

The architecture generalizes. Replace fontTools with any domain library:

- **Image manipulation**: PIL/Pillow → "Remove the background and add a shadow"
- **Audio processing**: pydub → "Normalize volume and add fade-out"
- **3D modeling**: trimesh → "Scale the model to fit a 10cm cube"

MCP is the bridge between natural language and programmatic control. Build servers for your domain, and suddenly Claude can do things it was never trained to do.

![A heist scene where anthropomorphized font glyphs (the letters A, B, C) are escaping through a server room, carrying bags of typographic treasure, comic book style](https://pixy.vexy.art/)

---

*Next: Part V begins—the economics of LLMs, from subscriptions to cost optimization.*
