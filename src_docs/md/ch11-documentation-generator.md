# Chapter 11: Case study — the documentation generator

*Turning code into docs without losing your mind.*

> "The documentation is like a love letter to your future self."
> — Damian Conway

## The problem

Every developer has a documentation graveyard. API specs that stopped matching reality six months ago. README files that explain how to install version 1.0. Inline comments that describe what the code did before the refactor.

Writing documentation is tedious. Keeping it updated is worse. Most teams give up.

This case study builds a system that:

1. **Reads your codebase** to understand what it does
2. **Generates documentation** in your preferred format
3. **Keeps docs in sync** when code changes
4. **Answers questions** about your code

By the end, you'll have a working documentation agent that earns its keep.

## Imagine...

Imagine a new employee who joins your team. On day one, they sit down and read every piece of code you've ever written—not skimming, but actually understanding the intent behind each function, the relationships between classes, the subtle patterns in your naming conventions.

After an hour, they can answer questions about your codebase better than people who've worked on it for years. "Where does the authentication flow start?" They know. "Why does this function return a tuple instead of a dict?" They can explain the design decision from when you made it.

That's what a documentation agent does. It ingests your entire codebase, builds a mental model of how everything fits together, and then speaks both human and code fluently. When you ask "What does this module do?", it doesn't just grep for comments—it synthesizes understanding from structure, naming, patterns, and actual behavior.

The documentation it generates isn't boilerplate. It's what that impossibly thorough new employee would write after truly understanding your system.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Codebase      │────►│   Doc Agent      │────►│  Documentation  │
│   (.py, .ts)    │     │   (RAG + LLM)    │     │  (.md, OpenAPI) │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │ Code    │  │ Vector  │  │ Template│
              │ Parser  │  │ Index   │  │ Engine  │
              └─────────┘  └─────────┘  └─────────┘
```

**Components:**

- **Code Parser**: AST-based extraction of functions, classes, and types
- **Vector Index**: Searchable embeddings of code chunks
- **Template Engine**: Output formatting for different doc styles
- **Doc Agent**: Orchestrates generation and updates

## Step 1: parsing code

We need structured data, not raw text. AST parsing beats regex.

### Python Code Parser

```python
import ast
from pathlib import Path
from pydantic import BaseModel

class FunctionInfo(BaseModel):
    name: str
    docstring: str | None
    parameters: list[dict]  # name, type, default
    return_type: str | None
    source: str
    file_path: str
    line_number: int

class ClassInfo(BaseModel):
    name: str
    docstring: str | None
    methods: list[FunctionInfo]
    attributes: list[dict]
    base_classes: list[str]
    file_path: str
    line_number: int

class ModuleInfo(BaseModel):
    path: str
    docstring: str | None
    imports: list[str]
    functions: list[FunctionInfo]
    classes: list[ClassInfo]

def parse_python_file(file_path: Path) -> ModuleInfo:
    """Parse a Python file into structured components."""
    source = file_path.read_text()
    tree = ast.parse(source)

    functions = []
    classes = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(extract_function(node, source, str(file_path)))
        elif isinstance(node, ast.ClassDef):
            classes.append(extract_class(node, source, str(file_path)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.extend(extract_imports(node))

    return ModuleInfo(
        path=str(file_path),
        docstring=ast.get_docstring(tree),
        imports=imports,
        functions=functions,
        classes=classes
    )

def extract_function(node: ast.FunctionDef, source: str, file_path: str) -> FunctionInfo:
    """Extract function information from AST node."""
    parameters = []
    for arg in node.args.args:
        param = {"name": arg.arg, "type": None, "default": None}
        if arg.annotation:
            param["type"] = ast.unparse(arg.annotation)
        parameters.append(param)

    # Handle defaults
    defaults = node.args.defaults
    for i, default in enumerate(reversed(defaults)):
        parameters[-(i+1)]["default"] = ast.unparse(default)

    return FunctionInfo(
        name=node.name,
        docstring=ast.get_docstring(node),
        parameters=parameters,
        return_type=ast.unparse(node.returns) if node.returns else None,
        source=ast.get_source_segment(source, node),
        file_path=file_path,
        line_number=node.lineno
    )

def extract_class(node: ast.ClassDef, source: str, file_path: str) -> ClassInfo:
    """Extract class information from AST node."""
    methods = []
    attributes = []

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            methods.append(extract_function(item, source, file_path))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attributes.append({
                "name": item.target.id,
                "type": ast.unparse(item.annotation) if item.annotation else None
            })

    return ClassInfo(
        name=node.name,
        docstring=ast.get_docstring(node),
        methods=methods,
        attributes=attributes,
        base_classes=[ast.unparse(base) for base in node.bases],
        file_path=file_path,
        line_number=node.lineno
    )
```

### TypeScript Parser

For TypeScript, we use the compiler API via a Node script:

```typescript
// parse_ts.ts
import * as ts from "typescript";

interface FunctionInfo {
  name: string;
  parameters: { name: string; type: string }[];
  returnType: string | null;
  jsDoc: string | null;
  filePath: string;
  lineNumber: number;
}

function parseFile(filePath: string): FunctionInfo[] {
  const program = ts.createProgram([filePath], {});
  const sourceFile = program.getSourceFile(filePath);
  const functions: FunctionInfo[] = [];

  function visit(node: ts.Node) {
    if (ts.isFunctionDeclaration(node) && node.name) {
      const signature = program.getTypeChecker().getSignatureFromDeclaration(node);
      functions.push({
        name: node.name.text,
        parameters: node.parameters.map(p => ({
          name: p.name.getText(),
          type: p.type?.getText() || "any"
        })),
        returnType: signature?.getReturnType().toString() || null,
        jsDoc: ts.getJSDocCommentsAndTags(node).map(c => c.comment).join("\n") || null,
        filePath,
        lineNumber: sourceFile!.getLineAndCharacterOfPosition(node.getStart()).line + 1
      });
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile!);
  return functions;
}

// Output JSON for Python to consume
console.log(JSON.stringify(parseFile(process.argv[2])));
```

```python
# Python wrapper
import subprocess
import json

def parse_typescript_file(file_path: Path) -> list[dict]:
    """Parse TypeScript using Node.js."""
    result = subprocess.run(
        ["npx", "ts-node", "parse_ts.ts", str(file_path)],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

## Step 2: building the index

Parse the codebase and create searchable embeddings.

```python
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
qdrant = QdrantClient(path="./docs_db")

# Create collection
qdrant.recreate_collection(
    collection_name="code",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

def embed(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

def index_codebase(root: Path) -> int:
    """Index all code files in the codebase."""
    point_id = 0

    for py_file in root.glob("**/*.py"):
        if "__pycache__" in str(py_file):
            continue

        module = parse_python_file(py_file)

        # Index module
        if module.docstring:
            qdrant.upsert(
                collection_name="code",
                points=[PointStruct(
                    id=point_id,
                    vector=embed(f"Module {py_file.name}: {module.docstring}"),
                    payload={
                        "type": "module",
                        "name": py_file.name,
                        "path": str(py_file),
                        "content": module.docstring
                    }
                )]
            )
            point_id += 1

        # Index functions
        for func in module.functions:
            doc_text = create_function_doc(func)
            qdrant.upsert(
                collection_name="code",
                points=[PointStruct(
                    id=point_id,
                    vector=embed(doc_text),
                    payload={
                        "type": "function",
                        "name": func.name,
                        "path": str(py_file),
                        "line": func.line_number,
                        "source": func.source,
                        "docstring": func.docstring
                    }
                )]
            )
            point_id += 1

        # Index classes
        for cls in module.classes:
            doc_text = create_class_doc(cls)
            qdrant.upsert(
                collection_name="code",
                points=[PointStruct(
                    id=point_id,
                    vector=embed(doc_text),
                    payload={
                        "type": "class",
                        "name": cls.name,
                        "path": str(py_file),
                        "line": cls.line_number,
                        "methods": [m.name for m in cls.methods],
                        "docstring": cls.docstring
                    }
                )]
            )
            point_id += 1

    return point_id

def create_function_doc(func: FunctionInfo) -> str:
    """Create searchable text for a function."""
    params = ", ".join([
        f"{p['name']}: {p['type'] or 'Any'}"
        for p in func.parameters
    ])
    return f"""
Function: {func.name}
Parameters: {params}
Returns: {func.return_type or 'None'}
Documentation: {func.docstring or 'No documentation'}
"""

def create_class_doc(cls: ClassInfo) -> str:
    """Create searchable text for a class."""
    methods = ", ".join([m.name for m in cls.methods])
    return f"""
Class: {cls.name}
Base classes: {', '.join(cls.base_classes) or 'None'}
Methods: {methods}
Documentation: {cls.docstring or 'No documentation'}
"""
```

## Step 3: the documentation agent

An agent that generates and maintains documentation.

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class DocumentationResult(BaseModel):
    content: str
    format: str  # "markdown", "openapi", "docstring"
    files_documented: list[str]
    missing_docs: list[str]

doc_agent = Agent(
    model="claude-sonnet-4",
    result_type=DocumentationResult,
    system_prompt="""You are a technical documentation expert. Generate clear,
accurate documentation from code analysis.

Guidelines:
- Lead with what the code DOES, not how
- Use consistent terminology
- Include type information
- Add usage examples when helpful
- Note dependencies and side effects
- Be concise but complete"""
)

@doc_agent.tool
async def search_code(query: str) -> str:
    """Search the codebase for relevant code."""
    query_embedding = embed(query)
    results = qdrant.search(
        collection_name="code",
        query_vector=query_embedding,
        limit=10
    )

    output = []
    for hit in results:
        p = hit.payload
        if p["type"] == "function":
            output.append(f"Function {p['name']} in {p['path']}:{p['line']}\n{p['source']}")
        elif p["type"] == "class":
            output.append(f"Class {p['name']} in {p['path']}:{p['line']}\nMethods: {p['methods']}")
        else:
            output.append(f"Module {p['name']}: {p['content']}")

    return "\n\n---\n\n".join(output)

@doc_agent.tool
async def read_file(path: str) -> str:
    """Read a source file."""
    return Path(path).read_text()

@doc_agent.tool
async def list_files(pattern: str) -> str:
    """List files matching a glob pattern."""
    files = list(Path(".").glob(pattern))
    return "\n".join([str(f) for f in files[:50]])
```

### Generating Documentation

```python
async def generate_module_docs(module_path: str) -> str:
    """Generate documentation for a Python module."""
    result = await doc_agent.run(
        f"""Generate comprehensive documentation for the module at {module_path}.

Include:
1. Module overview
2. All public functions with parameters and return types
3. All classes with their methods
4. Usage examples

Format as Markdown."""
    )
    return result.output.content

async def generate_api_docs(api_dir: str) -> str:
    """Generate OpenAPI documentation for an API."""
    result = await doc_agent.run(
        f"""Analyze the API endpoints in {api_dir} and generate OpenAPI 3.0 documentation.

Include:
- All endpoints with paths and methods
- Request parameters and body schemas
- Response schemas with status codes
- Authentication requirements

Output as YAML."""
    )
    return result.output.content
```

## Step 4: keeping docs in sync

Documentation rots. We prevent that with Git hooks and CI.

### Pre-commit Hook

```python
#!/usr/bin/env python3
# .git/hooks/pre-commit

import subprocess
import sys

def get_changed_python_files():
    """Get Python files staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True
    )
    return [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]

def check_docstrings(files):
    """Check if public functions have docstrings."""
    missing = []
    for file_path in files:
        module = parse_python_file(Path(file_path))
        for func in module.functions:
            if not func.name.startswith("_") and not func.docstring:
                missing.append(f"{file_path}:{func.line_number} - {func.name}")
        for cls in module.classes:
            if not cls.name.startswith("_") and not cls.docstring:
                missing.append(f"{file_path}:{cls.line_number} - {cls.name}")
    return missing

if __name__ == "__main__":
    files = get_changed_python_files()
    if not files:
        sys.exit(0)

    missing = check_docstrings(files)
    if missing:
        print("Missing docstrings:")
        for item in missing:
            print(f"  {item}")
        print("\nAdd docstrings or use --no-verify to bypass.")
        sys.exit(1)
```

### CI Documentation Check

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build documentation index
        run: python build_index.py

      - name: Check documentation coverage
        run: python check_coverage.py --min-coverage 80

      - name: Generate docs
        run: python generate_docs.py

      - name: Upload docs artifact
        uses: actions/upload-artifact@v4
        with:
          name: documentation
          path: docs/
```

### Incremental Updates

```python
import hashlib
import json
from pathlib import Path

HASH_FILE = Path(".doc_hashes.json")

def get_file_hash(path: Path) -> str:
    """Get hash of file contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()

def load_hashes() -> dict:
    """Load previous file hashes."""
    if HASH_FILE.exists():
        return json.loads(HASH_FILE.read_text())
    return {}

def save_hashes(hashes: dict):
    """Save current file hashes."""
    HASH_FILE.write_text(json.dumps(hashes, indent=2))

async def update_changed_docs(root: Path):
    """Only regenerate docs for changed files."""
    old_hashes = load_hashes()
    new_hashes = {}
    changed_files = []

    for py_file in root.glob("**/*.py"):
        file_hash = get_file_hash(py_file)
        new_hashes[str(py_file)] = file_hash

        if old_hashes.get(str(py_file)) != file_hash:
            changed_files.append(py_file)

    if changed_files:
        print(f"Regenerating docs for {len(changed_files)} changed files")
        for file_path in changed_files:
            docs = await generate_module_docs(str(file_path))
            doc_path = Path("docs") / file_path.with_suffix(".md").name
            doc_path.write_text(docs)

    save_hashes(new_hashes)
    return len(changed_files)
```

## Step 5: question answering

Beyond generation, let users ask questions about the code.

```python
class CodeQuestion(BaseModel):
    question: str
    answer: str
    relevant_files: list[str]
    code_snippets: list[str]

qa_agent = Agent(
    model="claude-sonnet-4",
    result_type=CodeQuestion,
    system_prompt="""You answer questions about codebases.

When answering:
1. Search for relevant code first
2. Reference specific files and line numbers
3. Include code snippets when helpful
4. Explain the WHY, not just the WHAT
5. Note any caveats or edge cases"""
)

@qa_agent.tool
async def search(query: str) -> str:
    """Search the code index."""
    results = qdrant.search(
        collection_name="code",
        query_vector=embed(query),
        limit=10
    )
    return format_search_results(results)

@qa_agent.tool
async def read(path: str) -> str:
    """Read a file."""
    return Path(path).read_text()

async def ask_about_code(question: str) -> CodeQuestion:
    """Ask a question about the codebase."""
    result = await qa_agent.run(question)
    return result.output

# Example
answer = await ask_about_code("How does the authentication system handle expired tokens?")
print(answer.answer)
print("Relevant files:", answer.relevant_files)
```

## Complete CLI tool

```python
# doc_gen.py
import asyncio
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer()
console = Console()

@app.command()
def index(root: Path = Path(".")):
    """Index the codebase for documentation."""
    console.print(f"[blue]Indexing {root}...")
    count = index_codebase(root)
    console.print(f"[green]Indexed {count} code elements")

@app.command()
def generate(
    output: Path = Path("docs"),
    format: str = "markdown"
):
    """Generate documentation for the entire codebase."""
    output.mkdir(exist_ok=True)

    async def gen():
        files = list(Path(".").glob("**/*.py"))
        for file_path in track(files, description="Generating..."):
            if "__pycache__" in str(file_path):
                continue
            docs = await generate_module_docs(str(file_path))
            doc_path = output / file_path.with_suffix(".md").name
            doc_path.write_text(docs)

    asyncio.run(gen())
    console.print(f"[green]Documentation written to {output}")

@app.command()
def update():
    """Update documentation for changed files only."""
    async def upd():
        count = await update_changed_docs(Path("."))
        console.print(f"[green]Updated {count} files")

    asyncio.run(upd())

@app.command()
def ask(question: str):
    """Ask a question about the codebase."""
    async def qa():
        result = await ask_about_code(question)
        console.print(f"\n[bold]{result.answer}[/bold]\n")
        if result.relevant_files:
            console.print("[dim]Relevant files:[/dim]")
            for f in result.relevant_files:
                console.print(f"  - {f}")

    asyncio.run(qa())

@app.command()
def coverage():
    """Check documentation coverage."""
    total = 0
    documented = 0

    for py_file in Path(".").glob("**/*.py"):
        if "__pycache__" in str(py_file):
            continue
        module = parse_python_file(py_file)
        for func in module.functions:
            if not func.name.startswith("_"):
                total += 1
                if func.docstring:
                    documented += 1

    pct = (documented / total * 100) if total else 0
    color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    console.print(f"Coverage: [{color}]{pct:.1f}%[/{color}] ({documented}/{total})")

if __name__ == "__main__":
    app()
```

## Cost analysis

For a 50,000 LOC codebase:

| Operation | Frequency | Monthly Cost |
|-----------|-----------|--------------|
| Initial indexing | Once | ~$5 |
| Embedding updates | Daily | ~$10 |
| Doc generation | Weekly | ~$50 |
| Q&A queries | 100/day | ~$30 |
| **Total** | | ~$95/month |

Using Gemini's free embedding tier and Claude Sonnet, this is affordable for most teams.

## Lessons learned

### What Worked

1. **AST > Regex**: Structured parsing caught edge cases that regex missed.

2. **Incremental updates**: Only regenerating changed files cut costs by 80%.

3. **Q&A as a feature**: Developers used the ask command more than generated docs.

### What Didn't

1. **Over-generation**: Early versions added too much boilerplate. Less is more.

2. **Cross-file context**: Single-file analysis missed important relationships. Solution: include import context.

3. **Generated examples**: LLM-generated code examples sometimes had bugs. Added validation.

### V2 Improvements

- **Architecture diagrams**: Generate Mermaid diagrams from import graphs
- **Changelog generation**: Diff-based release notes
- **Doctest integration**: Auto-generate tests from documentation
- **Multi-language**: Rust and Go support

## Getting started: two approaches

**Simple approach**: Ask Claude directly. No infrastructure needed.

```python
from anthropic import Anthropic
from pathlib import Path

client = Anthropic()

def document_file_simple(file_path: str) -> str:
    """Generate documentation for a single file using Claude."""
    code = Path(file_path).read_text()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Generate comprehensive documentation for this Python module.

Include:
- Module overview
- Each function with parameters, return types, and description
- Each class with its purpose and methods
- Usage examples

Format as Markdown.

```python
{code}
```"""
        }]
    )
    return response.content[0].text

# Document a single file
docs = document_file_simple("src/auth.py")
Path("docs/auth.md").write_text(docs)
```

Paste code, get docs. Works for one-off documentation needs. Costs ~$0.02 per file.

**Production approach**: The full system from this chapter—AST parsing, vector index, agent orchestration—for continuous documentation at scale.

```bash
# CLI for production documentation
$ python doc_gen.py index                    # Build searchable index
$ python doc_gen.py generate --output docs/  # Generate all docs
$ python doc_gen.py update                   # Regenerate changed files only
$ python doc_gen.py ask "How does auth work?" # Query the codebase
$ python doc_gen.py coverage                  # Check documentation coverage
```

```python
# Programmatic usage
from doc_generator import DocAgent

agent = DocAgent(codebase_path="./src")
agent.index()  # One-time setup

# Generate on demand
docs = await agent.generate_module_docs("src/auth.py")

# Keep in sync with CI
changed = await agent.update_changed_docs()

# Answer questions
answer = await agent.ask("Where is the token refresh logic?")
```

The simple approach documents files on demand. The production approach maintains living documentation that stays in sync with your code. Start simple when you need quick docs; invest in the pipeline when documentation freshness matters.

## The takeaway

Documentation doesn't have to rot. With the right automation:

1. **Parse** code into structured data
2. **Index** for searchable retrieval
3. **Generate** on demand, not manually
4. **Sync** with CI/CD pipelines
5. **Answer** questions interactively

The documentation generator isn't a replacement for human writing—it's a first draft that humans refine. It catches the stuff developers forget (missing docstrings, outdated examples) and handles the tedium (formatting, consistency).

Most importantly, it answers the question developers actually ask: "What does this code do?" Without opening a browser or searching through stale wikis.

![A robot librarian organizing code scrolls into documentation shelves, steampunk library aesthetic](https://pixy.vexy.art/)

---

*Next: Chapter 12 builds an MCP server that connects Claude to FontLab for automated font manipulation.*
