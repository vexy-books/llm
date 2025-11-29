# Chapter 12B: The real-time collaborator

*Live font editing with AI assistance.*

> "The best collaborative work comes from blending the human need to be in the flow with the machine's ability to never forget."
> — John Maeda

## Beyond request-response

Chapter 12 built an MCP server that processes font commands one at a time. You ask, it does, you see the result. That works for batch operations.

But creative work happens differently. A designer adjusts a curve, sees how it looks, adjusts again. The feedback loop is tight—seconds, not minutes. They want to say "make the 'g' feel lighter" and watch it happen while they speak.

This chapter builds a real-time font editing system:

- **WebSocket MCP servers** for persistent, bidirectional connections
- **Streaming glyph modifications** that show changes incrementally
- **Collaborative editing** where AI and human work on the same file simultaneously

The result: AI that feels like a co-designer sitting next to you, not a command-line tool you invoke.

## Imagine...

Imagine a pair programming session where your partner never gets tired, never loses context, and can execute changes as fast as you describe them. You say "let's try making the stem thicker" and your partner's hands are already moving before you finish the sentence.

They see your screen. They know what you're looking at. When you hesitate over a detail, they suggest options. When you make a change, they remember why. The conversation and the work happen simultaneously—talking isn't separate from doing.

Real-time collaboration with AI aims for this feeling. Not a chatbot you message between edits, but a presence in your workflow that sees what you see, understands what you're trying to achieve, and can act instantly when you give direction.

The technical challenge: maintaining state, streaming updates, and handling the chaos of two entities (human and AI) modifying the same data simultaneously.

---

## WebSocket MCP architecture

Traditional MCP uses stdio—standard input/output. Great for CLI tools, but no persistence between calls. WebSockets give us:

- **Persistent connection**: Server remembers state across interactions
- **Bidirectional communication**: Server can push updates to client
- **Low latency**: No connection setup overhead per message

### Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   Claude Code   │◄───────────────────│   Font Editor   │
│   (AI Agent)    │    WebSocket       │   (Human UI)    │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │ WebSocket                            │ WebSocket
         │                                      │
         └──────────────┬───────────────────────┘
                        │
               ┌────────▼────────┐
               │  Collaboration  │
               │     Server      │
               └────────┬────────┘
                        │
               ┌────────▼────────┐
               │   Font State    │
               │  (fontTools)    │
               └─────────────────┘
```

### The server

```python
# collab_server.py
import asyncio
import json
from pathlib import Path
from datetime import datetime
from fontTools.ttLib import TTFont

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Font Collaboration Server")

class FontState:
    """Manages the current font being edited."""

    def __init__(self):
        self.font: Optional[TTFont] = None
        self.font_path: Optional[Path] = None
        self.history: list[dict] = []
        self.clients: list[WebSocket] = []

    async def load_font(self, path: str) -> dict:
        """Load a font file into memory."""
        self.font_path = Path(path)
        self.font = TTFont(self.font_path)
        self.history = []

        info = self._get_font_info()
        await self.broadcast({
            "type": "font_loaded",
            "data": info
        })
        return info

    def _get_font_info(self) -> dict:
        """Extract current font information."""
        if not self.font:
            return {}

        name_table = self.font["name"]
        def get_name(name_id):
            record = name_table.getName(name_id, 3, 1, 1033)
            return record.toUnicode() if record else ""

        return {
            "family": get_name(1),
            "style": get_name(2),
            "glyph_count": self.font["maxp"].numGlyphs,
            "units_per_em": self.font["head"].unitsPerEm,
            "ascender": self.font["hhea"].ascent,
            "descender": self.font["hhea"].descent,
        }

    async def modify(self, operation: dict) -> dict:
        """Apply a modification to the font."""
        if not self.font:
            return {"error": "No font loaded"}

        op_type = operation.get("type")
        result = {}

        if op_type == "set_metric":
            result = self._set_metric(
                operation["metric"],
                operation["value"]
            )
        elif op_type == "scale_glyph":
            result = self._scale_glyph(
                operation["glyph"],
                operation["factor"]
            )
        elif op_type == "adjust_spacing":
            result = self._adjust_spacing(
                operation["glyph"],
                operation["left_delta"],
                operation["right_delta"]
            )

        # Record history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "result": result
        })

        # Broadcast change to all clients
        await self.broadcast({
            "type": "modification",
            "operation": operation,
            "result": result,
            "font_state": self._get_font_info()
        })

        return result

    def _set_metric(self, metric: str, value: int) -> dict:
        """Set a font metric."""
        if metric == "ascender":
            self.font["hhea"].ascent = value
            if "OS/2" in self.font:
                self.font["OS/2"].sTypoAscender = value
        elif metric == "descender":
            self.font["hhea"].descent = value
            if "OS/2" in self.font:
                self.font["OS/2"].sTypoDescender = value
        elif metric == "line_gap":
            self.font["hhea"].lineGap = value

        return {"metric": metric, "new_value": value}

    def _scale_glyph(self, glyph_name: str, factor: float) -> dict:
        """Scale a glyph by a factor."""
        # Note: Actual glyph transformation requires more complex code
        # This is simplified for illustration
        hmtx = self.font["hmtx"]
        if glyph_name in hmtx.metrics:
            width, lsb = hmtx[glyph_name]
            new_width = int(width * factor)
            new_lsb = int(lsb * factor)
            hmtx[glyph_name] = (new_width, new_lsb)
            return {"glyph": glyph_name, "new_width": new_width}
        return {"error": f"Glyph {glyph_name} not found"}

    def _adjust_spacing(
        self,
        glyph_name: str,
        left_delta: int,
        right_delta: int
    ) -> dict:
        """Adjust sidebearings for a glyph."""
        hmtx = self.font["hmtx"]
        if glyph_name in hmtx.metrics:
            width, lsb = hmtx[glyph_name]
            new_lsb = lsb + left_delta
            new_width = width + left_delta + right_delta
            hmtx[glyph_name] = (new_width, new_lsb)
            return {
                "glyph": glyph_name,
                "new_width": new_width,
                "new_lsb": new_lsb
            }
        return {"error": f"Glyph {glyph_name} not found"}

    async def save(self, path: str = None) -> dict:
        """Save the current font state."""
        if not self.font:
            return {"error": "No font loaded"}

        save_path = Path(path) if path else self.font_path
        self.font.save(save_path)

        await self.broadcast({
            "type": "font_saved",
            "path": str(save_path)
        })

        return {"saved": str(save_path)}

    async def undo(self) -> dict:
        """Undo the last operation (reload font for now)."""
        if self.font_path:
            # Simple undo: reload original
            return await self.load_font(str(self.font_path))
        return {"error": "Cannot undo"}

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception:
                pass  # Client disconnected

    async def add_client(self, websocket: WebSocket):
        """Register a new client."""
        await websocket.accept()
        self.clients.append(websocket)

        # Send current state
        if self.font:
            await websocket.send_json({
                "type": "sync",
                "font_state": self._get_font_info(),
                "history": self.history[-10:]  # Last 10 operations
            })

    def remove_client(self, websocket: WebSocket):
        """Unregister a client."""
        if websocket in self.clients:
            self.clients.remove(websocket)


# Global state
font_state = FontState()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket handler for collaboration."""
    await font_state.add_client(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "load":
                result = await font_state.load_font(data["path"])
            elif action == "modify":
                result = await font_state.modify(data["operation"])
            elif action == "save":
                result = await font_state.save(data.get("path"))
            elif action == "undo":
                result = await font_state.undo()
            elif action == "query":
                result = {"font_state": font_state._get_font_info()}
            else:
                result = {"error": f"Unknown action: {action}"}

            await websocket.send_json({
                "type": "response",
                "action": action,
                "result": result
            })

    except WebSocketDisconnect:
        font_state.remove_client(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

## Streaming modifications

For large operations, stream progress incrementally.

### Streaming glyph updates

```python
import asyncio
from typing import AsyncGenerator

class StreamingFontOperations:
    """Operations that stream their progress."""

    def __init__(self, font_state: FontState):
        self.state = font_state

    async def batch_scale_glyphs(
        self,
        glyph_names: list[str],
        factor: float
    ) -> AsyncGenerator[dict, None]:
        """Scale multiple glyphs, yielding progress."""
        total = len(glyph_names)

        for i, glyph_name in enumerate(glyph_names):
            result = self.state._scale_glyph(glyph_name, factor)

            yield {
                "type": "progress",
                "glyph": glyph_name,
                "result": result,
                "progress": (i + 1) / total,
                "completed": i + 1,
                "total": total
            }

            # Allow other tasks to run
            await asyncio.sleep(0)

        yield {
            "type": "complete",
            "total_modified": total
        }

    async def global_spacing_adjust(
        self,
        percent_change: float
    ) -> AsyncGenerator[dict, None]:
        """Adjust spacing for all glyphs."""
        hmtx = self.state.font["hmtx"]
        glyph_names = list(hmtx.metrics.keys())
        total = len(glyph_names)

        for i, glyph_name in enumerate(glyph_names):
            width, lsb = hmtx[glyph_name]

            # Adjust by percentage
            delta = int(width * percent_change / 100)
            left_delta = delta // 2
            right_delta = delta - left_delta

            hmtx[glyph_name] = (width + delta, lsb + left_delta)

            if i % 50 == 0:  # Report every 50 glyphs
                yield {
                    "type": "progress",
                    "progress": (i + 1) / total,
                    "last_glyph": glyph_name
                }

            await asyncio.sleep(0)

        yield {
            "type": "complete",
            "glyphs_modified": total,
            "percent_change": percent_change
        }


# Add to WebSocket handler
@app.websocket("/ws/stream")
async def streaming_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming operations."""
    await websocket.accept()

    streaming_ops = StreamingFontOperations(font_state)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "batch_scale":
                async for update in streaming_ops.batch_scale_glyphs(
                    data["glyphs"], data["factor"]
                ):
                    await websocket.send_json(update)
                    await font_state.broadcast(update)

            elif action == "global_spacing":
                async for update in streaming_ops.global_spacing_adjust(
                    data["percent"]
                ):
                    await websocket.send_json(update)
                    await font_state.broadcast(update)

    except WebSocketDisconnect:
        pass
```

## The AI collaborator

Now connect an AI agent that can participate in real-time editing.

### AI client

```python
import websockets
import json
from pydantic_ai import Agent
from pydantic import BaseModel

class FontEditSuggestion(BaseModel):
    operation: str
    parameters: dict
    reasoning: str
    confidence: float

class AICollaborator:
    """AI agent that participates in real-time font editing."""

    def __init__(self, server_url: str = "ws://localhost:8765/ws"):
        self.server_url = server_url
        self.ws = None
        self.current_font_state = {}

        self.agent = Agent(
            model="claude-sonnet-4",
            result_type=FontEditSuggestion,
            system_prompt="""You are a typography expert collaborating on font design.

When given the current font state and a design goal:
1. Analyze what modifications would achieve the goal
2. Suggest specific, measurable changes
3. Explain your reasoning in typographic terms
4. Rate your confidence in the suggestion

Available operations:
- set_metric: Change ascender, descender, line_gap
- scale_glyph: Scale a glyph by a factor
- adjust_spacing: Modify left/right sidebearings

Be conservative with changes. Small adjustments compound."""
        )

    async def connect(self):
        """Connect to the collaboration server."""
        self.ws = await websockets.connect(self.server_url)

        # Wait for sync message
        response = await self.ws.recv()
        data = json.loads(response)
        if data["type"] == "sync":
            self.current_font_state = data["font_state"]
            print(f"Connected. Font: {self.current_font_state.get('family')}")

    async def listen(self):
        """Listen for updates and potentially respond."""
        async for message in self.ws:
            data = json.loads(message)

            if data["type"] == "modification":
                self.current_font_state = data.get("font_state", {})
                print(f"Font updated: {data['operation']}")

            elif data["type"] == "font_loaded":
                self.current_font_state = data["data"]
                print(f"Font loaded: {self.current_font_state.get('family')}")

    async def suggest_modification(self, goal: str) -> FontEditSuggestion:
        """Generate a modification suggestion based on a goal."""
        prompt = f"""Current font state:
{json.dumps(self.current_font_state, indent=2)}

Design goal: {goal}

Suggest a single modification to move toward this goal."""

        result = await self.agent.run(prompt)
        return result.data

    async def apply_suggestion(self, suggestion: FontEditSuggestion):
        """Apply a suggestion to the font."""
        operation = {
            "type": suggestion.operation,
            **suggestion.parameters
        }

        await self.ws.send(json.dumps({
            "action": "modify",
            "operation": operation
        }))

        # Wait for response
        response = await self.ws.recv()
        return json.loads(response)

    async def collaborative_edit(self, goal: str, max_iterations: int = 5):
        """Iteratively suggest modifications toward a goal."""
        print(f"Working toward: {goal}")

        for i in range(max_iterations):
            suggestion = await self.suggest_modification(goal)
            print(f"\nIteration {i+1}:")
            print(f"  Suggestion: {suggestion.operation}")
            print(f"  Parameters: {suggestion.parameters}")
            print(f"  Reasoning: {suggestion.reasoning}")
            print(f"  Confidence: {suggestion.confidence:.0%}")

            if suggestion.confidence < 0.3:
                print("  Low confidence, stopping.")
                break

            # Apply the suggestion
            result = await self.apply_suggestion(suggestion)
            print(f"  Result: {result}")

            # Small delay to let human review
            await asyncio.sleep(1)


# Usage
async def main():
    collaborator = AICollaborator()
    await collaborator.connect()

    # Start listening in background
    listen_task = asyncio.create_task(collaborator.listen())

    # Collaborative editing session
    await collaborator.collaborative_edit(
        "Make this font feel more modern and open"
    )

    listen_task.cancel()

asyncio.run(main())
```

## Collaborative editing protocol

When multiple agents (human + AI) edit simultaneously, conflicts arise. A simple protocol handles this.

### Operational transformation (simplified)

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class Operation:
    """A single editing operation."""
    id: str
    timestamp: float
    client_id: str
    op_type: str
    target: str  # glyph name, metric name, etc.
    value: any
    parent_id: Optional[str] = None  # Previous operation this depends on

class CollaborationProtocol:
    """Handles concurrent editing from multiple clients."""

    def __init__(self):
        self.operations: list[Operation] = []
        self.operation_id = 0

    def create_operation(
        self,
        client_id: str,
        op_type: str,
        target: str,
        value: any
    ) -> Operation:
        """Create a new operation."""
        self.operation_id += 1
        parent = self.operations[-1].id if self.operations else None

        return Operation(
            id=f"op_{self.operation_id}",
            timestamp=time.time(),
            client_id=client_id,
            op_type=op_type,
            target=target,
            value=value,
            parent_id=parent
        )

    def can_apply(self, operation: Operation) -> bool:
        """Check if operation can be applied (no conflicts)."""
        # Check if parent exists
        if operation.parent_id:
            parent_exists = any(
                op.id == operation.parent_id
                for op in self.operations
            )
            if not parent_exists:
                return False

        # Check for conflicting concurrent operations
        for op in self.operations:
            if (op.timestamp > operation.timestamp - 1 and  # Within 1 second
                op.target == operation.target and
                op.client_id != operation.client_id):
                return False

        return True

    def apply(self, operation: Operation) -> dict:
        """Apply an operation if possible."""
        if not self.can_apply(operation):
            return {
                "status": "conflict",
                "operation_id": operation.id,
                "message": "Conflicting operation detected"
            }

        self.operations.append(operation)
        return {
            "status": "applied",
            "operation_id": operation.id
        }

    def get_history_since(self, operation_id: str) -> list[Operation]:
        """Get all operations since a given operation."""
        found = False
        result = []
        for op in self.operations:
            if found:
                result.append(op)
            if op.id == operation_id:
                found = True
        return result
```

### Conflict resolution UI

```python
class ConflictResolver:
    """Resolve editing conflicts between human and AI."""

    def __init__(self, protocol: CollaborationProtocol):
        self.protocol = protocol
        self.pending_conflicts: list[tuple[Operation, Operation]] = []

    async def handle_conflict(
        self,
        human_op: Operation,
        ai_op: Operation,
        websocket: WebSocket
    ):
        """Present conflict to human for resolution."""
        await websocket.send_json({
            "type": "conflict",
            "human_operation": {
                "type": human_op.op_type,
                "target": human_op.target,
                "value": human_op.value
            },
            "ai_operation": {
                "type": ai_op.op_type,
                "target": ai_op.target,
                "value": ai_op.value
            },
            "options": [
                {"id": "human", "description": "Keep human change"},
                {"id": "ai", "description": "Keep AI change"},
                {"id": "both", "description": "Apply both (if compatible)"},
                {"id": "neither", "description": "Discard both"}
            ]
        })

        # Wait for resolution
        response = await websocket.receive_json()
        choice = response.get("resolution")

        if choice == "human":
            self.protocol.apply(human_op)
        elif choice == "ai":
            self.protocol.apply(ai_op)
        elif choice == "both":
            self.protocol.apply(human_op)
            # Rebase AI operation
            ai_op.parent_id = human_op.id
            self.protocol.apply(ai_op)

        return {"resolved": choice}
```

## Real-time preview

Show font changes instantly in the browser.

### WebSocket font preview

```html
<!DOCTYPE html>
<html>
<head>
    <title>Font Collaborator</title>
    <style>
        #preview {
            font-size: 48px;
            padding: 20px;
            border: 1px solid #ccc;
            min-height: 200px;
        }
        #metrics {
            font-family: monospace;
            background: #f5f5f5;
            padding: 10px;
        }
        .operation {
            padding: 5px;
            margin: 2px 0;
            background: #e0e0e0;
        }
        .ai-operation { background: #d4edda; }
        .human-operation { background: #cce5ff; }
    </style>
</head>
<body>
    <h1>Font Collaboration</h1>

    <div id="preview">
        <span id="sample-text">Hamburgefonts</span>
    </div>

    <div id="metrics">
        <pre id="font-info">No font loaded</pre>
    </div>

    <div id="history">
        <h3>Operation History</h3>
        <div id="operations"></div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8765/ws');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'sync' || data.type === 'font_loaded') {
                updateFontInfo(data.font_state || data.data);
                refreshFontFace();
            }

            if (data.type === 'modification') {
                updateFontInfo(data.font_state);
                addOperationToHistory(data.operation);
                refreshFontFace();
            }

            if (data.type === 'progress') {
                updateProgress(data);
            }
        };

        function updateFontInfo(state) {
            document.getElementById('font-info').textContent =
                JSON.stringify(state, null, 2);
        }

        function addOperationToHistory(op) {
            const div = document.createElement('div');
            div.className = `operation ${op.source === 'ai' ? 'ai-operation' : 'human-operation'}`;
            div.textContent = `${op.type}: ${JSON.stringify(op)}`;
            document.getElementById('operations').prepend(div);
        }

        function refreshFontFace() {
            // Force font reload by adding timestamp
            const preview = document.getElementById('preview');
            const fontUrl = `/font?t=${Date.now()}`;

            const style = document.createElement('style');
            style.textContent = `
                @font-face {
                    font-family: 'EditingFont';
                    src: url('${fontUrl}') format('truetype');
                }
                #preview { font-family: 'EditingFont', sans-serif; }
            `;
            document.head.appendChild(style);
        }

        function updateProgress(data) {
            console.log(`Progress: ${(data.progress * 100).toFixed(1)}%`);
        }
    </script>
</body>
</html>
```

## Complete collaboration server

```python
# full_collab_server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fontTools.ttLib import TTFont
from pathlib import Path
import asyncio
import json

app = FastAPI()

# Serve static files (HTML preview)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
class CollaborativeFont:
    def __init__(self):
        self.font: TTFont | None = None
        self.font_path: Path | None = None
        self.clients: list[WebSocket] = []
        self.protocol = CollaborationProtocol()

    async def broadcast(self, message: dict, exclude: WebSocket = None):
        for client in self.clients:
            if client != exclude:
                try:
                    await client.send_json(message)
                except:
                    pass

collab_font = CollaborativeFont()

@app.get("/")
async def get_preview():
    return FileResponse("static/preview.html")

@app.get("/font")
async def get_font():
    """Serve current font file."""
    if collab_font.font:
        # Save to temp and serve
        temp_path = Path("/tmp/current_font.ttf")
        collab_font.font.save(temp_path)
        return FileResponse(temp_path, media_type="font/ttf")
    return {"error": "No font loaded"}

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    collab_font.clients.append(websocket)

    # Send current state
    if collab_font.font:
        await websocket.send_json({
            "type": "sync",
            "font_state": get_font_info(collab_font.font)
        })

    try:
        while True:
            data = await websocket.receive_json()
            await handle_message(data, websocket)
    except WebSocketDisconnect:
        collab_font.clients.remove(websocket)

async def handle_message(data: dict, sender: WebSocket):
    action = data.get("action")

    if action == "load":
        collab_font.font_path = Path(data["path"])
        collab_font.font = TTFont(collab_font.font_path)
        await collab_font.broadcast({
            "type": "font_loaded",
            "data": get_font_info(collab_font.font)
        })

    elif action == "modify":
        op = collab_font.protocol.create_operation(
            client_id=str(id(sender)),
            op_type=data["operation"]["type"],
            target=data["operation"].get("target", "global"),
            value=data["operation"]
        )

        result = collab_font.protocol.apply(op)

        if result["status"] == "applied":
            apply_operation(collab_font.font, data["operation"])
            await collab_font.broadcast({
                "type": "modification",
                "operation": data["operation"],
                "font_state": get_font_info(collab_font.font)
            })

    elif action == "save":
        if collab_font.font and collab_font.font_path:
            save_path = data.get("path", str(collab_font.font_path))
            collab_font.font.save(save_path)
            await collab_font.broadcast({
                "type": "saved",
                "path": save_path
            })

def get_font_info(font: TTFont) -> dict:
    name = font["name"]
    def get_name(id):
        r = name.getName(id, 3, 1, 1033)
        return r.toUnicode() if r else ""

    return {
        "family": get_name(1),
        "ascender": font["hhea"].ascent,
        "descender": font["hhea"].descent,
        "units_per_em": font["head"].unitsPerEm,
        "glyph_count": font["maxp"].numGlyphs
    }

def apply_operation(font: TTFont, operation: dict):
    op_type = operation["type"]

    if op_type == "set_metric":
        metric = operation["metric"]
        value = operation["value"]
        if metric == "ascender":
            font["hhea"].ascent = value
        elif metric == "descender":
            font["hhea"].descent = value

    elif op_type == "scale_glyph":
        glyph = operation["glyph"]
        factor = operation["factor"]
        if glyph in font["hmtx"].metrics:
            w, lsb = font["hmtx"][glyph]
            font["hmtx"][glyph] = (int(w * factor), int(lsb * factor))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

## The takeaway

Real-time collaboration changes how AI assists creative work:

1. **WebSockets enable persistence**: The server maintains state, clients stay synchronized
2. **Streaming shows progress**: Large operations feel responsive with incremental updates
3. **Conflict resolution matters**: When human and AI edit simultaneously, someone must arbitrate
4. **Live preview completes the loop**: Seeing changes instantly maintains creative flow

The same architecture applies beyond fonts:
- **Code editors**: AI suggesting refactors while you type
- **Design tools**: AI adjusting layouts as you add elements
- **Document editing**: AI fixing grammar as you write

The key insight: AI collaboration works best when it feels like a conversation happening in parallel with work, not a separate step before or after.

![Two hands—one human, one robotic—working on the same letter design, with streams of data flowing between them, watercolor technical illustration](https://pixy.vexy.art/)

---

*Next: Part V covers the economics—subscriptions, cost optimization, and what's coming next.*
