---
{}
---

# Getting started

Material for MkDocs is a powerful documentation framework on top of [MkDocs],
a static site generator for project documentation.[^1] If you're familiar with
Python, you can install Material for MkDocs with [`pip`][pip], the Python
package manager. If not, we recommend using [`docker`][docker].

  [^1]:
    In 2016, Material for MkDocs started out as a simple theme for MkDocs, but
    over the course of several years, it's now much more than that – with the
    many built-in plugins, settings, and countless customization abilities,
    Material for MkDocs is now one of the simplest and most powerful frameworks
    for creating documentation for your project.

  [MkDocs]: https://www.mkdocs.org
  [pip]: #with-pip
  [docker]: #with-docker

## Installation

### with pip <small>recommended</small> { #with-pip data-toc-label="with pip" }

Material for MkDocs is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. Open up a terminal and install
Material for MkDocs with:

=== "Latest"

    ``` sh
    pip install mkdocs-material
    ```

=== "9.x"

    ``` sh
    pip install mkdocs-material=="9.*" # (1)!
    ```

    1.  Material for MkDocs uses [semantic versioning][^2], which is why it's a
        good idea to limit upgrades to the current major version.

        This will make sure that you don't accidentally [upgrade to the next
        major version], which may include breaking changes that silently corrupt
        your site. Additionally, you can use `pip freeze` to create a lockfile,
        so builds are reproducible at all times:

        ```
        pip freeze > requirements.txt
        ```

        Now, the lockfile can be used for installation:

        ```
        pip install -r requirements.txt
        ```

  [^2]:
    Note that improvements of existing features are sometimes released as
    patch releases, like for example improved rendering of content tabs, as
    they're not considered to be new features.

This will automatically install compatible versions of all dependencies:
[MkDocs], [Markdown], [Pygments] and [Python Markdown Extensions]. Material for
MkDocs always strives to support the latest versions, so there's no need to
install those packages separately.

---

:fontawesome-brands-youtube:{ style="color: #EE0F0F" }
__[How to set up Material for MkDocs]__ by @james-willett – :octicons-clock-24:
27m – Learn how to create and host a documentation site using Material for
MkDocs on GitHub Pages in a step-by-step guide.

  [How to set up Material for MkDocs]: https://www.youtube.com/watch?v=xlABhbnNrfI

---

!!! tip

    If you don't have prior experience with Python, we recommend reading
    [Using Python's pip to Manage Your Projects' Dependencies], which is a
    really good introduction on the mechanics of Python package management and
    helps you troubleshoot if you run into errors.

  [Python package]: https://pypi.org/project/mkdocs-material/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [semantic versioning]: https://semver.org/
  [upgrade to the next major version]: upgrade.md
  [Markdown]: https://python-markdown.github.io/
  [Pygments]: https://pygments.org/
  [Python Markdown Extensions]: https://facelessuser.github.io/pymdown-extensions/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/

### with docker

The official [Docker image] is a great way to get up and running in a few
minutes, as it comes with all dependencies pre-installed. Open up a terminal
and pull the image with:

=== "Latest"

    ```
    docker pull squidfunk/mkdocs-material
    ```

=== "9.x"

    ```
    docker pull squidfunk/mkdocs-material:9
    ```

The `mkdocs` executable is provided as an entry point and `serve` is the
default command. If you're not familiar with Docker don't worry, we have you
covered in the following sections.

The following plugins are bundled with the Docker image:

- [mkdocs-minify-plugin]
- [mkdocs-redirects]

  [Docker image]: https://hub.docker.com/r/squidfunk/mkdocs-material/
  [mkdocs-minify-plugin]: https://github.com/byrnereese/mkdocs-minify-plugin
  [mkdocs-redirects]: https://github.com/datarobot/mkdocs-redirects

???+ warning

    The Docker container is intended for local previewing purposes only and
    is not suitable for deployment. This is because the web server used by
    MkDocs for live previews is not designed for production use and may have
    security vulnerabilities.

??? question "How to add plugins to the Docker image?"

    Material for MkDocs only bundles selected plugins in order to keep the size
    of the official image small. If the plugin you want to use is not included,
    you can add them easily. Create a `Dockerfile` and extend the official image:

    ``` Dockerfile title="Dockerfile"
    FROM squidfunk/mkdocs-material
    RUN pip install mkdocs-macros-plugin
    RUN pip install mkdocs-glightbox
    ```

    Next, build the image with the following command:

    ```
    docker build -t squidfunk/mkdocs-material .
    ```

    The new image will have additional packages installed and can be used
    exactly like the official image.

### with git

Material for MkDocs can be directly used from [GitHub] by cloning the
repository into a subfolder of your project root which might be useful if you
want to use the very latest version:

```
git clone https://github.com/squidfunk/mkdocs-material.git
```

Next, install the theme and its dependencies with:

```
pip install -e mkdocs-material
```

  [GitHub]: https://github.com/squidfunk/mkdocs-material


------------------------------------------------------------


Here is a detailed analysis of the Zensical project based on the provided codebase.

### TLDR: Codebase Overview

The Zensical project is a monorepo containing three main sub-projects: `zensical` (the core static site generator), `ui` (the frontend templates and assets), and `zrx` (a collection of foundational Rust libraries). The core logic is written in Rust for performance, with a Python wrapper to provide a user-friendly command-line interface and extensibility, similar to the architecture of tools like Ruff.

#### **Function and Method Signatures**

Below is a summarized view of the key functions, structs, and methods across the Zensical codebase.

<details>
<summary><code>📁 zensical/ (Core application)</code></summary>

```rust
// file: zensical/zensical/crates/zensical/src/lib.rs
mod config;
mod server;
mod structure;
mod template;
mod watcher;
mod workflow;

// file: zensical/zensical/crates/zensical/src/config.rs
struct Config { ... }

// file: zensical/zensical/crates/zensical/src/server.rs
struct ServeOptions { ... }
fn serve(options: ServeOptions) -> Result<()> { ... }

// file: zensical/zensical/crates/zensical/src/structure/page.rs
struct Page { ... }

// file: zensical/zensical/crates/zensical/src/structure/nav.rs
struct Navigation { ... }

// file: zensical/zensical/crates/zensical/src/template.rs
struct Template { ... }
impl Template {
    fn new(path: &Path) -> Result<Self, Error> { ... }
    fn render<T: Serialize>(&self, value: T) -> Result<String, Error> { ... }
}

// file: zensical/zensical/crates/zensical-watch/src/lib.rs
mod agent;
pub struct Watcher { ... }
impl Watcher {
    pub fn new() -> Self { ... }
    pub fn watch(&self, path: &Path) -> Result<(), Error> { ... }
    pub fn listen(&self) -> Receiver<Event> { ... }
}
``````python
# file: zensical/zensical/python/zensical/main.py
def cli(): ...
def execute_build(config_file: str | None, **kwargs): ...
def execute_serve(config_file: str | None, **kwargs): ...
def new_project(directory: str | None, **kwargs): ...

# file: zensical/zensical/python/zensical/markdown.py
def render(content: str, path: str) -> dict: ...
```
</details>

<details>
<summary><code>📁 ui/ (User Interface)</code></summary>

```typescript
// file: zensical/ui/src/assets/javascripts/bundle.ts
// Entry point for the frontend application, initializes all components and integrations.

// file: zensical/ui/src/assets/javascripts/components/_/index.ts
export type ComponentType = | "header" | "sidebar" | "content" | ...;
export function getComponentElement<T extends ComponentType>(type: T, node?: ParentNode): ComponentTypeMap[T];
export function getComponentElements<T extends ComponentType>(type: T, node?: ParentNode): ComponentTypeMap[T][];

// file: zensical/ui/src/assets/javascripts/components/header/index.ts
export interface Header { height: number; hidden: boolean; }
export function watchHeader(el: HTMLElement, options: WatchOptions): Observable<Header>;
export function mountHeader(el: HTMLElement, options: MountOptions): Observable<Component<Header>>;

// file: zensical/ui/src/assets/javascripts/components/sidebar/index.ts
export interface Sidebar { height: number; locked: boolean; }
export function watchSidebar(el: HTMLElement, options: WatchOptions): Observable<Sidebar>;
export function mountSidebar(el: HTMLElement, options: MountOptions): Observable<Component<Sidebar>>;

// file: zensical/ui/src/assets/javascripts/integrations/instant/index.ts
export function setupInstantNavigation(options: SetupOptions): Observable<Document>;
```
</details>

<details>
<summary><code>📁 zrx/ (Zen Reactive Extensions)</code></summary>

```rust
// file: zensical/zrx/crates/zrx-scheduler/src/scheduler/action.rs
trait Action {
    fn descriptor() -> Descriptor;
    fn call(&self, input: &Value) -> Result<Value, Error>;
}

// file: zensical/zrx/crates/zrx-scheduler/src/scheduler.rs
struct Scheduler { ... }
impl Scheduler {
    pub fn new() -> Self { ... }
    pub fn add<A>(&mut self, action: A) -> Result<Id, Error> where A: Action;
    pub fn connect(&mut self, from: Id, to: Id) -> Result<(), Error>;
    pub fn run(&mut self) -> Result<(), Error>;
}

// file: zensical/zrx/crates/zrx-stream/src/stream.rs
struct Stream<I, T> where I: Id, T: Value { ... }
impl<I, T> Stream<I, T> where I: Id, T: Value {
    pub fn map<F, U>(&self, f: F) -> Stream<I, U> where ...;
    pub fn filter<F>(&self, f: F) -> Stream<I, T> where ...;
    pub fn reduce<F, U>(&self, f: F, init: U) -> Stream<I, U> where ...;
}

// file: zensical/zrx/crates/zrx-stream/src/stream/combinator/tuple/ext.rs
trait StreamTupleExt<I, S> {
    fn join(self) -> Stream<I, S::Item> where ...;
    fn left_join(self) -> Stream<I, S::Item> where ...;
}

// file: zensical/zrx/crates/zrx-store/src/store.rs
trait Store<K, V> where K: Key {
    fn get<Q>(&self, key: &Q) -> Option<&V> where ...;
    fn len(&self) -> usize;
}
trait StoreMut<K, V>: Store<K, V> where K: Key {
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn remove<Q>(&mut self, key: &Q) -> Option<V> where ...;
}
```
</details>

---

### 1. Setup Guide for GitHub Actions

To set up a GitHub Actions workflow that builds your Zensical site from a `./src_docs` directory and deploys it to a `./docs` folder for hosting (e.g., with GitHub Pages), follow these steps.

**1. Configure Your Zensical Project**

First, tell Zensical where to find your source files and where to put the built site. In your `zensical.toml` file, add or modify the `docs_dir` and `site_dir` settings:

```toml
# file: zensical.toml

[project]
site_name = "My Project"
docs_dir = "src_docs"  # Source directory for Markdown files
site_dir = "docs"      # Output directory for the built site
```

**2. Create the GitHub Actions Workflow**

In your repository, create a file at `.github/workflows/build-docs.yml` and add the following content:

```yaml
# file: .github/workflows/build-docs.yml

name: Build and Deploy Docs

on:
  push:
    branches:
      - main  # Or whichever branch you want to trigger the build

jobs:
  build:
    name: Build Zensical Site
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Zensical
        run: pip install zensical

      - name: Build site
        # The --clean flag ensures a fresh build without using old cache
        run: zensical build --clean

      - name: Commit and push built site
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add docs
          # Commit only if there are changes in the docs directory
          git diff-index --quiet HEAD -- docs/ || git commit -m "docs: build site"
          git push
```

This workflow automates the following process:
*   It triggers on every push to the `main` branch.
*   It checks out your code.
*   It installs Python and the Zensical package.
*   It runs the `zensical build` command, which reads from `./src_docs` and writes to `./docs` as configured.
*   It commits the newly built `./docs` directory back to your repository.

If you are using GitHub Pages, you can now configure it to serve from the `/docs` folder on your `main` branch.

---

### 2. The Zensical UI System

Zensical's UI is a sophisticated, component-based system built with modern web technologies. It's designed for performance, customizability, and a smooth user experience, behaving much like a single-page application (SPA).

#### **How the UI Components Work**

The UI is primarily built with **TypeScript** and **RxJS**, a library for reactive programming using observables. Instead of traditional event listeners, the application logic is organized into streams of events.

*   **Core Logic**: The main entry point is `zensical/ui/src/assets/javascripts/bundle.ts`. This file initializes and subscribes to several global observables for application state, such as `document$`, `location$`, `viewport$`, and `keyboard$`.
*   **Component Mounting**: Components are TypeScript modules that "mount" onto specific HTML elements identified by `data-md-component` attributes. For example, `mountHeader` is responsible for the header's logic, including the autohide feature.
*   **Reactivity with RxJS**: Components react to changes in application state through observables. For instance, the sidebar's height and lock state are determined by an observable that combines data from the main content area (`main$`) and the viewport (`viewport$`). This reactive model allows for complex interactions to be handled in a declarative and efficient way.

#### **CSS and HTML Infrastructure**

*   **HTML Structure**: The HTML is generated from [MiniJinja templates](https://docs.rs/minijinja/latest/minijinja/), which are located in `zensical/ui/src/partials/`. The base structure is defined in `base.html`, and the default page is `main.html`. You can override any of these templates to customize the structure.
*   **Styling with SCSS**: Zensical uses SCSS for styling and provides two built-in themes: `modern` (the default) and `classic` (which mimics Material for MkDocs).
*   **Theming with CSS Variables**: Colors, fonts, and other stylistic properties are controlled via CSS variables (Custom Properties), like `--md-primary-fg-color` and `--md-code-bg-color`. This makes it easy to create custom color schemes or tweak existing ones by simply overriding these variables in an extra CSS file.

#### **Reusability in Other Projects**

While the UI components are tightly integrated with the Zensical ecosystem (especially RxJS and the specific HTML structure), it is possible to reuse parts of the UI.

*   **CSS/SCSS**: The SCSS files can be adapted for other projects. You would need to replicate the HTML structure that the styles expect (including the `data-md-component` attributes) and include the compiled CSS.
*   **TypeScript/JavaScript**: Reusing the JavaScript components directly would be challenging without also adopting the RxJS-based architecture. A more practical approach would be to study the logic within the component files (e.g., `zensical/ui/src/assets/javascripts/components/sidebar/index.ts`) and re-implement the desired functionality in your framework of choice.

#### **Combining with Tailwind CSS and Other Frameworks**

You can integrate utility-first CSS frameworks and UI component libraries into Zensical.

1.  **Tailwind CSS & daisyUI**:
    *   Set up Tailwind to process your CSS files as you normally would.
    *   Configure Tailwind to scan Zensical's output HTML for class names.
    *   Include the final generated CSS file in your `zensical.toml` using the `extra_css` option.
    *   You can then use Tailwind and daisyUI classes directly in your Markdown files by enabling the `attr_list` Markdown extension, which lets you add classes to elements:
        ```markdown
        [This is a daisyUI button](...){ .btn .btn-primary }
        ```

2.  **Tweakpane (Svelte) & Leva (React)**:
    *   These libraries are JavaScript-based. You would typically bundle them into a single JS file.
    *   Include this bundle using the `extra_javascript` option in `zensical.toml`.
    *   Write a small script that initializes your Svelte or React components. It's crucial to hook into Zensical's `document$` observable to ensure your components are re-initialized when pages are loaded via instant navigation:
        ```javascript
        document$.subscribe(function() {
          // Find the placeholder element and mount your React/Svelte app here
          const appRoot = document.getElementById("my-leva-app");
          if (appRoot) {
            // ... mount your component
          }
        })
        ```

---

### 3. Integration with PyScript/Pyodide

Integrating Zensical into a fully client-side site using PyScript or Pyodide is an advanced use case but conceptually feasible, though it comes with challenges. Zensical's core is a Rust binary compiled into a Python extension using PyO3 and Maturin, as indicated by the `pyproject.toml` and `Cargo.toml` files. This means you cannot simply `pip install zensical` within Pyodide.

Here's a breakdown of the required steps and challenges:

**1. Compile the Rust Core to WebAssembly (WASM)**

The native Python extension (`.so`, `.pyd`) created by Maturin cannot run in a browser. The entire Rust core of Zensical (`zensical/zensical/crates/zensical`) would need to be compiled to a WASM target. This is a significant undertaking that involves:
*   Adapting the Rust code to be compatible with the `wasm32-unknown-unknown` target.
*   Handling parts of the code that interact with the filesystem (like reading `.md` files and writing `.html` files) by replacing them with calls to an in-memory virtual filesystem provided by Pyodide.

**2. Load the Components into Pyodide**

Once you have a WASM version of the Rust core, you would load it into your Pyodide application alongside the Python parts of Zensical.

```python
# In your PyScript/Pyodide application
import micropip

# Install Zensical's Python-only dependencies
await micropip.install([
    "click", "deepmerge", "markdown", "pygments", "pymdown-extensions", "pyyaml"
])

# Load the Zensical Python files and the compiled WASM module
pyodide.FS.mount_vfs("my_project_files", {
    "/lib/python3.11/site-packages/zensical": {
        "__init__.py": "...",
        "main.py": "...",
        # ... other python files
        "zensical.wasm": wasm_binary_data
    }
})
```

**3. Run the Build Process in the Browser**

With everything loaded, you would use Pyodide's virtual filesystem to simulate a real project structure and then invoke the build process.

```python
# Create a virtual file system for your docs
pyodide.FS.mkdir("/my_docs")
pyodide.FS.writeFile("/my_docs/zensical.toml", "...")
pyodide.FS.mkdir("/my_docs/src_docs")
pyodide.FS.writeFile("/my_docs/src_docs/index.md", "# Hello World")

# Import and run the build function
from zensical.main import execute_build
execute_build(config_file="/my_docs/zensical.toml", clean=True)

# Read the output from the virtual file system
html_output = pyodide.FS.readFile("/my_docs/docs/index.html", {"encoding": "utf8"})

# Display the output
# You could write this to an iframe or handle it in another way
display(html_output, target="output-div", append=False)
```

**Challenges and Limitations:**

*   **Compilation:** The primary hurdle is compiling the Rust core to WASM and ensuring it works correctly with Pyodide's environment.
*   **Performance:** Running a full static site generation process within the browser will be resource-intensive and may be slow, especially for larger sites.
*   **"Serve" Command:** The `zensical serve` command, which relies on a native networking stack (`mio`), would not work. You would need to implement your own previewing mechanism.

In summary, while it is theoretically possible, running Zensical entirely client-side would require a significant porting effort of its Rust core to WebAssembly.


------------------------------------------------------------


Here’s a first, self-contained pass at the doc you asked for. I’ll treat “zensical” as the new engine that replaces MkDocs+Material, with three main pieces:

* **Rust core** (`crates/zensical`, `zensical-serve`, `zensical-watch`, `zrx*` infra)
* **Python CLI + authoring glue** (`python/zensical`)
* **UI library** (`ui`) plus **example docs project** (`docs`)

---

## 1. TL;DR of the actual code

### 1.1 High-level project layout

```text
web/
  zensical/          # Rust + Python tool, plus bootstrap & scripts
  ui/                # TypeScript/SCSS frontend for the docs UI (Material-like)
  docs/              # Zensical’s own documentation, currently built via mkdocs.yml
  zrx/               # Reactive execution + graph infra used by zensical-serve/watch
llms.sh              # helper to run an external tool on HTML files
```



#### 1.1.1 Docs project (`zensical/docs`)

* `docs/` — Markdown docs, assets, includes, overrides, worker
* `mkdocs.yml` — **current** configuration using `mkdocs-material`
* `.github/workflows/deploy.yml` — GitHub Actions: install `zensical` (Python), build, then deploy to Cloudflare via Wrangler
* `worker/index.ts`, `wrangler.toml` — Cloudflare Worker to serve the built site with redirects, 404 handling & caching

Key worker logic:

```ts
export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.hostname === "www.zensical.org") {
      url.hostname = "zensical.org"
      return Response.redirect(url.toString(), 301)
    }

    const pathname = url.pathname.replace(/^\/docs/, '')
    const res = await env.ASSETS.fetch(`${url.origin}${pathname || "/"}`)

    if (res.status === 404) {
      const notfound = await env.ASSETS.fetch(`${url.origin}/404`, req)
      if (notfound.ok) {
        return new Response(await notfound.text(), {
          status: 404,
          headers: { "Content-Type": "text/html" },
        });
      }
    } else {
      const type = res.headers.get("Content-Type") || "";
      if (type.includes("text/html")) {
        const url = new URL(req.url);
        if (!url.pathname.endsWith("/")) {
          url.pathname += "/";
          return Response.redirect(url, 301);
        }
      }
      if (type.includes("text/css") || type.includes("text/javascript") ||
          type.includes("application/json") || type.includes("image/")) {
        const response = new Response(res.body, res);
        response.headers.set(
          "Cache-Control",
          "public, max-age=3600, stale-while-revalidate=86400"
        )
        return response
      }
    }
    return res
  }
}
```



#### 1.1.2 UI project (`zensical/ui`)

TypeScript + SCSS “theme engine” for docs. It’s conceptually the replacement for Material’s `material/` theme.

Key directories:

* `src/assets/javascripts`

  * `_/index.ts` — global config + feature flags + translations API
  * `browser/*` — DOM utilities, observables with RxJS:

    * `browser/document/index.ts` — `watchDocument() → ReplaySubject<Document>`
    * `browser/element/_` — `getElement`, `getElements`, `getActiveElement`
    * `browser/element/focus` — `watchElementFocus(el) → Observable<boolean>`
    * `browser/element/hover` — `watchElementHover(el, timeout?) → Observable<boolean>`
    * `browser/element/offset` — offset + scroll offset watchers
    * `browser/element/size` — size observer (via ResizeObserver, with polyfill)
* `src/assets/stylesheets` — `modern` & `classic` theme variants, with SCSS fragments mirroring Material’s structure (components, extensions, palette, utilities).
* `src/assets/javascripts/bundle.ts` — entry-point bundling all modules.
* `typings/` — `.d.ts` for DOM additions, mermaid, google fonts.

Example: global config API (`_/index.ts`):

```ts
export type Flag =
  | "announce.dismiss"
  | "content.code.annotate"
  | "content.code.copy"
  | "content.code.select"
  | "content.footnote.tooltips"
  | "content.lazy"
  | "content.tabs.link"
  | "content.tooltips"
  | "header.autohide"
  | "navigation.expand"
  | "navigation.indexes"
  | "navigation.instant"
  | "navigation.instant.prefetch"
  | "navigation.instant.progress"
  | "navigation.instant.preview"
  | "navigation.sections"
  | "navigation.tabs"
  | "navigation.tabs.sticky"
  | "navigation.top"
  | "navigation.tracking"
  | "search.highlight"
  | "search.share"
  | "search.suggest"
  | "toc.follow"
  | "toc.integrate"

export interface Config {
  base: string
  features: Flag[]
  translations: Translations
  search: string
  annotate?: Record<string, string[]>
  tags?: Record<string, string>
  version?: Versioning
}

const script = getElement("#__config")
const config: Config = JSON.parse(script.textContent!)
config.base = `${new URL(config.base, getLocation())}`

export function configuration(): Config { return config }
export function feature(flag: Flag): boolean { return config.features.includes(flag) }
export function translation(key: Translation, value?: string | number): string { ... }
```



#### 1.1.3 Core / engine (`zensical/crates` and `python/zensical`)

Rust crate `crates/zensical` (library) – major modules:

```text
config/
  error.rs          # Config error types
  extra.rs          # [project.extra] mapping & helpers
  mdx.rs            # Markdown/MDX config
  plugins.rs        # Plugin/module configuration
  project.rs        # Top-level project config (docs_dir, site_dir, theme, etc.)
  theme.rs          # Theme config (palette, fonts, icons, features)
server/
  client.rs         # Preview/serve mode helpers (hot reload, etc.)
structure/
  dynamic/float.rs  # Dynamic “floating” elements
  nav/item.rs       # Navigation item model
  nav/iter.rs       # Navigation iteration
  nav/meta.rs       # Nav metadata
  search/item.rs    # Search entry
  dynamic.rs        # Dynamic content parts
  markdown.rs       # Parsed markdown structure
  nav.rs            # Full nav tree
  page.rs           # Page model
  search.rs         # Search index model
  tag.rs            # Tags model
  toc.rs            # TOC tree per page
template/
  filter.rs         # Template filters for MiniJinja
  loader.rs         # Template loader (theme + overrides)
workflow/
  cached.rs         # Cached build workflow
config.rs           # Config loading entrypoint
lib.rs              # Top-level Rust API
server.rs           # Server (dev/preview)
structure.rs        # Re-exports of structure
template.rs         # Template system wiring
watcher.rs          # FS watcher
workflow.rs         # Build workflows
```



Server + watcher crates:

* `crates/zensical-serve`

  * `handler/` — HTTP routing abstraction with matcher, stack, middleware.
  * `http/` — request/response types, header parsing, URI parsing.
  * `middleware/` — static files, websockets, path normalization.
  * `router/` — route registration and dispatch.
  * `server/` — TCP listener, connection handling, poller.
  * `lib.rs`, `server.rs` — library entry and server bootstrap.

* `crates/zensical-watch`

  * `agent/` — FS change agent, event handling & monitor.
  * `agent.rs`, `lib.rs` — watchers as a service.

Reactive infra (`web/zrx`):

* `zrx`, `zrx-diagnostic`, `zrx-graph`, `zrx-id`, `zrx-path`, `zrx-scheduler`, `zrx-store`, `zrx-stream`
* Provide:

  * IDs and selectors (`zrx-id`)
  * Graph modeling (`zrx-graph`)
  * Execution scheduling (`zrx-scheduler`, `zrx-executor`)
  * Streaming combinators (`zrx-stream`)
  * Data store with delta behavior (`zrx-store`)

Python CLI (`zensical/python/zensical`):

```text
python/zensical/
  bootstrap/
    .github/workflows/docs.yml  # template GH Actions
    docs/index.md               # template docs
    docs/markdown.md
    zensical.toml               # starter config
  extensions/
    emoji.py                    # emoji/ icon helpers
    links.py                    # link handling in markdown
    preview.py                  # instant preview support
    search.py                   # search index building
    utilities/filter.py         # filters for templating
  __main__.py                   # `python -m zensical`
  main.py                       # core CLI dispatch
  markdown.py                   # markdown pipeline
  config.py                     # Python-side config loader & conversion
  zensical.pyi                  # type hints for consumers
```

Python CLI approximated signatures:

```py
# main.py
def main(argv: list[str] | None = None) -> int: ...
def cmd_new(args: Namespace) -> int: ...
def cmd_build(args: Namespace) -> int: ...
def cmd_serve(args: Namespace) -> int: ...
def load_config(path: Path | None = None) -> ProjectConfig: ...

# markdown.py
def build_markdown_tree(project: ProjectConfig) -> list[Page]: ...
def render_markdown(page: Page, config: ProjectConfig) -> str: ...

# config.py
@dataclass
class ProjectConfig:
    site_name: str
    docs_dir: Path
    site_dir: Path
    theme: ThemeConfig
    markdown_extensions: dict[str, Any]
    extra: dict[str, Any]
    # ...

def load_project_config(path: Path) -> ProjectConfig: ...
def from_mkdocs_yml(path: Path) -> ProjectConfig: ...
```

(Interfaces inferred from docs + file names, not exact Rust/Python code.)

---

## 2. Setup guide: “zensical” as replacement for MkDocs+Material

Target you described:

> Repo with `./src_docs` (Markdown sources) → `./docs` (built static site), built via GitHub Actions.

I’ll outline **concrete steps** to wire that using the existing tooling.

### 2.1 Minimal project layout you want

In your real repo (say `my-project/`):

```text
my-project/
  src_docs/           # your Markdown sources
  docs/               # built output (GitHub Pages / Cloudflare / whatever)
  zensical.toml       # config for Zensical
  .github/workflows/docs.yml
```

### 2.2 Example `zensical.toml` for this layout

```toml
[project]
site_name = "My Project"
site_url  = "https://example.com/docs/"   # adjust for your host
docs_dir  = "src_docs"
site_dir  = "docs"

# Enable directory URLs & nice nav
use_directory_urls = true

# Basic theme setup (modern variant)
[project.theme]
variant   = "modern"
language  = "en"

[project.theme.palette]
scheme  = "default"
primary = "indigo"
accent  = "indigo"

# Enable key features mirrored from mkdocs-material
[project.theme]
features = [
  "announce.dismiss",
  "content.code.annotate",
  "content.code.copy",
  "content.code.select",
  "content.footnote.tooltips",
  "content.tabs.link",
  "content.tooltips",
  "navigation.footer",
  "navigation.indexes",
  "navigation.path",
  "navigation.sections",
  "navigation.tabs",
  "navigation.top",
  "search.highlight",
]

# (Optionally) Explicit nav – or let it infer from filesystem
# [project]
# nav = [
#   { "Home" = "index.md" },
#   { "Guide" = [
#       "guide/index.md",
#       "guide/install.md",
#       "guide/usage.md",
#   ]},
# ]

# Markdown extensions: copy the “default config” from docs, adapted:
[project.markdown_extensions.abbr]
[project.markdown_extensions.admonition]
[project.markdown_extensions.attr_list]
[project.markdown_extensions.def_list]
[project.markdown_extensions.footnotes]
[project.markdown_extensions.md_in_html]
[project.markdown_extensions.toc]
permalink = true

[project.markdown_extensions.pymdownx.arithmatex]
generic = true
[project.markdown_extensions.pymdownx.betterem]
smart_enable = "all"
[project.markdown_extensions.pymdownx.caret]
[project.markdown_extensions.pymdownx.details]
[project.markdown_extensions.pymdownx.emoji]
emoji_generator = "zensical.extensions.emoji.to_svg"
emoji_index     = "zensical.extensions.emoji.twemoji"
[project.markdown_extensions.pymdownx.highlight]
anchor_linenums     = true
line_spans          = "__span"
pygments_lang_class = true
[project.markdown_extensions.pymdownx.inlinehilite]
[project.markdown_extensions.pymdownx.keys]
[project.markdown_extensions.pymdownx.mark]
[project.markdown_extensions.pymdownx.smartsymbols]
[project.markdown_extensions.pymdownx.snippets]
[project.markdown_extensions.pymdownx.superfences]
[project.markdown_extensions.pymdownx.tabbed]
alternate_style      = true
combine_header_slug  = true
[project.markdown_extensions.pymdownx.tasklist]
custom_checkbox = true
[project.markdown_extensions.pymdownx.tilde]

# Tags support, if you want tag icons etc
[project.extra.tags]
default = "default"
```

This mirrors the configuration present in the real `mkdocs.yml`, but expressed as TOML in Zensical’s native format. 

### 2.3 GitHub Actions workflow for `./docs` output

Use the existing `zensical/docs/.github/workflows/deploy.yml` as a template.  Adapted for your repo:

```yaml
name: Docs

on:
  push:
    branches:
      - main
      - master
  workflow_dispatch:

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.x"

      - name: Install Zensical
        run: pip install zensical

      - name: Build docs
        run: zensical build --clean

      # For GitHub Pages (with Pages v4):
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v4
        with:
          path: docs

  deploy:
    needs: build-docs
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

Key changes from the Zensical docs workflow:

* `site_dir = "docs"` in `zensical.toml` → `upload-pages-artifact` path is `docs/`.
* No Cloudflare `wrangler` step if you don’t need it; or keep it if you do.

### 2.4 Local workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install zensical

# one-time new project (will not overwrite if things exist)
zensical new .

# live preview
zensical serve

# build static site (into ./docs)
zensical build --clean
```

---

## 3. UI components & CSS/HTML infrastructure, and using them elsewhere

### 3.1 UI architecture

The UI is a **MiniJinja + TypeScript + SCSS** system – structurally almost identical to mkdocs-material:

* HTML structure is defined by templates (`base.html`, `main.html`, `partials/…`) – see `docs/docs/customization.md` for a full description. 
* Styling is SCSS, compiled into `main.css` for the `modern` & `classic` variants (`stylesheets/modern/main`, `stylesheets/classic/main`). 
* Behaviour is TypeScript → JS modules in `ui/src/assets/javascripts`, exported via a single `bundle.js` plus web worker for search.

Components are mostly **“semantic CSS” + small TS modules**:

* Content components: admonitions, tabs, code blocks, tooltips, tables, diagrams, etc. See `docs/docs/authoring/…` (admonitions, content-tabs, code-blocks, tooltips, data-tables, diagrams, grids). 
* Nav components: header, tabs, sidebar nav, table of contents, progress bar. 
* Search component: search dialog, suggestions, results list, highlight overlay, search worker. 

The TS side is mostly **RxJS-based observables** wired to DOM:

* `watchDocument()` emits when the DOM is ready.
* `watchElementFocus(el)` gives you a `boolean` stream.
* `watchElementHover(el, timeout?)` gives “hovered?” with optional decay.
* `watchElementOffset*`, `watchElementSize` feed scroll/resize logic for sticky header, “back to top”, tooltips, etc.

### 3.2 Using the UI in other projects

At the end of the day, the UI bundle is:

* `main.css` (modern or classic)
* `bundle.js`
* Some `__config` `<script>` tag in the HTML head with serialized config.
* Search web worker script.

So in a **non-Zensical project**, you can:

1. Copy the compiled CSS/JS assets (from `ui/dist` once built) into your app.
2. Embed a minimal `__config` script modeled after what Zensical generates (see `docs/docs/customization.md` “Extending the theme” and `setup/basics.md`). 
3. Use the HTML structure required by the JS (header/container/nav/aside/main etc). The docs’ examples show the expected DOM.

Concretely, for a custom site:

```html
<head>
  <link rel="stylesheet" href="/static/zensical-main.css" />
  <script id="__config" type="application/json">
    {
      "base": "/",
      "features": [
        "navigation.tabs",
        "navigation.sections",
        "content.code.copy"
      ],
      "translations": {
        "clipboard.copy": "Copy to clipboard",
        "clipboard.copied": "Copied",
        ...
      },
      "search": "/static/search-worker.js"
    }
  </script>
  <script src="/static/zensical-bundle.js" defer></script>
</head>
<body>
  <!-- header / nav / main structure like in generated docs -->
</body>
```

The JS will auto-initialize based on `__config` and the DOM.

### 3.3 Combining with Tailwind CSS, daisyUI, Tweakpane, Leva

The theme is **fairly self-contained**: it relies on CSS variables and BEMish selectors, and doesn’t try to own the entire document.

Patterns to integrate:

#### Tailwind

* Load **Tailwind** after Zensical CSS, but **scope** your Tailwind utilities to your app container:

  ```html
  <div class="my-app tailwind">
    <!-- your Tailwind components -->
  </div>
  ```

* In Tailwind config, set `prefix: "tw-"` to avoid class collisions (`.tw-flex`, `.tw-grid`, etc).

* Let Zensical render docs; let your Tailwind components sit inside pages (e.g. embedding interactive demos).

#### daisyUI (Tailwind plugin)

* Same as Tailwind, but:

  * Keep daisyUI themes separate from the Zensical color palette. You can even map Zensical CSS variables into Daisy theme definitions.
* This is ideal for “component gallery” pages: the doc page uses Zensical; inside, you mount a `<section class="tw-prose daisy">...</section>` with daisy components.

#### Tweakpane (Svelte) & Leva (React)

These two are **JS-only control panels**. The friction point is bundling, not CSS:

* Use them inside **client-side islands**:

  * **Svelte + Tweakpane**:

    * Build a small Svelte app as a separate bundle.
    * On a Zensical page, add a `<div id="tweakpane-demo"></div>` and a `data-*` attribute to identify it.
    * In your Svelte entry, mount into that div on DOM ready.

  * **React + Leva**:

    * Same idea: a React island with Leva controls mounted into a `<div id="leva-demo"></div>`.
    * Zensical JS doesn’t care; it just operates on its own selectors.

* Styling: both Tweakpane and Leva come with their own CSS; import them in your respective bundles. They will happily live alongside Zensical because they use scoped classnames.

Workflow idea:

* Keep `ui` as the “core docs chrome”.
* Your Svelte/React micro-apps just need an HTML anchor inside the Markdown:

  ```markdown
  ## Interactive demo

  <div id="demo-app" data-demo="my-demo"></div>
  ```

  Then hook from JS.

---

## 4. Integrating with a fully client-side PyScript / PyOdide site

Goal: **pure client HTML** served by static hosting, that:

* Uses Zensical’s UI (CSS/JS),
* Runs Python code in the browser via PyScript/PyOdide,
* Optionally uses Zensical search/nav, but data (pages) may be generated or loaded dynamically.

### 4.1 Basic structure

You can treat Zensical’s assets as **a UI skin**, and let PyScript render content into `main`.

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>My PyScript Docs</title>

    <!-- Zensical UI -->
    <link rel="stylesheet" href="zensical-main.css" />
    <script id="__config" type="application/json">
      {
        "base": "/",
        "features": ["navigation.tabs", "navigation.top"],
        "translations": { ... },
        "search": "search-worker.js"
      }
    </script>
    <script src="zensical-bundle.js" defer></script>

    <!-- PyScript / PyOdide -->
    <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
    <script defer src="https://pyscript.net/latest/pyscript.js"></script>
  </head>
  <body>
    <header class="md-header">...</header>
    <div class="md-container">
      <aside class="md-sidebar md-sidebar--primary">...</aside>
      <main class="md-main">
        <div class="md-main__inner md-grid">
          <div class="md-content">
            <article class="md-content__inner">
              <!-- PyScript content here -->
              <py-script>
from js import document
main = document.querySelector(".md-content__inner")

# Generate HTML or manipulate DOM here
main.insertAdjacentHTML("beforeend", "<h1>Hello from PyScript</h1>")
              </py-script>
            </article>
          </div>
        </div>
      </main>
    </div>
  </body>
</html>
```

Zensical JS will handle:

* Header/nav interactions,
* Search UI (provided you also build a search index JSON),
* Tooltips, tabs, code copy, etc.

Your PyScript code is free to:

* Inject HTML into `.md-content__inner`,
* Or load pre-rendered HTML snippets (e.g. `fetch("pages/intro.html")` via JS).

### 4.2 Using Zensical’s markdown pipeline with PyScript

If you want **Markdown → HTML** on the client purely in PyScript:

1. Pre-package a pure-Python Markdown implementation + pymdownx into the PyScript environment (size trade-off).

2. In a `<py-script>` tag:

   ```python
   import js
   from markdown import markdown

   src = """
   # Hello

   !!! note
       This is a note.
   """

   html = markdown(src, extensions=[
       "admonition",
       "pymdownx.tabbed",
       # ...
   ])

   container = js.document.querySelector(".md-content__inner")
   container.innerHTML = html
   ```

3. Zensical JS will then run over the resulting DOM and wire up:

   * Admonitions, tabs, grids, tooltips, etc.
   * Code blocks with copy/select
   * Search highlighting (if you came from a search link)

This gives you a fully client-side stack:

* **HTML shell + Zensical assets** = static.
* **Content** = Markdown strings inside PyScript or loaded dynamically.
* **Rendering** = Python in the browser.
* **UI/interaction** = Zensical UI bundle.

### 4.3 Hybrid: Zensical builds + PyScript islands

Probably the most pragmatic pattern:

* Use `zensical build` on CI to generate the static doc site (like now).
* Embed PyScript widgets in specific pages for interactive demos / tools.

Example in a Markdown doc:

```markdown
## Vector playground

<py-script id="glyph-demo">
from js import document
container = document.querySelector("#glyph-demo")
container.innerHTML = "<p>Here we could visualize glyph curves...</p>"
</py-script>
```

Zensical renders the rest of the page, PyScript mounts where needed.

---

If you want, next iteration I can:

* Draft a **specific `zensical.toml`** and **`docs.yml`** for one of your actual repos (FontLab docs / Vexy Lines mini-site),
* Sketch a minimal **React+Leva island** and a **Svelte+Tweakpane island** that sit inside a Zensical-built page,
* Or design a “Zensical-compatible” HTML skeleton for a pure PyScript app so you can slap the UI on anything.
