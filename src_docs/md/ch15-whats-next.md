# Chapter 15: What's next

*Where LLMs are headed and what to watch.*

> "The best way to predict the future is to invent it."
> — Alan Kay

## The prediction problem

Every AI prediction ages badly. In 2020, GPT-3 seemed like magic. By 2023, it was a commodity. In early 2024, GPT-4 was the undisputed champion. By late 2024, Claude 3.5 Sonnet had taken the coding crown.

Predicting 2026 from late 2025 is a fool's errand. This chapter won't try. Instead, we'll identify trends with momentum and technologies worth watching—hedged bets, not prophecies.

> **2025 Update**: This chapter has aged well. Llama 4 launched and is competitive. Claude 4 and GPT-5 arrived with significant capability gains. OpenAI's Operator and Codex agents are now production tools. Context windows reached 2M tokens (Gemini). Multimodality is standard—Sora-class video generation works. The commodity prediction came true: most tasks work fine with cheaper models. The anti-trends persist: hallucinations remain, regulation arrived (EU AI Act enforcement began), and energy concerns intensified. The advice stands: build portable systems, stay skeptical, focus on applications.

## Imagine...

Imagine trying to predict the internet in 1996. You could see the potential—email was useful, websites were multiplying, e-commerce was emerging. But predicting Google, social media, or smartphones? Impossible. The technology existed, but the applications hadn't been invented yet.

We're in 1996 for AI. The transformer architecture is our TCP/IP—fundamental infrastructure that everything else will build upon. Current applications (chatbots, code assistants, image generators) are our email and static websites—useful, but primitive compared to what's coming.

Nobody knows what the Google of AI will be. Or the Facebook. Or the iPhone. The patterns aren't clear yet because the inventors are still tinkering in garages. What we can do is watch the infrastructure—the protocols, the tools, the capabilities—and bet on directions rather than destinations.

The wisest stance: stay curious, stay portable, and build on solid foundations. The landscape will shift beneath us regardless. Might as well learn to surf.

---

## Trend 1: commoditization accelerates

### What's happening

The performance gap between frontier models is shrinking. GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro are interchangeable for most tasks. Open-source models (Llama, Qwen, Mistral) now match last year's commercial leaders.

### What it means

- **Price pressure**: Margins compress, prices fall
- **Differentiation shifts**: Speed, tooling, and ecosystem matter more than raw capability
- **Build portability**: Don't lock into one provider

### What to watch

- **Llama 4**: Meta's next generation. If quality matches Claude/GPT, everything changes.
- **Qwen 3**: Alibaba's models are underestimated in the West. Watch for parity.
- **Open-source tooling**: vLLM, TGI, and other inference servers mature.

## Trend 2: small models, big impact

### What's happening

The industry is splitting: massive models for frontier capabilities, tiny models for deployment. Phi-3, Gemma 2, and Llama 3.2 prove that 3-7B parameter models can handle many production tasks.

### What it means

- **Edge deployment**: Real AI on phones and laptops, not just API calls
- **Cost collapse**: 100x cheaper inference for most use cases
- **Latency wins**: Local models respond in milliseconds

### What to watch

- **Apple Intelligence**: Apple's on-device models will set consumer expectations
- **Quantization advances**: 4-bit models approaching 16-bit quality
- **Speculative decoding**: Small models accelerating large model inference

## Trend 3: agents go production

### What's happening

2024 was the year of agent demos. 2025 is the year of agent deployments. Browser automation (Stagehand), coding assistants (Cursor, Claude Code), and research agents (Deep Research) moved from impressive to useful.

### What it means

- **Reliability over capability**: Production agents need error handling, not more features
- **Observability becomes critical**: You need to see what agents are doing
- **Cost management**: Agentic loops can spiral expensively

### What to watch

- **Anthropic's computer use**: General-purpose GUI agents are coming
- **OpenAI's Operator**: Rumored agent product for automated workflows
- **Standardization**: MCP or something like it becoming the tool protocol

## Trend 4: multimodality matures

### What's happening

Vision, audio, and video understanding are becoming default capabilities, not premium features. GPT-4o, Gemini, and Claude all handle images natively. Real-time voice is here.

### What it means

- **Document understanding**: PDFs, screenshots, diagrams become first-class inputs
- **Video as data**: Security, manufacturing, and media analysis unlock
- **Voice-first interfaces**: Beyond chatbots to genuine voice assistants

### What to watch

- **Native video models**: Sora-class generation, but also understanding
- **Real-time audio**: Sub-100ms voice conversations
- **Embodied AI**: Robotics finally meeting LLM reasoning

## Trend 5: context windows keep growing

### What's happening

Gemini's 1M token context was the headline of early 2024. By late 2025, it's standard. Long context enables workflows impossible before.

### What it means

- **RAG simplification**: Some retrieval becomes unnecessary
- **Codebase-scale reasoning**: Entire repos in context
- **Document processing**: Books, legal filings, research papers in one shot

### What to watch

- **10M+ context**: Where's the ceiling?
- **Context utilization**: Models struggle to use long context effectively (the "lost in the middle" problem). Better attention architectures needed.
- **Memory vs context**: Persistent memory across sessions

## Trend 6: inference gets creative

### What's happening

Chain-of-thought, tree-of-thought, and extended thinking (like Claude's) show that inference-time compute matters. Spending more time "thinking" improves results.

### What it means

- **Quality/cost tradeoff**: Pay more for better reasoning
- **New interfaces**: "Extended thinking" modes become standard
- **Hybrid approaches**: Fast responses for simple queries, deep reasoning for hard ones

### What to watch

- **OpenAI o1**: Extended reasoning as a product
- **Test-time scaling**: Mathematical foundations emerging
- **Constitutional AI**: Self-critique and revision at inference time

## Trend 7: specialization

### What's happening

General-purpose models are good at everything, great at nothing. Specialized models (code, medical, legal) outperform on specific domains.

### What it means

- **Domain experts**: Finance, healthcare, and legal get dedicated models
- **Vertical integration**: Industry-specific training data becomes valuable
- **Model portfolios**: Teams run multiple specialized models

### What to watch

- **Code models**: Codex, StarCoder, and domain-specific coding assistants
- **Scientific models**: AlphaFold-style breakthroughs in other domains
- **Enterprise fine-tuning**: Custom models on proprietary data

## The anti-trends

Not everything gets better:

### Hallucinations persist

Despite improvements, models still confidently state falsehoods. Retrieval augmentation helps. Training improvements help. The problem doesn't disappear.

**Implication**: Always verify. Never trust blindly. Build fact-checking into workflows.

### Regulation arrives

EU AI Act, US executive orders, state-level laws. The regulatory environment is hardening.

**Implication**: Compliance becomes a feature. Audit trails matter. Opt-out mechanisms required.

### Energy concerns

Training and inference require enormous compute. Data center power demands strain grids.

**Implication**: Efficiency becomes a selling point. On-device models get environmental credibility.

## What to build in 2026

If I were starting a project today:

### Safe bets

1. **RAG applications**: Knowledge retrieval isn't going away. Build the plumbing.

2. **Developer tools**: Coding assistance is addictive. Build specialized tools for niche languages or workflows.

3. **Document processing**: Turn unstructured documents into structured data. Boring but valuable.

4. **Internal assistants**: Help employees navigate corporate knowledge bases. Low risk, high impact.

### Risky bets

1. **Autonomous agents**: The potential is enormous. The reliability isn't there yet. High risk, high reward.

2. **Consumer products**: Hard to compete with OpenAI and Google on general chat. Find a niche.

3. **Model training**: Unless you're Meta-scale, focus on fine-tuning, not pre-training.

### Don't build

1. **Another chatbot**: The market is saturated. ChatGPT and Claude exist.

2. **AI wrappers without moats**: If you're just calling GPT-4 and adding a UI, you're a feature, not a product.

3. **Speculative AGI applications**: Build for today's capabilities, not tomorrow's hype.

## The skills that matter

What to learn:

### Technical skills

- **Prompt engineering**: Still underrated. Great prompts beat mediocre fine-tunes.
- **Evaluation**: Measuring model quality is harder than improving it.
- **MLOps**: Deploying models at scale requires engineering discipline.
- **RAG architecture**: The plumbing of practical AI applications.

### Strategic skills

- **Cost modeling**: LLM economics are unintuitive. Learn to forecast.
- **Use case identification**: Most AI projects fail from wrong problem selection, not technical failure.
- **Risk management**: When to use AI, when not to, and how to fail gracefully.

## The honest assessment

Here's what we know:

**LLMs are transformative.** They've changed software development, content creation, and knowledge work. The impact is real and growing.

**LLMs are overhyped.** AGI is not imminent. Most enterprise AI projects still fail. The gap between demos and production is vast.

**The tools are good enough.** You can build useful things today with commodity models and open-source frameworks. You don't need the next breakthrough.

**The fundamentals win.** Clear problem definition, good data, solid engineering—these matter more than model choice.

## A personal take

I've spent hundreds of hours with these tools. Written thousands of prompts. Built production systems. Here's what I believe:

**The best use of LLMs is augmentation, not replacement.** Human judgment plus AI capability beats either alone.

**The moat is in the application.** Base models are commodities. Value comes from how you integrate them into workflows.

**Speed matters more than perfection.** Ship, learn, iterate. The landscape changes too fast for waterfall development.

**Stay skeptical.** The AI industry is full of hype. Test claims. Measure results. Trust data over demos.

## Getting started: two approaches

**Simple approach**: Build with today's stable tools. Don't chase every new release.

```python
# future_proof.py - Stack that'll work for the next 2 years

# 1. Use an abstraction layer (LiteLLM) so you can swap providers
from litellm import completion

# 2. Stick with proven patterns (RAG, structured output)
from pydantic import BaseModel
from qdrant_client import QdrantClient

# 3. Keep it simple
def query_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Provider-agnostic LLM call."""
    response = completion(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# This code will work regardless of which model is best next year.
# The abstraction layer lets you swap in Llama 4, Claude 5, or whatever's next.
```

The simple approach: pick stable tools, use abstraction layers, ignore 90% of AI news. Your code stays working while others chase every new release.

**Production approach**: Build a learning system that adapts as the field evolves.

```python
# adaptive_stack.py - Architecture that evolves with the field

from dataclasses import dataclass
from typing import Protocol

class LLMProvider(Protocol):
    """Abstract interface - swap implementations as providers change."""
    def complete(self, prompt: str) -> str: ...
    def embed(self, text: str) -> list[float]: ...

@dataclass
class ModelRegistry:
    """Track model capabilities so you can route intelligently."""
    models: dict[str, dict] = None

    def __post_init__(self):
        self.models = {
            # Update this quarterly as models change
            "gpt-4o-mini": {"cost": 0.0015, "quality": 0.85, "speed": "fast"},
            "claude-3-haiku": {"cost": 0.00075, "quality": 0.83, "speed": "fast"},
            "gpt-4o": {"cost": 0.025, "quality": 0.95, "speed": "medium"},
            "claude-3-5-sonnet": {"cost": 0.015, "quality": 0.94, "speed": "medium"},
        }

    def best_for_task(self, task: str) -> str:
        """Route to best model - update logic as field evolves."""
        if task in ["classify", "extract"]:
            return min(self.models, key=lambda m: self.models[m]["cost"])
        return max(self.models, key=lambda m: self.models[m]["quality"])

# Stay current workflow:
# 1. Quarterly: Update ModelRegistry with new models and pricing
# 2. Monthly: Test new models against your evals
# 3. Weekly: Read one good source (Simon Willison, Hugging Face blog)
# 4. Daily: Build and ship with stable tools
```

The production approach: abstract interfaces that let you swap providers, a registry you update quarterly, and a disciplined information diet. Stay informed without getting distracted.

Both approaches share a core insight: the field moves fast, but good architecture doesn't. Build portable systems, use abstraction layers, and update your model choices periodically rather than constantly. The best code you can write today is code that'll still work when Claude 6 and GPT-6 arrive.

## The closing

This book started with a detective story—a typographer hunting a font through a digital underworld. We've traveled through providers, packages, protocols, and patterns. Built systems that identify fonts, generate documentation, and manipulate glyphs.

The tools will change. The principles won't.

- **Understand your problem** before choosing technology
- **Start simple** and add complexity only when forced
- **Measure everything** because intuition fails at scale
- **Stay curious** because the best is yet to come

Language models are tools. Powerful tools, but tools nonetheless. What you build with them is up to you.

Now go make something useful.

![A sunrise over a city skyline, with streams of data flowing between buildings like light, hopeful futuristic style](https://pixy.vexy.art/)

---

*The End*

---

## Appendix: resources

### Official documentation

- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com)
- [Google AI Studio](https://aistudio.google.com)
- [MCP Specification](https://modelcontextprotocol.io)

### Essential libraries

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API
- [PydanticAI](https://github.com/pydantic/pydantic-ai) - Type-safe agents
- [LangChain](https://github.com/langchain-ai/langchain) - Orchestration framework
- [Instructor](https://github.com/jxnl/instructor) - Structured outputs

### Community resources

- [Hugging Face](https://huggingface.co) - Models and datasets
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) - Open-source community
- [Simon Willison's Blog](https://simonwillison.net) - LLM news and tools

### Books (besides this one)

- *Designing Machine Learning Systems* by Chip Huyen
- *Natural Language Processing with Transformers* by Tunstall, von Werra, Wolf
- *Build a Large Language Model (From Scratch)* by Sebastian Raschka

---

*Thank you for reading. Good luck.*
