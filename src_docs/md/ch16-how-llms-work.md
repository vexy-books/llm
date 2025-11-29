# Chapter 16: How LLMs actually work

> "The question is not whether machines think, but whether men do."
> — B.F. Skinner

*No calculus. No linear algebra. Just metaphors you can hold in your head while debugging at 2 AM.*

## The librarian who never sleeps

Imagine a library containing every book ever written in every language. Not just novels and textbooks—every email, every Reddit comment, every Wikipedia article, every line of code pushed to GitHub. Petabytes of human expression, captured and cataloged.

Now imagine the librarian has read it all. Every word. Multiple times. But here's the twist: this librarian has no memory of what any of it *means*. They don't know that Paris is a city or that water is wet. What they remember, with perfect precision, is which words tend to follow other words.

"The cat sat on the..." → Most likely "mat," "floor," "couch."
"Paris is the capital of..." → Almost certainly "France."
"def __init__(self," → Probably "name," "args," or "**kwargs."

That's an LLM. A statistical model of text patterns, trained on most of human written output, making predictions about what comes next.

## Imagine...

Imagine you've been locked in a room with a million books but no teacher. You can't ask questions. You can't look things up. All you can do is read, over and over, until patterns emerge from the noise.

After years of this, you notice something strange. Certain words cluster together. "Melancholy" appears near "sadness" and "grief" but not near "celebration." "Electron" hangs out with "proton," "atom," and "charge." Without anyone telling you, you've built a map of meaning—not through understanding, but through association.

Then someone opens the door and hands you a prompt: "Complete this sentence: The electron orbited the..."

You've never seen an electron. You don't know what orbiting is. But you've seen these words together thousands of times, and the pattern is clear: "nucleus."

You got it right without knowing anything. That's the uncanny valley where LLMs live.

---

## The telephone game at scale

Remember the telephone game from childhood? One person whispers a message, and by the time it reaches the last player, "Send reinforcements, we're going to advance" has become "Send three and fourpence, we're going to a dance."

Neural networks work like a telephone game in reverse. Instead of corrupting information, they extract patterns by passing data through layers of "players" (neurons), each one filtering and transforming the signal.

The first layer might notice basic patterns: vowels, consonants, punctuation.

The middle layers start grouping: words, phrases, syntax.

The final layers extract meaning: sentiment, intent, relationships between concepts.

Each layer builds on the previous one. Raw text in, probability distribution out. Simple components, emergent complexity.

## Tokens: the atoms of language

Before an LLM can process text, it breaks everything into tokens—fragments that the model can count and manipulate. Most tokens are partial words:

- "unhappiness" → ["un", "happiness"]
- "tokenization" → ["token", "ization"]
- "LLM" → ["LL", "M"]

English averages about 4 characters per token. A 100,000-token context window holds roughly 75,000 words—a short novel.

Why tokens instead of words or characters? Efficiency. With around 50,000 tokens in the vocabulary, the model can represent any text while keeping the math tractable. Rare words get split into common fragments. "Pneumonoultramicroscopicsilicovolcanoconiosis" becomes manageable.

When you hit a token limit, you're not running out of memory—you're running out of attention. The model can only consider so many tokens at once.

## Embeddings: GPS for meaning

Here's where it gets interesting. Each token becomes a vector—a list of numbers representing its position in a high-dimensional space. These numbers aren't random; they encode relationships.

Think of embeddings as GPS coordinates, but instead of latitude and longitude, you have hundreds or thousands of dimensions capturing different aspects of meaning:

- Distance from "happy" to "joy": very small
- Distance from "happy" to "quantum": very large
- Distance from "king" to "queen": similar to "man" to "woman"

The famous example: take the vector for "king," subtract "man," add "woman." The nearest neighbor? "Queen." The math captures analogies that humans recognize intuitively.

When you search a vector database in RAG, you're asking: "What stored content lives near this query in meaning-space?" The answer might surprise you—embeddings capture semantic similarity that keyword search misses entirely.

## Attention: the spotlight mechanism

The breakthrough behind modern LLMs is the attention mechanism. It solves a fundamental problem: in a long sequence, which parts matter for predicting the next word?

In the sentence "The trophy doesn't fit in the suitcase because it's too big," what does "it" refer to? Humans instantly know: the trophy (big things don't fit). A model needs to learn this by attending to the right words.

Attention works like a spotlight. For each token being processed, the model calculates relevance scores for every other token in context. High scores mean "pay attention here." Low scores mean "ignore this."

Self-attention is the model talking to itself: "Given the word 'it,' which previous words help me understand what 'it' means?"

Cross-attention connects different sequences: "Given this prompt, which parts of my training are relevant?"

The "transformer" architecture runs multiple attention heads in parallel, each learning to focus on different relationship types. One head might track grammar. Another tracks entities. A third captures sentiment. Together, they build understanding from multiple angles.

## The training loop: a million small corrections

Training an LLM looks nothing like programming. There's no explicit rule-writing, no decision trees, no hand-crafted logic. Instead:

1. Show the model some text
2. Hide the next word
3. Ask it to predict
4. Compare prediction to reality
5. Nudge the numbers slightly toward correct
6. Repeat billions of times

Each nudge is tiny—a fraction of a percent. But scaled across trillions of tokens and months of compute, patterns crystallize. The model learns that "Paris" follows "capital of France" not because someone wrote that rule, but because that pattern appeared thousands of times in training data.

This is why LLMs hallucinate. They're not retrieving facts—they're predicting likely sequences. If a plausible-sounding but false statement fits the pattern, it gets generated.

## Temperature: the confidence dial

When an LLM predicts the next token, it doesn't output a single answer—it outputs a probability distribution over all 50,000 tokens.

**Temperature** controls how to sample from that distribution:

- **Temperature 0** (greedy): Always pick the highest-probability token. Deterministic, repetitive, safe.
- **Temperature 0.7** (default): Sample probabilistically, favoring high-probability tokens but allowing variation.
- **Temperature 1.0+**: More randomness, more creativity, more chaos.

Low temperature makes the model a conservative librarian: "Based on everything I've seen, this word is most likely."

High temperature makes it a jazz musician: "Sure, that word would work, but what if we tried something unexpected?"

For code generation, use low temperature (0.0-0.3). For creative writing, go higher (0.7-1.0). For brainstorming, crank it up and see what emerges.

## Context windows: the working memory

A context window is how much text the model can consider at once. Everything outside the window might as well not exist.

In 2023, 4K tokens was standard. By late 2025:
- Claude: 200K tokens
- GPT-4: 128K tokens
- Gemini: 1 million tokens (2 million in preview)

Larger context windows enable new workflows:
- Feed entire codebases for analysis
- Process book-length documents in one pass
- Maintain conversation history across extended sessions

But context isn't memory. The model doesn't remember previous conversations unless you include them in the current prompt. Each API call starts fresh.

## Prompt engineering: speaking the model's language

LLMs don't understand instructions—they predict continuations of prompts. This sounds pedantic, but it changes how you write prompts.

**Bad prompt**: "Tell me about Python."
**Why it's bad**: Too many valid continuations. The model might explain the snake, the language, or Monty Python.

**Good prompt**: "You are a senior software engineer explaining Python's garbage collection mechanism to a junior developer. Focus on reference counting and the generational collector. Use concrete examples."
**Why it works**: Constrains the continuation space. The model has fewer plausible directions.

System prompts work because they establish patterns. When you tell Claude "You are a helpful assistant that writes concise code," you're setting up patterns that influence every subsequent prediction.

Chain-of-thought prompting ("Let's think step by step") works because reasoning steps appear in the model's training data. Mathematical proofs, debugging sessions, and analytical essays all include explicit reasoning. By triggering that pattern, you get better results.

## Fine-tuning: teaching new tricks

Pre-training creates a general model. Fine-tuning specializes it:

- **Instruction tuning**: Train on prompt-response pairs to make the model better at following directions
- **RLHF (Reinforcement Learning from Human Feedback)**: Train on human preferences to make outputs more helpful
- **Domain adaptation**: Train on specialized text (legal documents, medical records) to improve performance in that domain

The breakthrough insight: you don't need to retrain from scratch. A model that cost $100 million to train can be fine-tuned for thousands of dollars.

This is why base models feel raw while chat models feel polished. Same architecture, different finishing school.

## Why hallucinations happen (and won't go away)

The term "hallucination" misleads. It implies a malfunction—as if the correct behavior is waiting to be fixed. But LLMs hallucinate by design.

The model doesn't store facts. It stores patterns. When you ask "Who invented the telephone?" the model isn't retrieving a fact—it's predicting that "Alexander Graham Bell" is the most likely continuation of text matching that pattern.

Usually the pattern is right. Sometimes it's wrong. The model has no way to know the difference.

This is why RAG (retrieval-augmented generation) matters. Instead of asking the model to remember, you retrieve relevant documents and include them in the prompt. Now the model is *reading*, not *remembering*—a task it handles much better.

## The energy question nobody wants to answer

Training GPT-4 consumed an estimated 50 GWh of electricity—enough to power 4,500 American homes for a year. Each query to a frontier model uses 10-100x more energy than a Google search.

The carbon footprint varies wildly by location. A query processed in Iceland (geothermal) has different impact than one processed in West Virginia (coal).

This matters for two reasons:
1. At scale, energy costs dominate operational expenses
2. The environmental impact of AI infrastructure is non-trivial

The models keep getting bigger. The context windows keep expanding. The energy demands keep growing. Whether this trajectory is sustainable depends on who you ask.

## The compression hypothesis

One way to think about LLMs: they're compression algorithms for human knowledge.

When you compress a file, you identify patterns and encode them efficiently. A 1GB file might compress to 100MB because the compression algorithm found redundancies to exploit.

LLM training compresses the internet into model weights. Those weights encode statistical patterns that let you decompress *similar* content—not identical, but drawn from the same distribution.

This explains both the power and the limits. The model can generate text that *sounds like* training data because it captured the underlying patterns. It can't generate truly novel content that differs fundamentally from what it learned.

---

## The takeaway

LLMs are pattern-matching engines trained on human text. They predict likely continuations without understanding meaning. They hallucinate because prediction and memory are different tasks. They seem intelligent because human language is highly structured, and they've learned the structure.

None of this diminishes their utility. A calculator doesn't understand mathematics, but it's still useful for computation. An LLM doesn't understand language, but it's still useful for language tasks.

The key is knowing what you're working with: a remarkably capable pattern-matcher with no actual knowledge, no memory between calls, and a tendency to generate plausible nonsense when the pattern isn't clear.

Use RAG when you need facts. Use low temperature when you need consistency. Use large context windows when you need the model to consider more information. And never assume the output is true without verification.

Understanding the machine doesn't make it less useful. It makes you better at using it.

![An abstract visualization of meaning-space, points of light connected by glowing threads representing word relationships, dark background, constellation aesthetic](https://pixy.vexy.art/)

---

*Next: Chapter 17 examines the ethics of AI-assisted creation—when the machine becomes collaborator.*
