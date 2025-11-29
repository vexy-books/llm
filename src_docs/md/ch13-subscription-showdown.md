# Chapter 13: Subscription showdown

*Paying for AI: what $20/month actually gets you.*

> "There's no such thing as a free lunch."
> — Milton Friedman

## The subscription landscape

Every major LLM provider now sells subscriptions. ChatGPT Plus, Claude Pro, Gemini Advanced—all priced around $20/month, all promising "access to the latest models."

But the details matter. Rate limits differ. Feature sets vary. Some subscriptions are worth it; others are API credits in disguise.

This chapter breaks down what you actually get for your money.

> **2025 Update**: Subscription tiers have expanded significantly. ChatGPT now offers Plus ($20/mo) and Pro ($200/mo). Claude has Pro ($20/mo) and Max tiers ($100-200/mo). Google released Gemini 3 Pro. API pricing dropped across all providers—Claude Haiku is now $0.25/1M input tokens, GPT-4o Mini is $0.60/1M. The free tier wars continue: Google AI Studio still offers generous free access for testing.

## Imagine...

Imagine three competing gyms on the same block. Each charges $20/month. Each promises "full access to equipment." But one gym has a pool, another has personal trainers, and the third has 24-hour access.

The pricing looks identical. The value depends entirely on what you actually use. The swimmer doesn't care about personal training. The night owl needs 24-hour access more than any equipment. The beginner might need the trainer to make any progress at all.

AI subscriptions work the same way. ChatGPT Plus, Claude Pro, and Gemini Advanced all cost roughly the same. But one excels at image generation, another at nuanced writing, the third at processing million-token documents. The best subscription isn't the "objectively best" one—it's the one that matches how you actually work.

Most people pick based on brand familiarity or feature lists they'll never use. The smart play: know your workflow, then match it to the subscription that supports it. Or realize that free tiers cover your needs entirely.

---

## The big three compared

### ChatGPT Plus ($20/month)

**What you get:**
- GPT-4o access (higher rate limits than free)
- GPT-4o with Canvas (collaborative editing)
- Advanced Data Analysis (code execution)
- DALL-E 3 image generation
- Custom GPTs (create and use)
- Browse with Bing
- Voice mode with GPT-4o
- Priority access during high demand

**Rate limits:**
- GPT-4o: ~80 messages/3 hours
- DALL-E 3: ~50 images/day
- Advanced Data Analysis: Session-based

**Best for:**
- General-purpose AI assistant use
- Creative work (images, writing)
- Non-technical users wanting maximum features

**Limitations:**
- No API access included
- Limits are soft (can be throttled during demand spikes)
- Can't use GPT-4 Turbo via chat interface

### Claude Pro ($20/month)

**What you get:**
- Claude 3.5 Sonnet (higher limits)
- Claude 3 Opus (when selected)
- 5x more usage than free tier
- Priority access during peak times
- Projects (organize conversations)
- Artifacts (interactive code/documents)

**Rate limits:**
- ~100 messages/8 hours on Sonnet
- ~30 messages/day on Opus
- Limits reset on rolling window

**Best for:**
- Long-form writing and analysis
- Code review and generation
- Users who value thoughtful responses over speed
- Document analysis (PDFs, long texts)

**Limitations:**
- No image generation
- No web browsing (yet)
- No API access included
- Opus is significantly rate-limited

### Google Gemini Advanced ($19.99/month via Google One AI Premium)

**What you get:**
- Gemini 1.5 Pro (1M token context)
- Gemini in Gmail, Docs, Sheets, Slides
- Deep Research feature
- Gems (custom Gemini personalities)
- 2TB Google One storage
- Google Workspace integration

**Rate limits:**
- Generous (exact limits not published)
- 1M token context available

**Best for:**
- Google Workspace power users
- Long document analysis (1M context)
- Research tasks
- Users already paying for Google One

**Limitations:**
- Quality inconsistency vs GPT-4o/Claude
- Deep integration means deep lock-in
- Image generation via Imagen (limited)

## Side-by-side comparison

| Feature | ChatGPT Plus | Claude Pro | Gemini Advanced |
|---------|--------------|------------|-----------------|
| **Price** | $20/mo | $20/mo | $19.99/mo |
| **Best Model** | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro |
| **Context Window** | 128K | 200K | 1M |
| **Image Gen** | DALL-E 3 | No | Imagen |
| **Code Execution** | Yes | No | Yes |
| **Web Browse** | Yes | No | Yes |
| **File Upload** | Yes | Yes | Yes |
| **Voice Mode** | Yes | No | Yes |
| **API Access** | No | No | No |
| **Custom Bots** | GPTs | Projects | Gems |

## Value analysis

### ChatGPT Plus: The Swiss Army Knife

**Value proposition**: Everything in one place.

GPT-4o handles most tasks adequately. Add DALL-E 3, code execution, and web browsing, and you have a complete toolkit. For users who want one subscription that does everything, ChatGPT Plus delivers.

**ROI calculation:**
- DALL-E 3 alone costs $0.04-0.12/image via API
- 50 images/month = $2-6 API equivalent
- GPT-4o API: $2.50/1M input, $10/1M output
- Average chat session ~2K tokens = ~$0.02
- 80 sessions/day × 30 days = 2,400 sessions = ~$48 API equivalent

**Verdict**: Worth it if you use multiple features. Overkill if you only chat.

### Claude Pro: The Writer's Choice

**Value proposition**: Quality over quantity.

Claude excels at nuanced writing, careful analysis, and long-context work. The 200K context window (free tier gets 200K too, but with usage limits) handles entire codebases and long documents.

**ROI calculation:**
- Claude 3.5 Sonnet API: $3/1M input, $15/1M output
- Average session ~3K tokens = ~$0.05
- 100 sessions/day = $5/day = $150/month API equivalent
- 5x free tier = ~$30-50 value

**Verdict**: Worth it for heavy users who value Claude's writing quality.

### Gemini Advanced: The Google Play

**Value proposition**: Integration + storage + AI.

The 2TB storage alone costs $9.99/month. Gemini integration across Google Workspace is unique. The 1M context window is unmatched.

**ROI calculation:**
- Google One 2TB: $9.99/month
- Gemini 1.5 Pro API: $1.25/1M input (up to 128K), higher for longer
- 1M context operations via API: expensive
- Workspace integration: priceless if you're in the ecosystem

**Verdict**: Worth it for Google Workspace users. Less compelling otherwise.

## The premium tier: $200/month power users

For those who hit limits constantly, premium tiers emerged in 2025:

### ChatGPT Pro ($200/month)

- Unlimited GPT-4o and o1-Pro access
- Extended thinking mode without quotas
- Operator (autonomous browser agent)
- Codex agent for coding tasks
- Sora video generation
- Priority infrastructure access

**Best for**: Researchers, developers, and power users who need unlimited access to the most capable models.

### Claude Max ($100-200/month)

- **Max 5x** ($100/mo): 5x Pro rate limits
- **Max 20x** ($200/mo): 20x Pro rate limits, Claude 4.1 Opus access
- Extended output limits
- Early access to new features
- Highest priority during peak times

**Best for**: Professional users who consistently hit Claude Pro limits—heavy coding, document analysis, or continuous daily use.

### The $200/month question

Is 10x the price worth it? Depends on your bottleneck:

- **Hit limits daily**: Premium pays for itself in productivity
- **Occasional heavy use**: Stick with Pro, wait out limits
- **Building products**: API is still more cost-effective

Most users overestimate their need for premium tiers. Track your actual limit hits for a week before upgrading.

## The teams/enterprise tier

When individual plans aren't enough:

### ChatGPT Team ($25/user/month, annual)

- Higher rate limits
- Admin console
- No data training on your conversations
- Workspace for team collaboration
- 32K context (vs 8K free)

### Claude Team ($25/user/month, minimum 5 users)

- Higher rate limits
- Admin controls
- SOC 2 Type II compliance
- Data not used for training
- Centralized billing

### Gemini Business ($20/user/month via Google Workspace)

- Enterprise-grade security
- Admin controls
- Data stays in your Google Workspace
- Gemini across all Workspace apps

### When Teams Plans Make Sense

1. **Data sensitivity**: You need guarantees about training data
2. **Compliance**: SOC 2, HIPAA requirements
3. **Centralized billing**: Company expense management
4. **Collaboration**: Shared workspaces matter

## API vs Subscription

Sometimes the API is cheaper. Sometimes it's not.

### When API Wins

**Light usage (< 50K tokens/day)**:
- Claude Sonnet API: ~$0.15/day for 50K tokens
- Claude Pro subscription: $0.67/day

**Specific model access**:
- Need GPT-4 Turbo specifically? API only.
- Need Claude 3 Opus without rate limits? API.

**Automation**:
- Building products? You need the API anyway.
- Subscription is for humans, not bots.

### When Subscription Wins

**Heavy interactive use**:
- 100+ conversations/day makes API expensive
- Subscription = predictable costs

**Multi-modal needs**:
- DALL-E 3 generation adds up fast via API
- Subscription bundles everything

**Feature access**:
- Advanced Data Analysis (code interpreter) isn't available via API
- Custom GPTs/Projects need subscription

### Hybrid Strategy

Many teams use both:

```
Human exploration → Subscription
  - Brainstorming
  - Research
  - Writing drafts

Production workloads → API
  - Automated pipelines
  - Customer-facing features
  - Batch processing
```

## The free tier reality

Before subscribing, exhaust the free options:

| Provider | Free Offering |
|----------|--------------|
| **Claude** | Limited Sonnet, web only |
| **ChatGPT** | GPT-4o mini, limited features |
| **Gemini** | Gemini 1.5 Flash, generous limits |
| **Mistral (Le Chat)** | Multiple models, good limits |
| **Perplexity** | 5 Pro searches/day |

### Free Tier Strategy

1. **Use Gemini** for research (1M context, generous limits)
2. **Use ChatGPT free** for quick questions
3. **Use Claude free** for writing tasks
4. **Use Perplexity** for web-grounded answers

You can go surprisingly far without paying anything.

## Specialized subscriptions

Beyond the big three:

### Perplexity Pro ($20/month)

- Unlimited Pro searches (uses Claude/GPT-4)
- File analysis
- Focus modes (Academic, YouTube, Reddit)
- API credits included

**Best for**: Researchers who need web-grounded answers.

### GitHub Copilot ($10/month individual, $19/month business)

- Code completion in IDE
- Chat interface
- CLI integration
- Pull request summaries

**Best for**: Developers writing code daily.

### Notion AI ($10/month add-on)

- AI writing assistance in Notion
- Q&A over workspace content
- Autofill databases

**Best for**: Notion power users.

### Cursor Pro ($20/month)

- AI-powered IDE
- Claude/GPT-4 integration
- Codebase-aware completions
- Multi-file editing

**Best for**: Developers wanting AI-native coding.

## Decision framework

### Choose ChatGPT Plus if:

- You want one tool for everything
- Image generation matters
- You use voice mode
- You build/use Custom GPTs
- You're non-technical

### Choose Claude Pro if:

- Writing quality is paramount
- You analyze long documents
- You prefer thoughtful over fast
- You want fewer hallucinations
- Code review is a use case

### Choose Gemini Advanced if:

- You're deep in Google ecosystem
- You need 1M token context
- You want the storage bundle
- You work heavily in Docs/Sheets

### Choose None if:

- Free tiers meet your needs
- You need API anyway
- Your usage is < 20 requests/day
- You can use multiple free tiers

## The portfolio approach

Some power users subscribe to multiple:

**The Writer's Stack**:
- Claude Pro for drafts and editing
- ChatGPT Plus for research and images
- Cost: $40/month

**The Developer's Stack**:
- GitHub Copilot for code completion
- Claude Pro for architecture discussions
- Cost: $30/month

**The Researcher's Stack**:
- Perplexity Pro for web research
- Gemini Advanced for long documents
- Cost: $40/month

## Cost over time

Annual costs add up:

| Subscription | Monthly | Annual |
|--------------|---------|--------|
| ChatGPT Plus | $20 | $240 |
| Claude Pro | $20 | $240 |
| Gemini Advanced | $20 | $240 |
| Two subscriptions | $40 | $480 |
| Three subscriptions | $60 | $720 |

Compare to API costs for equivalent usage:
- Light user (~$10/month API): Subscription likely overkill
- Medium user (~$50/month API): Subscription saves money
- Heavy user (~$200/month API): Subscription definitely saves money

## Getting started: two approaches

**Simple approach**: Don't subscribe at all. Stack free tiers strategically.

```
Daily workflow using only free tiers:

Morning research:
→ Gemini (1M context) for long document analysis
→ Perplexity (5 free Pro searches) for web research

Writing and editing:
→ Claude.ai free for drafts (limited but capable)
→ ChatGPT free for quick rewrites

Code assistance:
→ Claude.ai for architecture discussions
→ GitHub Copilot free tier for completions

Creative work:
→ Bing Image Creator (DALL-E 3, free)
→ Ideogram free tier for text-in-images
```

This covers 80% of use cases without spending anything. Hit limits? Wait, use a different tool, or batch your work.

**Production approach**: One subscription plus API for overflow.

```python
# Track actual usage before subscribing
usage_log = {
    "claude_messages": 0,
    "gpt_messages": 0,
    "images_generated": 0,
    "long_docs_analyzed": 0
}

# After 2 weeks of tracking:
# - 60 Claude messages/day → Claude Pro ($20) covers this
# - 10 GPT messages/day → Free tier sufficient
# - 5 images/week → Bing Creator free tier works
# - 3 long docs/week → Gemini free handles it

# Decision: Claude Pro + API credits for overflow
# Monthly cost: $20 subscription + ~$5 API = $25
# vs. ChatGPT Plus + Claude Pro + Gemini = $60
```

```bash
# Hybrid strategy: subscription for interactive, API for automation
# ~/.config/llm/cost-strategy.yaml

interactive:
  primary: claude-pro-subscription
  fallback: chatgpt-free
  research: gemini-free

automated:
  provider: anthropic-api  # Claude Haiku for batch work
  budget_cap: $10/month
  model: claude-3-haiku

# Result: Great AI access for ~$30/month instead of $60+
```

The simple approach costs nothing and works for most people. The production approach picks ONE subscription strategically and supplements with free tiers and cheap API calls. Track your actual usage before deciding—most people subscribe to things they don't need.

## The takeaway

Subscriptions make sense when:

1. **Usage is high**: >50 messages/day
2. **Features matter**: Image gen, code execution, voice
3. **Predictability matters**: Fixed costs > variable API
4. **Convenience matters**: No API key management

Skip subscriptions when:

1. **Usage is low**: <20 messages/day
2. **You're building products**: API is required anyway
3. **Free tiers suffice**: Gemini + Mistral + Claude free covers a lot
4. **You need specific models**: Subscription doesn't offer them

The best strategy: Start free, measure usage, subscribe only when limits hurt. Most people overestimate their AI usage and overpay for subscriptions they barely touch.

![A person juggling subscription cards from OpenAI, Anthropic, and Google, with coins falling from their pockets, cartoon editorial style](https://pixy.vexy.art/)

---

*Next: Chapter 14 digs into cost optimization—how to get the same results for less.*
