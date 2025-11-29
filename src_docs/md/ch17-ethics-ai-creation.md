# Chapter 17: The ethics of AI-assisted creation

> "We shape our tools, and thereafter our tools shape us."
> — Marshall McLuhan (attributed)

*Every tool changes its user. The question is whether you're changing intentionally.*

## The collaborator who never sleeps

The font specimen arrived from a client: sixteen weights of a new sans-serif, designed entirely in collaboration with Claude. The designer had used AI to generate variable font masters, interpolate weights, and suggest optical size adjustments. The type was beautiful—clean, contemporary, with subtle humanist touches that made it feel warm without being twee.

But something nagged at me.

The kerning felt familiar. The proportions of the lowercase 'e' reminded me of something. I ran the specimen through my font analysis tool and found it: the letter shapes borrowed heavily from Helvetica Now, with curve handling suspiciously similar to Inter.

The AI hadn't created something new. It had averaged its training data.

This chapter isn't about whether AI-assisted creation is good or bad. It's about the questions you need to ask—and keep asking—as the tools evolve.

## Imagine...

Imagine a ghostwriter who has read every book in your genre. They know exactly which tropes resonate with readers. They understand pacing, structure, the rhythm of sentences that keeps pages turning.

Now imagine hiring them. They write quickly, tirelessly, for pennies per word. The output is competent—sometimes brilliant. But gradually you notice something: your voice has changed. The ghostwriter's patterns have infected your thinking. You can't write a sentence anymore without hearing their cadence underneath.

That's the bargain with AI assistance. Capable partners who work for free, but charge in ways that don't show up on invoices.

---

## Attribution and authorship

When I use an LLM to write code, who wrote that code?

The model was trained on billions of lines of human-written code. My prompt shaped its output. My judgment decided which parts to keep. The final result reflects neither human nor machine exclusively—it's a hybrid artifact.

Current legal frameworks weren't designed for this. Copyright requires human authorship. The US Copyright Office has ruled that AI-generated content receives no copyright protection. But what about AI-assisted content? The line blurs.

**The practical implications:**

If you generate marketing copy with Claude, you own it—the prompt was your creative input. But if you ask for "a logo design" and accept the output unchanged, that's murkier. The more the AI contributed, the weaker your claim.

For code, the stakes differ. Most code isn't protected by copyright anyway—it's protected by being secret. But if you're building on AI-generated code that derived from someone else's repository, you might be inadvertently incorporating patterns that originated elsewhere.

**What I do:**

I treat AI outputs as first drafts that require human refinement. The finished work reflects my judgment, even if the raw material came from a machine. This isn't just legal prudence—it's practical wisdom. Unedited AI output usually needs work anyway.

## The derivative problem

LLMs generate by predicting likely continuations. This means their outputs necessarily reflect their training data. When you ask Claude for a font recommendation, it draws on patterns from thousands of discussions about fonts—including discussions that quoted specific recommendations from specific designers.

This creates a derivative works problem at scale.

**Individual level**: Unlikely to matter. If Claude suggests "Consider Satoshi for a modern sans-serif," that suggestion emerges from aggregate patterns, not any single source.

**Aggregate level**: Potentially troubling. If millions of users ask Claude for font recommendations, and those recommendations cluster around a few options that were prominent in training data, the AI is effectively amplifying certain voices while silencing others.

The designers whose work and opinions shaped the training data receive no compensation, no attribution, no even awareness that their expertise is being distributed.

**The photography parallel:**

Early photographers faced similar questions. Was a photograph art or merely mechanical reproduction? Did the photographer "create" the image, or did the camera? Courts eventually decided that creative choices—framing, lighting, timing—constituted authorship.

We're heading toward a similar resolution with AI. The creative choices are the prompts, the curation, the refinement. But the resolution isn't settled, and the edge cases multiply.

## Skill atrophy: use it or lose it

The typographer who relies on AI for kerning suggestions gradually loses their eye for spacing. The developer who generates boilerplate code loses the muscle memory of writing it. The writer who outsources first drafts loses the struggle that produces original voice.

This isn't hypothetical. I've watched it happen.

A junior developer on a project started using Claude for every coding task. Their output improved immediately—clean, well-structured code that followed best practices. But when the AI produced a subtle bug, they couldn't find it. They'd never learned to debug because they'd never learned to code.

**The calculator analogy breaks down:**

People argue that AI is like calculators for mental math—we don't need the skill anymore. But math fundamentals remain teachable because the domain is bounded. You can verify calculator output by estimation.

Language and design aren't bounded. You can't estimate whether code is correct. You can't verify whether prose is true. If you've never developed the underlying skill, you can't evaluate the AI's output.

**What I do:**

I deliberately practice without AI assistance. Not for everything—that would be wasteful. But regularly, for skills I want to maintain. I write first drafts by hand sometimes. I kern manually sometimes. The struggle is part of staying sharp.

## Environmental cost

Let's talk numbers.

Training GPT-4 consumed approximately 50 GWh of electricity—enough to power 4,500 average American homes for a year. Each inference query uses 10-100x more energy than a traditional web search. A reasonable estimate puts Claude at 0.5-5 Wh per query depending on response length.

Sounds small. Now multiply by billions of queries per day across all providers.

The AI industry's energy consumption is growing faster than data centers can add renewable capacity. Most inference runs in the United States, where grid mix varies from 90% renewable (Pacific Northwest) to 90% fossil fuels (certain Midwest regions).

**The uncomfortable math:**

If you make 100 queries per day to Claude (a busy developer's workload), you're consuming roughly 50-500 Wh daily. That's 18-182 kWh annually—equivalent to running a refrigerator for 2-20 weeks.

The individual impact seems manageable. The aggregate impact is significant. And the trajectory points toward more computation, not less.

**What I do:**

I don't pretend this is solved. I try to use AI efficiently—batching questions, avoiding redundant queries, preferring smaller models when they suffice. But I'm not going to stop using the tools. The productivity gains are too substantial.

This is the uncomfortable truth of many environmental questions: individual choices matter less than systemic changes. The industry needs cleaner power, more efficient models, and better utilization. Users can push for that, but we can't deliver it.

## The homogenization problem

Ask Claude for a business plan, and you'll get a business plan that looks like every other business plan Claude generates. The structure, the sections, the phrases—they converge toward an average.

This is fine for boilerplate. Boilerplate should be average by definition.

But creative work isn't boilerplate. If every novelist uses AI for first drafts, novels will homogenize toward the mean of novels in the training data. If every designer uses AI for layouts, layouts will converge toward common patterns.

We might be trading diversity of human expression for a kind of statistical smoothness.

**The font analogy:**

Consider what happened to fonts after personal computers. Before DTP, font selection was limited by what typesetters owned. Diversity was constrained by economics.

After DTP, anyone could use any font. You'd expect diversity to explode. Instead, we got decades of Arial and Times New Roman—the defaults. Unlimited choice led to clustering around familiar options.

AI might have the same effect on creative expression. Not through limitation, but through convenience. Why struggle for originality when "good enough" is instant?

**What I do:**

I use AI for tasks where average is acceptable: documentation, boilerplate, routine analysis. For creative work, I start without AI, develop my own direction, then use AI to accelerate execution—not ideation.

## Dependency and fragility

Your workflow depends on Claude. Claude depends on Anthropic. Anthropic depends on compute providers, electricity grids, investor patience, and regulatory environments.

That's a lot of dependencies for a critical tool.

**What happens when:**

- Anthropic raises prices 10x?
- Your government bans AI assistants?
- A model update breaks your carefully-tuned prompts?
- The API goes down during a deadline?

Each of these has happened to someone. Prices change. Governments regulate. Models update without notice. Services experience outages.

**The mitigation playbook:**

- Don't build workflows that require AI
- Build workflows that are *accelerated* by AI
- Maintain manual capabilities
- Document your prompt strategies in case models change
- Have fallback providers configured

The goal isn't to avoid dependency—that ship has sailed. The goal is to make dependency manageable.

## The authenticity question

When you publish a blog post written with AI assistance, do you disclose that?

When you ship code that Claude helped write, do you mention it in comments?

When you design a font with AI-assisted kerning, do you credit the tool?

There's no consensus. Arguments exist for full disclosure, partial disclosure, and no disclosure.

**The case for disclosure:**

Readers deserve to know how content was created. AI-assisted work may contain patterns derived from others' work. Transparency builds trust.

**The case against disclosure:**

No one discloses their spell-checker. No one credits their calculator. Tools are tools. What matters is the output.

**My position:**

Disclose when the AI's contribution was creative. Don't bother for mechanical assistance.

If Claude wrote your introduction and you kept it mostly unchanged, disclose. If Claude fixed your grammar, don't. If Claude suggested a color palette you used verbatim, disclose. If Claude calculated HSL values for you, don't.

The line is fuzzy. Draw it honestly.

## Consent and training data

The text that trained Claude came from somewhere. Books, websites, forums, social media—billions of tokens scraped from the internet. Most of it was published by people who never imagined their words would train a machine.

Is that fair?

**The legal answer:**

Training on publicly available data is generally considered fair use in the US, though cases are ongoing. The EU's approach differs. The question isn't settled.

**The ethical answer:**

More complicated. Fair use is a legal doctrine, not a moral one. A creative writing forum might have been technically "public" while also representing a community that would never have consented to mass scraping.

**The practical answer:**

The training data is already incorporated. We can't un-ring that bell. What we can influence is the future: what data gets collected going forward, how people are compensated, what consent looks like.

**What I do:**

I advocate for better data practices going forward. I support legislation requiring disclosure of training data sources. I use providers who are working on compensation mechanisms for creators. But I also recognize that I'm benefiting from a fait accompli.

---

## The takeaway

AI assistance isn't neutral. It shapes what you create, how you think, and which skills atrophy from disuse. The tools are too powerful to ignore, too consequential to use carelessly.

The questions don't have clean answers:
- Who owns AI-assisted work?
- What do you owe the humans in the training data?
- How do you maintain skills while using shortcuts?
- When must you disclose AI involvement?
- What's the environmental cost, and who bears it?

What I know: engaging with these questions matters more than resolving them. The person who uses AI thoughtlessly loses something—not legally or financially, but intellectually. The struggle to use these tools well, ethically, sustainably—that struggle is part of staying human in an age of capable machines.

The kerning on that client's typeface? I told them. The AI had converged on familiar solutions because familiar solutions dominated its training data. We went back and redrew the most derivative characters. The final type was still AI-assisted, but it was theirs—shaped by human choices, not just statistical averages.

That's the job now. Not to avoid AI, but to direct it. To use the machine without being used by it. To stay sharp while using tools that make sharpness optional.

It's harder than letting the AI do everything. That's rather the point.

![Hands of a craftsman and a machine intertwined, working on the same piece, renaissance style drawing with digital elements](https://pixy.vexy.art/)

---

*This concludes The Vexy Book of Large Language Models. What you build with these tools is up to you.*
