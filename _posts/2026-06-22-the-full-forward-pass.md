---
layout: post-article
title: "The Full Forward Pass: Putting Every Piece on the Belt"
date: 2026-06-22
permalink: /posts/the-full-forward-pass/
excerpt: "Tokens. Embeddings. Attention. MLPs. Residual stream. Unembedding. Six concepts. One factory floor. Let's run a real prompt through a real model and watch every stage produce its tiny contribution to a final next-token guess."
read_time_label: "11 min read"
accent: amber
math: true
---

We've spent six posts taking a transformer apart into pieces. Now I want to put it back together and run something through it, end to end, with real numbers, in front of you.

Because, and this is the bit that's hard to internalise from words alone, every piece we've talked about runs in *one shot*. Tokenize, embed, attention block 0, MLP block 0, attention block 1..., unembed. Single forward pass. A few milliseconds on a GPU. Out comes a probability distribution over 50,000 tokens. That's an inference. Repeat once for every word a chatbot generates.

So this post is short. The demo is the point. Let's run the floor.

---

## The grand tour

{% include demos/grand-tour.html %}

Pick a preset. Press **Step forward**. Or smash the auto-play. Watch each stage. The little token strip up top is the residual stream, showing the logit-lens prediction at every position. It evolves as you step from embedding to block 0 to block 5. By the time you reach the unembed stage, the rightmost cell has converged on the model's actual answer.

Try the "A B C D E F G A B C" preset for something interesting. You'll see how, in middle layers, the model picks up the *pattern* from the first half of the sequence and starts predicting that the next token will be "D", pure in-context induction.

## The seven stages, narrated

For the canonical prompt *"The capital of France is"*:

**1. Tokenize.** `[464, 3139, 286, 4881, 318]`. Five tokens. Note that "capital" gets its own token; "France" gets its own. "is" gets its own. The tokenizer doesn't know anything about the meaning yet.

**2. Embed.** Each ID becomes a 768-dim vector. The residual stream is initialised. Logit-lens at each position basically just says "the token itself" because there's no context-mixing yet.

**3. Block 0 (attn + MLP).** The first attention pass starts mixing information across positions. "France" gets a peek at "of" and "capital". The MLP starts annotating each token with surface features (proper noun, geographic entity, etc.). Logit-lens predictions are still mostly just the input tokens; the model hasn't "decided" anything yet.

**4. Blocks 1–4.** This is where the real semantic work happens. Names and geographic relations consolidate. Around block 3, the prediction at the final position starts looking distinctly French, country names, region names, "the". Block 4 starts pushing toward city names. By block 4 the top guess is often already "Paris", though with low confidence.

**5. Block 5 (final).** The last block sharpens the answer. Confidence on " Paris" climbs. Other candidates (" the"," France"," French") get suppressed.

**6. Unembed.** Project the final residual vector at the last position against all 50,257 token embeddings. Softmax. Out comes the distribution. " Paris" tops it, ~80%+ probability for distilGPT2. The rest of the mass is spread over " France", " Europe", " Britain", " Germany", and increasingly far-fetched options.

That is one forward pass. That is, mechanically, what every chatbot is doing every time it produces a token.

## The numbers, for the curious

For distilGPT2 specifically, on a 5-token prompt:

| Stage | Tensor shape | What lives here |
|------|-------------|----------|
| Embedding | $[5, 768]$ | one 768-dim vector per token |
| Each block input | $[5, 768]$ | the residual stream |
| QKV in each head | $[5, 64]$ × 3 | per-head query, key, value |
| Attention weights | $[5, 5]$ per head | what each position attends to |
| MLP intermediate | $[5, 3072]$ | 3,072 neuron activations per position |
| Final residual | $[5, 768]$ | post-block-5 stream |
| Logits | $[5, 50257]$ | unembedded score per token in vocab |

Multiply across layers and heads:
- 6 blocks × (12 heads + 1 MLP) = ~78 distinct sub-components.
- Each MLP has 3,072 neurons. So 6 × 3,072 = **18,432 MLP neurons** to potentially poke at.
- Plus 6 × 12 = **72 attention heads** to characterise.

That is the full surface area of distilGPT2. Modern frontier models are hundreds of times bigger.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The whole point of mechanistic interpretability is to take that surface area and label it. Every neuron, every head, what is it doing? What does it read from the residual stream? What does it write back? When we know that for every component, we can reconstruct the algorithm the model actually learned. We're a long way from solving that for frontier models, but we have the tools, and they get sharper every year.</p>
</aside>

## The thing I want you to notice

Run the demo on different prompts. Step through stages. Watch the logit-lens predictions evolve column by column.

Three observations that took me a while to internalise:

**1. Most of the work happens in the middle.** Embedding is dumb. The last block is decisive but small. The middle four blocks do most of the lifting. Look at how the prediction quality (the saturation, the confidence) rises in the middle layers and only locks in at the end.

**2. The non-final positions are not predicting anything useful.** The prediction at position 0 of "The capital of France is" is just the model's best guess for what comes after "The" alone, which is approximately useless out of context. Only the final position is asking the actual question. Earlier positions are doing *information accumulation* that the final position will retrieve via attention.

**3. The model doesn't reason in steps the way we do.** It does not first identify "France", then look up "is a country", then look up "capitals of countries", then output "Paris". It does, in parallel and across layers, a fuzzy soup of all of these things. The logit lens shows you the average direction, but the underlying computation is a giant linear-algebra-and-gates blob. The reason it works is the same reason gradient descent works: enough small contributions in the right direction add up to the right answer.

This last one is uncomfortable for anyone who wants neat causal stories. It's also why MI is hard. You can't ask "what happened at step 3?" because there is no step 3. You can ask "what did head 7 in layer 4 contribute?", and that you can answer.

## What you now know

After 13 posts: enough to read most modern interpretability papers without reaching for the dictionary every paragraph. Specifically:

- The architectural vocabulary: tokens, embeddings, attention (QK + OV), MLPs (key-value memories), residual stream, unembedding.
- The operational mental model: every block reads from and writes to a shared running sum.
- The investigative tools: logit lens, direct logit attribution, head/neuron characterisation.
- The conceptual furniture: superposition, polysemanticity, in-context induction.

That's a real foundation. It's enough to start poking at small models in code. It's enough to follow what the people at Anthropic, DeepMind, and the academic labs are publishing each month.

## Where this goes next

Up next, the curriculum widens out. The next chapter is about **features and circuits**, taking the architectural picture you now have and using it to actually find concrete pieces of computation inside trained models. Sparse autoencoders, the IOI circuit in detail, polysemanticity solved (mostly), the hunt for "concepts" in feature space.

But that's a different post. For now: you understand a transformer. The factory floor is no longer mysterious. It is, in fact, kind of boring. Six stations and a belt.

Which is exactly the point.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani, A. et al. · 2017 · the architecture in one paper</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021 · the residual-stream view used throughout this series</div></a></li>
  <li><a class="research-card" href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" target="_blank" rel="noopener"><div class="research-card__title">Interpreting GPT: the logit lens</div><div class="research-card__authors">Nostalgebraist · LessWrong, 2020 · the technique driving the demo</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson, C. et al. · Anthropic, 2022</div></a></li>
  <li><a class="research-card" href="https://huggingface.co/Xenova/distilgpt2" target="_blank" rel="noopener"><div class="research-card__title">Xenova/distilgpt2</div><div class="research-card__authors">Hugging Face · the actual model running in the demos above</div></a></li>
  <li><a class="research-card" href="https://github.com/neelnanda-io/TransformerLens" target="_blank" rel="noopener"><div class="research-card__title">TransformerLens</div><div class="research-card__authors">Nanda, N. · the standard library for poking at internals</div></a></li>
</ul>
