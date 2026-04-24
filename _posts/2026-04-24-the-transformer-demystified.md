---
layout: post-article
title: "The Transformer, Demystified: A Factory Floor That Runs on Language"
date: 2026-04-24
permalink: /posts/the-transformer-demystified/
excerpt: "Strip the scary name away and a transformer is six boring stations on a conveyor belt. Text goes in. Numbers travel. A word comes out. Here's what happens at each station."
read_time_label: "12 min read"
accent: amber
math: true
---

Okay. Let's do the thing where we stop being afraid of the word "transformer."

I know. It sounds like something that should require a PhD and a warning label. The truth is almost insulting: a transformer is a factory floor with about six stations on it. A sentence walks in one end. A guess at the next word walks out the other. In between is a conveyor belt. And every station does one small, boring job.

That's it. That's ChatGPT. That's Claude. That's the thing people are worried might take over the world. Six stations and a belt.

The reason it *feels* complicated is that the belt is really long, the stations run in parallel, and the vectors being shuffled around have several hundred dimensions that we can't visualise. But the structure? The structure is dumb in the best possible way.

Let's walk the floor.

---

## Tour the factory

{% include demos/factory-floor.html %}

Click a station. Press play. Watch the little token slide along the belt. That is — genuinely — the shape of what happens when you type a message to an LLM. All the rest is detail.

## The six stations, briefly

**1. Tokenizer.** The text comes in as characters, and characters are annoying to do math on. The tokenizer chops the text into subword pieces and assigns each piece an integer ID. `"unhappily"` might become `["un", "happ", "ily"]` → `[403, 7829, 6148]`. We'll spend a whole post on this in the next blog, because it's where half the model's weirdest bugs live.

**2. Embedding.** Each integer ID gets looked up in a giant table. Out comes a 768-dimensional vector (for GPT-2 small; 12,288 for GPT-3, 16,384 for the really big stuff). Similar-meaning tokens get similar vectors. This is the model's first, rough guess at what each word "means."

**3. Attention.** This is the one everyone talks about. At every position in the sentence, the model asks: *"which other positions should I pay attention to, and what information should I pull from them?"* The word `"it"` notices the noun three tokens back. The verb notices its subject. Nothing magical, just a weighted average steered by the vectors themselves.

**4. MLP.** A per-position feed-forward network. Same-shape input, same-shape output, but in between it does non-linear processing. This is where a lot of the model's "knowledge" seems to live — facts, transformations, little lookup patterns.

**5. Repeat.** Attention + MLP is one *block*. You stack a pile of these. GPT-2 small has 12. Llama 3 8B has 32. GPT-4 is thought to have something like 120. Every block reads from the conveyor belt and writes back to it.

**6. Unembedding.** At the very end, the final vector is compared against every word in the vocabulary. The comparison produces a score for each word, which gets softmaxed into a probability distribution. Sample from that, and you have your next token.

That is the entire pipeline. I am not hiding anything. There is no secret extra box.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>The transformer is a factory that takes half-formed meaning and refines it, one station at a time. The belt carries everyone's thoughts at once. Each station reads the belt, adds its own contribution, and puts the updated thoughts back. By the time the belt reaches the end of the floor, the vector at the last position is confident enough to name a single next word.</p>
</aside>

## The belt, which is a very big deal

Notice I keep saying "the belt." In the official literature this is called the **residual stream**, and it is arguably the single most important idea in modern transformer interpretability.

Here's the trick. Each block does not *replace* the previous block's output. It *adds* to it. If $x_i$ is the vector on the belt at position $i$ before a block, then after the block:

$$x_i \;\leftarrow\; x_i + \text{block}(x_i)$$

<div class="math-translate">In words: the new belt vector equals the old belt vector plus whatever the block computed. Nothing gets overwritten. Everything accumulates.</div>

Sounds like a tiny detail. It is the reason transformers work.

- Information from early layers can flow all the way to the end without being forgotten.
- Each block only has to produce a small *correction* to the running total, which is way easier to train than "compute everything from scratch."
- For us MI-folk, the residual stream gives us a single, canonical place to *read the model's mind* at any point in the computation. Layer 3? Just look at the belt after block 3.

We're going to spend an entire post on the residual stream — it's that important — so I won't belabour it here. File it away: **the belt is the main character.**

## Why this thing beat everything else

For a long time, the reigning architecture for language was the **recurrent neural network** (RNN) — a model that reads text one token at a time, maintaining a little hidden state as it goes. Like reading a book by passing a single post-it note from page to page and updating it. It works. Sort of. Two problems:

1. **You can't parallelise it.** Token $t$ depends on token $t-1$'s hidden state. You have to process the sentence sequentially. On a GPU, that's a crime against nature.
2. **The post-it note forgets.** Long-range dependencies fade. By the end of a paragraph the model has basically lost the beginning.

Transformers ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) fixed both problems in one stroke by replacing the post-it note with attention. Now every position can look at every other position *directly*, all at once, in parallel. No sequential dependency, no forgetting. The belt is wide. The lookups are fast. The GPU is happy.

The paper was called, memorably, *"Attention Is All You Need,"* which turned out to be one of the most accurate titles in ML history.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>Every modern LLM you've heard of — GPT-4, Claude, Llama, Gemini — is a transformer. Different sizes, different training data, different tweaks. But the factory floor is the same six stations. Learn this once and you've got the scaffolding for every frontier model.</p>
</aside>

## "Decoder-only," and why we only care about that flavour

The original 2017 paper had an *encoder* (for reading the input) and a *decoder* (for writing the output). Machine-translation, that's what they were after.

Then people noticed: if you just want to generate text, you don't need the encoder. Chop it off. Keep the decoder. Train it to predict the next token, given all previous tokens. That's a **decoder-only transformer** — GPT-2, GPT-3, Claude, Llama, all of them.

For this post and every post after, when I say "transformer," I mean decoder-only. Simpler. One stream of tokens in, one probability distribution out. The structure you clicked through above.

(There are also *encoder-only* models like BERT, which are great for classification but don't generate text. We won't touch those here.)

## A quick note on "autoregressive"

One more word that sounds scarier than it is. **Autoregressive** means the model generates one token, appends it to the input, and generates the next one using the now-longer input. Then repeats.

```
input:  "The cat sat on the"
model:  → "mat"
input:  "The cat sat on the mat"
model:  → "."
input:  "The cat sat on the mat."
model:  → "<end>"
```

That's all it means. One token at a time, each conditioned on everything before it. The factory floor runs once per token. When you watch a chatbot "type," you are literally watching the factory floor run, over and over, as fast as the GPU allows.

## What's ahead on this belt

The next few posts walk through each station in detail:

- **Tokens** — what they are, why they're not words, and why this causes subtle chaos.
- **The residual stream** — the belt, in all its glory, plus the single most useful mental model for understanding an LLM.
- **Attention** — how a position decides what to look at, and the QK/OV decomposition that turned MI into a real science.
- **MLPs** — the feed-forward layers, and why we now think of them as *key-value memories*.
- **The full forward pass** — everything wired up, end to end, with real numbers from a real model.

By the end of that run, you'll have the complete vocabulary to read any transformer interpretability paper. The next blog starts with tokens.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani, A. et al. · 2017 · the original transformer paper</div></a></li>
  <li><a class="research-card" href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noopener"><div class="research-card__title">The Illustrated Transformer</div><div class="research-card__authors">Alammar, J. · 2018 · the friendliest diagram-led tour of the architecture</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021 · the residual-stream view used throughout this series</div></a></li>
  <li><a class="research-card" href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noopener"><div class="research-card__title">Language Models are Unsupervised Multitask Learners</div><div class="research-card__authors">Radford, A. et al. · OpenAI, 2019 · GPT-2</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener"><div class="research-card__title">Language Models are Few-Shot Learners</div><div class="research-card__authors">Brown, T. et al. · 2020 · GPT-3, the "scale it up" paper</div></a></li>
</ul>
