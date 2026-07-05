---
layout: post-article
title: "Never Lost in the Middle Again: The U-Shape Is a Training Artifact"
date: 2025-08-24
permalink: /posts/notes-on-never-lost-in-the-middle-again/
excerpt: "Long-context LLMs read the beginning of a document. They read the end. They mostly hallucinate the middle. Turns out the middle isn't hard — the training data just told the model not to bother."
read_time_label: "9 min read"
accent: warn
---

Companion note to [Never Lost in the Middle Again: Teaching LLMs to Care About the Center of Long Documents](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5877962).

Here's the thing that started this paper.

Load a 32k-token context into a big frontier model. Put the fact you want the model to retrieve at the *very start* of the context. Ask the question. Model nails it. Move that same fact to *the very end*. Model nails it. Now move it to the middle. Model confidently invents something.

Nothing about the model changed. Nothing about the question changed. Only the position changed. The accuracy dropped from 91% to 54%. This is real, robust, replicated across model families, and it's the reason "long context" is often more marketing than useful.

This paper is why it happens and how to make it stop.

---

## Look at the shape yourself

Scrub the fine-tuning slider from 0 to 100. Click any depth bucket to see what the model actually says at that position.

{% include demos/lost-in-middle.html %}

The red dashed curve is the baseline — every long-context LLM comes with this U pre-installed. The blue curve is what happens as we fine-tune with our recipe. Watch the middle bucket climb.

<aside class="callout callout--key">
  <div class="callout__label">The finding, in one line</div>
  <p>The lost-in-the-middle effect is mostly a training-data artifact. Fix the data distribution, most of the U flattens. Don't touch the architecture at all.</p>
</aside>

## Why the U-shape exists (two causes, one paper)

Two things create the shape. Only one of them is really our problem.

### Cause 1: positional priors baked in by pretraining data

Where does important information live in the pretraining corpus? At the start (headlines, ledes, topic sentences) and at the end (conclusions, TL;DRs). Rarely in the middle.

The model observes this pattern billions of times and learns *"the middle is where filler goes."* Then at inference time, when the answer really is in the middle, the model has an inductive bias that says otherwise. So it fabricates.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Ask any journalist where the important sentence in a story lives — they'll say the lede or the kicker. Ten thousand years of that lesson is what the model has been trained on. It's not stupid. It's exactly as biased as its training data.</p>
</aside>

### Cause 2: softmax dispersion at very long distances

The other cause is architectural. Softmax attention normalizes across all positions, and at very long context lengths the distribution flattens toward uniform. Combined with positional-encoding artifacts (RoPE, ALiBi behave differently), attention weight *tends* to concentrate on the earliest and latest positions.

This one is harder to fix without touching the architecture. Our paper punts on it.

## What we did — the data-side fix

Three pieces. None are architectural. All are cheap.

### 1. Center-emphasized fine-tuning corpus

Build a dataset where the answer is deliberately placed in the middle 60% of a long context. Not the first 20%. Not the last 20%. The middle.

Sources are synthetic (long documents constructed by concatenating natural passages) plus filtered natural (a small subset of NarrativeQA and similar). The construction is straightforward; the *quality control* is not — you have to check that the "answer at position X" is genuinely answerable from position X alone, not accidentally leaked from surrounding context.

### 2. Position-balanced sampling

Even with a good corpus, batch composition can re-introduce the bias. If a batch happens to have most of its answerable-facts at the start, the model regresses on that batch.

Fix: stratify. Every batch has a fixed fraction of examples at each depth bucket. Middle buckets are always represented. Simple, boring, works.

### 3. Depth-conditional evaluation

Here's the trick most papers get wrong. If your training loss is a single number, and your eval is a single number, you cannot *see* the U-shape flattening during training. Aggregate accuracy averages the improvement over positions.

We instrumented training with a depth-conditional eval that plotted the U-shape in real time. That's how we knew our recipe was actually flattening the curve and not just trading endpoints for middle.

<aside class="callout callout--warning">
  <div class="callout__label">Instrument what you're optimizing for</div>
  <p>If your goal is "raise accuracy in the middle without hurting the ends," your training loop must be able to see both. A single aggregate number can hide the fact that you're just moving accuracy around, not adding it.</p>
</aside>

## The trade-off, told honestly

The endpoints of the curve dip a couple of points after fine-tuning. That's real, and I want to name it.

We are trading peak accuracy at the start of context (95% → 92%) for accuracy at the middle (54% → 88%). That's a great trade — the *worst case* of the model moved from unusable to actually useful, and the *best case* barely moved. If you're deploying an LLM in a long-context product, that's the shape you want.

It is not a strictly better model everywhere. It is a strictly better model *where it matters*.

## What the fix does NOT solve

Push context length far enough beyond what we tested and softmax dispersion reasserts itself. Toggle "longer context" in the demo and watch a small residual dip at the true center. That's the second cause — the architectural one — reintroducing itself.

<aside class="callout callout--key">
  <div class="callout__label">The honest limit</div>
  <p>Our fix is data-side. It works out to about 32k, degrades gracefully to 128k, and won't help you at 1M. At extreme context lengths you need architectural changes — sparse attention, linear-time attention, or SSM-style state. Data alone is not enough.</p>
</aside>

## What I'd do next

- **Position-conditional learned encodings.** Instead of static positional priors, learn a residual that specifically counteracts the center-of-context dip. There is reason to think this can be done at pre-training, not just fine-tuning.
- **Combine with sparse-attention long-context architectures.** Our data fix and their architectural fix should compose. Somebody with a big model and a lot of compute should try.
- **Extend to multi-document contexts.** Real users don't paste a single long document; they paste eight medium ones. The lost-in-the-middle effect on multi-document context is under-studied and I suspect worse.

Full paper: [SSRN](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5877962).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noopener"><div class="research-card__title">Lost in the Middle: How Language Models Use Long Contexts</div><div class="research-card__authors">Liu et al. · TACL 2024 · the paper that named the effect</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noopener"><div class="research-card__title">RoFormer: Enhanced Transformer with Rotary Position Embedding</div><div class="research-card__authors">Su et al. · 2021 · RoPE, one of the positional encoding schemes at play</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noopener"><div class="research-card__title">Train Short, Test Long: Attention with Linear Biases (ALiBi)</div><div class="research-card__authors">Press, Smith, Lewis · ICLR 2022 · ALiBi</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2404.02258" target="_blank" rel="noopener"><div class="research-card__title">Mixture-of-Depths</div><div class="research-card__authors">Raposo et al. · 2024 · another angle on long-context efficiency</div></a></li>
  <li><a class="research-card" href="https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5877962" target="_blank" rel="noopener"><div class="research-card__title">Never Lost in the Middle Again</div><div class="research-card__authors">Challagundla et al. · 2025 · this paper</div></a></li>
</ul>
