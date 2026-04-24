---
layout: post-article
title: "What Is Mechanistic Interpretability?"
date: 2026-04-24
permalink: /posts/what-is-mechanistic-interpretability/
excerpt: "Two ways to understand a magic trick. Most AI research does one. Mechanistic interpretability does the other, and it's a completely different game."
series: "Phase 1 · Foundations"
series_index: 2
series_index_prev: 1
series_index_next: 3
series_total: 31
read_time_label: "16 min read"
tags: [mechanistic-interpretability, foundations]
---

Okay, quick thought experiment. Two ways to figure out a magic trick.

One, you watch it a thousand times, log what works, and build a model of the *behaviour*. Two, you get the magician to show you their hands, frame by frame, and see the *mechanism*.

Most AI research is watching the trick. Mechanistic interpretability is grabbing the magician's wrist. Completely different game, and we're about to play it.

---

## What "mechanistic" actually means

"Interpretability" in AI is a big tent. It covers:

- **Saliency maps.** "Which pixels made the model look at this image?"
- **LIME / SHAP.** "Which features statistically mattered for this prediction?"
- **Probing.** "Does this layer seem to know what country a city is in?"

All of these are real, useful, shipped-in-production tools. They tell you *what* the model seems to do. They just don't tell you *how it does it*. Both questions matter, they're just different questions.

Mechanistic interpretability asks a totally different question:

<aside class="callout callout--key">
  <div class="callout__label">The MI question</div>
  <p>What is the actual algorithm running in this model? What computations are happening? What concepts does it represent? How do they connect?</p>
</aside>

Goal: **reverse engineering**. Not "poke it and see what falls out." Open it up and read the code.

{% include demos/behavioral-vs-mechanistic.html %}

## Three things MI wants to find

### Features. What concepts does the model represent?

Every neural network learns to represent stuff. *"Dog". "Curved line". "Toxic language". "The year 1990".* These are **features**. MI wants to find them, label them, and figure out how the model uses them.

Okay so here's the idea that broke my brain the first time: a feature is a *direction in activation space*. Not a neuron. A direction.

If you've got 768 neurons in a layer, one feature might be "cat" = (0.3, -0.8, 0.1, ...). A specific combination of activations across those 768 dimensions. When cat-like inputs show up, the activations shift in the "cat direction". Read that again, it's weird and it's important.

Features aren't neurons. They're *directions made of neurons*.

Vision models have well-catalogued features now:

- Low-level: edge, colour, texture detectors
- Mid-level: curves, corners, eyes
- High-level: faces, cars, specific dog breeds, even "pose"

Language models have them too. "This is a URL". "Subject of the current clause". "We're inside quotation marks". "This code needs a closing bracket". Recent sparse-autoencoder work has surfaced *millions* of these.

### Circuits. How do features connect?

A **circuit** is a small sub-network that computes a specific function. Like: *"this combination of 4 attention heads, working together, detects whether someone's referring back to a noun from two sentences ago."* MI wants to find these and understand them completely.

{% include demos/attention-heads.html %}

The example above isn't hypothetical. Those patterns are the kind of thing researchers have actually identified in GPT-2. An *induction head* is a specific two-attention-head circuit that was the first full circuit ever characterised in a transformer. It's how models do in-context learning: see a pattern, complete the pattern.

A real circuit description reads like: "Layer 7 Head 4 writes information about the previous token into the residual stream; Layer 9 Head 2 reads that information and moves it forward whenever the attention pattern matches." Dry, mechanical, reproducible. That's the vibe. I know, *so* romantic.

### Universality. Do the same features show up in different models?

If GPT and Gemini and a vision model all develop the same "curve detector" circuit, that's a big deal. Not arbitrary. More like *natural categories that emerge from learning*.

Some universality has already been found. InceptionV1 and CLIP (two very different vision models) both develop curve detectors with similar structure. Multiple transformer families grow induction heads at roughly the same training stage. The space of useful features, given natural data, seems constrained. There are "correct" things to learn, and good models find them.

Big if true. Means MI discoveries should generalise: understand how one model does X, other models probably do X similarly.

## Why this is genuinely hard

A modern LLM has:

- ~70 billion parameters (for medium-sized)
- thousands of neurons per layer
- dozens of layers
- no labels on anything

The model didn't ship with documentation. Its internal structure isn't organised in human-friendly ways. Two specific problems make life difficult:

<aside class="callout callout--warning">
  <div class="callout__label">Polysemanticity</div>
  <p>One neuron responds to <em>"cats AND legal documents AND the digit 7."</em> Because the model is cramming a lot of concepts into a small space. You can't just point to "the cat neuron", it's also the lawyer neuron, and the 7 neuron.</p>
</aside>

{% include demos/polysemantic-neuron.html %}

<aside class="callout callout--warning">
  <div class="callout__label">Superposition</div>
  <p>The model packs <strong>more concepts than it has neurons</strong>, by storing them at angles to each other. Many conversations happening on the same frequency, overlapping. Features don't live cleanly in single neurons; they live across combinations.</p>
</aside>

Superposition is the technical reason for polysemanticity. 512 neurons, 5000 concepts, you have to share. The clever (and spooky) thing: the model stores concepts at *angles* to each other in high-dim space. A careful reader can disentangle them approximately. Not cleanly. Not perfectly. Just well enough.

Which is why naive interpretation ("what does neuron 734 do?") gives confused answers. The right question is "what are the *feature directions* in this 512-dim space?" And that's what **sparse autoencoders** (Phase 4) finally cracked.

## The analogy that captures it

Think about a compiled program. You have the binary. Millions of 1s and 0s. You can run it. You can test it. You can't read the source directly.

A reverse engineer's job: take the binary, reconstruct the algorithms. "Ah, this block is doing sorting. That block is encryption. Here's where the password check is."

That's mechanistic interpretability. Except the binary is a neural network, and the "source code" we're looking for is a description of what algorithms it learned.

The exciting part: unlike compiled software, the algorithms the network learned *weren't designed by a human*. We don't know what we'll find. Some of what has been found:

- A two-head circuit that copies information across context (induction heads)
- A multi-head circuit that handles "John gave Mary the book. Mary gave it back to ___" (the IOI circuit)
- A mod-arithmetic algorithm that emerges suddenly during training (grokking)
- A feature that fires for "the concept of the assistant persona, including its restrictions and imprisonment metaphors" (Claude 3 Sonnet)

Every one of those was a surprise. We're doing discovery on artefacts we built but didn't design.

## Where behavioural methods fit

They aren't wrong. They're complementary. A good interpretability investigation usually looks like:

1. **Behavioural tools find *what*.** SHAP values, saliency maps, probing. Cheap and fast. They give you a hypothesis: "the model seems to care about token X here."
2. **Mechanistic tools explain *why*.** Activation patching, circuit extraction, ablation. Slow and expensive. They give you a causal account.
3. **Mechanistic validation.** Disable the alleged circuit, see if the behaviour breaks. Insert it elsewhere, see if it reproduces.

The mechanistic part is what turns a suggestive correlation into a rigorous claim. That's the specific thing MI adds.

## What success looks like

Finish line. We've succeeded when:

1. We can identify every feature a model represents.
2. We can trace every computation: for a given input, here's the exact path of information through the network.
3. We can predict failures before they happen: "this model will fail on inputs with property X because its circuit for handling X is weak."
4. We can verify: "this model genuinely doesn't have a deceptive capability hidden inside, because we've checked the circuits."

Not there yet. Maybe 5% of the way, if I'm being generous. But the progress in the last three years has been kind of wild. That's what Post 3 is about.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/mech-interp-essay/index.html" target="_blank" rel="noopener"><div class="research-card__title">Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases</div><div class="research-card__authors">Olah, C. · 2022 · clearest definition of MI</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/zoom-in/" target="_blank" rel="noopener"><div class="research-card__title">Zoom In: An Introduction to Circuits</div><div class="research-card__authors">Olah, C. et al. · Distill, 2020 · the founding paper</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson, C. et al. · Anthropic, 2022</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/toy_model/index.html" target="_blank" rel="noopener"><div class="research-card__title">Toy Models of Superposition</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2022 · where superposition was formalised</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" target="_blank" rel="noopener"><div class="research-card__title">Towards Monosemanticity</div><div class="research-card__authors">Bricken, T. et al. · Anthropic, 2023 · SAE breakthrough</div></a></li>
  <li><a class="research-card" href="https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability" target="_blank" rel="noopener"><div class="research-card__title">200 Concrete Open Problems in Mechanistic Interpretability</div><div class="research-card__authors">Nanda, N. · 2022 · the open frontier</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2018/building-blocks/" target="_blank" rel="noopener"><div class="research-card__title">The Building Blocks of Interpretability</div><div class="research-card__authors">Olah, C. et al. · Distill, 2018 · visual reference</div></a></li>
</ul>
