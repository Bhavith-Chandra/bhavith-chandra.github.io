---
layout: post-article
title: "Who's Doing This, and Why Now?"
date: 2026-02-11
permalink: /posts/whos-doing-this-and-why-now/
excerpt: "A quick tour of the labs, the people, and the 8-year sprint that took mechanistic interpretability from a weird hobby to a central safety bet."
read_time_label: "12 min read"
---

Here's a fun piece of history.

In the 1950s a small group of biologists set out to do something that sounded, at the time, a little mad: describe *exactly* how a single neuron in a living animal works. Not the nervous system. Not neurons in general. One neuron. What signal, in response to what input, with what timing.

That one obsession grew into all of modern neuroscience.

We're at something like that moment with AI. Early 1950s, give or take. Handful of people. Wild energy. Real progress.

---

## The players

MI is small. Delightfully small. Handful of central teams, a growing cloud of academics, and a surprising number of solo researchers doing real work. The field is tight-knit enough that you'll see the same names on most of the landmark papers. Which is a huge opportunity, by the way. Here's the rough map.

{% include demos/researcher-grid.html %}

### Anthropic Interpretability Team

The team most associated with the current wave. Founded by **Chris Olah**, one of the original Distill.pub researchers. They produced:

- The circuits framework (2020)
- A Mathematical Framework for Transformer Circuits (2021)
- In-context Learning and Induction Heads (2022)
- Toy Models of Superposition (2022)
- Towards Monosemanticity (2023), the SAE breakthrough
- Scaling Monosemanticity (2024), applied to Claude 3 Sonnet

Explicitly motivated by AI safety: *if we can't understand what the model has learned, we can't verify it's safe*. Most of their work lives on [transformer-circuits.pub](https://transformer-circuits.pub/). Beautifully written, often interactive. Worth reading linearly if you want the field's intellectual history. They publish rarely and thoroughly; each release is a small event.

### Neel Nanda · DeepMind + independent

Probably the most accessible voice in MI. Produces extremely well-explained research, runs workshops, wrote *200 Concrete Open Problems*. His work on **grokking** and **circuits** is foundational. His **TransformerLens** library is what most researchers actually use.

If Anthropic's style is "careful flagship releases", Nanda's style is "constant, generous, hands-on output". YouTube walkthroughs. Twitter threads explaining new papers. The ARENA curriculum. Paper-replication exercises. A huge fraction of the current generation of MI researchers got started by working through his material.

### Academic research groups

Short, non-exhaustive list:

- **MIT (Jacob Andreas, David Bau).** Language-model representations, model editing (ROME, MEMIT).
- **Berkeley (Jacob Steinhardt).** Training dynamics, emergent capabilities, benchmarks.
- **Harvard / Kempner.** Broader ML-theory work, feature learning.
- **Northeastern (Bau Lab).** Causal tracing, factual knowledge localisation.
- **NYU, Princeton, CMU.** Various smaller efforts on probing, circuits, evaluation.

Academia contributes mostly *methods* and *theory* rather than scale, and that's genuinely load-bearing. Labs don't always have Claude to poke at, but they have the time to develop the cleaner formal tools everyone ends up using.

### Independent researchers

One of the beautiful things about MI: you can contribute without institutional affiliation. The [Alignment Forum](https://www.alignmentforum.org/), LessWrong, and [EleutherAI](https://www.eleuther.ai/) have produced real research from independent contributors. The field is young and open. Novel findings from unknown researchers show up there regularly and get taken seriously.

A few that matter:

- [Apart Research](https://www.apartresearch.com/). Hackathons and distributed research.
- [FAR AI](https://www.far.ai/). Independent lab, explicit safety focus.
- [Redwood Research](https://www.redwoodresearch.org/). Circuit-level studies; the IOI paper came from here.

Not at a lab and want to get published? This is the culture you're entering. More forum than conference.

## The tools you'll actually use

{% include demos/mi-tools.html %}

Three rough tracks for picking one:

1. **Learning MI from zero.** Start with ARENA (structured exercises), then pick up TransformerLens and follow Neel Nanda's tutorial to replicate IOI or induction heads on GPT-2. You'll hit every major concept.
2. **SAE research.** SAELens plus Neuronpedia. Train a small autoencoder, find interesting features, label them, compare against published lists.
3. **Frontier model work.** nnsight is the only realistic option outside the frontier labs themselves, because it gives you intervention access to hosted large models.

## The timeline of discoveries

Click a milestone.

{% include demos/discovery-timeline.html %}

Headline: the field went from "curves detectable in vision models" (2019) to "millions of features decomposed in production Claude" (2024) in five years. That compression is almost unprecedented in ML research.

## Why now, the urgency

Not purely academic. There's a race.

<aside class="callout callout--warning">
  <div class="callout__label">The race</div>
  <p>AI is getting more powerful <strong>faster</strong> than we're gaining understanding of it.</p>
</aside>

Figure out how to interpret AI systems *before* they're making high-stakes autonomous decisions, we can verify they're safe. Miss that window, interpretability keeps lagging capability, we're in a much harder position.

This is why Anthropic, DeepMind, and others put serious resources into it. Not just interesting science. One of the most important technical bets for making AI go well. Concrete things MI would enable if it were solved:

- Detecting deceptive behaviour before deployment, not during incidents.
- Auditing a model for dangerous capabilities (bioweapons knowledge, cyber tools, persuasion) mechanistically, not just behaviourally.
- Explaining individual decisions in high-stakes applications (medicine, law, hiring).
- Verifying that fine-tuning didn't introduce hidden backdoors.

None of this requires fully solving MI. Partial progress helps. That's why the field is so alive right now: every small win is immediately useful.

## You can do this too

Field is young enough that smart outsiders contribute. Methods are learnable. Papers are readable. Code is open source.

<aside class="callout callout--key">
  <div class="callout__label">If you're starting today</div>
  <p>The on-ramp is clearer than you think:</p>
  <ol>
    <li>Skim <a href="https://distill.pub/2020/circuits/zoom-in/">Zoom In</a> and <a href="https://transformer-circuits.pub/2021/framework/index.html">the Mathematical Framework</a>. Even if the math is hard, read through for shape.</li>
    <li>Pick up TransformerLens. Follow <a href="https://www.youtube.com/@neelnanda2469">Neel Nanda's walkthrough</a> and replicate a small result.</li>
    <li>Do one exercise from ARENA. Ship something crappy.</li>
    <li>Write it up on the Alignment Forum. People will respond.</li>
  </ol>
  <p>Six weeks of focused effort and you'll understand more of the frontier than 99% of ML engineers.</p>
</aside>

That's the *why*. The *how* is a longer story: what a feature actually is, what a circuit actually is, how we find them. It all starts with the raw materials: neurons, weights, layers, and training.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://distill.pub/2019/activation-atlas/" target="_blank" rel="noopener"><div class="research-card__title">Activation Atlas</div><div class="research-card__authors">Carter, S. et al. · Distill, 2019</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/zoom-in/" target="_blank" rel="noopener"><div class="research-card__title">Zoom In: An Introduction to Circuits</div><div class="research-card__authors">Olah, C. et al. · Distill, 2020</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson, C. et al. · Anthropic, 2022</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2211.00593" target="_blank" rel="noopener"><div class="research-card__title">Interpretability in the Wild · a Circuit for IOI in GPT-2</div><div class="research-card__authors">Wang, K. et al. · 2022</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2301.05217" target="_blank" rel="noopener"><div class="research-card__title">Progress measures for grokking via mechanistic interpretability</div><div class="research-card__authors">Nanda, N. et al. · 2023</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" target="_blank" rel="noopener"><div class="research-card__title">Towards Monosemanticity</div><div class="research-card__authors">Bricken, T. et al. · Anthropic, 2023</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html" target="_blank" rel="noopener"><div class="research-card__title">Scaling Monosemanticity</div><div class="research-card__authors">Templeton, A. et al. · Anthropic, 2024</div></a></li>
  <li><a class="research-card" href="https://rome.baulab.info/" target="_blank" rel="noopener"><div class="research-card__title">ROME · Locating and Editing Factual Associations in GPT</div><div class="research-card__authors">Meng, K. et al. · MIT / Northeastern, 2022</div></a></li>
</ul>
