---
layout: post-article
title: "Temporal Intelligence: Thirty-Five Years of State, in One Long Look"
date: 2024-12-19
permalink: /posts/notes-on-temporal-intelligence-and-sequential-learning/
excerpt: "The Elman network from 1990 had already committed to the whole idea. Keep some state. Evolve it as new inputs arrive. Predict from it. Every generation since is engineering — the philosophy hasn't moved."
read_time_label: "10 min read"
accent: teal
---

Companion note to [Temporal Intelligence: Foundation to State-of-the-art Advancements of Sequential Learning Units and Models](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nV1hdiMAAAAJ&citation_for_view=nV1hdiMAAAAJ:roLk4NBRz8UC).

The paper is a survey. Which — full disclosure — I have complicated feelings about, because a lot of surveys read like Wikipedia articles with a bibliography. So this post is the *opinionated version*. The story I'd tell over coffee, not the neutral one I had to write for print.

The story, in one sentence: **the Elman network from 1990 had already committed to the whole idea, and everything since is engineering.**

---

## The one idea that hasn't moved

Take a stream of inputs. Keep some *state*. Evolve the state as new inputs arrive. Predict from the state. That's every sequential learner ever built.

Vanilla RNNs, LSTMs, GRUs, transformers, state-space models — they all subscribe to that commitment. What they disagree on is:

- **What form does the state take?** A vector? A cache? A structured continuous flow?
- **How does the state evolve?** Written by a gate? Grown by concatenation? Integrated by a transition matrix?
- **What compute regime is that evolution cheap in?** Sequential? Parallel? Both?

Scrub the year slider and watch the frontier move. The x-axis is calendar year; the y-axis is practical effective context length in log scale.

{% include demos/sequential-learners.html %}

There are four visible knees. Each one is a *specific idea* that unlocked the next era. Let's go through them.

---

## Knee 1: 1997 — LSTM's gate

The vanilla RNN is correct in theory: state is passed forward, gradient flows backward, learning happens. In practice, the gradient either vanishes or explodes past about ten steps. You can prove it in math. You feel it after five minutes of training.

The **LSTM** replaced multiplicative gradient flow with *additive* flow along a "cell state" that a gate can *choose* to read from or write to. Gradients now travel through addition, which is much better behaved. Suddenly hundreds of steps became tractable.

<aside class="callout callout--key">
  <div class="callout__label">Why gates matter</div>
  <p>A gate is a small learned function that decides <em>whether</em> to update part of the state on this step. It's the difference between "the state is overwritten every step" (vanilla RNN, unstable) and "the state is updated only when it's worth updating" (LSTM, stable).</p>
</aside>

Everything else stayed the same. Sequential compute. One step at a time. Slow to train, slow to deploy. But it *worked*, which none of the vanilla RNNs really did.

## Knee 2: 2014-2016 — attention

The seq2seq bottleneck. I wrote about this one [earlier](/posts/notes-on-neural-seq2seq-with-attention/) and won't repeat myself. The short version: the encoder compressed everything into one vector, the decoder read from that vector, and information got lost. Attention let the decoder *look back at the whole encoder state, per step*.

This one wasn't really a change to the state itself. It was a change to *what you're allowed to read from*. Recurrence stayed. But the idea that "the state can include everything you've ever seen, and you can query it dynamically" was going to be huge.

## Knee 3: 2017 — attention is the whole model

The transformer paper (Vaswani et al.) does one radical thing: **remove recurrence entirely**. No hidden state that gets passed forward. Instead, at every step, attend over every past step directly.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>An LSTM is a scribe who reads a paragraph and takes notes as they go. A transformer is a scribe who keeps the whole paragraph in front of them at all times and glances back whenever they need to. Slower per word if the paragraph is long, but never forgets.</p>
</aside>

The compute trade is different: attention is quadratic in sequence length, and the KV cache grows linearly. LSTMs were linear in compute, constant in state size. Transformers pay more per step and remember more per step. On the hardware of 2017, that was a great trade — GPUs love parallelism, and transformer training is embarrassingly parallel across positions in a way LSTM training just isn't.

That parallelism is why transformers ate everything for the next five years. The idea that state should be a *growing cache* rather than a *fixed vector* turned out to be the right call for the hardware of the era.

## Knee 4: 2022+ — state-space models

Then context windows started getting really long, and the quadratic-in-length attention cost stopped being cute.

**State-space models** (S4, Mamba, and family) bring recurrence back — but with the good parts of the transformer era intact. Structured continuous-time dynamics for the state. Parallel training via clever math (associative scan, convolutional view). Linear-time inference because they're recurrent underneath.

<aside class="callout callout--key">
  <div class="callout__label">The SSM bet</div>
  <p>You can have parallel training AND linear-time inference AND long-range gradients, if you're willing to constrain the state's dynamics to a specific mathematical form. That form makes some kinds of information easy to preserve and other kinds hard. The interesting research question is <em>which</em>.</p>
</aside>

The current frontier is a mix. Hybrids that alternate SSM blocks with attention blocks. Pure-attention scaled to million-token contexts. Pure-SSM competing at the language-model frontier for the first time. The bet isn't settled.

---

## What the story is actually about

Two axes, running underneath the whole thing:

- **What compute is cheap.** Sequential CPUs → recurrent networks. Parallel GPUs → transformers. Long-context inference → SSMs. Every era's dominant architecture is downstream of what the hardware makes free.
- **How you encode time-dependence.** Explicitly gated (LSTM), implicitly via a growing cache (transformer), or through structured dynamics (SSM). Each makes different things easy and different things hard.

If you strip out the marketing, "which architecture is best?" is really "which pairing of state-form × compute-regime × task-mix fits my constraints?" There is no universal winner. There will not be a universal winner.

## Why the story isn't over

The current sequence models are stunningly good at *predict the next token in a stream where recent context matters most*. They're still not great at:

- **Very long-range dependencies** where the relevant signal is a specific event a million tokens ago. Attention can theoretically reach it. Attention often doesn't.
- **Hierarchical time.** Events on wildly different timescales, in the same sequence. Human language has this — a novel and a tweet are the same "language" but need different temporal reasoning. Current models don't differentiate.
- **World-model rollouts.** State that's a good enough summary that you can *plan* against it, not just predict against it. This is the thing I care most about right now.

<aside class="callout callout--warning">
  <div class="callout__label">Where I think this is heading</div>
  <p>The next era isn't about even longer contexts. It's about state that's <em>meaningful</em> — where you can inspect it, plan against it, and detect when it's about to be wrong. That's what world models are for. That's where interpretability, sequence modeling, and safety start to meet in the middle.</p>
</aside>

That's the segue to the papers I'm writing now — [Never Lost in the Middle](/posts/notes-on-never-lost-in-the-middle-again/) and [One Lens, Many Worlds](/posts/notes-on-one-lens-many-worlds/) — both of which are asking questions about state that the survey couldn't quite reach.

Full paper on [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nV1hdiMAAAAJ&citation_for_view=nV1hdiMAAAAJ:roLk4NBRz8UC).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1" target="_blank" rel="noopener"><div class="research-card__title">Finding Structure in Time</div><div class="research-card__authors">Elman · Cognitive Science, 1990 · the Elman network</div></a></li>
  <li><a class="research-card" href="https://www.bioinf.jku.at/publications/older/2604.pdf" target="_blank" rel="noopener"><div class="research-card__title">Long Short-Term Memory</div><div class="research-card__authors">Hochreiter & Schmidhuber · 1997 · LSTM</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener"><div class="research-card__title">Neural Machine Translation by Jointly Learning to Align and Translate</div><div class="research-card__authors">Bahdanau, Cho, Bengio · ICLR 2015 · attention</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani et al. · NeurIPS 2017 · the transformer</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2111.00396" target="_blank" rel="noopener"><div class="research-card__title">Efficiently Modeling Long Sequences with Structured State Spaces (S4)</div><div class="research-card__authors">Gu, Goel, Ré · ICLR 2022 · the SSM revival paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener"><div class="research-card__title">Mamba: Linear-Time Sequence Modeling with Selective State Spaces</div><div class="research-card__authors">Gu & Dao · 2023 · Mamba</div></a></li>
  <li><a class="research-card" href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nV1hdiMAAAAJ&citation_for_view=nV1hdiMAAAAJ:roLk4NBRz8UC" target="_blank" rel="noopener"><div class="research-card__title">Temporal Intelligence: Foundation to State-of-the-art Advancements</div><div class="research-card__authors">Challagundla · 2024 · this paper</div></a></li>
</ul>
