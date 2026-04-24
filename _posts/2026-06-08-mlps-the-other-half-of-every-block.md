---
layout: post-article
title: "MLPs: The Other Half of Every Block"
date: 2026-06-08
permalink: /posts/mlps-the-other-half-of-every-block/
excerpt: "Attention gets the press. But two-thirds of a transformer's parameters live in the MLPs. They are the model's filing cabinet, where facts, transformations, and unreasonably specific patterns end up stored as little key-value pairs."
read_time_label: "12 min read"
accent: amber
math: true
---

If attention is the part of the model that *retrieves* information from context, the MLP is the part that *stores* information from training. About two-thirds of a transformer's parameters sit in the MLPs. If you want to know where the model's knowledge lives, you point at these.

For a long time the MLP got short shrift in the interpretability literature. Attention was sexy. Attention had named heads (induction! copy! name-mover!). MLPs were a featureless wall of weights, a single number times another single number, repeated 3,072 times.

That changed around 2021, when [Geva et al.](https://arxiv.org/abs/2012.14913) made a beautifully simple argument: an MLP layer can be read as a **key-value memory**. Each neuron is a *key* (a pattern in the residual stream) paired with a *value* (a vector to write back). Suddenly MLPs had structure. Suddenly we could name them.

Let's make that concrete.

---

## Poke at some neurons first

{% include demos/neuron-inspector.html %}

Cycle through the dropdown. Read the contexts. Notice that some neurons are crisply about one thing (Python keywords, capital letters, France-related facts) and others are weirdly about four unrelated things at once. That second case is **superposition**, which I'll get to. But first, the basic structure.

## What an MLP is, in one paragraph

A transformer MLP block has two linear layers separated by a non-linearity:

$$\text{MLP}(x) = W_{\text{out}} \cdot \sigma(W_{\text{in}} \cdot x)$$

<div class="math-translate">Project the residual vector up to a much wider dimension, apply a non-linearity (in modern LLMs, GELU or SwiGLU, but always a smooth squash-or-pass-through gate), and project back down. The output gets added to the residual stream.</div>

The "wider dimension" is typically $4 \times d_{\text{model}}$. So GPT-2 small has $d_{\text{model}} = 768$ and an MLP intermediate dimension of 3,072. Llama 3 8B has $d_{\text{model}} = 4096$ and intermediate around 14,336. Each of those intermediate units is a **neuron**.

Three things to notice:

1. **MLPs operate on each position independently.** Unlike attention, there is no mixing across token positions. Each token's MLP runs on its own residual vector, in parallel.
2. **The wider intermediate dimension matters.** That's where the actual storage capacity lives. 3,072 neurons in a single block of GPT-2, multiply by 12 blocks and you get 36,864 distinct units to study.
3. **The non-linearity is what makes it a feature detector.** Without GELU, two stacked linear layers collapse into a single linear layer and the MLP can't represent anything interesting. The non-linearity makes each neuron a thresholded-gate: "fire if the input matches my pattern, otherwise don't."

## Reading MLPs as key-value memories

Here is the move. The first matrix, $W_{\text{in}}$, has shape $[d_{\text{model}}, d_{\text{ffn}}]$. Each *column* is a vector in residual-stream space, call it the **key** for neuron $n$. The dot product $W_{\text{in}}[:, n] \cdot x$ measures *how much $x$ matches that key*.

The second matrix, $W_{\text{out}}$, has shape $[d_{\text{ffn}}, d_{\text{model}}]$. Each *row* is a vector in residual-stream space, call it the **value** for neuron $n$. When neuron $n$ fires (its activation is large), it writes that value (scaled by the activation) back to the residual stream.

So the MLP layer's update can be rewritten as:

$$\text{MLP}(x) = \sum_{n=1}^{d_{\text{ffn}}} \sigma(\langle x, k_n \rangle) \cdot v_n$$

<div class="math-translate">For each of the (thousands of) neurons, compute how much the input matches that neuron's key, gate that match through the non-linearity, and add the corresponding value vector to the output. Sum over all neurons.</div>

This is a **soft key-value lookup**. The MLP looks up a giant database keyed by patterns and writes back the corresponding payloads. Every neuron contributes a tiny bit. The aggregate output is the sum.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The key-value framing is what made MLPs interpretable. Once you can read each neuron's key (its trigger pattern) and value (its written contribution), you can label neurons. Each one becomes a tiny "if you see X, write Y." This is the basis for ROME (Meng 2022), which can edit specific facts in a model by surgically modifying the relevant key-value pair.</p>
</aside>

## What kinds of patterns do real neurons learn?

A few archetypes show up reliably across transformers:

### Surface-feature neurons

Early-layer neurons that fire on simple lexical patterns: capital letters, punctuation, specific morphemes, code tokens. The "Python-keyword neuron" in the demo is one. They pre-process the stream into useful tags that downstream blocks consume.

### Syntactic neurons

Mid-layer neurons that fire after specific grammatical patterns, possessives, definite articles, sentence beginnings. They write contributions that bias the next-token distribution toward syntactically valid continuations.

### Factual-recall neurons

These are the famous ones. A small set of neurons in mid-to-late MLPs encode specific facts. The "France/Paris" neuron in the demo is a stylised version. [Meng et al., 2022](https://rome.baulab.info/) showed you can locate a specific factual association ("the Eiffel Tower is in Paris") to a small group of neurons in a small range of layers, and edit it surgically. Suddenly the model thinks the Eiffel Tower is in Rome. (The model insists on this firmly. It's funny and slightly horrifying.)

### Abstract pattern neurons

Late-layer neurons that fire on more semantic patterns: sentiment shifts, sarcasm, irony, discourse markers. These are harder to characterise cleanly because the patterns themselves are abstract.

### Junk drawer

A nontrivial fraction of neurons just don't have a clean interpretation when you look at top-activating contexts. Sometimes they're polysemantic (next section). Sometimes they're doing something the literature hasn't named yet. Don't be alarmed if a randomly chosen neuron looks confusing.

## Polysemanticity, the inconvenient truth

Here's the thing nobody mentions in their first introduction to MI: **most real neurons are not monosemantic.**

The "Python-keyword neuron" looks clean. The "polysemantic mixed bag" neuron in the demo also looks clean, at exactly four totally different things. DNA letters. Car brands. Weekday names. Latin botanical names. There is no single concept that ties these together. The neuron just happens to fire strongly on all of them.

Why? Because a transformer needs to represent *more concepts than it has neurons*. GPT-2 small has 36,864 MLP neurons total. The number of distinct concepts it has learned to distinguish is far larger than that. So the model packs concepts on top of each other. This is called **superposition**.

The mechanism: features are stored as *directions* in the high-dimensional residual stream, not as single coordinates. Two features that are unlikely to ever co-occur in the same input can share the same direction (more or less) without interfering, because when one fires, the other isn't relevant. The non-linearity in the MLP lets the model *unpack* superposed features by attending to whichever one is contextually present.

Practical consequences:

- Most individual neurons look polysemantic when you read their top-activating examples.
- The "real" features are linear combinations of neurons, not individual ones.
- To recover monosemantic features, you need a tool that can decompose a high-dimensional space into its underlying directions, even when those directions outnumber the dimensions.

That tool turned out to be **sparse autoencoders**, which is the punchline of the whole second half of the modern interpretability era. We'll meet them properly later. For now, just know:

> When you see a polysemantic neuron, you're not looking at the model's actual feature. You're looking at a tangled superposition of multiple features that the SAE can untangle.

## Direct logit attribution, MLP edition

Same idea as for attention. Because each MLP block writes to the residual stream additively, you can ask: *for a specific output prediction, how much did this MLP contribute?*

Concretely, take the "Paris" neuron in the demo. Its output direction is roughly aligned with the unembedding vector for the token " Paris". When the neuron fires (because the residual stream encodes something like "capital of France"), it adds a vector to the stream that bumps the logit for " Paris" upward. The DLA contribution of this single neuron to the prediction of " Paris" is positive and measurable.

Run that analysis for every neuron in every layer, and you can decompose any prediction into "which neurons pushed the answer in which direction." That's a great deal of what circuit-level interpretability does in practice.

## Why MLPs are *where* the knowledge is

Stepping back. Why do we believe MLPs store factual knowledge, while attention does context routing?

Three pieces of evidence:

1. **Parameter count.** MLPs hold roughly two-thirds of the weights. If knowledge is in weights, statistically most of it is in MLPs.
2. **Editing experiments.** ROME and MEMIT can edit a specific fact (say "the Eiffel Tower is in Paris" → "in Rome") by tweaking only a small number of MLP neurons in mid layers. Editing attention parameters does not produce the same effect.
3. **Probing.** When researchers probe for the presence of factual information at intermediate layers, the signal jumps sharply at MLP layers, not at attention layers. The MLP is where information gets *added* to the stream.

A clean way to internalise it: **attention moves information around. MLPs add new information.** Both contribute to the final answer, but in different ways. This is also why MLPs at different depths specialise, early MLPs add surface features, mid-layer MLPs add facts and syntax, late MLPs add abstract semantics.

## The thing nobody tells you about GELU

A small implementation note that becomes important if you read MI papers. The non-linearity choice, GELU vs ReLU vs SwiGLU, affects how cleanly you can decompose the MLP.

ReLU is a hard gate: $\sigma(x) = \max(0, x)$. Either the neuron fires or it doesn't. Easy to interpret as a binary decision.

GELU and SwiGLU are softer. They produce small non-zero outputs even for slightly negative inputs. This means a neuron's "off" state still leaks a small contribution, which makes the per-neuron analysis a bit messier. In practice, you compensate by looking at activations as continuous and analysing only neurons whose activations exceed a threshold.

Modern frontier models almost all use SwiGLU. When you read papers about MLPs in Llama or Claude, that's the non-linearity in play. The key-value framing still applies, the math just has an extra gating term.

## Wrap

MLPs are 60% of a transformer's parameters. They are the model's stored memory. They can be read as key-value pairs: each neuron has a key (a pattern in the residual stream that activates it) and a value (a vector it writes back). Some neurons are clean, Python keywords, capital letters, individual facts. Many neurons are polysemantic, packing multiple concepts on top of each other through superposition.

Attention does retrieval. MLPs do storage. Together they alternate, block by block, accumulating contributions on the residual stream until the unembedding layer can read off a confident next token.

Which means: we now have all the parts. Tokens, embeddings, attention, MLPs, residual stream, unembedding. The next blog will run a full forward pass end-to-end, with real numbers, to make sure every piece you've read about is in the right place.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/2012.14913" target="_blank" rel="noopener"><div class="research-card__title">Transformer Feed-Forward Layers Are Key-Value Memories</div><div class="research-card__authors">Geva, M. et al. · 2021 · the foundational paper</div></a></li>
  <li><a class="research-card" href="https://rome.baulab.info/" target="_blank" rel="noopener"><div class="research-card__title">Locating and Editing Factual Associations in GPT (ROME)</div><div class="research-card__authors">Meng, K. et al. · 2022 · MLPs hold facts and we can edit them</div></a></li>
  <li><a class="research-card" href="https://memit.baulab.info/" target="_blank" rel="noopener"><div class="research-card__title">Mass-Editing Memory in a Transformer (MEMIT)</div><div class="research-card__authors">Meng, K. et al. · 2023 · scaling fact-editing to thousands of edits</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/toy_model/index.html" target="_blank" rel="noopener"><div class="research-card__title">Toy Models of Superposition</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2022 · why polysemanticity is rational</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" target="_blank" rel="noopener"><div class="research-card__title">Towards Monosemanticity</div><div class="research-card__authors">Bricken, T. et al. · Anthropic, 2023 · sparse autoencoders untangle the mess</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/zoom-in/" target="_blank" rel="noopener"><div class="research-card__title">Zoom In: An Introduction to Circuits</div><div class="research-card__authors">Olah, C. et al. · Distill, 2020 · vision-side, but the framing applies</div></a></li>
</ul>
