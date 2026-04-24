---
layout: post-article
title: "MLPs: The Other Half of Every Block"
date: 2026-06-08
permalink: /posts/mlps-the-other-half-of-every-block/
excerpt: "MLPs hold ~⅔ of a transformer's parameters and act as key-value memories: each neuron is a learned (key, value) pair that adds to the residual stream when the key matches. Where most factual knowledge lives."
read_time_label: "12 min read"
accent: amber
math: true
---

The **MLP** (multi-layer perceptron, or feed-forward network) is the second sublayer in every transformer block. It contains ~⅔ of the model's parameters and operates position-wise: each token's residual stream vector is processed independently.

This post defines the MLP, derives the **key-value memory** interpretation ([Geva et al., 2021](https://arxiv.org/abs/2012.14913)) that underlies most modern MLP interpretability, covers neuron archetypes and superposition, and connects the framework to factual editing (ROME/MEMIT) and sparse autoencoders.

---

## Demo: neuron activations

{% include demos/neuron-inspector.html %}

Cycle through neurons. Some are monosemantic (Python keywords, capital letters, France-related contexts). Some are **polysemantic**, firing on multiple unrelated concepts. The polysemantic case is explained by superposition (below).

## Definition

For input $x \in \mathbb{R}^{d_\text{model}}$ at one position, a transformer MLP computes:

$$\text{MLP}(x) = W_\text{out}\, \sigma(W_\text{in}\, x + b_\text{in}) + b_\text{out}$$

where:
- $W_\text{in} \in \mathbb{R}^{d_\text{ffn} \times d_\text{model}}$
- $W_\text{out} \in \mathbb{R}^{d_\text{model} \times d_\text{ffn}}$
- $\sigma$ is a non-linearity (GeLU, ReLU, or in modern models SwiGLU)
- $d_\text{ffn} = 4 \cdot d_\text{model}$ is the standard ratio

Sizes:

| Model | $d_\text{model}$ | $d_\text{ffn}$ | Neurons per block |
|---|---|---|---|
| GPT-2 small | 768 | 3,072 | 3,072 |
| GPT-2 XL | 1,600 | 6,400 | 6,400 |
| Llama 3 8B | 4,096 | 14,336 | 14,336 |
| GPT-3 175B | 12,288 | 49,152 | 49,152 |

Three properties:

1. **Position-wise.** No mixing across token positions. Operates in parallel on each token's residual vector.
2. **Up-projection then down-projection.** Hidden width $4\times$ wider than the residual stream. Storage capacity scales with $d_\text{ffn}$.
3. **Non-linearity is essential.** Without $\sigma$, two stacked linear layers collapse to one and the MLP cannot represent any nonlinear pattern.

## Key-value memory interpretation

Decompose $W_\text{in}$ row-wise and $W_\text{out}$ column-wise:

- Row $n$ of $W_\text{in}$, written $k_n^\top$, is a vector in $\mathbb{R}^{d_\text{model}}$: the **key** of neuron $n$.
- Column $n$ of $W_\text{out}$, written $v_n$, is a vector in $\mathbb{R}^{d_\text{model}}$: the **value** of neuron $n$.

Then:

$$\text{MLP}(x) = \sum_{n=1}^{d_\text{ffn}} \sigma(k_n^\top x + b_n)\, v_n$$

The MLP is a sum of $d_\text{ffn}$ scaled value-vectors, where each scaling coefficient is a non-linearly gated dot product of $x$ with the corresponding key.

Equivalently:
- The key $k_n$ tests whether $x$ matches a specific pattern (large $k_n^\top x$ ⇒ match).
- The non-linearity gates: only neurons whose match exceeds threshold contribute.
- Each contributing neuron writes its value $v_n$ to the residual stream, scaled by activation.

This is a soft, sparse key-value lookup over a learned database of $d_\text{ffn}$ entries per block. ([Geva et al., 2021](https://arxiv.org/abs/2012.14913))

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The key-value framing reduces MLP interpretability to two independent questions per neuron: (1) what input pattern activates k<sub>n</sub>? (2) what does v<sub>n</sub> write to the residual stream? Both are tractable. Top-activating dataset examples answer (1); projecting v<sub>n</sub> onto W<sub>U</sub> answers (2).</p>
</aside>

## Neuron archetypes

Empirically, MLP neurons fall into recurring categories:

### Surface-feature neurons (early layers)

Fire on lexical patterns: capital letters, punctuation, specific morphemes, code-syntax tokens. Their values write tags downstream blocks consume.

### Syntactic neurons (mid layers)

Fire after grammatical patterns: possessives, definite articles, sentence-initial positions. Values bias the next-token distribution toward syntactically valid continuations.

### Factual-recall neurons (mid-to-late layers)

Encode specific facts. [Meng et al. (2022, ROME)](https://arxiv.org/abs/2202.05262) demonstrated that "the Eiffel Tower is in Paris" can be located to a small set of neurons in mid layers and surgically edited (so the model claims the Eiffel Tower is in Rome) by modifying $W_\text{out}$ columns at those positions.

### Abstract / semantic neurons (late layers)

Fire on higher-level patterns: sentiment, sarcasm, discourse markers. Harder to characterize from top-activating examples alone.

### Uninterpretable from top examples

A non-trivial fraction of neurons have no clean concept-level description. Often these are polysemantic.

## Polysemanticity and superposition

Most real neurons are **polysemantic**: top-activating contexts span multiple unrelated concepts.

[Elhage et al. (2022, "Toy Models of Superposition")](https://transformer-circuits.pub/2022/toy_model/index.html) explain why. When features are sparse (most off most of the time), a $d$-dimensional space can represent ~$d / \log d$ features by overlapping them at non-orthogonal angles. The non-linearity in the MLP allows partial recovery: only one feature in a superposed pair is typically active in any given input, so interference is bounded.

Consequences:

1. The "real" interpretable features are *directions* (linear combinations of neurons), not single neurons.
2. Reading individual neuron activations gives a tangled, polysemantic picture.
3. To recover monosemantic features, train an overcomplete dictionary on cached activations: a **sparse autoencoder (SAE)**.

[Bricken et al. (2023, "Towards Monosemanticity")](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Templeton et al. (2024, "Scaling Monosemanticity")](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) trained SAEs on Claude 3 Sonnet and recovered millions of monosemantic features ranging from "the Golden Gate Bridge" to "code with security vulnerabilities."

## Direct logit attribution for MLPs

Because each neuron's contribution to the residual stream is $\sigma(k_n^\top x) v_n$, its contribution to the final logit of token $w$ is:

$$\Delta\text{logit}_n(w) = \sigma(k_n^\top x)\, v_n^\top W_U[:, w]$$

Sort neurons by $|\Delta\text{logit}_n(w)|$ to identify which neurons drove the prediction. This is **MLP-level DLA**.

```python
# in TransformerLens
mlp_act = cache["post", layer, "mlp"]      # [seq, d_ffn]
W_out = model.W_out[layer]                 # [d_ffn, d_model]
W_U = model.W_U[:, answer_id]              # [d_model]
neuron_dla = mlp_act[-1] * (W_out @ W_U)   # [d_ffn]
top_neurons = neuron_dla.argsort(descending=True)[:10]
```

## Why MLPs hold the knowledge

Three lines of evidence support the claim that factual knowledge lives in MLPs:

1. **Parameter share.** MLPs are ~⅔ of total parameters. Most learned content is statistically there.
2. **Editing.** ROME and [MEMIT](https://memit.baulab.info/) edit specific facts by modifying MLP weights at specific layers (typically mid-layers, around layer 5–8 in GPT-2 medium). Editing attention weights does not produce the same effect.
3. **Causal tracing.** [Meng et al. (2022)](https://arxiv.org/abs/2202.05262) corrupt subject tokens, then restore individual layers' activations one at a time and measure which restoration recovers the correct prediction. The signal localizes to mid-layer MLPs.

A clean operational summary: **attention moves information; MLPs add new information.** Both contribute additively to the residual stream. Their roles are complementary.

## Activation functions in modern models

| Model | Non-linearity | Form |
|---|---|---|
| GPT-2 / GPT-3 | GeLU | $x \cdot \Phi(x)$ |
| Original Transformer | ReLU | $\max(0, x)$ |
| PaLM, Llama, Mistral | SwiGLU | $\text{Swish}(W_g x) \odot (W_\text{in} x)$ |

SwiGLU adds a gating branch:

$$\text{MLP}_\text{SwiGLU}(x) = W_\text{out}\, (\text{Swish}(W_g x) \odot W_\text{in} x)$$

This requires three matrices instead of two, but the key-value interpretation extends: each neuron's "key" is now a (gate, input) pair, and the value is still the corresponding $W_\text{out}$ column. Most MLP interpretability tooling generalizes with minor modification.

## What we have so far

| Component | Role | Reads | Writes |
|---|---|---|---|
| Embedding | Token → vector | token IDs | residual stream |
| Attention | Cross-position routing | residual stream (all positions) | residual stream (current position) |
| MLP | Stored knowledge / transforms | residual stream (current position) | residual stream (current position) |
| Unembedding | Vector → logits | residual stream (last position) | output distribution |

All four components communicate exclusively via the residual stream. Every interpretability tool in this series operates on that interface.

The next post runs a full forward pass through GPT-2 small, end-to-end, with concrete numbers at every stage.

## Resources

### Foundational papers

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/2012.14913" target="_blank" rel="noopener"><div class="research-card__title">Transformer Feed-Forward Layers Are Key-Value Memories</div><div class="research-card__authors">Geva et al., 2021 · the key-value framing</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2202.05262" target="_blank" rel="noopener"><div class="research-card__title">Locating and Editing Factual Associations in GPT (ROME)</div><div class="research-card__authors">Meng et al., 2022 · causal tracing + rank-one MLP edits</div></a></li>
  <li><a class="research-card" href="https://memit.baulab.info/" target="_blank" rel="noopener"><div class="research-card__title">Mass-Editing Memory in a Transformer (MEMIT)</div><div class="research-card__authors">Meng et al., 2023 · scaling ROME to thousands of edits</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/toy_model/index.html" target="_blank" rel="noopener"><div class="research-card__title">Toy Models of Superposition</div><div class="research-card__authors">Elhage et al., Anthropic 2022 · why polysemanticity is rational</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" target="_blank" rel="noopener"><div class="research-card__title">Towards Monosemanticity</div><div class="research-card__authors">Bricken et al., Anthropic 2023 · SAEs on a 1-layer transformer</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html" target="_blank" rel="noopener"><div class="research-card__title">Scaling Monosemanticity</div><div class="research-card__authors">Templeton et al., Anthropic 2024 · SAEs on Claude 3 Sonnet, millions of features</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2002.05202" target="_blank" rel="noopener"><div class="research-card__title">GLU Variants Improve Transformer</div><div class="research-card__authors">Shazeer, 2020 · why SwiGLU replaced GeLU in modern LLMs</div></a></li>
</ul>

### Tools and code

<ul class="research-list">
  <li><a class="research-card" href="https://neuronpedia.org/" target="_blank" rel="noopener"><div class="research-card__title">Neuronpedia</div><div class="research-card__authors">browse top-activating contexts for MLP neurons and SAE features across many models</div></a></li>
  <li><a class="research-card" href="https://github.com/jbloomAus/SAELens" target="_blank" rel="noopener"><div class="research-card__title">SAELens</div><div class="research-card__authors">train and analyze sparse autoencoders on any HF transformer</div></a></li>
  <li><a class="research-card" href="https://rome.baulab.info/" target="_blank" rel="noopener"><div class="research-card__title">ROME · code & demo</div><div class="research-card__authors">Bau Lab · reproduce factual editing in GPT-2 / GPT-J</div></a></li>
  <li><a class="research-card" href="https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html#MLP-Layers" target="_blank" rel="noopener"><div class="research-card__title">TransformerLens · MLP analysis</div><div class="research-card__authors">cache MLP activations, decompose neuron contributions to logits</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/zoom-in/" target="_blank" rel="noopener"><div class="research-card__title">Zoom In: An Introduction to Circuits</div><div class="research-card__authors">Olah et al., Distill 2020 · the original feature-and-circuit framing (vision)</div></a></li>
</ul>
