---
layout: post-article
title: "The Residual Stream: The Belt That Runs the Whole Factory"
date: 2026-03-28
permalink: /posts/the-residual-stream/
excerpt: "The residual stream is a per-position running sum that every block reads from and writes to. Because it is additive and lives in a single coordinate frame, it admits direct linear decomposition: the foundation of the logit lens, direct logit attribution, and activation patching."
read_time_label: "13 min read"
accent: amber
math: true
---

The **residual stream** is the central data structure of a transformer. It is a tensor of shape $[T, d_\text{model}]$ where $T$ is the sequence length and $d_\text{model}$ is the model dimension. Each block reads it, computes an additive update, and writes it back. The stream is never overwritten.

This post defines the stream formally, explains why the additive structure is load-bearing, and introduces the two interpretability primitives derived from it: the **logit lens** and **direct logit attribution**.

---

## Demo: logit lens trajectory

{% include demos/residual-stream.html %}

Each cell shows the model's top-1 prediction at layer $\ell$, position $t$. Saturation = confidence. Click a cell for the full top-5 distribution and the residual norm.

## Formal definition

For a transformer with $L$ blocks operating on $T$ tokens, the residual stream is the sequence of states $\{X_0, X_1, \ldots, X_L\}$, each $X_\ell \in \mathbb{R}^{T \times d_\text{model}}$.

**Initial state:** $X_0 = E + P$ where $E$ is the token embedding and $P$ is the positional encoding (or $X_0 = E$ for models with rotary/RoPE applied inside attention).

**Block update:**

$$X_{\ell+1} = X_\ell + \text{Attn}_\ell(\text{LN}(X_\ell)) + \text{MLP}_\ell(\text{LN}(X_\ell + \text{Attn}_\ell(\text{LN}(X_\ell))))$$

The layer norms ($\text{LN}$) appear inside each sublayer in the **pre-norm** configuration used by GPT-2, Llama, and most modern models. Schematically:

$$X_{\ell+1} = X_\ell + \Delta_\ell^\text{attn} + \Delta_\ell^\text{mlp}$$

**Final read:** logits = $\text{LN}(X_L) \cdot W_U$, where $W_U \in \mathbb{R}^{d_\text{model} \times V}$ is the unembedding matrix.

The architectural choice that everything depends on is the `+`: each $\Delta$ is *added*, never substituted.

## Why additive matters

Compare the residual update with a non-residual update $X_{\ell+1} = f_\ell(X_\ell)$. Two structural problems with the latter:

**Vanishing information.** Any signal computed at layer $\ell$ must be re-encoded by $f_{\ell+1}, f_{\ell+2}, \ldots$ to survive. After 30 layers of arbitrary nonlinear transformations, layer-1 signals are effectively destroyed.

**Vanishing gradients.** Backprop multiplies gradients through every $f_\ell$. With layer-norm and sigmoid/tanh nonlinearities the gradient norm shrinks geometrically. Pre-2015 networks rarely trained stably past 20 layers.

Residual connections ([He et al., 2015](https://arxiv.org/abs/1512.03385)) solve both: $X_{\ell+1} = X_\ell + f_\ell(X_\ell)$ has an identity path from layer $\ell$ to $\ell+1$. Information and gradients flow through the `+` without distortion. This is what made GPT-2's 48-layer and GPT-3's 96-layer networks trainable.

For interpretability, the additive structure has a stronger consequence: the final state is *literally* a sum.

$$X_L = X_0 + \sum_{\ell=0}^{L-1} \Delta_\ell^\text{attn} + \sum_{\ell=0}^{L-1} \Delta_\ell^\text{mlp}$$

Every component's contribution is a linear term. This makes the residual stream **linearly decomposable**.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>A shared document where every editor can only add comments, not delete. The final document is the sum of all contributions. To attribute the final state to one editor, look at their diff.</p>
</aside>

## The logit lens

Final-layer logits are computed as $\text{LN}(X_L) \cdot W_U$. Because every $X_\ell$ lives in $\mathbb{R}^{T \times d_\text{model}}$, the same projection is well-defined at every layer:

$$\text{logits}_\ell := \text{LN}(X_\ell) \cdot W_U$$

[Nostalgebraist (2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) called this the **logit lens**. It returns the model's "current best guess" at each intermediate layer, treating the partial residual stream as if it were the final state.

Empirically (visible in the demo above for "Paris is the capital of"):

- Layers 0–2: predictions are close to a unigram distribution. The model has not yet aggregated context.
- Layers 3–6: top-k starts ranking semantically related tokens (countries, cities).
- Layers 7–11: the correct answer (`France`) reaches top-1 with high probability.

The lens is not exact, intermediate $X_\ell$ has different statistics than $X_L$, but it is informative and free. Refinements include the **tuned lens** ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)), which learns a per-layer affine correction.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The logit lens is a training-free decoder for any layer's residual stream. Activation patching, direct logit attribution, and most circuit-discovery techniques inherit from this idea. Internalizing it makes the literature readable.</p>
</aside>

## Direct logit attribution (DLA)

Because $X_L$ is a sum, the logit for any vocabulary token $w$ is also a sum:

$$\text{logit}(w) = (X_0 \cdot W_U[:, w]) + \sum_{\ell=0}^{L-1} (\Delta_\ell^\text{attn} \cdot W_U[:, w]) + \sum_{\ell=0}^{L-1} (\Delta_\ell^\text{mlp} \cdot W_U[:, w])$$

Each term is a scalar: how much that component pushed the prediction toward $w$. This is **direct logit attribution**.

Practical use:

```python
# in TransformerLens
import transformer_lens as tl
model = tl.HookedTransformer.from_pretrained("gpt2")
tokens = model.to_tokens("Paris is the capital of")
logits, cache = model.run_with_cache(tokens)

# decompose final residual stream into per-component contributions
per_layer = cache.decompose_resid(layer=-1, return_labels=True)
# project each onto W_U for the answer token
answer_id = model.to_single_token(" France")
W_U = model.W_U[:, answer_id]
contributions = per_layer[0] @ W_U  # one scalar per component
```

The largest entries in `contributions` identify the layers/heads/MLPs that drove the answer. DLA is the starting point for circuit analysis: keep zooming in (head → query/key/value → input neurons) until you have a mechanism.

{% include demos/dla-bars.html %}

## Subspaces and superposition

The stream has $d_\text{model}$ dimensions but generally encodes far more *features* than that. Components write to and read from **subspaces** of the stream, generally not axis-aligned.

[Elhage et al. (2022, "Superposition")](https://transformer-circuits.pub/2022/toy_model/index.html) characterize this: when features are sparse (most are off most of the time), a $d$-dim space can represent ~$d / \log d$ features by overlapping them. The cost is interference: reading one feature picks up small projections from others.

Consequences:

1. Single neurons are typically **polysemantic** (active for multiple unrelated concepts).
2. Single residual coordinates are not interpretable; *directions* are.
3. **Sparse autoencoders (SAEs)** ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html); [Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)) recover interpretable directions by training an overcomplete dictionary on cached residual streams.

Provisional model: think of the stream as a high-dimensional space where many features overlap, recoverable by linear probes or SAEs but not by reading individual coordinates.

## Reading the heatmap

In the demo, watch:

1. **Vertical evolution.** The same column (token position) refines its top prediction across layers.
2. **Horizontal differences.** Earlier positions are *not* trying to predict the next token, they are accumulating information that attention will later pull into the final position. Their logit-lens predictions are largely incidental.
3. **Final column saturation.** This is where the actual next-token prediction happens. Saturation increases monotonically (with rare exceptions in degenerate prompts).

## BOS and attention sinks

The first token's residual stream typically accumulates "housekeeping" state. Attention heads with no relevant key in a given query often place mass on the BOS token as a default, the **attention sink** ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)). [Templeton et al. (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) found that BOS-position SAE features are systematically distinct from content-position features.

Treat the BOS column as anomalous when interpreting diagnostics.

## The unifying claim

> Every transformer mechanism can be expressed as "component $C$ reads from subspace $R$ of the residual stream and writes to subspace $W$ of the residual stream."

Examples:
- **Copy heads** (attention): read content from position $i$, write the same content to position $j$.
- **Induction heads**: read a match-detection signal at the previous position, write a "copy this token" signal at the current.
- **Factual-recall MLPs** ([Meng et al., 2022, ROME](https://arxiv.org/abs/2202.05262)): read subject embedding from subject tokens, write attribute information back.
- **IOI circuit** ([Wang et al., 2022](https://arxiv.org/abs/2211.00593)): a chain of read/write heads juggling name and position information.

{% include demos/activation-patching.html %}

The next two posts cover attention and MLPs as readers/writers in detail.

## Resources

### Foundational papers

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener"><div class="research-card__title">Deep Residual Learning for Image Recognition</div><div class="research-card__authors">He et al., 2015 · ResNet, the original residual-connection paper</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage et al., Anthropic 2021 · introduces the residual-stream view; foundational for this series</div></a></li>
  <li><a class="research-card" href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" target="_blank" rel="noopener"><div class="research-card__title">Interpreting GPT: the logit lens</div><div class="research-card__authors">Nostalgebraist, LessWrong 2020 · the original logit-lens post</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2303.08112" target="_blank" rel="noopener"><div class="research-card__title">Eliciting Latent Predictions from Transformers with the Tuned Lens</div><div class="research-card__authors">Belrose et al., 2023 · per-layer affine refinement of the logit lens</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/toy_model/index.html" target="_blank" rel="noopener"><div class="research-card__title">Toy Models of Superposition</div><div class="research-card__authors">Elhage et al., Anthropic 2022 · why features overlap in the residual stream</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html" target="_blank" rel="noopener"><div class="research-card__title">Scaling Monosemanticity</div><div class="research-card__authors">Templeton et al., Anthropic 2024 · SAE features in Claude 3 Sonnet's residual stream</div></a></li>
</ul>

### Tools and code

<ul class="research-list">
  <li><a class="research-card" href="https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html" target="_blank" rel="noopener"><div class="research-card__title">TransformerLens · Main Demo</div><div class="research-card__authors">cache hooks, decompose_resid, direct logit attribution in code</div></a></li>
  <li><a class="research-card" href="https://github.com/AlignmentResearch/tuned-lens" target="_blank" rel="noopener"><div class="research-card__title">tuned-lens</div><div class="research-card__authors">official tuned-lens implementation; works on any HF causal LM</div></a></li>
  <li><a class="research-card" href="https://www.neelnanda.io/mechanistic-interpretability/glossary" target="_blank" rel="noopener"><div class="research-card__title">Neel Nanda's MI Glossary</div><div class="research-card__authors">definitions for residual stream, DLA, activation patching</div></a></li>
  <li><a class="research-card" href="https://arena3-chapter1-transformer-interp.streamlit.app/" target="_blank" rel="noopener"><div class="research-card__title">ARENA · Transformer Interpretability</div><div class="research-card__authors">guided exercises building DLA and the logit lens from scratch</div></a></li>
</ul>
