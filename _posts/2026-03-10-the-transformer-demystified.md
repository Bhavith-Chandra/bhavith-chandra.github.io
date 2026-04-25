---
layout: post-article
title: "The Transformer, Demystified: A Factory Floor That Runs on Language"
date: 2026-03-10
permalink: /posts/the-transformer-demystified/
excerpt: "A decoder-only transformer is a stack of N identical blocks operating on a residual stream of token vectors. Six stations, one conveyor belt, one probability distribution per step."
read_time_label: "12 min read"
accent: amber
math: true
---

A **decoder-only transformer** is a stack of $N$ identical blocks operating on a sequence of $T$ token vectors. Input: a token sequence. Output: a probability distribution over the vocabulary for the next token. The internals consist of six stations on a conveyor belt called the **residual stream**.

This post defines each station precisely, explains why the architecture replaced RNNs, and sets up the vocabulary used in the rest of the series.

---

## Tour the factory

{% include demos/factory-floor.html %}

Click a station, press play, watch one token's vector traverse the pipeline. The structure is identical for every modern LLM: GPT-2, GPT-3, GPT-4, Claude, Llama, Gemini.

## The six stations

**1. Tokenizer.** Maps a string to a sequence of integer IDs using a fixed vocabulary $V$ (typical sizes: 50,257 for GPT-2, 100,277 for GPT-4, 128,000 for Llama 3). Modern tokenizers use byte-pair encoding (BPE) or its variants. Example: `"unhappily"` → `["un", "happ", "ily"]` → `[403, 7829, 6148]`.

**2. Embedding.** A learned lookup table $W_E \in \mathbb{R}^{V \times d_\text{model}}$ maps each token ID to a $d_\text{model}$-dimensional vector. Common sizes: $d_\text{model} = 768$ (GPT-2 small), 4,096 (Llama 3 8B), 12,288 (GPT-3 175B). The $i$-th row of $W_E$ is the embedding for token $i$.

**3. Self-attention.** At each position $t$, the model computes weighted sums over all positions $\le t$ (causal mask). Weights are derived from query/key dot products. Output: a new vector at each position that has read from earlier positions. Multiple heads run in parallel ($n_\text{heads} = 12, 32, 96, \ldots$), each in a lower-dimensional subspace.

**4. MLP.** A position-wise feed-forward network: two linear layers with a non-linearity (GeLU, SwiGLU). Hidden dimension is typically $4 d_\text{model}$. Same shape in, same shape out. Operates independently on each position.

**5. Repeat.** One **block** = attention + MLP + residual connections + layer norms. Stack $N$ of them. GPT-2 small: $N=12$. Llama 3 8B: $N=32$. GPT-3 175B: $N=96$. GPT-4 (rumoured): $N \approx 120$.

**6. Unembedding.** Multiply the final residual stream vector by $W_U \in \mathbb{R}^{d_\text{model} \times V}$ to produce **logits**. Apply softmax to get a probability distribution over the vocabulary. Sample → next token.

Total parameters scale roughly as $N \cdot d_\text{model}^2 \cdot 12$ for the standard recipe.

{% include demos/softmax-temperature.html %}

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Each station reads the conveyor belt, computes a small contribution, and adds it back. By the final station the vector at the last position has accumulated enough information to identify the next token.</p>
</aside>

## The residual stream

In the literature this conveyor belt is the **residual stream**. Each block's output is *added* to its input, not substituted:

$$x_i \;\leftarrow\; x_i + \text{block}(x_i)$$

<div class="math-translate">In words: the new belt vector equals the old belt vector plus whatever the block computed. Nothing is overwritten; contributions accumulate.</div>

{% include demos/stream-accumulator.html %}

Three consequences follow directly from this additive structure:

1. **Gradient flow.** Information from layer 0 reaches layer $N$ along the identity path. This is why deep transformers train in the first place.
2. **Per-block delta.** Each block computes a *correction*, not a full representation. Easier optimization target.
3. **Linear decomposability.** The output is a sum of token embeddings + each block's contribution. Mechanistic interpretability uses this directly: ablate one head, see the difference in the logits.

Elhage et al. ([2021, "Mathematical Framework"](https://transformer-circuits.pub/2021/framework/index.html)) formalize this view. The residual stream is the central object in the rest of this series.

## Why transformers replaced RNNs

**Recurrent neural networks (RNNs)** process tokens sequentially, maintaining a hidden state $h_t = f(h_{t-1}, x_t)$. Two structural problems:

1. **No parallelism.** Token $t$ depends on $h_{t-1}$. Cannot batch positions on a GPU.
2. **Vanishing dependencies.** Gradients through 1,000 timesteps decay or explode. LSTMs and GRUs help but do not solve it at scale.

Transformers ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) replace recurrence with self-attention: every position reads from every other position in parallel via a single matrix multiplication. Sequence length $T$ → $O(T^2)$ time, but fully parallel within a sequence and across batch dimensions. On modern GPUs the constant factor wins.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>Every modern LLM is a decoder-only transformer with the same six stations. Different sizes, different training data, different fine-tuning. The interpretability tools from this series transfer directly across all of them.</p>
</aside>

## Decoder-only

The original 2017 paper had two halves: an **encoder** (reads input) and a **decoder** (writes output), wired for machine translation. For autoregressive text generation, only the decoder is needed.

A **decoder-only transformer** uses **causal self-attention**: position $t$ may only attend to positions $0, 1, \ldots, t$. This enforces left-to-right generation and matches the next-token-prediction training objective.

This series is exclusively about decoder-only transformers. (Encoder-only models like BERT are used for classification and embedding tasks, not generation.)

## Autoregressive generation

The transformer produces one probability distribution per forward pass. To generate text:

```
input:  "The cat sat on the"          → logits → sample → "mat"
input:  "The cat sat on the mat"      → logits → sample → "."
input:  "The cat sat on the mat."     → logits → sample → "<eos>"
```

This is **autoregressive decoding**. Each generated token is appended to the input and the model runs again. Sampling strategies (greedy, top-k, top-p, temperature) determine *which* token gets picked from the distribution. KV-caching avoids recomputing attention over previous tokens, making the per-step cost roughly $O(T)$ rather than $O(T^2)$.

## Key dimensions, by model

| Model | $N$ | $d_\text{model}$ | $n_\text{heads}$ | $V$ | Parameters |
|---|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 12 | 50,257 | 124M |
| GPT-2 XL | 48 | 1,600 | 25 | 50,257 | 1.5B |
| GPT-3 | 96 | 12,288 | 96 | 50,257 | 175B |
| Llama 3 8B | 32 | 4,096 | 32 | 128,000 | 8B |
| Llama 3 70B | 80 | 8,192 | 64 | 128,000 | 70B |

Memorize one row (GPT-2 small is most common in MI papers) and use it as the reference point.

{% include demos/scaling-explorer.html %}

## What's ahead

Subsequent posts cover each station in mechanistic detail:

- **Tokens.** BPE, vocabulary quirks, the bugs they cause.
- **Residual stream.** The most important MI primitive; logit lens; linear decomposition.
- **Attention.** QK/OV decomposition; induction heads; IOI circuit.
- **MLPs.** Key-value memory framing; superposition; sparse autoencoders.
- **Full forward pass.** End-to-end with real GPT-2 numbers.

The next post is on tokenization.

## Resources

### Foundational papers

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani et al., 2017 · the original transformer paper</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage et al., Anthropic 2021 · residual-stream view used throughout this series</div></a></li>
  <li><a class="research-card" href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noopener"><div class="research-card__title">Language Models are Unsupervised Multitask Learners</div><div class="research-card__authors">Radford et al., OpenAI 2019 · GPT-2 architecture and training</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener"><div class="research-card__title">Language Models are Few-Shot Learners</div><div class="research-card__authors">Brown et al., 2020 · GPT-3, scaling laws in practice</div></a></li>
</ul>

### Tutorials and code

<ul class="research-list">
  <li><a class="research-card" href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noopener"><div class="research-card__title">The Illustrated Transformer</div><div class="research-card__authors">Jay Alammar · diagram-led walkthrough of the original architecture</div></a></li>
  <li><a class="research-card" href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noopener"><div class="research-card__title">nanoGPT</div><div class="research-card__authors">Andrej Karpathy · ~300-line PyTorch implementation of GPT-2</div></a></li>
  <li><a class="research-card" href="https://www.youtube.com/watch?v=kCc8FmEb1nY" target="_blank" rel="noopener"><div class="research-card__title">Let's build GPT: from scratch, in code, spelled out</div><div class="research-card__authors">Karpathy · 2-hour lecture; pairs with nanoGPT</div></a></li>
  <li><a class="research-card" href="https://transformerlensorg.github.io/TransformerLens/" target="_blank" rel="noopener"><div class="research-card__title">TransformerLens</div><div class="research-card__authors">Neel Nanda et al. · the standard MI library; load any HF model and inspect activations</div></a></li>
  <li><a class="research-card" href="https://www.neelnanda.io/mechanistic-interpretability/getting-started" target="_blank" rel="noopener"><div class="research-card__title">A Comprehensive Mechanistic Interpretability Explainer</div><div class="research-card__authors">Neel Nanda · glossary and recommended reading order</div></a></li>
</ul>
