---
layout: post-article
title: "Attention: How Every Position Decides Who to Listen To"
date: 2026-05-26
permalink: /posts/attention-how-every-position-decides-who-to-listen-to/
excerpt: "Attention is a dot-product-based routing mechanism. Each head decomposes into a QK circuit (where to attend) and an OV circuit (what to write back), enabling head-level interpretability."
read_time_label: "15 min read"
accent: amber
math: true
---

**Self-attention** is the mechanism that lets each position in a sequence read from every other (causally) position. A single attention head consists of three learned linear maps and a softmax. A multi-head layer runs $n_\text{heads}$ such heads in parallel.

This post defines the operation, derives the **QK / OV decomposition** that underlies head-level interpretability, and walks through four head archetypes plus the indirect-object identification circuit.

---

## Demo: 72 real attention heads

{% include demos/attention-explorer.html %}

3 prompts × 6 routing-pattern reproductions × matrix and flow views. The patterns shown ("previous-token", "BOS sink", "induction", "duplicate-token", "name-mover", "self") are reproductions of the canonical patterns observed in real GPT-2 small heads.

## Definition

Each attention head has three weight matrices:

$$W_Q, W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$$

Typical sizes: GPT-2 small has $d_\text{model} = 768$, $n_\text{heads} = 12$, $d_\text{head} = 64$ per head ($d_\text{head} = d_\text{model} / n_\text{heads}$).

For input $X \in \mathbb{R}^{T \times d_\text{model}}$:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V \quad \in \mathbb{R}^{T \times d_\text{head}}$$

**Attention scores** (causal, scaled):

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_\text{head}}} + M\right)$$

where $M_{ij} = -\infty$ for $j > i$ (causal mask), 0 otherwise. $A \in \mathbb{R}^{T \times T}$ is the attention pattern.

**Output:**

$$Z = AV \quad \in \mathbb{R}^{T \times d_\text{head}}$$

Multi-head: concatenate $n_\text{heads}$ outputs $[Z^{(1)}, \ldots, Z^{(h)}]$ and project through $W_O \in \mathbb{R}^{(n_\text{heads} \cdot d_\text{head}) \times d_\text{model}}$ to write back to the residual stream.

The scaling factor $\sqrt{d_\text{head}}$ keeps the dot products in a numerically stable range ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), §3.2.1).

## QK and OV: two circuits per head

Each head can be analyzed as two independent linear maps composed by softmax + sum.

### QK circuit (where to attend)

The attention score is bilinear in the inputs:

$$Q_i K_j^\top = (X_i W_Q)(X_j W_K)^\top = X_i (W_Q W_K^\top) X_j^\top$$

The product $W_{QK} := W_Q W_K^\top \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ is the **QK matrix**. It maps pairs (query position content, key position content) → score. Eigendecomposing or projecting $W_{QK}$ onto interpretable subspaces reveals the routing rule.

### OV circuit (what to write)

The output written back to the residual stream from source position $j$, weighted by $A_{ij}$, is:

$$\Delta_i = \sum_j A_{ij}\, X_j W_V W_O^{(h)}$$

The product $W_{OV} := W_V W_O^{(h)} \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ is the **OV matrix**. It maps (source residual content) → (write contribution). Reading the eigenstructure of $W_{OV}$ describes what kind of information the head copies.

**The two are independent.** Routing (QK) and payload (OV) are trained jointly but are mathematically separate objects. Most interpretability claims about a head reduce to characterizing $W_{QK}$ and $W_{OV}$ separately. ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html))

{% include demos/qk-ov-toggle.html %}

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>"What does this head do?" decomposes into two questions: "What does QK select for?" and "What does OV copy?" Almost every head archetype in the literature (induction, copy, name-mover, S-inhibition) is named after its OV behavior with a description of QK as the routing condition.</p>
</aside>

## Four head archetypes

### 1. Previous-token heads

- **QK**: position $i$ attends primarily to $i-1$. Often pure positional (the QK matrix is approximately a shift operator after positional encoding).
- **OV**: copies the source token's embedding into the destination.
- **Where**: layer 0–2 in GPT-2 small.
- **Use**: feeds shifted-token information into later heads. A prerequisite for induction.

### 2. Induction heads

In-context bigram completion: if the prefix contains `…A B…` and the current token is a later `A`, the head attends to the position right after the prior `A` and copies that token (`B`) forward.

- **QK**: at position of the second `A`, query matches keys at positions whose previous token equals `A`. This requires the previous-token information that previous-token heads write.
- **OV**: copies the source token.
- **Where**: typically appears around layer 5–6 in GPT-2 small (after a previous-token head feeds layer 0).
- **Significance**: [Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) argue induction heads are the mechanistic basis of in-context learning.

### 3. Name-mover heads

- **QK**: the final position ("___") attends to name tokens earlier in the sentence.
- **OV**: copies the name's embedding to the final position, increasing that name's logit.
- **Where**: layer 9–10 in GPT-2 small.
- **Use**: the output stage of the IOI circuit (below).

{% include demos/attention-painter.html %}

### 4. Attention sinks (BOS sink)

Many heads route most of their attention to position 0 (BOS) on tokens where the head has nothing useful to do. Softmax forces the weights to sum to 1, so the head must attend somewhere; the BOS slot acts as a "rest" position with low informational impact.

- **QK**: queries that don't match any content key default to the BOS key.
- **Where**: layers 1–3, many heads.
- **Reference**: [Xiao et al. (2023)](https://arxiv.org/abs/2309.17453); also discussed in [Templeton et al. (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

## The IOI circuit

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) reverse-engineered the algorithm GPT-2 small uses to predict `Mary` for the prompt:

> *"When John and Mary went to the store, John gave a drink to ___"*

The circuit involves ~26 attention heads across layers 0–11, organized into named functional groups. Sketch:

| Stage | Heads (layer.head) | Role |
|---|---|---|
| Duplicate Token | 0.1, 0.10, 3.0 | Detect repeated names. Output: "this name appears twice." |
| Previous Token | 2.2, 4.11 | Move name info to positions before/after each name. |
| Induction | 5.5, 5.8, 5.9, 6.9 | Pattern-match across the sentence using duplicate-token features. |
| S-Inhibition | 7.3, 7.9, 8.6, 8.10 | Write a "John is the subject, suppress John" signal at the final position. |
| Name Mover | 9.6, 9.9, 10.0 | Attend from final position to names. Suppression from S-Inhibition makes them attend to Mary, not John. Output: Mary's logit goes up. |
| Negative Name Mover | 10.7, 11.10 | Slightly suppress the answer (regularization-like). |

The paper validates each role via path-patching ablations: zeroing out a single head's contribution to the relevant downstream component degrades the answer. Reproducible in TransformerLens with ~50 lines of code.

This was the first complete circuit reverse-engineered in a language model.

## Causal masking

Decoder-only models enforce $A_{ij} = 0$ for $j > i$ via the mask $M$. Two reasons:

1. **Training objective.** Predicting token $t$ given $0, \ldots, t-1$. If position $t$ could attend to $t+1$, the loss would leak the answer.
2. **Generation.** At inference, future tokens don't exist yet.

The mask is added to attention scores *before* softmax, with $-\infty$ in the masked positions, so masked weights become exactly zero.

In matrix form, $A$ is lower-triangular. Visible in every demo above.

## Multi-head attention

Why $h$ heads instead of one bigger head? Each head learns a different $(W_{QK}, W_{OV})$, allowing the layer to perform multiple routings simultaneously: head 1 might do "previous token" while head 2 does "subject of the sentence" while head 3 acts as a BOS sink. With one head these would have to share the same projection.

```
attn_layer(X):
    heads = []
    for h in range(n_heads):
        Q = X @ W_Q[h]; K = X @ W_K[h]; V = X @ W_V[h]
        A = softmax(Q @ K.T / sqrt(d_head) + causal_mask)
        heads.append(A @ V)
    return concat(heads, dim=-1) @ W_O
```

In code, this is one batched matmul with a head dimension. Conceptually, $h$ independent attention operations.

## Softmax: the source of selectivity (and BOS sinks)

The softmax is what makes attention *selective*. A linear weighting would average; softmax allows sharp, peaky distributions where one position gets most of the weight.

The constraint is that $\sum_j A_{ij} = 1$. The head must attend somewhere. When no key matches the query, it defaults to whatever residual key is closest, often the BOS token, which becomes the default sink.

Some recent architectures replace softmax with linear attention (Linformer, RetNet, Mamba) or kernelized variants to avoid this and to get sub-quadratic time. Softmax remains the standard for frontier LLMs.

## Attention as associative memory

A useful frame: attention performs **content-addressable retrieval** from the context. The query is an address; the keys are stored entries; the softmax picks the closest match. The OV circuit decides what to retrieve from the matched entry.

This is why attention scales gracefully across context length and why it pairs naturally with MLPs: attention retrieves relevant context-specific information; MLPs apply training-time-stored transformations to it. The next post is on MLPs.

## Resources

### Foundational papers

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani et al., 2017 · the original mechanism</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage et al., Anthropic 2021 · QK / OV decomposition formalized</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson et al., Anthropic 2022 · induction heads as the basis of ICL</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2211.00593" target="_blank" rel="noopener"><div class="research-card__title">Interpretability in the Wild: a Circuit for IOI in GPT-2</div><div class="research-card__authors">Wang et al., 2022 · the IOI circuit, end-to-end reverse engineering</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2309.17453" target="_blank" rel="noopener"><div class="research-card__title">Efficient Streaming Language Models with Attention Sinks</div><div class="research-card__authors">Xiao et al., 2023 · the BOS attention-sink phenomenon</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noopener"><div class="research-card__title">RoFormer: Enhanced Transformer with Rotary Position Embedding</div><div class="research-card__authors">Su et al., 2021 · RoPE, used by Llama, Mistral, GPT-NeoX</div></a></li>
</ul>

### Tutorials and code

<ul class="research-list">
  <li><a class="research-card" href="https://transformerlensorg.github.io/TransformerLens/generated/demos/Exploratory_Analysis_Demo.html" target="_blank" rel="noopener"><div class="research-card__title">TransformerLens · Exploratory Analysis</div><div class="research-card__authors">cache attention patterns, run path patching, replicate IOI</div></a></li>
  <li><a class="research-card" href="https://github.com/callummcdougall/ARENA_3.0" target="_blank" rel="noopener"><div class="research-card__title">ARENA 3.0 · Chapter 1.4 Indirect Object Identification</div><div class="research-card__authors">Callum McDougall · code-along reproduction of the IOI circuit</div></a></li>
  <li><a class="research-card" href="https://www.youtube.com/watch?v=ML4u0vDdf4Y" target="_blank" rel="noopener"><div class="research-card__title">Neel Nanda · A Walkthrough of Reverse Engineering Modular Addition</div><div class="research-card__authors">applied QK / OV analysis on a toy task</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2016/augmented-rnns/" target="_blank" rel="noopener"><div class="research-card__title">Attention and Augmented Recurrent Neural Networks</div><div class="research-card__authors">Olah & Carter, Distill 2016 · classic illustrated intro</div></a></li>
  <li><a class="research-card" href="https://github.com/jessevig/bertviz" target="_blank" rel="noopener"><div class="research-card__title">BertViz</div><div class="research-card__authors">Jesse Vig · interactive attention pattern visualizer for any HF model</div></a></li>
</ul>
