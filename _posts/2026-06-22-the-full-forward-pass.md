---
layout: post-article
title: "The Full Forward Pass: Putting Every Piece on the Belt"
date: 2026-06-22
permalink: /posts/the-full-forward-pass/
excerpt: "End-to-end execution: tokens → embeddings → six attention+MLP blocks → unembedding. Tensor shapes, layer-by-layer logit-lens trajectory, and the surface area available for mechanistic analysis."
read_time_label: "11 min read"
accent: amber
math: true
---

A transformer forward pass is a single deterministic function from a token sequence to a probability distribution over the next token. This post traces that function end-to-end through GPT-2 / distilGPT2, with concrete tensor shapes, the logit-lens trajectory at each stage, and the resulting surface area available for mechanistic analysis.

---

## Demo: layer-by-layer trajectory

{% include demos/grand-tour.html %}

Step through the 9 stages: tokenize → embed → block 0 → ... → block 5 → unembed. The token strip shows the logit-lens prediction at every position; the final-stage panel shows the actual output distribution.

Try the `A B C D E F G A B C` preset to observe in-context induction: middle layers detect the repeated bigram and predict the continuation `D` from the first occurrence.

## Stage-by-stage walkthrough

Reference prompt: `"The capital of France is"`. Model: distilGPT2 ($L=6$, $d_\text{model}=768$, $n_\text{heads}=12$, $V=50{,}257$).

### 1. Tokenize

```
"The capital of France is"
→ token IDs: [464, 3139, 286, 4881, 318]
→ pieces:    ["The", " capital", " of", " France", " is"]
```

GPT-2 BPE. 5 tokens. Note " France" is a single token (common proper noun); " capital" includes its leading space.

### 2. Embed

```
input_ids: [5]   →  embeddings: [5, 768]
```

Token embedding $W_E[\text{ids}] \in \mathbb{R}^{5 \times 768}$ plus learned positional embedding $W_P[0:5] \in \mathbb{R}^{5 \times 768}$.

Logit lens at this stage approximately returns the input tokens themselves (no context mixing has occurred). Final-position prediction is meaningless; the model has only seen the token " is" in isolation.

### 3. Block 0 (attention + MLP)

```
residual:     [5, 768]
attn output:  [5, 768]    (12 heads × 64 dim, projected back via W_O)
mlp output:   [5, 768]    (3072 neurons → 768 via W_out)
new residual: [5, 768]    (sum of three)
```

Block 0 is dominated by previous-token heads (attending one position back) and surface-feature MLP neurons (capitalization, punctuation, common morphemes). The logit lens still returns near-token-identity at most positions.

### 4. Blocks 1–4

```
residual: [5, 768] → ... → [5, 768]
```

Semantic consolidation. Geographic relations form: " France" gathers context from " capital" and " of". By block 3, the final-position logit-lens prediction includes country and city names in the top-5. By block 4, " Paris" has typically reached top-1, but with low confidence (~30–50%).

This is also where induction heads activate on patterned prompts. For `A B C D E F G A B C`, blocks 2–4 detect the prefix repetition and route the continuation forward.

### 5. Block 5 (final block)

```
residual: [5, 768] → [5, 768]
```

Sharpening. Late-layer name-mover-style heads pull " Paris" embedding into the final position; the answer's probability mass concentrates. Competing candidates (" the", " France") get suppressed by negative-name-mover-style components.

### 6. Unembed

```
final_residual[-1]: [768]
W_U:                [768, 50257]
logits:             [50257]
softmax(logits):    [50257] probability distribution
```

Apply final layer norm, then project the last position's residual through $W_U$ to produce a logit for every token in the vocabulary. Softmax gives the probability distribution. For distilGPT2 on this prompt, " Paris" is top-1 with ~80% probability; the remaining mass is distributed over " France", " Europe", " Britain", " Germany", and a long tail.

The model commits one token. To generate more, append the chosen token and run the forward pass again.

## Tensor shapes summary

| Stage | Tensor | Shape | Memory (fp16) |
|---|---|---|---|
| Token IDs | input | $[5]$ | 20 B |
| Embedding | $X_0$ | $[5, 768]$ | 7.5 KB |
| Per-head Q/K/V | per head | $[5, 64]$ | 640 B each |
| Attention pattern | per head | $[5, 5]$ | 50 B per head |
| MLP hidden | per block | $[5, 3072]$ | 30 KB |
| Final residual | $X_L$ | $[5, 768]$ | 7.5 KB |
| Logits | output | $[50257]$ | 100 KB |

## Surface area for analysis

distilGPT2 contains:

- **6 blocks** × (12 attention heads + 1 MLP) = **78 sub-components**
- **6 × 12 = 72 attention heads** (each with $W_{QK}, W_{OV}$ to characterize)
- **6 × 3072 = 18,432 MLP neurons** (each with $k_n, v_n$)
- **6 × $768^2$ = ~3.5M attention parameters**
- **6 × 2 × 768 × 3072 ≈ 28.3M MLP parameters**

Total: ~82M parameters (the embedding/unembedding tables add another ~38M).

For comparison:

| Model | $L$ | Heads/block | Neurons/block | Total components |
|---|---|---|---|---|
| distilGPT2 | 6 | 12 | 3,072 | 78 |
| GPT-2 small | 12 | 12 | 3,072 | 156 |
| GPT-2 XL | 48 | 25 | 6,400 | 1,248 |
| Llama 3 8B | 32 | 32 | 14,336 | 1,056 |
| Claude / GPT-4 class | ~100+ | ~100+ | ~50,000+ | tens of thousands |

The MI program: characterize each of these components in terms of what it reads from and writes to the residual stream. This is fully tractable for distilGPT2 and GPT-2 small (the IOI circuit is one example). It is partially tractable for 8B-class open models with sparse autoencoders. It is an open research problem at frontier scale.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The forward pass is one deterministic function. Every output token is the result of running the same circuit on a different input. Understanding the circuit at one input often generalizes; that's why a single-prompt analysis (IOI on one sentence) yields claims about a head's behavior across thousands of prompts.</p>
</aside>

## Three structural observations

**1. Most computation happens mid-stack.** Embedding produces near-token-identity; the final block sharpens but rarely overturns; the middle blocks (1–4 in distilGPT2; 4–10 in GPT-2 small) do the semantic work. The logit-lens trajectory shows confidence rising mid-stack and saturating at the top.

**2. Only the final position predicts the next token.** All earlier positions accumulate context that attention will later retrieve into the final position. Logit-lens predictions at non-final positions are largely incidental: the model is not optimizing them.

**3. Computation is parallel and distributed, not sequential.** The model does not execute "identify France → look up capitals → output Paris" as discrete steps. All blocks compute simultaneously on their inputs; the result is an additive sum on the residual stream. There is no step 3. There are 78 components contributing in parallel.

This is why MI does not ask "what happened at step 3?" but instead "what did head 7.4 contribute?". The latter has a precise numerical answer (DLA gives a scalar); the former does not.

## What this series has covered

| Concept | Post |
|---|---|
| The black box problem and why MI exists | post 1 |
| What mechanistic interpretability is | post 2 |
| Who's doing this work, and current goals | post 3 |
| Neurons, weights, and forward propagation | posts 4–5 |
| Layers, depth, and training | posts 6–7 |
| Transformer architecture overview | post 8 (this series) |
| Tokens and BPE | post 9 |
| Residual stream, logit lens, DLA | post 10 |
| Attention, QK / OV, IOI circuit | post 11 |
| MLPs, key-value memory, superposition | post 12 |
| Full forward pass | post 13 |

Sufficient to read most current MI papers without ambiguity.

## Where this goes next

The next chapter is **features and circuits**: applying this architectural foundation to find concrete computational structures inside trained models. Topics:

- Sparse autoencoders in depth (Anthropic 2023, 2024).
- Activation patching and causal scrubbing.
- The IOI circuit reproduction in code.
- Feature visualization and concept geometry.
- Mech interp on production models (Claude, Llama).

## Resources

### Foundational

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani et al., 2017 · the architecture</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage et al., Anthropic 2021 · residual-stream view, QK / OV decomposition</div></a></li>
  <li><a class="research-card" href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" target="_blank" rel="noopener"><div class="research-card__title">Interpreting GPT: the logit lens</div><div class="research-card__authors">Nostalgebraist, LessWrong 2020 · the lens used throughout this series</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson et al., Anthropic 2022</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2211.00593" target="_blank" rel="noopener"><div class="research-card__title">Interpretability in the Wild: a Circuit for IOI in GPT-2</div><div class="research-card__authors">Wang et al., 2022 · the canonical end-to-end circuit reverse-engineering</div></a></li>
</ul>

### Code, tools, courses

<ul class="research-list">
  <li><a class="research-card" href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noopener"><div class="research-card__title">nanoGPT</div><div class="research-card__authors">Karpathy · ~300-line PyTorch GPT-2 implementation; read the forward pass directly</div></a></li>
  <li><a class="research-card" href="https://transformerlensorg.github.io/TransformerLens/" target="_blank" rel="noopener"><div class="research-card__title">TransformerLens</div><div class="research-card__authors">the standard MI library; load any HF model and inspect every activation</div></a></li>
  <li><a class="research-card" href="https://huggingface.co/Xenova/distilgpt2" target="_blank" rel="noopener"><div class="research-card__title">Xenova/distilgpt2</div><div class="research-card__authors">Hugging Face · the model used in the grand-tour demo</div></a></li>
  <li><a class="research-card" href="https://arena3-chapter1-transformer-interp.streamlit.app/" target="_blank" rel="noopener"><div class="research-card__title">ARENA · Transformer Interpretability</div><div class="research-card__authors">guided exercises: build the logit lens, DLA, induction heads, IOI from scratch</div></a></li>
  <li><a class="research-card" href="https://www.neelnanda.io/mechanistic-interpretability/getting-started" target="_blank" rel="noopener"><div class="research-card__title">Neel Nanda's MI Getting Started</div><div class="research-card__authors">curated reading order, problem sets, and study tips</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/" target="_blank" rel="noopener"><div class="research-card__title">Transformer Circuits Thread</div><div class="research-card__authors">Anthropic · the canonical venue for new MI results; follow it to stay current</div></a></li>
</ul>
