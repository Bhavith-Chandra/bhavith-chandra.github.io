---
layout: post-article
title: "Attention: How Every Position Decides Who to Listen To"
date: 2026-05-26
permalink: /posts/attention-how-every-position-decides-who-to-listen-to/
excerpt: "The word 'it' has to figure out what noun it refers to. The verb has to find its subject. All of this happens through attention, the routing mechanism that took NLP from sad to superhuman in five years."
read_time_label: "15 min read"
accent: amber
math: true
---

At the risk of being obvious about it: if you want to understand the sentence *"The cat sat on the mat because it was warm,"* the word that has to do some work is **it**. Does "it" refer to the cat, or the mat?

Humans do this effortlessly. Early language models were terrible at it. The fix, the single biggest architectural idea in NLP in the last decade, was attention.

Attention is a mechanism by which every position in the sequence gets to peek at every other position and decide how much each one matters. The word "it" doesn't sit there clueless. It looks around the sentence, computes a weight for every other word, and pulls in the relevant information. If "cat" wins most of the weight, "it" now knows what it refers to.

That's the mechanical description. The interesting question, the one that animates modern interpretability, is: *what specific rules do specific attention heads actually learn to follow?* The answer, it turns out, is that different heads learn genuinely different, human-describable routing policies. Copy this token. Attend to the previous noun. Look for the same word earlier in the sequence and copy what came after it.

We can go read them.

---

## Play with a real model first

{% include demos/attention-explorer.html %}

This is 72 real attention heads (6 layers × 12 heads) of distilGPT2, running on prompts in your browser. Pick the John/Mary sentence. Click through layers and heads. Watch the patterns. Some heads attend only to the previous token. Some dump everything on the first position. Some have sharp, specific routes like "the first John attends to the second John." Every pattern you see is a tiny learned rule.

## The mechanism, from first principles

Each attention head has three small weight matrices: $W_Q$, $W_K$, $W_V$ (query, key, value). For every position $t$ in the sequence, with residual-stream vector $x_t$:

$$q_t = x_t W_Q \qquad k_t = x_t W_K \qquad v_t = x_t W_V$$

<div class="math-translate">Each position produces a query (what it's looking for), a key (what it offers to be found by), and a value (what it actually delivers if selected). Three linear projections of the same input vector.</div>

The attention weight from position $i$ looking at position $j$ is a softmaxed dot product of queries and keys:

$$a_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_h})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_h})}$$

<div class="math-translate">For the query at <em>i</em>, score every key. Big dot product means "these two match." Softmax turns the scores into weights that sum to 1. The <em>√</em> in the denominator keeps the scores from saturating the softmax too early.</div>

And the output at each position is a weighted sum of values:

$$o_i = \sum_j a_{ij} v_j$$

That's the head. Three tiny linear layers, a dot-product similarity, a softmax, a weighted sum. Per head. Per layer.

A *multi-head* attention layer just has many of these running in parallel with their own $W_Q, W_K, W_V$, whose outputs are concatenated and projected back into the residual stream via an output projection $W_O$. GPT-2 small has 12 heads per layer. Each one can learn a different routing rule.

## The QK circuit and the OV circuit

Here's where interpretability gets a lot of mileage. You can think of each head as implementing two separate computations, and they can be analysed independently.

### The QK circuit: "who should I look at?"

The dot product $q_i \cdot k_j$, decomposed:

$$q_i \cdot k_j = (x_i W_Q)(x_j W_K)^T = x_i (W_Q W_K^T) x_j^T$$

The matrix $W_Q W_K^T$ is the **QK circuit**. It's a bilinear form mapping pairs of residual-stream vectors to attention scores. This tells you, given the content of position $i$ and position $j$, *how much $i$ should attend to $j$*.

If we factor it: *"what kinds of queries does this head pose, and what kinds of keys light up for them?"*

### The OV circuit: "what should I write back?"

Symmetrically, the output of the head goes through $W_V$ and then $W_O$:

$$\text{write}_i = \sum_j a_{ij} x_j W_V W_O$$

The matrix $W_V W_O$ is the **OV circuit**. It maps a residual-stream vector (at a source position) to the vector that will be written to the target position (with weight given by $a_{ij}$). In plain English: *"given the head decided to pay attention to source $j$, what information does it copy from $j$ to $i$?"*

This is a powerful decomposition. **Attention is not one computation; it is two orthogonal computations composed together.** QK decides the routing. OV decides the payload. They are trained jointly but they are *separate objects*, and they can be understood separately.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The QK/OV split is the single most important analytical tool for attention. Almost every interpretability result about a specific head is ultimately an answer to "what does QK do?" plus "what does OV do?" This frame is from <a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener">Elhage et al., 2021</a> and has structured the field ever since.</p>
</aside>

## Four flavours of head you'll actually see

MI researchers have, over the last few years, catalogued several recurring head archetypes. You can hunt for them in the demo above.

### Previous-token heads

QK pattern: attend to the token one position back. OV: copy the embedding of that token. Very common in early layers. They're essentially building "what was the word just before this one?" into each position. Fundamental building block for a lot of higher-order routing.

### Induction heads

The most famous one. An induction head does **in-context copying**: if earlier in the sequence we saw the bigram "$A$ $B$", then the next time we see "$A$", the head attends to "$B$" and copies it forward. Which means: the model learns to continue patterns on the fly, without ever being trained on that specific pattern. This is widely believed to be the mechanistic basis of in-context learning ([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)).

In the demo, try the "A B C D … A B" preset and look at layer 2–3 heads. When the attention matrix has strong off-diagonal structure pointing a few tokens back, often one token back from an earlier match, you're looking at something induction-flavoured.

### Positional / syntactic heads

Heads that attend based mostly on position (e.g., "the start of this sentence") or syntactic role (e.g., "the subject of the main clause"). Often appear in middle layers.

### The BOS sink

This is a weirder one. Many heads in middle layers dump most of their attention mass onto the beginning-of-sequence token (position 0). Why? Because the softmax always has to sum to 1, it can't choose *not to attend*, so when a head has nothing useful to do on a given token, it "rests" by attending to BOS. The residual-stream update from this is small (because attending to a near-constant vector adds a near-constant amount), which is basically the head's way of staying out of the way. Sometimes called the "attention sink" ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)).

In the demo, the "BOS sink" diagnostic fires on many heads, especially in layers 1-3. Don't be alarmed.

## The indirect object identification circuit

Okay, we have to talk about the most famous circuit in MI, because it is genuinely one of the most beautiful things in the field.

[Wang et al., 2022](https://arxiv.org/abs/2211.00593) took the prompt *"When John and Mary went to the store, John gave a drink to ___"* and asked: how does GPT-2 Small correctly predict "Mary"? It's a nontrivial problem. The model has to figure out that John is the subject, Mary is the indirect object, and therefore Mary is who the drink was given to.

They traced the full circuit. It uses (roughly) seven specific attention heads working in concert. Here's the sketch:

1. **"Duplicate token" heads** (layer 0-3) notice that "John" appears twice in the sentence.
2. **"Previous-token" heads** relay information about adjacent words.
3. **"S-Inhibition" heads** (layer 7-8) write *"the subject John is not the answer"* into the residual stream at the final position.
4. **"Name-mover" heads** (layer 9-10) attend from the final position back to names, but their attention is actively suppressed toward John (because of the S-inhibition signal) and flows to Mary instead.
5. The logit for "Mary" goes up. The model predicts correctly.

The paper is a line-by-line autopsy of which head does what, verified by ablations (remove a head, watch the answer fall apart). It is essentially a reverse-engineered algorithm, in the model's own language, and it turned out to be a small piece of code you could almost write down in Python.

This was the first time a nontrivial "circuit" inside a language model got fully mapped. It launched a whole subfield.

In the demo, the John/Mary preset is the setup. Browse layers 7-10 and see which heads have strong attention from the final token toward names earlier in the sentence. You can *see* the name-mover heads.

## Causal attention and why autoregressive models only look backward

One small but important technical thing. In a **decoder-only** model (GPT, Claude, Llama, all of them), attention is *causal*: position $i$ can only attend to positions $j \le i$. This is enforced by masking, we set $a_{ij} = 0$ for $j > i$ before softmaxing.

Why? Because the model is trained to predict the next token given previous tokens. If position 5 could look at position 6, the model could cheat by just reading the answer. Causal masking forces every prediction to be made using only past information.

This is why the attention matrices in the demo are always lower-triangular, everything above the diagonal is zero. If a head wants to attend to "the future" it cannot. The only information it has is the tokens it has already seen.

## The softmax is doing something load-bearing

One thing worth pausing on. The softmax in attention is not just a normalisation; it is the mechanism that gives attention its selectivity. If it were a simple sum or average, every position would contribute equally and the routing would be useless. Softmax is the reason the head can produce sharp, peaky attention patterns, a single attended-to position getting most of the weight.

That selectivity is also a constraint. Softmax forces the weights to sum to exactly 1, which creates the "must attend to something" effect that produces BOS sinks. It's a design choice with downstream consequences. Some recent architectures (YOCO, RetNet) swap softmax for other mixing schemes, partly to escape these quirks.

## Attention is the memory

Zooming out. Why does attention work so well? One frame that's been useful to me: **attention is associative memory**.

When you type a question to an LLM, the attention mechanism is letting the final token (where the prediction happens) reach back through the entire context and *retrieve* relevant pieces. It's not storage. It's retrieval.

The MLPs, by contrast, store things, weights learned during training encode facts that don't change per-prompt. Attention is the look-up. MLPs are the filing cabinet. A transformer alternates them because generation requires both: retrieve what's in front of me, transform it using what I've memorised, repeat.

That's a useful duality to hold onto. I'll talk about the MLP half in the next blog.

## The takeaway, in three bullets

1. Attention is a dot-product-based routing mechanism between positions. Every position queries, every position offers keys, the soft-matched queries pull values.
2. Each head is best understood as two orthogonal matrices: **QK** (who to attend to) and **OV** (what to write back). Both are small, both are analyzable.
3. Real heads learn concrete, human-readable rules, previous-token, induction, name-movers. Finding and characterising them is what circuit-level MI does.

Next up: the MLP. The other half of every block, and where most of the model's factual knowledge turns out to live.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani, A. et al. · 2017 · the original mechanism</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021 · QK/OV decomposition</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson, C. et al. · Anthropic, 2022</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2211.00593" target="_blank" rel="noopener"><div class="research-card__title">Interpretability in the Wild: a Circuit for IOI in GPT-2</div><div class="research-card__authors">Wang, K. et al. · 2022 · the indirect-object identification circuit</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2309.17453" target="_blank" rel="noopener"><div class="research-card__title">Efficient Streaming Language Models with Attention Sinks</div><div class="research-card__authors">Xiao, G. et al. · 2023 · the BOS-sink phenomenon</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2016/augmented-rnns/" target="_blank" rel="noopener"><div class="research-card__title">Attention and Augmented Recurrent Neural Networks</div><div class="research-card__authors">Olah, C. & Carter, S. · Distill, 2016 · best illustrated early intro</div></a></li>
</ul>
