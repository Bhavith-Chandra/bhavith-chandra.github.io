---
layout: post-article
title: "The Residual Stream: The Belt That Runs the Whole Factory"
date: 2026-05-12
permalink: /posts/the-residual-stream/
excerpt: "If you understand one thing about transformers, make it this. The residual stream is the shared scratchpad that every block reads, edits, and writes to. It is where the model's thoughts live, and it is the main object of modern interpretability."
read_time_label: "13 min read"
accent: amber
math: true
---

I'm going to do something slightly unusual. I'm going to spend an entire post on a single idea, because that idea is load-bearing for basically every interpretability result from 2021 onward. If you skim this post, skim something else instead.

The idea: the **residual stream**.

Informally: the conveyor belt from the last post. Formally: a running sum, one vector per token position, that every block in the network reads from and writes to, and that stays coherent, in the same "language", from the input embeddings all the way to the final logits.

That last part is the deep bit. Most of the cleverness in modern MI comes from taking that fact seriously.

---

## Run one yourself first

{% include demos/residual-stream.html %}

Click a preset. The model will download once (~85MB, cached), run a real forward pass, and show you what I'm about to explain. Each cell is the logit-lens prediction at that layer × token position. Watch the final column evolve as you read bottom-to-top: early layers say nothing meaningful, middle layers drift toward a semantic category, the top layer locks in the answer.

This is not a toy. That's literally what a real model does when you prompt it.

## What the stream actually is

A transformer with $L$ blocks, processing a sequence of $T$ tokens with hidden dimension $d$, maintains a tensor of shape $[T, d]$ throughout the forward pass. Call this $X$.

Starting state: embed each token. $X_0 = E$ (plus positional info).

Then each block updates:

$$X_{\ell+1} = X_\ell + \text{Attn}_\ell(X_\ell) + \text{MLP}_\ell(X_\ell)$$

<div class="math-translate">In words: at each block, take whatever is currently on the belt, compute an attention update and an MLP update, and <strong>add them to the belt</strong>. Do not overwrite. Do not replace. Add.</div>

After $L$ blocks you have $X_L$. The unembedding layer reads $X_L$ at the last position and produces the distribution over next tokens.

The key feature is that **plus sign**. It changes everything.

## Why "+" is the whole ball game

Imagine you designed the network differently. Each block *replaces* the previous state:

$$X_{\ell+1} = \text{block}_\ell(X_\ell)$$

This is how classical feed-forward networks work, and it's how RNNs work. It has two problems:

**Problem 1: vanishing information.** Layer 1 computes something useful, say "this token is a noun." Layer 2 transforms it. Layer 3 transforms it further. By layer 40, that "noun" information has been funnelled through 39 non-linear transformations and is almost certainly gone, unless every intermediate layer somehow chose to preserve it. Which they won't, because they're not designed to.

**Problem 2: vanishing gradients.** Training requires sending error signals backward through every layer. With each multiplication the signal shrinks. 40 layers in, the gradient that should be telling the early layers what to learn is numerically zero. The deep parts of the network stop learning entirely.

Both problems were well known by 2015. The fix, in vision and then in language, was residual connections ([He et al., 2015](https://arxiv.org/abs/1512.03385)). Add instead of replace. Now:

- Information from layer 1 can flow unchanged all the way to layer $L$, it's still in there, mixed with everything added after.
- Gradients flow backward through the "+" without shrinking. Deep networks train stably.

For a transformer, this becomes the residual stream: one shared vector per position, accumulated across all blocks. Each block writes a (usually small) correction. The final state is the sum of every correction ever made to each position.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Picture a Google Doc that a whole team of editors is working on at once. Nobody's allowed to delete anything; everyone can only add a comment or a suggestion. At the end of the day the document is the sum of everyone's contributions. To figure out what the document "means," you don't have to interview every editor, you just read the document. To figure out what <em>one editor</em> contributed, you look at the diff they made. The document is the residual stream. The editors are the layers.</p>
</aside>

## Reading the stream: the logit lens

Here's the move that made modern MI possible.

The final output of the transformer is produced by projecting $X_L$ through the **unembedding matrix** $W_U$ and taking a softmax:

$$p(\text{next token}) = \text{softmax}(X_L \cdot W_U)$$

The clever observation ([Nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)): because every intermediate $X_\ell$ lives in the same vector space as $X_L$, we can apply $W_U$ to $X_\ell$ as well. This gives us, at every layer, the model's *current best guess* at the next token if it had to commit right now.

This is called the **logit lens**, and it's what the demo above is showing you.

Run it on "Paris is the capital of ___" and watch the evolution.

- **Embedding layer.** Junk. Maybe "the" or "a". The raw token embedding for "of" has no idea what it wants to predict next, it just contains info about the token itself.
- **Block 0–1.** Still pretty junky, but starting to look vaguely like geography words. "the", "a", "this".
- **Block 2–3.** Converging on the semantic type. Country names start showing up in the top-5. "Europe", "France", "Germany".
- **Block 4–5.** Top-1 is "France" with high confidence. The answer has arrived.

The logit lens *works* because the residual stream is coherent from start to finish. That's a non-trivial fact about the architecture. You couldn't do this on most other networks.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The logit lens is a free, training-free tool for seeing what a model is thinking at every intermediate layer. Many modern techniques, tuned lens, direct logit attribution, activation patching, are refinements of or close cousins to this idea. If you learn one interpretability technique, learn this one first.</p>
</aside>

## Direct logit attribution

Now the payoff. Because the residual stream is a *sum* of contributions, you can decompose the final logit of any token into a sum of contributions from every block:

$$\text{logit}(w) = \left(\sum_{\ell=0}^{L} \Delta_\ell\right) \cdot W_U[:, w] = \sum_{\ell=0}^{L} \Delta_\ell \cdot W_U[:, w]$$

<div class="math-translate">The final score for word <em>w</em> is the sum of dot products between every layer's contribution and <em>w</em>'s unembedding vector. Each term says: "how much did <em>this</em> layer push the prediction toward <em>w</em>?"</div>

This is **direct logit attribution (DLA)** and it's the Swiss army knife of transformer interpretability. Want to know which block was responsible for the model answering "France" correctly? Compute each layer's contribution to the logit of "France". The one with the biggest contribution is where the key computation happened. Drill further: attention or MLP? Then which head or which neuron? This is how you trace a model's reasoning back to specific components.

## Why the "channels" metaphor is helpful (and also a lie)

You'll sometimes hear the residual stream described as a bunch of "channels", loosely, dimensions, each carrying a specific piece of information. Position 0 channel 5 might be "this token is a verb." Position 0 channel 73 might be "this token is the subject of the sentence."

This is kind of true and kind of not. The real situation ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)):

- The residual stream has $d$ dimensions (768 for GPT-2, 4096 for Llama 3, much bigger for frontier models).
- Blocks read from and write to specific *subspaces* of the stream.
- Those subspaces are generally not axis-aligned, a "feature" is a direction in the space, not a single coordinate.
- Many features live on top of each other in **superposition**, more features than there are dimensions, packed in by relying on sparsity.

So "channels" is a useful first approximation. The refined view is "directions in a high-dimensional space, many of them overlapping." We'll spend serious time on superposition in later posts, because it's *why* neurons are often polysemantic. For now, think of the stream as a busy 768-lane highway where lanes overlap and cars can ride partly in two lanes at once. Fine? Fine.

## A concrete thing to believe

Let me drive one fact home because it will save you time reading papers.

> **Every major transformer phenomenon can be described as "something reads from the residual stream at some position, computes something, and writes something back to the residual stream at (possibly a different) position."**

Copy heads read from an earlier position and write a copy to the current position. Induction heads read a match-detection signal from one direction and write a "copy this token" signal to another. Factual-recall MLPs read from subject tokens and write facts about them. IOI heads juggle subject and object between positions to figure out who the indirect object is.

All of it. Read-from, compute, write-to. The *language* of the model is "operations on the residual stream."

This is why Elhage et al. call the residual stream **"the central object of the transformer."** Not the attention heads. Not the MLPs. The stream.

## Reading the logit lens heatmap

Pop back up to the demo and stare at the heatmap colour.

Amber saturation = the model's confidence in its top guess at that cell. Watch how the saturation grows as you scan from bottom (embedding) to top (final block). For most prompts you'll see:

1. The saturation increases monotonically.
2. The *top-1 token* changes 2-3 times as layers refine.
3. The final column (the last token of the input) is where prediction happens, earlier columns are mostly the model maintaining information about those tokens rather than predicting anything.

That last observation is important. The model only needs to predict *one* next token. All the work on earlier positions is setup, those positions are accumulating information that the attention mechanism will later pull into the final position. They're not trying to predict; they're trying to be *useful*.

## The BOS token scratchpad

One charming empirical finding from [Scaling Monosemanticity (Templeton et al., 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): the beginning-of-sequence token's residual stream ends up carrying a bunch of model-wide "housekeeping" state. Averaged running statistics, attention sinks, odd summary features. Kind of like a blank page at the front of the doc where the model scribbles notes to itself that don't correspond to any particular input token.

You don't need to understand why right now, just note that the first token's stream often looks weird in diagnostics, and this is the reason.

## What to take away

1. **The residual stream is a running sum**, not a sequence of replacements.
2. **Every block adds a correction.** Nothing is ever overwritten.
3. **Intermediate states live in the same space as the output**, which is what makes the logit lens work.
4. **Interpretability, at its core, is the study of what gets read from and written to the stream, and by what components.**

Hold onto that last one. When we get into attention next, I'll unpack the QK/OV decomposition, which is exactly the language of "what this head reads, and what it writes." Same with MLPs after that. Every component we study is best understood as a reader-writer on this shared belt. I'll get to attention in the next blog.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener"><div class="research-card__title">Deep Residual Learning for Image Recognition</div><div class="research-card__authors">He, K. et al. · 2015 · ResNet, the original residual-connection paper</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021 · introduces the residual-stream view</div></a></li>
  <li><a class="research-card" href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" target="_blank" rel="noopener"><div class="research-card__title">Interpreting GPT: the logit lens</div><div class="research-card__authors">Nostalgebraist · LessWrong, 2020 · the technique used in the demo above</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2303.08112" target="_blank" rel="noopener"><div class="research-card__title">Eliciting Latent Predictions from Transformers with the Tuned Lens</div><div class="research-card__authors">Belrose, N. et al. · 2023 · a refined version of the logit lens</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" target="_blank" rel="noopener"><div class="research-card__title">In-context Learning and Induction Heads</div><div class="research-card__authors">Olsson, C. et al. · Anthropic, 2022 · heavy use of residual-stream decomposition</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html" target="_blank" rel="noopener"><div class="research-card__title">Scaling Monosemanticity</div><div class="research-card__authors">Templeton, A. et al. · Anthropic, 2024 · residual-stream features in Claude 3 Sonnet</div></a></li>
</ul>
