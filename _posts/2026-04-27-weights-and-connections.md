---
layout: post-article
title: "Weights & Connections: Where Knowledge Actually Lives"
date: 2026-04-27
permalink: /posts/weights-and-connections/
excerpt: "Every fact, every grammar rule, every pattern a model ever learned is compressed into a pile of numbers. Nobody designed a single one of them."
series: "Phase 2 · Building Blocks"
series_index: 5
series_index_prev: 4
series_index_next: 6
series_total: 31
read_time_label: "10 min read"
tags: [mechanistic-interpretability, building-blocks]
---

Quick puzzle to kick us off. Say someone told you: *"Hide a trillion facts inside a pile of numbers. Go."* How would you do it?

Neural networks somehow figured this out. Every fact, every grammar rule, every pattern a model ever learned is in the weights. Not in any single weight. Not labelled. Just pressed into the collection, in a way nobody designed and nobody fully understands.

Weights are the most important thing in a neural network. They're also the hardest to read.

So let's learn to read them.

---

## What a weight actually is

A weight is one number. It lives on a connection between two neurons.

- **Positive** weight: "when the sending neuron is active, push the receiver to be more active too."
- **Negative** weight: "when the sender is active, push the receiver to be less active."
- **Near-zero** weight: "I don't care what that neuron does."

One number, one relationship, one direction of influence.

Scale this up. A model with 70 billion parameters has 70 billion of these little relationships. All learned from data. All working together to spit out something coherent on the other end.

The staggering bit: nobody wrote a single one of them. Nobody sat down and decided *"the word 'not' should have a negative weight on the sentiment neuron."* The model figured all of it out, by reading enough examples. That still feels slightly like magic to me, honestly.

{% include demos/weight-editor.html %}

Play with the demo. Flip the weight from *word positivity* to negative and watch the model's prediction invert. Positive reviews now get classified as negative. Try the **Broken** preset, then **Trained**. Nothing "inside" the model changed except a handful of numbers. That's all weights are.

## The weight matrix

When every neuron in one layer connects to every neuron in the next, you get a **weight matrix**.

Layer A has 4 neurons. Layer B has 3 neurons. You have a 4×3 grid of weights. 12 numbers, each the strength of one connection.

To calculate layer B's activations, you multiply: `B = W · A`. Matrix multiplication.

This is the fundamental operation of a neural network. Everything (attention, MLP layers, embeddings) is built from variations of this.

For interpretability, the weight matrix is where we look for structure:

- Are there patterns in which neurons have high weights to each other?
- Are there clusters all strongly positive or negative with each other?
- Can we factor the weight matrix into simpler components that mean something?

That last one, **matrix factorisation**, is a major MI technique. If `W` decomposes into `A × B`, then `A` and `B` might represent something interpretable.

## What trained weights look like

Random weights, before training: all small, roughly centred on zero. The network produces noise.

After training on language: the weights organise into structure. Not structure we designed. Structure that reflects the regularities in language.

### Word embeddings

Words get encoded as high-dimensional vectors. The weights arrange these vectors so that:

```
"king"  − "man"    + "woman"  ≈  "queen"
"Paris" − "France" + "Italy"  ≈  "Rome"
```

Not programmed. Emerges from the weights learning which words appear in similar contexts.

### Attention weight patterns

In transformers, attention weights form patterns like:

- Heads that always attend to the previous token
- Heads that look for subject-verb agreement
- Heads that copy information from far back in the sequence

These patterns live in the weight matrices. Finding them is a big chunk of mechanistic interpretability. We used the Behavioural-vs-Mechanistic demo in Post 2 to visualise exactly this.

## Why weights are hard to read directly

Here's the annoying part. Print out a weight matrix, you see a grid of numbers like `0.023, −0.41, 0.0017, 1.3, −0.88`... and?

It tells you almost nothing. The numbers only mean something in combination. One weight doesn't represent a concept. The whole matrix does.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>You can't read weights the same way you read code. You need other tools: activation analysis, weight visualisation, singular-value decomposition, probing. The weight matrix is the <em>storage</em>. The activity of neurons running through it is the <em>readout</em>. MI needs both.</p>
</aside>

## Weights vs activations

This one trips people up.

**Weights** are fixed after training. Don't change when you give the model a new input. They're the *structure*. The compiled knowledge of everything the model learned.

**Activations** are dynamic. Computed fresh for every input. The model's *current state of processing* your specific prompt.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Weights are the circuitry of a calculator. Activations are the numbers currently on the screen. The circuits are fixed; the computations change.</p>
</aside>

For MI: most research looks at activations (what's the model thinking about *this* input?) but relates them back to weights (what in the structure caused this pattern?). Both matter.

## Gradient descent made the weights

One sentence on how they got this way: during training, the model sees an example, makes a prediction, measures how wrong it was, and nudges every weight slightly in the direction that would've made it less wrong.

Do that billions of times, across trillions of words. Yes, literally trillions. Yes, it feels absurd. It also works.

The weights that emerge encode the statistical regularities of everything the model was trained on. Grammar. Facts. Logic. Poetry. Chemistry. Slang. All of it. Compressed into numbers.

Post 7 explains this in full. For now, "it's gradient descent" is enough.

## The MI connection

When an MI researcher asks *"what did this model learn to do?"*, they're asking *"what do these weights mean?"* Finding the answer requires figuring out which directions in weight space correspond to human-interpretable concepts. Which is the project of the whole field.

Next post: layers. Many weight matrices stacked on top of each other, each doing something different to the flow of information.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1301.3781" target="_blank" rel="noopener"><div class="research-card__title">Efficient Estimation of Word Representations in Vector Space</div><div class="research-card__authors">Mikolov, T. et al. · 2013 · Word2Vec</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener"><div class="research-card__title">A Mathematical Framework for Transformer Circuits</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2021 · sections on weight matrices</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2209.02535" target="_blank" rel="noopener"><div class="research-card__title">Analyzing Transformers in Embedding Space</div><div class="research-card__authors">Dar, G. et al. · 2022 · reading weight matrices directly</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2012.14913" target="_blank" rel="noopener"><div class="research-card__title">Transformer Feed-Forward Layers Are Key-Value Memories</div><div class="research-card__authors">Geva, M. et al. · 2021 · MLP weights encode factual associations</div></a></li>
  <li><a class="research-card" href="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/" target="_blank" rel="noopener"><div class="research-card__title">Neural Networks, Manifolds, and Topology</div><div class="research-card__authors">Olah, C. · 2014 · visual intuition for weight-matrix geometry</div></a></li>
</ul>
