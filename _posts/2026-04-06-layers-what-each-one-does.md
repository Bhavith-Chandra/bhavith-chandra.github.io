---
layout: post-article
title: "Layers: What Each Floor of the Building Does"
date: 2026-04-06
permalink: /posts/layers-what-each-one-does/
excerpt: "Layer 1 sees edges. Layer 5 sees this specific person. Same pixels, different lens at every level. That's why deep learning is deep."
read_time_label: "11 min read"
---

Picture this: you show a trained network a photo of a face.

Layer 1 sees edges. Diagonal lines, curves, horizontal stripes.
Layer 2 sees eye corners, nose tips, ear lobes.
Layer 5 sees *this person*. Their mood. Whether they're wearing glasses.

Same pixels going in. Totally different lens at every level. Each layer is watching the one below it and writing down what it noticed. It's like a game of telephone, except the message gets *smarter* at every hop.

That's what depth is. That's literally why "deep" learning is deep.

---

## What a layer actually is

A layer is a group of neurons that all receive the same inputs and all produce outputs that feed forward together.

Every neuron in a layer:

- Takes all the activations from the previous layer as input
- Applies its own weights to them
- Produces its own single activation value
- Passes that to every neuron in the next layer

Net result: each layer **transforms its inputs into a new representation**. Same information, viewed from a different angle, with different things highlighted.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>A series of photographers all shooting the same scene through different lenses. Early photographers use wide angles, capturing everything abstractly. Later photographers zoom in on specific meaningful details. By the end the picture has become a sentence: <em>"this is a cat, outdoors, late afternoon."</em></p>
</aside>

{% include demos/layer-explorer.html %}

## Input and output layers

**Input layer.** No math. Just receives the raw data and passes it forward. Images: pixel values. Text: token embedding vectors. Tabular data: feature values. The input layer is pure format conversion. "Here's the world, in number form."

**Output layer.** The final decision. Classification: a probability over each class (softmax). Language modelling: a probability over every word in the vocabulary (often 50,000+ options). Regression: a single continuous number.

Everything interesting happens between these two.

## Hidden layers, the middle of the machine

Don't let the word fool you. "Hidden" just means "not the input, not the output." These layers are where all the action happens, and where MI spends basically all its time.

In early vision models, researchers noticed the hidden layers had a striking, almost biological structure. Here's the rough gradient:

### Layer 1: Gabor filters

Neurons respond to oriented edges. Horizontal, vertical, 45-degree. Nobody programmed this. Every model trained on natural images independently discovers it. Emerges from the statistics of images themselves.

### Layer 2: Textures and simple shapes

Combinations of edges form textures. Checkerboards. Crosshatches. Dots. Neurons looking for local patches that match a pattern.

### Layer 3–4: Object parts

Eyes. Wheels. Leaves. Neurons now looking for parts of real objects. Things that have names in human language.

### Layer 5–7: Objects and scenes

Full objects, faces, specific categories. High-level human concepts represented in the network's internal language.

This progression is called **hierarchical feature extraction**, and it appears in every deep network trained on natural data. Images, text, audio. The depth lets the model compose simple features into complex ones, repeatedly.

## What layers do in language models

In text transformers, the same hierarchy exists, but it's harder to see because language doesn't have the obvious spatial structure of images.

Research into transformer layers has found patterns like:

- **Early (1–4).** Local, syntactic. Nearby word relationships. Part of speech. Simple co-occurrence.
- **Middle (5–16).** Syntactic structure. Subject-verb relationships. Clause boundaries. Entity tracking.
- **Late (17–final).** Semantic, pragmatic. What the text means. Who's saying what to whom. What should come next.

Not perfectly clean (features mix across layers), but the gradient from syntax to semantics is consistent and has been verified by probing experiments across many models.

### The logit lens

A cool MI technique that reads out the model's *best guess* at each layer, before the final output. Early layers, the guess is mostly garbage. Middle layers, it starts approaching the right semantic category. Late layers, it converges on the final answer. Shows you how the model builds its answer progressively. We'll build a demo of this in Phase 7.

## Depth vs width

**Width** = more neurons per layer. More "workers" doing parallel analysis at each step.

**Depth** = more layers. More steps of abstraction before the final answer.

Both help, but differently.

Width gives the model more capacity to represent complex things at each level of abstraction. Depth gives the model more steps to compose simple patterns into complex ones.

Modern networks are both wide and deep. GPT-4 is estimated to have around 120 layers. Many of those layers have ~25,000 neurons each. Which, you know, is a lot.

For interpretability: more layers means more places for information to transform. Also means there's more "room" for information to be stored in intermediate representations. Which is one reason large models are more capable but also harder to interpret.

## Skip connections, the highway system

In modern networks (including all transformers), there's a trick that changed everything: **residual connections**. Also called skip connections.

Instead of each layer *replacing* the previous layer's output entirely, it **adds** to it. The output of layer N+1 = what layer N produced + what layer N+1 computed.

Sounds small. It's enormous.

Means information can flow directly from early layers to late layers without passing through every layer in between. Early features don't get "forgotten" or overwritten.

Also means each layer can focus on adding a *small correction*, rather than computing everything from scratch. Makes training much more stable and allows much deeper networks.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>The <strong>residual stream</strong> (the accumulated sum of all layers' outputs) is one of the core concepts in transformer interpretability. You'll see it everywhere in Phase 3. For now, remember: in a modern network, information doesn't get overwritten layer by layer; it gets added to.</p>
</aside>

## The MI connection

Understanding what each layer does is one of the central projects of mechanistic interpretability. Not "layer 7 does something useful". *Exactly* what. Which features live in which layers. Which operations happen where. When we know that, we can start to decompose a model's behaviour the same way you'd decompose a program into functions.

Okay, one more thing before we go. Every weight we've talked about was set by a single procedure: gradient descent. That's what made any of this possible. Let's actually watch it work.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1311.2901" target="_blank" rel="noopener"><div class="research-card__title">Visualizing and Understanding Convolutional Networks</div><div class="research-card__authors">Zeiler, M. & Fergus, R. · 2013 · the original CNN layer-visualisation paper</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2017/feature-visualization/" target="_blank" rel="noopener"><div class="research-card__title">Feature Visualization</div><div class="research-card__authors">Olah, C. et al. · Distill, 2017 · beautiful interactive article</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/zoom-in/" target="_blank" rel="noopener"><div class="research-card__title">Zoom In: An Introduction to Circuits</div><div class="research-card__authors">Olah, C. et al. · Distill, 2020 · the layer hierarchy in vision</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener"><div class="research-card__title">Deep Residual Learning for Image Recognition</div><div class="research-card__authors">He, K. et al. · 2015 · ResNet, the skip-connection paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1905.05950" target="_blank" rel="noopener"><div class="research-card__title">BERT Rediscovers the Classical NLP Pipeline</div><div class="research-card__authors">Tenney, I. et al. · 2019 · syntax early, semantics late</div></a></li>
  <li><a class="research-card" href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" target="_blank" rel="noopener"><div class="research-card__title">Interpreting GPT: the logit lens</div><div class="research-card__authors">Nostalgebraist · LessWrong, 2020 · layer-by-layer prediction evolution</div></a></li>
</ul>
