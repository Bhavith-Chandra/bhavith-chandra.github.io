---
layout: post-article
title: "Neurons & Activations: The On/Off Switch That Isn't"
date: 2026-04-26
permalink: /posts/neurons-and-activations/
excerpt: "Your brain has 86 billion neurons. GPT-4 has about 1.8 trillion. Same word, not remotely the same thing. And the AI version is stunningly simple."
series: "Phase 2 · Building Blocks"
series_index: 4
series_index_prev: 3
series_index_next: 5
series_total: 31
read_time_label: "9 min read"
tags: [mechanistic-interpretability, building-blocks]
---

Your brain: 86 billion neurons. GPT-4: about 1.8 trillion "neurons".

Same word. Not remotely the same thing, and this bothers me a little.

The AI version is embarrassingly simple. I could sketch one on a napkin and you'd get it in thirty seconds. Stack trillions of napkin sketches together, though, and you get something that writes poetry, debugs code, and beats grandmasters at chess. That mismatch is basically the whole story.

---

## What a neuron actually does

An artificial neuron does one thing: takes a list of numbers, does some arithmetic, spits out a single number. That's the entire spec.

Say the neuron's job is to decide *is this review positive?* It gets three inputs:

- `0.9`. How positive the words are.
- `0.2`. How many exclamation marks.
- `0.7`. How long the review is.

Multiply each by its own **weight**, which is just a number saying how much the neuron cares about that input:

- positivity × `+0.8` (a lot)
- exclamations × `+0.1` (a little)
- length × `−0.3` (negative: long reviews tend to be more critical)

Add them up. That sum is the neuron's *raw opinion*.

Then (this is the part nobody explains) push that sum through an **activation function** before passing it on.

{% include demos/neuron-builder.html %}

## Why the activation function matters

Okay, bear with me for a sec. Without an activation function, stacking layers is completely pointless. Every layer does the same kind of math, so they all collapse into one. You literally just get a line. No matter how many layers you pile on. Twelve layers, twelve hundred layers, same line.

The activation function is what lets a deep network represent things that can't be drawn with a straight line. Bends the space. Curves become possible. *Non-linearity*, if you want the jargon.

Three you'll meet:

- **Sigmoid.** Squishes any input into 0 to 1. An S-curve. Used to mean "probability". Classic, elegant, still gets used in output layers. It was the default for a long time and earned its keep.
- **Tanh.** Same S, but outputs −1 to 1. A nice upgrade on sigmoid. Still pops up in places.
- **ReLU.** `max(0, x)`. Negative in, zero out. Positive in, pass it through unchanged. That's the whole function.

ReLU's simplicity is the superpower. Doesn't saturate. Doesn't vanish. And when a ReLU neuron outputs zero, it's not half-committed. It's *silent*. A clean, readable state. Which turns out to matter a lot when you're trying to reverse-engineer what the thing is doing.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Sigmoid is a dimmer switch with friction. It resists going all the way on or off. ReLU is a gate: nothing, then a straight open pipe. Gates are easier to reason about. That's most of why ReLU won.</p>
</aside>

## What "firing" actually means

Pop-sci loves *"a neuron fires"*. Great image, very evocative, gets people interested. Just not quite what's happening here.

Artificial neurons are dimmer switches, not lightbulbs.

With ReLU you're either silent (zero) or linearly active (some positive number proportional to how excited the neuron is). With softmax, used at the output layer, every neuron outputs a number and the set sums to 1.0. A probability distribution.

The useful thing isn't on vs off. It's the *degree*. *Slightly*, *strongly*, *barely*. That continuous answer is what lets networks handle fuzziness. Which is most of what real-world data is.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>When we say a neuron "responds to cats", we mean: inputs with cats make this neuron fire harder than average inputs. We can measure that. We can rank the inputs that excite it most. That's how we find features.</p>
</aside>

## Layers, a panel of critics

One neuron doesn't do much. A layer of them (512, 1024, 4096 neurons all looking at the same input, each with different weights) can represent rich structure.

Think of a layer as a panel of critics watching the same movie. One obsesses over pacing. One about dialogue. One only cares about cinematography. Each outputs a score. Together their scores paint a richer picture than any single critic could.

The next layer reads the scores and forms *opinions about opinions*. Each layer summarises the one below it into something more compact and more meaningful. That's how depth becomes abstraction.

## Bias, the neuron's default mood

One last thing, almost always skipped: biases.

Every neuron has a **bias**, a number added to its weighted sum before the activation. The neuron's default opinion before it sees any input. Positive bias? The neuron is eager; it takes real negative input to shut it up. Negative bias? It's reluctant; only strong signals wake it.

Biases let each neuron have its own firing threshold. Without them, every neuron would need its inputs to sum to exactly zero to stay silent, which is far too rigid.

## The MI connection

This whole setup (weights times inputs, plus bias, through an activation) is what mechanistic interpretability is trying to read at scale.

*Finding features* = figuring out which neurons fire, on which inputs, why.
*Finding circuits* = figuring out which neurons talk to which other neurons, and what that conversation computes.

Everything in this series is built on this one tiny unit. Next post: what happens when you wire a million of them together, every neuron talking to every other neuron. That's where things get weird.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://www.nature.com/articles/nature14539" target="_blank" rel="noopener"><div class="research-card__title">Deep learning</div><div class="research-card__authors">LeCun, Bengio, Hinton · Nature, 2015 · canonical overview</div></a></li>
  <li><a class="research-card" href="https://proceedings.mlr.press/v9/glorot10a.html" target="_blank" rel="noopener"><div class="research-card__title">Understanding the difficulty of training deep feedforward networks</div><div class="research-card__authors">Glorot & Bengio · 2010 · the vanishing-gradient paper</div></a></li>
  <li><a class="research-card" href="https://icml.cc/Conferences/2010/papers/432.pdf" target="_blank" rel="noopener"><div class="research-card__title">Rectified linear units improve restricted Boltzmann machines</div><div class="research-card__authors">Nair & Hinton · ICML 2010 · the original ReLU paper</div></a></li>
  <li><a class="research-card" href="https://transformer-circuits.pub/2022/solu/index.html" target="_blank" rel="noopener"><div class="research-card__title">Softmax Linear Units</div><div class="research-card__authors">Elhage, N. et al. · Anthropic, 2022 · MI-motivated activation redesign</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/" target="_blank" rel="noopener"><div class="research-card__title">Thread: Circuits</div><div class="research-card__authors">Cammarata, N. et al. · Distill, 2020</div></a></li>
</ul>
