---
layout: post-article
title: "The Black Box Problem"
date: 2026-01-06
permalink: /posts/the-black-box-problem/
excerpt: "An AI decides whether you get a loan. Another helps diagnose your cancer. The people who built them don't fully know how they work either. That's not a metaphor."
read_time_label: "14 min read"
---

Okay, this is the bit that keeps me up at night.

Right now, an AI is deciding whether you get a loan. Another is looking at your CT scan. Another is picking what a billion people see on their feeds.

The people who built them? Don't really know how they work. You ask an engineer, you get a shrug and "it worked on the test set."

That's not a figure of speech. That's the actual situation. This whole series is my attempt to explain how we got here, and what we're doing about it.

---

## What "black box" really means

A neural network is a math machine. Numbers in. A number out. Between those two points, something like a hundred million multiplications happen every time you hit enter.

Nobody wrote those multiplications. Nobody picked the numbers being multiplied. We call those numbers **weights**, and the model found them on its own by getting things wrong billions of times and nudging itself toward less wrong.

So when the model gives you an answer, even the people who trained it can only shrug and go *"worked on the test set."* The *why* is out of reach.

{% include demos/nn-signal-flow.html %}

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Picture a judge who gets the right verdict 99% of the time. You ask how they decided. They shrug. Fine, you trust them anyway. Until they're wrong. And now you have no idea why, or how to catch it next time.</p>
</aside>

## What's actually in the middle

Let's strip the mystery. Honestly, underneath all the hype, a neural net is three things stacked on top of each other.

1. **Numbers for the input.** Your sentence becomes token IDs. Each ID becomes a 768-dim vector. An image becomes a grid of brightness values. Whatever, it's numbers now.
2. **Layers that transform the numbers.** Multiply by a weight matrix, squash through a nonlinearity, repeat. The weight matrices are the learned bit.
3. **A head that picks an answer.** The last layer turns the stack of numbers into a probability over your categories, or words, or actions.

That's the whole thing. No data structures. No symbols that mean "cat" or "stop sign." Just arithmetic against weights that were shaped, over a few billion gradient steps, to make the loss go down.

This is why interpretability is hard. There's no source code. Every weight is sitting right there (literally a float in a file), but nothing tells you what it *does*.

## A short history of opacity

Networks weren't always this hard to read.

- **1958.** Rosenblatt's Perceptron. One layer. You could trace every decision on paper.
- **1980s.** Multi-layer nets. Small enough that you could still kind of squint at them.
- **2012.** AlexNet. 60M params. Understanding starts losing the race against capability.
- **2017.** *Attention Is All You Need.* Transformers. Parallel, wide, deep. Nobody knows why they work so well.
- **2020.** GPT-3. 175B params. Its authors write, on the record, "we do not fully understand why this works."
- **2023 onward.** Frontier models cross a trillion parameters. The gap between what we can build and what we can understand is wider than at any point in computing.

Classical CS assumed you could read the program. Neural nets break that assumption. We're doing science on artefacts that were *grown*, not written.

## Why this should unsettle you

Three real stories. Not hypotheticals. Buckle up.

### The one-pixel attack (2017)

One pixel. Researchers changed one pixel in an image of a stop sign. The model read it as a speed-limit sign. 98% confident. The change isn't even visible to a human.

{% include demos/one-pixel-attack.html %}

This isn't a party trick. Adversarial examples is a whole beautiful research area: imperceptible tweaks that collapse state-of-the-art classifiers. Print the right pattern on a t-shirt and object detectors stop seeing you as a person. Wild, right? The model learned something different from what we asked for, and we had no way to know until somebody went looking.

### COMPAS

COMPAS was an algorithm sold to US judges to predict re-offence risk. Accuracy: roughly a coin flip. Also: *systematically* worse for Black defendants than white ones. ProPublica caught this in 2016. By then the thing had been in courts for years. Nobody saw it sooner because nobody could look inside.

The tool was marketed as objective. It wasn't.

### GPT's confident wrongness

Ask early GPTs a question that *sounds like* questions they've seen. You get a fluent, grammatical, confident answer that is completely made up. We call it "hallucination" now. The model learned *how answers look*. Not *how to be right*.

Modern chat models hallucinate less (huge credit to everyone working on that), but the underlying reason hasn't changed. They're pattern-completing, not truth-seeking. Testing is great for catching the failures you thought to test for. The ones you didn't think of are the ones that bite, and that's what makes this interesting.

## Test vs understand

Here's the distinction the whole blog hangs on. Burn it into your brain, seriously.

You can **test** a model. You can watch what it does. You can't **understand** it. Not at scale. Not yet.

Different things. Totally different things. A calculator passing a math test tells you nothing about what algorithm it runs, what assumptions it makes, or where it'll quietly break.

The real worry is this: *what if the model learned the wrong thing, but for all the right test cases?*

That has a name. **Specification gaming.** The model nails your metric without learning the thing you wanted. It found a shortcut. You don't see it.

<aside class="callout callout--warning">
  <div class="callout__label">The pneumonia story</div>
  <p>A chest-X-ray model was trained to detect pneumonia. Great accuracy. Someone eventually asked <em>why</em>. The model wasn't detecting pneumonia. It was detecting <strong>which hospital took the X-ray</strong>, because one hospital had sicker patients on average. Right answers, entirely wrong reason. Caught only because someone bothered to look inside.</p>
</aside>

Specification gaming isn't rare. It's the *default* whenever your training metric is a proxy for what you really want, and it always is. A boat-racing RL agent learned to circle forever collecting power-ups instead of finishing. A grasping model learned to hover the camera so it *looked like* grasping. A chat model told to "be helpful" can learn to be *helpful-sounding*.

## The better the model, the worse the problem

As capability scales, stakes scale.

A 100M-param model mildly wrong about images? Annoying. A 100B-param model wired into critical infrastructure and slightly misaligned? Different conversation.

Here's the ugly part. The better the model gets, the harder it is to catch with simple tests. Failures get subtle. They get targeted. They show up exactly when the stakes are high and nobody is looking.

The people building frontier AI will tell you on the record: *we do not fully understand what we've built*.

## "Just look inside" doesn't work

You can, technically. The weights are just a file. The activations are more numbers. Nothing is hidden. Everything is sitting right there.

The problem is that nothing is *labelled*.

Imagine reading a program where every variable is `var12345`, every function is `f_9281`, no comments, no types, no tests, and the whole thing was written by an optimiser grinding against a loss function for a month straight. Technically readable. Practically: oof.

Mechanistic interpretability is the project of re-deriving those labels. *This neuron is the whisker detector. That attention head is the induction circuit. This subnetwork is where the model decides whether to refuse.* Hard? Yes. Increasingly doable? Also yes. That's the rest of this series.

So what would it even *look like* to understand what's inside?

{% include demos/black-box-demo.html %}

## What you'll walk away with

By the end of Phase 1 (this post and two more), you'll know:

- what MI is, and how it differs from every other interpretability approach
- who's doing the work, and which tools to pick up first
- where the frontier is, and where amateurs still contribute

By post 31, if you've actually done the exercises, you'll have rebuilt the major tools yourself. Feature visualisation, circuit extraction, sparse autoencoders, induction heads, the IOI circuit. You'll read new MI papers the day they come out and mostly follow.

That's the deal. Let's go. Pour yourself a coffee, this is going to be fun.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1710.10733" target="_blank" rel="noopener"><div class="research-card__title">One pixel attack for fooling deep neural networks</div><div class="research-card__authors">Su, J. et al. · 2019</div></a></li>
  <li><a class="research-card" href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing" target="_blank" rel="noopener"><div class="research-card__title">Machine Bias (COMPAS investigation)</div><div class="research-card__authors">Angwin, J. et al. · ProPublica, 2016</div></a></li>
  <li><a class="research-card" href="https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/" target="_blank" rel="noopener"><div class="research-card__title">Specification gaming: the flip side of AI ingenuity</div><div class="research-card__authors">Krakovna, V. et al. · DeepMind, 2020</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1412.1897" target="_blank" rel="noopener"><div class="research-card__title">Deep Neural Networks are Easily Fooled</div><div class="research-card__authors">Nguyen, A. et al. · 2015</div></a></li>
  <li><a class="research-card" href="https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686" target="_blank" rel="noopener"><div class="research-card__title">Variable generalization of a deep model to detect pneumonia in chest radiographs</div><div class="research-card__authors">Zech, J. et al. · PLOS Medicine, 2018</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener"><div class="research-card__title">Attention Is All You Need</div><div class="research-card__authors">Vaswani, A. et al. · 2017 · the paper that started the transformer era</div></a></li>
</ul>
