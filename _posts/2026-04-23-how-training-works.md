---
layout: post-article
title: "How Training Works: The Ball Rolling Downhill"
date: 2026-04-23
permalink: /posts/how-training-works/
excerpt: "Every weight started life as a random number. All the grammar, all the facts, learned by being wrong billions of times. This has a name, and it's the closest thing AI has to a creation story."
read_time_label: "12 min read"
---

Every weight in a neural network started life as a random number. Really. Random.

All the grammar, all the facts, all the reasoning patterns. Learned by being wrong, measuring how wrong, and nudging. Billions of times. For months.

This procedure has a name: **gradient descent**. And honestly, it's the closest thing AI has to a creation story.

---

## The problem, how do you teach a machine?

You can't write rules for recognising cats. Smart people spent decades on this, did genuinely incredible work along the way, and it turned out the hand-written-rules approach just hits a ceiling. That's how we ended up over here.

You can't manually set 70 billion weights either. Even if you magically knew what they should be (you don't, nobody does), it'd take longer than the age of the universe.

So what do you do? Show the model examples. Tell it when it's wrong. Let it adjust itself. Repeat.

Do that enough times and, somehow, it works. That's machine learning in one sentence. "Gradient descent" is just the specific algorithm that does the adjusting.

## The loss function, measuring wrongness

Before the model can learn from being wrong, you need a way to measure *how wrong* it is. That's the **loss function**.

Classification problem: you ask the model *is this email spam?*. It outputs a probability: `0.73` (73% chance spam). The correct answer is `1.0`. The loss is a number measuring the gap. How far off were you?

Common loss functions:

- **Cross-entropy** (main one for classification). Measures the difference between the model's probability distribution and the true distribution. 99% confident in the right answer, loss ≈ 0. 1% confident in the right answer, loss is large.
- **Mean squared error** (regression). The average squared distance between predictions and correct answers.

The loss is a single number. Low = good predictions. High = bad predictions.

Goal of training: **minimise the loss**, averaged over millions of examples.

## The loss landscape, a mountain range of wrongness

Way to picture it.

Imagine a two-dimensional landscape. Hills and valleys. Every point corresponds to a specific setting of all the weights. The height at any point is the loss. How wrong those weights make the model.

The model starts at a random point in this landscape. A random mountain, somewhere. Our job: walk it to the nearest valley. The point of lowest loss.

Here's the thing though: this landscape isn't 2D. It has as many dimensions as there are weights. A small model: millions of dimensions. GPT-4: hundreds of billions. Try picturing that for a second. Actually don't, you'll hurt yourself.

Nobody can visualise this landscape. But we can navigate it, one step at a time, using the **gradient**.

{% include demos/loss-landscape.html %}

The demo above is only 2D, but the algorithm is exactly what runs inside every deep-learning training loop. Run it with learning rate `0.1`. Now `1.3`. Now `0.01`. Notice how LR changes everything. And how the ball sometimes settles into the *shallow* valley instead of finding the deeper one.

## The gradient, which way is downhill?

The gradient is a vector that points in the direction of steepest *increase* in loss. It tells you: *"if you move this way, you'll get worse."*

So we go the opposite direction.

The gradient with respect to a single weight tells you: *"if I increase this weight slightly, does the loss go up or down, and by how much?"*

- Increasing the weight makes loss go up (positive gradient): **decrease** the weight.
- Increasing the weight makes loss go down (negative gradient): **increase** the weight.
- Magnitude tells you how much to change it.

Do this for every weight simultaneously. Take a step in the direction of decreasing loss. That's gradient descent.

```
new_weight = old_weight − (learning_rate × gradient)
```

## Learning rate, the size of each step

The learning rate is one of the most important numbers in training. Controls how big each step is.

- **Too large.** You overshoot. Jump past the valley, up the other side, then back again. The model oscillates, never settles.
- **Too small.** Every step is tiny. You'll get there eventually, but it'll take forever. You might also get stuck in a small local minimum.
- **Just right.** Fast enough to learn, small enough to settle into a good solution.

Finding the right LR is part science, part art. Modern training uses *adaptive* learning rates. Algorithms like **Adam** adjust the step size for each weight individually based on how the gradient has been behaving.

## Backpropagation, computing the gradient efficiently

Problem: a model has billions of weights. Computing the gradient for all of them by testing each one (*"what if I nudged this weight up a tiny bit?"*) would take forever.

**Backpropagation** (backprop) solves this with the chain rule of calculus. An algorithm that computes the gradient for every weight in a single backward pass through the network. In essentially the same time it takes to run a forward pass.

Key idea: the gradient flows backwards. Start from the loss (at the output). Compute how much each weight contributed to it, layer by layer, going backwards toward the input.

You don't need to understand backprop's math to understand MI. But know this: it's fast, it's exact, and every major training framework (PyTorch, JAX, TensorFlow) does it automatically. You define your model, you compute the loss, you call `.backward()`. Gradients appear, pre-computed, on every weight.

## Batches and epochs

You don't compute the gradient on one example at a time. You compute it on a **batch**. Typically 128, 512, or 2048 examples at once.

Why batches?

- Averaging over many examples makes the gradient estimate more accurate (less noise).
- Modern hardware (GPUs) is optimised for processing many things in parallel.
- Bigger batches = faster progress per gradient step.

One **epoch** = one full pass through the entire training dataset. Models are typically trained for many epochs. Same data multiple times.

- **Stochastic gradient descent (SGD).** Batch size = 1. Noisy but fast.
- **Mini-batch gradient descent.** Batch size = 32 to 2048. The standard.
- **Batch gradient descent.** Use the full dataset. Too slow for large datasets.

Modern training uses mini-batches with Adam. An improved version of SGD with adaptive learning rates and momentum.

## Overfitting

A model can memorise its training data. Loss goes to zero. On new data it's never seen (test data), the model fails badly.

That's **overfitting**. The model learned the specific examples, not the underlying patterns.

Detecting it: track loss on *training data* and on *held-out validation data* separately. Validation loss starts rising while training loss keeps falling, the model is overfitting. Training usually stops here.

Techniques to prevent it: **dropout** (randomly silence neurons during training), **weight decay** (penalise large weights), **data augmentation** (artificially expand training data).

<aside class="callout callout--warning">
  <div class="callout__label">Memorisation vs learning</div>
  <p>Overfitted models have memorised patterns rather than learned generalisable circuits. The circuits MI studies are ones that <em>generalise</em>. Because those are the algorithms the model actually learned, not the random noise it memorised.</p>
</aside>

## What gradient descent cannot tell us

Gradient descent optimises loss. That's all it does.

Does not guarantee the model learned the right algorithm. Does not guarantee it will generalise. Does not guarantee it's doing what you think it's doing.

The model might find a shortcut. A way to get low loss without learning the intended behaviour. This is the "specification gaming" problem in a nutshell. Gradient descent won't catch it. It has no idea what the "intended behaviour" is. It just minimises the number you gave it.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>Gradient descent produces the weights. MI is how we figure out what those weights <em>actually</em> learned. The relationship between these two procedures is one of the most interesting open questions in the field. <em>Why</em> does gradient descent so reliably produce interpretable circuits? Why does the same curve detector show up in every vision model? Is it inevitable?</p>
</aside>

## Wrap

Alright, real talk: between this post and the ones before it, you now know how a single neuron works, how a bunch of them chain into layers, how weights store knowledge, and how all of that was *produced* in the first place. That's enough vocabulary to follow almost any mechanistic-interpretability paper on modern neural networks.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://www.nature.com/articles/323533a0" target="_blank" rel="noopener"><div class="research-card__title">Learning representations by back-propagating errors</div><div class="research-card__authors">Rumelhart, Hinton, Williams · Nature, 1986 · the original backprop paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1412.6980" target="_blank" rel="noopener"><div class="research-card__title">Adam: A Method for Stochastic Optimization</div><div class="research-card__authors">Kingma, D. & Ba, J. · 2014 · the optimiser almost everyone uses</div></a></li>
  <li><a class="research-card" href="https://www.deeplearningbook.org/contents/optimization.html" target="_blank" rel="noopener"><div class="research-card__title">Deep Learning · Chapter 8: Optimization</div><div class="research-card__authors">Goodfellow, Bengio, Courville · free online textbook</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1712.09913" target="_blank" rel="noopener"><div class="research-card__title">Visualizing the Loss Landscape of Neural Nets</div><div class="research-card__authors">Li, H. et al. · 2018 · 2D slices of real loss surfaces</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2301.05217" target="_blank" rel="noopener"><div class="research-card__title">Progress measures for grokking via mechanistic interpretability</div><div class="research-card__authors">Nanda, N. et al. · 2023 · training dynamics through an MI lens</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2201.02177" target="_blank" rel="noopener"><div class="research-card__title">Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets</div><div class="research-card__authors">Power, A. et al. · 2022 · the grokking paper</div></a></li>
</ul>
