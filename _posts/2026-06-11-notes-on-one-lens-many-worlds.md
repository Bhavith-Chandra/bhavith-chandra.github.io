---
layout: post-article
title: "One Lens, Many Worlds"
date: 2026-06-11
permalink: /posts/notes-on-one-lens-many-worlds/
excerpt: "World-model research is fragmenting across at least three architectural families, and interpretability tooling doesn't compose across them. We proposed a small type system that fixes the composition problem without asking the field to agree on a single representation."
read_time_label: "12 min read"
accent: teal
math: true
---

A world model, in the current usage of the term, is any neural network trained to predict future states of an environment from present ones. Three architecturally distinct families are actively producing state-of-the-art results at the time of writing: recurrent state-space models in the Dreamer lineage, tokenized transformers in the Genie and GAIA lineage, and joint-embedding predictive architectures in the JEPA lineage. Each family has developed its own conventions for representing state, its own choices about what to decode back to observation space, and — most consequentially for our purposes — its own private interpretability tooling.

This is not a problem of taste. It is a problem of composition. An analysis written against a Dreamer-style recurrent state-space model does not run, without substantial rewriting, on a Genie-style token transformer. A probe designed for the JEPA latent has no natural extension to a model whose latent is a discrete token distribution. Every lab that studies world models is either confined to one family or paying the cost of maintaining three parallel implementations of the same idea.

Our paper argues that this cost is unnecessary, and that the standard route around it — a single unified framework — is worse than the disease. What we propose instead is a small, sharp *type system* over a set of adapter capabilities. Analyses declare the capabilities they need. Architectures declare the capabilities they expose. Compatibility is a static check. Composition, not unification.

## Why unification fails

We tried unification first. The design is intuitive: identify the common structure across world-model families, expose it through a single interface, and write analyses against that interface. The problem is that the common structure is thin.

Recurrent state-space models carry two distinct latent objects — a deterministic hidden state and a stochastic latent — that co-evolve through a learned transition. Token transformers carry no such distinction; state is implicit in a growing key-value cache, and the model's "prediction" is a distribution over next tokens in a discrete space. Joint-embedding predictive architectures carry a continuous latent and, critically, *do not decode back to observation space at all*. Any unified state representation either erases these differences (making some analyses impossible) or expands to accommodate them (making the abstraction weak enough that it does not constrain implementations).

Every prior attempt at unification in world-model research has landed somewhere in this trade-off. The tooling ends up architecture-specific in practice even when the interface claims otherwise.

<span class="sidenote"><sup>1</sup><span>A partial exception is the RL-community convention of treating world models as *environments with a step function*, which composes across families for some tasks. It is not fine-grained enough for interpretability work: you cannot patch a latent through the step function without knowing what a latent means for the underlying model.</span></span> The failure mode is characteristic: the framework is presented, adoption starts, and within eighteen months every user has forked to add architecture-specific extensions that were not anticipated by the original design.

## The move that worked

We stopped trying to describe world models in terms of what they *are* and started describing them in terms of what they *can do*. Each adapter implements a small set of capabilities. The core four are:

- `encode(observation) → state`
- `predict(state, action?) → next_state`
- `rollout(state, k) → trajectory`
- `extract(layer_name) → activations`

These are present in every serious world-model architecture we surveyed. Optional heads absorb the differences: `decode(state) → observation` for architectures that reconstruct pixels; `value_head(state) → scalar` for architectures with an explicit value estimator; `rollout_for_agent(agent_id)` for multi-agent settings.

Analyses declare which capabilities they need. Causal tracing requires `predict` and `extract`. Reconstruction attribution requires `decode`. Theory-of-mind probes require `rollout_for_agent`. When a user attempts to run an analysis on an architecture whose adapter does not expose the required capabilities, the harness reports the incompatibility *at load time*, before any compute is spent.

{% include demos/capability-adapter.html %}

The figure above walks through the resolution mechanically. RSSMs expose six of seven capabilities (they lack agent-conditional rollout, which most implementations do not have); token transformers expose a different six (they include agent-conditional rollout via agent tokens but often lack a value head); JEPAs expose only the required four (no decoder, no value head, no agent conditioning). Every analysis's compatibility follows from set containment on the exposed capability set.

## What this buys, and what it does not

The immediate benefit is that library analyses are written once. Causal tracing implemented against the capability interface runs on every architecture that exposes `predict` and `extract`, which is all three families under discussion. Mutual-information probes, trajectory-geometry tooling, and dynamical-systems analyses all inherit the same portability. Reconstruction attribution, being decoder-dependent, portably runs on RSSMs and token transformers and portably *does not* run on JEPAs — reported as such, rather than silently producing meaningless output.

<span class="sidenote"><sup>2</sup><span>The silent-nonsense failure mode is the one that keeps me up. A framework that claims universal applicability, produces plots, and lets you draw conclusions from an analysis that was never actually valid for the architecture you ran it on is worse than a framework that refuses to run.</span></span> The static compatibility check is the mechanism that prevents this class of error.

The framework does not resolve semantic mismatch between architectures. Two adapters can both expose `predict`, and the state returned can mean subtly different things — the mean of a distribution in one implementation, a sample from that distribution in another. Capability-typing catches structural mismatch; it does not catch this. We flag it in the paper as an open problem.

Nor does it eliminate the cost of writing adapters. A new architecture requires an adapter implementation; we have made this as small as we can, but the cost is not zero. What the framework does is amortize that one-time adapter cost across all subsequent analyses, which is a substantially better position than the current status quo of writing one implementation per (architecture, analysis) pair.

## Why now

Two things converged that made this the correct time to propose a shared substrate.

The first is that the world-model field is fragmenting on a shorter timescale than the interpretability field can keep up with. A new architectural family has emerged in each of the last three years. If interpretability tooling has to be rewritten from scratch for each, the tools will lag the models by definition, and the lag will grow.

The second is that the safety case for deploying agents that plan against learned world models — which is happening, at scale — depends on being able to inspect those world models. Interpretability that only works for one architecture family is not safety infrastructure. It is a research artifact of that family. For interpretability to serve a safety role, it has to compose across the architectures a deployed system might actually use. The type system is our attempt to make that composition possible.

## What ships

[WorldModelLens](https://github.com/Bhavith-Chandra/WorldModelLens), released with the paper, contains adapter implementations for the three families discussed above and reference implementations of the five analyses shown in the figure. The intent is that a research group with a new world-model architecture can write an adapter — roughly fifty lines of code, most of which is method plumbing — and get every analysis in the library for free.

The current work I am doing along these lines extends the framework in two directions: a set of capability-typed analyses over the geometry of learned latent spaces, and a set of energy-based priors that compose with the `rollout` capability to shape trajectory sampling. Both are drafts. Both are the reason the framework needed to exist.

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/2606.09936" target="_blank" rel="noopener"><div class="research-card__title">One Lens, Many Worlds: A Capability-Typed Interface for World-Model Interpretability</div><div class="research-card__authors">Challagundla, Pandey, Thakkar, Mallagundla, Gogireddy, Lu, Roy Choudhury, Challagundla, Deraz Nasr, Deshpande · 2026</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2301.04104" target="_blank" rel="noopener"><div class="research-card__title">Mastering Diverse Domains through World Models (DreamerV3)</div><div class="research-card__authors">Hafner, Pasukonis, Ba, Lillicrap · 2023</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2402.15391" target="_blank" rel="noopener"><div class="research-card__title">Genie: Generative Interactive Environments</div><div class="research-card__authors">Bruce et al. · 2024</div></a></li>
  <li><a class="research-card" href="https://ai.meta.com/vjepa/" target="_blank" rel="noopener"><div class="research-card__title">V-JEPA: Video Joint-Embedding Predictive Architecture</div><div class="research-card__authors">Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas · Meta AI, 2024</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2202.05262" target="_blank" rel="noopener"><div class="research-card__title">Locating and Editing Factual Associations in GPT (ROME)</div><div class="research-card__authors">Meng, Bau, Andonian, Belinkov · NeurIPS 2022</div></a></li>
  <li><a class="research-card" href="https://distill.pub/2020/circuits/zoom-in/" target="_blank" rel="noopener"><div class="research-card__title">Zoom In: An Introduction to Circuits</div><div class="research-card__authors">Olah, Cammarata, Schubert, Goh, Petrov, Carter · Distill, 2020</div></a></li>
</ul>
