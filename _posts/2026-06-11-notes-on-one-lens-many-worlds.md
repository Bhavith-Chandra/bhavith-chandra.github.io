---
layout: post-article
title: "One Lens, Many Worlds: A Type System for World-Model Interpretability"
date: 2026-06-11
permalink: /posts/notes-on-one-lens-many-worlds/
excerpt: "World-model research is fragmented. Every architecture family has its own tools, its own probes, its own vocabulary. This is the paper where we tried to fix that with a small, sharp adapter interface instead of a giant unified framework — and honestly, this is the direction I'm betting the next few years on."
read_time_label: "12 min read"
accent: teal
math: true
---

Companion note to [One Lens, Many Worlds: A Capability-Typed Interface for World-Model Interpretability](https://arxiv.org/abs/2606.09936). This is the paper I'm most excited about, and I want to explain *why* rather than just what.

---

## The problem, in one paragraph

World models come in wildly different shapes.

- **Recurrent state-space models** (Dreamer, DreamerV3) carry a deterministic hidden state and a stochastic latent, evolved by a learned transition.
- **Token transformers** (Genie, GAIA) treat world dynamics as autoregressive next-token prediction over tokenized observations.
- **Joint-embedding predictive architectures** (JEPA, V-JEPA) predict future *representations*, never decoding back to pixels.

Every interpretability technique — probes, causal traces, activation patching, information-flow analysis — was written against *one* of these families. None of them cleanly transfer to the others.

That means every interpretability team is either confined to studying one architecture family or spending most of their compute re-implementing the same analysis three ways. Which is exactly the state world-model interpretability was in when we started.

## The move that worked

We tried the "unified framework" approach first. Big abstraction over all world models, single common representation, generic analyses on top. It fought us the whole way.

RSSMs care about a distinction between deterministic and stochastic state that transformers don't have. Token models have a KV cache that RSSMs don't. JEPAs *have no decoder at all*, so any analysis that assumes reconstruction dies at the door.

<aside class="callout callout--key">
  <div class="callout__label">Give up on unification. Type the capabilities.</div>
  <p>Instead of one common representation, expose a small set of <strong>capabilities</strong> — encode, predict, rollout, extract. Every analysis declares what capabilities it needs. Every architecture's adapter declares what it exposes. A static check tells you whether an analysis can run on a model, before any compute is spent.</p>
</aside>

That's it. Pick an architecture. Pick an analysis. The type system resolves the match.

{% include demos/capability-adapter.html %}

Try picking JEPA and then "Reconstruction attribution." The harness reports that JEPA can't decode back to observations, so the analysis is incompatible. It doesn't crash. It doesn't silently produce nonsense. It says *"this analysis needs a decoder head, this model doesn't have one, I'm skipping."*

That is the entire safety property that makes this work as a shared library across labs.

## The four required capabilities

Every world-model adapter has to implement four methods. That's the whole contract.

```
encode(observation) → state
predict(state, action?) → next_state
rollout(state, k) → [state_1, state_2, ..., state_k]
extract(layer_name) → activations
```

Optional heads handle things like `decode(state) → observation`, `value_head(state) → scalar`, or `agent_conditional(state, agent_id) → state` for multi-agent world models. Analyses that need optional capabilities declare so, and get gracefully skipped when the model doesn't have them.

Every serious world-model architecture I've looked at implements the four core methods trivially. The optional heads are where architectures differ, and that's exactly where the type system does its work.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Think of it like WebGL. You don't write one giant abstraction over every possible GPU. You declare which features you need (compute shaders, floating-point textures, whatever) and the runtime tells you whether the GPU supports them. Analyses are shaders. Adapters are GPUs. The type system is the feature-detection layer.</p>
</aside>

## What lives in the library

Once the adapter interface is in place, you write each analysis *once* and it runs on every compatible architecture. That's the leverage.

- **Causal tracing.** Patch a state at layer L, re-run the rollout, measure the divergence in downstream predictions. Works anywhere `predict` and `extract` are available — which is everywhere.
- **Information-theoretic layer analysis.** Mutual information between successive states, entropy of the latent, disentanglement metrics. Only needs `encode` and `predict`.
- **Trajectory geometry.** Clustering, dimensionality analysis, and dynamical-systems tools (Lyapunov exponents on the state trajectory) applied to `rollout` outputs.
- **Multi-agent theory-of-mind probes.** The harder end — only runs on architectures with `agent_conditional`. Most models don't have it, and the harness cleanly skips.
- **Reconstruction attribution** (SHAP, LIME, integrated gradients over the observation space). Needs `decode`. Runs on RSSMs and token transformers. Cleanly skipped on JEPA.

Every one of those analyses used to be written per-architecture. Now they're written once. The compound effect on research velocity is bigger than any single result the library ships.

## Why "capability-typed" and not "unified"

There are two temptations when you're building a shared library, and both are wrong.

**Temptation 1: unify everything under one abstract representation.** Pick a "state" type that supposedly captures every architecture's state. This fails because the abstraction is either so weak it can't express what any specific analysis needs, or so strong it excludes half the architectures. There is no free lunch here.

**Temptation 2: write one analysis per architecture and let library users pick.** This is what everyone was doing before. It fragments the field. It also means every analysis has three subtly different implementations that drift out of sync.

**Capability-typed adapters split the difference.** The core methods are minimal enough that every reasonable architecture supports them. The optional heads absorb the differences. And the type system makes the differences *observable* to the analysis writer before compute is spent.

<aside class="callout callout--warning">
  <div class="callout__label">The move that took me longest to accept</div>
  <p>Giving up on the "single common state representation" was hard. It felt like a failure of abstraction. It's actually the win — trying to force a common representation is what made every previous framework brittle. Static checks over exposed capabilities let each architecture keep the state form its analyses need, without the framework flattening it.</p>
</aside>

## Why now

Two things converged that made this the right time.

**The field is fragmenting fast.** Every large lab has its own architecture family and its own private tooling. Interpretability results across labs are getting *harder* to compare, not easier. Without a shared substrate, that trajectory only gets worse.

**The safety case for interpretable world models is getting sharper.** We're going to deploy agents that plan against learned world models. We already are. Being able to inspect those world models has to work *across architectures*, because we don't get to pick which architecture the deployed system uses. Interpretability that's architecture-specific isn't safety infrastructure. It's a research artifact.

<aside class="callout callout--key">
  <div class="callout__label">Why I care about this personally</div>
  <p>Every safety story about learned world models assumes we can look inside them. Right now we mostly can't, because "look inside" doesn't compose across architectures. This paper is my attempt to make it compose. If it doesn't compose, agentic-AI safety cases don't work at all.</p>
</aside>

## What's shipping alongside the paper

[**WorldModelLens**](https://github.com/Bhavith-Chandra/WorldModelLens) — open source. Adapter implementations for the three architecture families discussed above, plus every analysis I described. It's early. There will be bugs. Pull requests welcome.

The plan is that if you have a world model of any architecture, you can install this library, wrap your model in ~50 lines of adapter code, and run every analysis it supports. The adapter should feel like implementing four methods, not a rewrite.

## What I'm working on next

Two directions, both live papers I'm in the middle of.

- **Latent-space geometry** across world-model families. Same architecture-agnostic framework, sharpened analyses on the shape of the state manifold. Is the JEPA state manifold structurally different from an RSSM latent? If yes, what does that predict about planning behavior?
- **Energy-based priors** for world-model rollouts. Learned scalar energy over trajectories, used as a soft filter on rollout samples. Composes with the capability interface — energy heads become an optional adapter method.

The through-line for both is the same. Once you can inspect world models across architectures with a shared toolkit, you can start asking questions about their state that were impossible when everyone had a private framework.

## The honest limitations

- **Capability granularity is a judgment call.** Right now the harness has ~7 named capabilities. Adding a new one requires updating adapters. This is not free, and it's the closest thing to a "framework tax" the library imposes.
- **Some analyses want state semantics, not just state.** Two architectures can both expose `predict` but mean subtly different things by "state" — mean of a distribution vs. sample from it, for example. The type system catches capability mismatches, not semantic mismatches. The paper flags this as an open problem.
- **We're not the first to notice the fragmentation.** Prior attempts to unify — under a shared framework or a shared benchmark — didn't work for the reasons I described above. Ours might not work either. What we've argued in the paper is that the capability-typed shape has better composition properties than any unified alternative. Time will tell.

Full paper: [arXiv:2606.09936](https://arxiv.org/abs/2606.09936). Code: [WorldModelLens on GitHub](https://github.com/Bhavith-Chandra/WorldModelLens).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1912.01603" target="_blank" rel="noopener"><div class="research-card__title">Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)</div><div class="research-card__authors">Hafner et al. · ICLR 2020 · the RSSM paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2301.04104" target="_blank" rel="noopener"><div class="research-card__title">DreamerV3: Mastering Diverse Domains through World Models</div><div class="research-card__authors">Hafner et al. · 2023</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2402.15391" target="_blank" rel="noopener"><div class="research-card__title">Genie: Generative Interactive Environments</div><div class="research-card__authors">Bruce et al. · 2024 · token-transformer world model</div></a></li>
  <li><a class="research-card" href="https://ai.meta.com/vjepa/" target="_blank" rel="noopener"><div class="research-card__title">V-JEPA: Video Joint Embedding Predictive Architecture</div><div class="research-card__authors">Meta AI · 2024 · JEPA for video</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2202.05262" target="_blank" rel="noopener"><div class="research-card__title">Locating and Editing Factual Associations in GPT (ROME)</div><div class="research-card__authors">Meng et al. · NeurIPS 2022 · causal tracing on transformers</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2606.09936" target="_blank" rel="noopener"><div class="research-card__title">One Lens, Many Worlds: A Capability-Typed Interface for World-Model Interpretability</div><div class="research-card__authors">Challagundla et al. · 2026 · this paper</div></a></li>
</ul>
