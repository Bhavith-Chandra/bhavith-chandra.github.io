---
layout: post-article
title: "APEX: Load Balance Was Never the Right Loss"
date: 2025-03-11
permalink: /posts/notes-on-apex-adaptive-per-token-expert-loss/
excerpt: "Every MoE model has a load-balancing loss. Every one is doing the same wrong thing at the token level. This post is the ten-minute version of why, and what the fix looks like when you push the loss down to per-token."
read_time_label: "9 min read"
accent: teal
---

Companion note to [APEX: Adaptive Per-Token Expert Loss](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5877982).

Here is the thing about mixture-of-experts models. They shouldn't work.

You have 8 or 64 or 128 "experts" — sub-networks — and a small router that picks two of them per token. The router is untrained at initialization. Its picks are essentially random. The gradients flowing back into the router are noisy and sparse.

Left to itself, the router collapses. It finds one or two favorites and picks them for *every* token, and the other experts die, having never gotten enough training signal to become useful. So every MoE paper adds an **auxiliary load-balancing loss** — a penalty that says "please use all your experts roughly equally."

The standard load-balance loss works, in a sense. Everyone stays alive. Nothing specializes. This is the failure mode APEX exists to fix.

---

## The setup, in one paragraph

The router assigns each token a distribution over experts. The load-balance loss penalizes the deviation of *global* expert usage — averaged over the whole batch — from uniform. So if expert 3 is picked 40% of the time and expert 7 is picked 5%, the loss says "route more to expert 7."

That objective is *batch-level*. It doesn't care which token went where. It only cares about the aggregate.

<aside class="callout callout--key">
  <div class="callout__label">The subtle wrongness</div>
  <p>The standard load-balance loss is happy with "the word <em>the</em> always goes to expert 3" as long as some other token compensates by getting routed to expert 5. Which is exactly wrong — <em>the</em> should be routed by content, not by expert habit. And a rare, domain-specific token that would benefit from concentrating on one expert is fighting the loss the entire time.</p>
</aside>

The right per-token intuition is the *opposite* shape:

- If the router is **confident** about a token (sharp distribution over 1-2 experts), leave it alone. That's specialization forming; that's what we want.
- If the router is **uncertain** about a token (flat distribution), lean harder on load-balance. Uncertainty means specialization hasn't formed yet, so nudge harder.

APEX is that intuition, expressed as a loss.

## The demo

Run 400 tokens through an 8-expert router. Toggle APEX on and off. Watch what happens.

{% include demos/moe-router.html %}

With **APEX on**, three things happen. Fewer dead experts. Each surviving expert leans toward one topic (the colored share on the load bar shows which). Mean specialization goes up.

With **APEX off** — the standard load-balance loss — every expert sees every topic in roughly equal share. Load is balanced. Nothing else is. The whole point of MoE is that experts specialize; the standard loss actively prevents that.

## What the loss actually looks like

The load-balancing loss is a weighted sum where the *weights adapt per-token, per-step, based on router confidence*.

For each token, the router outputs a probability distribution over experts. Take the top-2 gap — the difference between the top expert's probability and the second's. Call that `c`, for confidence.

- The auxiliary loss's contribution from this token is scaled by `1 / (1 + c)`. High-confidence tokens contribute weakly. Low-confidence tokens contribute strongly.
- The main loss (task loss) is unaffected. APEX only touches the auxiliary term.

That's the whole change. Twelve lines of code. Comes for free at training time — a rounding-error compute cost. And it removes an assumption that was hiding in every MoE paper for years.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Standard load-balance is a manager who forces every employee to work on every project so nobody feels left out. APEX is a manager who lets the specialists specialize and only intervenes when someone genuinely has no idea what they're doing.</p>
</aside>

## What it actually buys you

Not a jaw-dropping perplexity number. A few points at matched compute, real but not staggering. If you were chasing a headline benchmark result, APEX by itself won't get you a headline.

The gains that matter to *me* are structural:

- **Fewer dead experts** as you scale expert count. This is the number that lets you actually use large MoE models — a lot of production MoE deployments quietly waste half their experts.
- **Cleaner emergent specialization**, visible earlier in training. If you probe individual expert activations, you can name what they're for. That's not something you can typically do with standard load-balance.
- **Better sample efficiency in the router.** Because the loss lets router confidence shape training, the router gets to trust its own decisions and refine them, instead of being constantly overridden.

<aside class="callout callout--warning">
  <div class="callout__label">Not a free lunch</div>
  <p>APEX <em>can</em> collapse to a single expert if you turn the load-balance term all the way off. There's still a floor on how weak the auxiliary loss gets. The paper's hyperparameters keep the floor safe — but if you push them, be careful.</p>
</aside>

## The point I want you to leave with

The whole reason I care about this paper isn't the number. It's the principle underneath.

**Auxiliary losses that try to shape a distribution should be conditioned on the distribution.** Static, batch-averaged auxiliary losses throw away information they didn't have to.

Once you notice that principle, you see it everywhere:

- KL regularization in VAEs uses a fixed β. Should be per-token.
- Contrastive losses use a fixed temperature. Should be adaptive.
- Dropout uses a fixed rate. Should be per-activation.

Not every one of those is worth chasing. But the shape of the argument is the same. If a loss is trying to reshape a distribution, don't wave it around uniformly. Wave it harder where the distribution's wrong, softer where it isn't.

## What I'd do next

- **APEX + expert pruning.** If the router is confidently ignoring an expert, kill it. Reallocate the parameters. This is a real trick nobody's fully pulled off yet.
- **Curriculum on the confidence weighting.** Early in training, use a flatter weighting (help the router bootstrap). Later, use a sharper weighting (let specialization form). We used a fixed schedule; a learned one would probably do better.
- **Actually deploy this in a large MoE at frontier scale.** The paper is on smaller models. Someone with more compute than me should try it on a real 100B+ MoE. Please.

Full paper: [APEX on SSRN](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5877982).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1701.06538" target="_blank" rel="noopener"><div class="research-card__title">Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer</div><div class="research-card__authors">Shazeer et al. · ICLR 2017 · the MoE paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2101.03961" target="_blank" rel="noopener"><div class="research-card__title">Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity</div><div class="research-card__authors">Fedus, Zoph, Shazeer · JMLR 2022 · Switch Transformer, standard load-balance loss</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2202.08906" target="_blank" rel="noopener"><div class="research-card__title">Mixture-of-Experts with Expert Choice Routing</div><div class="research-card__authors">Zhou et al. · NeurIPS 2022 · alternate routing scheme</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2405.02412" target="_blank" rel="noopener"><div class="research-card__title">A Closer Look into Mixture-of-Experts in Large Language Models</div><div class="research-card__authors">Lo et al. · 2024 · analysis of dead experts and specialization</div></a></li>
  <li><a class="research-card" href="https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5877982" target="_blank" rel="noopener"><div class="research-card__title">APEX: Adaptive Per-Token Expert Loss</div><div class="research-card__authors">Challagundla, Challagundla · 2025 · this paper</div></a></li>
</ul>
