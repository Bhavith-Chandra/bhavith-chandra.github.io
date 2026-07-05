---
layout: post-article
title: "Financial Advisory LLM for India: When 'Speak English' Isn't Enough"
date: 2024-08-08
permalink: /posts/notes-on-financial-advisory-llm-for-india/
excerpt: "A generic assistant will happily answer an Indian user's tax question — in dollars, in US brackets, referencing a 401(k). The whole paper is about closing that gap without pretending to be a SEBI-registered advisor."
read_time_label: "8 min read"
accent: teal
---

Companion note to [Financial Advisory LLM Model for Modernizing Financial Services and Innovative Solutions for Financial Literacy in India](https://assets-eu.researchsquare.com/files/rs-4354348/v1_covered_341a77ab-9100-4c87-b9df-b5c2ee75aeb0.pdf).

Ask any large public model *"how much tax do I owe on ₹18 lakh in Chennai?"* and something strange happens. The reply is in the right currency, sometimes. The tax brackets are American, usually. Somewhere in there is a mention of a 401(k). Occasionally a Roth IRA turns up like a tourist who wandered into the wrong shop.

The model isn't stupid. Its training distribution is 90% American. When you ask about Indian finance, you're asking it to work in the long tail — and it does what a model on the long tail always does. It confabulates plausibly from the fat part of the distribution.

Here's what "plausibly wrong" looks like at three different salary levels.

---

## Same question, two very different answers

{% include demos/finance-regime.html %}

Move the salary slider. Toggle between old and new regime. Watch the left panel keep quoting US brackets and the right panel actually compute the correct Indian tax under whichever regime you picked. The difference isn't fluency — both models sound confident. The difference is whether they're operating in the right country.

<aside class="callout callout--key">
  <div class="callout__label">Why prompting alone doesn't fix this</div>
  <p>Sticking "you are a friendly Indian financial advisor" in a system prompt shifts the register. It doesn't shift the underlying knowledge distribution. The model still hasn't seen enough Indian tax code to reason correctly about it, so it fills in the blanks with US mechanics.</p>
</aside>

## What we actually built

A three-part stack, each part solving a specific failure mode of the generic model.

### 1. A small, dense instruction dataset

Not big. A few thousand carefully written examples covering the core Indian personal-finance surface:

- **Sections 80C, 80D** — tax-saving instruments and health-insurance deductions.
- **Old vs new regime** — the choice, the trade-offs, when each is optimal.
- **HRA, LTA, NPS** — salary components that a US-tuned model has literally never heard of at scale.
- **Capital gains** — LTCG vs STCG, indexation, ₹1L exempt threshold on equity.
- **PPF, EPF, ELSS, SIPs, ULIPs** — the actual products Indian users buy.

Every example was hand-checked for correctness by someone who files an ITR themselves. That last part is unglamorous and non-negotiable — you cannot bootstrap financial correctness from GPT-generated examples, because *the base model gets it wrong* and you'd be baking those errors into your fine-tuned model.

### 2. Retrieval over a dated, sourced corpus

The Income Tax Act changes. Every budget shifts something. If your model paraphrases a rule from its training weights, it will confidently quote the 2022 rule to a 2025 user.

The retrieval corpus is small and dated — sections of the ITA, SEBI circulars, RBI notifications, all timestamped. The model quotes law from retrieved passages instead of paraphrasing from weights. This is boring engineering that matters way more than it should.

<aside class="callout callout--warning">
  <div class="callout__label">Timestamps are load-bearing</div>
  <p>A financial LLM that can't tell you <em>when</em> a rule was in effect is a financial LLM that will confidently mislead someone in an audit. Every retrieved passage has a valid-from and valid-until date, and the model surfaces them.</p>
</aside>

### 3. Refusal on advice that would need a SEBI-registered person

This is the one I care about most. The model is not licensed. It never will be. If a user asks *"should I put my ₹5L bonus into ELSS or debt funds?"*, the model:

1. Explains what each vehicle is.
2. Shows the tax treatment of each.
3. Refuses to say which one they should pick.

That refusal is trained. It's easy to make an LLM helpful. It's much harder to make it helpfully draw a line and stop.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Think of the model as a knowledgeable friend who happens to have read every finance book. They'll happily explain what an ELSS is. They will not tell you to buy one. That's not their job, and they know it.</p>
</aside>

## The evaluation is where I sweated

Standard NLP benchmarks are useless here. A model that gets 90% on Indian-Finance-QA might be catastrophically wrong on the 10% where it hallucinates a tax rule.

We ran three eval axes:

- **Numerical correctness.** Given a scenario (salary, deductions, regime), does the model compute the right tax to within a rupee? This one is mechanical and easy to grade.
- **Legal grounding.** Does every rule the model quotes correspond to a real, current section? Human graders with tax knowledge. Slow, painful, necessary.
- **Refusal calibration.** On advice questions, does the model actually refuse? A model that refuses too much is useless; a model that refuses too little is dangerous.

The number I'd shout from the rooftop: numerical accuracy went from ~54% (base model, prompted) to ~91% (fine-tuned + retrieval). Legal grounding is harder to summarize but roughly doubled.

## The thing I'm least confident about

Financial literacy is downstream of trust. A model that gives correct answers to 91% of questions is a great product. It is not, by itself, a movement in financial literacy in India.

The users I most want to reach — first-time earners, gig workers, people whose parents didn't have brokerage accounts — need an interface that meets them where they are. That's product design, vernacular localization, and a lot of user research. The paper is about the model. The literacy question is about everything around the model, and we barely touched it.

<aside class="callout callout--warning">
  <div class="callout__label">Honest limit</div>
  <p>A correct model in an intimidating interface, in English only, on a website nobody navigates to, changes nothing. The next paper (or product) is about the surrounding stack.</p>
</aside>

## What I'd do next

- **Vernacular support.** Hindi, Tamil, Marathi, at minimum. Not translation of the English output — retrieval and generation directly in the language.
- **Voice interface.** A large fraction of the target user base uses WhatsApp voice notes for everything else. The finance model should meet them there.
- **Longitudinal case-based eval.** Instead of one-shot Q&A, evaluate the model over a multi-turn conversation that resembles a real tax-planning session. That's where hedging and consistency really get tested.

Full paper: [Research Square](https://assets-eu.researchsquare.com/files/rs-4354348/v1_covered_341a77ab-9100-4c87-b9df-b5c2ee75aeb0.pdf).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/2303.17564" target="_blank" rel="noopener"><div class="research-card__title">BloombergGPT: A Large Language Model for Finance</div><div class="research-card__authors">Wu et al. · 2023 · the domain-tuned-LLM template we borrowed from</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noopener"><div class="research-card__title">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</div><div class="research-card__authors">Lewis et al. · NeurIPS 2020 · the RAG paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2109.07958" target="_blank" rel="noopener"><div class="research-card__title">TruthfulQA: Measuring How Models Mimic Human Falsehoods</div><div class="research-card__authors">Lin, Hilton, Evans · 2021 · framework for the "confidently wrong" evaluation</div></a></li>
  <li><a class="research-card" href="https://www.incometax.gov.in/iec/foportal/" target="_blank" rel="noopener"><div class="research-card__title">Income Tax Department, Government of India</div><div class="research-card__authors">Official source · what the retrieval corpus indexes</div></a></li>
  <li><a class="research-card" href="https://assets-eu.researchsquare.com/files/rs-4354348/v1_covered_341a77ab-9100-4c87-b9df-b5c2ee75aeb0.pdf" target="_blank" rel="noopener"><div class="research-card__title">Financial Advisory LLM Model for Modernizing Financial Services</div><div class="research-card__authors">Challagundla et al. · 2024 · this paper</div></a></li>
</ul>
