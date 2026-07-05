---
layout: post-article
title: "Multi-Lead ECG: The Diagnosis Lives in the Disagreement"
date: 2024-04-29
permalink: /posts/notes-on-multi-lead-ecg-arrhythmia/
excerpt: "A cardiologist doesn't stare at one squiggle. They read twelve of them side by side, and the arrhythmia usually gives itself away in the way the leads disagree. Most ML models throw that signal in the bin."
read_time_label: "8 min read"
accent: warn
---

Companion note to [Advanced Neural Network Architecture for Enhanced Multi-Lead ECG Arrhythmia Detection](https://arxiv.org/pdf/2404.15347), IJFMR 2024.

Here is the thing that made me want to write this paper. A cardiologist reading an ECG doesn't stare at one waveform. They read twelve at once — different electrodes, different vantage points on the same heart — and they look at *how the leads disagree*. That disagreement is where a lot of arrhythmias give themselves away.

The vast majority of published ECG classifiers are single-lead. They throw twelve leads' worth of signal away and read the shortest one.

---

## Twelve leads, twelve angles on the same heart

Every electrode watches the heart from a different direction. Lead II sees the standard rhythm-strip view. Lead V1 sits on the right side of the chest and picks up the right ventricle. V5 sits on the left. aVF looks bottom-up. Same electrical event, four (or twelve) different projections.

A normal heartbeat looks pretty similar across all of them, so single-lead does fine on healthy patients. But diseased hearts *break the symmetry*. In left bundle branch block, the QRS complex widens on the leads that view the left ventricle and stays narrow on the ones that don't. You literally cannot see that from one lead.

Click through the conditions and watch what happens to the leads — and where a fusion model actually looks.

{% include demos/multi-lead-ecg.html %}

Look at what the fusion attention does on LBBB. It ignores Lead II (the one every single-lead paper uses) and locks onto V1 and V5, because *the disagreement between them* is the diagnosis. That's the entire pitch of the paper in one screenshot.

## What the architecture actually does

Two changes. Neither is fancy on its own; together they do most of the work.

<aside class="callout callout--key">
  <div class="callout__label">Per-lead encoders, then a fusion block</div>
  <p>Each lead gets its own 1-D CNN stack. Then a shared attention layer reads across the four (or twelve) per-lead feature vectors and pools them into a diagnosis. Mirrors how a cardiologist reads: lead by lead, then integrate.</p>
</aside>

The alternative most people default to is *channel-stacking* — treat the 12 leads as 12 input channels of a single CNN. It works, but the model can't ever look at leads independently; every filter is a joint function of all leads. The disagreement signal gets mashed at the earliest possible layer, where the model has the least information to know what to preserve.

<aside class="callout callout--key">
  <div class="callout__label">Beat-level auxiliary supervision</div>
  <p>The dataset gives you segment labels ("this 10-second strip is AFib"). But a lot of arrhythmias are defined by <em>individual beat morphology</em>. We add an auxiliary head that has to localize the QRS complexes. Forces the shared encoder to notice beats, not just windowed statistics.</p>
</aside>

The auxiliary head is not scored at test time. It only exists to shape the encoder during training. Turning it off costs us about 4 F1 points on the harder classes and barely moves the aggregate number, which is the exact shape of "this actually mattered."

## What actually moved the numbers

Class-conditional recall. Not aggregate F1. Aggregate F1 on ECG datasets is dominated by the majority class (normal sinus rhythm), so a model that just says *"normal"* 90% of the time already looks 90% good.

The interesting question is *what happens on SVT, LBBB, and PVC* — the classes where single-lead classifiers tend to collapse. That's where multi-lead pays off, and where our recall improvements were.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Aggregate accuracy on ECG data is like a weather forecaster who says "sunny" every day in Los Angeles. Technically right most of the time. Utterly useless on the days you actually needed the forecast.</p>
</aside>

## The bit I still lose sleep over

I never did the clean ablation that separated the architectural gain from the auxiliary-loss gain. Both changes were in the final model, both changes were in the ablation-off baseline. I have suspicions about which one is doing more of the work, but suspicions aren't a paper.

Also — and this is bigger — everything in the paper is trained and tested on public datasets (MIT-BIH, PTB-XL). Real hospital ECG data has noise, lead misplacement, patient motion, and 12 leads that don't always agree with the public-dataset conventions. I have no idea how the model holds up on that. Nobody who publishes on public ECG data does, really.

<aside class="callout callout--warning">
  <div class="callout__label">Health-ML pattern I keep seeing</div>
  <p>A paper reports 98% on a clean public dataset. Someone tries to deploy it and gets 62% in the ICU. The gap isn't the model — it's the sixty implicit assumptions the public dataset baked in.</p>
</aside>

## What I'd do next

- **Learn a per-lead noise gate.** In practice, leads drop out — an electrode falls off, a patient moves. The fusion block should be able to say *"lead V5 is garbage this window, downweight it."* We didn't have that.
- **Domain-shift-aware training.** Mix hospital data (even a little) into training, or explicitly augment for common real-world noise.
- **Report ROC-AUC per class alongside F1.** F1 hides a lot. Per-class ROC is where you actually see how the model behaves at different operating points, which is what a cardiologist actually cares about.

Full paper: [arXiv:2404.15347](https://arxiv.org/pdf/2404.15347).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://www.nature.com/articles/s41591-018-0268-3" target="_blank" rel="noopener"><div class="research-card__title">Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network</div><div class="research-card__authors">Hannun et al. · Nature Medicine, 2019 · the classic single-lead paper</div></a></li>
  <li><a class="research-card" href="https://physionet.org/content/mitdb/1.0.0/" target="_blank" rel="noopener"><div class="research-card__title">MIT-BIH Arrhythmia Database</div><div class="research-card__authors">Moody & Mark · PhysioNet · the benchmark almost every ECG paper reports on</div></a></li>
  <li><a class="research-card" href="https://physionet.org/content/ptb-xl/1.0.3/" target="_blank" rel="noopener"><div class="research-card__title">PTB-XL: a large publicly available electrocardiography dataset</div><div class="research-card__authors">Wagner et al. · Scientific Data, 2020 · the 12-lead dataset that made multi-lead papers viable</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1810.04231" target="_blank" rel="noopener"><div class="research-card__title">ECG Arrhythmia Classification Using a 2-D Convolutional Neural Network</div><div class="research-card__authors">Jun et al. · 2018 · one of the strong single-lead baselines</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2404.15347" target="_blank" rel="noopener"><div class="research-card__title">Advanced Neural Network Architecture for Enhanced Multi-Lead ECG Arrhythmia Detection</div><div class="research-card__authors">Challagundla · IJFMR, 2024 · this paper</div></a></li>
</ul>
