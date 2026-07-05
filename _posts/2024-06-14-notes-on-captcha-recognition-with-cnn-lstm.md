---
layout: post-article
title: "CAPTCHA With CNN + LSTM: A Small Model, Meanly Trained"
date: 2024-06-14
permalink: /posts/notes-on-captcha-recognition-with-cnn-lstm/
excerpt: "Big models overfit. Small models generalize. The whole paper is basically that one sentence, applied to distorted-text CAPTCHAs, backed up by one aggressive augmentation and a loss function nobody uses anymore."
read_time_label: "7 min read"
accent: amber
---

Companion note to [Efficient CAPTCHA Image Recognition Using CNNs and LSTMs](https://www.ijarise.org/index.php/ijarise/article/download/82/78) (IJARISE, 2024).

Every modern OCR paper wants to be a transformer. And for street signs and receipts, sure, be a transformer. For distorted-text CAPTCHAs, transformers are hilariously overpowered. They overfit to the specific distortions of your training set and fall apart on anything new.

The paper is a small model, meanly trained. This post is why.

---

## The regime nobody talks about

CAPTCHAs occupy a very specific corner of OCR:

- Short glyph sequences — usually 4 to 8 characters.
- Fixed character set — usually 30-ish alphanumeric symbols.
- No linguistic prior — the strings are random, so a language model on top is useless.
- The whole point of the image is to be *geometrically weird*. That's the entire task.

The task boils down to: read stroke topology through distortion. That's a job CNNs are already excellent at, and one where sequence order is basically left-to-right. So the pipeline writes itself: **CNN feature extractor → BiLSTM → CTC loss**. No pretraining, no transformer, no beam search on top.

## The augmentation that did all the work

Move the slider. This is the single most important variable in the whole paper.

{% include demos/captcha-distortion.html %}

Turn the augmentation checkbox off, push amplitude past ~0.35, and the model falls off a cliff. Turn it on, and the model degrades gracefully out into pretty ugly distortion. Same architecture. Same training compute. Just, one of them was fed images it hadn't seen at training time and one wasn't.

<aside class="callout callout--key">
  <div class="callout__label">The one-line thesis</div>
  <p>The augmentation distribution IS the model's OOD prior. If your training distortion covers what the test set does, you generalize. If it doesn't, no amount of architecture cleverness will save you.</p>
</aside>

## Why CTC, not per-character cross-entropy

The naive move is: input image → 6 outputs, one per character position, cross-entropy on each. It works when characters are cleanly aligned to horizontal image regions.

Under elastic distortion, they aren't. Character positions drift, sometimes overlap, sometimes leave gaps. Per-position cross-entropy penalizes the model for correctly reading a character that ended up in the "wrong" column.

**CTC** — Connectionist Temporal Classification — lets the model produce any number of intermediate blank tokens between characters, and defines the loss over all input-output alignments that collapse to the target string. Alignment is handled by the loss function itself, for free. This is why every serious sequence-OCR paper for the last ten years uses CTC.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Per-position cross-entropy is a metronome that demands each note land exactly on the beat. CTC is a jazz teacher that says "just play the melody, we'll worry about the timing." One of them writes better music.</p>
</aside>

## Why the backbone stayed small

Every paper reviewer asked the same question: *why not use a bigger backbone?*

I tried. Bigger backbones got *worse* on test accuracy, even as they overfit training accuracy toward 100%. Bigger backbones burn their extra capacity memorizing the specific distortion patterns in the training set. That's a great way to hit 99.7% train / 71% test. That's a terrible way to ship anything.

<aside class="callout callout--warning">
  <div class="callout__label">Small models are often correct</div>
  <p>"Bigger is always better" is a slogan, not a law. Model capacity beyond what the task actually needs mostly buys you overfitting. On narrow perceptual tasks with strong augmentation, small models win.</p>
</aside>

The specific numbers: a 4-block CNN with about 1.2M parameters beat an 8-block CNN with about 8M parameters on the harder test splits by 3-5 accuracy points. Same augmentation. Same data. Bigger just hurt.

## The failure mode I didn't fix

Adversarial CAPTCHAs — the ones with color-noise overlays, overlapping glyphs, or 3D-rendered characters — still fool the model.

That's because my augmentation didn't cover them. And that's the point of the paper's thesis: *my model's OOD prior only extends as far as my augmentation distribution does*. Beyond it, the model has no reason to be right.

The next paper would be about learning the augmentation distribution instead of specifying it by hand. That's a hard, unsolved problem, and I didn't want to solve it in a five-page paper.

## What I'd do differently

- **Report per-character-count accuracy.** A model that's great on 5-char CAPTCHAs and useless on 8-char ones has a specific bug, and averaging hides it.
- **A tiny attention block after the LSTM** for very long sequences. LSTMs at 10+ steps of distorted characters start to drift. A single head of self-attention on the character features would probably fix it and add almost no parameters.
- **Test on wild CAPTCHAs.** I trained and evaluated on synthetic. Real-world CAPTCHAs from scraped forms would tell me a much more honest number.

Full paper: [IJARISE PDF](https://www.ijarise.org/index.php/ijarise/article/download/82/78).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://www.cs.toronto.edu/~graves/icml_2006.pdf" target="_blank" rel="noopener"><div class="research-card__title">Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with RNNs</div><div class="research-card__authors">Graves et al. · ICML 2006 · the CTC paper</div></a></li>
  <li><a class="research-card" href="https://ieeexplore.ieee.org/document/1227801" target="_blank" rel="noopener"><div class="research-card__title">Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis</div><div class="research-card__authors">Simard, Steinkraus, Platt · 2003 · where elastic distortion for OCR augmentation was introduced</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1507.05717" target="_blank" rel="noopener"><div class="research-card__title">An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)</div><div class="research-card__authors">Shi, Bai, Yao · 2015 · the CNN+BiLSTM+CTC canonical paper</div></a></li>
  <li><a class="research-card" href="https://www.ijarise.org/index.php/ijarise/article/download/82/78" target="_blank" rel="noopener"><div class="research-card__title">Efficient CAPTCHA Image Recognition Using CNNs and LSTMs</div><div class="research-card__authors">Challagundla et al. · IJARISE, 2024 · this paper</div></a></li>
</ul>
