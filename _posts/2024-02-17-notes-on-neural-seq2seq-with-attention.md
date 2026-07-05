---
layout: post-article
title: "Neural Seq2Seq With Attention: The Bottleneck, and How to Kill It"
date: 2024-02-17
permalink: /posts/notes-on-neural-seq2seq-with-attention/
excerpt: "The old encoder-decoder tried to cram an entire paragraph into a single vector. It went about as well as you'd expect. Attention is the small idea that fixed it, and this is the paper where we chased that fix into summarization."
read_time_label: "7 min read"
accent: teal
---

Companion note to [our IJMLC paper on neural seq2seq with attention](https://arxiv.org/pdf/2404.08685).

Here's the punchline first: the vanilla encoder-decoder was trying to summarize a paragraph by first compressing it into *one vector*. One. Vector. For a whole paragraph. Then generating a summary from that vector alone.

Reader, it did not go well.

---

## The bottleneck, in one sentence

The encoder reads the source and produces a final hidden state. The decoder generates from that state. Everything the model knows about the source has to survive the trip through that final hidden state.

For a 20-word sentence, fine. For a 400-word news article, that vector is trying to hold a plot, three named entities, a date, and a tone. Something's getting dropped.

<aside class="callout callout--key">
  <div class="callout__label">The bottleneck</div>
  <p>Vanilla seq2seq compresses the entire source into a single fixed-length vector before the decoder ever runs. Attention lets the decoder go back and re-read the source, one look per output token. That's the whole idea.</p>
</aside>

## What attention actually does

Instead of one vector, attention keeps *all* the encoder states around. At each decoding step, the decoder computes a weighted read over them — a soft lookup — and mixes that read into the next generation step.

Weights come from a compatibility score between the current decoder state and each encoder state. Softmax makes them a distribution. Weighted sum makes them a vector. That vector is fresh for every output token.

Try it. Click through the output tokens and watch which source words the model actually looks at.

{% include demos/seq2seq-attention.html %}

Flip on the bottleneck toggle at the bottom and see what a vanilla encoder-decoder is stuck with. The arcs go red and flat. Every output token gets the same, uniform, uninformative read. That's exactly the failure mode attention was invented to kill.

## What we did in the paper

Nothing wildly new architecturally — a **bidirectional GRU encoder** with **Bahdanau (additive) attention** and a **coverage loss**. But there's a specific reason each of those was picked.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Vanilla seq2seq is a note-taker who summarizes a lecture from memory afterwards. Attention seq2seq is a note-taker who gets to look at the whiteboard again for every sentence they write. Guess which one gets the equations right.</p>
</aside>

- **Bidirectional encoder.** Each token gets to see both left and right context before it becomes a key to attend to. This one detail moved ROUGE-2 by about 0.6 points on its own.
- **Additive attention over dot-product.** Sources in our corpus were short-ish (<400 tokens). Additive attention has more parameters and does slightly better at that scale. At long-context scale you'd flip this and take dot-product for speed.
- **Coverage loss.** Vanilla attention has a habit of re-attending to the same source region over and over — you get repetitive summaries like *"The company announced. The company announced. The company announced."* The coverage loss penalizes attention distributions that overlap too much with prior steps. It doesn't help ROUGE much on average. It stops the model from embarrassing itself.

## The metric-invisible regression

Here's the thing that scared me while writing the paper.

Turning off the coverage loss dropped ROUGE-1 by less than 1%. Barely a blip. If you'd only looked at the metric, you'd say "eh, ship without it."

Then you actually read the outputs. Half the summaries had a repeated sentence. Some had the *same clause* twice in a single sentence. Fluency was gone. ROUGE didn't care because the n-grams were still in the ballpark.

<aside class="callout callout--warning">
  <div class="callout__label">Lesson I keep re-learning</div>
  <p>Some regressions don't show up in your top-line metric. If the qualitative outputs feel worse, they probably are, even if the number says otherwise. Skim outputs before you ship.</p>
</aside>

## What I'd do differently, now

- **Warm start from a pretrained encoder.** We trained from scratch. That was already outdated when we published. Free performance we left on the table.
- **Report factuality alongside ROUGE.** ROUGE tells you the summary *looks* like the reference. It won't tell you it's *true*. For a summarizer, that's the exact bug we care about.
- **Report per-length results.** Attention shines most on longer sources. Averaging across all lengths hides that.

## Why the paper still matters to me

Attention feels obvious now — every model has it, transformers are literally *attention all the way down*. But sitting inside the encoder-decoder regime, watching the ROUGE curve lift the moment the model stopped trying to squeeze everything through one vector, was the first time I felt what a good architectural inductive bias actually *does*. It doesn't add capacity. It removes a stupid constraint.

Most of the wins in deep learning are shaped like that. Somebody notices a constraint the model didn't need to have. Somebody removes it. Numbers go up.

Full paper: [arXiv:2404.08685](https://arxiv.org/pdf/2404.08685).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener"><div class="research-card__title">Neural Machine Translation by Jointly Learning to Align and Translate</div><div class="research-card__authors">Bahdanau, Cho, Bengio · ICLR 2015 · original additive attention</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1409.3215" target="_blank" rel="noopener"><div class="research-card__title">Sequence to Sequence Learning with Neural Networks</div><div class="research-card__authors">Sutskever, Vinyals, Le · NeurIPS 2014 · the encoder-decoder paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1704.04368" target="_blank" rel="noopener"><div class="research-card__title">Get To The Point: Summarization with Pointer-Generator Networks</div><div class="research-card__authors">See, Liu, Manning · ACL 2017 · where coverage loss earned its keep</div></a></li>
  <li><a class="research-card" href="https://aclanthology.org/W04-1013/" target="_blank" rel="noopener"><div class="research-card__title">ROUGE: A Package for Automatic Evaluation of Summaries</div><div class="research-card__authors">Lin · 2004 · the metric everyone lives and dies by</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2404.08685" target="_blank" rel="noopener"><div class="research-card__title">Neural Sequence-to-Sequence Modeling with Attention for Abstractive Summarization</div><div class="research-card__authors">Challagundla, Peddavenkatagari · IJMLC, 2024 · this paper</div></a></li>
</ul>
