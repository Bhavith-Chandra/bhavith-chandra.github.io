---
layout: post-article
title: "Large-Scale PDF Retrieval: The Boring Bits Are Where the Accuracy Lives"
date: 2024-11-05
permalink: /posts/notes-on-large-scale-pdf-retrieval-with-faiss/
excerpt: "Every RAG tutorial hands you 50 lines and a happy demo. At the scale the demo doesn't matter and FAISS starts to earn its name, the accuracy is decided by chunking, sharding, and cache eviction. None of which anyone writes about."
read_time_label: "9 min read"
accent: amber
---

Companion note to [End-to-end Neural Embedding Pipeline for Large-scale PDF Document Retrieval Using Distributed FAISS and Sentence Transformer Models](https://www.researchgate.net/profile/Yugandhar-Gogireddy/publication/384681635).

There is a specific size of corpus at which RAG stops being a Colab notebook and starts being an engineering system. Somewhere around a few million documents, or a few hundred million chunks. Below that, everything works. Above it, everything is fragile in ways that don't show up in tutorials.

This paper (and this post) is about above that.

---

## The thing nobody writes tutorials about: chunking

Every RAG tutorial glosses over chunking. *"Just split into 512-token chunks."* Done, moving on.

Turns out that decision is where most of your recall lives, and 512 is very rarely the right answer.

{% include demos/retrieval-chunking.html %}

Move the slider. Look at what happens to recall for factoid, definitional, and multi-hop queries. **They peak at different chunk sizes.** A factoid ("what year was the CEO born?") wants a tiny, tight chunk. A multi-hop query ("how does the acquisition affect the pension plan?") wants a big chunk with enough context to link concepts.

There is no universally correct chunk size. There's only *the size that matches your query mix.*

<aside class="callout callout--key">
  <div class="callout__label">The paper's chunking rule</div>
  <p>Sentence-aware primary boundaries, plus a hard token-length ceiling as a safety net, plus a small overlap to prevent boundary-split answers from vanishing. Boring, unglamorous, moves recall more than anything else in the pipeline.</p>
</aside>

## The parts of chunking that will hurt you

- **Chunks that split mid-sentence** produce embeddings the model wasn't trained to produce. Sentence Transformers were trained on complete sentences. Half-sentences embed weirdly, cosine similarity behaves weirdly, retrieval quality degrades in a way that's invisible to eyeball testing.
- **Chunks that split mid-table** vaporize the table. Column headers in one chunk, cell values in another. The embedding of "23.4, 17.9, 41.2" is nearly useless without "Revenue Q1, Revenue Q2, Revenue Q3" attached to it.
- **Chunks that split mid-code-block** are the same, worse. Function signature in chunk A, function body in chunk B. Neither embeds anywhere near the query "how does auth work in this repo."

You fix this by *structure-aware chunking* — parse the document first, then chunk within structural units (paragraph, section, table, code block). This is unglamorous engineering. It is worth 5-10 recall points on real corpora.

## Index choice is a latency-recall tradeoff — but that's the boring axis

Every FAISS tutorial talks about the recall-vs-latency tradeoff. IVF-PQ is fast and lossy. HNSW is memory-hungry and accurate. Etc.

That's not the axis you'll actually spend all your time on.

<aside class="callout callout--warning">
  <div class="callout__label">The axis that actually matters at scale</div>
  <p>Update cost. How fast can you add new documents to the index without downtime? IVF-PQ makes this painful — you either rebuild the whole index or accept degraded recall until you do. If your corpus grows daily, you feel this immediately.</p>
</aside>

The paper's answer was mixing index types by access pattern:

- **HNSW shards** for query-heavy, low-churn data. Fast, accurate, tolerable rebuild cost because the data doesn't change often.
- **IVF-Flat shards** for high-churn data. Slightly worse recall per query, but adding new vectors is O(1). Rebuild happens on a much slower cadence.

Both sit behind one query interface. The router picks the shard based on document ID. This is deeply unglamorous. It is also what makes the system actually work.

## The distributed-FAISS part

Ninety percent of "distributed FAISS" is not FAISS at all. It's:

1. **Consistent hashing** over document IDs to route to shards. Ring layout, virtual nodes, standard stuff.
2. **A top-k merger** that fetches k results from each shard and re-ranks globally to produce the final top-k.
3. **A per-shard LRU cache** for hot query patterns.
4. **An eviction policy** for that cache that doesn't leak memory under adversarial query patterns.

FAISS the library does exactly one thing well (approximate nearest-neighbor search over a vector index). Everything around it — routing, merging, caching, eviction, sharding for growth — you build.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>FAISS is a very fast engine. "Distributed FAISS" is a car — you needed the engine, sure, but you also needed a chassis, wheels, brakes, dashboard, a fuel tank that doesn't leak, and something to steer with. Nobody publishes papers about the fuel tank.</p>
</aside>

## The failure mode I only caught in production

Embedding-model version drift.

You start with `sentence-transformers/all-MiniLM-L6-v2`. You embed 200M chunks with it. Six months in, someone upgrades to `all-MiniLM-L12-v2` because it's slightly better. Now you have half your index in one embedding space and half in another. Cosine similarities across model versions are not directly comparable. Retrieval quality falls off a cliff and nobody knows why.

The fix — a version tag on every stored vector, a query-time router that ensures your query embedder matches the shard's embedder — is boring plumbing. And absent from every RAG tutorial you'll read.

<aside class="callout callout--warning">
  <div class="callout__label">Continuous eval matters more than you think</div>
  <p>Log per-query recall against a small held-out ground-truth set continuously in production. Not once at launch. Not quarterly. Continuously. Drift creeps in from a dozen directions — embedder version, chunker changes, index staleness — long before it shows up in user complaints.</p>
</aside>

## What I'd do differently

- **Learned chunk-size selection.** Predict, per document type, what chunk-size profile fits. Small classifier on document-level features. We hand-tuned per corpus. That doesn't scale to lots of corpora.
- **Hybrid dense + sparse retrieval.** BM25 as a first-pass filter is embarrassingly effective on long-tail factoid queries. Rerank with the embedding. Better numbers, small implementation cost.
- **A colder cold-start path.** New shards take a while to warm up their LRU caches. Prewarming with a query-pattern prior would help.

Full paper on [ResearchGate](https://www.researchgate.net/profile/Yugandhar-Gogireddy/publication/384681635).

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1908.10084" target="_blank" rel="noopener"><div class="research-card__title">Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</div><div class="research-card__authors">Reimers & Gurevych · EMNLP 2019 · the Sentence Transformer paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1702.08734" target="_blank" rel="noopener"><div class="research-card__title">Billion-scale similarity search with GPUs</div><div class="research-card__authors">Johnson, Douze, Jégou · 2017 · the FAISS paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1603.09320" target="_blank" rel="noopener"><div class="research-card__title">Efficient and robust approximate nearest neighbor search using HNSW</div><div class="research-card__authors">Malkov & Yashunin · 2016 · the HNSW paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noopener"><div class="research-card__title">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</div><div class="research-card__authors">Lewis et al. · NeurIPS 2020 · the RAG paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2004.04906" target="_blank" rel="noopener"><div class="research-card__title">Dense Passage Retrieval for Open-Domain Question Answering</div><div class="research-card__authors">Karpukhin et al. · EMNLP 2020 · DPR</div></a></li>
  <li><a class="research-card" href="https://www.researchgate.net/profile/Yugandhar-Gogireddy/publication/384681635" target="_blank" rel="noopener"><div class="research-card__title">End-to-end Neural Embedding Pipeline for Large-scale PDF Document Retrieval</div><div class="research-card__authors">Challagundla et al. · 2024 · this paper</div></a></li>
</ul>
