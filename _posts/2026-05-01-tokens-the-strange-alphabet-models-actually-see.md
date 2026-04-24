---
layout: post-article
title: "Tokens: The Strange Alphabet Models Actually See"
date: 2026-05-01
permalink: /posts/tokens-the-strange-alphabet-models-actually-see/
excerpt: "A language model doesn't read words. It doesn't read letters either. It reads this weird third thing called tokens — and once you understand them, half the model's mysterious quirks stop being mysterious."
read_time_label: "10 min read"
accent: amber
math: true
---

Here's a small puzzle that has ruined a thousand pub conversations.

Ask GPT-4, *"How many r's are in strawberry?"* and for most of 2024, it said two. Many rounds of "are you sure?" would not fix it. Two. Final answer. Two.

The joke is that the model is bad at reading. The truth is stranger: **the model never sees letters at all.** It sees tokens. "strawberry" is roughly one token. There is no mechanism inside the machine for counting the `r`s in it, because — from the model's perspective — the word isn't a sequence of letters. It's a single opaque symbol, a blob, a hieroglyph that simply *means* strawberry.

Welcome to tokenization. It is responsible for surprisingly many of the dumb things language models do.

---

## Play with it first

{% include demos/tokenizer-playground.html %}

Try the "Long number" preset. Notice how `123456789` becomes multiple tokens — and not in the way a human would split it. Try the emoji preset. Watch `❤️` explode into three pieces (because a real emoji is not one byte, and the tokenizer's table mostly covers English bytes). Try the Japanese preset — each character might be its own token, or even multiple tokens per character.

Every weird thing above is doing real work. Let's unpack.

## What a token actually is

A **token** is a subword piece of text that lives in a fixed vocabulary of (typically) 30,000 to 200,000 entries. For GPT-2 the number is 50,257. For Claude and GPT-4 it's bigger.

The vocabulary is learned once, before training, by looking at a huge pile of text and running an algorithm called **Byte-Pair Encoding (BPE)**:

1. Start with every single byte as its own token. (So 256 starter tokens.)
2. Find the pair of adjacent tokens that co-occurs most often in the training corpus.
3. Merge it. You now have token 257: `" t"` say, because space-then-t was the most common pair.
4. Repeat 50,000 times.

What comes out is a vocabulary where the most common words are single tokens (`"the"`, `" and"`, `" is"`) and rare words get split into pieces. The result is a pretty decent compression of English at roughly 4 characters per token — but it gets wonky fast when you leave English.

<aside class="callout callout--analogy">
  <div class="callout__label">Analogy</div>
  <p>Imagine you had to write English but could only use pre-cut paper scraps from a bag of 50,000 shapes. Most common words each have their own scrap. Rare words don't, so you tape together a few scraps to spell them. "dog" is one scrap. "oncology" is three. You are the tokenizer. Welcome aboard.</p>
</aside>

## Why we don't just use characters

Seems like an obvious simplification, right? 128 ASCII tokens, done. No BPE, no weirdness, no strawberry problem.

Two issues:

**1. Sequence length blows up.** A 500-word article is maybe 2,500 characters. As tokens (at ~4 chars/token) that's 625 tokens. As characters it's 2,500. Attention scales with the square of sequence length, so that's *16× more compute* per forward pass. Untenable for long texts.

**2. The model has to learn English from scratch.** With subword tokens, `" dog"` as an embedding already carries most of the "doggishness" the model needs. With characters, the model has to assemble meaning from sequences like `d-o-g` — which is possible, but a huge chunk of its capacity gets burned on learning spelling before it can even think about meaning.

Subword tokenization is the compromise. Frequent things get their own slot. Rare things get pieced together. It's not elegant. It just works.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>Every interpretability result you'll read is at the <em>token</em> level. "Neuron 1523 fires on the DNA token." "Head 7 in layer 4 copies the previous token." When a paper talks about "what the model sees at position 3," position 3 is a token, not a letter. Internalise this and the whole literature reads smoother.</p>
</aside>

## The weird things this causes

### Numbers are a mess

Pick the "Long number" preset and stare. In GPT-2's tokenizer, `123456789` splits into chunks of unpredictable size, often 3 digits at a time, because 3-digit chunks are what happened to be most common in training text (prices, years, whatever). Different numbers tokenize differently. Which means the model is doing arithmetic on *symbols it has to recombine*, not on the digits it sees. That's why LLMs suck at multiplication and are merely okay at addition — and it's also why *instruction-tuned* models that get trained with each digit as its own token (Llama 3, recent GPTs) got dramatically better at math.

### Spaces are part of the token

This one trips everyone up on their first day. The token for "the" as a standalone is different from the token for " the" (with a leading space). The space is *included in the token*. A sentence is really a sequence of (space-prefixed word) tokens, which is why the model handles word-boundary information naturally and why you see " cat" and "cat" as separate items in the vocabulary.

### Emoji and non-ASCII get murdered

Open the emoji preset. `❤️` is one grapheme to you, but three bytes in UTF-8, and since BPE was trained mostly on English the table has no merge for those specific bytes. So the tokenizer falls back to emitting each byte as its own token. Three tokens, one heart. Which means if you want the model to reliably manipulate emoji, you're spending 3× the token budget per character. Some tokenizers (Claude, GPT-4o) have improved this by training on more multilingual data, but the ratio is still worse than ASCII.

### The strawberry problem

Alright — the question that opened this post. Why does the model get "count the r's in strawberry" wrong?

`strawberry` tokenizes as one or two tokens, depending on the model. Either way, there is no point in the forward pass where the model is *looking at individual letters*. It's looking at a few dense vectors that represent the whole thing. To count `r`s, the model would have to have learned, during training, something like *"when asked about letter counts, internally decompose tokens back into their spelling and count."* That's a specific, non-trivial skill — and it turns out models mostly didn't learn it, because it's a rare task in training data. Recent models have been explicitly trained on this and do better, but it remains a beautiful example of how **tokenization shapes capability**.

## Why different models see different tokens

Here's a subtle one. **Every model has its own tokenizer.** Not just different vocab sizes — different merges, different byte-level conventions.

- GPT-2: 50,257 tokens, byte-level BPE.
- GPT-4: ~100,277 tokens, `cl100k_base` encoding with more multilingual coverage.
- Llama 3: 128,256 tokens, mostly overlapping with GPT-4 but not identical.
- Claude: ~100,000+ tokens, byte-pair based.

Which means an MI result on GPT-2 at "token position 7" is not directly comparable to the same prompt at position 7 in Claude. The tokenization has to be recomputed per model. One of the first things you do when studying a new model is dump its tokenizer and look at the vocabulary.

## Tokens as the unit of everything

This is the crucial reframe. Inside the transformer:

- The **embedding table** has one row per token.
- **Attention** computes a weight between every pair of *token positions*.
- The **loss** during training is averaged over tokens.
- **Context length** is measured in tokens (not words, not characters).
- **Cost** of API calls is priced in tokens.

The token is the atom. Every operation inside the machine is an operation on tokens. If you want to understand a model's internals, you have to train your eye to see text the way it does — as a sequence of these weird little glued-together shards.

## A small thing about "BOS" and special tokens

One more bit of housekeeping. Most models inject a **beginning-of-sequence** (BOS) token at the start of every input, plus optional system, user, and assistant role markers in chat models. These aren't drawn from the regular vocabulary — they're special reserved IDs. You'll see them in interpretability papers as `<|endoftext|>` or `<s>` or `[INST]`. They're part of the sequence the model processes, and some surprisingly deep results ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)) found that BOS tokens accumulate a lot of interesting internal state — kind of a scratchpad the model can use however it likes. More on that when we get to the residual stream.

## Wrap

That's tokens. The lesson to tattoo on your forearm: **the model's alphabet is not your alphabet.** Every inference, every attention pattern, every feature the MI community has ever found — it's all defined relative to this 50,000-to-200,000-entry subword vocabulary that got distilled from billions of words of training data by a frequency-counting algorithm.

Tokens go into the embedding table and come out as 768-dimensional vectors, which then get dropped onto the residual stream. Which is the conveyor belt. Which is the main character of the entire architecture and which I'll write about in the next blog.

## Research referenced in this post

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1508.07909" target="_blank" rel="noopener"><div class="research-card__title">Neural Machine Translation of Rare Words with Subword Units</div><div class="research-card__authors">Sennrich, R. et al. · 2015 · the BPE paper</div></a></li>
  <li><a class="research-card" href="https://github.com/openai/tiktoken" target="_blank" rel="noopener"><div class="research-card__title">tiktoken</div><div class="research-card__authors">OpenAI · open-source tokenizer used by GPT-3.5/4</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2305.14788" target="_blank" rel="noopener"><div class="research-card__title">Tokenization and the Noiseless Channel</div><div class="research-card__authors">Zouhar, V. et al. · 2023 · why tokenization choice changes downstream performance</div></a></li>
  <li><a class="research-card" href="https://www.harmdevries.com/post/context-length/" target="_blank" rel="noopener"><div class="research-card__title">In the long (context) run</div><div class="research-card__authors">de Vries, H. · 2023 · tokens, context length, and what the numbers really mean</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2402.14903" target="_blank" rel="noopener"><div class="research-card__title">Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs</div><div class="research-card__authors">Singh, A. et al. · 2024 · why digit-level tokenization helps math</div></a></li>
</ul>
