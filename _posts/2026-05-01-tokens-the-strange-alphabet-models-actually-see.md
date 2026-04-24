---
layout: post-article
title: "Tokens: The Strange Alphabet Models Actually See"
date: 2026-05-01
permalink: /posts/tokens-the-strange-alphabet-models-actually-see/
excerpt: "A token is a subword unit drawn from a fixed vocabulary of 30K, 200K entries. Tokenization shapes capability: arithmetic, multilingual coverage, character-level reasoning all depend on it."
read_time_label: "10 min read"
accent: amber
math: true
---

A **token** is the atomic unit of input and output for a language model. It is not a word and not a character. It is a subword piece drawn from a fixed vocabulary $V$ of size 30,000 to 200,000.

This post defines tokens precisely, explains how the vocabulary is constructed (Byte-Pair Encoding), and walks through five concrete failure modes traceable directly to tokenization.

---

## Demo: tokenize anything

{% include demos/tokenizer-playground.html %}

Switch between presets. Note how `123456789` does not split into individual digits; how `❤️` consumes three tokens; how Japanese text consumes more tokens per character than English.

## Definition

A tokenizer is a deterministic function $\text{tok}: \text{string} \to [V]^*$ that maps text to a sequence of integer IDs. The vocabulary $V$ is a fixed lookup table containing strings called **token pieces**.

Modern LLMs use **byte-level BPE** (Byte-Pair Encoding):

```
1. Initialize V := {byte 0, byte 1, ..., byte 255}        # 256 entries
2. While |V| < target_size:
     a. Count adjacent token pairs in the training corpus.
     b. Find the most frequent pair (a, b).
     c. Add ab to V as a new token.
     d. Replace all occurrences of (a, b) in the corpus with ab.
3. Return V and the merge order.
```

Encoding new text greedily applies merges in the order they were learned. The output is the list of resulting token IDs.

Example (GPT-2 BPE):

```
"unhappily"  → ["un", "happ", "ily"]   → [403, 7829, 6148]
" the cat"   → [" the", " cat"]        → [262, 3797]
"strawberry" → ["str", "aw", "berry"]  → [2536, 707, 27078]
```

Common English words tend to be a single token. Rare or compound words split. Average compression: ~4 characters per token for English, less for other scripts.

## Why not characters?

A character-level model is conceptually simpler (256 ASCII tokens) but breaks at scale.

**1. Sequence length.** Self-attention is $O(T^2)$ in the sequence length $T$. A 2,500-character article is 625 BPE tokens vs 2,500 characters: a 16× compute increase per forward pass.

**2. Capacity.** Subword tokens already encode high-level meaning. The vector for `" doctor"` carries semantic content that a character-level model would have to assemble from `d`, `o`, `c`, `t`, `o`, `r`, burning model capacity on spelling.

BPE is the standard compromise: dense for frequent strings, sparse for rare ones.

<aside class="callout callout--key">
  <div class="callout__label">Why this matters for MI</div>
  <p>Every interpretability claim ("head 7 in layer 4 attends to the previous token", "neuron 1523 fires on DNA") is stated at the token level. "Position 3" means token 3, not character 3 or word 3. Mismatched tokenization breaks reproducibility across models.</p>
</aside>

## Five capability failures caused by tokenization

### 1. Arithmetic

In GPT-2's tokenizer, `123456789` splits into `["123", "456", "789"]` (or similar 3-digit chunks). Different numbers tokenize differently: `1234` may be `["12", "34"]` while `12345` may be `["123", "45"]`. The model performs arithmetic on these symbolic chunks, not on individual digits.

**Fix.** Recent models (Llama 3, GPT-4o math fine-tunes) use single-digit tokenization for numbers. Singh et al. ([2024](https://arxiv.org/abs/2402.14903)) show this alone yields large gains on multi-digit arithmetic.

### 2. Spaces are part of the token

`" cat"` (with leading space) and `"cat"` (without) are different tokens with different embedding rows. A sentence is typically a sequence of space-prefixed words: `"the cat sat"` → `["the", " cat", " sat"]`. The first word lacks a leading space; subsequent words include it. This is why prompts that omit a leading space sometimes produce subtly different outputs.

### 3. Non-ASCII

UTF-8 encodes `❤️` as 3 bytes (`\xe2\x9d\xa4` for `❤`, plus the variation selector). BPE trained mostly on English never merged these byte sequences. Result: each emoji byte is one token. Consequences:

- 3× token cost per emoji.
- Models cannot reliably match-and-modify emoji at the substring level.
- Multilingual scripts (Hindi, Arabic, CJK) suffer similarly. Llama 3's larger 128K vocabulary mitigates this; older 50K vocabularies do not.

### 4. The "strawberry" problem

`strawberry` is one or two tokens depending on the model. The forward pass never sees individual letters: it operates on dense vectors representing whole subwords. To answer "how many r's in strawberry?" the model must have *learned* to internally spell tokens back out, a non-trivial skill that requires it in training. As of 2024 most models still failed this; explicit training on letter-counting tasks fixes it.

### 5. Per-model vocabularies are incompatible

| Model | Vocabulary size | Encoding |
|---|---|---|
| GPT-2 | 50,257 | byte-level BPE |
| GPT-3.5 / GPT-4 | 100,277 | `cl100k_base` |
| GPT-4o | 200,019 | `o200k_base` |
| Llama 3 | 128,256 | tiktoken-based BPE |
| Claude | ~100K+ | proprietary BPE |

A given prompt produces a different number of tokens at different positions in each model. Position-indexed MI results do not transfer without re-tokenization.

## Tokens are the unit of everything

Inside the transformer:

- **Embedding table** $W_E \in \mathbb{R}^{V \times d_\text{model}}$: one row per token.
- **Attention scores** are computed between pairs of token positions.
- **Training loss** is cross-entropy averaged over predicted tokens.
- **Context length** is in tokens (e.g. GPT-4o: 128K tokens, Claude: 200K tokens).
- **API pricing** is per token.

Every operation in the model is an operation on tokens.

## Special tokens

Most models reserve specific IDs outside the regular BPE vocabulary:

- `<|endoftext|>` (`<bos>`, `<eos>`): document boundary.
- `<|im_start|>`, `<|im_end|>`: chat role markers (OpenAI ChatML format).
- `[INST]`, `[/INST]`: instruction markers (Llama 2/3 chat format).
- `<|system|>`, `<|user|>`, `<|assistant|>`: chat-tuned models.

These tokens behave like any other input but carry structural meaning. The BOS token in particular accumulates state that several attention heads use as a "rest" position, often called the **BOS sink** ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453); also discussed in [Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)).

## Practical: tokenize and inspect

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
ids = enc.encode("How many r's are in strawberry?")
pieces = [enc.decode([i]) for i in ids]
print(list(zip(ids, pieces)))
# [(4438, 'How'), (1690, ' many'), (436, ' r'), (596, "'s"), ...]
```

For Llama / Mistral / open models:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
print(tok.tokenize("strawberry"))
# ['str', 'aw', 'berry']
```

When studying a new model, dump its vocabulary and search for prompt fragments before drawing conclusions about positions.

## Wrap

The model's alphabet is not the user's alphabet. Tokens are the atom: every embedding lookup, attention computation, and loss term is defined per token. Tokenization choice has measurable downstream effects on arithmetic, multilingual fluency, and character-level reasoning.

The next post is on the residual stream: what those token vectors do once they enter the transformer.

## Resources

### Foundational papers

<ul class="research-list">
  <li><a class="research-card" href="https://arxiv.org/abs/1508.07909" target="_blank" rel="noopener"><div class="research-card__title">Neural Machine Translation of Rare Words with Subword Units</div><div class="research-card__authors">Sennrich et al., 2015 · the BPE paper</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2305.14788" target="_blank" rel="noopener"><div class="research-card__title">Tokenization and the Noiseless Channel</div><div class="research-card__authors">Zouhar et al., 2023 · how tokenization choice affects downstream loss</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2402.14903" target="_blank" rel="noopener"><div class="research-card__title">Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs</div><div class="research-card__authors">Singh et al., 2024 · single-digit tokenization fixes arithmetic</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/2309.17453" target="_blank" rel="noopener"><div class="research-card__title">Efficient Streaming Language Models with Attention Sinks</div><div class="research-card__authors">Xiao et al., 2023 · BOS-token attention sinks</div></a></li>
  <li><a class="research-card" href="https://arxiv.org/abs/1808.06226" target="_blank" rel="noopener"><div class="research-card__title">SentencePiece: A simple and language independent subword tokenizer</div><div class="research-card__authors">Kudo & Richardson, 2018 · the alternative to BPE used by T5/Llama</div></a></li>
</ul>

### Tools and code

<ul class="research-list">
  <li><a class="research-card" href="https://github.com/openai/tiktoken" target="_blank" rel="noopener"><div class="research-card__title">tiktoken</div><div class="research-card__authors">OpenAI · fast BPE tokenizer for GPT-3.5/4/4o</div></a></li>
  <li><a class="research-card" href="https://platform.openai.com/tokenizer" target="_blank" rel="noopener"><div class="research-card__title">OpenAI Tokenizer Playground</div><div class="research-card__authors">visualize how any string tokenizes for GPT-3.5/4/4o</div></a></li>
  <li><a class="research-card" href="https://github.com/karpathy/minbpe" target="_blank" rel="noopener"><div class="research-card__title">minbpe</div><div class="research-card__authors">Karpathy · minimal BPE implementation; train your own tokenizer in &lt;200 lines</div></a></li>
  <li><a class="research-card" href="https://www.youtube.com/watch?v=zduSFxRajkE" target="_blank" rel="noopener"><div class="research-card__title">Let's build the GPT Tokenizer</div><div class="research-card__authors">Karpathy · 2-hour deep dive on BPE construction and edge cases</div></a></li>
  <li><a class="research-card" href="https://huggingface.co/docs/tokenizers/index" target="_blank" rel="noopener"><div class="research-card__title">Hugging Face Tokenizers</div><div class="research-card__authors">production-grade BPE, WordPiece, Unigram implementations</div></a></li>
</ul>
