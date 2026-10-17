# 42. Tokens, embeddings, and the transformer block

*Track B (LLM fundamentals). Book: Appendix F (Under the Hood) and Chapter 16 (Resource-Aware Optimization). Prerequisite: 00. Course orientation. Recommended before 43. Attention, the KV cache, GQA/MLA, and MoE.*

Every budget you set in an agent is denominated in tokens. Every latency you measure is a count of forward passes. Every "context window" argument you have with a teammate is an argument about a matrix shape. This lesson opens the box far enough that those quantities stop being abstractions. You will implement the two mechanisms that matter most for an agent engineer: how text becomes token ids, and what one decoder block does to them.

## The problem

The support agent from 00. Course orientation is answering "Why is Acme complaining about their invoice?" It pulled a Slack thread, a Notion page, a Stripe invoice JSON, and a SQL result into the context. The run failed with a context-length error. The engineer on call looks at the inputs: about 40,000 characters. "That is nowhere near 128K," they say, and open a ticket against the provider.

The provider is right and the engineer is wrong, for three reasons that all live in this lesson.

First, the limit is in tokens, not characters, and the ratio is not fixed. English prose tends to run around four characters per token. JSON with quoted keys, UUIDs, and stack traces can run two or fewer. The Stripe payload was 12,000 characters and 7,000 tokens.

Second, the "128K" window is shared between input and output, plus a reserved output budget the API refuses to overrun.

Third, the same text costs different amounts in different tokenizers. Moving from one model to another changed the count by 20 percent without anyone changing a prompt.

```text
naive mental model:            what the model sees:

  "text"  --->  model            "text" --> tokenizer --> [ids] --> embedding rows
   40K chars, fine                  40K chars -> 19K tokens (JSON heavy)
                                    + 4K system prompt + 8K tool schemas
                                    + 16K reserved for output
                                    = 47K...  but the deploy used a 32K model
```

The fix was not a bigger model. It was measuring the right unit, trimming tool results (see 34. Context Engineering), and understanding why the count came out the way it did. That understanding is what this lesson builds.

### Why tokens are not words

A language model does not read characters or words. It reads integers from a fixed vocabulary, typically 32K to 256K entries. The tokenizer is a deterministic compression scheme that maps bytes to those integers. Modern models use byte-level Byte Pair Encoding (BPE): start with the 256 possible bytes as the vocabulary, then repeatedly merge the most frequent adjacent pair in a training corpus into a new symbol until the vocabulary budget is reached. Llama 3 uses a 128K-entry vocabulary built this way (Llama 3 model report, 2024).

Three consequences fall directly out of the algorithm:

| Fact about BPE | Consequence for you |
|---|---|
| Frequent strings become single tokens | "invoice" is one token; "invoices" may be three |
| Anything can fall back to bytes | No out-of-vocabulary error, but rare strings (IDs, hashes, non-Latin text) cost many tokens |
| The merge table is fixed at training time | Tokenization is deterministic; the same string always costs the same in the same model |

This is also why models are bad at counting letters and doing arithmetic on long numbers. The model never saw "strawberry" as letters; it saw one or two opaque ids. Digits get chunked into tokens whose boundaries have nothing to do with place value.

### Where cost comes from

Cost has two components: what you pay the provider per token, and what the hardware does per token. Both matter for an agent because agents burn tokens in loops.

```text
one agent step:
  prefill:  process the whole prompt (system + history + tool results)  -> pay for every token
  decode:   generate the response one token at a time                   -> pay again per output token
  repeat:   next step re-sends the whole transcript                      -> prefill cost grows every step
```

A 20-step agent whose transcript grows by 2K tokens per step re-reads a growing prefix each step. Without prompt caching (see 38. LLM Serving Infrastructure) the total input tokens are quadratic in the number of steps. Understanding what a forward pass actually computes tells you which of those tokens are expensive and which are nearly free.

## Build it without a framework

Start with tokenization. This is a toy BPE trainer on a tiny corpus about invoices, but the algorithm is the real one: count adjacent pairs, merge the most frequent, repeat. Encoding applies the learned merges in order.

```{code-cell} python
from collections import Counter

def bpe_train(corpus: str, num_merges: int):
    """Byte-level BPE: start from single bytes, merge the most frequent adjacent pair."""
    words = Counter(corpus.split())  # frequency-weighted word list
    # Each word is a tuple of byte-symbols; the trailing marker keeps word boundaries.
    vocab = {tuple(bytes([b]).decode("latin-1") for b in w.encode("utf-8")) + ("</w>",): n for w, n in words.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, n in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # tie-break deterministically
        merges.append(best)
        merged = {}
        for symbols, n in vocab.items():
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1]); i += 2
                else:
                    out.append(symbols[i]); i += 1
            merged[tuple(out)] = n
        vocab = merged
    return merges

def bpe_encode(word: str, merges):
    """Apply learned merges in order. The output is the token sequence."""
    symbols = [bytes([b]).decode("latin-1") for b in word.encode("utf-8")] + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols

corpus = ("the invoice was paid the invoice was refunded the customer asked about the invoice "
          "invoice invoice invoice refund refund the the the")
merges = bpe_train(corpus, num_merges=12)
print("first merges:", merges[:5])
print("invoice ->", bpe_encode("invoice", merges))
print("invoices ->", bpe_encode("invoices", merges))
print("zebra   ->", bpe_encode("zebra", merges))
print("ünïcode ->", bpe_encode("ünïcode", merges))

assert len(bpe_encode("invoice", merges)) < len(bpe_encode("zebra", merges)), "frequent word gets fewer tokens"
assert "</w>" in "".join(bpe_encode("zebra", merges)), "no OOV: rare word falls back to bytes"
assert len(bpe_encode("ünïcode", merges)) > len("ünïcode"), "non-ASCII costs more tokens than characters"
```

Look at the output. "invoice" is one token because it was frequent. "invoices" is six, because the plural never appeared and the merges that built "invoice" stop early on the different byte sequence. The accented word costs more tokens than it has characters, because each non-ASCII character is two UTF-8 bytes and none of the pairs were merged. Real tokenizers see far more data, so their merges are better, but the shape of the outcome is identical: frequent, English, prose-like text is cheap; rare identifiers, code, and other scripts are expensive.

Now the model. A decoder-only transformer is a stack of identical blocks operating on a matrix of shape `(sequence_length, d_model)`. That matrix is called the **residual stream**. Each block reads from it, computes an update, and adds the update back. Nothing replaces the stream; everything adds to it. That is what makes 80 layers trainable and it is why interpretability work talks about "reading" and "writing" to the residual stream.

```text
token ids  [3, 17, 17, 42, 8, 3]
    |
    v  embedding lookup: E[ids]  -> (T, d_model)          <- one row per id
    |
    v  block 1:  h = h + Attn(RMSNorm(h))                  <- mix information across positions
    |            h = h + MLP(RMSNorm(h))                   <- transform each position independently
    v  block 2 ... block L  (same shape in, same shape out)
    |
    v  final norm, then  logits = h @ E^T  -> (T, vocab)   <- score every next token, at every position
```

Here is a single Llama-style block in numpy: RMSNorm, causal multi-head self-attention, residual add, RMSNorm, SwiGLU MLP, residual add. Then a tied unembedding to get logits.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(0)
V, d_model, n_heads, T = 50, 32, 4, 6          # toy sizes
d_head = d_model // n_heads
d_ff = 4 * d_model

# Parameters. In a real model these are the learned weights.
E = rng.normal(0, 0.02, (V, d_model))          # embedding matrix: one row per token id
W_q, W_k, W_v, W_o = (rng.normal(0, 0.02, (d_model, d_model)) for _ in range(4))
W_gate, W_up = (rng.normal(0, 0.02, (d_model, d_ff)) for _ in range(2))
W_down = rng.normal(0, 0.02, (d_ff, d_model))
g1, g2 = np.ones(d_model), np.ones(d_model)   # RMSNorm scales

def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def causal_attention(x):
    """x: (T, d_model). Split into heads, mask the future, mix values, merge heads."""
    T = x.shape[0]
    q = (x @ W_q).reshape(T, n_heads, d_head).transpose(1, 0, 2)   # (H, T, d_head)
    k = (x @ W_k).reshape(T, n_heads, d_head).transpose(1, 0, 2)
    v = (x @ W_v).reshape(T, n_heads, d_head).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)             # (H, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)                # True above the diagonal = future
    scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores)
    out = (weights @ v).transpose(1, 0, 2).reshape(T, d_model)
    return out @ W_o, weights

def silu(z):
    return z / (1 + np.exp(-z))

def swiglu_mlp(x):
    return (silu(x @ W_gate) * (x @ W_up)) @ W_down

def decoder_block(h):
    """Pre-norm block, Llama style. The residual stream h is read and written, never replaced."""
    attn_out, weights = causal_attention(rmsnorm(h, g1))
    h = h + attn_out                      # residual add #1
    h = h + swiglu_mlp(rmsnorm(h, g2))    # residual add #2
    return h, weights

token_ids = np.array([3, 17, 17, 42, 8, 3])
h0 = E[token_ids]                          # embedding lookup is just row indexing
h1, weights = decoder_block(h0)
logits = h1 @ E.T                          # tied unembedding: back to vocabulary space
probs = softmax(logits)

print("residual stream shape:", h1.shape, " logits shape:", logits.shape)
print("attention weights, head 0, last position:", np.round(weights[0, -1], 3))
assert h1.shape == (T, d_model) and logits.shape == (T, V)
assert np.allclose(weights.sum(-1), 1.0), "each row of attention weights is a distribution"
assert np.all(weights[:, 0, 1:] == 0), "position 0 cannot attend to the future"
assert np.allclose(probs.sum(-1), 1.0)
# Same token id at positions 1 and 2 gets identical embeddings but different block outputs,
# because attention mixes in different contexts.
assert np.allclose(h0[1], h0[2]) and not np.allclose(h1[1], h1[2])
# Residual stream: the block adds to the input rather than replacing it, so the update is
# a perturbation of the stream, not a fresh representation.
assert np.linalg.norm(h1 - h0) < np.linalg.norm(h0) * 5
```

Read the asserts as the lesson. The embedding lookup is row indexing: token 17 gets the same vector wherever it appears. Only attention can make position 1 and position 2 differ, because only attention looks across positions. The MLP is applied to each row independently; it is where most of the parameters live and where most "knowledge" is stored, but it cannot move information between tokens. The causal mask zeroes the upper triangle so position `i` sees only positions `0..i`. That single constraint is what makes the KV cache in the next lesson exact.

One thing this cell omits: positional information. Without it, attention is permutation-invariant. Llama-style models inject position by rotating query and key vectors (RoPE), which is covered in 43. Attention, the KV cache, GQA/MLA, and MoE because it interacts with the cache.

### What a forward pass costs

For a dense model the compute per token is dominated by matrix multiplications against the weights, and each weight participates in one multiply and one add per token. So the rule of thumb is:

```text
FLOPs per token  ~=  2 x (number of parameters)        (ignores the attention n^2 term)

Llama-3-70B:  2 x 70e9 = 1.4e11 FLOPs per token
8,000-token prompt (prefill):   8,000 x 1.4e11 = 1.1e15 = 1,100 TFLOP
H100 at ~990 TFLOPS BF16:       about 1.1 s of pure compute for prefill, if perfectly utilized
```

Let me reconstruct the parameter count from the published shapes so the number is not magic.

```{code-cell} python
def dense_params(n_layers, d_model, d_ff, vocab, n_heads, n_kv_heads):
    d_head = d_model // n_heads
    attn = d_model * (n_heads * d_head) * 2 + d_model * (n_kv_heads * d_head) * 2   # Wq, Wo, Wk, Wv
    mlp = 3 * d_model * d_ff                                                          # gate, up, down
    return n_layers * (attn + mlp) + vocab * d_model   # + embeddings (untied output head doubles this)

llama70 = dict(n_layers=80, d_model=8192, d_ff=28672, vocab=128_256, n_heads=64, n_kv_heads=8)
p = dense_params(**llama70)
print(f"Llama-3-70B non-embedding+embedding params ~ {p/1e9:.1f}B")
flops_per_token = 2 * p                       # one multiply-add per weight per token
prompt_tokens, output_tokens = 8_000, 500
prefill = flops_per_token * prompt_tokens
decode = flops_per_token * output_tokens
print(f"prefill {prefill/1e12:.0f} TFLOP, decode {decode/1e12:.0f} TFLOP (excluding attention's n^2 term)")
h100_bf16 = 990e12
print(f"ideal prefill time on one H100: {prefill/h100_bf16*1000:.0f} ms")
assert 65e9 < p < 72e9, "should land near the advertised 70B"
assert prefill > decode * 10, "prefill dominates FLOPs for long prompts; decode dominates wall time"
```

The last assert names the paradox that drives the whole serving lesson. Prefill does far more arithmetic than decode, but decode takes far longer, because each decode step must read all 140 GB of weights from memory to produce one token. Prefill is compute-bound; decode is memory-bandwidth-bound (see 38. LLM Serving Infrastructure). Knowing which regime you are in tells you whether batching helps, whether quantization helps, and why output tokens are priced higher than input tokens.

## How it shows up in the stack

An agent engineer rarely touches these matrices directly. But the numbers they produce are the constraints you design around.

**Context budgets are matrix dimensions.** The `T` in `(T, d_model)` is the context length. Every token you put in the prompt is a row that every layer must process, and a row of K and V that must be stored. When 34. Context Engineering tells you to keep tool results small and to compact old turns, this is why: the cost is per row, per layer, every step.

**Count tokens, not characters, and count them with the right tokenizer.** `langchain_core` messages expose `usage_metadata` after a model call with `input_tokens` and `output_tokens`. Before a call, providers expose count-tokens endpoints; use them for admission control on tool results rather than a characters-divided-by-four guess, which is off by 2x on JSON.

**Tool schemas are tokens.** Every tool you bind is serialized into the prompt. Thirty tools with verbose descriptions can be 8K tokens of prefix that the model re-reads every step. This is the mechanical reason behind tool retrieval (load only relevant tools per step) and behind prompt caching (make the expensive prefix identical across calls so the provider can reuse its prefill work).

**Prefill versus decode explains pricing and latency.** Time-to-first-token is prefill. Tokens-per-second after that is decode. An agent step that reads 30K tokens and writes 200 has a TTFT problem; one that writes a 3K-token plan has a decode problem. They need different fixes.

```text
agent step timeline:
|<---- prefill (compute-bound, all prompt tokens at once) ---->|<- decode: one token per pass ->|
 TTFT                                                           TPOT x n_out
```

**Embeddings for retrieval are the same object.** The `E` matrix maps ids to vectors inside the model. A retrieval embedding model is a transformer whose final residual stream is pooled into one vector per passage. When 45. Retrieval internals compares dense and sparse retrieval, "dense" means exactly this kind of vector.

**Tokenization explains failure modes you will see in evals.** The model cannot reliably count characters, reverse strings, or do long arithmetic, because it does not see characters. Give it a tool. The book's Chapter 5 says this in terms of tool use; the tokenizer says it in terms of mechanism.

## What goes wrong in production

1. **Character-based budgets.** Symptom: intermittent context-length errors that correlate with JSON-heavy or non-English inputs. Cause: dividing characters by four. Fix: count with the target model's tokenizer, budget per message type (system, tools, history, tool results, reserved output), and enforce the budget in code before the call.

2. **Tokenizer drift across model swaps.** Symptom: a fallback route to a second provider starts truncating or overflowing. Cause: different vocabularies tokenize the same prefix differently; a stable "20K prompt" is 24K somewhere else. Fix: budgets are per-model; recompute on route change; keep 10 to 15 percent headroom.

3. **Identifier-heavy tool results.** Symptom: a Stripe or SQL result "looks short" but consumes half the window. Cause: UUIDs, timestamps, and hashes fall back to near-byte tokenization. Fix: project tool results to the fields the model needs, replace ids with short aliases the application maps back, and store the full payload outside the context (see 34. Context Engineering on just-in-time retrieval).

4. **Quadratic transcript cost in long loops.** Symptom: cost per run grows much faster than steps per run. Cause: every step re-prefills the whole growing transcript. Fix: stable prefixes for caching, compaction of old tool results, and hard step budgets. Measure input tokens per run, not per call.

5. **Asking the model to do tokenizer-hostile work.** Symptom: wrong letter counts, wrong sums, mangled string reversals, then a "fix" via prompting. Cause: the model never sees characters. Fix: a calculator or code tool. Evals should include these cases so regressions are caught.

6. **Trusting the reserved output budget by accident.** Symptom: a long structured output is silently cut mid-JSON. Cause: `max_tokens` was reached, the finish reason said so, and nobody checked it. Fix: check the finish reason on every call; treat truncation as an error with a repair path (see 44. Decoding, reasoning models, and structured output).

## What changed by 2026

The transformer block in this lesson is, in its essentials, the block from "Attention Is All You Need" (2017) with the refinements that Llama-family models standardized: pre-norm with RMSNorm, SwiGLU MLPs, rotary positions, and no biases. What changed is around the block.

Vocabularies got bigger. Llama 3 moved to a 128K-entry tokenizer, and large vocabularies are now common, because more merges mean fewer tokens per document, which is cheaper for the user and gives the model more text per unit of context. Multilingual and code tokenization improved for the same reason.

Context windows grew from 4K to hundreds of thousands of tokens, but the cost per token did not disappear. The response from the agent-engineering side was context engineering: compaction, structured note-taking outside the window, subagents with their own windows, and just-in-time retrieval instead of preloading. Anthropic's guidance on context engineering frames the window as a budget to be managed rather than a bucket to be filled.

Provider prompt caching became standard. Because prefill is deterministic given the same prefix, providers can store the intermediate state for a prefix and charge a discount when it is reused. This turned "keep the system prompt and tool schemas byte-identical across calls" from a nicety into a cost lever. Current provider docs describe cached-input pricing at a large discount to uncached input; check the current numbers rather than memorizing them.

Attention itself became sparser and cheaper in the KV dimension, through grouped-query and latent attention, and the MLP became conditional through mixture-of-experts. Those are the subject of the next lesson.

What did not change: the model still sees integers, the residual stream is still `(T, d_model)`, and every token you add is a row every layer must touch.

## Interview questions

**Q1. Why are tokens not words, and why should an application engineer care?**

Strong: Modern models use byte-level BPE. The tokenizer starts from raw bytes and greedily merges the most frequent adjacent pairs until it hits a vocabulary budget, so frequent strings become single tokens and rare strings fall back toward bytes. That means token count is data-dependent: English prose is around four characters a token, JSON and identifiers can be two or fewer, and non-Latin scripts cost more. I care because every limit and price is in tokens, the ratio differs across models, and it explains model weaknesses like letter counting and arithmetic, which I route to tools.

Weak: "A token is about three-quarters of a word."

**Q2. Walk me through one decoder block.**

Strong: The input is the residual stream, a matrix of one vector per token. A pre-norm block applies RMSNorm, then causal multi-head self-attention, which is the only place information moves between positions, and adds the result back into the stream. Then RMSNorm again, a position-wise MLP such as SwiGLU that holds most of the parameters, and another residual add. Same shape in and out, so blocks stack. At the top, a final norm and a projection to vocabulary logits at every position.

Weak: "Attention then a feed-forward layer."

**Q3. What is the residual stream and why does it matter?**

Strong: It is the running `(T, d_model)` representation that each block reads from and adds to. Because blocks add rather than replace, gradients flow directly through the identity path, which is what makes very deep stacks trainable, and any layer can write a feature that a much later layer reads. It is also the reason a KV cache and interpretability probes are well-defined: each layer's contribution is an additive update to a shared space.

Weak: "It's a skip connection."

**Q4. Estimate the FLOPs for a forward pass.**

Strong: Roughly two FLOPs per parameter per token for a dense model, because each weight does one multiply and one add. For a 70B model that is about 1.4e11 per token, so an 8K-token prompt is about 1.1 petaFLOP of prefill, or around a second on one H100 at full utilization. Attention adds a term quadratic in sequence length, which starts to matter at long contexts. The more useful point is that prefill has lots of FLOPs per byte of weights read and is compute-bound, while decode reads every weight for one token and is bandwidth-bound.

Weak: "It depends on the GPU."

**Q5. Why is a model bad at counting letters in a word?**

Strong: It never sees letters. The tokenizer hands it one or two integer ids for "strawberry", and the embedding for that id encodes usage, not spelling. Any spelling knowledge has to be learned indirectly from contexts where words were spelled out. The same applies to arithmetic on long numbers, where digit grouping into tokens ignores place value. The engineering answer is to give the model a tool for that class of task and to include those cases in evals.

Weak: "The model isn't smart enough yet."

**Q6. Your agent's cost per run is growing faster than its step count. Why?**

Strong: Each step re-sends the whole transcript, so input tokens per step grow linearly and total input tokens per run grow quadratically. Tool results dominate that growth because they are identifier-heavy and tokenize poorly. Fixes are stable prefixes so the provider's prompt cache hits, compaction or externalization of old tool results, projecting results to needed fields, and a hard step budget. I would instrument input tokens per run and cache hit rate, not just calls.

Weak: "Use a cheaper model."

## Build it yourself

Extend the toy tokenizer and block into a small cost model.

1. Add `bpe_encode_text(text, merges)` that splits on whitespace, encodes each word, and returns the total token count.
2. Train merges on a prose corpus, then measure tokens per character for three inputs: an English paragraph, a JSON object with UUID values, and a short string of accented text.
3. Add a `count_cost(prompt_tokens, output_tokens, price_in, price_out)` function and simulate a 15-step agent loop whose transcript grows by a fixed number of tokens each step, with and without a cached prefix that costs 10 percent of full price.
4. In the decoder block, add a second block and verify the residual stream keeps its shape and that the logits at the last position change.

Acceptance checks:

- `tokens/char` for the JSON input is at least 1.5x the prose input on the same merges.
- The accented string produces more tokens than characters.
- The uncached 15-step total input token count is quadratic: doubling the steps roughly quadruples the sum of prompt tokens.
- The cached variant's total cost is less than half the uncached cost when the fixed prefix is at least 70 percent of each prompt.
- Stacking two blocks preserves `(T, d_model)` and `assert not np.allclose(logits_1, logits_2)`.

## Key takeaways

- Models read integers from a fixed BPE vocabulary; token count is data-dependent and never a fixed fraction of characters.
- Frequent prose is cheap; identifiers, code, and non-Latin text are expensive, and this drives context-length surprises.
- The embedding matrix is a lookup; only attention moves information between positions; the MLP holds most parameters.
- The residual stream is added to, never replaced, which makes deep stacks trainable and caches well-defined.
- A dense forward pass costs about two FLOPs per parameter per token; prefill is compute-bound, decode is bandwidth-bound.
- Agent loops re-read a growing transcript every step, so input cost is quadratic in steps without caching and compaction.
- Tokenizer-hostile tasks (counting, arithmetic) belong in tools, and in evals.
