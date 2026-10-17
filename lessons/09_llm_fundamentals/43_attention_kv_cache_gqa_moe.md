# 43. Attention, the KV cache, GQA/MLA, and MoE

*Track B (LLM fundamentals). Book: Appendix F (Under the Hood) and Chapter 16 (Resource-Aware Optimization). Prerequisite: 42. Tokens, embeddings, and the transformer block.*

The previous lesson built one decoder block. This lesson goes inside the attention half of it, because attention is where the memory goes, where the context-length limits come from, and where the last three years of architecture work concentrated. By the end you will be able to size a KV cache on a whiteboard, explain why a 47B-parameter model runs like a 13B one, and say what FlashAttention actually changed.

## The problem

The support agent is being moved from a hosted API to a self-hosted Llama-3-70B so that customer data stays inside the VPC. The platform team asks a simple question: how many concurrent conversations can one H100 node handle at an 8K context?

The naive answer counts weights. 70B parameters in FP16 is about 140 GB, so two 80 GB H100s hold the model with 20 GB spare. "Plenty of room for requests."

Then the first load test runs and the server rejects new sessions at a handful of concurrent users. Weights were never the problem. Each active conversation carries its own attention state, and that state is large.

```text
GPU memory during serving:

|<--------- weights (fixed) --------->|<----- KV cache (grows with tokens x users) ----->|<-workspace->|
   ~140 GB for 70B FP16                   320 KiB per token for Llama-3-70B
                                          8K tokens  x 1 user  = 2.5 GiB
                                          8K tokens  x 32 users = 80 GiB   <- one whole H100
```

To answer the platform team you need the KV cache formula, and to understand why the formula has the shape it does you need to know what attention stores and why it can store it.

### Attention, in one formula

Self-attention lets each position build its representation as a weighted mix of every earlier position's values. Three projections of the residual stream: queries `Q` (what am I looking for), keys `K` (what do I contain), values `V` (what do I pass along if selected).

```text
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Q: (T, d_k)   K: (T, d_k)   V: (T, d_v)
Q K^T: (T, T)  one score per (query position, key position)
softmax along each row -> weights sum to 1 -> weighted sum of value rows
```

"Attention Is All You Need" (2017) introduced the `1/sqrt(d_k)` scaling. The reason is variance: if the components of `q` and `k` have unit variance, their dot product has variance `d_k`. At `d_k = 128` raw scores have standard deviation about 11, and softmax of numbers that spread apart is nearly one-hot. Gradients vanish and the head stops learning. Dividing by `sqrt(d_k)` brings the variance back to one.

### Causal masking is what makes caching legal

A decoder predicts the next token, so position `i` may only attend to positions `0..i`. Implementation: set the scores above the diagonal to negative infinity before the softmax, so their weights are exactly zero.

The consequence that matters for serving: position `i`'s output depends only on positions `<= i`. When token `i+1` arrives, nothing about the keys and values of tokens `0..i` changes. They can be computed once and stored. That store is the **KV cache**. Queries are not cached because each query is used once, by its own position, and then never again.

### Position: RoPE

Attention as written is permutation-invariant; shuffle the rows and you shuffle the outputs. Something must encode position. Rotary Position Embedding (RoFormer, 2021) does it by rotating each 2-D pair of `q` and `k` components by an angle proportional to the absolute position `m`, with a different frequency `theta_i = base^(-2i/d)` per pair. Because a dot product of two rotated vectors depends only on the difference of their angles, the attention score depends on relative position `m - n`. Llama 3 raised the base from 10,000 to 500,000 to stretch the usable range.

RoPE does not solve length extrapolation on its own. Beyond the training length the low-frequency pairs see angles they never saw during training and quality falls off. Position Interpolation, NTK-aware scaling, and YaRN (2023) rescale the frequencies so a model trained at 4K can be adapted to 32K or more with little fine-tuning. Interview one-liner: RoPE encodes relative position through rotation; extrapolation fails because unseen angles appear, and the fixes rescale the angle schedule.

## Build it without a framework

First, attention itself, with the causal mask and a demonstration of what `sqrt(d_k)` protects against.

```{code-cell} python
import numpy as np
rng = np.random.default_rng(0)

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def attention(Q, K, V, causal=True, scale=True):
    """softmax(Q K^T / sqrt(d_k)) V for one head. Q:(Tq,d) K,V:(Tk,d)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)
    if causal:
        Tq, Tk = scores.shape
        # query i (at absolute position Tk-Tq+i) may see keys 0..Tk-Tq+i
        mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=Tk - Tq + 1)
        scores = np.where(mask, -np.inf, scores)
    w = softmax(scores)
    return w @ V, w

T, d = 8, 64
Q, K, V = (rng.normal(size=(T, d)) for _ in range(3))
out, w = attention(Q, K, V)
print("weights row 3:", np.round(w[3], 3))
assert out.shape == (T, d)
assert np.allclose(w.sum(-1), 1) and np.all(w[np.triu_indices(T, 1)] == 0)

# Why sqrt(d_k): unscaled dot products have variance d_k, so softmax saturates as d grows.
for dk in (16, 256, 4096):
    q, k = rng.normal(size=(2000, dk)), rng.normal(size=(2000, dk))
    raw = (q * k).sum(-1)
    print(f"d_k={dk:5d}  var(q.k)={raw.var():8.1f}  var(q.k/sqrt(d_k))={(raw/np.sqrt(dk)).var():.2f}")
_, w_unscaled = attention(rng.normal(size=(T, 4096)), rng.normal(size=(T, 4096)), V, scale=False)
_, w_scaled = attention(rng.normal(size=(T, 4096)), rng.normal(size=(T, 4096)), V, scale=True)
print("max weight, last row, unscaled vs scaled:", w_unscaled[-1].max().round(3), w_scaled[-1].max().round(3))
assert w_unscaled[-1].max() > 0.99, "unscaled attention at large d_k collapses to one-hot"
assert w_scaled[-1].max() < 0.9, "scaled attention stays soft"
```

Row 3 of the weights has four non-zero entries (positions 0 to 3) and zeros after. The variance table shows the raw dot product variance tracking `d_k` exactly and the scaled variance sitting at one. At `d_k = 4096` the unscaled softmax puts everything on one key; that is a dead head.

### The KV cache formula

For every token in the context, every layer stores one key vector and one value vector per KV head.

```text
KV bytes per token = 2 x n_layers x n_kv_heads x head_dim x bytes_per_value
                     ^ K and V

Llama-3-70B, FP16:   2 x 80 x 8 x 128 x 2 = 327,680 B = 320 KiB per token
8,192 tokens:        320 KiB x 8,192 = 2.5 GiB per sequence
batch of 32:         80 GiB               <- one H100's entire memory
same model with MHA (64 KV heads instead of 8): 2,560 KiB per token, 8x more
```

The figures come straight from the published Llama 3 shapes: 80 layers, 64 query heads, 8 KV heads, head dimension 128. Put the formula in code with the worked figures as asserts so you can never misremember it in an interview.

```{code-cell} python
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value   # 2 = K and V

KiB, GiB = 1024, 1024 ** 3
llama70 = dict(n_layers=80, n_kv_heads=8, head_dim=128)
per_tok = kv_bytes_per_token(**llama70)
print(f"Llama-3-70B FP16: {per_tok:,} B/token = {per_tok/KiB:.0f} KiB/token")
print(f"  8,192-token context: {per_tok*8192/GiB:.2f} GiB;  batch 32: {per_tok*8192*32/GiB:.0f} GiB")
mha = kv_bytes_per_token(80, 64, 128)
print(f"  same model with MHA (64 KV heads): {mha/KiB:.0f} KiB/token ({mha//per_tok}x)")
assert per_tok == 327_680 and per_tok / KiB == 320
assert abs(per_tok * 8192 / GiB - 2.5) < 0.01
assert per_tok * 8192 * 32 / GiB == 80
assert mha // per_tok == 8
assert kv_bytes_per_token(32, 8, 128) / KiB == 128, "the interview variant: 32 layers, 8 KV heads"
# FP8 KV cache halves it
assert kv_bytes_per_token(**llama70, bytes_per_value=1) / KiB == 160
```

Now the claim that the cache is exact. Decode token by token, keeping only past K and V, and compare against recomputing attention over the full sequence each step.

```{code-cell} python
d_model = 32
W_q, W_k, W_v = (rng.normal(0, 0.1, (d_model, d_model)) for _ in range(3))
x = rng.normal(size=(6, d_model))                 # six token embeddings arriving one at a time

# Full recompute: attend over the whole prefix every step.
full_out, _ = attention(x @ W_q, x @ W_k, x @ W_v)

# Cached: keep K and V for past tokens, project only the new token.
K_cache = np.zeros((0, d_model)); V_cache = np.zeros((0, d_model)); cached_out = []
for t in range(x.shape[0]):
    x_t = x[t:t + 1]
    K_cache = np.vstack([K_cache, x_t @ W_k]); V_cache = np.vstack([V_cache, x_t @ W_v])
    o, _ = attention(x_t @ W_q, K_cache, V_cache)   # one query row against all cached keys
    cached_out.append(o[0])
cached_out = np.array(cached_out)
print("max |cached - recompute| =", np.abs(cached_out - full_out).max())
assert np.allclose(cached_out, full_out), "causal masking makes past K/V immutable, so caching is exact"
assert K_cache.shape == (6, d_model)
# Q is never cached: each new token's query is used once and discarded.
```

The difference is floating-point noise. Each decode step does one row of projections and one row of attention instead of `T` rows, at the price of storing `2 x T x d_model` numbers per layer. Every serving system is built on this trade.

### MHA, MQA, GQA, MLA: shrinking what is stored

The formula has `n_kv_heads` in it, and that term is the lever.

| Scheme | K/V heads | KV size relative to MHA | Cost |
|---|---|---|---|
| Multi-Head (MHA) | one per query head | 1x | baseline quality, largest cache |
| Multi-Query (MQA) | one shared | 1/n_heads | quality drop, training instability reported |
| Grouped-Query (GQA) | g groups shared by query heads | g/n_heads | near-MHA quality; Llama 3 uses 8 of 64 |
| Multi-head Latent (MLA) | K,V compressed to a low-rank latent | reported ~4-14% of MHA | extra projection at decode time |

GQA (2023) showed you can take an MHA checkpoint, mean-pool its KV heads into groups, and recover quality with about five percent of the original pretraining compute. MLA (DeepSeek-V2, 2024) stores a compressed latent vector per token and reconstructs keys and values from it. The distinction to say aloud: GQA shares heads, MLA compresses what is stored.

### FlashAttention: the same math, less memory traffic

Naive attention materializes the `(T, T)` score matrix in GPU high-bandwidth memory, then reads it back for the softmax, then again to multiply by `V`. At long contexts that matrix is the bottleneck, and the bottleneck is bytes moved, not FLOPs. FlashAttention (2022) tiles `Q`, `K`, `V` into blocks that fit in on-chip SRAM, computes the softmax incrementally ("online softmax", carrying a running max and running sum), and never writes the full matrix. It is exact attention, not an approximation, and it is faster because it is IO-aware. FlashAttention-3 (2024) added Hopper-specific asynchrony and FP8 and reported roughly 75 percent H100 utilization versus roughly 35 percent for the previous version.

The interview point: "why is it faster if it does the same FLOPs?" Because it moves fewer bytes, and attention is bandwidth-bound.

### Mixture of Experts: decouple parameters from compute

In a dense block every token goes through the same MLP. In a mixture-of-experts block the MLP is replaced by `N` expert MLPs and a small router. The router scores each token, picks the top-`k` experts, and only those run. All `N` experts must be in memory; only `k` do work per token.

```text
dense:   token -> [ MLP ]                     params = compute
MoE:     token -> router -> top-2 of 8 -> [E3] [E7] -> weighted sum
         memory: all 8 experts resident       compute: 2 experts per token
```

Mixtral 8x7B (2024) has about 47B total parameters and about 13B active per token (top-2 of 8). DeepSeek-V3 (2024) has 671B total and 37B active, routing to 8 of 256 experts plus shared experts. The headline "47B parameters but runs like 13B" is exactly this: memory like the big model, per-token FLOPs like the small one.

```{code-cell} python
def moe_layer(x, experts, router_W, top_k):
    """x:(T,d). Route each token to top_k experts; only those experts run."""
    logits = x @ router_W                        # (T, n_experts)
    top = np.argsort(-logits, axis=-1)[:, :top_k]
    gates = softmax(np.take_along_axis(logits, top, axis=-1))
    out = np.zeros_like(x); calls = 0
    for t in range(x.shape[0]):
        for j, e in enumerate(top[t]):
            W1, W2 = experts[e]
            out[t] += gates[t, j] * (np.maximum(x[t] @ W1, 0) @ W2); calls += 1
    return out, calls, top

n_experts, top_k, d, d_ff, T = 8, 2, 16, 64, 5
experts = [(rng.normal(0, .1, (d, d_ff)), rng.normal(0, .1, (d_ff, d))) for _ in range(n_experts)]
router_W = rng.normal(size=(d, n_experts))
y, calls, top = moe_layer(rng.normal(size=(T, d)), experts, router_W, top_k)
total = n_experts * 2 * d * d_ff
active = top_k * 2 * d * d_ff
print(f"expert calls: {calls} (= {T} tokens x top-{top_k});  params total {total:,}, active per token {active:,}")
print("routing:", top.tolist())
assert calls == T * top_k and y.shape == (T, d)
assert active / total == top_k / n_experts
# Mixtral-style headline: ~47B total, ~13B active means memory for 47B, compute for 13B.
mixtral_total, mixtral_active = 46.7, 12.9
assert mixtral_active / mixtral_total < 0.3
```

The operational costs of MoE are real: the router must be load-balanced or a few experts saturate while others idle; in multi-GPU serving, tokens must be shuffled to whichever GPU holds their expert (all-to-all communication); and you need enough batch to keep every expert busy, so MoE favors high-throughput serving over single-user latency.

## How it shows up in the stack

**Context budgets are KV cache budgets.** Every token in the agent's transcript is 320 KiB of cache on a 70B model. A 100K-token context is 31 GiB for one user. When a provider prices long-context requests higher or limits concurrency, this is the resource being rationed. It is the mechanical justification for compaction in 34. Context Engineering.

**Prompt caching is KV cache reuse.** The immutability that made the incremental decode demo exact is what lets a server keep the K/V for a shared prefix (system prompt, tool schemas) and attach new requests to it. vLLM's prefix caching and SGLang's RadixAttention are the open implementations; hosted APIs expose the same idea as discounted cached-input tokens. It only works if the prefix is byte-identical, which is why 34. Context Engineering tells you to put stable content first and volatile content last.

**Position matters, and not just for cost.** "Lost in the Middle" (2023) measured a U-shaped curve: models use information at the start and end of a long context better than information in the middle. For a RAG node that means rank your chunks and place the strongest first or last rather than stuffing the window (see 45. Retrieval internals).

**Choosing a model to self-host is choosing a KV budget.** A GQA model with 8 KV heads gives you 8x the concurrent users of an MHA model of the same size at the same context length. An MLA model gives you more still. An MoE model gives you cheaper tokens per second at high batch but needs the memory of its total parameter count.

**Tool result size is not just a token cost.** Each tool result the agent keeps in the transcript is cache the server must hold for the life of the conversation. Trimming a 6K-token SQL result to 400 tokens saves 1.75 GiB of cache per user on a 70B model, per step, for every following step.

```text
agent transcript          KV cache (70B, FP16)
system + tools    4K  ->  1.25 GiB   shared across users if the prefix is stable
history           6K  ->  1.9 GiB    per user
tool results     12K  ->  3.75 GiB   per user, mostly ids and JSON
------------------------------------------------
                 22K  ->  6.9 GiB per active conversation
```

## What goes wrong in production

1. **Capacity planned on weights alone.** Symptom: OOM or request rejection at low concurrency. Cause: KV cache not budgeted. Fix: `memory = weights + kv_per_token x tokens x batch + workspace`; set the server's memory fraction accordingly and cap max context or max concurrent sequences.

2. **Prefix cache never hits.** Symptom: cached-input tokens near zero in billing, TTFT high. Cause: something volatile early in the prompt (a timestamp, a request id, a per-user name) breaks byte-identity. Fix: put all variable content after the stable prefix; check cache hit metrics in the harness.

3. **Long-context regression.** Symptom: retrieval evals pass at top-5, fail at top-30. Cause: relevant chunk buried mid-context, the lost-in-the-middle effect. Fix: rerank and cap the number of chunks; place the best first or last; measure answer quality against chunk position.

4. **Extrapolating past the training length.** Symptom: a self-hosted model produces degraded output on prompts just above its native context even though the server accepted them. Cause: RoPE angles beyond training range. Fix: use a checkpoint with a properly extended context (interpolation or YaRN fine-tuned), and enforce the true limit in your gateway.

5. **MoE at low batch.** Symptom: self-hosted MoE is slower per request than the dense model it replaced. Cause: with few tokens in flight most experts are idle, and their weights are still read from memory. Fix: MoE pays off at high throughput; for a low-QPS internal tool a dense model may be cheaper.

6. **Attention memory blowup on long inputs.** Symptom: OOM during prefill of a long document, not during decode. Cause: a kernel that materializes the `(T, T)` scores. Fix: ensure the serving stack uses FlashAttention-style kernels; chunk very long prefill.

## What changed by 2026

The attention formula did not change. Everything around it did.

Grouped-query attention became the default for new open models because the KV cache, not weights, limits serving concurrency. Latent compression (MLA) followed, first in the DeepSeek line, trading a little decode compute for a much smaller cache. The direction is clear: store less per token so long contexts and many users fit.

Mixture-of-experts moved from research to the frontier. Models with hundreds of billions of total parameters and tens of billions active are now common at the top of leaderboards, and serving them well is a systems problem (routing, expert parallelism, batch sizing) as much as a modeling one.

IO-aware kernels became table stakes. FlashAttention and its successors are what make long contexts practical; every serious serving engine ships them. The lesson for engineers is that "faster" in this field usually means "moved fewer bytes," not "did fewer FLOPs."

Prompt caching reached the hosted APIs. What was an internal serving trick (prefix KV reuse) became a pricing line item, which changed how agent prompts are laid out. Stable prefix first, volatile content last, byte-identical tool schemas across calls.

Context windows grew into the hundreds of thousands of tokens, and the "lost in the middle" result made engineers cautious about using them naively. Current practice is retrieve-rerank-place rather than stuff, and Anthropic's context engineering guidance says as much.

On the installed stack in this course, none of these are things you configure in LangGraph; they are things you design for. The agent's transcript shape (system, tools, history, tool results) is a KV cache layout, and a good harness keeps it stable and small.

## Interview questions

**Q1. Why divide by `sqrt(d_k)`?**

Strong: The dot product of two vectors with unit-variance components has variance equal to the dimension, so at `d_k = 128` the raw scores have a standard deviation around eleven. Softmax of scores that spread that far is nearly one-hot, gradients through it are tiny, and the head stops learning. Dividing by `sqrt(d_k)` normalizes the variance back to one so the softmax stays in its sensitive range. It is a training-stability fix that also keeps attention soft at inference.

Weak: "To normalize the scores."

**Q2. Why cache K and V but not Q?**

Strong: With causal masking, position `i` attends only to positions `<= i`, so once a token's key and value are computed they are used by every later position and never change. Storing them turns each decode step from `O(T)` projections into `O(1)`. The query is different: it is consumed once, by its own position, and no later token needs it. Caching queries would cost memory and save nothing.

Weak: "Q changes every step."

**Q3. Size the KV cache for a model with 32 layers, 8 KV heads, head dimension 128, FP16.**

Strong: Two for K and V, times 32 layers, times 8 heads, times 128 dimensions, times 2 bytes: 131,072 bytes, which is 128 KiB per token. At 16K context that is 2 GiB per sequence, and 64 concurrent users is 128 GiB, so the cache dwarfs an 8B model's 16 GB of weights. FP8 KV halves it; going to MHA with 32 KV heads would quadruple it.

Weak: "A few gigabytes."

**Q4. GQA versus MQA versus MLA in one line each.**

Strong: MQA shares one key-value head across all query heads for the smallest cache but with a quality hit. GQA shares each key-value head across a group of query heads, giving most of the memory savings with near-MHA quality, and Llama 3 uses 8 KV heads for 64 query heads. MLA compresses keys and values into a low-rank latent per token and reconstructs them at attention time, so the cache is a small fraction of MHA's. GQA shares heads; MLA compresses what is stored.

Weak: "They're all ways to make attention cheaper."

**Q5. What does RoPE encode, and why do models still fail past their training length?**

Strong: RoPE rotates each two-dimensional pair of the query and key by an angle proportional to absolute position, with a different frequency per pair. Because the dot product of two rotated vectors depends only on the angle difference, attention scores depend on relative position. Past the training length the low-frequency pairs reach angles the model never saw, and quality drops. Position interpolation, NTK-aware scaling, and YaRN rescale the frequency schedule so a short-context model can be adapted to a long one.

Weak: "RoPE lets the model handle any length."

**Q6. FlashAttention computes the same result. Why is it faster?**

Strong: Attention on GPUs is limited by memory traffic, not arithmetic. Naive kernels write the full `T x T` score matrix to HBM, read it back for softmax, and read it again to multiply by V. FlashAttention tiles Q, K, V into blocks that fit in SRAM, computes softmax incrementally with a running max and sum, and never materializes the matrix. Same FLOPs, far fewer bytes moved, exact result, and linear rather than quadratic memory.

Weak: "It's an approximation that's good enough."

**Q7. "47B parameters but runs like 13B." What does that mean and what is the catch?**

Strong: It is a mixture-of-experts model. The MLP in each block is replaced by eight expert MLPs and a router that picks the top two per token, so only about 13B parameters do work for any token while all 47B must be resident in memory. The catches are that memory cost tracks the total, the router must be load-balanced, multi-GPU serving needs all-to-all communication to move tokens to their experts, and you need enough batch to keep experts busy, so it favors throughput over single-user latency.

Weak: "It's compressed."

## Build it yourself

Extend the incremental decode demo into a small serving-capacity calculator.

1. Generalize `attention` and the cache loop to `n_heads` query heads and `n_kv_heads` KV heads (GQA): query head `h` uses KV head `h // (n_heads // n_kv_heads)`.
2. Verify the GQA cached decode matches full recompute.
3. Write `max_concurrent(gpu_bytes, weight_bytes, context_tokens, kv_per_token, workspace_bytes)` and evaluate it for Llama-3-70B FP16 on 2 x 80 GB at 8K and 32K context, and for the same model with FP8 KV.
4. Add a RoPE function that rotates pairs of dimensions by `m * theta_i` and verify that `q_m . k_n` depends only on `m - n`.

Acceptance checks:

- `assert np.allclose(gqa_cached, gqa_full)` with `n_heads=8, n_kv_heads=2`.
- The GQA cache tensor is exactly `n_heads / n_kv_heads` times smaller than the MHA cache.
- `max_concurrent` at 8K context on 2 x 80 GB with 140 GB of weights and 2 GB workspace is at most 7.
- `max_concurrent` with FP8 KV is roughly double the FP16 figure.
- For RoPE, `assert np.isclose(score(m=5, n=2), score(m=13, n=10))`.

## Key takeaways

- Attention is `softmax(QK^T / sqrt(d_k)) V`; the scaling keeps scores from saturating softmax as dimension grows.
- Causal masking means past keys and values never change, which makes the KV cache exact and makes prefix caching possible.
- KV bytes per token = `2 x layers x kv_heads x head_dim x bytes`; Llama-3-70B FP16 is 320 KiB per token, 80 GiB for 32 users at 8K.
- GQA shares KV heads across query groups; MLA compresses what is stored; both exist because the cache, not the weights, caps concurrency.
- FlashAttention is exact and faster because it moves fewer bytes; attention is bandwidth-bound.
- MoE decouples parameter count from per-token compute: memory for the total, FLOPs for the active experts, at the cost of routing and batching complexity.
- Models use the start and end of long contexts better than the middle; rerank and place, do not stuff.
