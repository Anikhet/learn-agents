# 38. LLM Serving Infrastructure: why your agent is slow and expensive

*Track D (infra + interviews). Book: this is a course addition with no book chapter; it underpins Chapter 16 (Resource-Aware Optimization). Prerequisites: 16. Resource-Aware Optimization, 34. Context Engineering.*

Every lesson so far treated `model.invoke(messages)` as a black box with a price and a latency. This lesson opens the box. Not to make you a kernel engineer, but because the two complaints you will hear most about a production agent are "it is slow" and "it is expensive," and both are explained by four facts about how a transformer is served: prefill is compute-bound, decode is memory-bandwidth-bound, the KV cache is the scarce resource, and batching is how the GPU is paid for.

Hold those facts and prompt caching, tool-result truncation, streaming, speculative decoding, and quantization stop being vendor features and become consequences you could have predicted.

## The problem

The support agent from this course runs in a Slack channel. A customer success manager types:

> "Why is Acme complaining about their invoice?"

The agent makes six model calls: plan, search Slack, look up the CRM, fetch Stripe, reconcile, write the answer. The system prompt plus tool schemas is 6,000 tokens. Each tool result adds 2,000 to 20,000 tokens. The user waits 38 seconds and the run costs $0.41. The team's first instinct is "use a smaller model." That fixes neither problem, because neither problem is the model's intelligence.

Here is what actually happened on the serving side, for one of those six calls:

```text
request arrives ──► queue ──► PREFILL (read 6,000 + 14,000 prompt tokens,
                                       build KV cache, one big matmul pass)
                              ──► first token leaves  ....... TTFT
                              ──► DECODE step 1  (read ALL weights + ALL KV, emit 1 token)
                              ──► DECODE step 2  (read ALL weights + ALL KV, emit 1 token)
                              ──► ...  x 400 output tokens ....... 400 x TPOT
                              ──► done ....... E2E ~ TTFT + TPOT x (n_out - 1)
```

Two different machines are hiding in that timeline.

**Prefill** processes the whole prompt in one pass. It is a large matrix multiply with thousands of tokens in flight, so it saturates the GPU's arithmetic units. It is compute-bound. Twenty thousand prompt tokens cost real FLOPs and real time, every call, unless the KV for the shared prefix was cached from a previous call.

**Decode** produces one token per step. For each step the GPU must read every weight of the model (say 16 GB for an 8B model in FP16) plus the entire KV cache of the sequence, to do a tiny amount of arithmetic. That is memory-bandwidth-bound. An H100 moves about 3.35 TB/s, so a single 8B FP16 sequence cannot decode faster than roughly 200 tokens per second no matter how clever the kernel is, and the arithmetic units sit almost idle.

Which leads to the first rule of serving economics: **decode at batch size 1 wastes the GPU, so the server batches many sequences and reads the weights once per step for all of them.** Your request is sharing the hardware with dozens of strangers, and the scheduler's job is to keep the GPU busy without blowing anyone's latency target.

The metrics that describe this:

| Metric | Meaning | What drives it |
|---|---|---|
| TTFT | time to first token | queue wait + prefill (prompt length, prefix cache hits) |
| TPOT | time per output token, averaged | batch size, weight bytes, KV bytes |
| ITL | inter-token latency, the per-gap distribution | interference from other requests' prefill |
| E2E | end-to-end latency | TTFT + TPOT x (n_out - 1) |
| throughput | tokens/s across the server | batching efficiency |
| goodput | requests/s meeting both the TTFT and TPOT SLO | scheduling policy |

Goodput is the one to remember. Throughput can rise while goodput falls: pack the batch harder and total tokens/s climbs, but every request's TPOT slips past its SLO and the user-visible product gets worse. The DistServe paper (OSDI 2024) made this the headline metric for a reason.

Now reread the agent's 38 seconds. Six calls, each re-prefilling a mostly identical 20,000-token prefix, each queued behind other tenants, each decoding 400 tokens at a TPOT set by how full the batch was. The fixes follow: keep the prefix stable so it is cached, shrink tool results, ask for fewer output tokens, stream so TTFT is what the user perceives, and pick a scheduler that protects your SLO. The rest of this lesson makes each precise.

## Build it without a framework

We cannot run a GPU in a notebook, but we can compute what one would do. Every number below comes from a formula you can verify by hand, which is what a system-design interviewer wants to watch you do.

### Prefill versus decode on a roofline

The roofline model asks how many FLOPs an operation performs per byte moved from memory. Below the hardware's own ratio (peak FLOPs / peak bandwidth) the operation is bandwidth-bound and more compute will not help.

```text
arithmetic intensity  = FLOPs / bytes moved
hardware ridge point  = peak FLOPs / peak bandwidth
H100 (BF16):          ~990 TFLOPS / 3.35 TB/s  ~= 295 FLOPs per byte

decode, batch B:      FLOPs  ~= 2 x params x B         (one matmul pass per token)
                      bytes  ~= params x bytes/param    (weights read once per step)
                      intensity ~= B x 2 / bytes_per_param  =  B  (for FP16, 2 bytes)
```

Decode intensity is roughly the batch size, so below about 295 sequences an H100 decoding an FP16 model is bandwidth-bound. That one line explains why servers batch, why weight quantization speeds up low-batch decode (fewer bytes per step), and why speculative decoding works (idle arithmetic can verify several tokens at once).

```{code-cell} python
# A roofline model of one decode step. Numbers are H100 SXM datasheet figures
# (BF16 dense ~990 TFLOPS, HBM3 3.35 TB/s); the shape of the result is what matters.
PEAK_FLOPS = 990e12
PEAK_BW = 3.35e12
RIDGE = PEAK_FLOPS / PEAK_BW              # FLOPs per byte at which compute takes over


def decode_step(params, bytes_per_param, batch, kv_bytes_per_seq=0):
    """Time for one decode step; returns which resource bounds it."""
    flops = 2 * params * batch                       # one MAC per weight per sequence
    bytes_moved = params * bytes_per_param + batch * kv_bytes_per_seq
    t_compute = flops / PEAK_FLOPS
    t_memory = bytes_moved / PEAK_BW
    bound = "compute" if t_compute > t_memory else "bandwidth"
    return max(t_compute, t_memory), bound, flops / bytes_moved


P8B = 8.0e9
print(f"ridge point: {RIDGE:.0f} FLOPs/byte")
for batch in (1, 8, 64, 256, 512):
    t, bound, intensity = decode_step(P8B, 2, batch)
    tok_per_s = batch / t
    print(f"batch {batch:>4}: step {t*1e3:6.2f} ms  intensity {intensity:6.1f}  "
          f"{bound:>9}-bound  {tok_per_s:9.0f} tok/s total  {1/t:6.0f} tok/s per user")

t1, bound1, _ = decode_step(P8B, 2, 1)
assert bound1 == "bandwidth"
assert abs(1 / t1 - PEAK_BW / (P8B * 2)) < 1        # ~209 tok/s ceiling for 1 user
_, bound512, _ = decode_step(P8B, 2, 512)
assert bound512 == "compute"
# Quantizing weights to 4 bits cuts bytes per step, so single-user decode speeds up:
t4, _, _ = decode_step(P8B, 0.5, 1)
assert t4 < t1 / 3
print(f"INT4 weights, batch 1: {1/t4:.0f} tok/s (vs {1/t1:.0f} FP16)")
```

At batch 1 the GPU does about 1 FLOP per byte and is 99.7% idle on arithmetic. At batch 256 the step takes essentially the same time but serves 256 users. That near-free scaling is the business model of hosted inference. We ignored KV reads; add them and long-context decode stays bandwidth-bound even at large batch, because at 32K tokens one sequence's KV can rival the weights in size.

### How much GPU memory does that take?

The KV cache limits batch size: every token of every active sequence keeps its keys and values for every layer resident in GPU memory.

```text
KV bytes per token = 2 (K and V) x n_layers x n_kv_heads x head_dim x bytes_per_element

Llama-3-8B, FP16:   2 x 32 x 8 x 128 x 2  = 131,072 bytes = 128 KiB per token
Llama-3-70B, FP16:  2 x 80 x 8 x 128 x 2  = 327,680 bytes = 320 KiB per token

GPU memory = weights + KV cache + activations/workspace + ~1-2 GB CUDA context
```

Both models use grouped-query attention with 8 KV heads, so KV per token is 4x (8B) and 8x (70B) smaller than the query-head count suggests. GQA is a serving optimization dressed as an architecture choice.

```{code-cell} python
GiB = 2**30
GB = 1e9


def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem=2):
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem


def plan(name, params, n_layers, n_kv_heads, head_dim, weight_bytes_per_param,
         gpu_gb=80, n_gpus=1, kv_bytes_per_elem=2, overhead_gb=1):
    """Where the memory goes. overhead_gb stands in for CUDA context + workspace per GPU."""
    weights = params * weight_bytes_per_param
    per_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_bytes_per_elem)
    kv_room = gpu_gb * GB * n_gpus - weights - overhead_gb * GB * n_gpus
    tokens = int(kv_room // per_tok) if kv_room > 0 else 0
    print(f"{name:<34} weights {weights/GB:6.1f} GB  KV/token {per_tok/1024:4.0f} KiB  "
          f"KV room {max(kv_room,0)/GB:5.1f} GB  ->  {tokens/1000:6.1f}K resident tokens")
    return weights, per_tok, tokens


# The interview classic: 8B model, FP16, 64 concurrent users at 16K context.
w8, kv8, _ = plan("Llama-3-8B FP16, 1xH100", 8e9, 32, 8, 128, 2)
assert kv8 == 128 * 1024
need = 64 * 16_384 * kv8
print(f"64 users x 16K ctx = {need/GiB:.0f} GiB of KV on top of {w8/GB:.0f} GB weights")
assert need == 128 * GiB                      # does not fit on one 80 GB GPU

# 70B: FP16 needs two H100s; INT4 weight-only (~4.5 bits with group scales) fits one.
_, kv70, tok_fp16 = plan("Llama-3-70B FP16, 2xH100", 70e9, 80, 8, 128, 2, n_gpus=2)
_, _, tok_fp8 = plan("Llama-3-70B FP8,  1xH100", 70e9, 80, 8, 128, 1)
_, _, tok_int4 = plan("Llama-3-70B INT4, 1xH100", 70e9, 80, 8, 128, 0.56)
assert kv70 == 320 * 1024
assert tok_int4 > tok_fp16 > tok_fp8          # FP8 70B on one GPU leaves little KV room
print("FP8 KV cache on the INT4 model doubles resident tokens:")
_, _, tok_int4_fp8kv = plan("Llama-3-70B INT4 + FP8 KV, 1xH100", 70e9, 80, 8, 128, 0.56, kv_bytes_per_elem=1)
assert tok_int4_fp8kv > 1.9 * tok_int4
```

The 8B example is worth memorizing as a spoken answer: "16 gigabytes of weights plus 64 times 16K times 128 KiB, which is 128 GiB of KV cache; it does not fit on one card, and the KV cache, not the weights, is the problem." The 70B example gives the other half: FP16 needs two H100s and leaves on the order of 50-60K tokens of KV; INT4 weight-only fits one card with over 100K tokens; FP8 weights fit one card but leave so little KV room that FP8 70B is in practice a two-GPU deployment. Real servers reserve a further slice (vLLM's `gpu_memory_utilization` defaults to 0.9), so treat these as upper bounds.

### Continuous batching and goodput

Static batching collects a batch, runs it until the *longest* sequence finishes, then admits new requests; short replies are held hostage by long ones. Orca (OSDI 2022) introduced iteration-level scheduling: after every decode step, finished sequences leave and waiting ones join. vLLM, SGLang, and TensorRT-LLM all do this.

```{code-cell} python
import random
from collections import deque

random.seed(7)
STEP_MS = 25.0             # one decode iteration for the batch
SLOTS = 32                 # KV-limited concurrent sequences
TTFT_SLO, TPOT_SLO = 500.0, 60.0


def workload(n, mean_gap_ms=110):
    t, reqs = 0.0, []
    for i in range(n):
        t += random.expovariate(1 / mean_gap_ms)
        reqs.append({"id": i, "arrive": t, "n_out": int(random.lognormvariate(4.5, 0.6)) + 1})
    return reqs


def simulate(reqs, continuous):
    """Discrete-time GPU: each step emits one token for every active sequence."""
    reqs = [dict(r) for r in reqs]
    pending, active, done, now = deque(reqs), [], [], 0.0
    while pending or active:
        can_admit = continuous or not active            # static: only when the batch drains
        while can_admit and pending and len(active) < SLOTS and pending[0]["arrive"] <= now:
            r = pending.popleft()
            r.update(start=now, emitted=0)              # prefill folded into the first step
            active.append(r)
        if not active:
            now = pending[0]["arrive"]
            continue
        now += STEP_MS
        for r in active:
            r["emitted"] += 1
            if r["emitted"] == 1:
                r["first"] = now
        finished = [r for r in active if r["emitted"] >= r["n_out"]]
        for r in finished:
            r["end"] = now
            done.append(r)
        active = [r for r in active if r["emitted"] < r["n_out"]]
    return done, now


def report(done, wall, label):
    ttft = [r["first"] - r["arrive"] for r in done]
    tpot = [(r["end"] - r["first"]) / max(r["n_out"] - 1, 1) for r in done]
    good = sum(1 for a, b in zip(ttft, tpot) if a <= TTFT_SLO and b <= TPOT_SLO)
    total_tok = sum(r["n_out"] for r in done)
    p95 = sorted(ttft)[int(0.95 * len(ttft)) - 1]
    print(f"{label:<11} p95 TTFT {p95:7.0f} ms  throughput {total_tok/wall*1000:6.0f} tok/s  "
          f"goodput {good/wall*1000:5.2f} req/s  ({good}/{len(done)} in SLO)")
    return good / wall


reqs = workload(600)
g_static = report(*simulate(reqs, continuous=False), "static")
g_cont = report(*simulate(reqs, continuous=True), "continuous")
assert g_cont > g_static * 2
print("iteration-level scheduling admits work every step, so short requests stop waiting on long ones")
```

The simulator is crude (constant step time, prefill folded into the first step), but the effect is real: same hardware, same traffic, several times the goodput, because the batch is never partially idle.

The crude part hides a real problem. When a sequence joins, its prefill runs in the same step as everyone else's decode; a 20,000-token prefill can turn a 25 ms step into 300 ms, and every other user sees a stall. That is the ITL spike. **Chunked prefill** bounds it by interleaving prefill chunks with decode steps. **Disaggregated serving** (DistServe, OSDI 2024) removes it by running prefill and decode on separate GPU pools and shipping KV across NVLink or RDMA, so the two phases scale and tune independently. NVIDIA Dynamo, Mooncake, and modes in vLLM and SGLang implement this. The cost is KV transfer and orchestration; the win is that TTFT and TPOT stop fighting over the same hardware.

### PagedAttention and prefix caching

If KV memory is the bottleneck, wasting it is the cardinal sin. Before vLLM, engines reserved a contiguous KV region per sequence sized for the *maximum* length; 60 to 80 percent of it sat unused. The PagedAttention paper (SOSP 2023) borrowed virtual memory: KV lives in fixed-size blocks (16 tokens in vLLM), a per-sequence block table maps logical to physical blocks, and blocks are allocated on demand. Waste fell below 4 percent and throughput rose 2 to 4x purely from fitting more sequences.

Blocks make **prefix caching** nearly free. Two requests that share the same first 6,000 tokens share their first 375 blocks. Hash each block by its prefix content and the second request points its block table at blocks already computed, skipping that prefill. vLLM calls this automatic prefix caching; SGLang's RadixAttention keeps the same idea in a radix tree with LRU eviction. Hosted APIs expose it as "prompt caching" with discounted cached-input pricing.

```{code-cell} python
import hashlib

BLOCK = 16


class PagedKV:
    """Block-granular KV allocation with content-hashed prefix sharing."""

    def __init__(self, n_blocks):
        self.free = n_blocks
        self.by_hash = {}        # prefix hash -> physical block id (shared, refcounted)
        self.refs = {}
        self.next_id = 0

    def allocate(self, tokens):
        table, prefix_hash, reused = [], "", 0
        for i in range(0, len(tokens), BLOCK):
            chunk = tuple(tokens[i:i + BLOCK])
            full = len(chunk) == BLOCK
            prefix_hash = hashlib.sha1((prefix_hash + repr(chunk)).encode()).hexdigest()
            if full and prefix_hash in self.by_hash:          # identical prefix: share the block
                block = self.by_hash[prefix_hash]
                self.refs[block] += 1
                reused += 1
            else:
                if self.free == 0:
                    raise MemoryError("KV cache full")         # in vLLM: preempt or queue
                block, self.next_id, self.free = self.next_id, self.next_id + 1, self.free - 1
                self.refs[block] = 1
                if full:
                    self.by_hash[prefix_hash] = block
            table.append(block)
        return table, reused


def contiguous_reserved(n_requests, max_len):
    return n_requests * max_len                                 # tokens reserved, pre-vLLM style


system_prompt = list(range(6000))                               # token ids of a stable prefix
requests = [system_prompt + [100_000 + r * 1000 + k for k in range(random.randint(300, 2500))]
            for r in range(40)]

kv = PagedKV(n_blocks=40_000)
allocated = reused_total = 0
for toks in requests:
    table, reused = kv.allocate(toks)
    allocated += len(table)
    reused_total += reused
used_tokens = (allocated - reused_total) * BLOCK
reserved = contiguous_reserved(len(requests), max_len=32_768)
print(f"paged+prefix-cached blocks in use: {allocated - reused_total:,} "
      f"({used_tokens/reserved:.1%} of what contiguous max-length reservation would hold)")
print(f"prefill skipped via prefix reuse: {reused_total * BLOCK:,} tokens across 40 requests")
assert reused_total >= 39 * (6000 // BLOCK)                     # every request after the first hits
assert used_tokens < 0.2 * reserved

# Change the FIRST token of the prompt and every cached block misses:
_, reused_after_edit = kv.allocate([-1] + system_prompt[1:] + [1, 2, 3])
assert reused_after_edit == 0
print("editing the start of the prompt invalidates the whole prefix; keep volatile content at the END")
```

That last assertion is the operational lesson. A timestamp, a user name, or a randomly ordered tool list at the *top* of your prompt defeats caching for every call. The MCP specification's current revision asks servers to return tools in deterministic order for exactly this reason.

### Speculative decoding

Decode leaves arithmetic idle. Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) spends that slack: a small drafter proposes gamma tokens, the target verifies them all in one pass, and rejection sampling keeps the output distribution identical to the target's. With per-token acceptance probability alpha, expected tokens per target step is:

```text
E[tokens per step] = (1 - alpha^(gamma+1)) / (1 - alpha)

alpha = 0.8, gamma = 4:  (1 - 0.8^5) / 0.2  = 3.36 tokens per step   (~3x fewer steps)
alpha = 0.5, gamma = 4:  (1 - 0.5^5) / 0.5  = 1.94
```

Medusa adds drafting heads to the target; EAGLE drafts at the feature level, with later versions reporting several-x speedups. All share the failure condition the roofline predicts: the free lunch is idle compute at low batch. At high batch the target is compute-saturated, verification costs gamma times more, and the speedup collapses. It also collapses at low alpha, which happens on unusual domains and on structured tool-call JSON the drafter was not trained on.

```{code-cell} python
def expected_tokens_per_step(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, batch, verify_cost_per_token=1.0):
    """Wall-clock speedup vs plain decode, on the roofline model from cell 1.

    Below the ridge a verification pass over gamma+1 tokens costs about the same as
    one token (bandwidth-bound); above it, cost scales with tokens verified.
    """
    base_t, _, _ = decode_step(P8B, 2, batch)
    verify_t, _, _ = decode_step(P8B, 2, batch * (gamma + 1))
    draft_t = 0.05 * base_t * gamma                              # a ~5% sized drafter
    return expected_tokens_per_step(alpha, gamma) * base_t / (verify_t + draft_t)


assert abs(expected_tokens_per_step(0.8, 4) - 3.3616) < 1e-3
for batch in (1, 16, 64, 128, 512):
    row = "  ".join(f"a={a}: {speedup(a, 4, batch):4.2f}x" for a in (0.5, 0.8, 0.95))
    print(f"batch {batch:>4}   {row}")
assert speedup(0.8, 4, 1) > 2.0
assert speedup(0.8, 4, 512) < 1.0
print("speculation pays at low batch and high acceptance; it costs you at compute saturation")
```

### Quantization choices

Quantization is a memory and bandwidth decision before it is a quality decision:

| Scheme | What is quantized | Helps | Hurts / caveat |
|---|---|---|---|
| W4A16 (GPTQ, AWQ, GGUF) | weights only, 4-bit | memory footprint; low-batch decode (fewer bytes/step) | must dequantize before matmul, so no help at high batch; quality drop on hard tasks |
| W8A8 INT8 (SmoothQuant) | weights and activations | prefill and high-batch decode on INT8 tensor cores | activation outliers make this hard; SmoothQuant migrates the difficulty into weights |
| FP8 (Hopper and later) | weights and activations | prefill and high-batch throughput with small quality loss | needs H100-class hardware; still 1 byte/param, so 70B does not fit one GPU comfortably |
| FP8 KV cache | the cache itself | halves KV memory, so ~2x resident tokens | slight quality impact on very long contexts |

After quantizing, re-run your *task* evals, not just perplexity. A model can lose almost nothing in perplexity and still emit malformed tool-call arguments.

## How it shows up in the stack

You will probably never configure a GPU server for the support agent. You will call a hosted API, or a platform team will run vLLM for you. The serving facts still reach you through every line of your agent's cost and latency.

**Prompt caching is prefix caching.** Providers charge a fraction of the input price for tokens that hit their cache, and hit only when the prefix is byte-identical. Structure every prompt as stable-to-volatile: system prompt, then tool schemas in fixed order, then reference material, then the transcript, then the newest turn. Never put a date, a request id, or a shuffled tool list before the stable part. 34. Context Engineering and Appendix A treat this as a prompt-design rule; now you know it is a block-table hash.

**Tool results are prefill.** A 20,000-token Stripe dump appended to the transcript is 20,000 tokens of compute-bound prefill on *every subsequent call* in the loop, plus the KV memory to hold it. The tool-design guidance to paginate, truncate, and offer concise versus detailed response formats is serving economics. Claude Code's cap on tool results at 25,000 tokens is a serving budget expressed as a product rule.

**Output tokens are decode steps.** Each is a full pass over the weights. Asking for a terse structured answer instead of prose, or setting `max_tokens` honestly, is the single cheapest latency win. A 400-token answer at 50 ms TPOT is 20 seconds; a 60-token JSON answer is 3.

**Streaming does not make decode faster; it makes TTFT the perceived latency.** Stream the final answer to the user and stream intermediate "searching Slack" status so the 38 seconds feel like activity rather than silence. Appendix B covers the UX; the reason it works is the timeline diagram at the top of this lesson.

**Engine positioning**, as a synthesis rather than a benchmark claim. vLLM is the breadth choice: PagedAttention, automatic prefix caching, the widest model and hardware support, an OpenAI-compatible server. SGLang leads with RadixAttention for heavy prefix sharing and strong constrained decoding for JSON, which matters for tool calls. TensorRT-LLM is NVIDIA-only, compiles models into engines, gets FP8 and FP4 first, and trades iteration speed for peak performance. For an agent with a long shared system prompt and many tool calls, benchmark prefix-cache behavior and structured decoding first.

**The LLM gateway** is where agent teams meet all of this. It sits between agents and providers and owns per-tenant token buckets, retries with backoff and jitter on transient errors only, fallback chains with health-based routing, exact caches (semantic caches are risky outside FAQ-shaped traffic), SSE passthrough so streaming survives the hop, TTFT and TPOT monitoring per route, partial JSON parsing for streamed tool calls, cost attribution per agent run, PII redaction, and guardrails. One trap is specific to agents: providers differ in tool-call dialects and `tool_call_id` validation, so a mid-run failover can break the transcript. Route at the start of a run, not in the middle.

```text
agents ──► gateway ──► provider A (primary)
              │   ├──► provider B (fallback, same tool schema dialect)
              │   └──► self-hosted vLLM (cheap tier for classification/routing)
              │
              ├─ rate limits per tenant      ├─ TTFT/TPOT/goodput per route
              ├─ retry only transient        ├─ cost per agent run
              ├─ exact cache (idempotent)    └─ prefix stability linting
```

Here is the measurement that turns all of this into evidence. It is the only cell in this lesson that needs a real model.

```{code-cell} python live
import os
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
stable_prefix = SystemMessage("You are the support agent for a SaaS billing team. " * 200)


def measure(messages):
    started = time.perf_counter()
    first = None
    n = 0
    for chunk in model.stream(messages):
        if first is None and chunk.content:
            first = time.perf_counter()
        n += 1
    end = time.perf_counter()
    ttft = (first or end) - started
    tpot = (end - (first or end)) / max(n - 1, 1)
    return ttft, tpot, n


for attempt in (1, 2):                      # the second call should hit the provider's prompt cache
    ttft, tpot, n = measure([stable_prefix, HumanMessage(f"Reply with one short sentence. Attempt {attempt}.")])
    print(f"call {attempt}: TTFT {ttft*1000:6.0f} ms  TPOT {tpot*1000:5.1f} ms  chunks {n}")
```

## What goes wrong in production

1. **TTFT spikes with no change in your code.** Symptom: p95 TTFT triples during business hours. Cause: queueing at the provider or on your shared vLLM pool; other tenants' long prefills interfere with your decode. Fix: measure TTFT and TPOT separately and alert on each; ask your platform for chunked prefill or disaggregated serving; move latency-sensitive routing calls to a dedicated small-model tier.

2. **Prompt cache hit rate is zero.** Symptom: input costs are the full price although the system prompt never changes. Cause: something volatile precedes the stable prefix: a timestamp, a per-request id, tool schemas in dict-iteration order, or a gateway injecting a header. Fix: lint prompt construction so stable content comes first and tools are sorted; verify with the provider's cached-token counters.

3. **Cost per run grows quadratically with steps.** Symptom: step 12 of a run costs six times step 2. Cause: every tool result stays in the transcript, so each call re-prefills everything before it. Fix: compaction and offloading of large tool results to files or a store (34. Context Engineering); tool-side truncation and pagination (05. Tool Use). Serving explains why the fix works: fewer prefill tokens and fewer KV blocks.

4. **Self-hosted server OOMs or preempts under load.** Symptom: requests fail with out-of-memory or latency collapses at a concurrency the weights alone said was fine. Cause: KV cache, not weights, is the limit; a 32K-context user takes the memory of eight 4K users. Fix: do the memory arithmetic from this lesson per model and context; set the max model length honestly; consider FP8 KV; add admission control so the scheduler queues rather than thrashes.

5. **Quantized model passes benchmarks and fails tool calls.** Symptom: perplexity is within 1 percent of the FP16 model, but 3 percent of tool calls have malformed JSON arguments. Cause: quantization error concentrates in rare, precise tokens; perplexity averages it away. Fix: gate quantization on task evals with tool-call validity and trajectory metrics (37. Trajectory Evals), and prefer constrained decoding for structured output.

6. **Throughput up, users angry.** Symptom: the platform team reports 30 percent more tokens per second after a tuning change; the support team reports the agent "feels slower." Cause: larger batches raised TPOT past the SLO; throughput rose, goodput fell. Fix: make goodput the SLO metric and give the scheduler a TPOT target, not a utilization target.

## What changed by 2026

- **Prompt caching became the default economic model for agents.** Hosted providers discount cached input tokens, and the MCP specification's current revision asks servers to return tool lists in deterministic order and marks listings as cacheable with TTLs, explicitly to improve prompt-cache hits. Prefix stability is now a protocol-level concern, not a trick.
- **Disaggregated prefill and decode moved from paper to product.** The DistServe idea ships in NVIDIA Dynamo, Mooncake, and modes of vLLM and SGLang. Current docs describe KV transfer over NVLink or RDMA as the main cost. For agent teams this means TTFT and TPOT can be tuned as separate SLOs.
- **Reasoning models changed the token mix.** A reasoning model may spend thousands of hidden tokens before its first visible token. That moves cost into decode and lengthens TTFT for the visible stream. OpenAI's reasoning guidance treats reasoning effort as a tuning knob; budget it per step rather than globally, and do not ask reasoning models to "think step by step" in the prompt.
- **Structured decoding became a serving feature.** Constrained decoding for JSON schemas is now built into engines like SGLang and vLLM, which makes tool-call validity a property of the server rather than a prayer. 24. Structured Output covers the agent side.
- **Quantization moved to FP8 and FP4 on new hardware.** Current docs describe FP8 as the default high-throughput format on Hopper-class and later GPUs, with FP4 arriving first in TensorRT-LLM. Re-run task evals after every precision change.

## Interview questions

**Q1. Why is LLM decode memory-bandwidth-bound, and what follows from that?**

Strong: Each decode step generates one token per sequence but has to read every weight of the model, plus the sequence's KV cache, out of HBM. For an 8B FP16 model that is 16 GB of reads to do about 16 GFLOPs of work, roughly one FLOP per byte, while an H100 can do about 295 FLOPs per byte. So at low batch the arithmetic units sit idle and the step time is set by bandwidth: about 16 GB divided by 3.35 TB/s, or 5 ms, a ceiling of about 200 tokens per second for one user. Three things follow: batching is nearly free until you approach the ridge point, weight quantization speeds up low-batch decode because it cuts bytes per step, and speculative decoding works because the spare compute can verify several tokens at once.

Weak: "Decode is slow because it is sequential."

**Q2. Explain TTFT, TPOT, and goodput, and how batching trades them off.**

Strong: TTFT is queue wait plus prefill, so it is driven by prompt length and prefix-cache hits. TPOT is the average gap between output tokens, driven by batch size and weight and KV bytes. End-to-end latency is roughly TTFT plus TPOT times output tokens. Bigger batches raise total throughput but also raise every request's TPOT, and admitting long prefills into a decode batch causes inter-token stalls. Goodput is requests per second that meet both SLOs; it is the right objective because throughput can rise while goodput falls. Chunked prefill bounds the stalls; disaggregating prefill and decode onto separate pools removes the interference entirely at the cost of KV transfer, which is why it only makes sense at a scale where interference is the bottleneck.

Weak: "Batching improves throughput."

**Q3. What problem does PagedAttention solve?**

Strong: KV cache memory is the limit on concurrency, and pre-vLLM engines allocated a contiguous KV region per sequence sized for the maximum length, so most of it was reserved but unused, with 60 to 80 percent waste reported. PagedAttention stores KV in fixed-size blocks with a per-sequence block table, like virtual memory pages, allocating on demand. Waste dropped under 4 percent and throughput rose 2 to 4x from fitting more sequences. The block structure also enables prefix caching by content hash and copy-on-write for parallel sampling.

Weak: "It makes attention faster."

**Q4. Estimate the GPU memory for an 8B model serving 64 users at 16K context in FP16.**

Strong: Weights are 8 billion parameters times 2 bytes, 16 GB. KV per token is 2 for K and V, times 32 layers, times 8 KV heads with grouped-query attention, times head dim 128, times 2 bytes, which is 128 KiB. Sixty-four users times 16K tokens times 128 KiB is 128 GiB. Total is about 144 GB before activations and CUDA context, so it does not fit on one 80 GB card; the cache dominates, not the weights. Options are two GPUs, FP8 KV to halve the cache, a shorter max context, or admitting fewer concurrent users.

Weak: "16 GB, it fits."

**Q5. When does speculative decoding stop helping?**

Strong: It helps because decode is bandwidth-bound and a verification pass over gamma plus one tokens costs about the same as a pass over one at low batch. Expected tokens per step is (1 minus alpha to the gamma plus one) over (1 minus alpha), so with 80 percent acceptance and four drafts you get about 3.4 tokens per step. It stops helping when the batch is large enough that the target is compute-bound, because then verifying five tokens costs five times as much, or when acceptance drops because the drafter does not predict the target well, for example on unfamiliar domains or structured tool-call output. Enable it per tier and measure acceptance on real traffic.

Weak: "It always gives 2 to 3x."

**Q6. INT4 AWQ or FP8 for a high-QPS deployment on H100s?**

Strong: FP8. Weight-only INT4 shrinks memory and speeds up low-batch decode because it cuts weight bytes per step, but the weights are dequantized to run the matmul, so at high batch where you are compute-bound it gives nothing back. FP8 quantizes weights and activations and runs on the Hopper FP8 tensor cores, so it speeds up prefill and high-batch decode. If memory were the constraint, for example fitting a 70B model on one GPU with lots of KV room, INT4 would win. Either way I would gate the change on task evals including tool-call validity, not perplexity.

Weak: "INT4 because it is smaller."

**Q7. Why does a long system prompt get cheaper over time, and how do you keep it that way?**

Strong: Engines cache KV blocks by prefix content hash, so a second request that shares the first N tokens skips that prefill and reuses the blocks; providers pass that on as discounted cached-input pricing. The catch is that the match is byte-exact from the start, so anything volatile placed before the stable content, a timestamp, a request id, tools in nondeterministic order, invalidates every block after it. Structure prompts stable-to-volatile, sort tool schemas, and verify with the provider's cached-token counters. The MCP spec's current revision asks servers to return tools in deterministic order for this reason.

Weak: "The provider caches the prompt."

## Build it yourself

Extend the batching simulator into a scheduler that models the real trade-offs:

1. Give each request a prompt length, and make a step's time `STEP_MS + prefill_ms_per_token x tokens prefilled this step`. Show the ITL spike a 20,000-token arrival causes for everyone else.
2. Add chunked prefill with a configurable chunk size and show the spike bounded.
3. Add a KV budget in blocks, using `PagedKV` from this lesson, and make admission depend on free blocks. Model requests that share the 6,000-token prefix and requests that do not.
4. Report TTFT p95, TPOT p95, throughput, and goodput for four configurations: static, continuous, continuous plus chunked prefill, continuous plus chunked prefill plus prefix caching.

Acceptance checks:

- With chunked prefill disabled, the maximum inter-token gap for a decoding request exceeds `5 x STEP_MS` when a 20,000-token request arrives; with a 2,048-token chunk it stays under `2 x STEP_MS`.
- Prefix caching admits at least 30 percent more concurrent sequences than no caching under the same block budget when 80 percent of requests share the prefix.
- Goodput is non-decreasing across the four configurations for the same workload and seed.
- An `assert` pins the KV math: 64 x 16K x 128 KiB == 128 GiB.

## Key takeaways

- Prefill is compute-bound and proportional to prompt length; decode is bandwidth-bound and proportional to weight bytes plus KV bytes per step.
- The KV cache, not the weights, is what limits concurrency; compute it as 2 x layers x kv_heads x head_dim x bytes per token before sizing anything.
- Continuous batching, chunked prefill, and disaggregated serving are all scheduling answers to one question: how to keep the GPU full without breaking TTFT and TPOT SLOs. Judge them by goodput.
- PagedAttention makes KV allocation block-granular; prefix caching is those blocks shared by content hash, and it only works if your prompt is stable from the first byte.
- Speculative decoding and weight-only quantization spend idle compute and save bytes at low batch; both stop paying at compute saturation. Re-run task evals after any precision change.
- For an agent engineer, the levers are prompt structure, tool-result size, output length, streaming, and a gateway that routes at run boundaries and measures TTFT, TPOT, and cost per run.
