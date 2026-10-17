# 45. Retrieval internals: BM25, hybrid fusion, rerankers, HNSW and IVF-PQ

*Track C (retrieval + production AI). Course addition; goes under the hood of the book's Chapter 14 (Knowledge Retrieval). Prerequisites: 14. Knowledge Retrieval (RAG), 25. Embeddings, Clustering, and Theme Discovery. Companion to 29. Advanced Production RAG.*

Lesson 14 built retrieval as "embed, cosine, top-k" over a numpy matrix, which is enough for a few hundred thousand chunks. This lesson opens the four boxes production retrieval runs on: the lexical scorer (BM25), the fusion of lexical and dense results (RRF), the reranker (cross-encoders), and the approximate nearest-neighbor indexes that make dense search fast at scale (HNSW, IVF-PQ). Each is built in stdlib or numpy, then measured with the metrics interviewers expect on a whiteboard: recall@k, MRR, and nDCG.

## The problem

The support/ops agent has a `search_kb` tool over 40 million chunks from Notion runbooks, Slack threads, and past tickets. Three tickets arrive in the same hour.

```text
ticket A  "Getting E-4012 on every webhook since this morning"
          dense search returns: generic webhook retry docs          ✗
          the runbook that says "E-4012 = rotate the signing secret" is rank 14

ticket B  "Our client is unable to authenticate incoming notifications"
          keyword search returns: "Slack alerts for billing notifications"   ✗
          the right doc says "validate event payloads", no shared keywords

ticket C  "Acme was charged twice"   (tenant filter: acme's workspace only)
          vector DB returns 3 results for k=10, none relevant       ✗
```

Each failure lives in a different layer.

- **A is a dense-retrieval failure.** An embedding model has no dedicated dimension for the string `E-4012`. It encodes "some error code about webhooks," and every webhook doc looks similar.
- **B is a lexical failure.** "Authenticate notifications" and "validate event payloads" mean the same thing and share no tokens.
- **C is an index failure.** The approximate index found its 10 nearest neighbors over *all* tenants, then the filter threw away the 7 that were not Acme's.

The naive fixes make things worse: raising k to 100 floods the model's context (and lost-in-the-middle hides the right chunk), and switching to brute force over 40 million 768-dimensional vectors is about 123 GB of float32 scanned per query.

The production answer is a pipeline, where each stage narrows the candidate set and spends more compute per candidate:

```text
query
  │
  ├──► BM25 (inverted index)        top 100    ~ms, exact on identifiers
  │
  ├──► dense ANN (HNSW / IVF-PQ)    top 100    ~ms, semantic, approximate
  │
  ▼
RRF fusion                          top 50     microseconds, rank-based
  │
  ▼
cross-encoder reranker              top 5      tens to hundreds of ms, joint scoring
  │
  ▼
model context (5 chunks, strongest first)
```

Every box is a place where recall can be lost, and a lost document is never recovered by a later stage. That sentence is the whole design philosophy: early stages are cheap and recall-oriented, late stages are expensive and precision-oriented.

## Build it without a framework

The corpus is a small slice of the agent's knowledge base. Small enough to read, structured to reproduce tickets A, B, and C.

### BM25: TF-IDF that saturates and normalizes

BM25 (Robertson and colleagues, the Okapi BM25 family from the 1990s) scores a document against a query as a sum over query terms. For each term, it multiplies how rare the term is in the corpus (IDF) by how much the term appears in this document (a saturating TF).

```text
score(D, Q) = Σ_{t ∈ Q}  IDF(t) · f(t,D)·(k1 + 1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )

IDF(t)      = ln( 1 + (N − n_t + 0.5) / (n_t + 0.5) )        (Lucene's form, never negative)

f(t,D)  term count in D      |D|    length of D in tokens      avgdl  mean document length
N       number of documents  n_t    documents containing t

k1 ≈ 1.2–2.0   term-frequency saturation.  As f → ∞ the TF factor → k1 + 1.
b  ≈ 0.75      length normalization.       b = 0: ignore length. b = 1: full normalization.

Worked example: N = 1000, n_t = 10, f = 3, |D| = 150, avgdl = 100, k1 = 1.2, b = 0.75
  IDF   = ln(1 + 990.5 / 10.5)            = ln(95.33)  = 4.557
  norm  = 1.2 · (1 − 0.75 + 0.75 · 1.5)   = 1.2 · 1.375 = 1.65
  TF    = 3 · 2.2 / (3 + 1.65)            = 6.6 / 4.65 = 1.419
  score = 4.557 · 1.419                   = 6.47
```

The intuition for the two knobs is what interviewers want. **k1** answers "is the tenth mention worth as much as the first?" No: the first mention says the doc is about the term, the tenth adds little. With k1 = 1.2 the TF factor can never exceed 2.2, however many times a spammy page repeats "webhook." **b** answers "should a long document win because it has more words?" No: a 5,000-word guide mentions everything once. b scales the saturation point by relative length, so a term in a short, focused doc counts for more.

```{code-cell} python
import math
import re
from collections import Counter

DOCS = {
    "kb-webhook-sig": "Error E-4012: webhook signature verification failed. Rotate the signing secret "
                      "in the Stripe dashboard and redeploy the endpoint.",
    "kb-webhook-retry": "Stripe retries failed webhook deliveries with exponential backoff for up to three days.",
    "kb-payload-check": "If a customer cannot validate incoming event payloads, check the signing secret "
                        "and the clock skew tolerance on their server.",
    "kb-refund-policy": "Refunds for annual plans are prorated. Enterprise customers can request a refund within 60 days.",
    "kb-sso": "SAML single sign-on setup requires the Enterprise plan and an identity provider metadata URL.",
    "kb-duplicate-charge": "Duplicate invoice charges usually come from a retried payment intent without an idempotency key.",
    "kb-slack-alerts": "Configure Slack alerts for failed payments on the billing notifications page.",
    "kb-api-keys": "Rotate API keys every 90 days. Revoked keys return error E-4011 unauthorized.",
    "kb-webhook-guide": "Webhooks guide. A webhook is an HTTP callback. Each webhook event has a type, a payload and a "
                        "timestamp. Register the webhook endpoint, choose webhook event types, test webhook delivery "
                        "from the dashboard, monitor webhook logs, and read the changelog for new event types, "
                        "pagination, versioning and limits.",
}

STOP = set("a an the and or to for in of on with if is are can up their from without every each has this that was".split())


def tokenize(text):
    # Keep hyphenated identifiers like e-4012 as one token: that is BM25's superpower.
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.ids, self.k1, self.b = list(docs), k1, b
        self.tf = [Counter(tokenize(docs[i])) for i in self.ids]    # the "inverted index", flattened
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.dl) / len(self.dl)
        self.N = len(self.ids)
        self.df = Counter(t for c in self.tf for t in c)

    def idf(self, t):
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query, j):
        s = 0.0
        for t in tokenize(query):
            f = self.tf[j].get(t, 0)
            if f:
                norm = self.k1 * (1 - self.b + self.b * self.dl[j] / self.avgdl)
                s += self.idf(t) * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query, k=10):
        scored = [(self.score(query, j), self.ids[j]) for j in range(self.N)]
        return [d for s, d in sorted(scored, key=lambda x: (-x[0], x[1])) if s > 0][:k]


def bm25_term(f, dl, avgdl, N, n_t, k1=1.2, b=0.75):
    idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
    return idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))


bm25 = BM25(DOCS)
print("ticket A:", bm25.search("E-4012 webhook error")[:3])
print("ticket B:", bm25.search("client unable to authenticate incoming notifications")[:3])
print("ticket C:", bm25.search("customer was charged twice")[:3])

# The worked example from the formula box.
assert abs(bm25_term(f=3, dl=150, avgdl=100, N=1000, n_t=10) - 6.47) < 0.01
# k1: ten mentions are worth far less than ten times one mention, and never more than (k1 + 1) x IDF.
one, ten = bm25_term(1, 100, 100, 1000, 10), bm25_term(10, 100, 100, 1000, 10)
assert ten / one < 2.0 and ten < 2.2 * math.log(1 + 990.5 / 10.5)
# b: the same single mention scores lower in a document 5x the average length.
assert bm25_term(1, 500, 100, 1000, 10) < bm25_term(1, 100, 100, 1000, 10)
# Identifiers: the exact error code wins ticket A. Vocabulary mismatch loses ticket B and C.
assert bm25.search("E-4012 webhook error")[0] == "kb-webhook-sig"
assert bm25.search("client unable to authenticate incoming notifications")[0] == "kb-slack-alerts"
assert "kb-duplicate-charge" not in bm25.search("customer was charged twice")
```

BM25 nails ticket A because `e-4012` appears in exactly one document, so its IDF is the largest in the corpus. It fails B (it matched "notifications" in the Slack-alerts doc) and C ("charged" is not "charges", "twice" is not "duplicate"). No amount of k1/b tuning fixes vocabulary mismatch. That is the job of a dense retriever.

### Dense retrieval, and fusing the two with RRF

A dense retriever (a bi-encoder, see 25. Embeddings) maps query and document separately to vectors trained so that paraphrases land close together. Documents are embedded offline once; a query costs one encoder pass plus a nearest-neighbor search.

We cannot load a real embedding model offline, so the cell below uses a deliberately tiny stand-in: a hand-written concept map where "authenticate", "validate", and "signature" share a dimension, and every error code collapses into a single `<id>` dimension. That second choice is the honest part of the simulation. A real encoder also blurs rare identifiers, because it has never learned a meaning for `E-4012` specifically. The `live` cell later in this section swaps in a real embedder.

**Fusion.** We now have two ranked lists with incompatible scores: BM25 scores are unbounded (6.47 above), cosine is in [-1, 1]. Adding them lets BM25 dominate. **Reciprocal Rank Fusion** (Cormack, Clarke, and Buettcher, SIGIR 2009) ignores scores and uses only ranks:

```text
RRF(d) = Σ_{lists i}  1 / (k + rank_i(d))        k = 60, rank starts at 1

doc X: rank 1 in BM25, absent from dense        1/61                = 0.01639
doc Y: rank 2 in BM25, rank 2 in dense          1/62 + 1/62         = 0.03226   ← wins
```

The k = 60 flattens the curve so that agreement between retrievers matters more than a single first place. No normalization, no training, no tuning per corpus. That is why it is the default fusion in most hybrid-search products. The alternative, a weighted sum of min-max-normalized scores, can beat RRF when tuned on labeled data, and is fragile when score distributions shift.

```{code-cell} python
import numpy as np

# Toy dense encoder: a concept vocabulary instead of a learned one (see prose).
CONCEPTS = {
    **dict.fromkeys(["webhook", "webhooks", "event", "events", "notifications", "callback",
                     "deliveries", "delivery"], "webhook"),
    **dict.fromkeys(["verify", "verification", "validate", "authenticate", "signature", "signing"], "verify"),
    **dict.fromkeys(["error", "failed", "unable", "cannot", "fails"], "error"),
    **dict.fromkeys(["charged", "charges", "charge", "payment", "payments"], "charge"),
    **dict.fromkeys(["twice", "duplicate", "retried", "double"], "duplicate"),
    **dict.fromkeys(["invoice", "billing"], "invoice"),
    **dict.fromkeys(["refund", "refunds"], "refund"),
    **dict.fromkeys(["customer", "customers", "client"], "customer"),
    **dict.fromkeys(["secret", "keys", "key", "rotate"], "secret"),
    **dict.fromkeys(["payload", "payloads", "incoming"], "payload"),
}
DIMS = sorted(set(CONCEPTS.values())) + ["<id>", "<other>"]


def embed(text):
    v = np.zeros(len(DIMS))
    for t in tokenize(text):
        if t in CONCEPTS:
            v[DIMS.index(CONCEPTS[t])] += 1.0
        elif re.fullmatch(r"[a-z]+-\d+", t):
            v[DIMS.index("<id>")] += 1.0          # every error code looks the same to the encoder
        else:
            v[DIMS.index("<other>")] += 0.3
    return v / np.linalg.norm(v)                  # L2-normalize: dot product == cosine


DOC_IDS = list(DOCS)
DOC_MATRIX = np.stack([embed(DOCS[d]) for d in DOC_IDS])     # embedded once, offline


def dense_search(query, k=10):
    sims = DOC_MATRIX @ embed(query)
    order = sorted(range(len(DOC_IDS)), key=lambda j: (-sims[j], DOC_IDS[j]))
    return [DOC_IDS[j] for j in order[:k]]


def rrf(rankings, k=60, top=10):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))][:top]


def hybrid_search(query, k=10):
    return rrf([bm25.search(query, 100), dense_search(query, 100)], top=k)


for q in ["E-4012 webhook error", "client unable to authenticate incoming notifications",
          "customer was charged twice"]:
    print(f"{q[:40]:40s} bm25={bm25.search(q)[:1]} dense={dense_search(q)[:1]} hybrid={hybrid_search(q)[:2]}")

# Dense fixes vocabulary mismatch (B, C) and fails the identifier (A).
assert dense_search("client unable to authenticate incoming notifications")[0] == "kb-payload-check"
assert dense_search("customer was charged twice")[0] == "kb-duplicate-charge"
assert dense_search("E-4012 webhook error")[0] != "kb-webhook-sig"
# Hybrid keeps BM25's identifier win and dense's paraphrase win.
assert hybrid_search("E-4012 webhook error")[0] == "kb-webhook-sig"
assert hybrid_search("client unable to authenticate incoming notifications")[0] == "kb-payload-check"
# The RRF arithmetic from the formula box: agreement beats a single first place.
fused = rrf([["X", "Y"], ["Z", "Y"]])
assert fused[0] == "Y" and abs(sum(1 / (60 + r) for r in (2, 2)) - 0.03226) < 1e-4
```

Look at ticket C in the output. Hybrid puts `kb-duplicate-charge` in the top 2 but not first, because BM25 put a wrong doc first and dense put that same wrong doc fourth, and RRF rewards agreement. Fusion raises the floor; it does not pick the best document. That is the reranker's job.

### Rerankers: joint scoring on a shortlist

A bi-encoder never sees query and document together. It compresses each into one vector independently, so it cannot notice that "charged twice" is answered by the sentence about retried payment intents rather than the one about refund windows. A **cross-encoder** feeds `[query] [SEP] [passage]` through a transformer and outputs one relevance score, with full attention between every query token and every passage token.

```text
bi-encoder       q ──enc──► q⃗      d ──enc──► d⃗ (offline)     score = q⃗ · d⃗
cross-encoder    [q ; d] ──enc──► score                        one forward pass PER candidate
late interaction q ──enc──► q⃗₁..q⃗ₙ   d ──enc──► d⃗₁..d⃗ₘ (offline)
  (ColBERT)      score = Σ_i max_j q⃗ᵢ · d⃗ⱼ   ("MaxSim": per-token vectors, stored per document)

Cost, computed: 100 candidates × (~40 query + ~260 passage tokens)
              = 100 forward passes, ~30,000 tokens of encoder compute per query.
Embedding the query for dense search: 1 pass, ~40 tokens.
```

That cost is why cross-encoders only ever see the top 50 to 200 fused candidates, never the corpus. ColBERT-style late interaction sits in between: token-level matching quality, with document vectors precomputed, at the price of storing a vector per token. LLM rerankers (the model reads the shortlist and orders it) are strongest and most expensive. For the support agent, reranking 50 candidates down to 5 is usually the single biggest quality gain per millisecond, because it fixes both the "right doc at rank 7" problem and lost-in-the-middle.

### HNSW: greedy search on a navigable graph

Brute-force dense search is one matrix-vector product per query: exact, and O(N·d). At 40 million vectors it is too slow and too memory-hungry. **HNSW** (Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", 2016) replaces the scan with a walk.

```text
layer 2   ●──────────────────────●                 few nodes, long links  (entry point)
          │                      │
layer 1   ●────●─────────●───────●────●            more nodes
          │    │         │       │    │
layer 0   ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●         every vector, M·2 short links each

search: at each upper layer, greedily hop to the neighbor closest to q, then drop down.
        at layer 0, run a best-first beam search keeping the efSearch closest seen so far.
        stop when the closest unexpanded candidate is farther than the worst of the beam.
```

Three knobs, and which one costs memory:

| Knob | Controls | Up means | Memory? |
|---|---|---|---|
| `M` | links per node | higher recall, better on hard data, slower build | **yes**, roughly linear |
| `efConstruction` | beam width while inserting | better graph, slower build | no |
| `efSearch` | beam width while querying (must be ≥ k) | higher recall, higher latency | no |

The memory arithmetic, computed: a 768-dimensional float32 vector is 4 × 768 = 3,072 bytes. Layer 0 stores about 2M neighbor ids of 4 bytes, so M = 16 adds roughly 8 × 16 = 128 bytes. That is about 3.2 KB per vector, and **100 million vectors need about 320 GB of RAM** before any metadata. That number is why IVF-PQ exists.

The cell builds a two-layer graph from exact k-nearest-neighbor lists (real HNSW inserts incrementally and prunes links with a diversity heuristic; the search is the same), then sweeps `efSearch`.

```{code-cell} python
import heapq

rng = np.random.default_rng(7)
N, D, M = 3000, 24, 12
centers = rng.normal(size=(30, D))
X = (centers[rng.integers(0, 30, N)] + rng.normal(size=(N, D))).astype(np.float32)
queries = (centers[rng.integers(0, 30, 50)] + rng.normal(size=(50, D))).astype(np.float32)


def knn_graph(ids, m):
    V = X[ids]
    sq = (V ** 2).sum(1)
    dist2 = sq[:, None] + sq[None, :] - 2 * V @ V.T
    np.fill_diagonal(dist2, np.inf)
    nearest = np.argsort(dist2, axis=1)[:, :m]
    graph = {int(ids[i]): {int(ids[j]) for j in nearest[i]} for i in range(len(ids))}
    for u in list(graph):                     # make links bidirectional, as HNSW does
        for v in list(graph[u]):
            graph[v].add(u)
    return graph


layer0 = knn_graph(np.arange(N), M)
upper_ids = rng.choice(N, size=N // 30, replace=False)     # ~3% of nodes get promoted
layer1 = knn_graph(upper_ids, M)
entry_point = int(upper_ids[0])
distance_calls = [0]


def dist(q, i):
    distance_calls[0] += 1
    diff = X[i] - q
    return float(diff @ diff)


def beam_search(graph, q, entry, ef):
    d0 = dist(q, entry)
    visited, candidates, beam = {entry}, [(d0, entry)], [(-d0, entry)]
    while candidates:
        d_c, c = heapq.heappop(candidates)                 # closest unexpanded node
        if len(beam) >= ef and d_c > -beam[0][0]:
            break                                          # nothing left can improve the beam
        for nb in graph[c]:
            if nb in visited:
                continue
            visited.add(nb)
            d_nb = dist(q, nb)
            if len(beam) < ef or d_nb < -beam[0][0]:
                heapq.heappush(candidates, (d_nb, nb))
                heapq.heappush(beam, (-d_nb, nb))
                if len(beam) > ef:
                    heapq.heappop(beam)                    # drop the worst
    return [i for _, i in sorted((-d, i) for d, i in beam)]


def ann_search(q, k=10, ef=10):
    entry = beam_search(layer1, q, entry_point, ef=1)[0]   # greedy descent on the upper layer
    return beam_search(layer0, q, entry, ef=max(ef, k))[:k]


def exact_search(q, k=10, candidates=None):
    ids = np.arange(N) if candidates is None else np.asarray(candidates)
    return [int(i) for i in ids[np.argsort(((X[ids] - q) ** 2).sum(1))[:k]]]


def recall_at_k(found, truth):
    return len(set(found) & set(truth)) / len(truth)


sweep = {}
for ef in [10, 20, 50, 100]:
    distance_calls[0] = 0
    recalls = [recall_at_k(ann_search(q, 10, ef), exact_search(q, 10)) for q in queries]
    sweep[ef] = (float(np.mean(recalls)), distance_calls[0] / len(queries))
    print(f"efSearch={ef:3d}  recall@10={sweep[ef][0]:.3f}  distances/query={sweep[ef][1]:6.0f}  (brute force: {N})")

links = sum(len(v) for v in layer0.values())
print(f"layer-0 links: {links:,} ({links * 4 / N:.0f} bytes/vector of ids) vs {D * 4} bytes/vector of floats")

assert sweep[10][0] < sweep[100][0]                      # efSearch buys recall...
assert sweep[10][1] < sweep[100][1]                      # ...with distance computations
assert sweep[100][0] >= 0.99 and sweep[100][1] < N / 2   # near-exact at a fraction of brute force

# Filtering: only 10% of vectors belong to tenant "acme" (ticket C).
tenant = np.where(rng.random(N) < 0.10, "acme", "other")
acme_ids = np.where(tenant == "acme")[0]
q = queries[0]
post_filtered = [i for i in ann_search(q, 10, ef=50) if tenant[i] == "acme"]
over_fetched = [i for i in ann_search(q, 100, ef=200) if tenant[i] == "acme"][:10]
pre_filtered = exact_search(q, 10, candidates=acme_ids)  # brute force over acme's slice only
print(f"post-filter: {len(post_filtered)} of 10 | over-fetch x10: {len(over_fetched)} | "
      f"pre-filter brute force: {len(pre_filtered)} using {len(acme_ids)} distances")
assert len(post_filtered) < 10                           # ticket C, reproduced
assert all(tenant[i] == "acme" for i in pre_filtered) and len(pre_filtered) == 10
assert len(acme_ids) < sweep[50][1] * 2                  # a selective filter makes brute force cheap
```

The sweep is the recall-latency curve you tune in production: efSearch goes up until recall@10 against brute force hits the target (often 0.95 to 0.99), and no higher. The filtering lines reproduce ticket C. Post-filtering returns fewer than k because the graph found the nearest vectors among *all* tenants. Over-fetching helps but guesses a multiplier. Pre-filtering to the tenant's slice and brute-forcing it is exact and, for a selective filter, cheaper than the graph walk. Engines now integrate the predicate into traversal (single-stage filtered search); ask your vendor what it does when a filter keeps 1% of the corpus.

### IVF and PQ: fewer candidates, smaller vectors

**IVF** (inverted file) runs k-means over the vectors to get `nlist` centroids and stores each vector in its nearest centroid's list. A query compares against the centroids, then scans only the `nprobe` closest lists. A common starting point is `nlist` on the order of √N. Vectors near a cell boundary are missed when their cell is not probed.

**PQ** (product quantization, Jégou, Douze, and Schmid, 2011) compresses each vector. Split the d dimensions into m subvectors, run k-means with 256 centroids in each subspace, and store each subvector as a one-byte centroid id.

```text
768-d float32                         = 768 × 4   = 3,072 bytes
PQ, m = 64 subvectors × 1 byte        =              64 bytes        48x smaller   (computed)
100M vectors: 307 GB raw  →  6.4 GB of codes (+ centroids and list overhead)

search (ADC): per query, build an m × 256 table of distances from each query subvector to
each centroid; a vector's approximate distance is m table lookups summed. No decompression.
refine: re-score the top few hundred candidates with exact vectors (from disk or RAM).
```

```{code-cell} python
def kmeans(V, k, iters=10, seed=0):
    r = np.random.default_rng(seed)
    C = V[r.choice(len(V), k, replace=False)].copy()
    for _ in range(iters):
        assign = ((V[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            members = V[assign == j]
            if len(members):
                C[j] = members.mean(0)
    return C, ((V[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)


rng = np.random.default_rng(11)
N2, D2 = 4000, 32
c2 = rng.normal(size=(40, D2))
Y = (c2[rng.integers(0, 40, N2)] + 0.6 * rng.normal(size=(N2, D2))).astype(np.float32)
qs = (c2[rng.integers(0, 40, 40)] + 0.6 * rng.normal(size=(40, D2))).astype(np.float32)

NLIST, PQ_M, KS = 64, 8, 256
coarse, cell = kmeans(Y, NLIST)                                 # IVF: train the coarse quantizer
inverted_lists = {j: np.where(cell == j)[0] for j in range(NLIST)}
SUB = D2 // PQ_M
codebooks, codes = [], np.zeros((N2, PQ_M), dtype=np.uint8)
for s in range(PQ_M):                                           # PQ: one codebook per subspace
    C, a = kmeans(Y[:, s * SUB:(s + 1) * SUB], KS, iters=8, seed=s)
    codebooks.append(C)
    codes[:, s] = a


def ivfpq_search(q, k=10, nprobe=4, refine=0):
    probe = np.argsort(((coarse - q) ** 2).sum(1))[:nprobe]
    cand = np.concatenate([inverted_lists[j] for j in probe])
    table = np.stack([((codebooks[s] - q[s * SUB:(s + 1) * SUB]) ** 2).sum(1) for s in range(PQ_M)])
    approx = table[np.arange(PQ_M)[None, :], codes[cand]].sum(1)   # ADC: m lookups per vector
    ranked = cand[np.argsort(approx)]
    if refine:
        top = ranked[:refine]
        ranked = top[np.argsort(((Y[top] - q) ** 2).sum(1))]       # exact re-score of a shortlist
    return ranked[:k], len(cand)


def truth(q, k=10):
    return np.argsort(((Y - q) ** 2).sum(1))[:k]


results = {}
for nprobe in [1, 4, 16]:
    for refine in [0, 50]:
        r = [recall_at_k(ivfpq_search(q, 10, nprobe, refine)[0], truth(q)) for q in qs]
        scanned = np.mean([ivfpq_search(q, 10, nprobe, refine)[1] for q in qs])
        results[(nprobe, refine)] = float(np.mean(r))
        print(f"nprobe={nprobe:2d} refine={refine:2d}  recall@10={np.mean(r):.3f}  codes scanned={scanned:6.0f} of {N2}")

raw_bytes, code_bytes = D2 * 4, PQ_M
print(f"bytes/vector: {raw_bytes} raw -> {code_bytes} PQ ({raw_bytes // code_bytes}x); "
      f"768-d at m=64: {768 * 4 // 64}x")

assert raw_bytes // code_bytes == 16 and 768 * 4 // 64 == 48
assert results[(4, 0)] < 0.8                     # PQ distances alone are lossy
assert results[(4, 50)] >= 0.95                  # refine with exact vectors recovers recall
assert results[(1, 50)] < results[(4, 50)]       # one probe misses neighbors across cell boundaries
```

The two failure modes in the output are the two that matter in production. PQ codes alone reorder near-ties, so the top 10 by approximate distance is often wrong; re-scoring a shortlist of 50 with exact vectors fixes it. And `nprobe = 1` misses true neighbors that live in the adjacent cell.

### Measuring retrieval: recall@k, MRR, nDCG

Retrieval quality is measured against labeled queries, separately from answer quality (see 22. Evaluation Reliability).

```text
recall@k   = |relevant ∩ top-k| / |relevant|         did we retrieve it at all? (feeds the reranker)
MRR        = mean over queries of 1 / rank_first_relevant     how high is the first hit?
DCG@k      = Σ_{i=1..k} (2^rel_i − 1) / log2(i + 1)           graded relevance, discounted by position
nDCG@k     = DCG@k / IDCG@k                                   IDCG = DCG of the ideal ordering

Worked example: ranked relevances [3, 0, 2], ideal [3, 2, 0]
  DCG  = 7/1 + 0/1.585 + 3/2     = 8.5
  IDCG = 7/1 + 3/1.585 + 0/2     = 8.893
  nDCG@3 = 0.956
```

```{code-cell} python
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked, qrels, k):
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg([qrels.get(d, 0) for d in ranked[:k]]) / ideal


def mrr(ranked, qrels):
    return next((1 / i for i, d in enumerate(ranked, 1) if qrels.get(d, 0) > 0), 0.0)


def recall(ranked, qrels, k):
    relevant = {d for d, r in qrels.items() if r > 0}
    return len(relevant & set(ranked[:k])) / len(relevant)


# Graded labels: 3 = answers the ticket, 2 = directly useful, 1 = related background.
EVAL_SET = {
    "E-4012 webhook error": {"kb-webhook-sig": 3, "kb-payload-check": 1, "kb-webhook-guide": 1},
    "client unable to authenticate incoming notifications": {"kb-payload-check": 3, "kb-webhook-sig": 2},
    "customer was charged twice": {"kb-duplicate-charge": 3},
    "E-4011": {"kb-api-keys": 3},
}
SYSTEMS = {"bm25": bm25.search, "dense": dense_search, "hybrid": hybrid_search}

report = {}
for name, search in SYSTEMS.items():
    per_query = [ndcg_at_k(search(q), rel, 3) for q, rel in EVAL_SET.items()]
    report[name] = {"ndcg@3": float(np.mean(per_query)), "worst": min(per_query),
                    "mrr": float(np.mean([mrr(search(q), rel) for q, rel in EVAL_SET.items()])),
                    "recall@5": float(np.mean([recall(search(q), rel, 5) for q, rel in EVAL_SET.items()]))}
    print(f"{name:7s} " + "  ".join(f"{m}={v:.3f}" for m, v in report[name].items()))

# The worked example.
assert abs(ndcg_at_k(["a", "b", "c"], {"a": 3, "c": 2}, 3) - 0.956) < 1e-3
assert mrr(["x", "y", "a"], {"a": 1}) == 1 / 3
# Hybrid is best on average and, more importantly, has no catastrophic query.
assert report["hybrid"]["ndcg@3"] >= max(report["bm25"]["ndcg@3"], report["dense"]["ndcg@3"])
assert report["hybrid"]["worst"] > max(report["bm25"]["worst"], report["dense"]["worst"])
assert report["bm25"]["worst"] == 0.0            # ticket C: BM25 retrieved nothing relevant
```

The `worst` column is the one to show in a design review. Averages hide the query class where a retriever returns nothing. BM25's worst query scores zero (ticket C); dense's worst is the identifier query; hybrid's worst is better than either. Four queries is a toy; a real eval set has hundreds, stratified by query type (identifiers, paraphrases, multi-hop, tenant-filtered), because the fixes differ by class.

## How it shows up in the stack

For an agent engineer, retrieval internals surface in five decisions.

**1. The retrieval tool's contract.** The agent calls `search_kb(query, k)`. Inside it, the pipeline is BM25 plus dense plus RRF plus rerank; outside, the model sees 5 chunks with ids. The tool result size is a context-budget decision (34. Context Engineering): 5 reranked chunks beat 20 fused ones, both in tokens and in lost-in-the-middle risk. Order the strongest chunk first.

**2. Where the filter goes.** Permissions and tenancy (ticket C) are enforced in the retrieval layer, never by the model. Metadata filters at query time, a re-check at result time against the source system's ACL, and over-fetch to compensate for late rejections. Hard isolation (per-tenant namespace or index) when the contract requires it. Never return counts that leak the existence of documents a user cannot see.

**3. Index choice by scale and memory.**

| Corpus | Index | Why |
|---|---|---|
| < ~1M vectors | flat (brute force) | exact, trivial, a matrix multiply; fine on CPU |
| 1M to ~100M, RAM available | HNSW | best recall/latency; memory ~ (4d + 8M) bytes/vector |
| 100M+, or RAM-bound | IVF-PQ (+ refine), or disk-based graphs | 30-50x compression; needs training; lower recall |
| heavy filtering | engine with filter-aware traversal, or per-tenant indexes | post-filter returns < k |

**4. Freshness.** HNSW handles inserts but deletes are tombstones that degrade the graph until a rebuild. IVF centroids are trained on a snapshot and drift as the corpus changes, so retrain periodically. Store the embedding model version with every vector and never mix two models in one index.

**5. Where the LangGraph node sits.** Retrieval is either a fixed node before the model or a tool the model can re-query (agentic RAG, see 14 and 29). Either way it is a function returning ids and text, and no prompt work recovers a document it never retrieved.

```{code-cell} python live
import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

# Swap the toy concept encoder for a real embedding model and re-run the eval set.
embedder = init_embeddings(os.environ.get("COURSE_EMBEDDINGS", "openai:text-embedding-3-small"))
real_matrix = np.array(embedder.embed_documents([DOCS[d] for d in DOC_IDS]))
real_matrix /= np.linalg.norm(real_matrix, axis=1, keepdims=True)


def real_dense_search(query, k=10):
    qv = np.array(embedder.embed_query(query))
    sims = real_matrix @ (qv / np.linalg.norm(qv))
    return [DOC_IDS[j] for j in np.argsort(-sims)[:k]]


def real_hybrid(query, k=10):
    return rrf([bm25.search(query, 100), real_dense_search(query, 100)], top=k)


for name, search in {"real dense": real_dense_search, "real hybrid": real_hybrid}.items():
    scores = [ndcg_at_k(search(q), rel, 3) for q, rel in EVAL_SET.items()]
    print(f"{name:11s} ndcg@3={np.mean(scores):.3f} worst={min(scores):.3f}")

# An LLM as a listwise reranker over the fused shortlist: strongest and most expensive stage.
model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
shortlist = real_hybrid("customer was charged twice", k=5)
listing = "\n".join(f"[{d}] {DOCS[d]}" for d in shortlist)
print(model.invoke(f"Order these by relevance to 'customer was charged twice'. Reply with ids only.\n{listing}").content)
```

## What goes wrong in production

1. **Identifiers vanish in dense-only search.** Symptom: queries with error codes, SKUs, invoice ids, or function names return generic docs. Cause: the encoder blurs rare strings. Fix: hybrid with BM25; also route queries that look like identifiers to exact-match lookup first.

2. **Score-sum fusion dominated by one retriever.** Symptom: "hybrid" behaves exactly like BM25. Cause: unbounded BM25 scores added to cosine in [-1, 1]. Fix: RRF, or normalize and tune weights on a labeled set, and monitor the per-retriever contribution.

3. **Filtered search returns fewer than k, or nothing.** Symptom: ticket C. Cause: post-filtering an ANN result. Fix: filter-aware search, pre-filter plus brute force for selective filters, or per-tenant indexes. Measure what fraction of the corpus survives typical filters.

4. **Recall silently drops as the corpus grows.** Symptom: quality regression with no code change. Cause: efSearch or nprobe tuned at 1M vectors, now serving 20M; tombstoned deletes degrading the HNSW graph; IVF centroids trained on last year's data. Fix: a scheduled job that measures ANN recall@k against brute force on a sample of real queries, and alerts.

5. **Reranker latency blows the budget.** Symptom: p99 of the tool call jumps from 150 ms to 2 s. Cause: reranking 500 candidates, or long passages. Fix: cap candidates (50-100), truncate passages, batch, and set a timeout with a fallback to the fused order.

6. **Mixed embedding models in one index.** Symptom: a subset of documents is never retrieved after a model upgrade. Cause: vectors from two models are not comparable. Fix: store the model version per vector, re-embed fully, and switch indexes atomically.

7. **Good recall, wrong answers.** Symptom: recall@10 is 0.95 and users still get wrong answers. Cause: the right chunk is at rank 8 in a 10-chunk prompt, or a stale version is ranked above it. Fix: rerank to fewer chunks, strongest first; dedupe versions; measure generation faithfulness separately.

## What changed by 2026

- **Hybrid became the default, not an optimization.** Major vector databases and search engines ship BM25 (or learned sparse such as SPLADE) alongside dense search with built-in fusion, and RRF is the common default. Pure dense retrieval is now the thing you justify.
- **Contextual enrichment of chunks.** Anthropic's contextual retrieval write-up reports top-20 retrieval failure falling from 5.7% to 3.7% with contextual embeddings, 2.9% adding contextual BM25, and 1.9% adding reranking. Each stage in this lesson's pipeline shows up as a measured step in that result.
- **Reranking is standard, and LLM rerankers are affordable at small k.** Cross-encoder reranking of the fused top 50 to 100 is now routine; LLM listwise reranking is used where latency allows.
- **Filter-aware ANN.** Current vendor docs describe filtering integrated into the index traversal (for example Weaviate's ACORN-based filtering and Qdrant's payload-aware links) instead of pre- or post-filtering; behavior on very selective filters is vendor-specific, so measure it.
- **Memory pressure moved indexes off RAM.** Quantization (PQ, scalar, binary), disk-resident graph indexes, and GPU index builds are common answers to the 320 GB-per-100M-vectors problem. Treat specific speedup claims as vendor benchmarks until you reproduce them.
- **Retrieval as an agent tool.** Anthropic's context-engineering guidance favors just-in-time retrieval (load context when needed), so an agent calls retrieval many times per task and the tool's precision compounds across steps.

## Interview questions

**Q1. How do you choose between a sparse retriever (BM25) and a dense retriever? When do you need both?** *(Microsoft, Perplexity, Glean)*

Strong: They fail on opposite query classes. BM25 is exact on rare tokens, identifiers, and names, needs no training, and is cheap, but misses paraphrase because it only matches tokens. Dense retrieval handles paraphrase and vocabulary mismatch but blurs identifiers it never learned and drifts out of domain. Enterprise support traffic has both classes, error codes and natural-language complaints, so I run both and fuse with RRF, which needs no score normalization. I justify it with nDCG and worst-query-class metrics on a labeled set stratified by query type, not with an average.

Weak: "Dense is better because it understands meaning."

**Q2. Explain BM25's k1 and b.**

Strong: k1 controls term-frequency saturation: the TF factor is f(k1+1)/(f + k1·norm), so it approaches k1+1 as f grows, and the tenth mention of a word adds little over the first. b controls length normalization: it scales that saturation point by document length relative to the average, so b = 0 ignores length and b = 1 fully normalizes; 0.75 is the usual default. IDF then weights each term by rarity, which is why an error code that appears in one document dominates the score. I would tune them only with a labeled eval set, and usually the defaults are fine.

Weak: "They're hyperparameters you tune."

**Q3. What is a reranker, when should you use one, and what does a cross-encoder cost you?** *(Cohere, Microsoft, Perplexity)*

Strong: A reranker re-scores a shortlist with a more expensive model. A cross-encoder reads query and passage together with full attention, so it catches relevance that independent embeddings miss, but it needs one forward pass per candidate and nothing can be precomputed. So it runs on the top 50 to 100 fused candidates, never the corpus: at 100 candidates of 300 tokens that is about 30,000 tokens of encoder compute per query, tens to hundreds of milliseconds. I use it when the right document is in the candidate set but not at the top, which recall@50 versus nDCG@5 tells me, and I cap candidates and set a timeout with a fallback to the fused order.

Weak: "A reranker makes results better."

**Q4. Explain HNSW's M, efConstruction, and efSearch. Which affects memory?**

Strong: HNSW is a layered proximity graph: search descends greedily through sparse upper layers to find an entry point, then runs a best-first beam search at the bottom layer. M is the number of links per node; it raises recall and is the only one that costs memory, about 8M bytes per vector on top of the raw vector. efConstruction is the beam width during insertion, trading build time for graph quality. efSearch is the beam width at query time, at least k, and it is the knob I tune online for the recall-latency trade-off. Memory is about 3.2 KB per 768-d float32 vector with M = 16, so 100M vectors is about 320 GB, which is when I look at quantization or IVF-PQ.

Weak: "Higher values mean better accuracy."

**Q5. Compare HNSW, IVF-PQ, and flat indexes. How do you pick?**

Strong: Flat is exact brute force, a matrix multiply, and is the right answer under roughly a million vectors or as the ground truth for measuring the others. HNSW gives the best recall at low latency but keeps full vectors and links in RAM and handles deletes poorly. IVF-PQ partitions with k-means and compresses each vector to m bytes, 48x smaller for 768-d at m = 64, which is how you fit billions on limited memory, at the cost of training, lower raw recall, and boundary misses; you recover recall by probing more lists and re-scoring a shortlist with exact vectors. I pick by corpus size, RAM budget, update rate, and filter selectivity, and I measure recall@k against flat on real queries.

Weak: "HNSW is the fastest, so use HNSW."

**Q6. A filtered vector search with k = 10 returns 3 results. What happened?** *(Microsoft, Glean, Palantir)*

Strong: Almost certainly post-filtering: the ANN index returned the 10 nearest vectors over the whole corpus and the tenant or permission filter discarded 7. Fixes depend on selectivity. For a very selective filter, pre-filter and brute-force the small slice, which is exact and cheap. For moderate filters, use an engine with filter-aware traversal, or over-fetch as a stopgap. For hard tenant isolation, use per-tenant namespaces. And the filter is enforced in the retrieval layer with a re-check against the source ACL, never by the model.

Weak: "Increase k."

**Q7. How would you evaluate retrieval, separately from generation?** *(Cohere)*

Strong: I build a labeled query set, ideally with graded relevance, stratified by query type: identifiers, paraphrases, tenant-filtered, multi-hop. Recall@k at the candidate-generation depth tells me whether the right document reaches the reranker at all; nDCG@k and MRR at the context depth tell me whether it is near the top where the model will use it. I report the worst query class, not just the mean, and I separately track ANN recall against brute force so I can tell an index regression from a model regression. Generation faithfulness and citation accuracy are a separate eval, because a correct retrieval can still produce a wrong answer.

Weak: "Look at whether the answers are good."

## Build it yourself

Add a reranking stage and a filter-aware search to this lesson's pipeline.

1. Write `rerank(query, candidates)` that scores each (query, doc) pair jointly, for example concept overlap plus a bonus when an identifier in the query appears verbatim in the doc. It must see both texts together, like a cross-encoder.
2. Build `pipeline(query, k=3) = rerank(query, hybrid_search(query, 50))[:k]` and evaluate it with `ndcg_at_k` on `EVAL_SET`.
3. Implement `filtered_ann_search(q, k, allowed)` that traverses the full HNSW-style graph but only admits allowed nodes into the result beam.
4. Add a `retrieval_budget` that counts documents scored at each stage.

Acceptance checks:

- `pipeline("customer was charged twice")[0] == "kb-duplicate-charge"` (the reranker fixes what RRF could not).
- Mean nDCG@3 of `pipeline` is at least that of `hybrid_search`, and its worst query is no lower.
- `filtered_ann_search(queries[0], 10, acme_ids)` returns 10 acme vectors with recall@10 ≥ 0.9 against `exact_search(..., candidates=acme_ids)`.
- The budget shows the reranker scored at most 50 documents per query while BM25 and dense each scored the full corpus.

## Key takeaways

- Production retrieval is a funnel: cheap, recall-oriented stages first, expensive precision stages last, and nothing lost early is recovered later.
- BM25 is IDF times a saturating, length-normalized TF; k1 sets saturation, b sets length normalization, and it is unbeatable on identifiers.
- Dense and sparse retrievers fail on opposite query classes, and RRF fuses them by rank with no score normalization.
- Cross-encoders score query and passage jointly and cost one forward pass per candidate, so they only rerank a shortlist.
- HNSW trades memory for speed; M is the knob that costs RAM, efSearch is the knob you tune online.
- IVF-PQ compresses vectors to a few bytes and probes a few cells; re-scoring a shortlist with exact vectors restores recall.
- Evaluate retrieval with recall@k, MRR, and nDCG per query class, and watch the worst class, not the mean.
