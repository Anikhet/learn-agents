# 29. Advanced Production RAG

*Track C (retrieval + production AI). Course addition; extends the book's Chapter 14 (Knowledge Retrieval). Prerequisites: 14. Knowledge Retrieval (RAG), 25. Embeddings, Clustering, and Themes.*

14 built RAG with nothing hidden: chunk, embed, cosine top-k, cite, verify. It works on the demo questions. Then it meets production traffic and fails in ways that have nothing to do with the model: an error code nobody's embedding understands, a paraphrase no keyword matches, the right chunk buried at rank 8, a contractor reading the compensation guide, last quarter's policy outranking this quarter's. This lesson is the production retrieval stack, built one measured fix at a time, and the metrics that tell you whether each fix helped.

## The problem

The support agent's knowledge base is Notion policies, runbooks, incident reports, CRM notes, and a few HR pages that should never reach a support answer. Here is a week of real questions against the 14 pipeline:

| Question | What the dense-only retriever did | Root cause |
|---|---|---|
| "customer reports E-4012" | ranked a refund policy first | embeddings blur exact identifiers |
| "can a large customer get their money back after six weeks?" | fine; keyword search found nothing | vocabulary mismatch (the other direction) |
| "enterprise refund window" | the right chunk says "The window is 60 days" with no mention of Enterprise | chunk lost its context when it was split |
| same question, next week | returned the superseded 45-day text | stale index, two versions live |
| a contractor asks about refund bonuses | returned the HR compensation page | no permission filtering |
| "what refund window applies to Acme?" | no single chunk answers it | multi-hop: plan in CRM, policy in Notion |

None of these is fixed by a bigger model. They are fixed by the pipeline:

```text
question
   │
   ▼
rewrite ─────────── multi-query, HyDE, decomposition (optional, agentic)
   │
   ▼
┌─────────────── candidate generation (cheap, high recall) ───────────────┐
│  ACL + freshness PRE-filter  (only chunks this user may see, latest)    │
│         │                                   │                           │
│     BM25 (sparse)                     dense (ANN)                       │
│         └──────────── RRF fusion ──────────┘                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼  top 20-200
                         rerank (cross-encoder, expensive, precise)
                               ▼
                         ACL RE-CHECK at result time
                               ▼  top 3-5
                         generate with citations → verify citations (14)
```

Each box answers one row of the table. The rest of this lesson builds them, in numpy and stdlib, and scores each with recall@k, MRR, and nDCG so "better" is a number, not a feeling.

## Build it without a framework

### The corpus, with the metadata that production needs

Every chunk carries more than text: which document and section it came from, who may read it, and which version it is. The chunk that says "The window is 60 days" is the realistic hard case. Structure-aware splitting (14) put it under the "Enterprise plans" heading, but the word "Enterprise" is not in its text.

Contextual retrieval addresses exactly this. Anthropic's write-up on it prepends a short, model-written context (roughly 50 to 100 tokens) to each chunk before indexing it, both for embeddings and for BM25. It reports that top-20 retrieval failures fell from 5.7% to 3.7% with contextual embeddings, to 2.9% when contextual BM25 was added, and to 1.9% with reranking on top. We use a deterministic prefix built from the title and section; in production a model writes it once per chunk at index time.

The first cell is the course's scripted model helper; the model appears only in the LangGraph section.

```{code-cell} python
import hashlib
import math
import re
from collections import Counter

import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(GenericFakeChatModel):
    """Replays scripted turns so the control flow, not the model, is under study."""

    def bind_tools(self, tools, **kwargs):
        return self  # a real model would attach tool schemas here


def scripted(*turns):
    """Each turn is either final text or a list of (tool_name, args) tool calls."""
    messages = []
    for turn in turns:
        if isinstance(turn, str):
            messages.append(AIMessage(content=turn))
        else:
            messages.append(AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
                for i, (name, args) in enumerate(turn)]))
    return ScriptedChatModel(messages=iter(messages))


CHUNKS = [
    {"id": "refunds#standard", "doc": "policies/refunds.md", "section": "Standard plans",
     "text": "Refunds are available for 30 days after purchase.", "acl": {"everyone"}, "version": 3},
    {"id": "refunds#enterprise@v3", "doc": "policies/refunds.md", "section": "Enterprise plans",
     "text": "The window is 60 days, and the request must come from an account admin.",
     "acl": {"everyone"}, "version": 3},
    {"id": "refunds#enterprise@v2", "doc": "policies/refunds.md", "section": "Enterprise plans",
     "text": "The window is 45 days for annual contracts.", "acl": {"everyone"}, "version": 2},
    {"id": "sso#errors", "doc": "runbooks/sso.md", "section": "Error codes",
     "text": "E-4012 means the SAML assertion expired. Ask the admin to resync the IdP clock.",
     "acl": {"everyone"}, "version": 1},
    {"id": "sso#setup", "doc": "runbooks/sso.md", "section": "Setup",
     "text": "Upload the identity provider metadata XML under Settings, then enable enforcement.",
     "acl": {"everyone"}, "version": 1},
    {"id": "billing#seats", "doc": "policies/billing.md", "section": "Seat changes",
     "text": "Seat reductions are prorated and credited on the next invoice.",
     "acl": {"everyone"}, "version": 1},
    {"id": "incident#2291", "doc": "incidents/2291.md", "section": "Summary",
     "text": "Login failures for EU tenants were caused by an expired signing certificate.",
     "acl": {"support", "eng"}, "version": 1},
    {"id": "crm#acme", "doc": "crm/ACME-88", "section": "Account notes",
     "text": "Acme Corp upgraded from Standard to Enterprise on 2026-08-02. Champion: Priya.",
     "acl": {"support"}, "version": 1},
    {"id": "roadmap#pricing", "doc": "product/roadmap.md", "section": "Pricing",
     "text": "Enterprise price increase planned for Q1. Do not share with customers.",
     "acl": {"support", "product"}, "version": 1},
    {"id": "hr#comp", "doc": "hr/compensation.md", "section": "Support bands",
     "text": "Compensation bands for support engineers and refund-handling bonuses.",
     "acl": {"hr"}, "version": 1},
]
TITLES = {"policies/refunds.md": "Refund policy", "runbooks/sso.md": "Single sign-on runbook",
          "policies/billing.md": "Billing policy", "incidents/2291.md": "Incident report",
          "crm/ACME-88": "CRM record for Acme Corp", "product/roadmap.md": "Product roadmap",
          "hr/compensation.md": "HR compensation guide"}
IDS = [c["id"] for c in CHUNKS]
BY_ID = {c["id"]: c for c in CHUNKS}
TOKEN = re.compile(r"[a-z0-9]+(?:-[0-9]+)?")      # keeps "e-4012" as one token


def tokens(text):
    return TOKEN.findall(text.lower())


def contextualize(chunk):
    """Contextual retrieval: situate the chunk before indexing. A model writes this in production."""
    return f"{TITLES[chunk['doc']]}, section '{chunk['section']}': {chunk['text']}"


print(contextualize(BY_ID["refunds#enterprise@v3"]))
assert "enterprise" not in tokens(BY_ID["refunds#enterprise@v3"]["text"])
assert "enterprise" in tokens(contextualize(BY_ID["refunds#enterprise@v3"]))
```

### Two retrievers with opposite blind spots

**BM25** is the classic lexical scorer. For each query term it adds the term's rarity (IDF) times a saturating function of how often it appears in the chunk, normalized by chunk length:

```text
score(D, Q) = Σ_t  IDF(t) · f(t,D)·(k1 + 1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )
IDF(t)      = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))              (Lucene form)

k1 ≈ 1.2-2.0   term-frequency saturation: the 5th "refund" adds little over the 2nd
b  ≈ 0.75      length normalization: long chunks do not win just by being long

worked example, N = 10 chunks, "e-4012" in 1 of them:
IDF = ln(1 + 9.5/1.5) = ln(7.33) ≈ 1.99      a word in 8 of 10 chunks: ln(1 + 2.5/8.5) ≈ 0.26
```

That IDF gap is why BM25 nails identifiers: a rare token like an error code dominates the score. It is also why BM25 scores zero for "money back" against a chunk that says "refunds."

**Dense retrieval** embeds query and chunk and compares vectors. Offline we cannot call an embedding model, so the stand-in maps words to shared concept dimensions (refund, money, reimbursement all land on one axis) plus a weak hashed residual for everything else. It models the one property that matters here, paraphrases landing near each other, and it inherits the real weakness too: a code like `e-4012` is just one faint residual dimension. The live cell swaps in a real embedder.

```{code-cell} python
class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.dl = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avgdl, n = self.dl.mean(), len(docs)
        df = Counter(t for tf in self.tf for t in tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.tf))
        for t in set(tokens(query)) & self.idf.keys():
            for i, tf in enumerate(self.tf):
                if f := tf.get(t, 0):
                    norm = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    out[i] += self.idf[t] * f * (self.k1 + 1) / norm
        return out


CONCEPTS = {
    "refund": ["refund", "refunds", "money", "reimburse", "reimbursement", "credited", "credit"],
    "enterprise": ["enterprise", "large", "big", "biggest"],
    "duration": ["days", "weeks", "window", "month", "months"],
    "sso": ["sso", "saml", "idp", "identity", "sign", "login", "log", "single"],
    "setup": ["configure", "setup", "set", "upload", "enable", "install"],
    "seats": ["seat", "seats", "licenses", "reductions", "reduce", "reducing"],
    "error": ["error", "failures", "fail", "fails", "expired", "broken"],
    "account": ["customer", "customers", "account", "admin", "client"],
}
LEX = {w: i for i, words in enumerate(CONCEPTS.values()) for w in words}
RESIDUAL = 64


def embed(texts):
    """Stand-in bi-encoder: synonyms share a concept axis; other tokens get a faint hashed axis."""
    out = np.zeros((len(texts), len(CONCEPTS) + RESIDUAL))
    for r, text in enumerate(texts):
        for tok in tokens(text):
            if tok in LEX:
                out[r, LEX[tok]] += 1.0
            out[r, len(CONCEPTS) + int(hashlib.md5(tok.encode()).hexdigest(), 16) % RESIDUAL] += 0.25
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


class Dense:
    def __init__(self, docs):
        self.matrix = embed(docs)

    def scores(self, query):
        return self.matrix @ embed([query])[0]      # cosine, rows are unit vectors


def ranked(scores, k=20):
    order = np.argsort(-scores, kind="stable")
    return [IDS[i] for i in order[:k] if scores[i] > 0]


plain = [c["text"] for c in CHUNKS]
ctx = [contextualize(c) for c in CHUNKS]
bm25_plain, bm25 = BM25(plain), BM25(ctx)
dense = Dense(ctx)

cases = {"exact id": "customer reports E-4012",
         "paraphrase": "can a large customer get their money back after six weeks?"}
for label, q in cases.items():
    print(f"{label:10s} bm25={ranked(bm25.scores(q), 3)}")
    print(f"{'':10s} dense={ranked(dense.scores(q), 3)}")
q = "enterprise refund window"
print("no context :", ranked(bm25_plain.scores(q), 3))
print("contextual :", ranked(bm25.scores(q), 3))

assert ranked(bm25.scores(cases["exact id"]))[0] == "sso#errors"
assert ranked(dense.scores(cases["exact id"]))[0] != "sso#errors"          # dense blurs the code
assert "refunds#enterprise@v3" not in ranked(bm25.scores(cases["paraphrase"]))
assert ranked(dense.scores(cases["paraphrase"]))[0] == "refunds#enterprise@v3"
assert "refunds#enterprise@v3" not in ranked(bm25_plain.scores(q), 3)
assert "refunds#enterprise@v3" in ranked(bm25.scores(q), 3)                  # context fixed it
```

Look at the last two lines of output. Without context, BM25's top hit for "enterprise refund window" is the HR compensation page, because it contains "refund". That is two production bugs in one result: a miss and a leak. With context, the right chunk is in the top 3, but so is the superseded 45-day version. We fix those in order: fusion, then metrics, then permissions and freshness, then reranking.

### Fusion, and the metrics that justify it

Two rankers with opposite blind spots should be combined. Their scores are not comparable (BM25 is unbounded, cosine is in [-1, 1]), so the robust move is to fuse **ranks**, not scores. Reciprocal Rank Fusion (Cormack, Clarke, and Buettcher, SIGIR 2009):

```text
RRF(d) = Σ_rankers 1 / (k + rank_r(d)),   k = 60

worked example: chunk ranked 1st by BM25, absent from dense top-20:  1/61          = 0.0164
                chunk ranked 3rd by BM25, 2nd by dense:              1/63 + 1/62   = 0.0320
Agreement beats a single strong vote. k damps the gap between rank 1 and rank 5.
```

Then measure. Three metrics, each answering a different question about a ranked list against labeled relevance:

```text
recall@k   fraction of the relevant chunks that appear in the top k        "is it in there?"
MRR        mean over queries of 1 / rank of the first relevant chunk       "how high is the first hit?"
nDCG@k     DCG = Σ_i (2^rel_i − 1) / log2(i + 1), divided by the ideal DCG "is the order right, with grades?"

worked example, labels {A: 2, B: 1}, ranking [C, A, B]:
recall@3 = 2/2 = 1.0     MRR = 1/2 = 0.5
DCG  = 0/1 + 3/log2(3) + 1/log2(4) = 0 + 1.893 + 0.5 = 2.393
IDCG = 3/1 + 1/log2(3)             = 3 + 0.631       = 3.631     nDCG@3 = 0.659
```

Recall@k is the metric for the candidate stage: a chunk that is not retrieved cannot be reranked into place. MRR and nDCG are the metrics for the final ordering the model sees.

```{code-cell} python
def rrf(rankings, k=60):
    fused = Counter()
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            fused[doc] += 1 / (k + pos)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def recall_at_k(ranking, rel, k):
    return len({d for d in ranking[:k] if rel.get(d, 0) > 0}) / sum(v > 0 for v in rel.values())


def mrr(ranking, rel):
    return next((1 / i for i, d in enumerate(ranking, 1) if rel.get(d, 0) > 0), 0.0)


def ndcg_at_k(ranking, rel, k):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranking[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


assert abs(ndcg_at_k(["C", "A", "B"], {"A": 2, "B": 1}, 3) - 0.659) < 1e-3   # the worked example

# Graded labels: 2 = answers the question, 1 = useful context. Superseded versions are NOT relevant.
EVAL = [
    ("customer reports E-4012", {"sso#errors": 2}),
    ("can a large customer get their money back after six weeks?",
     {"refunds#enterprise@v3": 2, "refunds#standard": 1}),
    ("enterprise refund window", {"refunds#enterprise@v3": 2}),
    ("how do I set up SAML single sign-on", {"sso#setup": 2}),
    ("reducing seats mid-contract, how is it credited", {"billing#seats": 2}),
    ("why did EU customers fail to log in", {"incident#2291": 2, "sso#errors": 1}),
]


def evaluate(retrieve, k=3):
    rows = [(recall_at_k(retrieve(q), rel, k), mrr(retrieve(q), rel), ndcg_at_k(retrieve(q), rel, k))
            for q, rel in EVAL]
    return tuple(round(float(np.mean(col)), 3) for col in zip(*rows))


systems = {
    "bm25": lambda q: ranked(bm25.scores(q)),
    "dense": lambda q: ranked(dense.scores(q)),
    "hybrid (RRF)": lambda q: rrf([ranked(bm25.scores(q)), ranked(dense.scores(q))]),
}
results = {name: evaluate(fn) for name, fn in systems.items()}
print(f"{'system':14s} recall@3   MRR   nDCG@3")
for name, (r, m, n) in results.items():
    print(f"{name:14s} {r:8.3f} {m:6.3f} {n:8.3f}")

assert results["hybrid (RRF)"][0] == 1.0
assert results["hybrid (RRF)"][0] > max(results["bm25"][0], results["dense"][0])
```

Hybrid reaches full recall@3 where each retriever alone misses something, and it lifts MRR and nDCG too. But MRR is still well short of 1.0. Fusion put the right chunks in the candidate set, not always first: the superseded version and loosely related chunks sit above them. That is the job of the next two stages.

### Permissions, freshness, and reranking

**Permission-aware retrieval** has one rule: the model is never the enforcement point. If a chunk the user cannot read reaches the prompt, it is already leaked, whatever the instructions say. The pattern:

```text
1. PRE-filter at query time    ACL groups are metadata on every chunk; the index only
                                scores chunks the caller may read.
2. RE-CHECK at result time      ACLs change after indexing (someone is removed from a group).
                                Check the survivors against the live source of truth.
3. OVER-FETCH                   late rejections shrink the list; fetch more than k so you
                                still return k.
4. NO EXISTENCE LEAKS           never say "3 results hidden"; counts are information.
```

Identity comes from the authenticated request, never from the model. A `search(query, user="ceo")` tool argument is an injection vector.

**Freshness** here means one live version per section. Store the version and a content hash on every chunk, supersede old versions when a new one is indexed, and never mix vectors from two embedding models in one index.

**Reranking** is the precision stage. A bi-encoder embeds query and chunk separately, which is what makes it fast enough for millions of chunks. A cross-encoder reads the query and one candidate together and outputs a relevance score. It sees interactions (does this passage answer *this* question?) and is much more accurate, but it costs one model pass per candidate. So it runs on the top 20-200 fused candidates, never on the corpus. Our stand-in scores concept coverage of the query plus exact-identifier matches; a real one is a transformer such as a BGE reranker or a hosted rerank API.

```{code-cell} python
def rerank_score(query, passage):
    """Stand-in cross-encoder: judges the pair jointly (coverage of the query's needs, exact ids)."""
    q_concepts = {LEX[t] for t in tokens(query) if t in LEX}
    p_concepts = {LEX[t] for t in tokens(passage) if t in LEX}
    coverage = len(q_concepts & p_concepts) / len(q_concepts) if q_concepts else 0.0
    q_ids = {t for t in tokens(query) if any(ch.isdigit() for ch in t)}
    exact = 1.0 if q_ids and q_ids <= set(tokens(passage)) else 0.0
    overlap = len(set(tokens(query)) & set(tokens(passage))) / len(set(tokens(query)) | set(tokens(passage)))
    return 2 * coverage + exact + 0.5 * overlap


# Freshness: one live version per (doc, section), by version number.
latest = {}
for c in CHUNKS:
    key = (c["doc"], c["section"])
    if c["version"] > latest.get(key, {"version": -1})["version"]:
        latest[key] = c
LIVE = {c["id"] for c in latest.values()}

# The live ACL service. Product revoked support's access to the roadmap after indexing.
ACL_NOW = {c["id"]: set(c["acl"]) for c in CHUNKS}
ACL_NOW["roadmap#pricing"] = {"product"}


def search(query, groups, k=3, fetch=20, rerank=True):
    visible = np.array([bool(c["acl"] & groups) and c["id"] in LIVE for c in CHUNKS])   # pre-filter
    candidates = rrf([ranked(np.where(visible, bm25.scores(query), 0), fetch),
                      ranked(np.where(visible, dense.scores(query), 0), fetch)])
    if rerank:
        candidates.sort(key=lambda cid: -rerank_score(query, contextualize(BY_ID[cid])))
    return [cid for cid in candidates if ACL_NOW[cid] & groups][:k]                      # re-check


SUPPORT, CONTRACTOR = {"everyone", "support"}, {"everyone"}
fused_only = evaluate(lambda q: search(q, SUPPORT, k=20, rerank=False))
reranked = evaluate(lambda q: search(q, SUPPORT, k=20))
print("hybrid + acl + fresh          ", fused_only)
print("hybrid + acl + fresh + rerank ", reranked)
print("contractor, 'refund bonuses'  ", search("refund bonuses for support engineers", CONTRACTOR))
print("support, 'enterprise price'   ", search("enterprise price increase", SUPPORT))
print("support, 'enterprise refund'  ", search("enterprise refund window", SUPPORT))

assert reranked[1] > fused_only[1] and reranked[2] > fused_only[2]      # rerank fixed the order
assert all("hr#comp" not in search(q, CONTRACTOR, k=20) for q, _ in EVAL)
assert "roadmap#pricing" not in search("enterprise price increase", SUPPORT)   # re-check caught it
assert "refunds#enterprise@v2" not in search("enterprise refund window", SUPPORT, k=20)
assert search("enterprise refund window", SUPPORT)[0] == "refunds#enterprise@v3"
```

The reranker moved MRR and nDCG, which is exactly what it is for. It cannot move recall: it only reorders what candidate generation found. That division of labor is the architecture: cheap and wide first, expensive and precise second. The roadmap chunk passed the pre-filter (its indexed ACL still said "support") and was caught by the re-check, which is why both layers exist.

### When brute force stops being enough

Everything above scores every chunk, which is fine to a few hundred thousand. Past that, dense retrieval uses an approximate nearest-neighbor (ANN) index, and the index choice interacts with everything in this lesson.

```text
HNSW    multi-layer proximity graph; greedy descent from the top layer, beam of efSearch at
        layer 0. M = links per node (recall and memory up), efConstruction = build quality,
        efSearch = query-time recall vs latency (>= k). Only M changes memory.
        memory ≈ 4d + ~8M bytes/vector → 768-d, M = 16: ≈ 3.2 KB/vector; 100M vectors ≈ 320 GB RAM.
IVF-PQ  k-means into nlist cells, probe nprobe of them; product quantization stores m bytes per
        vector → 768-d at m = 64: 64 B instead of 3,072 B (48x smaller). Needs training; refine
        top results with exact vectors. The billion-scale, limited-RAM option.
```

Two production consequences. **Filtering and ANN fight.** Pre-filtering a very selective ACL fragments the HNSW graph, and post-filtering returns fewer than k results (ask for 10, get 3). Vector databases address this with filter-aware traversal, and very selective filters often fall back to brute force over the survivors; per-tenant namespaces sidestep it entirely for hard isolation. **Freshness costs graph quality.** HNSW deletes are tombstones, so an index with heavy churn degrades until rebuilt, and IVF centroids drift as the corpus changes. Plan for periodic rebuilds and measure recall@k against exact search on a sample.

## How it shows up in the stack

For the agent, the whole pipeline above collapses into one tool: `search_kb(query)`. Everything interesting happens around it.

**Query-side techniques** are what the agent contributes. *Multi-query* rewrites the question several ways and fuses the results. *HyDE* (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels") asks the model to write a hypothetical answer and searches with that, because an answer-shaped passage sits closer to answer passages in embedding space than a short question does. *Decomposition and multi-hop* split "what refund window applies to Acme?" into "what plan is Acme on?" then "what is the refund window for that plan?", with the second query depending on the first result (IRCoT, Trivedi et al., 2022, interleaves retrieval with reasoning steps this way).

```text
model:        search_kb("Acme current plan")
application:  hybrid + ACL(user=support) → [crm#acme] "upgraded ... to Enterprise on 2026-08-02"
model:        search_kb("Enterprise plan customers can get a refund within a window of days")   ← HyDE
application:  → [refunds#enterprise@v3] "The window is 60 days ... account admin"
model:        "Acme is on Enterprise [crm#acme], so the window is 60 days [refunds#enterprise@v3]."
```

In LangGraph that is the agent loop from 00 with a custom retrieval node instead of `ToolNode`. The custom node exists for one reason: the user's groups come from the run's `config`, set by your authenticated API layer, and are never a tool argument the model can fill in. The node also enforces a retrieval budget, the fix for 14's "14 searches, $3, no answer."

```{code-cell} python
import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

MAX_SEARCHES = 3


def agent(state: MessagesState, config: RunnableConfig):
    return {"messages": [config["configurable"]["model"].invoke(state["messages"])]}


def retrieve(state: MessagesState, config: RunnableConfig):
    """search_kb executor. Identity and budget come from the runtime, not from the model."""
    groups = set(config["configurable"]["user_groups"])
    used = sum(isinstance(m, ToolMessage) for m in state["messages"])
    out = []
    for call in state["messages"][-1].tool_calls:
        if used >= MAX_SEARCHES:
            body = "budget exhausted: answer from what you have, or say the sources do not cover it"
        else:
            hits = search(call["args"]["query"], groups, k=2)
            body = json.dumps([{"id": h, "text": BY_ID[h]["text"]} for h in hits])
        used += 1
        out.append(ToolMessage(body, tool_call_id=call["id"]))
    return {"messages": out}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("retrieve", retrieve)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", lambda s: "retrieve" if s["messages"][-1].tool_calls else END,
                              ["retrieve", END])
builder.add_edge("retrieve", "agent")
rag_agent = builder.compile()

question = [HumanMessage("What refund window applies to Acme?")]
support_model = scripted(
    [("search_kb", {"query": "Acme current plan"})],
    [("search_kb", {"query": "Enterprise plan customers can get a refund within a window of days"})],
    "Acme is on Enterprise [crm#acme], so the refund window is 60 days and the request "
    "must come from an account admin [refunds#enterprise@v3].")
out = rag_agent.invoke({"messages": question},
                       {"configurable": {"model": support_model, "user_groups": ["everyone", "support"]}})
hops = [json.loads(m.content) for m in out["messages"] if isinstance(m, ToolMessage)]
print("hop 1:", [h["id"] for h in hops[0]])
print("hop 2:", [h["id"] for h in hops[1]])
print("answer:", out["messages"][-1].content)

# Same graph, same question, a contractor: the CRM note never enters the context.
contractor_model = scripted(
    [("search_kb", {"query": "Acme current plan"})],
    "I can't see Acme's account details. Standard plans have a 30-day window [refunds#standard].")
out_c = rag_agent.invoke({"messages": question},
                         {"configurable": {"model": contractor_model, "user_groups": ["everyone"]}})
seen = [h["id"] for m in out_c["messages"] if isinstance(m, ToolMessage) for h in json.loads(m.content)]
print("contractor saw:", seen)

assert hops[0][0]["id"] == "crm#acme" and hops[1][0]["id"] == "refunds#enterprise@v3"
cited = set(re.findall(r"\[([^\]]+)\]", out["messages"][-1].content))
assert cited <= {h["id"] for hop in hops for h in hop}, "every citation was actually retrieved"
assert "crm#acme" not in seen and "hr#comp" not in seen
```

The citation check at the end is 14's `verify_citations` in one line: every id the answer cites must be among the ids that retrieval actually returned to this user. Cheap, and it catches both hallucinated citations and a model citing something from its priors.

The same retrieval evaluation runs against a real embedder by swapping one function. This is the experiment to run before believing any retrieval change: same labeled set, same metrics, one variable changed.

```{code-cell} python live
import os
from langchain.embeddings import init_embeddings

embedder = init_embeddings(os.environ.get("COURSE_EMBEDDINGS", "openai:text-embedding-3-small"))


class RealDense:
    def __init__(self, docs):
        m = np.array(embedder.embed_documents(docs))
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def scores(self, query):
        v = np.array(embedder.embed_query(query))
        return self.matrix @ (v / np.linalg.norm(v))


real_dense = RealDense(ctx)
for name, fn in {"real dense": lambda q: ranked(real_dense.scores(q)),
                 "bm25 + real dense (RRF)": lambda q: rrf([ranked(bm25.scores(q)),
                                                          ranked(real_dense.scores(q))])}.items():
    print(f"{name:24s} recall@3, MRR, nDCG@3 =", evaluate(fn))
```

Where the pieces live in a real system: BM25 and dense usually live in one engine that supports hybrid queries (Elasticsearch/OpenSearch, Vespa, or a vector database with sparse vectors); the reranker is a hosted API or a GPU service; contextualization runs in the indexing pipeline with prompt caching over the parent document; ACLs sync from source systems by change events. The LangGraph layer sees none of that. It sees a tool, a budget, and an identity in `config`.

## What goes wrong in production

1. **Exact identifiers vanish.** Symptom: "E-4012", SKUs, invoice numbers, and people's names retrieve unrelated chunks. Cause: dense-only retrieval; rare tokens barely move an embedding. Fix: hybrid with BM25 or a learned sparse model, fused with RRF, and a tokenizer that keeps identifiers whole (`e-4012`, not `e` and `4012`).

2. **Good recall, wrong answers.** Symptom: recall@10 is 0.95 and users still get wrong answers. Cause: the right chunk is at rank 7 of 10 and the model anchors on the first ones, or a stale version sits above it. Fix: rerank to a short top 3-5, remove superseded versions, and measure MRR and nDCG, not just recall. Then check faithfulness separately (14's three-way split).

3. **Permission leakage.** Symptom: a contractor's answer quotes the HR guide or a roadmap they lost access to yesterday. Cause: filtering done by the prompt, done only at index time, or keyed on a user id the model supplied. Fix: ACL pre-filter from the authenticated identity, re-check at result time against the live source, over-fetch, per-tenant namespaces for hard isolation, and no counts of hidden results.

4. **Filtered search returns too few results.** Symptom: ask for k=10, get 3, and answer quality collapses for users with narrow permissions. Cause: post-filtering after ANN, or a pre-filter that fragments the HNSW graph. Fix: filter-aware ANN in your vector store, over-fetch proportional to filter selectivity, brute force over small filtered sets, and a monitor on "returned fewer than k."

5. **Stale and duplicated versions.** Symptom: the answer cites the 45-day policy, or both policies at once. Cause: re-indexing appended a new version without superseding the old; tombstones not applied. Fix: stable section keys, version and content hash per chunk, supersede on upsert, a freshness SLO measured with canary edits, and a full rebuild behind a version switch when the embedding model changes.

6. **Rewriting that drifts.** Symptom: HyDE or multi-query retrieves confidently about the wrong thing. Cause: the hypothetical answer encodes the model's prior, which may be the exact wrong fact you are trying to correct (a HyDE passage saying "30 days"). Fix: fuse rewritten-query results with the original query's results rather than replacing them, cap rewrites, and evaluate rewriting against the same labeled set; it does not always help.

7. **Reranker cost and latency.** Symptom: p95 latency doubles after adding reranking. Cause: reranking 500 candidates, or reranking on every agent hop. Fix: rerank 50-100 candidates, cache by (query, chunk id), and use the reranker's score as an abstention signal: if the best candidate scores low, say the sources do not cover it.

8. **No retrieval evals.** Symptom: every change is argued from three hand-picked examples. Cause: no labeled set. Fix: build one from real traffic and failed conversations, with graded labels, and track recall@k, MRR, and nDCG per change, sliced by query type (identifier, paraphrase, multi-hop) and by permission group, since labels must be judged under the user's permissions.

## What changed by 2026

- **Hybrid is the default, not the upgrade.** Production retrieval stacks fuse sparse and dense signals as a baseline, with RRF as the parameter-light fusion, because the two fail on disjoint query types.
- **Chunks are enriched, not just split.** Anthropic's contextual retrieval write-up made index-time context generation standard practice and published the failure-rate reductions quoted above. Prompt caching makes it affordable because the parent document is shared across all of its chunks.
- **Retrieval became a tool the agent drives.** Anthropic's context-engineering guidance describes just-in-time retrieval: keep lightweight identifiers and fetch content when needed. Multi-hop, rewriting, and HyDE are things the agent does with a search tool, bounded by a budget, rather than fixed pipeline stages.
- **Rerankers are a commodity stage.** Cross-encoder and hosted rerank APIs are a routine part of the pipeline, and LLM listwise reranking is used where quality justifies its cost.
- **Permissions moved into the retriever.** Enterprise assistants are expected to enforce source-system permissions at query time and re-check at result time, and interviewers ask this directly.
- **Retrieval evaluation is separated from answer evaluation.** Recall@k, MRR, and nDCG for the retriever; faithfulness and citation precision for the generator; both tracked on every change.
- **Long context changed k, not the need to retrieve.** Bigger windows let you pass more candidates, but cost, latency, and context rot still favor a short, reranked context.

## Interview questions

**Q1. How do you choose between BM25 and dense retrieval, and when do you need both?** *(reported at Microsoft, Perplexity, Glean)*

Strong: They fail on opposite query types. BM25 is exact and rewards rare tokens through IDF, so it wins on error codes, SKUs, names, and jargon, but it scores zero on paraphrase. Dense retrieval handles vocabulary mismatch, "money back" versus "refund," but blurs identifiers into a single faint dimension. Almost every real corpus has both query types, so I run both and fuse with reciprocal rank fusion, which works on ranks and avoids normalizing incomparable scores. I justify it with recall@k on a labeled set sliced by query type, where hybrid should beat each alone.

Weak: "Dense is newer and better; BM25 is legacy."

**Q2. What is a reranker, when do you use one, and what does a cross-encoder cost you?** *(reported at Cohere, Microsoft, Perplexity)*

Strong: A bi-encoder embeds query and document separately, which lets you precompute document vectors and search millions quickly, but it never sees the pair together. A cross-encoder reads query and candidate jointly and scores relevance, which is far more accurate but costs one forward pass per candidate at query time. So it is a second stage: retrieve 50 to 200 candidates cheaply, rerank them, pass the top 3 to 5 to the model. It improves MRR and nDCG, not recall, since it can only reorder what was retrieved. I watch its latency, cache by query and chunk, and use its top score as an abstention signal.

Weak: "It uses an LLM to pick the best documents."

**Q3. Design permission-aware retrieval: users must never see content they cannot access in the source system.** *(reported at Microsoft, Databricks, Glean, Palantir)*

Strong: ACLs are synced from source systems as metadata on every chunk, driven by change events plus periodic reconciliation. At query time the retriever pre-filters by the caller's groups, taken from the authenticated request and never from a model-supplied argument. After ranking, survivors are re-checked against the live permission service, because ACLs change after indexing, and I over-fetch so late rejections do not starve k. For hard tenant isolation I use separate namespaces or indexes. I never expose counts of hidden results. The model is never the enforcement point, because once text is in the prompt it is leaked.

Weak: "Tell the model which documents the user is allowed to see."

**Q4. Filtered search with k=10 returns 3 results. Why, and what do you do?**

Strong: Most likely the filter is applied after approximate search: the ANN index returned its 10 nearest neighbors overall and the filter removed 7. Pre-filtering fixes that but can fragment an HNSW graph when the filter is very selective, hurting recall. The options are filter-aware ANN if the store supports it, over-fetching proportional to selectivity, brute-force search over the filtered subset when it is small, or per-tenant partitions. I would measure what fraction of the corpus survives typical filters and alert on queries that return fewer than k.

Weak: "Increase k."

**Q5. How do you evaluate a retrieval system? Explain recall@k, MRR, and nDCG.** *(reported at Cohere)*

Strong: I build a labeled set from real queries, with graded relevance per chunk, judged under the asking user's permissions. Recall@k asks whether the relevant chunks are in the top k, which is the right metric for candidate generation because nothing downstream can recover a miss. MRR is the mean reciprocal rank of the first relevant result, which captures whether the best evidence is at the top. nDCG uses graded relevance with a log discount by position, normalized by the ideal ordering, so it rewards putting the most useful chunks first. I report them separately from generation metrics like faithfulness and citation precision, and slice by query type.

Weak: "We check if the answers look right."

**Q6. What is HyDE, and when does it beat standard dense retrieval?**

Strong: Hypothetical Document Embeddings: the model writes a plausible answer to the question, and you embed and search with that passage instead of the question. It helps when questions are short or phrased very differently from the documents, because an answer-shaped passage lands closer to answer passages. It hurts when the model's prior is wrong in a way that pulls retrieval toward the wrong fact, and it adds a model call of latency. I fuse HyDE results with the original query's results rather than replacing them, and I keep it only if the labeled set says it helps.

Weak: "It makes the query longer so it matches better."

**Q7. How do you keep the index fresh when the corpus changes continuously?** *(reported at Perplexity, Cursor)*

Strong: Stable ids per section, and a version and content hash per chunk, so an upsert supersedes the old version instead of adding a duplicate. Indexing is driven by change events with a reconciliation sweep for missed events. I define a freshness SLO and measure it with canary edits. ANN indexes need care: HNSW deletes are tombstones that degrade the graph, IVF centroids drift, so I schedule rebuilds and compare recall against exact search on a sample. An embedding-model change is a full rebuild behind a version switch, never a mixed index.

Weak: "Re-index every night."

**Q8. When is agentic, multi-hop retrieval worth the extra cost?**

Strong: When the answer genuinely needs facts that no single chunk contains, such as "what refund window applies to Acme," which needs the plan from the CRM and then the policy for that plan, or when the first query often fails and needs rewriting. For single-fact lookups a fixed pipeline is cheaper, faster, and easier to evaluate. When I do make retrieval a tool, I bound it with a search budget, fuse rewritten queries with the original, verify citations against what was actually retrieved, and evaluate the trajectory, not only the final answer.

Weak: "Agents are always better because they can search more."

## Build it yourself

Extend `search` into a production retriever with a report card:

1. Add a `tenant` field to every chunk and a `tenant` argument to `search`. Implement hard isolation: chunks from other tenants are never scored, not merely filtered afterwards.
2. Add a `multi_query(queries, groups, k)` that runs `search` for each rewrite plus the original and fuses the results with RRF.
3. Add a reranker abstention threshold: if the best candidate's `rerank_score` is below 1.0, return an empty list, and make the LangGraph `retrieve` node return "the sources do not cover this."
4. Write `report(system)` that prints recall@3, MRR, and nDCG@3 per query type (label each EVAL row as `identifier`, `paraphrase`, `context`, or `incident`) and overall.

Acceptance checks:

- A chunk from tenant B never appears in any `search(..., tenant="A")` result, and the BM25 scores for tenant B chunks are never computed (assert on a call counter).
- `multi_query(["large customer money back six weeks", "enterprise refund window"], SUPPORT, 3)` returns `refunds#enterprise@v3` first.
- `search("parking reimbursement for the office garage", SUPPORT)` returns `[]` with the threshold on, and the graph answers without citing anything.
- `report` shows the reranked system's MRR is at least the fused-only MRR in every query-type slice.
- Every assert from this lesson still passes.

## Key takeaways

- Sparse and dense retrieval fail on opposite queries; fuse their rankings with RRF instead of choosing one.
- Contextual retrieval puts a chunk's missing context back into its text before indexing, which helps both BM25 and embeddings.
- Candidate generation is judged by recall@k; reranking is judged by MRR and nDCG, and cannot fix recall.
- Permissions are enforced by the retriever from the authenticated identity, pre-filtered and re-checked, never by the model.
- Freshness is a data-model property: stable section ids, versions, content hashes, superseding upserts, and a measured SLO.
- Query rewriting, HyDE, and multi-hop belong to the agent, bounded by a retrieval budget and checked by citation verification.
- Every retrieval change is an experiment on a labeled set with graded relevance, sliced by query type and permission group.
