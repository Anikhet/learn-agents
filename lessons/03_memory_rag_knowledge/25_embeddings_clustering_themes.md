# 25. Embeddings, Clustering, and Theme Discovery

*Track C (retrieval + production AI). Course addition; extends the book's Chapter 14 (Knowledge Retrieval). Prerequisites: 14. Knowledge Retrieval. Used by 29. Advanced Production RAG and 08. Memory Management.*

Every retrieval system, memory store, and semantic router in this course rests on one primitive: turn text into a vector so that "close" means "similar." This lesson opens that box. It works through the geometry (why cosine, why normalize, when dot product is the same thing), builds k-means in numpy to discover themes in a pile of support tickets, uses a model to name the clusters, and then spends real time on the cases where the geometry lies to you.

## The problem

Ops asks the support agent:

> "What are customers complaining about this month? Give me the top themes with counts."

There are 4,000 tickets in the CRM. Reading them is not an option. Three naive approaches:

```text
Approach                     Where it breaks
─────────────────────────    ───────────────────────────────────────────────────
Keyword counts               "login", "sign in", "can't get in", "SSO broken" are
                             four keywords and one theme
Ask the model to summarize   4,000 tickets do not fit; sample 200 and the counts are
                             guesses; run it twice and the themes differ
Predefined categories        Whatever is new this month is by definition not in
                             last quarter's taxonomy
```

The workable pipeline uses embeddings for what they are good at (cheap similarity at scale) and a model for what it is good at (naming and explaining):

```text
tickets ──embed──► vectors ──cluster──► groups ──sample──► model names each group
                                          │
                                          └──► counts per group (exact, from assignments)
                                                    │
                                          new ticket ──nearest centroid──► theme
```

The numbers come from arithmetic on assignments, not from a model guessing. The names come from the model reading a handful of representative tickets per cluster. That division is the design.

To trust it you need to understand what an embedding is geometrically, and where it fails. The interview question this answers is "cosine or dot product, and why normalize?", followed by "when do embeddings mislead you?"

## Build it without a framework

The scripted model stands in for the real one so the geometry and the clustering are the object of study; the live cell swaps in a real embedder and a real model.

```{code-cell} python
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


print("scripted model ready")
```

### The geometry: cosine, dot product, and normalization

An embedding model is a bi-encoder: text goes in, a fixed-length vector comes out, and the model was trained (contrastively, with positives pulled together and negatives pushed apart) so that the angle between vectors tracks semantic similarity. The angle, not the length. Length carries noise: frequency effects, sentence length, artifacts of training.

Three similarity measures and how they relate:

```text
cosine(a, b)   = a·b / (|a| |b|)                 angle only, in [-1, 1]
dot(a, b)      = a·b                             angle and lengths
euclid(a, b)^2 = |a|^2 + |b|^2 - 2 a·b

If |a| = |b| = 1 (L2-normalized):
    dot(a, b)      = cosine(a, b)
    euclid(a, b)^2 = 2 - 2 cosine(a, b)

Worked example, unit vectors at 60 degrees:
    cosine = 0.5    dot = 0.5    euclid^2 = 2 - 1.0 = 1.0    euclid = 1.0
```

So: normalize once at index time, and cosine becomes a dot product, and nearest-by-euclidean equals nearest-by-cosine. Every vector database exploits this; it is why "which metric?" usually reduces to "did you normalize?" The exception is models trained with unnormalized dot product, where length is meaningful; use the metric the model card specifies, and use its query and passage prefixes if it has them.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(7)


def normalize(m):
    m = np.atleast_2d(m).astype(float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


a, b = rng.normal(size=8), rng.normal(size=8)
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
an, bn = normalize(a)[0], normalize(b)[0]

assert abs(an @ bn - cos) < 1e-12, "dot product of unit vectors is the cosine"
assert abs(np.sum((an - bn) ** 2) - (2 - 2 * cos)) < 1e-12, "squared euclidean = 2 - 2cos on the unit sphere"
# Scaling a vector changes the dot product but not the cosine: length is not meaning.
assert abs((3 * a) @ b - 3 * (a @ b)) < 1e-9 and abs(np.dot(normalize(3 * a)[0], bn) - cos) < 1e-12

# Ranking check: for unit vectors, argsort by dot == argsort by -euclidean.
M = normalize(rng.normal(size=(50, 8)))
q = normalize(rng.normal(size=8))[0]
by_dot = np.argsort(-(M @ q))
by_euclid = np.argsort(np.linalg.norm(M - q, axis=1))
assert np.array_equal(by_dot, by_euclid)
print(f"cosine={cos:.4f}  dot(unit)={an @ bn:.4f}  euclid^2={np.sum((an - bn) ** 2):.4f}  2-2cos={2 - 2 * cos:.4f}")
```

### A corpus with hidden themes

Offline we cannot call an embedding model, so the cell synthesizes embeddings with known structure: three theme directions on the unit sphere, each ticket a noisy copy of its theme's direction. This is a caricature of what a real embedder produces, but it lets us check clustering against ground truth, which you will never have in production. The live cell repeats the pipeline on real tickets with a real embedder.

```{code-cell} python
DIMS = 32
THEMES = ["login", "billing", "shipping"]
TICKETS = {
    "login": ["Cannot sign in after password reset", "SSO login loops back to the start page",
              "Two-factor code never arrives", "Locked out of my account this morning",
              "Login page says invalid credentials but they are right", "Can't get in via Google SSO"],
    "billing": ["Charged twice for the same invoice", "Discount was not applied to the renewal",
                "Invoice shows the wrong company name", "Need a refund for the duplicate charge",
                "Why did my plan price go up", "Credit note still not received"],
    "shipping": ["Order stuck in transit for two weeks", "Package marked delivered but never arrived",
                 "Wrong item shipped", "Tracking number does not work", "Shipping address was ignored"],
}

theme_dirs = normalize(rng.normal(size=(len(THEMES), DIMS)))
texts, labels, X = [], [], []
for t_idx, theme in enumerate(THEMES):
    for text in TICKETS[theme]:
        texts.append(text)
        labels.append(t_idx)
        X.append(theme_dirs[t_idx] + 0.15 * rng.normal(size=DIMS))    # theme direction + noise
X = normalize(np.array(X))
labels = np.array(labels)

# Sanity: within-theme similarity beats cross-theme similarity on average.
S = X @ X.T
same = S[labels[:, None] == labels[None, :]].mean()
diff = S[labels[:, None] != labels[None, :]].mean()
assert same > diff + 0.3
print(f"{len(texts)} tickets, {DIMS} dims; mean cosine same-theme {same:.2f}, cross-theme {diff:.2f}")
```

### k-means from scratch

k-means finds k centroids that minimize the sum of squared distances from each point to its assigned centroid. Two steps alternate until nothing changes: assign each point to its nearest centroid, then move each centroid to the mean of its points. On unit vectors, "nearest by euclidean" is "most similar by cosine," so this is spherical clustering in effect (renormalize centroids to keep it exact).

Initialization matters. Random init can put two centroids in one theme and none in another. k-means++ picks each new centroid with probability proportional to its squared distance from the nearest existing centroid, which spreads them out. We seed the generator so the run is deterministic.

Choosing k is the perennial problem. Inertia (total squared distance) always falls as k rises, so you look for the elbow, or use a silhouette-like score, or simply run several k and let the model-labeling step reveal when two clusters get the same name. In practice, you pick k a bit high and merge.

```{code-cell} python
def kmeans_pp_init(X, k, rng):
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        centroids.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centroids)


def kmeans(X, k, rng, iters=50):
    C = kmeans_pp_init(X, k, rng)
    for _ in range(iters):
        assign = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)   # E-step
        newC = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(k)])
        newC = normalize(newC)                                                     # stay on the sphere
        if np.allclose(newC, C):
            break
        C = newC
    # Cluster ids are arbitrary; renumber by first member so runs (and labels) line up.
    order = sorted(np.unique(assign), key=lambda j: np.flatnonzero(assign == j)[0])
    C, assign = C[order], np.argsort(order)[assign]
    inertia = float(((X - C[assign]) ** 2).sum())
    return C, assign, inertia


def purity(assign, labels):
    """Fraction of points whose cluster's majority label matches their own. Needs ground truth."""
    return sum(np.bincount(labels[assign == j]).max() for j in np.unique(assign)) / len(labels)


for k in (2, 3, 4, 5):
    C, assign, inertia = kmeans(X, k, np.random.default_rng(0))
    print(f"k={k}  inertia={inertia:6.2f}  purity={purity(assign, labels):.2f}")

C, assign, inertia = kmeans(X, 3, np.random.default_rng(0))
assert purity(assign, labels) == 1.0, "with clear themes and k=3, clusters recover the ground truth"
assert len(np.unique(assign)) == 3
counts = np.bincount(assign)
print("cluster sizes:", counts.tolist())
```

Look at the inertia column. It drops sharply from k=2 to k=3, then slowly. That is the elbow, and it coincides with the true number of themes because the synthetic data is clean. Real data has no clean elbow; the model-labeling step below is what tells you two clusters are really one.

### Label clusters with a model, then classify against frozen centroids

For each cluster, pick the tickets closest to the centroid (the most representative ones), and ask the model for a short label and a one-line description. Then freeze the centroids and the labels as a taxonomy version. New tickets are classified by nearest centroid, which is a matrix multiply, not a model call. Counts are exact because they come from assignments.

Two design rules that keep this honest: the model names clusters, it never decides membership; and the taxonomy is versioned, so counts from two months are comparable only if they used the same frozen centroids.

```{code-cell} python
def representatives(X, C, assign, j, n=3):
    idx = np.where(assign == j)[0]
    sims = X[idx] @ C[j]
    return idx[np.argsort(-sims)[:n]]


# One scripted label per cluster, in cluster order. A real model reads the samples.
label_model = scripted("Login and SSO access failures", "Billing and invoice errors", "Shipping and delivery problems")

taxonomy = {}
for j in range(len(C)):
    reps = representatives(X, C, assign, j)
    prompt = "Name this group of support tickets in five words or fewer:\n" + "\n".join(f"- {texts[i]}" for i in reps)
    taxonomy[j] = {"label": label_model.invoke(prompt).content, "centroid": C[j],
                   "n": int((assign == j).sum()), "examples": [texts[i] for i in reps]}

# Frozen taxonomy: classification of a new ticket is nearest centroid, no model call.
def classify(vec, taxonomy, floor=0.5):
    sims = {j: float(vec @ t["centroid"]) for j, t in taxonomy.items()}
    best = max(sims, key=sims.get)
    return (taxonomy[best]["label"], sims[best]) if sims[best] >= floor else ("uncategorized", sims[best])


new_login = normalize(theme_dirs[0] + 0.15 * rng.normal(size=DIMS))[0]
new_random = normalize(rng.normal(size=DIMS))[0]        # something the taxonomy has never seen

label, sim = classify(new_login, taxonomy)
assert label == "Login and SSO access failures", label
assert classify(new_random, taxonomy)[0] == "uncategorized", "off-taxonomy items must not be forced into a theme"
assert sum(t["n"] for t in taxonomy.values()) == len(texts)

for j, t in sorted(taxonomy.items(), key=lambda kv: -kv[1]["n"]):
    print(f"{t['n']:2d}  {t['label']:32s} e.g. {t['examples'][0]}")
print(f"new ticket -> {label} (cos {sim:.2f}); random vector -> {classify(new_random, taxonomy)[0]}")
```

The similarity floor in `classify` is the "uncategorized" bucket. Without it every new ticket is forced into the nearest theme, and the first sign that something new is happening (an outage, a new product) is hidden inside an old count. Track the uncategorized rate; when it climbs, re-cluster.

## How it shows up in the stack

Where an agent engineer meets this material:

```text
Component                    Embedding decision hiding inside it
───────────────────────────  ─────────────────────────────────────────────────────
Retrieval node / tool (14)   normalize once, dot-product search, k, similarity floor
Memory store (08)            InMemoryStore(index={"embed": ..., "dims": ...}) is exactly the
                             CosineIndex from lesson 14 with namespaces
Semantic router (02)         classify() above: nearest centroid of example utterances
Dedupe in memory writes      cosine above a threshold means "same memory"
Hybrid retrieval (29)        dense scores are one of two rankings fused by RRF
Eval clustering (19, 37)     cluster failed traces to find failure themes, label with a model
```

The clustering pipeline is itself an agent tool in the running example: `summarize_ticket_themes(month)` embeds, clusters, labels, and returns counts with examples, so the ops question at the top becomes one tool call with exact numbers. The model that answers the user never sees 4,000 tickets; it sees a table.

Two stack-level facts to carry:

- Embeddings are cheap, model calls are not. Embedding 4,000 tickets is one batched call and a few cents; asking a model to read them is thousands of times more. Put the model where judgment is needed (naming, explaining, edge cases) and arithmetic everywhere else.
- Embedding-model version is part of your data schema. Vectors from two models are not comparable, so a model upgrade means re-embedding everything, including frozen centroids, and re-running the labeling to confirm themes survived.

### The real thing

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
real_embed = init_embeddings(os.environ.get("COURSE_EMBEDDINGS", "openai:text-embedding-3-small"))

Xr = normalize(np.array(real_embed.embed_documents(texts)))
Cr, assign_r, _ = kmeans(Xr, 3, np.random.default_rng(0))
print("purity against hand labels:", round(purity(assign_r, labels), 2))
for j in range(3):
    reps = representatives(Xr, Cr, assign_r, j)
    name = real_model.invoke("Name this group of support tickets in five words or fewer:\n"
                             + "\n".join(f"- {texts[i]}" for i in reps)).content
    print(f"{(assign_r == j).sum():2d}  {name}")

# Negation check on a real embedder: expect these to be uncomfortably close.
pair = normalize(np.array(real_embed.embed_documents(["Refund was approved", "Refund was not approved"])))
print("cosine(approved, not approved) =", round(float(pair[0] @ pair[1]), 3))
```

## What goes wrong in production

1. **Negation and small-word blindness.** Symptom: "refund approved" and "refund not approved" land in the same cluster; a retrieval for "invoices without discounts" returns invoices with discounts. Cause: embeddings encode topic strongly and polarity weakly; "not" barely moves the vector. Fix: do not use embeddings alone for anything where polarity matters; add a lexical signal, a classifier, or let the model read the candidates (rerank). The live cell shows the effect on a real model.

2. **Anisotropy: everything is similar to everything.** Symptom: cosine scores between unrelated texts are 0.6 and between related texts 0.8; thresholds tuned on one corpus are useless on another. Cause: embedding spaces are anisotropic; vectors occupy a narrow cone, and a large common component dominates the dot product. Fix: never hard-code a similarity threshold from a tutorial; calibrate on your corpus (subtract the corpus mean before comparing, or set thresholds from a labeled sample); prefer rank-based fusion (29) over raw score fusion.

3. **Hubness.** Symptom: one short, generic ticket ("It doesn't work") is in the top-10 for every query. Cause: in high dimensions some points are near many others; short or generic texts often sit near the center of the cone. Fix: filter very short texts before indexing, penalize by frequency of appearance in results, or rerank.

4. **Clusters that mirror surface form instead of meaning.** Symptom: one cluster is "tickets written in Spanish" or "tickets that start with Hi team." Cause: the embedder encodes language, register, and format as strongly as topic. Fix: normalize inputs (strip greetings and signatures, translate or use a multilingual model), and check labels: a model-generated label like "Spanish tickets" is your signal.

5. **k-means forcing spherical, equal-ish clusters.** Symptom: a giant theme is split into three near-identical clusters while a small, distinct one is absorbed. Cause: k-means minimizes squared distance and prefers balanced, round clusters; it has no notion of density or outliers. Fix: over-cluster and merge by label; or use a density method (HDBSCAN) that produces an explicit noise bucket; always keep an uncategorized path.

6. **Unstable taxonomies.** Symptom: month-over-month theme counts jump because the clusters changed shape. Cause: re-clustering every run; different init, different k, different names. Fix: freeze centroids and labels as a versioned taxonomy; classify new data against the frozen version; re-cluster deliberately on a schedule, then map old themes to new with a confusion table.

7. **Mixed embedding models.** Symptom: retrieval quality collapses after an upgrade for half the documents. Cause: new vectors in an index built from the old model. Fix: embedding-model version on every vector, full reindex behind a version switch, never mix.

## What changed by 2026

- **Embedding models got cheaper, longer, and truncatable.** Current model cards describe long input windows and Matryoshka-style training that lets you truncate a vector to fewer dimensions with graceful quality loss, trading recall for memory. The geometry in this lesson is unchanged; the knobs multiplied.
- **Late chunking and contextual embeddings.** Instead of embedding isolated chunks, current practice embeds with surrounding document context (Anthropic's contextual retrieval prepends a model-written context to each chunk; late chunking pools token embeddings computed over the whole document). Both attack the "chunk loses its subject" failure from 14.
- **Evaluation moved to benchmarks and to your own data.** MTEB-style leaderboards rank embedders across retrieval, clustering, and classification tasks, but the consistent guidance is that the ranking on your corpus is what matters; a small labeled set of (query, relevant chunk) pairs beats a leaderboard position.
- **Theme discovery became an agent tool.** Rather than a quarterly analyst report, embed-cluster-label pipelines run as tools an agent calls on demand, with the model reserved for naming and explanation and counts coming from assignments. The failure modes above are what the tool's evals check.
- **Embeddings alone stopped being the retrieval answer.** Hybrid lexical plus dense with rank fusion and a reranker is the production default (29), precisely because of the negation, anisotropy, and hubness problems described here.

## Interview questions

**Q1. Cosine similarity or dot product for embedding search, and why do people normalize?**

Strong: Cosine measures the angle between vectors and ignores their lengths; dot product mixes angle and length. Embedding models are trained so that angle carries meaning and length is mostly noise, so cosine is usually what you want. If you L2-normalize every vector once at index time, the dot product equals the cosine and squared euclidean distance equals 2 minus 2 cosine, so all three metrics give the same ranking and search is a single matrix multiply. That is why vector databases ask you to normalize. The exception is a model trained with an unnormalized dot-product objective where length is meaningful; follow the model card, including any query and passage prefixes.

Weak: "Cosine is the standard for text."

**Q2. How would you find the top complaint themes in 4,000 tickets, with counts you would defend?**

Strong: Embed all tickets in one batched call, normalize, and run k-means with k-means++ initialization, trying several k and looking at inertia and at whether clusters get distinct names. For each cluster, take the few tickets nearest the centroid and have a model write a short label and description. Freeze centroids and labels as a versioned taxonomy. Counts come from assignments, so they are exact and reproducible; the model never decides membership. New tickets are classified by nearest centroid with a similarity floor so genuinely new issues land in an uncategorized bucket rather than inflating an old theme. I would validate on a hand-labeled sample and report purity, and I would show the representative tickets next to each count so a human can sanity-check the label.

Weak: "Ask the LLM to read the tickets and list the themes."

**Q3. When do embeddings mislead you?**

Strong: Whenever the thing you care about is not what the model was trained to encode strongly. Negation and polarity: "approved" and "not approved" are close. Exact identifiers: an error code or invoice number is one token the embedder barely distinguishes. Surface form: language, register, and boilerplate can dominate topic, so you get a "Spanish tickets" cluster. Anisotropy: all scores are compressed into a narrow band, so thresholds do not transfer between corpora. Hubness: generic short texts are near everything. The mitigations are the same each time: pair dense similarity with a lexical signal, calibrate thresholds on your own data, rerank candidates with a model that reads them, and keep an explicit uncategorized path.

Weak: "Embeddings work well as long as you use a good model."

**Q4. Why can't you compare vectors from two embedding models, and what does that imply operationally?**

Strong: Each model learns its own coordinate system; there is no shared basis, so a cosine between vectors from different models is meaningless even if the dimensions match. Operationally that means the embedding-model version is part of the schema on every stored vector, an upgrade is a full re-embedding of documents, queries, and any frozen centroids, and you cut over behind a version switch rather than mixing. It also means cached query embeddings are invalidated by an upgrade, and any similarity thresholds must be recalibrated.

Weak: "As long as the dimensions match, it works."

**Q5. How do you choose k for k-means, and what do you do when there is no clear elbow?**

Strong: Inertia always decreases with k, so the elbow heuristic looks for where the decrease flattens; silhouette-style scores compare within-cluster tightness to between-cluster separation. On real text there is rarely a clean elbow. My practical approach is to over-cluster, label every cluster with a model, and merge clusters whose labels are the same, which turns k selection into a review step a human can check. Alternatively use a density method like HDBSCAN that does not take k and produces a noise bucket. Either way, freeze the result as a taxonomy version so counts are comparable over time.

Weak: "Use the elbow method."

**Q6. Where does the model belong in an embed-cluster-label pipeline, and where does it not?**

Strong: The model belongs where judgment and language are needed: naming clusters from representative examples, writing a description, flagging that two clusters are really one, and explaining a specific ticket's assignment when a human asks. It does not belong in membership decisions or counting, because a model reading 4,000 tickets is expensive, non-deterministic, and gives you numbers you cannot reproduce. Embeddings and arithmetic handle membership and counts; the model handles meaning. That split is also what makes the pipeline testable: the numeric part has ground truth on a labeled sample, and the naming part can be reviewed.

Weak: "The model does the clustering."

## Build it yourself

Extend the pipeline into a versioned theme tool with drift detection:

1. Wrap embed, cluster, label, and count into a `discover_themes(texts, k)` function that returns a taxonomy dict with a `version` string (hash of the centroids) and `examples` per theme.
2. Add `classify_batch(vectors, taxonomy)` that returns counts per label plus an `uncategorized` count using the similarity floor.
3. Simulate a new month: add 8 tickets from a fourth theme ("mobile app crashes") with their own direction. Classify against the frozen taxonomy and assert the uncategorized rate exceeds 25%; then re-cluster with k=4 and assert purity returns to 1.0.
4. Implement mean-centering: subtract the corpus mean before normalizing, and show that cross-theme cosine drops toward zero while same-theme stays high (anisotropy correction).
5. Implement over-cluster-and-merge: run k=6, then merge clusters whose centroids have cosine above 0.9; assert exactly 3 clusters remain on the original data.

Acceptance checks:

- `discover_themes` is deterministic for a fixed seed: two runs produce the same `version`.
- The fourth-theme month yields `uncategorized >= 2` of the 8 new tickets before re-clustering and 0 after.
- After centering, `mean cross-theme cosine < 0.1` while `mean same-theme cosine > 0.5`.
- Over-cluster-and-merge on the original 17 tickets ends with 3 clusters and purity 1.0.
- Counts across all labels plus uncategorized always sum to the number of tickets.

## Key takeaways

- Embeddings encode meaning as direction; normalize once and cosine, dot product, and euclidean ranking coincide, making search one matrix multiply.
- Use the metric and prefixes the embedding model was trained with, and never compare vectors from different models.
- k-means with k-means++ recovers clean themes; choose k by elbow plus model labels, over-cluster and merge, and keep an explicit uncategorized bucket.
- The model names and explains clusters; arithmetic on assignments produces the counts. Keep that split for cost, determinism, and testability.
- Freeze centroids and labels as a versioned taxonomy so counts are comparable over time, and re-cluster deliberately when the uncategorized rate climbs.
- Embeddings mislead on negation, exact identifiers, surface form, anisotropy, and hubness; pair them with lexical signals, calibrated thresholds, and reranking.
- Embedding-model version is part of your data schema; an upgrade is a reindex, not a config change.
