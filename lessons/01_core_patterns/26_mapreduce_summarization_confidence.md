# 26. Map-Reduce Summarization and Confidence

*Track A (agent engineering). Book: course addition (builds on Chapter 3, Parallelization, and Chapter 19, Evaluation and Monitoring). Prerequisites: 03. Parallelization; 13. Human in the Loop; 14. Knowledge Retrieval (RAG).*

"Summarize these 400 documents" sounds like one prompt. It is actually a distributed aggregation problem with a trust problem attached. Every Monday the ops agent has to read last week's escalations from Slack, Notion incident notes, and support tickets, and tell leadership what is going wrong, how sure it is, and which items need a human before anything is published. This lesson builds map-reduce summarization from scratch, shows why the reduce belongs in code rather than in a model, turns "how confident are you?" into a number derived from evidence and then calibrated against history, and decides when to escalate. Then it rebuilds the pipeline in LangGraph with `Send`, reducers, a collapse loop, and an `interrupt()` for review.

## The problem

The request from the VP of Support:

> "What were the top customer issues last week, and how sure are we?"

The naive approach concatenates everything and asks once:

```text
[ticket 1][slack 2][notion 3] ... [ticket 400]  ──►  model  ──►  "Top issues: billing (87% confidence) ..."
```

It fails in five distinct ways:

```text
1. Size          400 docs do not fit, or fit and cost a fortune per run
2. Position      evidence in the middle of a long prompt is used less well than
                 evidence at the ends ("Lost in the Middle", Liu et al., 2023)
3. Counting      "billing: 12 customers" is a number the model estimated, not counted
4. Provenance    no way to check which documents support which claim
5. Confidence    "87%" is text the model generated; nothing measured it
```

Points 1 and 2 are why map-reduce exists. Points 3 to 5 are why most map-reduce implementations are still wrong: they fix the size problem and keep the trust problems.

The map-reduce shape:

```text
            ┌── map(doc 1) ──► {theme, quote, severity, doc_id}
            ├── map(doc 2) ──► {theme, quote, severity, doc_id}
docs ───────┼── ...                                                ──► reduce ──► digest
            └── map(doc N) ──► {theme, quote, severity, doc_id}
```

**Map** is a per-document extraction: small input, small structured output, independent of every other document, so it parallelizes and can use a cheap model. **Reduce** combines extractions into conclusions. If the extractions themselves do not fit one reduce call, you reduce in levels, which is **hierarchical summarization**:

```text
level 0:   e1 e2 e3 e4 | e5 e6 e7 e8 | e9 e10 e11 e12
level 1:      P1       |     P2      |      P3
level 2:                 FINAL
```

The central design decision of this lesson: **what is the reduce operation?** If it is "a model summarizes the summaries," every level loses information and invents some. Counts drift. A customer mentioned in two tickets becomes two customers. A quote gets paraphrased and then attributed. And the result depends on how you batched the documents, because summarizing is not associative:

```text
summarize(summarize(A, B), C)  ≠  summarize(A, summarize(B, C))
```

If the reduce is a **code merge of structured records** (union sets of customer ids, sum counts, keep exemplar quotes with their source ids), it is associative and commutative, so batching and ordering cannot change the answer. The model's job shrinks to two places where it is actually good: extracting from one document, and writing prose over a final table it did not compute.

### What "confidence" should mean

A model asked for a confidence score will give one. Published studies of verbalized confidence have often found it poorly calibrated, and more fundamentally, the number is not tied to anything you can audit. For a digest, confidence should be a function of evidence:

```text
support        how many DISTINCT customers mention it (not how many messages)
agreement      how consistently the extractor labels the same document (self-consistency)
verification   what fraction of extracted quotes actually appear in the source
```

That gives a raw score. A raw score is still not a probability. **Calibration** maps it to "claims with this score were confirmed by a human reviewer X% of the time," which is what lets you set an escalation threshold that means something. Guo et al., "On Calibration of Modern Neural Networks" (2017), is the standard reference for why raw model scores need this step and for the expected calibration error (ECE) metric used below.

## Build it without a framework

The corpus is twelve documents so the output fits on screen. The map step is a lookup keyed by document id, standing in for one small-model call per document with a structured-output schema. That is deliberate: a map worker must be a **pure function of its one input**. A scripted model that replays turns in order would be wrong here, because parallel workers call it in nondeterministic order, and that is itself the lesson. The scripted model (copied below, per course convention) is used only for the one sequential call at the end: writing the prose. A live cell shows real extraction.

Three samples per document stand in for self-consistency sampling at nonzero temperature. Document `d08` contains an invented quote, and `d06` and `d07` have extractions that disagree across samples.

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


from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# One week of escalations: (customer, source, text).
DOCS = {
    "d01": ("acme",     "slack",  "We were charged twice for the September invoice. Please fix."),
    "d02": ("globex",   "ticket", "Invoice INV-88 shows a duplicate line item for seats."),
    "d03": ("initech",  "ticket", "Our card was billed two times on the 3rd."),
    "d04": ("acme",     "slack",  "Following up: still seeing the double charge on our card."),
    "d05": ("umbrella", "notion", "SSO login fails with a SAML assertion error for all admins."),
    "d06": ("hooli",    "ticket", "Some users cannot sign in through Okta since Tuesday."),
    "d07": ("stark",    "slack",  "Webhooks for payment events arrive about 20 minutes late."),
    "d08": ("wayne",    "ticket", "CSV export spins forever on large workspaces."),
    "d09": ("wonka",    "ticket", "Export to CSV times out after 60 seconds."),
    "d10": ("pied",     "slack",  "Just wanted to say the new dashboard is great."),
    "d11": ("umbrella", "slack",  "SSO still broken, our whole company is locked out."),
    "d12": ("hooli",    "ticket", "We got two identical invoices this month."),
}

# Three extraction samples per doc: (theme, quote, severity 1-3).
MAP_SAMPLES = {
    "d01": [("invoice_duplicate", "charged twice", 2)] * 3,
    "d02": [("invoice_duplicate", "duplicate line item", 2)] * 3,
    "d03": [("invoice_duplicate", "billed two times", 2)] * 3,
    "d04": [("invoice_duplicate", "double charge", 2)] * 3,
    "d05": [("sso_login_failure", "SSO login fails", 3)] * 3,
    "d06": [("sso_login_failure", "cannot sign in through Okta", 3)] * 2
           + [("account_permissions", "cannot sign in", 2)],
    "d07": [("webhook_delay", "arrive about 20 minutes late", 2)] * 2
           + [("payments_outage", "payment events", 3)],
    "d08": [("export_timeout", "export fails with error 504", 1)] * 3,   # not in the source
    "d09": [("export_timeout", "times out after 60 seconds", 1)] * 3,
    "d10": [(None, "", 0)] * 3,
    "d11": [("sso_login_failure", "locked out", 3)] * 3,
    "d12": [("invoice_duplicate", "two identical invoices", 2)] * 3,
}


def map_doc(doc_id):
    """Map: majority extraction, its agreement rate, and a quote check against the source."""
    samples = MAP_SAMPLES[doc_id]
    (theme, quote, severity), votes = Counter(samples).most_common(1)[0]
    customer, source, text = DOCS[doc_id]
    return {"doc_id": doc_id, "customer": customer, "source": source, "theme": theme,
            "quote": quote, "severity": severity, "agreement": votes / len(samples),
            "verified": quote.lower() in text.lower()}


with ThreadPoolExecutor(max_workers=4) as pool:
    extractions = list(pool.map(map_doc, sorted(DOCS)))

for e in extractions:
    print(e["doc_id"], e["theme"], f"agree={e['agreement']:.2f}", "ok" if e["verified"] else "REJECTED")
assert [e["doc_id"] for e in extractions if not e["verified"] and e["theme"]] == ["d08"]
assert next(e for e in extractions if e["doc_id"] == "d06")["agreement"] < 1
```

The quote check is one line and catches the most damaging failure in summarization: a vivid, specific, invented quote that ends up on a slide in front of an executive. Substring matching is crude (it misses legitimate whitespace differences), and production versions normalize text or use fuzzy span matching, but the principle is fixed: **a displayed quote must be located in its source, by code, before it is displayed.**

### The reduce, in code

Each extraction becomes a one-row table. Merging two tables unions customer sets and document-id sets, concatenates agreement values, takes the max severity, and keeps a few exemplar quotes. Rejected extractions count as rejected and contribute nothing else.

```{code-cell} python
def table_of(e):
    if e["theme"] is None:
        return {}
    ok = e["verified"]
    return {e["theme"]: {"customers": {e["customer"]} if ok else set(),
                         "doc_ids": {e["doc_id"]} if ok else set(),
                         "agreements": [e["agreement"]] if ok else [],
                         "severity": e["severity"] if ok else 0,
                         "quotes": [(e["doc_id"], e["quote"])] if ok else [],
                         "rejected": 0 if ok else 1}}


def merge_tables(a, b):
    """Associative and commutative, so the shape of the reduce tree cannot matter."""
    out = {}
    for theme in sorted(a.keys() | b.keys()):
        x, y = a.get(theme), b.get(theme)
        if x is None or y is None:
            out[theme] = dict(x or y)
            continue
        out[theme] = {
            "customers": x["customers"] | y["customers"],
            "doc_ids": x["doc_ids"] | y["doc_ids"],
            "agreements": sorted(x["agreements"] + y["agreements"]),
            "severity": max(x["severity"], y["severity"]),
            "quotes": sorted(x["quotes"] + y["quotes"])[:3],     # exemplars; ids keep the rest
            "rejected": x["rejected"] + y["rejected"],
        }
    return out


def fold(batch):
    acc = {}
    for t in batch:
        acc = merge_tables(acc, t)
    return acc


def hierarchical_reduce(tables, batch_size=4):
    """Reduce in levels, as a context-limited reduce would have to."""
    levels = 0
    while len(tables) > 1:
        tables = [fold(tables[i:i + batch_size]) for i in range(0, len(tables), batch_size)]
        levels += 1
    return (tables[0] if tables else {}), levels


table, levels = hierarchical_reduce([table_of(e) for e in extractions])
flat = fold([table_of(e) for e in reversed(extractions)])      # other order, one level

for theme, row in sorted(table.items(), key=lambda kv: -len(kv[1]["customers"])):
    print(f"{theme:18s} customers={len(row['customers'])} mentions={len(row['doc_ids'])} "
          f"rejected={row['rejected']} severity={row['severity']}")
print("reduce levels:", levels)
assert table == flat                                       # batching and order do not matter
assert len(table["invoice_duplicate"]["customers"]) == 4   # acme wrote twice, counts once
assert len(table["invoice_duplicate"]["doc_ids"]) == 5
assert table["export_timeout"]["rejected"] == 1
assert levels == 2
```

`assert table == flat` is the property that makes hierarchical reduction safe. A two-level tree and a one-level fold over the reversed list produce identical tables. Try the same experiment with an LLM summarizing summaries and you will get different theme names, different counts, and different quotes on every batching.

Note the distinction the table keeps: **mentions** (5 documents about duplicate invoices) versus **support** (4 distinct customers, because Acme posted twice). Prevalence claims use support. A single angry customer who posts ten times is one data point with high severity, not ten votes.

### From evidence to a raw score

```{code-cell} python
import numpy as np


def raw_confidence(row, n_extracted):
    """Evidence-derived score in [0, 1]. Not a number the model made up."""
    support = min(len(row["customers"]) / 3, 1.0)                  # saturates at 3 customers
    agreement = float(np.mean(row["agreements"])) if row["agreements"] else 0.0
    verified_rate = len(row["doc_ids"]) / n_extracted if n_extracted else 0.0
    return round(0.5 * support + 0.3 * agreement + 0.2 * verified_rate, 3)


n_extracted = Counter(e["theme"] for e in extractions if e["theme"])
raw = {t: raw_confidence(r, n_extracted[t]) for t, r in sorted(table.items())}
print(raw)

# 600 past theme claims with their raw score and whether a reviewer confirmed them.
# Synthetic, but shaped like reality: the raw score is systematically overconfident.
rng = np.random.default_rng(7)
hist_raw = rng.uniform(0.2, 1.0, 600)
hist_ok = rng.random(600) < hist_raw ** 2
train, test = slice(0, 400), slice(400, 600)


def ece(scores, labels, bins=10):
    """Expected calibration error: bin-weighted gap between confidence and accuracy."""
    idx = np.minimum((scores * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(scores[idx == b].mean() - labels[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def fit_histogram_binning(scores, labels, bins=10):
    idx = np.minimum((scores * bins).astype(int), bins - 1)
    rates = np.array([labels[idx == b].mean() if (idx == b).any() else (b + 0.5) / bins
                      for b in range(bins)])
    rates = np.maximum.accumulate(rates)                 # higher raw score never maps lower
    return lambda s: rates[np.minimum((np.asarray(s) * bins).astype(int), bins - 1)]


calibrate = fit_histogram_binning(hist_raw[train], hist_ok[train])
before, after = ece(hist_raw[test], hist_ok[test]), ece(calibrate(hist_raw[test]), hist_ok[test])
calibrated = {t: float(calibrate(s)) for t, s in raw.items()}
print(f"held-out ECE: raw {before:.3f} -> calibrated {after:.3f}")
print({t: round(c, 2) for t, c in calibrated.items()})
assert after < before / 2
assert raw["invoice_duplicate"] > raw["sso_login_failure"] > raw["webhook_delay"]
```

The raw score for SSO failures is 0.80. Historically, claims scored around 0.8 were confirmed about 75% of the time, so the calibrated value is lower. Nothing about the formula changed; we learned what its numbers mean. Histogram binning is the simplest calibrator; Platt scaling and isotonic regression are the usual alternatives. All of them need labels, and the labels come from the review queue you are about to build. That loop, where escalations produce the data that calibrates the next threshold, is the real system.

The weights (0.5, 0.3, 0.2) are hand-set here. With enough labelled history you fit them too, typically with logistic regression on the same features, which gives you a calibrated score directly.

### Escalation: a threshold with a meaning

An escalation threshold should come from a stated requirement, not from taste. Requirement: "claims we auto-publish must be right at least 90% of the time." On held-out history, pick the lowest calibrated threshold that meets it, and read off the coverage, which is the fraction of claims that skip review. Coverage is the cost side of the trade.

```{code-cell} python
def pick_threshold(scores, labels, target_precision=0.9, min_count=20):
    """Lowest threshold whose auto-published claims meet the precision target."""
    for t in np.unique(scores):
        m = scores >= t
        if m.sum() >= min_count and labels[m].mean() >= target_precision:
            return float(t), float(m.mean())
    return 1.01, 0.0                       # nothing qualifies: everything goes to review


threshold, coverage = pick_threshold(calibrate(hist_raw[test]), hist_ok[test])
print(f"threshold {threshold:.2f}: {coverage:.0%} of claims auto-publish at >=90% precision")


def route(theme):
    row, c = table[theme], calibrated[theme]
    if c >= threshold:
        return "publish"
    if row["severity"] >= 3:
        return "escalate"                  # uncertain and high stakes: a human decides
    if len(row["customers"]) <= 1:
        return "emerging"                  # publish, labelled as a low-n signal
    return "escalate"


decisions = {t: route(t) for t in sorted(table)}
print(decisions)
assert decisions == {"export_timeout": "emerging", "invoice_duplicate": "publish",
                     "sso_login_failure": "escalate", "webhook_delay": "emerging"}
```

Three outcomes, not two. **Publish** when calibrated confidence clears the bar. **Escalate** when confidence is below the bar and being wrong is expensive (severity 3: a company locked out of SSO). **Emerging** for single-customer signals that are low stakes: they are published, but labelled as "1 customer, unconfirmed," which is honest and costs no reviewer time. The routing rule combines probability of being wrong with cost of being wrong, which is the whole theory of escalation in one function.

Only 10% of historical claims clear the 90% bar. That is the calibration talking: the raw scorer is overconfident, so the honest threshold is high. The fix for low coverage is a better scorer (more features, fitted weights), not a lower bar.

## How LangGraph does it

The from-scratch pipeline had five parts: a thread-pool map, a list the workers fill, a level-by-level reduce loop, scoring, and a review step. In LangGraph:

```text
START ──fan_out──► map_doc × N  (one Send per document)
                        │  extractions: Annotated[list, operator.add]
                        ▼
                  start_reduce ──► collapse ⟲ (while partials > 1)
                        │                │
                        └──────┬─────────┘
                               ▼
                             score ──► human_gate (interrupt) ──► write_report ──► END
```

```{code-cell} python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage


class DigestState(TypedDict):
    doc_ids: list[str]
    extractions: Annotated[list[dict], operator.add]    # fan-in: every map worker appends
    partials: list[dict]                                # replace: each collapse level overwrites
    decisions: dict
    reviewed: dict
    report: str


def fan_out(state: DigestState):
    """One Send per document. Each worker's state is only its own doc id."""
    return [Send("map_doc", {"doc_id": d}) for d in state["doc_ids"]]


def map_node(payload: dict):
    return {"extractions": [map_doc(payload["doc_id"])]}


def start_reduce(state: DigestState):
    ordered = sorted(state["extractions"], key=lambda e: e["doc_id"])   # never trust arrival order
    return {"partials": [table_of(e) for e in ordered]}


def collapse(state: DigestState):
    p = state["partials"]
    return {"partials": [fold(p[i:i + 4]) for i in range(0, len(p), 4)]}


def needs_collapse(state: DigestState):
    return "collapse" if len(state["partials"]) > 1 else "score"


def score(state: DigestState):
    final_table = state["partials"][0]
    n = Counter(e["theme"] for e in state["extractions"] if e["theme"])
    return {"decisions": {
        theme: {"confidence": round(float(calibrate(raw_confidence(row, n[theme]))), 2),
                "route": route(theme), "customers": len(row["customers"])}
        for theme, row in final_table.items()}}


def human_gate(state: DigestState):
    pending = {t: d for t, d in state["decisions"].items() if d["route"] == "escalate"}
    if not pending:
        return {"reviewed": {}}
    # Persists state and pauses. On resume the node re-runs from the top, which is
    # safe because everything before interrupt() is pure.
    return {"reviewed": interrupt({"review": pending})}


def write_report(state: DigestState):
    lines = [f"{t}: {d['route']} (conf {d['confidence']}, {d['customers']} customers)"
             for t, d in sorted(state["decisions"].items())]
    lines += [f"human: {t} -> {v}" for t, v in state["reviewed"].items()]
    prose = writer.invoke([HumanMessage("\n".join(lines))]).content   # prose over a fixed table
    return {"report": prose + "\n" + "\n".join(lines)}


b = StateGraph(DigestState)
for name, fn in [("map_doc", map_node), ("start_reduce", start_reduce), ("collapse", collapse),
                 ("score", score), ("human_gate", human_gate), ("write_report", write_report)]:
    b.add_node(name, fn)
b.add_conditional_edges(START, fan_out, ["map_doc"])
b.add_edge("map_doc", "start_reduce")
b.add_conditional_edges("start_reduce", needs_collapse, ["collapse", "score"])
b.add_conditional_edges("collapse", needs_collapse, ["collapse", "score"])
b.add_edge("score", "human_gate")
b.add_edge("human_gate", "write_report")
b.add_edge("write_report", END)
digest = b.compile(checkpointer=InMemorySaver())

writer = scripted("Weekly escalations: duplicate invoicing confirmed across 4 customers; "
                  "SSO failures confirmed by on-call; two low-n signals to watch.")
cfg = {"configurable": {"thread_id": "digest-2026-w39"}}
first = digest.invoke({"doc_ids": sorted(DOCS), "extractions": []}, cfg)
print("paused for review:", first["__interrupt__"][0].value)
final = digest.invoke(Command(resume={"sso_login_failure": "confirmed, page identity team"}), cfg)
print(final["report"])
assert list(first["__interrupt__"][0].value["review"]) == ["sso_login_failure"]
assert len(final["extractions"]) == 12
assert final["decisions"]["invoice_duplicate"]["route"] == "publish"
assert "human: sso_login_failure -> confirmed" in final["report"]
```

What each primitive is doing, in terms of the from-scratch code:

| From scratch | LangGraph |
|---|---|
| `pool.map(map_doc, docs)` | `fan_out` returning one `Send("map_doc", {...})` per doc; all run in one super-step |
| workers returning into `extractions` | the `operator.add` reducer on `extractions`; `map_doc` returns a one-item list |
| `while len(tables) > 1` | `collapse` with a conditional edge back to itself |
| `tables = [...]` each level | `partials` with no reducer: each level replaces the previous |
| a reviewer looking at a list | `interrupt()` persisting state; `Command(resume=...)` continuing the same thread |

Two details matter. First, `start_reduce` sorts extractions by id. Reducer append order follows completion order, which is not something to rely on; our reduce is order-independent anyway, but the report's exemplar quotes and any model call downstream should see a stable order. Second, the interrupt makes the review durable. The digest can sit waiting for the on-call lead for a day; the checkpointer holds the extractions, so resuming does not re-run 400 map calls.

The live cell replaces the lookup with a real extractor. Structured output enforces the record shape; the quote check stays in code because structured output guarantees the JSON is well-formed, not that the quote is real.

```{code-cell} python live
import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


class Extraction(BaseModel):
    theme: Optional[Literal["invoice_duplicate", "sso_login_failure", "webhook_delay",
                            "export_timeout", "other"]]
    quote: str = Field(description="exact span copied from the document, or empty")
    severity: int = Field(ge=0, le=3)


extractor = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"),
                            temperature=0.7).with_structured_output(Extraction)


def live_map_doc(doc_id, k=3):
    customer, source, text = DOCS[doc_id]
    samples = [extractor.invoke(f"Classify this support message and copy one exact supporting "
                                f"span.\n\n{text}") for _ in range(k)]
    (theme, quote, sev), votes = Counter((s.theme, s.quote, s.severity) for s in samples).most_common(1)[0]
    return {"doc_id": doc_id, "theme": theme, "quote": quote, "agreement": votes / k,
            "verified": bool(quote) and quote.lower() in text.lower()}


for d in ["d01", "d06", "d08"]:
    print(live_map_doc(d))
```

A note on the fixed theme list. Letting the map step invent theme names ("double billing", "duplicate charge", "billed twice") guarantees the reduce will see three themes for one problem. Either give the extractor a closed taxonomy, as here, or add a normalization step (embedding clustering, see 25. Embeddings, Clustering, and Themes) before the reduce.

## What goes wrong in production

1. **Counts that drift through the tree.** Symptom: the weekly digest says 31 customers; a SQL query over the same tickets says 22. Cause: the reduce is a model summarizing summaries, so every level re-estimates counts. Fix: carry ids through every level and count in code; the model writes prose over a table it did not compute.

2. **Mentions reported as support.** Symptom: an issue from one very loud customer tops the digest. Cause: counting documents rather than distinct customers. Fix: dedupe by customer (or account) before counting, report both mentions and support, and keep severity as a separate axis from prevalence.

3. **Invented quotes.** Symptom: a quote on a leadership slide that no customer ever wrote. Cause: extraction output displayed without checking. Fix: locate every displayed span in its source by code; drop and count the ones that fail. A rising rejection rate is also an early signal that the extractor or its prompt regressed.

4. **Theme fragmentation.** Symptom: "SSO issue," "login failure," and "Okta problem" appear as three small themes, none clearing the confidence bar. Cause: open-vocabulary extraction. Fix: a closed taxonomy with an `other` bucket, plus periodic review of `other` to extend the taxonomy.

5. **Map fan-out overload.** Symptom: 429s from the model provider on Monday morning. Cause: 400 simultaneous map calls. Fix: bound concurrency (`max_concurrency` in the run config, a semaphore per provider), retry with backoff, and let a failed map produce a "missing" record rather than blocking the reduce. The digest reports what it could not read (03. Parallelization).

6. **An uncalibrated threshold.** Symptom: "confidence > 0.8" auto-publishes claims that reviewers reject a third of the time. Cause: the threshold was picked by taste on a raw score. Fix: calibrate against reviewer labels, choose the threshold from a precision target, and recalibrate on a schedule, because model or prompt changes shift the score distribution.

7. **Review queues that nobody drains.** Symptom: escalations pile up and the digest ships late or ships without them. Cause: escalation designed without a service level. Fix: a deadline per escalation with a default action (publish as "unconfirmed" or withhold), a named owner, and escalation volume tracked as a metric. If most items escalate, fix the scorer, not the reviewers.

## What changed by 2026

- **Long context did not kill map-reduce.** Windows grew, but position effects, cost per run, and the counting and provenance problems did not go away. Map-reduce is now chosen for trust and cost, not only for size: per-document extraction is auditable and parallel, and a cheap model can do it.
- **Structured output is the default map contract.** With schema-constrained output widely available (24. Structured Output), the map step returns records, not prose, which is what makes a code reduce possible. It guarantees shape, not truth, so span verification stays.
- **Map-reduce as a runtime primitive.** `Send` plus reducers make dynamic fan-out a graph edge with checkpointing, so a crashed run resumes with completed map results intact. LangGraph's how-to material has long used summarization as its worked map-reduce example.
- **Confidence moved from prompts to evaluation.** Asking the model for a percentage has given way to evidence features, self-consistency, and calibration against reviewer labels, the same eval-driven loop as 19. Evaluation and Monitoring. Reasoning traces are not used as evidence of correctness; outcomes are verified.
- **Escalation as a designed interface.** Durable interrupts made "pause and ask a human" cheap to build, which moved the hard part to policy: thresholds tied to precision targets, severity-aware routing, and review queues with owners and deadlines.

## Interview questions

**Q1. How would you summarize 10,000 support tickets into a weekly report?**

Strong: Map-reduce. The map step runs one small-model call per ticket with a structured-output schema: theme from a closed taxonomy, a verbatim quote, severity, and the customer id. I verify each quote against its source in code. The reduce is a code merge: union customer ids per theme, count distinct customers and mentions, keep a few exemplar quotes. Because that merge is associative I can do it in levels without the result depending on batching. Only at the end does a model write prose over the final table. Then each theme gets an evidence-based confidence and a route: publish, label as emerging, or escalate.

Weak: "Chunk the tickets, summarize each chunk, then summarize the summaries."

**Q2. Why is "summarize the summaries" risky?**

Strong: Summarization is lossy and not associative, so the result depends on how you batched the inputs, and every level can drop or invent details. Counts are the worst case: each level re-estimates them, and customers mentioned in several documents get double-counted. Quotes get paraphrased and then treated as verbatim. The fix is to make intermediate results structured records with ids and do the merge in code, so the tree shape cannot change the answer, and to use the model only at the leaves and for final prose.

Weak: "It might lose some information."

**Q3. How do you produce a confidence score for a generated claim?**

Strong: Not by asking the model for a percentage; verbalized confidence tends to be poorly calibrated and is not tied to anything auditable. I compute features from evidence: distinct supporting sources, agreement across several sampled extractions, and the fraction of quotes that verified against the source. That gives a raw score. Then I calibrate it against human review outcomes with histogram binning, isotonic regression, or Platt scaling, and I track expected calibration error on held-out data. After calibration, 0.8 means claims like this were confirmed about 80% of the time.

Weak: "Ask the model how confident it is."

**Q4. What is calibration and how do you measure it?**

Strong: A score is calibrated if, among all predictions with score p, the fraction that are correct is about p. I measure it with a reliability table: bin predictions by score, compare mean score to observed accuracy in each bin, and summarize with expected calibration error, the bin-weighted average gap. Calibration is separate from discrimination: a scorer can rank claims well and still be overconfident. Recalibration fixes the mapping without changing the ranking, and it needs periodic refitting because model and prompt changes shift the score distribution.

Weak: "Calibration means the model is accurate."

**Q5. How do you decide when an agent should escalate to a human?**

Strong: Two inputs: the probability of being wrong, which is the calibrated confidence, and the cost of being wrong, which is severity or blast radius. I pick the auto-action threshold from a stated requirement, such as 90% precision on held-out labelled data, and read off the coverage to see what it costs in review volume. High-severity items below the bar escalate; low-stakes low-support items can be published with an explicit "emerging, one customer" label. Every escalation has an owner and a deadline with a default action, and reviewer decisions feed back as calibration labels.

Weak: "When confidence is below 0.7."

**Q6. How does map-reduce work in LangGraph?**

Strong: A conditional edge returns a list of `Send(node, payload)` objects, one per document, and the runtime executes that node once per Send, in parallel, each with only its payload as state. The results land in a state key with an `operator.add` reducer, which is the fan-in. For hierarchical reduction I add a collapse node with a conditional edge back to itself while more than one partial remains. With a checkpointer the completed map results survive a crash, and an `interrupt()` before publishing makes the human review step durable.

Weak: "You use Send for parallelism."

**Q7. One customer filed ten tickets about the same bug. How does that show up in your digest?**

Strong: As one customer with ten mentions, not ten customers. I count support as distinct customers or accounts and report mentions separately, because prevalence and intensity are different signals. Severity is a third axis: one customer locked out entirely can outrank a mild annoyance reported by many. Mixing these is how digests end up driven by whoever complains loudest.

Weak: "It would be the top issue."

## Build it yourself

Replace the hand-set weights with a fitted scorer and add a missing-data path:

1. Generate a labelled history where each claim has the three features (support, agreement, verified rate) and a reviewer label, deterministically with `np.random.default_rng`.
2. Fit logistic regression with numpy (a few hundred steps of gradient descent is enough) and use its output as the calibrated score.
3. Make `map_doc` raise for one document, catch the error in the map node, and emit `{"doc_id": ..., "theme": None, "missing": True}`. The report must list missing documents.
4. Add a severity-weighted threshold: severity-3 themes need 95% precision to auto-publish; others need 90%.

Acceptance checks:

- The fitted scorer's held-out ECE is lower than the raw hand-weighted score's ECE.
- The fitted weight on `support` is positive and larger than the weight on `verified_rate` in your synthetic data (by construction).
- A failing map worker does not fail the graph; the final report contains `missing: d0X`.
- `table == flat` still holds when the missing document is included.
- With the stricter severity-3 threshold, `sso_login_failure` still escalates and `invoice_duplicate` still publishes.

## Key takeaways

- Map-reduce summarization is a distributed aggregation problem: extract per document in parallel, then combine.
- Put the reduce in code over structured records so it is associative, and use the model only for extraction and final prose.
- Verify every displayed quote against its source by code before it is shown to anyone.
- Count distinct customers for support, report mentions separately, and keep severity as its own axis.
- Confidence should come from evidence features and be calibrated against reviewer labels, not asked of the model.
- Escalation combines calibrated probability of error with cost of error, and every escalation needs an owner and a deadline.
- In LangGraph, `Send` is the map, an `operator.add` reducer is the fan-in, a self-looping collapse node is hierarchical reduction, and `interrupt()` is the durable review.
