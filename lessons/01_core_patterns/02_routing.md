# 02. Routing

*Track A (agent engineering). Book: Chapter 2 (Routing). Prerequisites: 00. What an agent actually is; 01. Prompt Chaining.*

A chain has one path. The moment the right path depends on the input, you need a decision, and the component that makes it is a router. This lesson builds four kinds of router (rules, classifier, LLM, embedding), shows how to handle the case where the router is not sure, rebuilds the same decision as a LangGraph conditional edge and then as a `Command`, and finally explains why, with a capable model, most routing quietly turns into tool selection.

## The problem

The support agent's inbox is not one kind of request. In one hour it receives:

```text
"Acme's September invoice is wrong"                 → billing
"Export to CSV crashes with a 500"                  → technical
"Can you add three seats to our plan?"              → account changes (side effects!)
"Is the API rate limit per key or per org?"         → docs lookup
"hey are you a real person"                         → human handoff
"Remind me what we discussed yesterday"            → ...ambiguous
```

The naive design is one prompt with one giant set of instructions covering all of these. Lesson 01 explained why that degrades: instruction dilution. The next naive design is a chain, but a chain has one path and these requests need five.

```text
                    one prompt for everything
request ──────────► "handle billing, technical, account, docs, handoff..." ──► reply
                                    │
                                    └─ same tools, same tone, same risk controls for
                                       "what's the rate limit" and "add three seats"
```

That last line is the real problem, and it is not about quality. "Add three seats" changes what the customer pays. "What is the rate limit" does not. They need different tools, different approval gates, different logging, and probably different models. A single handler cannot give them different controls because it cannot tell them apart until it is already running.

Routing separates *deciding what kind of request this is* from *handling it*:

```text
request
   │
   ▼
┌────────┐   billing     ┌──────────────────┐
│ router │──────────────►│ billing handler  │  Stripe read tools, refund needs approval
│        │   technical   ├──────────────────┤
│        │──────────────►│ tech handler     │  logs, SQL, runbooks
│        │   account     ├──────────────────┤
│        │──────────────►│ account handler  │  CRM write tools, always approval-gated
│        │   docs        ├──────────────────┤
│        │──────────────►│ docs handler     │  retrieval only, cheap model
│        │   unsure      ├──────────────────┤
│        │──────────────►│ clarify / human  │  ask a question, never act
└────────┘               └──────────────────┘
```

Each handler is now a small, testable thing with its own prompt, tools, model, and controls. The book frames routing as the pattern that gives a workflow dynamic behavior: the path is chosen at runtime, from a fixed menu the developer wrote. That is still a workflow. The menu is fixed; only the choice is dynamic.

Two design questions follow. **How is the decision made?** And **what happens when the router is wrong or unsure?** Wrong is inevitable, so the second question matters more than the first.

## Build it without a framework

We need the scripted model for the LLM router.

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

### Four ways to decide

Every router has the same signature: `text -> (route, confidence)`. What differs is cost, latency, accuracy, and how it fails.

| Router | How it decides | Cost / latency | Strength | Weakness |
|---|---|---|---|---|
| Rules | keyword / regex match | free, microseconds | deterministic, auditable, unit-testable | brittle, misses paraphrase, needs maintenance |
| Classifier | small trained model, or nearest-centroid over embeddings | cheap, milliseconds | fast, calibrated confidence, cheap at scale | needs labeled data, fixed label set |
| Embedding similarity | cosine to example queries per route | one embedding call | handles paraphrase, no training loop | confidence is a similarity, not a probability |
| LLM | prompt the model for a label + reason | one model call, hundreds of ms | handles nuance and new phrasing, can explain | slowest, costliest, may be overconfident |

The stack in production is usually **rules first, then a cheap classifier, then the LLM only for the hard remainder.** Rules catch "refund" and "invoice" for free. The LLM sees the 10% that survived.

```{code-cell} python
import json
import re
import numpy as np

ROUTES = ("billing", "technical", "account", "docs", "human")


def rules_router(text):
    """Deterministic and auditable. Confidence 1.0 when it fires, 0.0 when it does not."""
    t = text.lower()
    if re.search(r"\b(invoice|refund|charge|billing)\b", t):
        return "billing", 1.0
    if re.search(r"\b(crash|error|500|stack trace|bug)\b", t):
        return "technical", 1.0
    if re.search(r"\b(add|remove|upgrade|downgrade)\b.*\b(seats?|plan)\b", t):
        return "account", 1.0
    return None, 0.0


# A tiny bag-of-words "embedding" so the mechanism runs offline. A real system
# would call an embedding model; the cosine + centroid logic is identical.
VOCAB = sorted({w for w in (
    "invoice refund charge paid wrong amount "
    "crash error export failing 500 slow "
    "add seats plan upgrade downgrade cancel "
    "rate limit api docs how does work pagination "
    "person human agent talk someone").split()})
EXAMPLES = {
    "billing":   ["invoice amount wrong", "refund charge", "paid twice"],
    "technical": ["export crash error", "500 failing", "slow export"],
    "account":   ["add seats plan", "upgrade plan", "cancel plan"],
    "docs":      ["how does rate limit work", "api pagination docs", "does api work"],
    "human":     ["talk person", "human agent", "someone talk"],
}


def embed(text):
    v = np.array([1.0 if w in text.lower().split() else 0.0 for w in VOCAB])
    n = np.linalg.norm(v)
    return v / n if n else v


CENTROIDS = {r: np.mean([embed(e) for e in ex], axis=0) for r, ex in EXAMPLES.items()}


def embedding_router(text):
    """Nearest centroid. Confidence is the margin between best and second best."""
    q = embed(text)
    scores = sorted(((float(q @ c), r) for r, c in CENTROIDS.items()), reverse=True)
    (best, route), (second, _) = scores[0], scores[1]
    return route, round(best - second, 3)


def llm_router(model, text):
    """The model returns a label and its own confidence. Treat the confidence as advisory."""
    raw = model.invoke(f"Classify into {ROUTES}. Return JSON with route and confidence.\n\n{text}").content
    data = json.loads(raw)
    route = data["route"] if data["route"] in ROUTES else None
    return route, float(data["confidence"])


print(rules_router("Acme's September invoice is wrong"))
print(embedding_router("the export is failing again today"))
print(llm_router(scripted('{"route": "docs", "confidence": 0.92}'), "Is the rate limit per key or per org?"))
assert rules_router("Acme's September invoice is wrong") == ("billing", 1.0)
assert embedding_router("the export is failing again today")[0] == "technical"
assert rules_router("remind me what we discussed yesterday")[0] is None
```

The embedding router's confidence is a **margin**, not a probability. If the best and second-best routes score almost the same, the router does not know, and it says so with a small number. That is the property we need for the next part.

### The cascade, with a fallback when unsure

Now compose them. Each tier tries to decide; if it cannot decide confidently, the next tier gets a turn; if nothing is confident, we do not guess. We route to `clarify`, which asks a question and never touches a tool with side effects.

```{code-cell} python
def route_request(text, llm, thresholds={"embedding": 0.15, "llm": 0.7}):
    """Rules -> embedding -> LLM -> clarify. Returns the route and a trace of who decided."""
    trace = []
    route, conf = rules_router(text)
    trace.append(("rules", route, conf))
    if route:
        return route, trace

    route, conf = embedding_router(text)
    trace.append(("embedding", route, conf))
    if conf >= thresholds["embedding"]:
        return route, trace

    route, conf = llm_router(llm, text)
    trace.append(("llm", route, conf))
    if route and conf >= thresholds["llm"]:
        return route, trace

    trace.append(("fallback", "clarify", 0.0))
    return "clarify", trace


# Case 1: rules decide, nothing else runs (the LLM script is never consumed).
r1, t1 = route_request("Acme's September invoice is wrong", scripted())
# Case 2: rules miss, embedding is confident.
r2, t2 = route_request("the export is failing again today", scripted())
# Case 3: ambiguous. Rules miss, embedding margin is tiny, LLM is unsure -> clarify.
r3, t3 = route_request("remind me what we discussed yesterday",
                       scripted('{"route": "account", "confidence": 0.45}'))

for r, t in ((r1, t1), (r2, t2), (r3, t3)):
    print(r, "<-", t)
assert (r1, len(t1)) == ("billing", 1)
assert (r2, t2[-1][0]) == ("technical", "embedding")
assert r3 == "clarify" and t3[-1][0] == "fallback"
```

Case 3 is the one that matters. "Remind me what we discussed yesterday" could be a follow-up on a seat change, a memory lookup, or a human request. The LLM leaned `account` at 0.45. If we had acted on that, we might have started an approval-gated seat change the customer never asked for. Instead the system asks. **When the router is unsure, the safe route is the one with no side effects.**

The trace is not decoration either. It records which tier decided and with what confidence, which is exactly what you need for the confusion matrix later: every misroute in production becomes a labeled example, and you can see which tier produced it.

### Two representations of the same decision

Notice that the router returned a string and a dispatcher picked a handler:

```python
handlers[route](text)
```

That `dict` lookup is a conditional edge. Hold that thought.

## How LangGraph does it

In LangGraph the router is a node that writes the decision into state, and the dispatch is a **conditional edge**: a function that reads state and returns the name of the next node. That is `handlers[route]` with the graph runtime doing the lookup.

```text
from scratch                          LangGraph
------------                          ---------
route_request(text) -> route          a `router` node writing state["route"]
handlers[route](text)                 add_conditional_edges("router", pick, path_map)
"clarify" fallback                    a node with no side-effecting tools
trace list                            a state key with an append reducer
```

```{code-cell} python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


class RouteState(TypedDict, total=False):
    text: str
    route: str
    trace: Annotated[list, operator.add]     # append: we want the history of who decided
    reply: str


def make_router(llm):
    def router(state: RouteState):
        route, trace = route_request(state["text"], llm)
        return {"route": route, "trace": trace}
    return router


def pick_handler(state: RouteState):
    """The conditional edge. It only reads state; the router node already decided."""
    return state["route"]


def handler(name, tools_note):
    def node(state: RouteState):
        return {"reply": f"[{name}] handled with {tools_note}: {state['text']}"}
    return node


def clarify(state: RouteState):
    return {"reply": "Could you tell me whether you want to change your seats or review a past conversation?"}


def build(llm):
    b = StateGraph(RouteState)
    b.add_node("router", make_router(llm))
    b.add_node("billing", handler("billing", "Stripe read tools"))
    b.add_node("technical", handler("technical", "logs + SQL"))
    b.add_node("account", handler("account", "CRM writes behind approval"))
    b.add_node("docs", handler("docs", "retrieval only"))
    b.add_node("human", handler("human", "handoff"))
    b.add_node("clarify", clarify)
    b.add_edge(START, "router")
    b.add_conditional_edges("router", pick_handler,
                            ["billing", "technical", "account", "docs", "human", "clarify"])
    for n in ("billing", "technical", "account", "docs", "human", "clarify"):
        b.add_edge(n, END)
    return b.compile()


g = build(scripted('{"route": "account", "confidence": 0.45}'))
out = g.invoke({"text": "remind me what we discussed yesterday"})
print(out["route"], "->", out["reply"])
print(out["trace"])
assert out["route"] == "clarify"
assert "seats" in out["reply"]

out2 = build(scripted()).invoke({"text": "Acme's September invoice is wrong"})
print(out2["reply"])
assert out2["route"] == "billing" and len(out2["trace"]) == 1
```

```text
START ──► router ──┬──► billing   ──► END
                   ├──► technical ──► END
                   ├──► account   ──► END
                   ├──► docs      ──► END
                   ├──► human     ──► END
                   └──► clarify   ──► END
```

The `path_map` (the list passed as the third argument) matters for more than drawing. It tells the compiler every destination the edge can take, so the graph can be validated and visualized before it runs, and so a typo in a route name fails at compile time rather than at 3 a.m.

### `Command`: deciding and updating in one node

The conditional-edge version splits the decision into two functions: the node writes `route`, the edge reads it. Sometimes that is awkward, especially when the decision and a state update belong together. `Command` lets a node return both:

```{code-cell} python
from langgraph.types import Command
from typing import Literal


def router_command(state: RouteState) -> Command[Literal["billing", "technical", "account", "docs", "human", "clarify"]]:
    """Same decision, but the node names its successor directly. No separate edge function."""
    route, trace = route_request(state["text"], scripted('{"route": "account", "confidence": 0.45}'))
    return Command(update={"route": route, "trace": trace}, goto=route)


b = StateGraph(RouteState)
b.add_node("router", router_command)       # the Literal return annotation is the path_map
for n, note in (("billing", "Stripe read"), ("technical", "logs"), ("account", "CRM writes"),
                ("docs", "retrieval"), ("human", "handoff")):
    b.add_node(n, handler(n, note))
    b.add_edge(n, END)
b.add_node("clarify", clarify)
b.add_edge("clarify", END)
b.add_edge(START, "router")
g_cmd = b.compile()

out = g_cmd.invoke({"text": "remind me what we discussed yesterday"})
print(out["route"], "->", out["reply"][:60])
assert out["route"] == "clarify"
```

Underneath, both are the same thing: after the node runs, the runtime looks at what it returned and schedules the next node. The conditional edge keeps routing logic separate from work; `Command` puts them together. Use the edge when the decision is a pure function of state (easier to test, visible in the graph). Use `Command` when the node has already computed the decision as a byproduct of its work, or when a node needs to jump to a node in a *parent* graph (`Command(graph=Command.PARENT, ...)`), which edges cannot express.

### When routing becomes tool selection

Here is the shift that makes this lesson different from the book's chapter. Look at what the LLM router is doing: given a request and a menu of five options with descriptions, pick one. That is precisely what a model does when it chooses among tools. So with a capable model you can drop the router entirely and expose the handlers *as tools*:

```{code-cell} python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def billing_lookup(customer: str, period: str) -> str:
    """Read invoices for a customer and period. Use for questions about charges, invoices, refunds."""
    return f"invoice in_9F2 for {customer} {period}: duplicate seat charge $480"


@tool
def change_seats(customer: str, delta: int) -> str:
    """Change the number of seats on a plan. Side effects: requires approval before execution."""
    return f"approval requested: {customer} seats {delta:+d}"


@tool
def ask_clarifying_question(question: str) -> str:
    """Use when the request is ambiguous. Ask the user before taking any action."""
    return f"asked: {question}"


tools = [billing_lookup, change_seats, ask_clarifying_question]
by_name = {t.name: t for t in tools}

# The model *is* the router now. Its "route" is the tool it picks.
model = scripted(
    [("ask_clarifying_question", {"question": "Do you want to change seats, or review yesterday's conversation?"})],
    "I've asked the customer to clarify.",
).bind_tools(tools)

msgs = [HumanMessage("remind me what we discussed yesterday")]
ai = model.invoke(msgs)
chosen = ai.tool_calls[0]["name"]
msgs += [ai, ToolMessage(content=by_name[chosen].invoke(ai.tool_calls[0]["args"]), tool_call_id=ai.tool_calls[0]["id"])]
final = model.invoke(msgs)
print("model routed to:", chosen)
print(final.content)
assert chosen == "ask_clarifying_question"
```

```text
explicit router                          tool selection
---------------                          --------------
router prompt: "pick one of 5 labels"    tool schemas: 5 descriptions
label -> conditional edge -> handler     tool call -> ToolNode -> function
"clarify" fallback route                 ask_clarifying_question tool
confidence threshold in your code        ...nothing. The model just picks.
```

The last row is the trade. Tool selection is simpler and lets the model pick *several* actions or chain them. But you lose the explicit confidence gate. With tool selection, the fallback has to be a tool the model is instructed to prefer when unsure, and the guardrails move to the tool layer: `change_seats` requires approval whatever the model's confidence, because you cannot read its confidence.

When to keep the explicit router:

- You need different **models** per route (cheap for docs, strong for technical).
- You need different **tool sets** per route, so the account handler never even sees `refund`.
- You need **deterministic** handling for some inputs (compliance, cost).
- You need a **measurable** router with a confusion matrix, because routing errors are your top failure.

When to collapse into tools: the handlers are cheap, share a model, and the request often needs more than one of them. In 2026 many production systems do both: a rules tier in front for the free wins and for compliance-mandated paths, then a tool-selecting agent for the remainder.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
agent = create_agent(real, tools=tools,
                     system_prompt="If the request is ambiguous, call ask_clarifying_question before any other tool.")
out = agent.invoke({"messages": [HumanMessage("remind me what we discussed yesterday")]})
for m in out["messages"]:
    print(type(m).__name__, m.content or m.tool_calls)
```

## What goes wrong in production

1. **Overlapping labels.** Symptom: "invoice export crashes" flips between billing and technical run to run. Cause: the routes are defined by topic words, and this request has both. Fix: define routes by *what handler is needed*, not by topic. If the export crash needs the technical handler regardless of what is being exported, say so in the route description. Keep an eval set of exactly these boundary cases.

2. **Confident misroutes with side effects.** Symptom: a seat change started from a request that was actually a question. Cause: the LLM router said 0.9 and the account handler trusted it. Fix: the handler's guardrails do not depend on router confidence. Side-effecting tools are approval-gated on every path (13. Human in the Loop). Confidence gates only the *choice*; they never authorize the *action*.

3. **Silent fallback.** Symptom: 30% of traffic lands in `general` and nobody knows. Cause: the fallback route is a catch-all handler that quietly does its best. Fix: the fallback should be visible: it asks a clarifying question or escalates, and it emits a metric. A rising fallback rate is your signal that the routes no longer match the traffic.

4. **Evaluating on happy paths.** Symptom: 98% accuracy in staging, complaints in production. Cause: the eval set was written by the person who wrote the routes. Fix: build the eval set from real traffic, sample misroutes weekly, and report a confusion matrix, not a single accuracy number. A router that never sends `account` to `docs` but often sends `docs` to `account` has a specific, fixable bias.

5. **Router latency added to every request.** Symptom: p50 rose by 600 ms after adding LLM routing. Cause: every request pays for the LLM tier. Fix: the cascade. Rules and embeddings handle the majority for free; the LLM sees only the uncertain remainder. Measure what fraction reaches each tier.

6. **Route drift after a model change.** Symptom: the LLM router's label distribution shifts after a model upgrade. Cause: different model, different reading of the route descriptions. Fix: the routing eval set runs in CI on every model or prompt change, with a threshold that blocks the deploy.

7. **The "routing" agent that re-does the work.** Symptom: the router's output includes a full draft answer and costs as much as the handler. Cause: the router prompt asked the model to "understand and handle" instead of "classify." Fix: a router emits a label and, at most, one sentence of reason. Anything else belongs in the handler.

## What changed by 2026

- **Routing collapsed into tool selection for many systems.** As models became reliable at choosing among well-described tools, the explicit label-then-dispatch step disappeared from simple agents. Anthropic's tool-design guidance (narrow, non-overlapping, namespaced tools with descriptions written like onboarding a new hire) is, in effect, routing guidance: a well-designed tool set *is* a well-designed route menu.
- **Cascades became the cost-control default.** Cheap tiers (rules, small classifiers, embeddings) in front of expensive tiers (LLM, agent) are now standard practice, both for latency and because per-request LLM routing at scale is a visible line item.
- **Model routing is its own layer.** Choosing *which model* handles a request (cheap for docs, reasoning model for hard technical) is now a routing decision in its own right, exposed by gateways and by frameworks' per-node model configuration. See 16. Resource-Aware Optimization.
- **Handoffs formalized routing between agents.** The OpenAI Agents SDK's Handoffs are routing where the destinations are agents rather than nodes: a triage agent delegates to a billing agent. Underneath it is the same idea as `Command(goto=...)`, with the twist that the destination agent receives the conversation. LangGraph docs describe the equivalent as a supervisor or handoff pattern built from `Command`. See 07. Multi-Agent Collaboration.
- **Routers are evaluated like classifiers.** Confusion matrices, per-route precision and recall, and drift monitoring on the label distribution are standard. Routing errors are cheap to measure and catastrophic to ignore, so this is often the first eval a team builds.
- **The fallback route got a name: escalation.** With human-in-the-loop primitives (`interrupt` in LangGraph, approval flows in harnesses) the "unsure" route is a first-class path to a person rather than a best-effort handler.

## Interview questions

**Q1. What is routing and how is it different from an agent deciding what to do?**

Strong: Routing is classifying an input and dispatching it to one of a fixed set of handlers the developer wrote in advance. The menu is fixed, the choice is dynamic, so it is still a workflow. An agent choosing its next action is the same act of choosing from a menu, but the menu is its tools and it may pick several in sequence. The difference is who owns the control flow after the choice. In routing, the handler is a fixed code path with its own tools, model, and guardrails. In an agent loop the model keeps choosing. I pick routing when the handlers need different controls or different models, and tool selection when the handlers are cheap and the request often needs more than one.

Weak: "Routing is an if statement on the user's intent."

**Q2. How would you build the router itself?**

Strong: As a cascade. Rules first for the free, deterministic, compliance-mandated cases. Then a cheap classifier or embedding nearest-centroid with a confidence margin. Then an LLM call for the remainder, asked for a label and one line of reason, not a solution. Each tier has a threshold; anything that falls through all of them goes to a clarify or escalate route that has no side effects. The trace records which tier decided, so every misroute in production becomes a labeled example for the eval set.

Weak: "Ask the LLM to classify the request."

**Q3. What do you do when the router is not confident?**

Strong: Never guess into a side-effecting path. The fallback is the route with no side effects: ask a clarifying question, or escalate to a human. I treat the fallback rate as a health metric, because when it rises the routes no longer match the traffic. And I keep the confidence gate separate from authorization: a high-confidence route into the account handler still cannot change seats without the approval gate, because router confidence is about the classification, not about whether the action is safe.

Weak: "Send it to a general handler."

**Q4. How does LangGraph implement routing, and what is the difference between a conditional edge and `Command`?**

Strong: A conditional edge is a function that reads state and returns the name of the next node; it is the `handlers[route]` dictionary lookup with the runtime doing the dispatch, and the path map lets the compiler validate every destination. `Command` lets a node return a state update and its successor together, so the decision and the work live in one place, and it is the only way to jump to a node in a parent graph. I use the edge when routing is a pure function of state, because it is easy to test and visible in the graph. I use `Command` when the node computes the decision as a byproduct of its work or for handoffs between subgraphs.

Weak: "Conditional edges are like if statements."

**Q5. How do you evaluate a router?**

Strong: Like a classifier. A labeled set built from real traffic, not written by the route author, with a confusion matrix rather than a single accuracy number, because the specific confusions tell you what to fix: which pair of routes overlaps, which route is over-absorbing. I track per-route precision and recall, the fallback rate, and the label distribution over time to detect drift after model or prompt changes. The eval runs in CI on every change to the router prompt, the route descriptions, or the model.

Weak: "Check accuracy on some test cases."

**Q6. Two of your routes keep getting confused. What do you do?**

Strong: First I look at the confused examples to see whether the routes are actually distinct by *handler need* or only by topic. Often the fix is to redefine them: if "invoice export crashes" always needs the technical handler regardless of the invoice, the route description should say so. If they are genuinely both, I either merge them into one handler that has both tool sets, or add a rules tier for the unambiguous signal. Then I add every confused example to the eval set so the fix is measurable and the confusion does not come back.

Weak: "Improve the prompt."

**Q7. When would you not build an explicit router at all?**

Strong: When the handlers share a model, are cheap, and a request often needs more than one of them. Then I expose the handlers as narrow, well-described tools and let the model pick; that is routing by tool selection, and it lets the model compose actions. The cost is losing the explicit confidence gate, so the guardrails move to the tools: side-effecting tools are approval-gated regardless of what the model chose, and there is an explicit ask-a-question tool for ambiguity. Anthropic's tool-design guidance on narrow, non-overlapping tools is really routing guidance in that setting.

Weak: "Always build a router, it is more controllable."

## Build it yourself

Extend the cascade and the graph:

1. Add a `confusion_matrix(examples)` function that runs `route_request` over a list of `(text, expected_route)` pairs and returns a `dict[(expected, actual)] -> count`, plus the fallback rate.
2. Add two boundary cases that deliberately confuse `billing` and `technical` ("the invoice PDF export returns a 500"). Decide and document which handler they should reach, then adjust the rules tier so they do.
3. Make the embedding threshold configurable through `config["configurable"]["margin"]` in the LangGraph version and show that lowering it to 0.0 sends the ambiguous case to `account` instead of `clarify`.
4. Add an `escalate` node reached when the trace shows the LLM tier ran and returned confidence below 0.3, distinct from `clarify`.

Acceptance checks:

- `confusion_matrix` on a 10-example set returns counts that sum to 10, and the fallback rate equals the fraction routed to `clarify`.
- The two boundary cases both route to `technical` and the rules tier is what decided (trace length 1).
- With `margin=0.0` the ambiguous case's trace ends with `("embedding", ...)` and the route is not `clarify`.
- An LLM response of `{"route": "docs", "confidence": 0.2}` on an otherwise-unroutable input reaches `escalate`, not `clarify`.

## Key takeaways

- Routing separates deciding what kind of request this is from handling it; the menu is fixed, the choice is dynamic, so it is still a workflow.
- Build the router as a cascade: rules, then a cheap classifier or embedding margin, then the LLM for the remainder.
- When unsure, route to the path with no side effects. Confidence gates the choice; it never authorizes the action.
- In LangGraph a conditional edge is `handlers[route]` with the runtime doing the lookup; `Command` combines the decision and the state update, and is the way to jump across subgraphs.
- With a capable model, routing often collapses into tool selection: well-designed tools are a well-designed route menu, and the guardrails move to the tool layer.
- Evaluate routers as classifiers, with a confusion matrix from real traffic and drift monitoring on the label distribution.
- Track the fallback rate as a health metric; when it rises, the routes no longer match the traffic.
