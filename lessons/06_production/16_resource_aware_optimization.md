# 16. Resource-Aware Optimization

*Track A (agent engineering) with Track C production concerns. Book: Chapter 16 (Resource-Aware Optimization). Prerequisites: 00. What an agent actually is, 02. Routing, 12. Exception Handling and Recovery.*

An agent loop that works is not the same as an agent loop you can afford. The demo that answered one question for four cents will, at ten thousand questions a day with a few runaway loops mixed in, produce an invoice that ends the project. This lesson treats tokens, dollars, and seconds as first-class state: budgets that live in the run, are decremented by every model call and tool call, and change the agent's behavior before they are exhausted rather than after. Then it covers the four levers that actually move cost: model routing, caching, context pruning, and controlling parallelism.

## The problem

The support agent handles "Why is Acme complaining about their invoice?" It also handles "What are your support hours?" Left alone, one architecture serves both:

```text
"What are your support hours?"
   -> strong model plans (2,100 input tokens, 180 output)
   -> tool: notion_search("support hours")  (3,400 tokens of results)
   -> strong model summarizes                (5,700 input, 60 output)
   -> tool: slack_search("support hours")   (unnecessary, 2,800 tokens)
   -> strong model answers                  (8,600 input, 40 output)
   total: ~16,400 input tokens, 280 output, 3 model calls, 11 seconds
   for a question with a fixed answer that fits in one sentence
```

Multiply by volume. If seventy percent of traffic is that shape, most of the spend buys nothing. Meanwhile the thirty percent that actually needs the strong model and five tool calls is being starved of latency budget because the queue is full of trivial questions.

The naive fix is a global cap: "kill any run over $0.50." That prevents the worst case and does nothing about the average. The better model is a budget the agent knows about and spends deliberately:

```text
                ┌──────────────────────────────────────────┐
   request ───► │  budget: tokens=20k  cost=$0.10  time=15s │
                └──────────────┬───────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ classify difficulty │  (cheap model or heuristic)
                    └──────┬──────┬───────┘
                      easy │      │ hard
                           ▼      ▼
                   cheap model   strong model
                   0 tools       up to N tools
                           │      │
                           ▼      ▼
                ┌──────────────────────────────┐
                │ after every step: spend -=   │
                │   tokens, $, seconds         │
                │ if remaining < reserve:       │
                │   stop tools, answer now      │
                └──────────────────────────────┘
```

The book's chapter on resource-aware optimization frames this as dynamic allocation: the agent monitors its resource use and adapts strategy, choosing a fast path when the request allows it and a thorough path when it does not. Three ideas from the chapter recur below: routing by difficulty, caching, and graceful degradation when a budget is close. The distributed-systems reader will recognize all three from serving systems. What is specific to agents is that the loop's length is decided at runtime by a model, so the budget has to be enforced inside the loop, not around it.

### The four things you pay for

| Resource | Measured in | Dominant driver in an agent | Lever |
|---|---|---|---|
| Input tokens | tokens per model call | context grows every step; tool results | pruning, caching, smaller tool results |
| Output tokens | tokens per model call | reasoning and verbose answers | output caps, concise formats |
| Latency | wall-clock per run | sequential model calls; slow tools | parallel tool calls, cheaper models, fewer steps |
| Rate limits | requests and tokens per minute | fan-out and subagents | bounded concurrency, queueing |

Input tokens dominate cost in agent loops because the context is re-sent on every call. A ten-step loop with a 4,000-token context at step one and 2,000 tokens added per step sends roughly 4k + 6k + 8k + ... + 22k = 130,000 input tokens for what a human would call one question. That quadratic growth is the single most important number in this lesson.

## Build it without a framework

The scripted model stands in for a real one so the budget logic is the object of study. Pricing numbers below are illustrative placeholders in dollars per million tokens, not quotes; put your provider's real price sheet in the table.

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

### Budgets live in state and are checked before every step

A budget is a small record with a limit and a spend for each resource. The loop charges it after every model call and tool call, and checks it before the next model call. The important design choice is the *reserve*: the loop stops calling tools when the remaining budget drops below what a final answer will cost, so the run degrades to "answer with what you have" instead of dying mid-tool-call with nothing to show.

```{code-cell} python
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

PRICE = {  # illustrative $ per 1M tokens: (input, output)
    "cheap":  (0.10, 0.40),
    "strong": (2.50, 10.00),
}

def approx_tokens(messages):
    """Stand-in for a tokenizer: ~4 characters per token."""
    return sum(len(str(m.content)) + len(str(getattr(m, "tool_calls", ""))) for m in messages) // 4


@dataclass
class Budget:
    max_tokens: int
    max_cost: float
    max_seconds: float
    tokens: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    reserve_tokens: int = 600          # enough for one final answer call
    ledger: list = field(default_factory=list)

    def charge_model(self, tier, prompt_tokens, completion_tokens, seconds):
        p_in, p_out = PRICE[tier]
        cost = (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000
        self.tokens += prompt_tokens + completion_tokens
        self.cost += cost
        self.seconds += seconds
        self.ledger.append((tier, prompt_tokens, completion_tokens, round(cost, 6)))

    def charge_tool(self, seconds):
        self.seconds += seconds

    def exhausted(self):
        return (self.tokens >= self.max_tokens or self.cost >= self.max_cost
                or self.seconds >= self.max_seconds)

    def can_afford_another_tool_round(self):
        # Stop calling tools while there is still room for the final answer.
        return self.tokens + self.reserve_tokens < self.max_tokens and not self.exhausted()


@tool
def notion_search(query: str) -> str:
    """Search the Notion knowledge base."""
    return "Support hours: 9am-6pm ET Mon-Fri. Enterprise plans have 24/7 paging. " * 6


@tool
def stripe_get_invoice(invoice_id: str) -> str:
    """Fetch a Stripe invoice."""
    return "invoice in_4471 open, $44.00, 3 failed attempts, card_declined " * 8


TOOL_LATENCY = {"notion_search": 0.8, "stripe_get_invoice": 0.4}
tools_by_name = {t.name: t for t in (notion_search, stripe_get_invoice)}


def run_budgeted(model, tier, user_text, budget, max_steps=8):
    messages = [HumanMessage(user_text)]
    for _ in range(max_steps):
        prompt_tokens = approx_tokens(messages)
        ai = model.invoke(messages)
        messages.append(ai)
        budget.charge_model(tier, prompt_tokens, approx_tokens([ai]), seconds=1.2)
        if not ai.tool_calls:
            return messages, "answered"
        if not budget.can_afford_another_tool_round():
            # Graceful degradation: drop the tool calls and ask for a final answer.
            messages.pop()
            messages.append(HumanMessage("Budget nearly exhausted. Answer now with what you know."))
            ai = model.invoke(messages)
            messages.append(ai)
            budget.charge_model(tier, approx_tokens(messages), approx_tokens([ai]), seconds=1.2)
            return messages, "degraded"
        for call in ai.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            budget.charge_tool(TOOL_LATENCY[call["name"]])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return messages, "step_limit"


# A run that wants three tool rounds but is only allowed a small token budget.
model = scripted(
    [("notion_search", {"query": "support hours"})],
    [("stripe_get_invoice", {"invoice_id": "in_4471"})],
    [("notion_search", {"query": "escalation policy"})],
    "Support hours are 9am-6pm ET; Acme's invoice is open after card declines.",
    "Support hours are 9am-6pm ET; Acme's invoice is open after card declines.",
)
budget = Budget(max_tokens=900, max_cost=0.05, max_seconds=15)
messages, outcome = run_budgeted(model, "strong", "Support hours, and why is Acme upset?", budget)

print("outcome:", outcome)
print(f"tokens={budget.tokens} cost=${budget.cost:.5f} seconds={budget.seconds:.1f}")
for row in budget.ledger:
    print("  model call:", row)
assert outcome == "degraded"                       # stopped tools before the budget died
assert budget.tokens < budget.max_tokens           # and still finished under the cap
assert sum(1 for m in messages if isinstance(m, ToolMessage)) == 2
```

Look at the ledger: prompt tokens climb every call because the whole transcript is re-sent. The per-call cost is dominated by input. That is the quadratic growth from the problem section, visible in four rows.

### Routing and caching: the two biggest levers

Routing sends easy requests to a cheap model with no tools. Caching avoids paying twice for the same work. There are three distinct caches and they are frequently confused:

```text
1. Result cache      key = (tool, args)        saves tool latency and result tokens
2. Semantic cache    key = embedding(query)     saves whole runs for repeat questions
3. Prompt cache      key = exact prompt prefix  provider-side; saves input cost on the
                                                 stable prefix (system prompt, tool schemas,
                                                 early transcript) across calls
```

Prompt caching is a provider feature: when consecutive calls share a long identical prefix, the provider reuses the computed KV state and charges a discounted rate for the cached portion (see 38. LLM Serving Infrastructure for why the prefix must be byte-identical). For an agent loop, where each call is the previous call plus a few messages, this is the difference between quadratic and roughly linear cost. Three rules make it work: put stable content first (system prompt, tool schemas), never inject anything that changes per call into the prefix (timestamps, request ids), and keep tool ordering deterministic. MCP's current spec revision asks servers to return tools in deterministic order for exactly this reason.

The next cell simulates the cost of a day of traffic under four configurations. The router is a heuristic (length and keywords); in production it is a cheap classifier call or a small fine-tuned model (see 30. Fine-Tuning, Routing, and Prompt Optimization).

```{code-cell} python
import functools

REQUESTS = (
    [("What are your support hours?", "easy")] * 60
    + [("How do I reset my password?", "easy")] * 40
    + [("Why is Acme complaining about their invoice?", "hard")] * 25
    + [("Reconcile Stripe and CRM MRR for Q3 and explain gaps", "hard")] * 15
)

HARD_MARKERS = ("why", "reconcile", "explain", "compare", "investigate")

def route(text):
    t = text.lower()
    return "strong" if (len(t.split()) > 7 or any(w in t for w in HARD_MARKERS)) else "cheap"

def run_cost(tier, steps, prefix_tokens=3000, per_step_tokens=1500, cached_fraction=0.0):
    """Cost of one agent run with `steps` model calls and a growing transcript."""
    p_in, p_out = PRICE[tier]
    total = 0.0
    for step in range(steps):
        prompt = prefix_tokens + per_step_tokens * step
        cached = prompt * cached_fraction if step > 0 else 0     # prefix hits from call 2 on
        total += ((prompt - cached) * p_in + cached * p_in * 0.1 + 150 * p_out) / 1_000_000
    return total

def day_cost(*, routing, prompt_cache, result_cache):
    seen = set()
    total, calls = 0.0, 0
    for text, difficulty in REQUESTS:
        if result_cache and text in seen:      # semantic/result cache hit: no model call
            continue
        seen.add(text)
        tier = route(text) if routing else "strong"
        steps = 1 if (tier == "cheap") else (5 if difficulty == "hard" else 3)
        total += run_cost(tier, steps, cached_fraction=0.8 if prompt_cache else 0.0)
        calls += steps
    return round(total, 4), calls

configs = {
    "baseline (strong, no caching)":   dict(routing=False, prompt_cache=False, result_cache=False),
    "+ prompt cache":                  dict(routing=False, prompt_cache=True,  result_cache=False),
    "+ routing":                       dict(routing=True,  prompt_cache=True,  result_cache=False),
    "+ result cache for repeats":      dict(routing=True,  prompt_cache=True,  result_cache=True),
}
results = {name: day_cost(**cfg) for name, cfg in configs.items()}
base = results["baseline (strong, no caching)"][0]
for name, (cost, calls) in results.items():
    print(f"{name:34s} ${cost:8.4f}  {calls:4d} model calls  ({cost / base:5.1%} of baseline)")

# Routing must not misroute the hard requests.
assert all(route(t) == "strong" for t, d in REQUESTS if d == "hard")
assert results["+ prompt cache"][0] < base * 0.6
assert results["+ result cache for repeats"][0] < results["+ routing"][0]
```

The relative sizes are what matter, not the absolute dollars. Prompt caching alone removes a large share of the input cost because most of every call is the previous call. Routing removes the strong-model cost from requests that never needed it. A result cache removes whole runs, but only for genuinely repeated requests, and it is also the most dangerous: a cached answer to "what is Acme's invoice status" is wrong the moment the invoice is paid. Cache reads, never writes, and key the cache on what the answer actually depends on.

## How LangGraph does it

LangGraph does not have a budget primitive. It has the pieces you need to build one, and each corresponds to something in the from-scratch code:

| From scratch | LangGraph | Underneath |
|---|---|---|
| `Budget` fields | keys in the state `TypedDict` with reducers | Reducers such as `operator.add` accumulate spend across nodes and across parallel branches |
| `can_afford_another_tool_round()` | conditional edge | A plain function over state that returns the next node name |
| `route()` | a router node plus conditional edge, or `Command(goto=...)` | The model choice is state; the edge reads it |
| result cache | `CachePolicy` on `add_node` with a compiled `cache=` | The executor hashes the node input (default key) and skips the node on a hit within `ttl` |
| context pruning | a node that rewrites `messages` (with `trim_messages`) | Nothing special: `messages` is state, and a node can shrink it |
| `max_steps` | `recursion_limit` | The Pregel step bound |

The next cell puts them together: a router node picks the tier, a budget reducer accumulates spend, a cached `fetch_invoice` node avoids a second Stripe hit, a `prune` node trims the transcript to a token cap before each model call, and a conditional edge degrades to a final answer when the reserve is reached.

```{code-cell} python
from operator import add
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    tier: str
    tokens_spent: Annotated[int, add]        # reducer: every node adds its own spend
    cost_spent: Annotated[float, add]
    account: str
    invoice: str
    outcome: str


class InvoiceRequest(TypedDict):
    account: str                             # the only field fetch_invoice reads, so the only cache key

MAX_TOKENS, RESERVE, CONTEXT_CAP = 2000, 400, 700
stripe_hits = {"n": 0}

def router(state: State):
    return {"tier": route(state["messages"][-1].content)}

def fetch_invoice(state: InvoiceRequest):
    stripe_hits["n"] += 1                    # counts real calls; cache hits skip this node
    return {"invoice": f"{state['account']} in_4471 open $44.00, 3 failed attempts"}

def prune(state: State):
    """Keep the system prompt and the most recent turns under CONTEXT_CAP tokens."""
    kept = trim_messages(state["messages"], max_tokens=CONTEXT_CAP,
                         token_counter=count_tokens_approximately, strategy="last",
                         include_system=True, start_on="human")
    # add_messages merges by id, so to *remove* messages we return RemoveMessage markers.
    from langchain_core.messages import RemoveMessage
    keep_ids = {m.id for m in kept}
    return {"messages": [RemoveMessage(id=m.id) for m in state["messages"] if m.id not in keep_ids]}

def call_model(state: State):
    prompt_tokens = count_tokens_approximately(state["messages"])
    ai = models[state["tier"]].invoke(state["messages"])
    p_in, p_out = PRICE[state["tier"]]
    return {"messages": [ai], "tokens_spent": prompt_tokens + 40,
            "cost_spent": (prompt_tokens * p_in + 40 * p_out) / 1_000_000}

def decide(state: State):
    if state["tokens_spent"] + RESERVE >= MAX_TOKENS:
        return "final"                       # degrade: no more tools
    return "model"

def final(state: State):
    return {"outcome": "degraded", "messages": [AIMessage("Answering with what I have: " + state["invoice"])]}

# Give each message an id so trim/remove can address them.
import uuid
def msg(text): return HumanMessage(text, id=str(uuid.uuid4()))

models = {
    "cheap":  scripted("Support hours are 9am-6pm ET, Mon-Fri."),
    "strong": scripted("Working on Acme...", "Still working...", "Acme's invoice is open after card declines."),
}

builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("fetch_invoice", fetch_invoice, input_schema=InvoiceRequest,
                 cache_policy=CachePolicy(ttl=300))
builder.add_node("prune", prune)
builder.add_node("model", call_model)
builder.add_node("final", final)
builder.add_edge(START, "router")
builder.add_edge("router", "fetch_invoice")
builder.add_edge("fetch_invoice", "prune")
builder.add_conditional_edges("prune", decide, {"model": "model", "final": "final"})
builder.add_edge("final", END)
# For the demo, a "strong" answer that does not end with a final decision loops back through prune.
builder.add_conditional_edges("model", lambda s: END if s["tier"] == "cheap"
                              or "invoice is open" in s["messages"][-1].content else "prune",
                              {END: END, "prune": "prune"})
graph = builder.compile(cache=InMemoryCache())

easy = graph.invoke({"account": "acme", "messages": [SystemMessage("You are the support agent.", id="sys"),
                                                     msg("What are your support hours?")]})
hard = graph.invoke({"account": "acme", "messages": [SystemMessage("You are the support agent.", id="sys"),
                                                     msg("Why is Acme complaining about their invoice?")]})

print("easy ->", easy["tier"], f"tokens={easy['tokens_spent']} cost=${easy['cost_spent']:.6f}")
print("hard ->", hard["tier"], f"tokens={hard['tokens_spent']} cost=${hard['cost_spent']:.6f}",
      "| final:", hard["messages"][-1].content)
print("stripe calls actually made:", stripe_hits["n"], "(second run hit the node cache)")
assert easy["tier"] == "cheap" and hard["tier"] == "strong"
assert stripe_hits["n"] == 1                       # CachePolicy skipped the second identical call
assert hard["cost_spent"] > easy["cost_spent"] * 5 # routing put the cheap request on the cheap model
assert hard["tokens_spent"] < MAX_TOKENS
```

Underneath, `CachePolicy` does what a `functools.lru_cache` keyed on the node's input does, with a TTL and a pluggable backend; the default key is a hash of the node input, so two runs with byte-identical state for that node share a result. The key is computed from whatever the node receives, which is why `fetch_invoice` declares `input_schema=InvoiceRequest`: it sees only `account`, so the second run hit the cache even though the user message differed. Leave the node on the full state and the key includes `messages`, which change every turn, so the cache never hits. Narrow the input to the fields that actually determine the result. The `tokens_spent` reducer is the `Budget.charge_model` call spread across nodes: each node reports its own spend and the reducer sums, which also makes it correct under parallel fan-out (see 03. Parallelization), where several branches charge the same budget concurrently.

### Parallelism versus rate limits

Fan-out reduces latency and increases instantaneous token rate. A `Send` to eight subagents that each make three model calls issues twenty-four calls in a burst. Provider rate limits are per minute, so the burst either gets 429s (which 12. Exception Handling taught you to retry with backoff, adding latency you were trying to remove) or you bound concurrency. The rule: parallel width is a budget too. Set it from the rate limit and the per-call token size, not from the number of things you could do at once. Anthropic's context-engineering guidance adds another reason to bound subagent count: each subagent's condensed summary (1,000 to 2,000 tokens is the suggested size) lands in the parent's context, so more subagents means more context rot for the orchestrator.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

cheap = init_chat_model(os.environ.get("COURSE_CHEAP_MODEL", "openai:gpt-5.6-mini"))
strong = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))

# Route with a real cheap classifier, then run the chosen tier and read token usage from metadata.
def live_route(text):
    verdict = cheap.invoke(f"Classify as EASY or HARD for a support agent. Reply one word.\n\n{text}").content
    return "strong" if "HARD" in verdict.upper() else "cheap"

for text in ["What are your support hours?", "Why is Acme complaining about their invoice?"]:
    tier = live_route(text)
    agent = create_agent(strong if tier == "strong" else cheap, tools=[notion_search, stripe_get_invoice])
    out = agent.invoke({"messages": [HumanMessage(text)]})
    usage = [m.usage_metadata for m in out["messages"] if isinstance(m, AIMessage) and m.usage_metadata]
    total_in = sum(u["input_tokens"] for u in usage)
    print(f"{tier:6s} {text[:45]:45s} model_calls={len(usage)} input_tokens={total_in}")
```

The live cell reads `usage_metadata` from each `AIMessage`. That is where real token counts come from; the `approx_tokens` stand-in in the offline cells exists only so the notebook runs without a provider. In production, charge the budget from `usage_metadata`, and log it per run so cost per task is a number you can plot, not an estimate.

## What goes wrong in production

1. **Budget enforced after the fact.** Symptom: runs are killed at the cap with nothing to show. Cause: a global timeout or cost kill-switch outside the loop. Fix: budget in state, checked before each model call, with a reserve that triggers a final-answer path. The user gets a partial answer and a note, not a timeout.

2. **Quadratic context.** Symptom: cost per run grows with the square of steps; the last call costs more than the first ten combined. Cause: the whole transcript re-sent every step; large tool results never pruned. Fix: cap tool result size at the tool (Anthropic's guidance notes Claude Code truncates tool results at 25,000 tokens), prune or compact the transcript between steps, and rely on prompt caching for the stable prefix.

3. **Prompt cache misses you did not notice.** Symptom: the cached-token count in usage metadata is near zero. Cause: something in the prefix changes per call: a timestamp in the system prompt, a request id, tools listed in a different order, a user name injected early. Fix: stable prefix first, volatile content last, deterministic tool order. Measure the cached fraction; it should be most of the prompt from the second call onward.

4. **Stale result cache.** Symptom: the agent reports an invoice as open after it was paid. Cause: caching a read of mutable data with a long TTL, or keying the cache on the query text rather than on the data it depends on. Fix: short TTLs for mutable sources, no caching after a write to the same entity, and cache keys that include the entity id and version where available.

5. **Router that is wrong in the expensive direction.** Symptom: quality complaints on a subset of tickets. Cause: the difficulty router sends hard requests to the cheap tier to save money. Fix: measure the router on a labelled set; bias toward the strong tier for ambiguous cases; add an escalation path where the cheap tier can say "I cannot answer this" and hand off (a cascade, see 30. Fine-Tuning, Routing, and Prompt Optimization).

6. **Fan-out that trips rate limits.** Symptom: 429s and retries make the parallel version slower than the sequential one. Cause: unbounded `Send` fan-out. Fix: a semaphore or worker pool sized from the provider's requests-per-minute and tokens-per-minute; queue the rest; treat concurrency as a budget field.

7. **Cost per task unknown.** Symptom: nobody can say what the agent costs per resolved ticket. Cause: token usage logged per call, not aggregated per run and per outcome. Fix: every run has a run id; every model call records `usage_metadata` against it; report cost per task by outcome (resolved, escalated, failed). Cost per *successful* task is the number that matters, and it rises sharply when failures are frequent.

8. **Optimizing tokens and losing quality.** Symptom: cheaper, faster, and wrong. Cause: pruning removed the one tool result the answer depended on; the cheap model hallucinated where the strong one would have called a tool. Fix: every cost change is an eval change. Run the eval suite (see 19. Evaluation and Monitoring) and plot cost against quality; pick an operating point on the curve deliberately.

## What changed by 2026

**Context engineering absorbed cost engineering.** Anthropic's guidance on effective context engineering treats the context window as a finite resource that degrades as it fills ("context rot"). The prescribed techniques are the same ones that cut cost: compaction that preserves decisions and open issues, structured note-taking outside the context, just-in-time retrieval through lightweight identifiers rather than preloading, and subagents that return condensed summaries. A team that manages context well gets the token bill down as a side effect.

**Caching became a protocol concern.** MCP's current revision returns `ttlMs` and `cacheScope` on `tools/list`, `prompts/list`, and `resources/read`, and asks servers to return tools in deterministic order so that prompt-cache prefixes stay stable. The tool listing that used to be fetched on every turn is now cacheable by contract.

**Model routing became an ordinary component.** In LangChain 1.3, `init_chat_model` takes a provider-prefixed string, so swapping tiers is a config change, and `create_agent` accepts any chat model. Middleware can select the model per call. Deep Agents 0.5 exposes provider and harness profiles. Routing logic that was once a custom wrapper is now a place in the stack.

**Reasoning effort is a cost knob.** Reasoning models expose an effort setting, and OpenAI's guidance treats it as a tuning parameter: low effort for routine steps, higher for hard ones. Reasoning tokens are billed as output, which changes the cost model of a loop; a step that "thinks" for 3,000 tokens costs more than a step that calls a tool. Budget for it, and do not prescribe verbose step-by-step reasoning in prompts for models that reason natively.

**Hosted harnesses meter for you.** Anthropic's managed-agents design and OpenAI's Agents API (as their current docs describe) run the loop server-side and handle compaction. The spend still lands on your bill, so budget policy remains yours; what you get is a reliable usage stream to enforce it against.

## Interview questions

**Q1. How would you keep an agent's cost under control in production?**

Strong: Budget in state, not a kill switch around the process. The run carries token, cost, and time budgets that every model call and tool call decrements, and the loop checks them before each step with a reserve so it can degrade to a final answer rather than die. Then the four levers: route easy requests to a cheap model with no tools, use provider prompt caching by keeping the stable prefix byte-identical, prune and compact the transcript so context does not grow quadratically, and bound parallel width against rate limits. Every one of those changes goes through the eval suite, because the cheapest agent that is wrong costs more than the expensive one that is right. And report cost per successful task, per run id, from real usage metadata.

Weak: "Set a max spend and use a cheaper model."

**Q2. Why does an agent's cost grow faster than linearly with the number of steps?**

Strong: Because each model call re-sends the whole transcript, and the transcript grows every step with the tool results. If step one has a 4k context and each step adds 2k, the total input over ten steps is the sum of an arithmetic series, about 130k tokens. Two things break that: prompt caching, which makes the shared prefix nearly free from the second call, and pruning or compaction, which caps the transcript. Capping tool result sizes at the tool is the cheapest fix of all.

Weak: "More steps means more calls."

**Q3. Explain prompt caching and what breaks it.**

Strong: The provider caches the computed attention state for a prompt prefix and reuses it when a later call shares that exact prefix, charging a fraction of the input price for the cached part. It works for agents because each call is the previous one plus a few messages. It breaks when anything early in the prompt varies per call: timestamps, request ids, user-specific content placed before the stable instructions, or tools listed in a different order. The fix is a layout rule: stable content first, volatile content last, deterministic tool ordering, and monitoring the cached-token fraction in usage metadata.

Weak: "It caches responses to identical prompts."

**Q4. What is the difference between a result cache, a semantic cache, and a prompt cache?**

Strong: A result cache is keyed on a tool call's name and arguments and saves the tool's latency and the tokens of its output; it must respect the data's mutability, so short TTLs for things like invoice status. A semantic cache is keyed on an embedding of the user's question and returns a previous whole answer for near-duplicate questions; it saves entire runs but risks returning stale or subtly wrong answers, so it belongs only on questions with stable answers. A prompt cache is provider-side and keyed on the exact token prefix; it saves input cost across the calls within one run. They sit at different layers and solve different problems.

Weak: "They all cache to save money."

**Q5. How do you decide which model to route a request to?**

Strong: Start with a labelled set of requests and outcomes so the router can be measured. The router itself can be a heuristic, a cheap classifier call, or a small fine-tuned model; what matters is its error asymmetry. Misrouting hard to cheap costs quality, misrouting easy to strong costs money, and quality is usually the expensive mistake, so bias toward strong on ambiguity. A cascade is often better than a pure router: try the cheap tier first and let it escalate when it lacks confidence. Then sweep the threshold and plot cost against quality; the operating point is a product decision.

Weak: "Use the small model for simple questions."

**Q6. How does parallel fan-out interact with rate limits?**

Strong: Fan-out trades latency for instantaneous token rate. Eight subagents doing three calls each is a burst of twenty-four requests against a per-minute limit, so either the provider returns 429s and your retries add back the latency you tried to remove, or you bound concurrency. Treat parallel width as a budget field: size it from requests-per-minute and tokens-per-minute divided by expected tokens per call, and queue the remainder. There is also a context cost: each subagent's summary lands in the orchestrator's context, so more subagents means a fuller parent context.

Weak: "Run everything in parallel to make it faster."

**Q7. How do you measure whether a cost optimization worked?**

Strong: Cost per successful task, not cost per call. Aggregate `usage_metadata` per run id, join to the outcome, and compare before and after on the same eval set. Report the quality metric next to it; if quality dropped, the optimization did not work even if the bill did. Also watch latency percentiles, because some optimizations, like aggressive retries after 429s, move cost into time.

Weak: "Check if the bill went down."

## Build it yourself

Extend the LangGraph graph in this lesson into a budget-aware agent with measured cost:

1. Add a `concurrency` budget: a `Send`-based fan-out node that queries three sources in parallel, gated by a semaphore-like counter in state so at most two branches run per step.
2. Add a `cached_tokens` field charged at 10% of the input price when the prefix (system prompt plus tool schemas) is unchanged between consecutive model calls, and 100% when the prefix changes. Inject a timestamp into the system prompt in one run and show the cost difference.
3. Give `fetch_invoice` a narrow `input_schema` containing only the invoice id, so the cache key ignores unrelated state.
4. Record a per-run summary `{tier, steps, tokens, cost, outcome}` and print a table over a batch of ten requests.

Acceptance checks:

- With the timestamp in the prefix, `cached_tokens` is 0 and total cost is at least 2x the stable-prefix run.
- Two runs with the same invoice id but different user messages make exactly one real `fetch_invoice` call.
- No run exceeds `MAX_TOKENS`, and any run that hits the reserve has `outcome == "degraded"` with a non-empty final message.
- The batch table shows cheap-tier runs with exactly one model call and strong-tier runs with at least two.

## Key takeaways

- Budgets are state: tokens, cost, time, and concurrency are checked before every step, with a reserve that lets the run degrade to an answer instead of dying.
- Input tokens dominate agent cost because the transcript is re-sent every step; prompt caching and pruning turn quadratic growth into roughly linear.
- Route by difficulty with a measured router or a cascade; the expensive mistake is sending hard requests to the cheap tier.
- Three caches, three layers: result cache (tool, args), semantic cache (question), prompt cache (exact prefix); each has its own staleness risk.
- Parallel width is a budget too; size it from rate limits, and remember every subagent summary lands in the parent's context.
- LangGraph gives you reducers, conditional edges, `CachePolicy`, and `trim_messages`; the budget policy is yours to write.
- Report cost per successful task from real `usage_metadata`, and treat every cost change as an eval change.
