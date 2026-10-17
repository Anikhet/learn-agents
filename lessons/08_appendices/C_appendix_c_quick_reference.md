# C. Quick Reference: choosing a pattern from the symptom

*Track A (agent engineering) with a Track D (interviews) flavor. Book: Appendix C (Quick Overview of Agentic Patterns) and the chapter summaries. Prerequisites: 00. What an agent actually is, D. Building an Agent. Read this after the core patterns, and again the night before an interview.*

This appendix is a lookup table with a spine. The table maps a symptom you can observe to the pattern that treats it, the LangGraph primitive that implements it, the lesson that teaches it, and the interview question it answers. The spine is a decision procedure: workflow first, agent only when the path cannot be known in advance, multiple agents only when one context window is the actual bottleneck.

You will build the lookup table as data, a recommender over it, and the workflow-versus-agent-versus-multi-agent decision tree as a function with asserts. Then you will compile the three canonical shapes in LangGraph and count what each one costs.

## The problem

The enterprise support/ops agent from lesson 00 has a backlog. Five tickets arrive in the same week:

```text
T1  "Answers about refund policy are sometimes wrong."
T2  "Every request goes through the same 9-step prompt, even password resets."
T3  "The Acme invoice investigation takes 4 minutes; Slack, CRM, and Stripe
     lookups run one after another."
T4  "The agent issued a $4,800 refund without anyone checking."
T5  "Leadership wants the agent to 'be more autonomous' across all tools."
```

The naive team reaches for patterns by name. Someone read about multi-agent systems, so T5 becomes "build a planner agent, a Stripe agent, a Slack agent, and a supervisor." Someone else read about reflection, so T1 becomes "add a critic that rereads every answer."

Here is where that breaks:

```text
Ticket  Pattern chosen by name     What actually happens
------  -------------------------  -------------------------------------------
T1      Reflection (self-critique) Critic rereads the same wrong policy text and
                                   approves it. Root cause was stale retrieval.
T2      Bigger prompt              Longer prompt, same waste. Root cause: no router.
T3      Multi-agent                3 agents, 3x tokens, same serial tool calls
                                   inside each. Root cause: no fan-out.
T4      "Be careful" in prompt     Prompt is a suggestion. The tool still exists.
T5      Supervisor + 4 agents      Context split across agents, handoffs lose
                                   facts, nobody can debug a trajectory.
```

Every one of those choices started from the pattern. The fix is to start from the **symptom**, name the **cause**, and only then pick the **smallest pattern** that removes that cause. That is the discipline the book's pattern catalog assumes and never states in one place, and it is what Anthropic's "Building effective agents" guidance means by "find the simplest solution possible, and only increase complexity when needed."

The mapping has a shape:

```text
observable symptom  ──►  cause  ──►  pattern  ──►  primitive  ──►  eval that proves it
      (trace)          (diagnosis)   (design)     (code)          (regression gate)
```

Skip the diagnosis step and you add overhead without removing the cause. Skip the eval step and you cannot tell whether the pattern helped.

## Build it without a framework

A reference table is data. Put it in code, and it becomes testable: you can assert that each symptom points to a pattern, that each pattern names a primitive and a lesson, and that the recommender prefers the cheap fix when two patterns both apply.

### The master table

| Symptom you can observe | Likely cause | Pattern | LangGraph / LangChain primitive | Lesson | Interview question it answers |
|---|---|---|---|---|---|
| One prompt does too much; quality drops as steps are added | Task has fixed stages | Prompt chaining | Linear `add_edge` sequence; typed state per stage | 01 | "When do you split a prompt into a chain?" |
| Every request pays for the most expensive path | Heterogeneous inputs, one handler | Routing | `add_conditional_edges` on a classifier; `Command(goto=...)` | 02 | "How do you route between cheap and expensive models?" |
| Independent lookups run one after another | Serial calls with no data dependency | Parallelization | Several edges from one node; `Send` for dynamic fan-out; reducer to fan in | 03 | "How do you cut latency in a multi-tool agent?" |
| Output has checkable defects the generator misses | Generation and checking share one pass | Reflection / evaluator-optimizer | Generator node + evaluator node + bounded loop edge | 04 | "When does self-critique help, and when is it theater?" |
| Needs facts from your systems or the present | Information is not in the prompt or weights | Tool use | `@tool`, `ToolNode`, `create_agent` | 05 | "How does function calling work end to end?" |
| Long task, next step depends on discoveries | Path is not knowable in advance | Planning + agent loop | Plan in state; agent node + tools loop; todo middleware | 06 | "Workflow or agent?" |
| One context cannot hold all the work | Breadth exceeds one context window | Multi-agent (orchestrator-workers) | Subgraphs as nodes; `Send` to workers; Deep Agents `task` subagents | 07 | "When does multi-agent break down?" |
| Forgets the user between sessions | Only thread-scoped state | Memory | Checkpointer (short-term) + `Store` (long-term) | 08 | "Design memory for a long-running agent." |
| Wrong answers about internal documents | Retrieval misses or is stale | RAG | Retrieval node; retriever tool; reranking | 14, 29 | "Evaluate retrieval and generation separately." |
| Tools come from many teams and vendors | Bespoke integrations per tool | MCP | MCP adapters load tools into the tool list | 10 | "MCP versus function calling?" |
| Transient 503s kill long runs | Exceptions escape the loop | Exception handling | `RetryPolicy` per node; errors as `ToolMessage` | 12 | "How do you handle tool timeouts and retries?" |
| Consequential action taken with no review | Nothing pauses before side effects | Human in the loop | `interrupt()` + checkpointer + `Command(resume=...)` | 13 | "Design HITL approval for consequential actions." |
| Cost or latency blows the budget | Frontier model on every step | Resource-aware optimization | Router to cheaper model; call-limit middleware | 16 | "How do you bound cost and steps?" |
| Unsafe inputs or outputs reach users | No checks at boundaries | Guardrails | Input/output validation nodes; PII middleware | 18 | "Layered defenses for an agent?" |
| "It works on my demo" | No measurement | Evaluation and monitoring | LangSmith datasets and evaluators; trajectory checks | 19, 37 | "How do you evaluate an agent, not a response?" |
| Quality decays after many steps | Context rot, bloated transcript | Context engineering | Summarization middleware; offload to files/store; subagents | 34 | "Your agent drifts after a long run. Diagnose it." |
| Crash at minute 40 restarts from minute 0 | No durable state | Durable execution | Checkpointer + `thread_id`; durability modes | 36 | "How do you resume a long task after a crash?" |
| Injected text in a web page triggers a tool call | Untrusted data treated as instructions | Security / capability limits | Tool allowlists per step; approval gates; sandbox | 31, 33 | "Defend against prompt injection." |

Read the "Likely cause" column twice. It is the column most people skip, and it is the column that decides whether the pattern works.

### The table as data, and a recommender over it

The recommender below does three things the table cannot. It matches free-text symptoms to entries by keywords. It ranks by a rough cost so that, when two patterns could apply, the cheaper one comes first. And it refuses to recommend anything when nothing matches, rather than defaulting to "add an agent."

```{code-cell} python
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    name: str
    lesson: str
    primitive: str
    keywords: tuple
    cost: int          # 1 = a few lines of control flow, 5 = new agents and new failure modes
    question: str


CATALOG = [
    Pattern("prompt chaining", "01", "add_edge sequence", ("stages", "too much", "one prompt"), 1,
            "When do you split a prompt into a chain?"),
    Pattern("routing", "02", "add_conditional_edges / Command(goto)", ("every request", "same path", "password reset", "cheap"), 1,
            "How do you route between cheap and expensive handlers?"),
    Pattern("parallelization", "03", "fan-out edges / Send + reducer", ("one after another", "serial", "sequential", "latency"), 2,
            "How do you cut latency in a multi-tool agent?"),
    Pattern("reflection", "04", "generator + evaluator + bounded loop", ("checkable defect", "format error", "fails validation"), 2,
            "When does self-critique help?"),
    Pattern("rag", "14", "retrieval node / retriever tool", ("wrong about policy", "internal document", "stale", "policy"), 2,
            "Evaluate retrieval and generation separately."),
    Pattern("human in the loop", "13", "interrupt + checkpointer", ("without review", "without anyone", "irreversible", "money movement"), 2,
            "Design HITL approval for consequential actions."),
    Pattern("durable execution", "36", "checkpointer + thread_id", ("crash", "restart", "resume"), 2,
            "How do you resume after a crash?"),
    Pattern("context engineering", "34", "summarization / offload / subagent", ("drift", "after many steps", "context"), 3,
            "Your agent drifts after a long run. Diagnose it."),
    Pattern("agent loop", "06", "agent node + ToolNode loop", ("depends on what it finds", "open-ended", "unknown path"), 3,
            "Workflow or agent?"),
    Pattern("multi-agent", "07", "subgraphs / Send to workers", ("exceeds one context", "breadth", "independent research"), 5,
            "When does multi-agent break down?"),
]


def recommend(symptom: str, catalog=CATALOG):
    """Keyword match, then cheapest first. No match means: diagnose more, build nothing."""
    text = symptom.lower()
    hits = [p for p in catalog if any(k in text for k in p.keywords)]
    return sorted(hits, key=lambda p: (p.cost, p.lesson))


TICKETS = {
    "T1": "Answers about refund policy are sometimes wrong; the policy doc is stale.",
    "T2": "Every request goes through the same path, even a password reset.",
    "T3": "Slack, CRM, and Stripe lookups run one after another.",
    "T4": "The agent issued a refund without anyone checking.",
    "T5": "Be more autonomous.",
}

for tid, text in TICKETS.items():
    recs = recommend(text)
    first = f"{recs[0].name} (lesson {recs[0].lesson}, {recs[0].primitive})" if recs else "no match: diagnose first"
    print(f"{tid}: {first}")

assert recommend(TICKETS["T1"])[0].name == "rag"                 # not reflection
assert recommend(TICKETS["T2"])[0].name == "routing"
assert recommend(TICKETS["T3"])[0].name == "parallelization"      # not multi-agent
assert "human in the loop" in [p.name for p in recommend(TICKETS["T4"])]
assert recommend(TICKETS["T5"]) == []                             # a wish is not a symptom
```

Two results deserve comment.

T4 matches "without anyone" (human in the loop) and nothing else, so the answer is an approval gate, not a better prompt. The prompt already said "be careful." A gate in code is the only thing that stops the tool.

T5 matches nothing, and that is the correct output. "Be more autonomous" names a wish, not a symptom. The right response in a design review, and in an interview, is a question: "Which task fails today, and what does the trace show?"

### The decision tree: workflow, agent, or multi-agent

The single most asked architecture question is some version of "workflow or agent?" Anthropic's guidance defines workflows as systems where LLMs and tools are orchestrated through predefined code paths, and agents as systems where the LLM dynamically directs its own process and tool usage. Multi-agent adds a third rung: several model contexts, each with its own loop.

```text
                     Can you write the steps down before the run starts?
                              │
                 ┌────── yes ─┴─ no ──────┐
                 ▼                        ▼
            WORKFLOW                 Does the work fit in one context
  (chain / route / parallel /        window with room to think?
   evaluator-optimizer in code)           │
                              ┌── yes ────┴──── no ──┐
                              ▼                      ▼
                        SINGLE AGENT        Can the work split into parts that
                   (tools + loop + bounds)  need little shared state mid-run?
                                                     │
                                          ┌── yes ───┴─── no ──┐
                                          ▼                    ▼
                                    MULTI-AGENT          SINGLE AGENT +
                                (orchestrator-workers,   context engineering
                                 condensed summaries)    (compaction, notes, files)
```

Overlay two cross-cutting checks on every leaf:

```text
Consequential or irreversible action?   → approval gate (interrupt) regardless of leaf
Run longer than a request timeout?      → checkpointer + thread_id regardless of leaf
```

Notice the bottom-right leaf. Multi-agent is not the answer to "the task is big." It is the answer to "the task is big *and* decomposes into parts that do not need to talk to each other while they work." If the parts are tightly coupled, splitting them across contexts creates handoff loss, and the better fix is one agent with better context management.

```{code-cell} python
from dataclasses import dataclass


@dataclass
class Task:
    name: str
    steps_known_in_advance: bool
    fits_one_context: bool = True
    parts_loosely_coupled: bool = False
    consequential_action: bool = False
    long_running: bool = False


def choose_architecture(t: Task) -> dict:
    if t.steps_known_in_advance:
        shape = "workflow"
    elif t.fits_one_context:
        shape = "single agent"
    elif t.parts_loosely_coupled:
        shape = "multi-agent"
    else:
        shape = "single agent + context engineering"
    add_ons = []
    if t.consequential_action:
        add_ons.append("interrupt approval gate")
    if t.long_running:
        add_ons.append("checkpointer + thread_id")
    return {"task": t.name, "shape": shape, "add_ons": add_ons}


tasks = [
    Task("password reset", steps_known_in_advance=True),
    Task("weekly churn report from SQL + CRM", steps_known_in_advance=True, long_running=True),
    Task("why is Acme complaining about their invoice?", steps_known_in_advance=False),
    Task("refund Acme's duplicate charge", steps_known_in_advance=False, consequential_action=True),
    Task("audit all 400 enterprise accounts for billing drift", steps_known_in_advance=False,
         fits_one_context=False, parts_loosely_coupled=True, long_running=True),
    Task("migrate a tangled billing module", steps_known_in_advance=False,
         fits_one_context=False, parts_loosely_coupled=False, long_running=True),
]

decisions = [choose_architecture(t) for t in tasks]
for d in decisions:
    print(f"{d['task'][:48]:48s} -> {d['shape']:36s} {d['add_ons']}")

shapes = {d["task"]: d["shape"] for d in decisions}
assert shapes["password reset"] == "workflow"
assert shapes["why is Acme complaining about their invoice?"] == "single agent"
assert shapes["audit all 400 enterprise accounts for billing drift"] == "multi-agent"
assert shapes["migrate a tangled billing module"] == "single agent + context engineering"
assert "interrupt approval gate" in decisions[3]["add_ons"]
# multi-agent appears exactly once among six realistic tasks: it is the exception
assert sum(d["shape"] == "multi-agent" for d in decisions) == 1
```

The last assert is the lesson. Across six realistic tasks for this agent, multi-agent is the right shape once. The account audit is broad (400 accounts do not fit one context), and each account can be checked independently, so workers can return condensed findings to an orchestrator. The billing migration is also broad, but every part touches every other part, so one agent with compaction and a notes file beats four agents passing half-understood diffs to each other.

## How LangGraph does it

The table above names a primitive for every pattern. Here are the three shapes of the decision tree compiled as real graphs, with stub nodes in place of model calls so the control flow is the object of study. Each stub records that it ran, so we can assert on the path taken and count model calls per shape.

```{code-cell} python
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class S(TypedDict, total=False):
    request: str
    route: str
    accounts: list
    findings: Annotated[list, operator.add]   # fan-in: parallel writes append
    calls: Annotated[list, operator.add]      # every "model call" logs here
    steps: int
    answer: str


# 1. WORKFLOW: route, then a fixed chain. The developer owns every edge.
def classify(s):  return {"route": "reset" if "password" in s["request"] else "billing", "calls": ["classify"]}
def reset(s):     return {"answer": "reset link sent", "calls": ["reset"]}
def lookup(s):    return {"calls": ["lookup"]}
def draft(s):     return {"answer": "billing draft", "calls": ["draft"]}

wf = StateGraph(S)
for name, fn in [("classify", classify), ("reset", reset), ("lookup", lookup), ("draft", draft)]:
    wf.add_node(name, fn)
wf.add_edge(START, "classify")
wf.add_conditional_edges("classify", lambda s: s["route"], {"reset": "reset", "billing": "lookup"})
wf.add_edge("lookup", "draft")
wf.add_edge("reset", END)
wf.add_edge("draft", END)
workflow = wf.compile()


# 2. SINGLE AGENT: the model chooses whether to loop. Here a stub "decides" to
#    call tools twice, then answer; a real model would decide from the transcript.
def agent(s):
    n = s.get("steps", 0) + 1
    return {"steps": n, "calls": ["agent"], "answer": "done" if n > 2 else ""}
def tools(s):     return {"calls": []}   # tools are not model calls

ag = StateGraph(S)
ag.add_node("agent", agent)
ag.add_node("tools", tools)
ag.add_edge(START, "agent")
ag.add_conditional_edges("agent", lambda s: END if s["answer"] else "tools", ["tools", END])
ag.add_edge("tools", "agent")
single_agent = ag.compile()


# 3. MULTI-AGENT (orchestrator-workers): Send creates one worker task per account
#    at runtime; the operator.add reducer on `findings` is the fan-in.
def orchestrator(s): return {"calls": ["orchestrator"]}
def fan_out(s):      return [Send("worker", {"accounts": [a]}) for a in s["accounts"]]
def worker(s):       return {"findings": [f"{s['accounts'][0]}: ok"], "calls": ["worker"]}
def synthesize(s):   return {"answer": f"{len(s['findings'])} accounts checked", "calls": ["synthesize"]}

ma = StateGraph(S)
ma.add_node("orchestrator", orchestrator)
ma.add_node("worker", worker)
ma.add_node("synthesize", synthesize)
ma.add_edge(START, "orchestrator")
ma.add_conditional_edges("orchestrator", fan_out, ["worker"])
ma.add_edge("worker", "synthesize")
ma.add_edge("synthesize", END)
multi_agent = ma.compile()

runs = {
    "workflow (reset)":   workflow.invoke({"request": "password reset please"}),
    "workflow (billing)": workflow.invoke({"request": "invoice is wrong"}),
    "single agent":       single_agent.invoke({"request": "why is Acme upset?"}),
    "multi-agent x5":     multi_agent.invoke({"accounts": [f"acct{i}" for i in range(5)]}),
}
for name, out in runs.items():
    print(f"{name:20s} model calls={len(out['calls']):2d} path={out['calls']} answer={out['answer']!r}")

assert runs["workflow (reset)"]["calls"] == ["classify", "reset"]        # router skipped the billing chain
assert len(runs["single agent"]["calls"]) == 3                           # loop count chosen at runtime
assert runs["multi-agent x5"]["answer"] == "5 accounts checked"          # reducer merged 5 parallel writes
assert len(runs["multi-agent x5"]["calls"]) == 7                         # 1 + N + 1: cost grows with N
```

Map each primitive back to the plain-Python version:

| Plain Python | LangGraph | What it does underneath |
|---|---|---|
| `if route == "reset": ... else: ...` | `add_conditional_edges(node, fn, mapping)` | After the node's super-step, the runtime calls `fn(state)` and schedules the named node next |
| `a(); b(); c()` | `add_edge("a", "b")` | A fixed trigger: when `a` writes, `b` runs in the next super-step |
| `while not done:` | conditional edge that points back to `agent` | The loop is a cycle in the graph; `recursion_limit` (1000 super-steps by default in current docs) bounds it |
| `for acct in accounts: pool.submit(worker, acct)` | `Send("worker", {...})` from a conditional edge | Creates one task per `Send` in the next super-step, each with its own input |
| `results.extend(r)` after joining futures | `Annotated[list, operator.add]` | The reducer merges concurrent writes to one channel instead of raising a conflict |
| `input("approve?")` | `interrupt(value)` + checkpointer | Saves state, raises out of the run, re-enters the node from the top on `Command(resume=...)` |

The model-call count is the cost line an interviewer is looking for. The workflow spent two calls on a password reset. The agent spent three because it chose to. The orchestrator-workers graph spends `N + 2`, and in a real system each worker is itself an agent loop, so it is closer to `N × (steps per worker) + 2`. Multi-agent systems buy breadth with tokens. Say that out loud.

### The same classification with a real model

The decision tree needs judgment on its first question ("can you write the steps down?") when the request is new. A cheap model with structured output can make that call and a human can audit it.

```{code-cell} python live
import os
from typing import Literal

from pydantic import BaseModel
from langchain.chat_models import init_chat_model


class ArchitectureChoice(BaseModel):
    steps_known_in_advance: bool
    fits_one_context: bool
    parts_loosely_coupled: bool
    consequential_action: bool
    rationale: str


model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
judge = model.with_structured_output(ArchitectureChoice)
for text in ["Reset my password", "Why is Acme complaining about their invoice?",
             "Audit all 400 enterprise accounts for billing drift"]:
    c = judge.invoke(f"Classify this support/ops request for architecture selection: {text}")
    print(text, "->", choose_architecture(Task(text, c.steps_known_in_advance, c.fits_one_context,
                                                 c.parts_loosely_coupled, c.consequential_action)))
```

The model fills the features; your code applies the tree. That is the "model chooses, software executes" split applied to architecture review.

### Comparison tables

#### Workflow vs single agent vs multi-agent

| Property | Workflow | Single agent | Multi-agent |
|---|---|---|---|
| Who chooses the path | Developer | Model, within bounds | Orchestrator model, plus each worker |
| Model calls per request | Fixed, small | Variable, bounded by step budget | Roughly workers × steps + orchestration |
| Predictability | High | Medium | Low |
| Debugging | Read the graph | Read one trajectory | Read N trajectories plus handoffs |
| Good for | Known procedures, high volume | Open-ended tasks in one domain | Broad, parallelizable research or audits |
| Main failure | Rigid on novel input | Loops, drift, wrong tool | Lost context at handoffs, duplicated work, cost |
| Eval style | Unit tests per node, output checks | Trajectory + outcome evals | Per-worker evals + end-to-end outcome |

#### Patterns that are often confused

| Pair | The difference |
|---|---|
| Prompt chaining vs planning | A chain's steps are fixed in code; a plan's steps are generated at runtime and can be revised. |
| Routing vs tool use | Routing picks one handler before work starts; tool use picks actions inside a loop, repeatedly. |
| Parallelization vs multi-agent | Parallelization fans out calls or nodes inside one design; multi-agent fans out whole loops with separate contexts. |
| Reflection vs evaluation | Reflection improves one output at run time; evaluation measures many outputs offline or in monitoring. |
| Memory vs RAG | Memory stores what the agent learned about users and tasks; RAG retrieves from a corpus someone else maintains. |
| Checkpointer vs Store | A checkpointer saves one thread's state step by step; a Store holds data shared across threads. |
| Guardrails vs HITL | Guardrails are automated checks; HITL is a human decision at a specific point. Consequential actions often need both. |
| MCP vs A2A | MCP connects one agent to tools and data; A2A connects agents to other agents as peers. |
| Function calling vs MCP (course framing) | Function calling is how the model asks the application for a tool; MCP is how the application discovers and invokes tools on a server. |

#### Where each LangGraph primitive shows up

| Primitive | Patterns that use it |
|---|---|
| `StateGraph` + typed state + reducers | All of them |
| `add_conditional_edges` | Routing, agent loop, reflection loop, goal checks |
| `Send` | Parallelization (dynamic), map-reduce, orchestrator-workers |
| `Command(goto=..., update=...)` | Routing from inside a node, handoffs between agents |
| `ToolNode` / `create_agent` | Tool use, agent loop |
| Checkpointer + `thread_id` | HITL, durable execution, short-term memory, time travel |
| `interrupt` | HITL, clarification questions, approval gates |
| `Store` | Long-term memory, user preferences, shared notes |
| Subgraph as a node | Multi-agent, reusable components |
| `RetryPolicy` | Exception handling on flaky nodes |

## What goes wrong in production

1. **Pattern before diagnosis.** Symptom: a new pattern ships and the metric does not move. Cause: the pattern treats a cause the system does not have (reflection on a retrieval failure). Fix: every design change starts from a failing trace and names the cause in the PR description; the eval that exposed it becomes a regression case.

2. **Pattern stacking.** Symptom: p95 latency triples over a quarter and nobody can say why. Cause: each fix added a node (router, critic, second retriever, summarizer) and none were removed. Fix: attach a measured benefit to every node and delete nodes whose ablation does not hurt the eval. Anthropic's harness posts make the same point about removing harness components as models improve.

3. **Multi-agent for a coupled task.** Symptom: workers produce contradictory outputs; the synthesizer papers over them. Cause: parts share state that no worker can see. Fix: go back one leaf on the tree: single agent plus context engineering, or tighter worker contracts with explicit inputs and outputs.

4. **Agent for a known procedure.** Symptom: a password reset sometimes takes 11 steps and sometimes 2. Cause: a model is choosing a path that a developer could have written down. Fix: route known intents to workflows; keep the agent for the open-ended tail.

5. **Cross-cutting checks attached to one leaf only.** Symptom: approval works in the agent path but a workflow path also calls `issue_refund` with no gate. Cause: the gate was added to a node, not to the capability. Fix: gate the tool itself (wrap the side-effecting function), so every path that can call it passes through `interrupt`.

6. **Unbounded fan-out.** Symptom: an audit over 40,000 accounts creates 40,000 concurrent worker tasks and rate-limits every downstream API. Cause: `Send` per item with no batching or concurrency cap. Fix: batch items per worker, cap concurrency (`max_concurrency` in the run config), and apply backpressure at the tool layer.

## What changed by 2026

- **The catalog stabilized; selection became the skill.** The five workflow patterns in Anthropic's "Building effective agents" (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) plus the agent loop cover most production designs. Interviews now probe *choice and justification* more than definitions.
- **Harnesses package the common stack.** `create_agent` in LangChain 1.x ships many table rows as middleware: retries, summarization, human-in-the-loop, call limits, PII handling, tool selection. Deep Agents adds planning, a file system for context, and subagents on top. The table still matters because you must know which row each middleware implements, and when you need it.
- **Context engineering moved into the table.** "Quality decays after many steps" is now a first-class symptom with named treatments: compaction, structured note-taking, just-in-time retrieval, and subagents that return condensed summaries, per Anthropic's context-engineering guidance.
- **Separating generation from evaluation became standard.** Anthropic's 2026 harness-design post reports that models tend to praise their own work, and uses a separate evaluator with a gradable rubric. That sharpened the reflection row: self-critique in the same context is weak; a separate evaluator with checkable criteria is strong.
- **Reasoning models changed the planning row.** Explicit "think step by step" scaffolding is no longer a pattern you add. Planning is about externalized, checkable state (a todo list, a plan in graph state), and verification is about outcomes, because written reasoning is not a faithful audit log.
- **Protocols entered the table.** MCP for tools and A2A for agent peers are now rows you choose, not research topics. Lessons 10 and 15 cover them.

## Interview questions

**Q1. Walk me through how you decide between a workflow, a single agent, and a multi-agent system.**

Strong: I start by asking whether I can write the steps down before the run. If yes, it is a workflow, because it is cheaper, testable per node, and predictable. If the next step depends on what the system discovers, it is an agent, with step, time, and cost bounds. I only split into multiple agents when the work does not fit one context and it decomposes into parts that need little shared state mid-run, like auditing independent accounts. Separately, any consequential action gets an approval gate and any long run gets a checkpointer, whatever the shape. Most real systems end up as a workflow with a small agentic core.

Weak: "Multi-agent is the most powerful, so I would start there."

**Q2. Your support agent gives wrong answers about refund policy. Which pattern do you add?**

Strong: None yet. I pull failing traces and look at what was retrieved. If the policy document retrieved was stale or the wrong section, the fix is in retrieval: freshness, chunking, maybe hybrid search and a reranker, and I evaluate retrieval separately from generation. If retrieval was right and the answer was wrong, then I look at the prompt or a separate evaluator with a checkable rubric. Adding a self-critique pass before diagnosing would reread the same wrong document and approve it.

Weak: "Add reflection so it double-checks itself."

**Q3. How many model calls does an orchestrator-workers design cost compared to a single agent?**

Strong: Roughly one orchestration call to plan, one synthesis call, and then each worker's own loop, so about N workers times their steps, plus two. A single agent on the same task is its step count. So multi-agent buys breadth and parallel wall-clock time with a multiple of the tokens. That trade is worth it when the task is broad and parallelizable and the value is high. It is not worth it for a task one context can hold.

Weak: "About the same, since the work is the same."

**Q4. When does multi-agent orchestration break down?** *(reported at Cognition)*

Strong: It breaks when the parts are coupled. Each agent sees only its slice, so decisions made in one context conflict with assumptions in another, and the handoff messages lose the reasoning behind choices. You see duplicated work, contradictory outputs, and a synthesizer that hides the conflict. It also breaks on debuggability: you have N trajectories and the bug is in the gaps between them. The fixes are to keep coupled work in one context with good context engineering, and to give workers narrow contracts and have them return condensed, structured results.

Weak: "When the agents disagree."

**Q5. Map these to LangGraph primitives: routing, fan-out, human approval, crash recovery.**

Strong: Routing is a conditional edge, or `Command(goto=...)` from inside a node when the node also updates state. Dynamic fan-out is `Send` from a conditional edge, one task per item, with a reducer like `operator.add` on the output key to fan in. Human approval is `interrupt()`, which needs a checkpointer and a `thread_id`; you resume with `Command(resume=...)`, and the interrupted node reruns from the top, so side effects before the interrupt must be idempotent. Crash recovery is the same checkpointer: re-invoke with the same `thread_id` and the run continues from the last saved super-step.

Weak: "LangGraph has nodes for all of those."

**Q6. What is the difference between reflection and evaluation?**

Strong: Reflection is a runtime pattern: generate, critique, revise, for one output, inside a request, bounded by a retry cap. Evaluation is measurement across many outputs, offline on a dataset before shipping and online on sampled traffic after. Reflection can improve a single answer; only evaluation tells you whether the system got better. Current practice also separates the critic from the generator, because a model grading its own work in the same context tends to approve it.

Weak: "They are the same thing, the model checks its answer."

**Q7. A stakeholder says, "make the agent more autonomous." What do you do?**

Strong: I turn the wish into a symptom. Which tasks fail or need a human today, how often, and what do their traces show? Autonomy usually means removing a human step, so I would look at which approval gates have near-100 percent approval rates on low-risk actions and consider raising the threshold there with monitoring, while keeping gates on irreversible actions. I would not restructure into more agents without a failing task that needs it.

Weak: "Give it access to more tools and let it plan."

## Build it yourself

Extend the recommender and the decision tree into a design-review assistant.

1. Add a `cause` field to `Pattern` and require `recommend` to return `(pattern, cause)` pairs, so every recommendation names what it treats.
2. Add rows for guardrails (18), memory (08), and exception handling (12) with keywords, and a symptom string for each that must route to it.
3. Add a `max_cost` argument to `recommend` that filters out patterns above that cost, and return an explicit "escalate to design review" string when every match is filtered.
4. Add a `tokens_per_call` estimate and a function `estimate_calls(shape, n_items, steps)` that returns the model-call count for each of the three graph shapes above.

Acceptance checks:

- `recommend("Stripe 503 errors kill the run")` returns exception handling first.
- `recommend("Slack lookups are serial", max_cost=1)` returns the escalation string, because parallelization costs 2.
- `estimate_calls("multi-agent", n_items=5, steps=1) == 7`, matching the compiled graph above.
- `estimate_calls("workflow", ...)` does not depend on `n_items` when routing to the reset path.
- Every `Pattern` in the catalog has a non-empty lesson, primitive, and question (assert over the list).

## Key takeaways

- Start from an observable symptom and a named cause, never from a pattern name.
- If you can write the steps down before the run, build a workflow; it is cheaper, testable, and predictable.
- Use an agent when the path depends on discoveries, and bound it in code by steps, time, and cost.
- Use multiple agents only when the work exceeds one context and splits into loosely coupled parts; they multiply token cost.
- Approval gates and checkpoints are cross-cutting: they attach to capabilities and long runs, not to one architecture.
- Every LangGraph primitive maps to a few lines of plain Python: conditional edges are `if`, `Send` is a task pool, reducers are merge rules, `interrupt` is a saved pause.
- Every added node should earn its place on an eval, and be removed when it stops earning it.
