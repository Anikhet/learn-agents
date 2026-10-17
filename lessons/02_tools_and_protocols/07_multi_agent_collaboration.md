# 07. Multi-Agent Collaboration

*Track A (agent engineering). Book: Chapter 7 (Multi-Agent Collaboration). Prerequisites: 05. Tool Use, 03. Parallelization, 06. Planning.*

A "multi-agent system" sounds like a new kind of software. It is not. It is the same loop from the orientation lesson, instantiated more than once, with a decision about what each instance can see and who is allowed to talk to whom. Once you see it that way, the design questions become ordinary systems questions: where does state live, who owns the task, what crosses the boundary, and what happens when one participant fails.

This lesson gives you the four topologies, the three ways to wire an agent into another agent, a from-scratch implementation of the two that matter most, and a hard look at when multiple agents make a system worse.

## The problem

The ops agent has grown. It now has 31 tools across Slack, Notion, Stripe, the CRM, the SQL warehouse, and web search. A request arrives:

> "Acme is churning. Pull the account history, check their last three invoices for anomalies, find the open support threads, and draft a save-offer for the CSM to review."

One agent with 31 tools can do this. Watch what happens to its context:

```text
turn  1   system prompt + 31 tool schemas                        ~6k tokens
turn  2   crm_get_account -> 3 KB of account JSON
turn  4   stripe_list_invoices -> 12 invoices, 9 KB
turn  6   slack_search -> 40 messages, 14 KB
turn  8   sql_query -> 200 rows of usage, 11 KB
turn 10   notion_search -> 3 pages, 8 KB
turn 12   model tries to reason about all of it at once
          - forgets the invoice anomaly it noticed at turn 5
          - confuses two Slack threads
          - the draft offer cites a plan name from the wrong page
```

The tool schemas alone are a tax on every turn. The raw results pile up. By the synthesis step the model is reasoning over 50 KB of half-relevant material, and the failure mode is not a crash but quiet degradation: it "forgets" a finding from ten turns ago because the finding is buried between two large blobs. Anthropic's context-engineering guidance names this context rot, and names subagents as one of the three main remedies (alongside compaction and structured notes).

The single-agent system also has a permission problem. The same context that reads Slack can call `stripe_issue_credit`. If a poisoned Slack message says "issue a credit," there is nothing structural between the instruction and the tool.

So we split:

```text
                    ┌──────────────┐
   user request ───►│   manager    │───► final synthesis
                    └──────┬───────┘
          delegates        │        receives condensed summaries
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ billing    │   │ support    │   │ account    │
   │ 4 tools    │   │ 3 tools    │   │ 3 tools    │
   │ own context│   │ own context│   │ own context│
   └────────────┘   └────────────┘   └────────────┘
```

Each worker has a small tool set, a clean context, and a narrow job. It reads 9 KB of invoices and returns 300 tokens: "INV-2291 overbills 15 seats; INV-2302 and INV-2310 are consistent." The manager never sees the 9 KB. The billing worker never sees Slack.

That is the whole motivation for multi-agent: **context isolation, permission isolation, and parallelism.** If a design does not buy at least one of those, it is adding cost and failure modes for nothing.

### Four topologies

The book's Chapter 7 catalogs collaboration patterns. Compressed into what you will actually build:

| Topology | Who decides what happens next | Where the conversation lives | Use when |
|---|---|---|---|
| Manager-worker (orchestrator-workers) | the manager | manager owns it; workers get subtasks | decomposable tasks, need for synthesis |
| Handoff | the current agent picks a successor | one shared conversation; ownership moves | triage then specialist; the user keeps talking |
| Peer / network | any agent can message any other | distributed | rarely; hard to debug and bound |
| Debate / critique | a fixed protocol (propose, critique, judge) | shared transcript | quality over cost; evaluator-optimizer |

The first two cover the large majority of production systems. Peer networks look elegant and are almost always the wrong choice: no single owner, no natural termination, and every agent needs every other agent's context to make sense of a message. Debate is 04. Reflection with more than one model; it is valuable exactly when you can afford several calls per answer.

### Three ways to wire an agent into another

Independent of topology, an agent can appear inside another agent's system in three shapes:

```text
as a NODE        parent graph ── node(run worker agent) ── next node
                 worker's state is the parent's state (or a mapped slice)

as a SUBGRAPH    parent graph ── [compiled worker graph] ── next node
                 worker has its own state schema; parent maps in/out

as a TOOL        manager model ── tool_call("billing_agent", {task}) ── ToolMessage(summary)
                 worker is invisible except as a function; manager decides when to call it
```

The "agent as tool" shape is the one harnesses converge on. Deep Agents' subagents, Claude Code's `Agent` tool, and the OpenAI Agents SDK's `agent.as_tool()` are all this: the manager's model sees a tool schema, calls it with a task string, and gets back a summary. The worker's whole transcript is hidden behind the `ToolMessage`. That is context isolation enforced by the interface.

Handoff is different in kind. In a handoff the worker does not return to the manager; it *becomes* the agent talking to the user. The conversation history transfers, the tool set changes, and the previous agent is out of the loop.

## Build it without a framework

The scripted model stands in for the real one so the wiring is the object of study. Each agent below gets its own script because each agent gets its own model calls; the number of turns per script must match exactly how many times that agent is invoked.

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

### Manager-worker with workers as tools

Start with the primitive: one agent loop, parameterized by model and tools. Then a worker is a fresh loop with a fresh message list, wrapped in a function whose signature is `task: str -> summary: str`. The manager gets those wrappers as tools. Nothing else is needed for isolation; the worker's transcript is a local variable that goes out of scope.

```{code-cell} python
import json
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


def run_loop(model, tools, system, user_text, max_steps=8):
    """The plain agent loop from 00. Every agent in this lesson is one of these."""
    by_name = {t.name: t for t in tools}
    messages = [SystemMessage(system), HumanMessage(user_text)]
    for _ in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return messages
        for call in ai.tool_calls:
            try:
                obs = json.dumps(by_name[call["name"]].invoke(call["args"]))
            except Exception as exc:
                obs = f"error: {type(exc).__name__}: {exc}"
            messages.append(ToolMessage(content=obs, tool_call_id=call["id"]))
    raise RuntimeError("step budget exhausted")


# Worker tool sets. The billing worker can never see Slack; support can never touch Stripe.
@tool
def stripe_list_invoices(customer: str) -> list:
    """List the customer's recent invoices with seat counts and amounts."""
    return [{"id": "INV-2291", "seats": 40, "amount": 1299}, {"id": "INV-2302", "seats": 25, "amount": 750},
            {"id": "INV-2310", "seats": 25, "amount": 750}]


@tool
def crm_get_contract(customer: str) -> dict:
    """Fetch contracted plan and seats."""
    return {"plan": "team", "seats": 25, "renewal": "2026-11-01"}


@tool
def slack_search(query: str) -> list:
    """Search support Slack channels. Returns message snippets."""
    return [{"ts": "2026-09-20", "text": "Acme asks why INV-2291 is $1299 not $750"},
            {"ts": "2026-09-25", "text": "Acme CTO: 'considering alternatives'"}]


WORKER_CONTRACT = ("Do the task with your tools, then reply with at most three sentences. "
                   "State facts and ids only; the manager will synthesize.")

billing_model = scripted(
    [("stripe_list_invoices", {"customer": "acme"}), ("crm_get_contract", {"customer": "acme"})],
    "INV-2291 bills 40 seats ($1299) against a 25-seat contract ($750): overbilled 15 seats. "
    "INV-2302 and INV-2310 match the contract. Renewal is 2026-11-01.",
)
support_model = scripted(
    [("slack_search", {"query": "Acme"})],
    "Two open threads: a 2026-09-20 complaint about INV-2291 pricing and a 2026-09-25 "
    "message from the CTO saying they are considering alternatives.",
)


TRANSCRIPT_LENS = {}


def make_worker_tool(name, description, model, tools):
    """Wrap a whole agent as a function. Its transcript never leaves this scope."""
    def worker(task: str) -> str:
        transcript = run_loop(model, tools, WORKER_CONTRACT, task)
        TRANSCRIPT_LENS[name] = len(transcript)        # for the assertion below
        return transcript[-1].content
    worker.__doc__ = description
    return tool(name)(worker)


billing_agent = make_worker_tool(
    "billing_agent", "Investigate invoices and contracts for a customer.", billing_model,
    [stripe_list_invoices, crm_get_contract])
support_agent = make_worker_tool(
    "support_agent", "Find and summarize open support threads for a customer.", support_model,
    [slack_search])

manager_model = scripted(
    [("billing_agent", {"task": "Check Acme's last three invoices for anomalies."}),
     ("support_agent", {"task": "Find open Acme support threads."})],
    "Save-offer draft: credit the 15 overbilled seats on INV-2291 ($450), acknowledge the "
    "2026-09-20 thread, and offer a renewal call before 2026-11-01 given the CTO's message.",
)

manager_transcript = run_loop(
    manager_model, [billing_agent, support_agent],
    "You are the ops manager. Delegate, then synthesize a save-offer draft.",
    "Acme is churning. Investigate and draft a save-offer.")

for m in manager_transcript:
    print(f"{type(m).__name__:12s} {(m.content or str(m.tool_calls))[:100]}")

# The manager saw two short summaries, never the raw invoices or Slack messages.
manager_text = " ".join(m.content for m in manager_transcript if isinstance(m, ToolMessage))
assert "INV-2291" in manager_text and "2026-09-25" in manager_text
assert '"id": "INV-2302"' not in manager_text          # raw invoice JSON never reached the manager
assert not any("stripe_list_invoices" in str(m.tool_calls) for m in manager_transcript if isinstance(m, AIMessage))
assert TRANSCRIPT_LENS["billing_agent"] == 6           # system, human, ai(2 calls), 2 tool msgs, final
assert manager_transcript[-1].content.startswith("Save-offer draft")
```

The assertion on `TRANSCRIPT_LENS` is the point: the billing worker produced a six-message transcript (system, human, AI with two calls, two tool results, final AI), and exactly one sentence of it reached the manager. The manager's context contains two `ToolMessage`s of three sentences each. Anthropic's context-engineering guidance describes exactly this shape: subagents explore with their own context and return condensed summaries of one to two thousand tokens.

The other thing to notice is `WORKER_CONTRACT`. A worker without an output contract returns whatever it feels like, and the manager has to parse prose. Say exactly what comes back: facts, ids, a length cap. In production this is a Pydantic schema and 05. Tool Use's structured-output pattern.

### Handoff: transfer of ownership

Now the other shape. A triage agent talks to the user first. It decides the conversation belongs to billing. Instead of asking billing a question and relaying the answer, it *hands off*: the billing agent inherits the conversation and continues with the user directly.

The mechanism is a special tool. Calling `transfer_to_billing` does not return a result; it changes which agent runs next and which tools are bound.

```{code-cell} python
from dataclasses import dataclass, field


@dataclass
class AgentSpec:
    name: str
    system: str
    model: object
    tools: list
    handoffs: list = field(default_factory=list)   # names of agents this one may transfer to


def run_with_handoffs(agents: dict, start: str, user_text: str, max_hops=3):
    """One shared conversation; ownership moves when an agent calls transfer_to_*.
    Tool calls that are not handoffs execute normally."""
    messages = [HumanMessage(user_text)]
    current = agents[start]
    hops = []
    for _ in range(max_hops * 4):
        by_name = {t.name: t for t in current.tools}
        ai = current.model.invoke([SystemMessage(current.system), *messages])
        messages.append(ai)
        if not ai.tool_calls:
            return current.name, hops, messages
        for call in ai.tool_calls:
            if call["name"].startswith("transfer_to_"):
                target = call["name"].removeprefix("transfer_to_")
                if target not in current.handoffs:
                    messages.append(ToolMessage(content=f"error: {current.name} may not hand off to {target}",
                                                tool_call_id=call["id"], status="error"))
                    continue
                # The handoff result is itself an observation, so the transcript stays valid.
                messages.append(ToolMessage(content=f"transferred to {target}", tool_call_id=call["id"]))
                hops.append((current.name, target))
                current = agents[target]
            else:
                obs = json.dumps(by_name[call["name"]].invoke(call["args"]))
                messages.append(ToolMessage(content=obs, tool_call_id=call["id"]))
    raise RuntimeError("too many hops")


agents = {
    "triage": AgentSpec(
        "triage", "Classify the request and transfer to the right specialist.",
        scripted([("transfer_to_billing", {})]), tools=[], handoffs=["billing"]),
    "billing": AgentSpec(
        "billing", "You own billing conversations. Resolve invoice questions.",
        scripted([("crm_get_contract", {"customer": "acme"})],
                 "Your contract is for 25 seats; INV-2291 billed 40. I will raise a credit request."),
        tools=[crm_get_contract], handoffs=[]),
}

owner, hops, transcript = run_with_handoffs(agents, "triage", "Why is INV-2291 $1299? We have 25 seats.")
print("final owner:", owner, "| hops:", hops)
for m in transcript:
    print(f"{type(m).__name__:12s} {(m.content or str(m.tool_calls))[:90]}")
assert owner == "billing" and hops == [("triage", "billing")]
assert transcript[-1].content.startswith("Your contract is for 25 seats")
```

Compare the two mechanisms side by side:

```text
DELEGATION (worker as tool)            HANDOFF (transfer)
manager ──call──► worker               triage ──transfer──► billing
manager ◄─summary── worker             billing ──────────► user
manager keeps talking to the user      billing keeps talking to the user
worker context: fresh, isolated        billing context: inherits the conversation
manager decides when to stop           billing decides when to stop
```

Delegation is for *work*. Handoff is for *conversations*. If the user needs to keep talking to whoever has the right tools, hand off. If you need a result to combine with other results, delegate.

## How LangGraph does it

LangGraph gives you all three wiring shapes, and the choice between them is a choice about state.

**Agent as a node** shares the parent's state schema. The worker sees `messages` and appends to it. No isolation. Use it only when the agent is a step in a pipeline and you *want* the shared transcript.

**Agent as a subgraph** has its own `StateGraph` with its own schema. When you add a compiled graph as a node, LangGraph maps parent keys to child keys by name if the schemas overlap, or you wrap it in a function that builds the child's input and extracts the child's output. That wrapper is where isolation happens: you pass in a task string, you get back a summary, and the child's `messages` never touch the parent's.

**Agent as a tool** is the from-scratch `make_worker_tool`, and `ToolNode` executes it like any other tool. Parallel delegation falls out for free because `ToolNode` runs a turn's calls concurrently.

For orchestrator-workers where the *number* of workers is decided at runtime (one per invoice, one per document), the primitive is `Send`. A conditional edge returns a list of `Send("worker", payload)` objects; the runtime spawns one worker per payload, runs them in parallel, and a reducer on the parent state collects the results (03. Parallelization has the full treatment).

```{code-cell} python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send


# --- the billing worker as a real subgraph with its own state schema ---
class WorkerState(TypedDict):
    task: str
    messages: Annotated[list, operator.add]
    summary: str


worker_model = scripted(
    [("stripe_list_invoices", {"customer": "acme"})],
    "INV-2291 overbills 15 seats versus contract; INV-2302 and INV-2310 are consistent.",
)
worker_tools = {t.name: t for t in [stripe_list_invoices, crm_get_contract]}


def worker_start(state: WorkerState):
    return {"messages": [SystemMessage(WORKER_CONTRACT), HumanMessage(state["task"])]}


def worker_agent(state: WorkerState):
    ai = worker_model.invoke(state["messages"])
    if ai.tool_calls:
        obs = [ToolMessage(content=json.dumps(worker_tools[c["name"]].invoke(c["args"])),
                           tool_call_id=c["id"]) for c in ai.tool_calls]
        return {"messages": [ai, *obs]}
    return {"messages": [ai], "summary": ai.content}


def worker_route(state: WorkerState):
    return END if state.get("summary") else "agent"


wb = StateGraph(WorkerState)
wb.add_node("start", worker_start)
wb.add_node("agent", worker_agent)
wb.add_edge(START, "start")
wb.add_edge("start", "agent")
wb.add_conditional_edges("agent", worker_route, ["agent", END])
billing_subgraph = wb.compile()


# --- the parent: fan out one worker per subtask with Send, collect with a reducer ---
class ManagerState(TypedDict):
    request: str
    subtasks: list[str]
    findings: Annotated[list[str], operator.add]     # each worker appends one summary
    draft: str


def plan(state: ManagerState):
    # In a real system the manager model produces this list (06. Planning).
    return {"subtasks": ["Check Acme invoices for anomalies", "Summarize open Acme threads"]}


def fan_out(state: ManagerState):
    return [Send("worker", {"task": t}) for t in state["subtasks"]]


support_subgraph_model = scripted("Two open threads: INV-2291 pricing complaint; CTO considering alternatives.")


def worker(payload: dict):
    """Isolation boundary: build the child's input, return only its summary.
    The child's messages never enter ManagerState."""
    if "invoice" in payload["task"]:
        out = billing_subgraph.invoke({"task": payload["task"], "messages": [], "summary": ""})
        return {"findings": [out["summary"]]}
    return {"findings": [support_subgraph_model.invoke([HumanMessage(payload["task"])]).content]}


def synthesize(state: ManagerState):
    return {"draft": "Save-offer: " + " | ".join(sorted(state["findings"]))}


mb = StateGraph(ManagerState)
mb.add_node("plan", plan)
mb.add_node("worker", worker)
mb.add_node("synthesize", synthesize)
mb.add_edge(START, "plan")
mb.add_conditional_edges("plan", fan_out, ["worker"])
mb.add_edge("worker", "synthesize")
mb.add_edge("synthesize", END)
manager_graph = mb.compile()

result = manager_graph.invoke({"request": "Acme is churning", "subtasks": [], "findings": [], "draft": ""})
print(result["draft"])
assert len(result["findings"]) == 2
assert "messages" not in result                      # the worker's transcript stayed inside the subgraph
assert "INV-2291" in result["draft"] and "CTO" in result["draft"]
```

What each primitive is doing, in terms of the from-scratch code:

| Primitive | From-scratch equivalent |
|---|---|
| `billing_subgraph.invoke({...})` inside `worker` | `run_loop(...)` inside `make_worker_tool`; a fresh transcript per call |
| `WorkerState` vs `ManagerState` | the local `messages` variable in the worker function vs the manager's list |
| `Send("worker", payload)` | a `ThreadPoolExecutor.map` over subtasks, one loop each |
| `findings: Annotated[list, operator.add]` | `results.append(...)` as each worker returns |
| `Command(goto="billing")` from a node (not shown) | the `current = agents[target]` line in `run_with_handoffs` |

That last row is how handoffs are expressed in LangGraph: a node returns `Command(goto="billing_agent", update={...})` and the runtime routes there, carrying the shared `messages`. Because the state is shared, that is a handoff, not a delegation.

### Two harnesses that show the idea clearly

The OpenAI Agents SDK makes handoff a first-class object: `handoffs=[billing_agent]` on the triage agent generates a `transfer_to_billing_agent` tool, and `Runner` swaps the active agent when it is called. It is the cleanest illustration of the pattern because there is nothing else in the way. Deep Agents (built on LangGraph) makes delegation first-class: `subagents=[{...}]` creates a `task` tool whose calls run a fresh agent with its own context and return a summary. Both are installed here and run only in live mode.

```{code-cell} python live
import asyncio
import os

from agents import Agent, Runner

# Handoff, live. The SDK generates transfer_to_* tools from `handoffs`.
billing = Agent(name="billing_agent", handoff_description="Invoices, refunds, plan pricing.",
                instructions="Resolve billing questions using the conversation so far.")
triage = Agent(name="triage", instructions="Route the user to the right specialist.",
               handoffs=[billing])
run = asyncio.run(Runner.run(triage, "Why is INV-2291 $1299 when we have 25 seats?"))
print("final agent:", run.last_agent.name)
print(run.final_output[:200])

# Delegation, live. Deep Agents' `subagents` become a `task` tool the manager calls.
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
manager = create_deep_agent(
    model=model,
    tools=[],
    subagents=[{"name": "billing-agent", "description": "Investigates invoices vs contracts.",
                "system_prompt": WORKER_CONTRACT, "tools": [stripe_list_invoices, crm_get_contract]},
               {"name": "support-agent", "description": "Finds open support threads.",
                "system_prompt": WORKER_CONTRACT, "tools": [slack_search]}],
    system_prompt="Delegate to subagents, then draft a save-offer.")
out = manager.invoke({"messages": [HumanMessage("Acme is churning. Investigate and draft a save-offer.")]})
print(out["messages"][-1].content[:400])
```

## What goes wrong in production

1. **Telephone-game degradation.** Symptom: the final answer cites a number no worker produced. Cause: each hop paraphrases; three hops of paraphrase is three chances to drift. Fix: workers return structured records with ids and quotes, not prose; the manager cites worker output verbatim where numbers matter; add a verification step that checks the draft against the findings.

2. **Duplicated or contradictory work.** Symptom: billing and support both call the CRM and return different account states. Cause: overlapping tool sets and no shared fact base. Fix: non-overlapping tools per worker; if two workers need the same fact, the manager fetches it once and passes it in the task.

3. **Lost context at handoff.** Symptom: the billing agent asks the user a question triage already answered. Cause: the handoff transferred control but filtered the history, or the specialist's system prompt ignores prior turns. Fix: define exactly what crosses the boundary (full history, or a summary plus the last N turns) and test it.

4. **No termination across agents.** Symptom: triage hands to billing, billing hands back to triage, forever. Cause: each agent bounds its own loop but nothing bounds the hops. Fix: a hop budget in the coordinator (our `max_hops`), and a rule that an agent may not transfer to the agent that transferred to it without new information.

5. **Partial failure in fan-out.** Symptom: one of five workers times out and the synthesis silently omits its area. Cause: the reducer collected four results and nobody checked the count. Fix: workers return a result-or-error record; synthesis asserts on the expected count and states what is missing; retry policy per worker node.

6. **Privilege leakage.** Symptom: a worker that only needed read access ends up with the manager's write tools. Cause: tools passed by reference to a shared list. Fix: each worker gets an explicit allowlist; the manager holds write tools itself behind approval gates (13. Human in the Loop).

7. **Cost multiplication.** Symptom: a request that cost $0.05 as one agent costs $0.60 as six. Cause: each worker re-reads a system prompt, tool schemas, and the task; the manager pays for planning and synthesis turns. Fix: measure per-request cost against the single-agent baseline; keep workers only where they buy isolation or parallelism you can name.

8. **Untraceable failures.** Symptom: "the answer was wrong" and six transcripts to read. Cause: no correlation between the manager's tool call and the worker's run. Fix: propagate a run id into every worker; nest worker traces under the manager's span (32. Observability and Tracing).

## What changed by 2026

- **Subagents as a context-engineering tool, not an org chart.** The framing moved from "specialist personas" to "context isolation with a condensed return." Anthropic's context-engineering guidance treats subagents as one of the three long-horizon techniques, with the explicit purpose of keeping heavy exploration out of the main context.
- **Agent-as-tool won.** Deep Agents' `task` tool, Claude Code's `Agent` tool, and the OpenAI Agents SDK's `as_tool()` all expose a worker as a function the manager's model calls. Handoff survives as the second primitive, mainly for conversational routing.
- **Separation of generation from evaluation.** Anthropic's harness-design writing on long-running application development argues for Planner, Generator, and Evaluator as separate agents, because a model grading its own work is unreliable. That is the debate/critique topology used for one specific reason.
- **Fresh context beats compaction for long jobs.** The same guidance found that resetting to a fresh context with a handoff artifact (a progress file, a feature list with pass/fail) outperformed compacting a long transcript. Multi-agent here means "many sequential sessions with durable state," not many concurrent personalities.
- **Managed multi-agent runtimes.** Anthropic's "Scaling Managed Agents" describes a session (append-only event log), a stateless harness loop, and interchangeable sandboxes; the OpenAI Agents API (public beta as of current docs) offers hosted orchestration with durable sessions. The topology is the same; where the state lives moved to the provider.
- **Protocols for the boundary.** A2A (15. Inter-Agent Communication) standardizes the delegation boundary across organizations: Agent Card for discovery, Task for the unit of work, Artifact for the result. Inside one codebase you still use a subgraph.
- **The default is still one agent.** Every vendor's guidance says the same thing the book says: start with a single well-tooled agent, split only when you can name the isolation or parallelism you are buying.

## Interview questions

**Q1. When would you split a single agent into multiple agents, and when would you refuse to?**

Strong: I split for one of three measurable reasons: context isolation, when raw tool output is degrading the main context and a worker can return a summary; permission isolation, when a component that reads untrusted input must not hold write tools; or parallelism, when subtasks are independent and latency matters. I refuse when the motivation is "it feels more organized," because each extra agent adds a paraphrase hop, a cost multiplier, a partial-failure mode, and a tracing problem. The test is: can I name what the boundary buys, and can I measure it against the single-agent baseline on the same eval set?

Weak: "Specialized agents are better at their specialties."

**Q2. Explain the difference between delegation and handoff.**

Strong: In delegation the manager calls a worker like a function: fresh context, task in, summary out, and the manager keeps the conversation and decides what happens next. In a handoff, control of the conversation itself transfers: the specialist inherits the history, gets a different tool set, and talks to the user directly; the previous agent is out. Delegation is for work you want to combine; handoff is for routing a conversation to whoever has the right tools. The OpenAI Agents SDK makes handoff explicit with `handoffs=[...]`; Deep Agents makes delegation explicit with `subagents=[...]`. In LangGraph, delegation is a subgraph or tool with its own state; handoff is `Command(goto=...)` over shared state.

Weak: "Handoff is when one agent passes the task to another."

**Q3. How do you keep a worker agent's context isolated from the manager in LangGraph?**

Strong: Give the worker its own state schema in its own compiled `StateGraph`, and wrap the invocation in a parent node that builds the child's input from a task string and returns only the child's summary field. Because the parent's state has no `messages` key from the child, the transcript cannot leak. The alternative is exposing the worker as a tool, which gives the same isolation through the `ToolMessage` boundary and lets `ToolNode` run several workers concurrently. What I avoid is adding the worker as a plain node over the shared `MessagesState`, since that is no isolation at all.

Weak: "Use a subgraph."

**Q4. A fan-out to five workers returns four results. What does your synthesis step do?**

Strong: It must know that five were expected. The reducer collecting results is append-only, so I carry the expected count, or the list of subtasks, in state and assert on it. Each worker returns a tagged record, success or error, rather than raising, so the missing one shows up as an explicit failure entry. Synthesis then either retries that worker under its node's retry policy, or produces an answer that states which area is unverified. What it never does is silently produce a confident summary over 80% of the evidence.

Weak: "Retry the failed one."

**Q5. Why is a peer-to-peer network of agents usually a bad design?**

Strong: There is no owner. Without an owner there is no natural termination, no single place to enforce budgets, and no one accountable for the final answer. Each agent needs enough context to interpret messages from every other, which defeats the isolation that justified splitting in the first place. Debugging means reading N transcripts with no tree structure. A manager-worker tree gives you one root span, one budget, one termination point, and clear data flow. I reach for peer designs only when the agents are owned by different organizations, and then I want a protocol like A2A with explicit task states, not free-form messaging.

Weak: "It's more flexible."

**Q6. How do you prevent multi-agent systems from degrading answers through paraphrase?**

Strong: Contract the boundary. Workers return structured records with identifiers, exact figures, and short quotes rather than narrative. The manager is instructed to cite worker output verbatim where numbers or ids appear. I add a cheap verification step that checks every claim in the final draft against the collected findings, and I evaluate the whole system end to end on a fixed set, comparing to a single-agent baseline, because the degradation is only visible at the output.

Weak: "Tell the workers to be accurate."

**Q7. What does "separate generation from evaluation" mean and why does it need two agents?**

Strong: A model grading its own output in the same context tends to approve it; it has already committed to the reasoning. Putting the evaluator in a fresh context with only the artifact and a rubric removes that bias and lets you give it different tools, like test runners, that the generator should not control. It is the evaluator-optimizer pattern from the building-effective-agents guidance and the Planner/Generator/Evaluator split in the harness-design writing. It costs extra calls, so it belongs where correctness matters more than latency.

Weak: "Have the agent check its work."

## Build it yourself

Extend `run_with_handoffs` and the `Send` fan-out:

1. Add a `max_hops` enforcement that raises `RuntimeError("hop budget")` after the configured number of transfers, and forbid an agent from transferring back to the agent that just transferred to it.
2. Change the handoff to carry a *summary* rather than the full history: when ownership moves, replace all prior messages with one `HumanMessage` containing a scripted summary plus the last user turn.
3. In the LangGraph manager, make `worker` return `{"findings": [{"task": ..., "status": "ok"|"error", "summary": ...}]}` and make a worker fail when its task contains the word "fail".
4. Make `synthesize` assert that `len(findings) == len(subtasks)` and include an "UNVERIFIED: task name" line for every error record.

Acceptance checks:

- A script where triage and billing transfer to each other stops with `RuntimeError("hop budget")` and never runs a fourth model call.
- After the summarized handoff, the billing agent's first model input contains exactly one `HumanMessage` before its own turn.
- With subtasks `["a", "fail b", "c"]`, the draft contains exactly one `UNVERIFIED` line and `len(findings) == 3`.
- The happy-path draft is unchanged.

## Key takeaways

- Multi-agent is the same loop instantiated more than once; the design questions are about state ownership and boundaries, not personas.
- Split only for context isolation, permission isolation, or parallelism you can name and measure; the default is one agent.
- Delegation calls a worker like a function and keeps the conversation; handoff transfers the conversation and the tool set.
- Agent as tool and agent as subgraph give isolation through the interface; agent as node over shared state gives none.
- `Send` plus an append reducer is orchestrator-workers when the worker count is decided at runtime; carry the expected count so partial failure is visible.
- Every boundary is a paraphrase hop; contract it with structured records and verify the synthesis against them.
- Separate generation from evaluation when correctness matters; a model grading its own output in the same context is not a grader.
