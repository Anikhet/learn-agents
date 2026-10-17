# 13. Human in the Loop

*Track A (agent engineering). Book: Chapter 13 (Human-in-the-Loop). Prerequisites: 00. What an agent actually is, 05. Tool Use, 08. Memory Management (for checkpointers). Companion: 18. Guardrails and Safety Patterns.*

The book frames human-in-the-loop as putting a person at the decision points where the agent's judgment is not enough: approving, correcting, escalating. That framing is right, and it hides the hard part. Asking a human is easy. *Waiting* for a human is the problem. The agent must stop mid-run, hold its state for minutes or days, survive deploys and crashes, and then continue exactly where it stopped, with the human's decision applied, without doing anything twice.

This lesson builds that machine. First by hand, so you see that "pause" means "serialize state and return." Then with LangGraph's `interrupt()` and `Command(resume=...)`, so you know what those primitives save, what they re-run, and which bug they will hand you if you do not read the fine print.

## The problem

The support agent is working a ticket from Slack:

> "Acme was double-charged on invoice in_1001. Please refund the duplicate."

It looks up Acme in the CRM, finds the invoice in Stripe, sees two charges of $1,200, and decides to refund one. The naive agent calls `stripe_refund` immediately. Most days that is fine. One day the model reads the wrong invoice, or the "duplicate" is actually a second seat purchase, and $1,200 leaves the company with nobody having looked.

The first fix everyone reaches for is a prompt line: "Always ask before issuing refunds." That is a suggestion, not a control. The tool is registered; the model can call it. The second fix is to put `input("Approve? ")` inside the tool. That works in a notebook and fails in production, because the agent runs in a worker process behind a Slack bot and nobody is at that terminal. The worker blocks a thread for hours, then gets killed by the next deploy, and the approval is lost.

```text
naive:     agent ── stripe_refund($1200) ──► money moves        nobody saw it

blocking:  agent ── input("approve?") ──► worker thread waits...
                                           deploy at 14:00 ─► process killed
                                           approval arrives 15:30 ─► nothing to resume

durable:   agent ── needs approval ──► save state, return "paused"
                                           │
                     (minutes, days, deploys, crashes)
                                           │
           human decides in Slack ──► load state, apply decision, continue
```

The durable version is the one you want. It has four parts, and every HITL system has all four whether it names them or not:

```text
1. POLICY       which actions pause?           code, not prompt
2. PAUSE        persist state + pending action  a checkpoint keyed by run id
3. DECISION     approve / edit / reject         a structured payload from a human
4. RESUME       load, apply, continue           exactly once, with an audit record
```

### Which actions pause

Gating everything is as bad as gating nothing. If the agent asks permission to read the CRM, reviewers learn to click "approve" without reading, and the gate becomes theater. Gate by risk:

| Action | Reversible? | Blast radius | Policy |
|---|---|---|---|
| `crm_lookup`, `stripe_get_invoice` | read-only | none | auto |
| `slack_post` to the requester's thread | mostly | one person | auto |
| `notion_update_page` | yes (history) | a team | auto, audited |
| `stripe_refund` under $100 | no | one customer | auto, audited |
| `stripe_refund` $100 or more | no | money | **approve** |
| `sql_execute` with `DELETE`/`UPDATE` | no | a table | **approve**, or not registered at all |
| `slack_post` to `#general` | no | the whole company | **approve** |

Notice the policy looks at arguments, not only tool names. A $40 refund and a $4,000 refund are the same tool. The policy is a function of `(tool, args, context)`, and it lives in your code where it can be tested.

## Build it without a framework

We reuse the scripted model from 00. What an agent actually is. The script replays fixed model turns, so the control flow is under study and the notebook runs without network access. A live cell later shows the same pattern with a real model.

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


from langchain_core.tools import tool

# A fake Stripe that honors idempotency keys, like the real one does.
LEDGER = {}      # idempotency_key -> refund record


@tool
def stripe_get_invoice(invoice_id: str) -> str:
    """Look up a Stripe invoice by id."""
    return f"{invoice_id}: Acme, two charges of $1200 on 2026-09-01 (ch_1, ch_2)"


@tool
def stripe_refund(charge_id: str, amount: int, idempotency_key: str = "") -> str:
    """Refund a charge. Irreversible. Amount in whole dollars."""
    key = idempotency_key or f"{charge_id}:{amount}"
    if key not in LEDGER:                      # a replay returns the original result
        LEDGER[key] = {"charge": charge_id, "amount": amount}
    return f"refunded ${LEDGER[key]['amount']} on {LEDGER[key]['charge']}"


def needs_approval(call: dict) -> str | None:
    """The policy. Returns a reason when the call must pause, None when it may run."""
    if call["name"] == "stripe_refund" and call["args"].get("amount", 0) >= 100:
        return f"refund of ${call['args']['amount']} is at or above the $100 limit"
    return None


print(needs_approval({"name": "stripe_refund", "args": {"amount": 1200}}))
assert needs_approval({"name": "stripe_refund", "args": {"amount": 40}}) is None
assert needs_approval({"name": "stripe_get_invoice", "args": {}}) is None
```

The policy is ordinary code. You can unit-test it, version it, and review changes to it like any other authorization rule. That is the first thing a strong interview answer says about HITL.

### Pause means "serialize and return"

Now the loop. When the model requests a call the policy flags, the loop does not wait. It writes the transcript and the pending call to durable storage and returns a `paused` status. `resume()` is a separate entry point, possibly in a different process on a different day, that loads the record, applies the human's decision, and re-enters the loop.

Three decisions cover almost every real review UI:

```text
approve   run the call exactly as the model proposed it
edit      run the call with human-corrected arguments ($600, not $1200)
reject    do not run it; tell the model why, and let it continue
```

Reject is not "abort the run." The model gets a `ToolMessage` saying the action was declined and why, and it can respond to the user sensibly ("A teammate declined the refund; I've flagged the ticket for billing."). That is the difference between a gate and a crash.

```{code-cell} python
import json
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, ToolMessage, messages_from_dict, messages_to_dict

DB = {}          # run_id -> JSON string. Stands in for Postgres.
AUDIT = []       # append-only: who decided what, when, on which exact action


def _save(run_id, messages, pending):
    DB[run_id] = json.dumps({"messages": messages_to_dict(messages), "pending": pending})


def _execute(call, tools_by_name):
    return ToolMessage(content=str(tools_by_name[call["name"]].invoke(call["args"])),
                       tool_call_id=call["id"])


def run_loop(model, tools, run_id, messages, max_steps=8):
    tools_by_name = {t.name: t for t in tools}
    for _ in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            DB.pop(run_id, None)
            return {"status": "done", "answer": ai.content}
        for call in ai.tool_calls:
            reason = needs_approval(call)
            if reason:
                _save(run_id, messages, {"call": call, "reason": reason})
                return {"status": "paused", "run_id": run_id, "action": call, "reason": reason}
            messages.append(_execute(call, tools_by_name))
    raise RuntimeError("step budget exhausted")


def resume(model, tools, run_id, decision, reviewer):
    record = json.loads(DB[run_id])                        # could be days later, new process
    messages = messages_from_dict(record["messages"])
    proposed = record["pending"]["call"]
    call = {**proposed, "args": dict(proposed["args"])}   # never mutate what was proposed
    AUDIT.append({"run_id": run_id, "reviewer": reviewer, "decision": decision["type"],
                  "proposed": proposed["args"],
                  "at": datetime.now(timezone.utc).isoformat()})
    if decision["type"] == "reject":
        messages.append(ToolMessage(content=f"Declined by {reviewer}: {decision['reason']}",
                                    tool_call_id=call["id"]))
    else:
        if decision["type"] == "edit":
            call["args"] = dict(decision["args"])
        call["args"]["idempotency_key"] = f"{run_id}:{call['id']}"   # stable across retries
        AUDIT[-1]["executed"] = call["args"]
        messages.append(_execute(call, {t.name: t for t in tools}))
    return run_loop(model, tools, run_id, messages)


tools = [stripe_get_invoice, stripe_refund]
model = scripted(
    [("stripe_get_invoice", {"invoice_id": "in_1001"})],
    [("stripe_refund", {"charge_id": "ch_2", "amount": 1200})],
    "Refunded the duplicate charge ch_2. Acme will see it in 5-10 days.",
).bind_tools(tools)

first = run_loop(model, tools, "ticket-4417", [HumanMessage("Acme was double-charged on in_1001.")])
print(first["status"], "-", first["reason"])
assert first["status"] == "paused" and LEDGER == {}          # nothing irreversible happened

# ... the process can die here. Only DB survives. A reviewer clicks "approve" in Slack.
final = resume(model, tools, "ticket-4417", {"type": "approve"}, reviewer="maria@corp")
print(final["status"], "-", final["answer"])
print("audit:", AUDIT[-1]["reviewer"], AUDIT[-1]["decision"], AUDIT[-1]["executed"])
assert final["status"] == "done" and len(LEDGER) == 1
assert "ticket-4417" not in DB and "idempotency_key" not in AUDIT[-1]["proposed"]
```

Four details carry the weight:

```text
json.dumps(...)              the pause survives a restart only if the state is serializable
return {"status":"paused"}   no thread is blocked; the worker is free for other tickets
AUDIT.append(...)            the record says what was PROPOSED and what was EXECUTED
idempotency_key=run:call     if resume runs twice (double click, retry), Stripe refunds once
```

The audit line deserves emphasis. After an edit, "proposed" and "executed" differ. An auditor asking "who authorized the $600 refund?" needs both, plus the reviewer, the time, and ideally the evidence the reviewer saw. Logging only "approved" loses that.

## How LangGraph does it

Everything above maps onto three LangGraph primitives:

| From scratch | LangGraph |
|---|---|
| `_save(run_id, ...)` to `DB` | a **checkpointer** saves state after every step, keyed by `thread_id` |
| `return {"status": "paused", ...}` | `interrupt(payload)` raises a special signal; the run stops and `invoke` returns with `__interrupt__` |
| `resume(..., decision)` | `graph.invoke(Command(resume=decision), config)` with the same `thread_id` |
| the value passed into `resume` | the return value of `interrupt(...)` inside the node |

Here is the part that surprises people. `interrupt()` does not freeze the Python frame. Python cannot serialize a suspended function. So when you resume, LangGraph **re-runs the interrupted node from the top**. This time, when execution reaches `interrupt()`, the call returns the resume value instead of pausing. Everything in the node *before* `interrupt()` runs twice.

```text
first invoke:   review node starts ─► line A ─► line B ─► interrupt(payload) ─► PAUSE
                                                                  (checkpoint saved)
resume invoke:  review node starts ─► line A ─► line B ─► interrupt(payload) returns decision
                                  ▲ runs again   ▲ runs again
```

Keep that picture in mind while reading the graph. The `review` node sits between the model and the tools. If the policy says nothing needs approval, it passes through. Otherwise it interrupts with a payload the reviewer UI can render, then routes on the decision with `Command(goto=...)`.

```{code-cell} python
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

LEDGER.clear()
model = scripted(
    [("stripe_get_invoice", {"invoice_id": "in_1001"})],
    [("stripe_refund", {"charge_id": "ch_2", "amount": 1200})],
    "Refunded $600 on ch_2 as adjusted by billing.",
).bind_tools(tools)


def agent(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}


def review(state: MessagesState) -> Command[Literal["tools", "agent"]]:
    ai = state["messages"][-1]
    flagged = [(c, needs_approval(c)) for c in ai.tool_calls if needs_approval(c)]
    if not flagged:
        return Command(goto="tools")
    call, reason = flagged[0]
    decision = interrupt({"action": call["name"], "args": call["args"], "reason": reason,
                          "allowed": ["approve", "edit", "reject"]})
    if decision["type"] == "reject":
        return Command(goto="agent", update={"messages": [ToolMessage(
            content=f"Declined by reviewer: {decision['reason']}", tool_call_id=call["id"])]})
    args = decision["args"] if decision["type"] == "edit" else call["args"]
    args = {**args, "idempotency_key": call["id"]}
    # Rewrite the pending AIMessage in place (same id => the reducer replaces it).
    edited = ai.model_copy(update={"tool_calls": [{**call, "args": args}]})
    return Command(goto="tools", update={"messages": [edited]})


def route(state: MessagesState):
    return "review" if state["messages"][-1].tool_calls else END


def build(checkpointer):
    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("review", review)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["review", END])
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


saver = InMemorySaver()                       # Postgres in production; same interface
config = {"configurable": {"thread_id": "ticket-4417"}}
out = build(saver).invoke({"messages": [HumanMessage("Acme was double-charged on in_1001.")]}, config)
pending = out["__interrupt__"][0].value
print("paused for:", pending["action"], pending["args"], "-", pending["reason"])
assert LEDGER == {}

# A brand-new graph object over the same checkpointer: the "days later, new deploy" case.
graph_after_deploy = build(saver)
print("next node on resume:", graph_after_deploy.get_state(config).next)
out = graph_after_deploy.invoke(
    Command(resume={"type": "edit", "args": {"charge_id": "ch_2", "amount": 600}}), config)
print(out["messages"][-1].content, "| ledger:", list(LEDGER.values()))
assert list(LEDGER.values()) == [{"charge": "ch_2", "amount": 600}]
```

Map it back once more. `saver` is `DB`. `thread_id` is `run_id`. The checkpoint holds the whole `MessagesState`, which is why the second graph object, built from scratch, knows exactly where it was: `get_state(config).next` reports `('review',)`. `interrupt(...)` is our `return {"status": "paused"}`, and its return value on resume is our `decision` argument. What LangGraph adds is that the pause can happen *inside* any node, at any depth, including inside subgraphs and parallel branches, without you writing a save function for each.

Two hard requirements fall out of the mechanism. There must be a checkpointer, because the resume needs somewhere to load from. And every call must carry a `thread_id`, because that is the key. Without either, `interrupt()` has nowhere to put the paused state and the run cannot continue.

`InMemorySaver` dies with the process, so it only simulates the restart here. In production the checkpointer is Postgres or another durable backend; the code does not change.

### The re-run trap

Now the bug. A reasonable engineer adds a Slack notification so the reviewer knows there is something to look at, and puts it just before `interrupt()`:

```{code-cell} python
SLACK_POSTS = []
SENT_KEYS = set()


def notify_naive(call_id):
    SLACK_POSTS.append(f"#billing-approvals: approval needed ({call_id})")


def notify_idempotent(call_id):
    key = f"approval-request:{call_id}"          # derived from state, stable across re-runs
    if key not in SENT_KEYS:
        SENT_KEYS.add(key)
        SLACK_POSTS.append(f"#billing-approvals: approval needed ({call_id})")


def make_gate(notify):
    def gate(state: MessagesState):
        call = state["messages"][-1].tool_calls[0]
        notify(call["id"])                           # side effect BEFORE interrupt
        decision = interrupt({"action": call["name"], "args": call["args"]})
        return {"messages": [AIMessage(content=f"decision: {decision['type']}")]}
    return gate


def run_gate(notify, thread):
    g = StateGraph(MessagesState)
    g.add_node("gate", make_gate(notify))
    g.add_edge(START, "gate")
    g.add_edge("gate", END)
    graph = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    ai = AIMessage(content="", tool_calls=[{"name": "stripe_refund", "id": "call_9",
                                            "args": {"charge_id": "ch_2", "amount": 1200}}])
    graph.invoke({"messages": [ai]}, cfg)
    return graph.invoke(Command(resume={"type": "approve"}), cfg)


SLACK_POSTS.clear(); run_gate(notify_naive, "t-naive")
naive_count = len(SLACK_POSTS)
SLACK_POSTS.clear(); run_gate(notify_idempotent, "t-fixed")
fixed_count = len(SLACK_POSTS)
print(f"naive notify posted {naive_count} times; idempotent notify posted {fixed_count} time")
assert naive_count == 2 and fixed_count == 1
```

Two posts for one approval is annoying. The same shape with `stripe_refund` before the `interrupt()` is a double refund. Three rules follow:

```text
1. Put side effects AFTER interrupt() or in a separate node that runs after the gate.
2. Anything that must run before it must be idempotent, keyed on something in state
   (the tool call id, the thread id), never on time or randomness.
3. Several interrupt() calls in one node are matched to resume values BY ORDER.
   Do not put interrupt() inside conditionals whose outcome can change between runs.
```

Rule 3 is the subtle one. If a node calls `interrupt()` twice, LangGraph pairs the first resume value with the first call and the second with the second, positionally. Reorder them or skip one on a re-run and the answers land on the wrong questions. Keep one interrupt per node when you can.

### The framework version: middleware and Deep Agents

Writing the `review` node by hand teaches the mechanism. Day to day, LangChain ships it as middleware for `create_agent`. `HumanInTheLoopMiddleware` takes a map from tool name to policy, interrupts before the tool node with a batch of `action_requests`, and expects a list of decisions back, one per request.

```{code-cell} python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

LEDGER.clear()
model = scripted(
    [("stripe_get_invoice", {"invoice_id": "in_1001"})],
    [("stripe_refund", {"charge_id": "ch_2", "amount": 1200})],
    "The refund was declined by billing; I've left the ticket open for them.",
)
hitl_agent = create_agent(
    model, tools=tools, checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "stripe_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
        "stripe_get_invoice": False,                 # reads never pause
    })])

cfg = {"configurable": {"thread_id": "ticket-4418"}}
out = hitl_agent.invoke({"messages": [HumanMessage("Refund Acme's duplicate on in_1001.")]}, cfg)
request = out["__interrupt__"][0].value
print("action_requests:", [(r["name"], r["args"]) for r in request["action_requests"]])
print("allowed:", request["review_configs"][0]["allowed_decisions"])

out = hitl_agent.invoke(Command(resume={"decisions": [
    {"type": "reject", "message": "ch_2 is a second seat, not a duplicate."}]}), cfg)
rejection = [m for m in out["messages"] if isinstance(m, ToolMessage)][-1]
print("model saw:", rejection.content)
print("final:", out["messages"][-1].content)
assert LEDGER == {} and "second seat" in rejection.content
```

The middleware's policy is per tool name, not per argument, so the "$100 threshold" rule does not fit it directly. Two common answers: split the tool (`stripe_refund_small` auto, `stripe_refund` gated), or keep a custom review node like ours when the policy needs arguments or context. Choose based on where you want the policy to live and be tested.

Deep Agents exposes the same idea as `interrupt_on` on `create_deep_agent`, with the same approve/edit/reject decision shape and the same checkpointer requirement. It is shown live because `deepagents` is optional and its signature has moved between releases; check the current docs for the exact names.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
deep = create_deep_agent(
    model=real_model,
    tools=[stripe_get_invoice, stripe_refund],
    system_prompt="You are a billing support agent. Verify before refunding.",
    interrupt_on={"stripe_refund": {"allowed_decisions": ["approve", "edit", "reject"]}},
    checkpointer=InMemorySaver(),
)
cfg = {"configurable": {"thread_id": "live-hitl-1"}}
out = deep.invoke({"messages": [HumanMessage("Acme was double-charged on in_1001. Refund the duplicate.")]}, cfg)
if "__interrupt__" in out:
    print("paused:", out["__interrupt__"][0].value)
    out = deep.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
print(out["messages"][-1].content)
```

## What goes wrong in production

1. **Side effects before the interrupt run twice.** Symptom: duplicate Slack pings, duplicate tickets, in the worst case duplicate refunds. Cause: the node re-runs from the top on resume. Fix: move side effects after the gate or into their own node; make anything pre-interrupt idempotent with keys derived from state.

2. **Approval fatigue.** Symptom: median review time drops to two seconds and approvals approach 100%. Reviewers are not reviewing. Cause: every action is gated, including reads and trivial writes. Fix: gate by risk (irreversibility times blast radius); track approve rate and time-to-decision per action type; a gate with a 99.9% approve rate over thousands of decisions is a candidate for an automated policy check instead.

3. **Reviewers approve without evidence.** Symptom: an approved refund later turns out to be wrong, and the reviewer says "the request looked fine." Cause: the interrupt payload was just `{"tool": "stripe_refund", "amount": 1200}`. Fix: the payload carries what a human needs to decide in under a minute: the customer, the invoice, both charges, the model's stated reason, and a link to the trace. Design it like an on-call page.

4. **Stale approvals.** Symptom: an approval lands three days later and refunds an invoice that was already refunded manually. Cause: the world changed while the run was parked. Fix: re-validate preconditions after resume (re-read the invoice before refunding), give pending approvals an expiry, and treat an expired approval as a reject.

5. **Lost pauses.** Symptom: approvals disappear after a deploy. Cause: `InMemorySaver` in production, or a thread id generated per request instead of per ticket. Fix: a durable checkpointer, and a thread id derived from the business object (the ticket id) so any worker can resume it.

6. **Double resume.** Symptom: two reviewers click approve at once and the action runs twice, or a resume races with the next user message. Cause: resume is not treated as a state transition that can only happen once. Fix: an idempotency key on the side-effecting tool (the fake `LEDGER` above), and a compare-and-set on the pending approval record in your approval service.

7. **No audit trail, or a lossy one.** Symptom: compliance asks who authorized a $4,000 refund and the log says "approved." Cause: only the decision type was stored. Fix: store the proposed action, the executed action, the reviewer identity, the time, the evidence shown, and the thread id, in an append-only log separate from the checkpoint (checkpoints get pruned; audit logs must not).

## What changed by 2026

- **HITL moved from pattern to primitive.** In the book's framing HITL is a design idea. In the current stack it is a runtime feature: `interrupt()` plus a checkpointer in LangGraph, `HumanInTheLoopMiddleware` in LangChain's `create_agent`, and `interrupt_on` in Deep Agents, all sharing the approve/edit/reject vocabulary. You configure it instead of building it.
- **Durable execution is the assumption.** Pauses that last days only work because checkpointing every step is now baseline. LangGraph's durability modes (persist synchronously, asynchronously, or only at exit) are a real knob; for approval flows, the state before a pause must be persisted before anything irreversible can follow.
- **Policy moved out of prompts.** "Ask before refunding" in a system prompt is no longer considered a control by anyone serious. Approval policy is code or configuration keyed on tool and arguments, reviewed like any authorization rule, and tied to agent identity and permissions (39. Agent Identity and Governance).
- **Harnesses expose permission modes.** Coding-agent harnesses popularized tiered permission settings (always ask, ask for writes, auto within a sandbox). The same tiering now shows up in enterprise agents: sandboxed or reversible actions run freely, irreversible ones pause (33. Sandboxed Agent Execution).
- **Protocols carry the pause.** The MCP 2026-07-28 revision's multi round-trip requests let a server answer "input required" and have the client come back with the input, and A2A tasks have an `INPUT_REQUIRED` state. Pauses for human input now cross process and vendor boundaries in standard ways (10. Model Context Protocol, 15. Inter-Agent Communication).
- **Reasoning text is not the evidence.** A model's explanation of why the refund is right is a useful hint, not proof. Reviewers should see the underlying records (both charges, the invoice), because chain-of-thought is not a faithful account of how the model decided.

## Interview questions

**Q1. Design human-in-the-loop approval for an agent that takes consequential actions.** (reported: OpenAI, Palantir)

Strong: I separate four things. A policy in code decides which actions pause, based on tool, arguments, and context, for example refunds of $100 or more, destructive SQL, or company-wide posts. When the policy fires, the run checkpoints its full state keyed by a stable id like the ticket id and returns; no thread blocks. The reviewer gets a payload with the evidence needed to decide quickly and can approve, edit, or reject; reject goes back to the model as an observation so it can respond gracefully. Resume loads the checkpoint, re-validates preconditions, and executes with an idempotency key, and every decision lands in an append-only audit log with proposed and executed arguments. I would track approve rate and time-to-decision to find gates that are theater.

Weak: "Ask the user to confirm before running the tool."

**Q2. How do `interrupt()` and `Command(resume=...)` work in LangGraph, and what is the most common bug?**

Strong: `interrupt(payload)` stops the run and surfaces the payload to the caller; the checkpointer has already saved state for that `thread_id`. Resuming means invoking the graph with `Command(resume=value)` on the same thread. Python cannot serialize a suspended frame, so LangGraph re-executes the interrupted node from the beginning, and this time `interrupt()` returns the resume value. The common bug is putting a side effect before the `interrupt()` call, which then runs twice. The fix is to move it after the gate or make it idempotent, keyed on something in state.

Weak: "It pauses the function and continues it later."

**Q3. The approval arrives three days after the agent paused. What can go wrong and how do you handle it?**

Strong: The world has changed. The invoice may have been refunded manually, the customer may have churned, or the agent's code may have been redeployed. So resume must re-read the facts the decision depended on before acting, and treat a mismatch as a reject with an explanation. Pending approvals should expire. The checkpoint must be durable and compatible with the deployed graph, which means being careful about renaming nodes or changing state schemas while runs are parked. And the thread id should be a business id so any worker can pick it up.

Weak: "The checkpointer handles it."

**Q4. How do you decide which actions need approval?**

Strong: By irreversibility and blast radius, not by tool category. Reads are auto. Reversible writes with small scope are auto but audited. Irreversible actions above a threshold, like refunds over a limit or destructive SQL, pause. Some actions should not be registered at all in production. Then I measure: if a gate's approve rate is near 100% with two-second review times, it is not adding safety and should become an automated check; if rejects cluster on a pattern, that pattern belongs in the policy or the prompt.

Weak: "Anything that writes data needs approval."

**Q5. How do you make agent actions auditable?** (reported: Palantir)

Strong: An append-only log per action with the thread id, the model's proposed call, the policy decision and reason, the reviewer identity if any, the executed call, the tool result, and timestamps. Proposed and executed are stored separately because edits happen. The log is separate from checkpoints, which are operational state and get pruned. Link each entry to the trace so an investigator can see what the model saw. Where possible, prefer reversible actions and keep the compensating action (the refund reversal, the message delete) next to the log entry.

Weak: "We log everything to stdout."

**Q6. Where does HITL belong: in the tool, in the graph, or in middleware?**

Strong: Not inside the tool, because a blocking prompt inside a tool cannot survive a restart and hides the policy. In the graph as a review node when the policy depends on arguments or context, or when you want the gate visible in the graph structure. In middleware like `HumanInTheLoopMiddleware` when a per-tool policy is enough and you are using `create_agent`; it is the same `interrupt()` underneath. The deciding factor is where the policy is easiest to test and audit.

Weak: "Wherever the framework puts it."

**Q7. What should the reviewer see?**

Strong: Enough to decide correctly in under a minute without opening five tabs: what the agent intends to do in plain language, the exact arguments, the evidence it relied on (the invoice and both charges), why the policy fired, what happens on reject, and a link to the full trace. Not the model's reasoning as the justification, because that text is not a faithful record of how it decided; show the underlying data. Offer edit, because many "wrong" proposals are nearly right.

Weak: "The tool name and arguments."

## Build it yourself

Extend the LangGraph `review` node into a production-shaped gate:

1. Support several flagged tool calls in one model turn: interrupt once with a list of action requests and accept a list of decisions in the same order.
2. Add an expiry: the interrupt payload carries `expires_at`; on resume, if the decision arrives after it, treat it as a reject with reason `"approval expired"`.
3. Re-validate on resume: before executing an approved refund, call `stripe_get_invoice` again and reject if the charge is no longer present in the result.
4. Write an audit record (thread id, reviewer, decision, proposed args, executed args) to an append-only list for every decision.

Acceptance checks:

- A turn with two $1,200 refunds produces exactly one interrupt whose payload lists two actions; approving one and rejecting the other leaves exactly one entry in `LEDGER`.
- A decision arriving after `expires_at` leaves `LEDGER` empty and the model receives a `ToolMessage` containing `"expired"`.
- Resuming the same thread twice with the same approval never produces more than one ledger entry.
- For an edited decision, the audit record's proposed and executed args differ.
- A run with only reads and a $40 refund never interrupts.

## Key takeaways

- Human-in-the-loop is four parts: a policy in code, a durable pause, a structured decision, and an exactly-once resume with an audit record.
- Pausing means persisting state and returning, never blocking a thread while a human decides.
- `interrupt()` needs a checkpointer and a `thread_id`; `Command(resume=value)` on the same thread continues the run.
- On resume the interrupted node re-runs from the top, so everything before `interrupt()` must be idempotent or moved after it.
- Gate by irreversibility and blast radius; gating everything trains reviewers to click approve without reading.
- Reject is an observation for the model, not a crash, and edits must be audited as proposed versus executed.
- `HumanInTheLoopMiddleware` and Deep Agents' `interrupt_on` are the same mechanism packaged; build it once by hand so you know what they do.
