# B. Agentic interactions: topologies, sync and async, streaming, approvals, background agents

*Track A (agent engineering). Book: Appendix B (AI Agentic Interactions: From GUI to Real World Environment). Prerequisites: 00. Course orientation, 13. Human in the loop; 36. Durable agent orchestration helps for the background-agent section.*

Most agent lessons study what happens inside the loop. This one studies its edges: who is on the other side of each message, how long they are willing to wait, how they find out what happened, and how they say yes or no to a consequential action.

These questions decide whether an agent is usable. A correct agent that shows a spinner for ninety seconds, or that finishes a two-hour task while the user has closed the tab and never tells them, is a failed product.

## The problem

The support agent's first deployment is a synchronous HTTP endpoint: the Slack bot posts the user's message, waits for the agent to finish, and posts the answer.

```text
Slack user ──► POST /agent ──► [ loop: model, tools, model, tools, model ] ──► 200 OK ──► reply
                   │                                                              │
                   └──────────────── 45 to 300 seconds, silent ───────────────────┘
```

Every assumption in that picture fails in production:

```text
assumption                          reality
──────────────────────────────────  ────────────────────────────────────────────────────
the user waits                      Slack expects a quick acknowledgement; users leave
one request = one task              a refund needs a manager's approval, maybe tomorrow
the only party is the user          approver, other agents, webhooks, a cron job
silence is fine while working       users retry, creating duplicate runs
the connection lives as long        load balancer idle timeouts, deploys, laptop sleeps
as the task
```

What the endpoint needs is an interaction design: a topology (who talks to whom), a time model (sync, streaming, async, background), an event protocol (what gets reported while work happens), a pause protocol (approvals), and a notification path (how the result finds the user). All of those are ordinary distributed-systems concerns wearing an agent costume.

### Topologies

```text
1. user ⇄ agent                  chat; the user is present and waiting
2. agent ⇄ tool                  the model chooses, your software executes (lesson 00)
3. agent ⇄ agent                 supervisor/subagent, handoff, or A2A across services
4. agent ⇄ approver              a human who is NOT the requester approves an action
5. event → agent → notification  no user present: webhook, schedule, or queue trigger
```

The running example uses all five. A customer asks in Slack (1). The agent reads the CRM and Stripe (2). It delegates a log search to a research subagent (3). A refund over the threshold goes to a finance approver (4). Every night a scheduled run reconciles failed payments and posts a summary to a channel (5).

Each edge has an owner of state and an expected latency. Deciding those two things per edge is most of the design.

## Build it without a framework

The scripted model is defined first because the LangGraph section needs it. It stands in for a real model so the interaction machinery, not the model's judgment, is under study; a live cell at the end streams a real model.

### An event protocol, by hand

The core idea: an agent run is not a function that returns an answer. It is a producer of a stream of typed events, and the answer is the last event. Once you see it that way, streaming, progress, approvals, and notifications are all consumers of the same stream.

```text
event            meaning                               UI behavior
───────────────  ────────────────────────────────────  ────────────────────────────
status           human-readable progress               "Checking Stripe..."
tool_start/end   an action began or finished           collapsible activity log
token            a piece of the final answer           append to the message
approval         the run is paused for a decision      buttons: approve / reject
final            the run finished                       close the activity log
```

A Python generator is exactly this: it yields events and can receive a value back with `send`. The approval pause is a `yield` that waits for the consumer's decision.

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


def support_run(account: str):
    """A run as a stream of events. The final answer is just the last event."""
    yield {"type": "status", "text": "Looking up the account"}
    yield {"type": "tool_start", "tool": "crm_lookup"}
    yield {"type": "tool_end", "tool": "crm_lookup", "result": f"{account}: INV-88 billed twice"}
    decision = yield {"type": "approval", "action": "stripe_refund", "amount_usd": 450}
    if decision == "approve":
        yield {"type": "tool_start", "tool": "stripe_refund"}
        yield {"type": "tool_end", "tool": "stripe_refund", "result": "refunded $450"}
        answer = "You were billed twice for INV-88. I refunded the duplicate $450."
    else:
        answer = "You were billed twice for INV-88. A billing specialist will follow up."
    for word in answer.split(" "):
        yield {"type": "token", "text": word + " "}
    yield {"type": "final", "text": answer}


def chat_ui(run, approver):
    """A consumer: renders progress, asks the approver, streams the answer."""
    log, shown, event = [], "", next(run)
    while True:
        log.append(event["type"])
        if event["type"] == "token":
            shown += event["text"]
        if event["type"] == "final":
            return log, shown.strip(), event["text"]
        reply = approver(event) if event["type"] == "approval" else None
        event = run.send(reply)


log, shown, final = chat_ui(support_run("Acme"), approver=lambda e: "approve")
print(log[:6], "...", log[-1])
print(shown)
assert log.index("status") < log.index("approval") < log.index("token")
assert shown == final
_, _, rejected = chat_ui(support_run("Acme"), approver=lambda e: "reject")
assert "specialist" in rejected
```

Three properties of this design carry through the rest of the lesson. The user sees progress before the answer exists. The approval is an ordinary event with a reply, not a special code path. And the UI is a consumer that could equally be a Slack bot, a web page, or a test.

What the generator cannot do is survive a process restart. Its paused state lives in the Python frame. If the approver answers tomorrow, the frame is gone. That is the gap LangGraph's checkpointer fills.

### Sync, streaming, async, background

Choose the time model from how long the task takes and whether the user is present:

| Time model | Typical duration | User present? | Transport | Example in the support agent |
|---|---|---|---|---|
| Sync | under a couple of seconds | yes | request/response | intent classification |
| Streaming | seconds to a minute | yes | SSE or WebSocket | answering an account question |
| Async with status | minutes | maybe | 202 Accepted + poll or push | a multi-system investigation |
| Background | minutes to hours, or triggered | no | queue + notification | nightly failed-payment reconciliation |

The line between streaming and async is not a number. It is the question "will the user still be looking when it finishes?" If not, the run must outlive the connection, so it needs an id, durable state, and a way to report back.

### A background agent with notifications

A background run is a job. It has an id, a state machine, a durable record, and subscribers that are notified on transitions. Notification delivery is at-least-once in any real system (webhooks retry), so notifications carry a key and the receiver deduplicates.

```text
queued ──► running ──► succeeded
              │  ▲          
              ▼  │ resume    
        waiting_approval ──► failed (rejected or timed out)
```

```{code-cell} python
import asyncio


class JobStore:
    """Durable job records would live in Postgres; a dict shows the state machine."""

    def __init__(self):
        self.jobs, self.subscribers, self.delivered = {}, [], set()

    def transition(self, job_id: str, state: str, **info):
        job = self.jobs.setdefault(job_id, {"history": []})
        job.update(state=state, **info)
        job["history"].append(state)
        for notify in self.subscribers:
            key = f"{job_id}:{state}"                 # idempotency key for at-least-once delivery
            if key not in self.delivered:
                self.delivered.add(key)
                notify(job_id, state, info)


async def reconcile_failed_payments(store: JobStore, job_id: str, approvals: asyncio.Queue):
    store.transition(job_id, "running")
    await asyncio.sleep(0)                            # stands in for tool calls
    store.transition(job_id, "waiting_approval", action="retry_charge", count=3)
    decision = await approvals.get()                  # the run is parked, not spinning
    if decision != "approve":
        store.transition(job_id, "failed", reason="rejected by approver")
        return
    store.transition(job_id, "running")
    await asyncio.sleep(0)
    store.transition(job_id, "succeeded", summary="3 charges retried, 2 succeeded")


async def main():
    store, inbox = JobStore(), []
    store.subscribers.append(lambda job, state, info: inbox.append((job, state)))
    approvals = asyncio.Queue()
    store.transition("job-42", "queued")
    worker = asyncio.create_task(reconcile_failed_payments(store, "job-42", approvals))
    while store.jobs["job-42"]["state"] != "waiting_approval":   # a client polling status
        await asyncio.sleep(0)
    store.transition("job-42", "waiting_approval")               # a duplicate webhook retry
    await approvals.put("approve")
    await worker
    return store, inbox


from concurrent.futures import ThreadPoolExecutor

# Jupyter already runs an event loop in this thread, so give the demo its own thread and loop.
with ThreadPoolExecutor(max_workers=1) as pool:
    store, inbox = pool.submit(asyncio.run, main()).result()
print(store.jobs["job-42"]["history"])
print(inbox)
assert store.jobs["job-42"]["state"] == "succeeded"
assert [s for _, s in inbox] == ["queued", "running", "waiting_approval", "succeeded"]
```

The inbox received exactly one notification per distinct state, even though `waiting_approval` was published twice and `running` was entered twice. That is a deliberate choice: the approver needs one ping, not three. Notifications are also a product decision. Notify on the transitions a human must act on (approval needed, failed) and on completion; do not notify on every tool call.

## How LangGraph does it

LangGraph gives you the event stream and the durable pause as primitives, and each maps back to the code above.

```text
from scratch                          LangGraph
────────────────────────────────────  ────────────────────────────────────────────────
generator yielding events             graph.stream(..., stream_mode=[...])
{"type": "status"}                    get_stream_writer() → "custom" mode
{"type": "tool_start/end"}            "updates" mode: one chunk per finished node
{"type": "token"}                     "messages" mode: LLM tokens with node metadata
yield approval / run.send(decision)   interrupt(value) / Command(resume=decision)
job record in JobStore                checkpointer + thread_id; graph.get_state(config)
job state "waiting_approval"          state.next non-empty and state.interrupts present
```

### Streaming modes

`stream_mode` accepts a list, and each chunk comes back as `(mode, payload)`. The modes answer different UI questions: `updates` says which step just finished, `messages` streams the model's tokens as they are generated, `custom` carries whatever progress events your nodes write, and `values` gives the full state after each step (useful for debugging, expensive for UIs).

One addition to the scripted model: the fake chat model drops tool calls when it streams, so the cell subclasses it to emit tool calls as one chunk, the way real providers stream them.

```{code-cell} python
import json
import re

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


@tool
def stripe_get_invoice(invoice_id: str) -> str:
    """Read one Stripe invoice: amount, status, and charge ids."""
    get_stream_writer()({"status": f"Reading {invoice_id} from Stripe"})   # custom progress event
    return "INV-88: $450, two successful charges ch_1 and ch_2"


class StreamingScriptedModel(ScriptedChatModel):
    """The fake model drops tool calls when streamed. Real providers stream them as
    tool-call chunks, so emit those in one chunk and stream text word by word."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._generate(messages, stop=stop, **kwargs).generations[0].message
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i,
                 "type": "tool_call_chunk"} for i, c in enumerate(msg.tool_calls)]))
            return
        for piece in re.split(r"(\s)", msg.content):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


tools = [stripe_get_invoice]
model = StreamingScriptedModel(messages=scripted(
    [("stripe_get_invoice", {"invoice_id": "INV-88"})],
    "INV-88 was charged twice (ch_1, ch_2).").messages).bind_tools(tools)


def agent(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END,
                              ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()

ui_events, tokens = [], []
for mode, payload in graph.stream({"messages": [HumanMessage("Why was Acme billed twice?")]},
                                  stream_mode=["updates", "messages", "custom"]):
    if mode == "custom":
        ui_events.append(("status", payload["status"]))
    elif mode == "updates":
        ui_events.append(("step_done", next(iter(payload))))
    elif mode == "messages":
        chunk, meta = payload
        if isinstance(chunk, AIMessageChunk) and chunk.content:      # only model tokens
            tokens.append(chunk.content)

print(ui_events)
print(f"{len(tokens)} token chunks -> {''.join(tokens)!r}")
assert ("status", "Reading INV-88 from Stripe") in ui_events
assert [e for e in ui_events if e[0] == "step_done"] == [
    ("step_done", "agent"), ("step_done", "tools"), ("step_done", "agent")]
assert "".join(tokens) == "INV-88 was charged twice (ch_1, ch_2)."
```

Two details are worth saying out loud. The `messages` mode carries more than model tokens: messages produced by other nodes appear too, so the UI filters for `AIMessageChunk` with content, or on the `langgraph_node` field in the metadata. And the stream writer inside the tool is the `yield {"type": "status"}` line from the generator, except that it works from any depth of the graph, including subgraphs when you stream with `subgraphs=True`.

### Approvals and background runs, durably

With a checkpointer, an `interrupt` is the generator's approval `yield` with the frame saved to storage. The run ends. The caller gets an `__interrupt__` payload. Hours later, any process with the same checkpointer and `thread_id` resumes it with `Command(resume=...)`. The job store's "waiting_approval" state is just the checkpoint's `next` field being non-empty.

```{code-cell} python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

REFUNDS, NOTIFICATIONS = [], []


def investigate(state: MessagesState):
    get_stream_writer()({"status": "Found duplicate charge ch_2 on INV-88"})
    return {"messages": [AIMessage("Duplicate charge ch_2 found.")]}


def approve_refund(state: MessagesState):
    # Nothing with side effects before interrupt(): on resume this node re-runs from the top.
    decision = interrupt({"action": "stripe_refund", "charge": "ch_2", "amount_usd": 450,
                          "requested_by": "support-agent", "approver_group": "finance"})
    if decision["approved"]:
        REFUNDS.append(("ch_2", 450))
        return {"messages": [AIMessage(f"Refunded $450 (approved by {decision['by']}).")]}
    return {"messages": [AIMessage("Refund rejected; escalated to billing.")]}


def notify(state: MessagesState):
    NOTIFICATIONS.append(state["messages"][-1].content)   # Slack post, email, webhook
    return {}


b = StateGraph(MessagesState)
b.add_node("investigate", investigate)
b.add_node("approve_refund", approve_refund)
b.add_node("notify", notify)
b.add_edge(START, "investigate")
b.add_edge("investigate", "approve_refund")
b.add_edge("approve_refund", "notify")
b.add_edge("notify", END)
checkpointer = InMemorySaver()                        # PostgresSaver in production
background = b.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "job-acme-inv-88"}}

# Request path: start the run, stream progress, return 202 when it pauses.
seen = [chunk for chunk in background.stream(
    {"messages": [HumanMessage("Refund Acme's duplicate charge")]}, config,
    stream_mode=["custom", "updates"])]
pause = [p["__interrupt__"][0].value for m, p in seen if m == "updates" and "__interrupt__" in p]
print("paused:", pause[0])

# Status endpoint: anyone can poll the durable record.
snapshot = background.get_state(config)
print("next:", snapshot.next, "| pending approvals:", len(snapshot.interrupts))
assert snapshot.next == ("approve_refund",) and REFUNDS == []

# Approval path: a different request, possibly a different process, resumes the run.
background.invoke(Command(resume={"approved": True, "by": "finance:maria"}), config)
print("refunds:", REFUNDS, "| notifications:", NOTIFICATIONS)
assert REFUNDS == [("ch_2", 450)] and background.get_state(config).next == ()
assert NOTIFICATIONS == ["Refunded $450 (approved by finance:maria)."]
```

Map it to the job store. `thread_id` is the job id. `get_state(config).next` is the job state. `snapshot.interrupts` is the approval inbox. The `notify` node is the subscriber. The difference from the dict version is that everything survives a restart, because the checkpointer wrote a snapshot after each step.

Three more primitives complete the picture. LangGraph's durability modes (`"sync"`, `"async"`, `"exit"` in current docs) trade write latency against how much progress a crash can lose. `create_agent` packages the approval pattern as `HumanInTheLoopMiddleware`, configured per tool, so you do not hand-write the interrupt node for each tool. And LangGraph's deployment server (the LangGraph Platform / LangSmith deployment docs) exposes the same model over HTTP: runs on threads, background runs, streaming endpoints, and webhooks on completion, which is the productized version of the cell above.

The same stream against a real model:

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
for chunk in real.stream("In two sentences, why do agents stream progress events?"):
    print(chunk.content, end="", flush=True)
print()
```

## What goes wrong in production

1. **Silent long runs.** Symptom: users click the button again, and there are now two agents refunding the same invoice. Cause: no acknowledgement or progress for tens of seconds. Fix: acknowledge immediately with a run id, stream status events, and deduplicate starts by an idempotency key derived from the request.

2. **Connection-bound work.** Symptom: runs die at exactly the load balancer's idle timeout, or on every deploy. Cause: the task lives inside the HTTP request. Fix: decouple the run from the connection; stream from a durable run that clients can reattach to by id, and move anything longer than a user will wait to a background run.

3. **Approval without context.** Symptom: approvers rubber-stamp everything, or reject everything. Cause: the approval card shows "Approve refund?" and nothing else. Fix: the interrupt payload carries the exact action, arguments, the evidence, who requested it, and what happens on reject; the approver can edit arguments, not only accept them (lesson 13).

4. **Side effects replayed on resume.** Symptom: a Slack message is posted twice after an approval. Cause: the node that calls `interrupt` posted the message before pausing, and a resumed node re-runs from its top. Fix: nothing with side effects before `interrupt()` in the same node, or split the side effect into its own node, plus idempotency keys.

5. **Notification storms and gaps.** Symptom: a channel gets forty updates for one job, or none when it fails. Cause: notifying on every event, and only on the happy path. Fix: notify on actionable transitions (needs approval, failed, finished), deduplicate by `job:state`, and alert when a job sits in `waiting_approval` beyond an SLA.

6. **Streaming the wrong thing.** Symptom: the UI shows raw tool-call JSON or internal messages as if they were the answer. Cause: rendering every `messages` chunk. Fix: filter by chunk type and node metadata; stream the final answer's tokens and map everything else to status events.

## What changed by 2026

- **Background agents are a default product shape.** Coding agents and support agents now routinely run as background jobs that report back through pull requests, tickets, or chat, rather than in a blocking chat turn. Anthropic's "Scaling Managed Agents" describes the session as an append-only durable event log that outlives any harness process; OpenAI's Agents API (a public beta, per its docs) exposes durable sessions and events for hosted runs.
- **Protocols model long-running work explicitly.** A2A tasks have states including `WORKING`, `INPUT_REQUIRED`, and `AUTH_REQUIRED`, support streaming over SSE, and deliver push notifications to a client webhook. MCP's 2026-07-28 revision added multi round-trip requests where a server returns `input_required` and the client retries with the answers, and long-running work is handled by the official Tasks extension, which clients poll.
- **Durable pause is a baseline expectation.** Checkpointed interrupts with resume from another process are how approvals are built, in LangGraph (`interrupt` and `Command(resume=...)`), in LangChain's `HumanInTheLoopMiddleware`, and in Deep Agents' `interrupt_on` configuration, which current docs describe with approve, edit, reject, and respond decisions.
- **Streaming is multi-channel.** Instead of one token stream, UIs consume several: tokens, step updates, custom progress, and interrupts, as LangGraph's stream modes expose.
- **The approver is often not the user.** Agents acting on behalf of an organization route approvals to role-based groups with their own SLAs and audit trails, which ties interaction design to agent identity and governance (lesson 39).

## Interview questions

**Q1. How do you design the UX for an agent task that takes two minutes?**

Strong: Acknowledge immediately with a run id so the user knows it started and a retry does not start a second run. Stream progress as typed events: status lines, tool starts and ends, then the final answer's tokens, so time to first useful output is seconds. Make the run independent of the connection: it lives on a durable thread, and the client can reattach by id. If the user may leave, switch to async and notify on completion or when input is needed.

Weak: "Show a loading spinner."

**Q2. When do you make an agent asynchronous instead of streaming?**

Strong: When the user will probably not be looking when it finishes, or when it may pause for someone else, like an approver. Then the run needs an id, durable state, a status endpoint, and a notification path. Streaming keeps the user connected; async lets them leave. The deciding question is presence, not a fixed number of seconds.

Weak: "When it takes longer than 30 seconds."

**Q3. Design human approval for a refund over $100 when the approver may respond tomorrow.**

Strong: The agent proposes the refund as a tool call; a policy gate in code, not the model, decides it needs approval and calls `interrupt` with the exact arguments, evidence, and requester. The checkpointer saves the run and the process can exit. The approval request goes to the finance group with an SLA. When someone approves, any worker resumes the thread with `Command(resume=...)`. Nothing with side effects runs before the interrupt in that node, and the refund itself uses an idempotency key. Rejections and timeouts are explicit outcomes with an escalation path.

Weak: "The agent asks the user if it should proceed."

**Q4. What do LangGraph's stream modes give you, and which would you use for a chat UI?**

Strong: `updates` emits one chunk per finished node, `messages` streams LLM tokens with metadata about which node produced them, `custom` carries events your nodes write with the stream writer, and `values` emits the full state after each step. For a chat UI I combine `messages` for the answer, filtered to model chunks from the answering node, `custom` for progress text, and `updates` for the activity log and interrupts. `values` is for debugging.

Weak: "Stream mode lets you stream the response."

**Q5. How does a background agent notify users without spamming them or missing failures?**

Strong: Treat the run as a job with a state machine and notify on actionable transitions: needs input, failed, finished. Deliveries are at-least-once, so each notification carries a job-and-state key and receivers deduplicate. I alert on jobs stuck in a waiting state beyond an SLA, because a silent pause is the worst outcome. Channels follow the audience: the requester gets a direct message, an approver group gets a queue item.

Weak: "Post to Slack when it's done."

**Q6. Compare agent-to-agent interaction inside one process with A2A across services.**

Strong: In-process, a supervisor calls a subagent as a function or a subgraph; state can be shared and failures are exceptions. Across services, the other agent is opaque: you discover it through its Agent Card, send it a task, and track the task through states like working, input-required, and completed, with streaming or push notifications. You lose shared state and gain independent deployment and ownership. The interaction design questions are the same: who owns state, how long does it take, and how do you learn it finished.

Weak: "A2A is how agents talk to each other."

## Build it yourself

Extend the durable background run into a small job API.

1. Write `start(request_id, text)` that derives the `thread_id` from `request_id`, so calling it twice with the same id does not start a second run.
2. Write `status(job_id)` returning one of `running`, `waiting_approval`, `succeeded`, `failed`, computed only from `graph.get_state(config)`.
3. Add a reject path: `Command(resume={"approved": False, "by": ...})` must end in `failed` with an escalation notification.
4. Add a `stale_approvals(now, sla_seconds)` function that lists jobs paused longer than the SLA, using a timestamp you store in the interrupt payload.

Acceptance checks:

- Calling `start` twice with the same `request_id` produces one checkpoint history, not two runs.
- `status` returns `waiting_approval` after `start` and `succeeded` after an approving resume.
- A rejection produces exactly one notification and no entry in `REFUNDS`.
- A job paused at `t=0` appears in `stale_approvals(now=7200, sla_seconds=3600)` and not in `stale_approvals(now=1800, sla_seconds=3600)`.

## Key takeaways

- An agent run is a stream of typed events; the final answer is only the last event.
- Choose sync, streaming, async, or background by whether the user will still be present, not by a fixed duration.
- Decouple the run from the connection: give it an id and durable state so clients can reattach and approvals can wait.
- An approval is a durable pause: `interrupt` saves the run, `Command(resume=...)` continues it from any process.
- Never put side effects before `interrupt()` in the same node, because the node re-runs from the top on resume.
- Notify on actionable transitions, deduplicate by job and state, and alert on runs stuck waiting.
- LangGraph's stream modes separate tokens, step updates, and custom progress; a good UI uses several and filters them.
