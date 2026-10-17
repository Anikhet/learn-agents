# 15. Inter-Agent Communication (A2A)

*Track A (agent engineering). Book: Chapter 15 (Inter-Agent Communication, A2A). Prerequisites: 07. Multi-Agent Collaboration, 10. Model Context Protocol (MCP), 13. Human in the Loop.*

In 07 every agent lived in your process. The manager imported the billing worker, compiled it as a subgraph, and called it. That works while one team owns every agent. It stops working the day the agent you need belongs to someone else, runs on another framework, sits behind another team's credentials, and takes three hours to finish because a human in Finance has to sign off.

This lesson is about that boundary. A2A (Agent2Agent) is the protocol for it. We will build the protocol's core by hand, a task state machine exchanging JSON between two in-process agents, then put a LangGraph node in front of it, then look at what breaks when the other agent is really across a network.

## The problem

Our running example is the enterprise support agent. It reads Slack, Notion, Stripe, the CRM, and SQL. A customer, Acme, disputes a $1,200 invoice line. Support has read access to Stripe. It does not, and should not, have write access to the ledger.

Finance has its own agent. It owns the ledger integration, the credit-memo templates, and the rule that any credit over $500 needs a named approver. It is built by a different team, on a different framework, deployed on a different cluster. Support cannot import it.

The naive integration is a tool that POSTs a prompt and waits for a string:

```text
support agent                                  finance agent
     │  POST /chat {"prompt": "please credit       │
     │   Acme $1,200 for INV-2291, duplicate line"} │
     │ ───────────────────────────────────────────► │
     │                                              │  needs approver... waits for a human
     │             (30 s client timeout)            │
     │ ◄──────────── 504 Gateway Timeout ────────── │
     │  retry the tool call                         │
     │ ───────────────────────────────────────────► │  second credit request created
```

Walk through where it breaks:

```text
Question the naive design cannot answer         What the protocol needs
───────────────────────────────────────         ───────────────────────
What can the finance agent do? How do I auth?   a discovery document (Agent Card)
The work takes hours. Where is it now?          a stateful Task with an id and states
Finance needs an approver name from me.         a state that means "blocked on input"
Did my retry create a second credit?            idempotent messages, task-scoped follow-ups
The result is a credit memo, not prose.         typed outputs (Artifacts made of Parts)
I disconnected. What happened meanwhile?        streaming, resubscribe, push notifications
I want to stop it.                              cancel, and terminal states that stay terminal
```

Every row is a distributed-systems problem, not an AI problem. A2A is what you get when you answer them once, in a spec, so two teams do not reinvent them pairwise.

### Vertical and horizontal

You already have MCP (10). Why a second protocol? The A2A project's own framing is that MCP is vertical and A2A is horizontal:

```text
                     ┌──────────────── A2A (horizontal) ────────────────┐
                     │                                                  │
             ┌───────┴───────┐                                  ┌───────┴───────┐
             │ support agent │   Task: resolve dispute INV-2291 │ finance agent │
             │  (LangGraph)  │ ───────────────────────────────► │  (other team) │
             └───────┬───────┘                                  └───────┬───────┘
                     │ MCP (vertical)                                   │ MCP (vertical)
        ┌────────────┼────────────┐                          ┌──────────┴──────────┐
     Slack        Notion        Stripe (read)              Ledger             Approvals
```

MCP connects one agent to its capabilities: tools, resources, prompts. The server is a function box. It does not plan, it does not have goals, and it should not see the conversation. A2A connects an agent to a peer that does plan, has its own tools and model, and is opaque to you. You do not see its prompt, its tools, or its memory. You see a card, a task, messages, and artifacts.

The line blurs at the edges. MCP's Tasks extension gives long-running MCP calls a task handle, and MCP's published roadmap mentions agentic messaging. The useful test is ownership and opacity: if you control the thing and it is a function, it is a tool (MCP or local). If someone else owns it and it decides how to do the work, it is a peer (A2A).

### The core objects

```text
Agent Card   JSON at a well-known URL: identity, endpoint, skills, input/output modes,
             capabilities (streaming, push notifications), auth schemes.
Task         The stateful unit of work. Has an id, a contextId, a status, a history.
Message      One turn: role "user" (the client agent) or "agent" (the remote agent),
             made of Parts. Follow-ups carry the taskId.
Part         One piece of content: text, a file (bytes or URL), or structured data (JSON).
Artifact     The task's output, also made of Parts. The credit memo, not the chat.
```

And the lifecycle. The v1.0 spec defines these task states:

```text
                       ┌──────────► REJECTED   (agent declines the work)
                       │
  SUBMITTED ──► WORKING ┼──────────► COMPLETED  (artifacts attached)
      │           ▲  │  ├──────────► FAILED
      │           │  │  └──────────► CANCELED
      │           │  ▼
      │     INPUT_REQUIRED / AUTH_REQUIRED   (interrupted: the client must act)
      └────────────────────────────► AUTH_REQUIRED / REJECTED / CANCELED

  terminal:     COMPLETED  FAILED  CANCELED  REJECTED     (never leave these)
  interrupted:  INPUT_REQUIRED  AUTH_REQUIRED             (resume with a message on the same taskId)
```

If this looks like LangGraph's `interrupt` from 13. Human in the Loop, that is the right instinct. `INPUT_REQUIRED` is an interrupt that crosses an organizational boundary.

## Build it without a framework

We build both sides in one process, but every byte between them is serialized JSON through a single `transport` function. That is the discipline that matters: if it works through `json.dumps`, it works over HTTP.

The method names and field names below follow the shape of the spec but are simplified for teaching. The spec defines several bindings (JSON-RPC, gRPC, HTTP+JSON) and exact names differ between them and between versions. In real code, use an A2A SDK and let it own the wire format.

The first cell is the scripted model helper used throughout the course. The finance agent's internals are deterministic here; the model only appears later, when the support agent writes its Slack reply.

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

### The Agent Card and the task state machine

The card is what a client reads before it sends anything. It is a contract about *what* the agent does, never *how*. Notice what is absent: no system prompt, no tool list, no model name. The finance team can change all of those without telling you.

The state machine is the other half of the contract. Illegal transitions raise. Terminal states have no exits. Without this, "the task completed, then failed, then completed again" is a real bug you will see in webhook logs.

```{code-cell} python
import json
from dataclasses import dataclass, field

FINANCE_CARD = {
    "name": "finance-disputes",
    "description": "Resolves invoice disputes and issues credit memos against the ledger.",
    "url": "https://finance-agents.internal.example/a2a",
    "version": "2.3.0",
    "capabilities": {"streaming": True, "pushNotifications": True},
    "securitySchemes": {"corp_oauth": {"type": "oauth2"}},
    "defaultInputModes": ["application/json", "text/plain"],
    "defaultOutputModes": ["application/json"],
    "skills": [{
        "id": "resolve_invoice_dispute",
        "name": "Resolve invoice dispute",
        "description": "Given invoice_id, amount_usd and reason, validates the dispute and "
                       "issues a credit memo. Credits over $500 need an approval_ref.",
        "tags": ["billing", "credit", "ledger"],
    }],
}

# Discovery: the client fetches the card from a well-known path on the agent's host.
WELL_KNOWN = {"https://finance-agents.internal.example/.well-known/agent-card.json":
              json.dumps(FINANCE_CARD)}

TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "REJECTED"}
TRANSITIONS = {
    "SUBMITTED": {"WORKING", "REJECTED", "CANCELED", "FAILED", "AUTH_REQUIRED"},
    "WORKING": {"COMPLETED", "FAILED", "CANCELED", "INPUT_REQUIRED", "AUTH_REQUIRED"},
    "INPUT_REQUIRED": {"WORKING", "CANCELED", "FAILED"},
    "AUTH_REQUIRED": {"WORKING", "CANCELED", "FAILED"},
    **{s: set() for s in TERMINAL},
}


class IllegalTransition(Exception):
    pass


@dataclass
class Task:
    id: str
    context_id: str
    state: str = "SUBMITTED"
    status_message: dict | None = None
    history: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    status_log: list = field(default_factory=lambda: ["SUBMITTED"])

    def to(self, new_state, message=None):
        if new_state not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"task {self.id}: {self.state} -> {new_state}")
        self.state, self.status_message = new_state, message
        self.status_log.append(new_state)


t = Task("t-demo", "ctx-demo")
t.to("WORKING"); t.to("COMPLETED")
try:
    t.to("WORKING")
    raise AssertionError("terminal state must be final")
except IllegalTransition as exc:
    print("refused:", exc)
assert t.status_log == ["SUBMITTED", "WORKING", "COMPLETED"]
assert all(TRANSITIONS[s] == set() for s in TERMINAL)
```

### Two agents, one wire

The server is the finance agent. It handles four methods: send a message, get a task, list tasks, cancel a task. Its "intelligence" is a policy function, because the point is the protocol, not Finance's reasoning. Three details carry most of the weight:

1. **Idempotency by `messageId`.** A retried message returns the existing task instead of creating a new one. This is the fix for the naive diagram's double credit.
2. **Follow-ups carry `taskId`.** Answering an `INPUT_REQUIRED` question is just another message, addressed to the existing task.
3. **Every state change is an event with a sequence number.** Streaming, resubscription, and push notifications are all views of this one log.

```{code-cell} python
import itertools

# Stand-in for the network: webhook URL -> the handler listening there.
WEBHOOK_ROUTES = {}


def http_post(url, body):
    if url in WEBHOOK_ROUTES:
        WEBHOOK_ROUTES[url](body)


class FinanceAgentServer:
    """The remote agent. Opaque to the client except through JSON over `handle`."""

    def __init__(self):
        self.tasks, self.by_message_id = {}, {}
        self.events, self.webhooks = [], {}
        self.ledger_writes = []
        self._ids = itertools.count(1)

    # -- wire entry point: JSON in, JSON out --------------------------------------
    def handle(self, raw: str) -> str:
        req = json.loads(raw)
        methods = {"message/send": self._send, "tasks/get": self._get,
                   "tasks/list": self._list, "tasks/cancel": self._cancel,
                   "tasks/events": self._events, "tasks/pushConfig/set": self._push_set}
        try:
            body = {"result": methods[req["method"]](req.get("params", {}))}
        except (IllegalTransition, KeyError, ValueError) as exc:
            body = {"error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}}
        return json.dumps({"jsonrpc": "2.0", "id": req["id"], **body})

    # -- methods ------------------------------------------------------------------
    def _send(self, params):
        msg = params["message"]
        if msg["messageId"] in self.by_message_id:        # retry: same answer, no new work
            return self._snapshot(self.tasks[self.by_message_id[msg["messageId"]]])
        if "taskId" in msg:                               # follow-up to an interrupted task
            task = self.tasks[msg["taskId"]]
            if task.state not in {"INPUT_REQUIRED", "AUTH_REQUIRED"}:
                raise ValueError(f"task {task.id} is {task.state}, not awaiting input")
        else:
            n = next(self._ids)
            task = Task(f"task-{n}", msg.get("contextId") or f"ctx-{n}")
            self.tasks[task.id] = task
            self._emit(task, "status", {"state": "SUBMITTED"})
        self.by_message_id[msg["messageId"]] = task.id
        task.history.append(msg)
        self._work(task, msg)
        return self._snapshot(task)

    def _get(self, params):
        return self._snapshot(self.tasks[params["id"]])

    def _list(self, params):
        wanted = params.get("state")
        return [self._snapshot(t) for t in self.tasks.values() if not wanted or t.state == wanted]

    def _cancel(self, params):
        task = self.tasks[params["id"]]
        self._transition(task, "CANCELED")
        return self._snapshot(task)

    def _events(self, params):
        """Resubscribe: everything for this task after the client's last seen sequence."""
        return [e for e in self.events
                if e["taskId"] == params["id"] and e["seq"] > params.get("after", 0)]

    def _push_set(self, params):
        self.webhooks[params["id"]] = params["url"]
        return {"ok": True}

    # -- the agent's own logic (opaque to the client) -----------------------------
    def _work(self, task, msg):
        data = {k: v for p in msg["parts"] if "data" in p for k, v in p["data"].items()}
        dispute = {k: v for m in task.history for p in m["parts"] if "data" in p
                   for k, v in p["data"].items()}
        if task.state == "SUBMITTED" and dispute.get("amount_usd", 0) > 10_000:
            return self._transition(task, "REJECTED", self._agent_text(
                "Disputes over $10,000 go to the controller, not this agent."))
        self._transition(task, "WORKING")
        if dispute["amount_usd"] > 500 and "approval_ref" not in data:
            return self._transition(task, "INPUT_REQUIRED", self._agent_text(
                "Credits over $500 need an approval_ref from a named approver."))
        memo_id = f"CM-{len(self.ledger_writes) + 1001}"
        self.ledger_writes.append((memo_id, dispute["invoice_id"], dispute["amount_usd"]))
        artifact = {"artifactId": f"{task.id}-memo", "name": "credit_memo",
                    "parts": [{"data": {"memo_id": memo_id, "invoice_id": dispute["invoice_id"],
                                        "amount_usd": dispute["amount_usd"],
                                        "approval_ref": dispute.get("approval_ref")}}]}
        task.artifacts.append(artifact)
        self._emit(task, "artifact", artifact)
        self._transition(task, "COMPLETED")

    # -- helpers ------------------------------------------------------------------
    def _agent_text(self, text):
        return {"role": "agent", "parts": [{"text": text}]}

    def _transition(self, task, state, message=None):
        task.to(state, message)
        self._emit(task, "status", {"state": state, "message": message})

    def _emit(self, task, kind, payload):
        event = {"seq": len(self.events) + 1, "taskId": task.id, "kind": kind, "payload": payload}
        self.events.append(event)
        if task.id in self.webhooks:
            http_post(self.webhooks[task.id], json.dumps(event))

    def _snapshot(self, task):
        return {"id": task.id, "contextId": task.context_id,
                "status": {"state": task.state, "message": task.status_message},
                "artifacts": task.artifacts}


class A2AClient:
    """The calling agent's side. It knows the card and the wire; nothing else."""

    def __init__(self, card_url, transport):
        self.card = json.loads(WELL_KNOWN[card_url])
        self.transport, self._rid = transport, itertools.count(1)

    def supports(self, skill_id):
        return any(s["id"] == skill_id for s in self.card["skills"])

    def call(self, method, **params):
        raw = json.dumps({"jsonrpc": "2.0", "id": next(self._rid), "method": method,
                          "params": params})
        resp = json.loads(self.transport(raw))
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def send(self, parts, message_id, task_id=None, context_id=None):
        msg = {"role": "user", "parts": parts, "messageId": message_id}
        if task_id:
            msg["taskId"] = task_id
        if context_id:
            msg["contextId"] = context_id
        return self.call("message/send", message=msg)


finance = FinanceAgentServer()
client = A2AClient("https://finance-agents.internal.example/.well-known/agent-card.json",
                   transport=finance.handle)
assert client.supports("resolve_invoice_dispute")

dispute = {"invoice_id": "INV-2291", "amount_usd": 1200, "reason": "duplicate seat line"}
first = client.send([{"text": "Acme disputes a duplicated line."}, {"data": dispute}],
                    message_id="slack-C042:1717:dispute", context_id="acme-case-88")
print("1st send :", first["id"], first["status"]["state"], "|",
      first["status"]["message"]["parts"][0]["text"])

# The HTTP response was lost; the client retries with the SAME messageId.
retry = client.send([{"data": dispute}], message_id="slack-C042:1717:dispute")
print("retry    :", retry["id"], retry["status"]["state"])

done = client.send([{"data": {"approval_ref": "APR-7781"}}],
                   message_id="slack-C042:1717:approval", task_id=first["id"])
memo = done["artifacts"][0]["parts"][0]["data"]
print("follow-up:", done["id"], done["status"]["state"], "|", memo)

try:
    client.call("tasks/cancel", id=first["id"])
except RuntimeError as exc:
    print("cancel   :", exc)

assert retry["id"] == first["id"] and len(finance.tasks) == 1, "retry must not create work"
assert finance.tasks[first["id"]].status_log == [
    "SUBMITTED", "WORKING", "INPUT_REQUIRED", "WORKING", "COMPLETED"]
assert len(finance.ledger_writes) == 1 and memo["approval_ref"] == "APR-7781"
assert client.call("tasks/list", state="COMPLETED")[0]["id"] == first["id"]
```

Read the transcript as a sequence of wire exchanges:

```text
client:  message/send {messageId: ...:dispute, parts: [text, data]}
server:  task-1 INPUT_REQUIRED  "Credits over $500 need an approval_ref"
client:  message/send {messageId: ...:dispute}            (network retry)
server:  task-1 INPUT_REQUIRED  (same task, no new work, no ledger write)
client:  message/send {taskId: task-1, messageId: ...:approval, data: {approval_ref}}
server:  task-1 COMPLETED  artifacts: [credit_memo CM-1001]
client:  tasks/cancel task-1
server:  error: COMPLETED -> CANCELED is illegal
```

The client never learned how Finance decides anything. It learned what Finance needs (from the card and the `INPUT_REQUIRED` message) and what Finance produced (the artifact). That opacity is the feature. It is why two teams can evolve independently.

### Streaming, resubscribe, and push notifications

A synchronous `message/send` works when the task finishes in seconds. Finance's approval can take hours. A2A gives three ways to follow a task, and they are three views of the same event log:

```text
polling         client calls tasks/get every N seconds       simple, wasteful, laggy
streaming       server holds an SSE stream open, sends        live UI, but the client must
                status-update and artifact-update events     stay connected
push            client registers a webhook; server POSTs      hours-long tasks, clients that
                each event to it                              sleep or scale to zero
```

Two properties of push delivery matter in practice. Webhooks are delivered at least once, so the receiver must dedupe. And a webhook endpoint is a public URL that mutates your state, so it must authenticate the sender. The spec leaves the mechanism to the auth configuration you register; the cell uses an HMAC over the body with a per-task secret, which is the common pattern.

```{code-cell} python
import hashlib
import hmac


class WebhookReceiver:
    """Support's endpoint for finance events. At-least-once in, exactly-once effect out."""

    def __init__(self, secret: bytes):
        self.secret, self.seen, self.applied, self.rejected = secret, set(), [], 0

    def sign(self, body: str) -> str:
        return hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()

    def receive(self, body: str, signature: str):
        if not hmac.compare_digest(signature, self.sign(body)):
            self.rejected += 1
            return 401
        event = json.loads(body)
        if event["seq"] in self.seen:           # duplicate delivery: acknowledge, do nothing
            return 200
        self.seen.add(event["seq"])
        self.applied.append((event["seq"], event["kind"], event["payload"].get("state")))
        return 200


finance2 = FinanceAgentServer()
client2 = A2AClient("https://finance-agents.internal.example/.well-known/agent-card.json",
                    transport=finance2.handle)
receiver = WebhookReceiver(secret=b"per-task-secret-from-registration")


def deliver(body):
    """The server's delivery loop. It re-sends every event once to simulate a lost ACK."""
    for _ in range(2):
        receiver.receive(body, receiver.sign(body))


# Open the task, then register the webhook for it (in real A2A this can ride on the first send).
t1 = client2.send([{"data": {"invoice_id": "INV-3004", "amount_usd": 900, "reason": "proration"}}],
                  message_id="m-3004-a")
WEBHOOK_ROUTES["https://support.internal.example/hooks/finance"] = deliver
client2.call("tasks/pushConfig/set", id=t1["id"], url="https://support.internal.example/hooks/finance")
client2.send([{"data": {"approval_ref": "APR-9012"}}], message_id="m-3004-b", task_id=t1["id"])

# A forged event from someone who does not hold the secret.
forged = json.dumps({"seq": 99, "taskId": t1["id"], "kind": "status",
                     "payload": {"state": "CANCELED"}})
assert receiver.receive(forged, "0" * 64) == 401

# The support agent's UI was disconnected after seq 3; it resubscribes from its cursor.
missed = client2.call("tasks/events", id=t1["id"], after=3)
for e in missed:
    print(e["seq"], e["kind"], e["payload"].get("state") or e["payload"].get("name"))
print("applied via webhook:", receiver.applied)

assert [s for _, _, s in receiver.applied] == ["WORKING", None, "COMPLETED"]
assert len(receiver.applied) == len(receiver.seen) == 3, "duplicates were deduped"
assert receiver.rejected == 1 and missed[-1]["payload"]["state"] == "COMPLETED"
```

The webhook only saw events after it was registered, which is realistic: register early, and use the event log (resubscribe) to fill the gap. The receiver's dedupe on `seq` is the same idea as the server's dedupe on `messageId`. Both sides of an at-least-once channel need it.

## How LangGraph does it

LangGraph does not implement A2A for you in the core library. What it gives you is the right place to put the client: a node, with a checkpointer underneath, so a task that waits hours for a human does not hold a process hostage.

The mapping is close to one-to-one:

| A2A concept | Where it lives in the support agent's graph |
|---|---|
| Task id | a field in graph state, persisted by the checkpointer |
| `INPUT_REQUIRED` | `interrupt()` in the delegating node, surfaced to a human in Slack |
| follow-up message with `taskId` | the resumed node sends the human's answer to the same task |
| `messageId` idempotency | deterministic ids derived from the thread, so re-execution is harmless |
| terminal state | a conditional edge: `COMPLETED` goes to reply, anything else to escalation |

One LangGraph fact makes the idempotency non-optional. On resume, the interrupted node **re-runs from the top**. Everything before `interrupt()` executes again. If the node generated a random `messageId`, the re-run would open a second task at Finance. Deterministic ids plus server-side dedupe make the replay a no-op.

```text
invoke #1:  delegate ─ send(:dispute) ─► task-1 INPUT_REQUIRED ─ interrupt() ─► pause, checkpoint
            ... hours pass, a human replies "APR-7781" in Slack ...
invoke #2:  delegate ─ send(:dispute) ─► task-1 (deduped, same snapshot)
                     ─ interrupt() returns "APR-7781"
                     ─ send(:approval, taskId=task-1) ─► COMPLETED
            ─► reply (model drafts the Slack message) ─► END
```

```{code-cell} python
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

finance3 = FinanceAgentServer()
finance_client = A2AClient("https://finance-agents.internal.example/.well-known/agent-card.json",
                           transport=finance3.handle)
reply_model = scripted("Acme: we issued credit memo CM-1001 for $1,200 on INV-2291 "
                       "(approved APR-7781). It will show on your next invoice.")


class SupportState(TypedDict, total=False):
    case_id: str
    dispute: dict
    task: dict
    reply: str


def delegate_to_finance(state: SupportState):
    """Runs again from the top on resume, so every send uses a deterministic messageId."""
    assert finance_client.supports("resolve_invoice_dispute")
    task = finance_client.send([{"data": state["dispute"]}],
                               message_id=f"{state['case_id']}:dispute",
                               context_id=state["case_id"])
    if task["status"]["state"] == "INPUT_REQUIRED":
        question = task["status"]["message"]["parts"][0]["text"]
        answer = interrupt({"remote_agent": finance_client.card["name"],
                            "task_id": task["id"], "question": question})
        task = finance_client.send([{"data": {"approval_ref": answer}}],
                                   message_id=f"{state['case_id']}:approval",
                                   task_id=task["id"])
    return {"task": task}


def route_on_task(state: SupportState):
    return "reply" if state["task"]["status"]["state"] == "COMPLETED" else "escalate"


def reply(state: SupportState):
    memo = state["task"]["artifacts"][0]["parts"][0]["data"]
    # The artifact is data from another trust domain: pass it as data, not as instructions.
    msg = reply_model.invoke([HumanMessage(f"Draft a Slack reply. Credit memo JSON: {json.dumps(memo)}")])
    return {"reply": msg.content}


def escalate(state: SupportState):
    return {"reply": f"Escalated to a human: finance task ended {state['task']['status']['state']}."}


builder = StateGraph(SupportState)
builder.add_node("delegate", delegate_to_finance)
builder.add_node("reply", reply)
builder.add_node("escalate", escalate)
builder.add_edge(START, "delegate")
builder.add_conditional_edges("delegate", route_on_task, ["reply", "escalate"])
builder.add_edge("reply", END)
builder.add_edge("escalate", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "acme-case-88"}}
paused = graph.invoke({"case_id": "acme-case-88",
                       "dispute": {"invoice_id": "INV-2291", "amount_usd": 1200,
                                   "reason": "duplicate seat line"}}, config)
pending = paused["__interrupt__"][0].value
print("paused on:", pending)
assert graph.get_state(config).next == ("delegate",)

final = graph.invoke(Command(resume="APR-7781"), config)
print("reply    :", final["reply"])
print("finance  :", finance3.tasks["task-1"].status_log)

assert len(finance3.tasks) == 1, "node re-run on resume must not open a second task"
assert len(finance3.ledger_writes) == 1
assert final["task"]["status"]["state"] == "COMPLETED" and "CM-1001" in final["reply"]
```

The checkpointer is doing what `Task` does on the server side: holding durable state keyed by an id so a later message can pick up the work. Two state machines, one on each side of the wire, each owned by the team that runs it. The `thread_id` is your task id; Finance's `task-1` is theirs. You store theirs in your state so you can always ask them about it.

### Three ways to reach another agent

07 introduced node, subgraph, and tool. A2A adds a fourth that looks like the tool from the caller's side but behaves differently underneath:

| | Subgraph | Agent-as-tool | A2A remote agent |
|---|---|---|---|
| Who owns the code | you | you | another team or vendor |
| Process boundary | same process | same process (usually) | network |
| State visibility | shared or mapped state | hidden; one summary back | hidden; card, task, artifacts |
| Duration | seconds to minutes | seconds to minutes | seconds to days |
| Mid-task input | parent's `interrupt` | awkward | `INPUT_REQUIRED` on the task |
| Failure model | exceptions | exceptions as tool errors | task states, timeouts, retries, auth |
| Pick it when | you own both and want one trace | you want context isolation | the boundary is organizational |

The rule of thumb: do not reach for A2A to connect two agents you own and deploy together. A subgraph gives you one checkpointer, one trace, and type-checked state. A2A pays off when independent deployment, independent ownership, or independent security domains are real.

### The live version: the remote agent as a tool the model chooses

In the graph above, code decided to delegate. Often you want the model to choose, with the A2A client hidden behind an ordinary tool. The model sees a narrow schema; your software owns the protocol.

```{code-cell} python live
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

live_finance = FinanceAgentServer()
live_client = A2AClient("https://finance-agents.internal.example/.well-known/agent-card.json",
                        transport=live_finance.handle)


@tool
def finance_resolve_dispute(invoice_id: str, amount_usd: int, reason: str,
                            approval_ref: str | None = None) -> str:
    """Ask the Finance team's agent to credit a disputed invoice line.
    Credits over $500 require approval_ref; if missing, the result says so."""
    parts = [{"data": {"invoice_id": invoice_id, "amount_usd": amount_usd, "reason": reason}}]
    task = live_client.send(parts, message_id=f"live:{invoice_id}:{amount_usd}")
    if task["status"]["state"] == "INPUT_REQUIRED" and approval_ref:
        task = live_client.send([{"data": {"approval_ref": approval_ref}}],
                                message_id=f"live:{invoice_id}:approval", task_id=task["id"])
    return json.dumps(task)


agent = create_agent(init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6")),
                     tools=[finance_resolve_dispute])
out = agent.invoke({"messages": [HumanMessage(
    "Acme was double charged $1,200 on INV-2291. Approval ref is APR-7781. Get it credited.")]})
print(out["messages"][-1].content)
print(live_finance.ledger_writes)
```

## What goes wrong in production

1. **Duplicate work on retry.** Symptom: two credit memos for one dispute. Cause: the client retried after a timeout with a fresh message id, or the LangGraph node re-ran on resume and generated a new one. Fix: deterministic `messageId`s derived from your own durable ids, server-side dedupe, and follow-ups addressed by `taskId`. Treat every cross-agent send as a non-idempotent HTTP call until proven otherwise.

2. **Orphaned interrupted tasks.** Symptom: Finance's dashboard shows 400 tasks in `INPUT_REQUIRED`, some months old. Cause: the client crashed or the human never answered, and nothing owns the timeout. Fix: a TTL on both sides. The client stores remote task ids in state and runs a sweep (the list-tasks operation makes this cheap) that cancels or escalates tasks older than the SLA.

3. **Lost or forged push notifications.** Symptom: support shows "pending" for a task that finished hours ago, or a task "cancels" itself. Cause: webhook deliveries dropped during a deploy, or an unauthenticated endpoint accepted a forged POST. Fix: authenticate every delivery, dedupe by sequence or event id, and reconcile periodically with a get or resubscribe call. The push channel is a latency optimization, not the source of truth.

4. **Artifacts treated as instructions.** Symptom: the support agent emails the customer list after reading a remote artifact. Cause: text from another agent was concatenated into the system prompt. Fix: remote messages and artifacts are untrusted input, the same as retrieved documents (31. Agent Security). Prefer structured `data` parts validated against a schema over free text. Consequential actions stay behind your own approvals.

5. **Confused deputy and credential sprawl.** Symptom: the finance agent credits an account the requesting user has no rights to. Cause: support called Finance with a powerful service credential, so Finance could not tell who the end user was. Fix: authenticate as the agent, carry the user's identity or a scoped delegated token, and let Finance authorize against it. `AUTH_REQUIRED` exists for the case where the remote agent needs more authority than it was given (39. Agent Identity and Governance).

6. **Broken traces at the boundary.** Symptom: "why did it take four hours?" and your trace ends at the send. Cause: no correlation between your run and their task. Fix: store the remote task id and `contextId` on your span, propagate a trace context header across the call, and log every state transition with its timestamp. The server's event log is also your latency breakdown.

7. **Delegation loops.** Symptom: agent A asks B, B asks C, C asks A, and the bill grows. Cause: horizontal protocols make any agent reachable from any other. Fix: a hop count or delegation budget in message metadata, refused past a limit, and per-caller rate limits on the server. This is the peer-network topology from 07, and it needs the same bounds.

8. **Card drift.** Symptom: calls that worked yesterday return `REJECTED`. Cause: the remote team changed a skill's input contract behind the same card URL. Fix: version the card, pin the skill contract you tested against, validate `data` parts against a schema on both sides, and run a contract test against the remote agent in CI.

## What changed by 2026

- **A2A reached 1.0 under neutral governance.** Google introduced A2A in April 2025 and donated it to the Linux Foundation in June 2025. The v1.0.0 spec was published in March 2026, with a 1.0.1 revision in May 2026. As of mid-2026 it is hosted by the Agentic AI Foundation, a Linux Foundation foundation formed in December 2025 that also hosts MCP. Both of the protocols in this track now sit under the same neutral body.
- **v1.0 tightened the data model.** The 1.0 spec flattened `Part`, added task listing, and modernized the OAuth flows (device code, PKCE). If you learned A2A from 2025 examples, expect field-level differences; use an SDK rather than hand-rolled JSON.
- **The MCP/A2A line is clearer and also narrower.** "MCP is vertical, A2A is horizontal" is the A2A project's own framing. At the same time, MCP's Tasks extension gives long-running tool calls task handles, and MCP's roadmap mentions agentic messaging. Expect overlap at the edges; choose by ownership and opacity, not by which protocol has a feature.
- **Frameworks expose and consume A2A.** Google ADK documents exposing an agent over A2A and consuming remote A2A agents (marked experimental in its current docs). Check your framework's current docs before assuming native support; wrapping an SDK client in a node or tool, as above, always works.
- **Durable execution made cross-agent waits cheap.** Checkpointed interrupts mean a run blocked on another organization's human costs storage, not a live process. That is what makes `INPUT_REQUIRED` practical rather than theoretical.
- **Identity moved to the front.** Agent identity, delegated credentials, and per-agent authorization are now first-class design topics, not afterthoughts. The cross-agent call is where they bite first.

## Interview questions

**Q1. What is A2A and how does it differ from MCP?**

Strong: MCP connects one agent to its capabilities: tools, resources, and prompts exposed by servers that do not plan and should not see the conversation. A2A connects an agent to a peer agent that has its own model, tools, and goals and is opaque to the caller. The A2A project frames it as MCP being vertical and A2A horizontal. Concretely, A2A's core objects are an Agent Card for discovery, a stateful Task with a defined lifecycle including input-required, Messages made of Parts, and Artifacts as typed outputs. A single system often uses both: the finance agent talks to its ledger over MCP and to the support agent over A2A.

Weak: "MCP is for tools and A2A is for chat between bots."

**Q2. Walk me through the A2A task lifecycle. Why does INPUT_REQUIRED exist?**

Strong: A task starts submitted, moves to working, and ends in one of four terminal states: completed, failed, canceled, or rejected. Two interrupted states, input-required and auth-required, mean the remote agent is blocked on the client. Input-required exists because real work needs clarification or approval mid-flight, and without it you either fail the task or block a connection for hours. The client answers with a new message carrying the same task id, and the task resumes. It is the cross-organization version of LangGraph's interrupt, and I enforce the transitions in code so terminal states stay terminal.

Weak: "The task is either running or done."

**Q3. When would you use A2A instead of a subgraph or an agent-as-tool?**

Strong: I choose by ownership and deployment, not by capability. If I own both agents and ship them together, a subgraph gives me one checkpointer, one trace, and typed state, and an agent-as-tool gives me context isolation cheaply. A2A earns its cost when the other agent belongs to another team or vendor, deploys independently, sits in a different security domain, or runs for hours with its own human approvals. Using A2A between two agents in one repo adds serialization, auth, and failure modes for no benefit.

Weak: "A2A is the standard, so every agent should speak it."

**Q4. The remote agent timed out. Your client retries. How do you avoid doing the work twice?**

Strong: A timeout tells you nothing about whether the work happened, so the send has to be idempotent. I derive the message id deterministically from my own durable ids, such as the case id plus the step, and the server dedupes on it and returns the existing task. Follow-ups are addressed to the task id, never re-sent as new tasks. On the LangGraph side this matters twice, because an interrupted node re-runs from the top on resume, so any id generated inside the node must be deterministic. Finally I reconcile with a get-task call instead of trusting my last response.

Weak: "Set a longer timeout."

**Q5. Polling, streaming, or push notifications for a long-running remote task?**

Strong: They are three views of one event log. Streaming over SSE is right when a client is connected and wants live progress, such as a UI. Push notifications are right when the task takes hours and the client may sleep, scale to zero, or restart; the server posts events to a registered webhook. Polling is the fallback and the reconciliation path. With push I authenticate every delivery, dedupe because delivery is at least once, and still reconcile periodically, because the webhook is an optimization, not the source of truth.

Weak: "Use websockets."

**Q6. How do you secure agent-to-agent calls?**

Strong: Start from the card: it declares the auth schemes, and the client authenticates as itself with short-lived credentials. The harder part is authorization on behalf of a user: the caller should carry the end user's identity or a scoped delegated token so the remote agent can refuse actions that user cannot take, which avoids the confused-deputy problem. Everything the remote agent returns is untrusted input, so artifacts are validated against a schema and never concatenated into instructions. Add hop limits against delegation loops and per-caller rate limits on the server.

Weak: "Use an API key."

**Q7. How does multi-agent orchestration break down across organizational boundaries?**

Strong: Inside one process, failures are exceptions and state is shared. Across organizations you inherit every distributed-systems failure: partial failure, duplicate delivery, lost callbacks, version skew in the other agent's contract, and traces that stop at your boundary. You also lose visibility, since the other agent is opaque by design, so you cannot debug it by reading its prompt. The mitigations are protocol discipline: explicit task states, idempotent sends, contract tests against the card, TTLs on interrupted tasks, and correlation ids on both sides. And the first question is always whether you need a peer at all or just a tool.

Weak: "Agents get confused talking to each other."

## Build it yourself

Extend the in-process pair into something you would trust with money:

1. Add a `metadata` field to messages carrying a `hops` counter. The server rejects (state `REJECTED`, with a reason message) any task whose `hops` exceeds 3.
2. Add a `created_at` logical clock to `Task` and a client-side `sweep(now, ttl)` that lists `INPUT_REQUIRED` tasks and cancels any older than `ttl`.
3. Make cancellation propagate: add a `cancel` node to the LangGraph version that, when the human answers "cancel" instead of an approval ref, calls `tasks/cancel` on the remote task and routes to `escalate`.
4. Validate artifacts on the client with a Pydantic model `CreditMemo(memo_id: str, invoice_id: str, amount_usd: int, approval_ref: str | None)` and route to `escalate` if validation fails.

Acceptance checks:

- A message with `hops=4` produces a task whose `status_log` ends in `REJECTED` and zero ledger writes.
- `sweep(now=100, ttl=50)` cancels a task created at `t=10` still in `INPUT_REQUIRED` and leaves a task created at `t=80` untouched.
- Resuming the graph with `Command(resume="cancel")` leaves the remote task `CANCELED`, `len(ledger_writes) == 0`, and the reply mentions escalation.
- An artifact with `amount_usd: "a lot"` is rejected by the Pydantic model and the graph ends in `escalate`.
- The happy path from this lesson still produces exactly one task and one ledger write.

## Key takeaways

- MCP connects an agent to its capabilities; A2A connects an agent to an opaque peer that plans and acts on its own.
- The Agent Card says what an agent does and how to reach it, never how it works internally.
- A Task is a state machine with four terminal states and two interrupted ones; enforce the transitions in code.
- `INPUT_REQUIRED` is an interrupt across an organizational boundary, and it pairs naturally with a checkpointed LangGraph `interrupt`.
- Every cross-agent send needs a deterministic message id, because timeouts, retries, and resumed nodes all replay it.
- Streaming, push, and polling are views of one event log; authenticate and dedupe pushes, and reconcile against the task.
- Use A2A when the boundary is organizational; use a subgraph or an agent-as-tool when you own both sides.
