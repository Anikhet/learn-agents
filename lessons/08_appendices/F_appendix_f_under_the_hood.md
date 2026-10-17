# F. Under the Hood: what a graph runtime actually does

*Track A (agent engineering) with a Track D flavor. Book: Appendix F (Under the Hood). Prerequisites: 00. Course Orientation, 13. Human in the Loop, 28. Stateful Graphs and Checkpoints, D. Building an Agent.*

You have used `StateGraph`, reducers, `interrupt`, `InMemorySaver` and `stream` for thirty lessons. This appendix opens the box. A graph runtime is not magic and it is not large: the scheduling core of LangGraph is a Pregel-style bulk-synchronous loop over channels. We will build one in about seventy lines, run the capstone's approval gate on it, then run the same graph on LangGraph and compare checkpoint for checkpoint.

The interview question this answers is "what does LangGraph actually do when I call `invoke`?" Most candidates say "it runs the nodes." The strong answer names supersteps, channels, reducers, the checkpoint boundary, and why `interrupt` re-runs a node.

## The problem

Take the capstone's approval flow and strip it to its skeleton:

```text
a ──► b ──┐
  │       ├──► gate (interrupt) ──► d
  └──► c ─┘
```

Node `a` writes the state. `b` and `c` both read what `a` wrote and both append to a log. `gate` pauses for a human. `d` finishes. Ask five precise questions about this picture and the naive "call the functions in order" runtime cannot answer any of them:

1. Do `b` and `c` run in parallel? If they do, does `c` see `b`'s write?
2. `b` and `c` both append to `log`. Who wins? Is anything lost?
3. The process dies during `gate`. What exactly is on disk, and from where does the run continue?
4. `gate` calls `interrupt()`. How does a function "pause" and later "return" a value it did not have?
5. When you `stream` the run, what are the units you receive: nodes, steps, tokens?

The naive runtime is a `for` loop over a list of functions:

```python
for node in [a, b, c, gate, d]:
    state = node(state)
```

It answers the questions badly. `b` and `c` are sequential, so `c` sees `b`'s write and the order of the log depends on the order of the list. There is no checkpoint, so a crash loses everything. `interrupt` cannot be implemented at all: a Python function cannot suspend itself and be resumed in another process. And there is nothing to stream except the final state.

Pregel, the graph-processing model that LangGraph borrows its core from, answers all five with one idea: **the superstep**.

```text
superstep t:
   1. every node scheduled for t reads the SAME snapshot of the channels
   2. every scheduled node runs (in any order, or in parallel)
   3. all of their writes are collected
   4. writes are merged into the channels through reducers
   5. a checkpoint is written
   6. the nodes triggered by those writes are scheduled for t+1
```

Everything the runtime offers falls out of that loop. Parallel nodes see the same snapshot because reads happen before any write is applied. Two writes to one channel go through the channel's reducer, so `operator.add` appends both and a last-value channel refuses the ambiguity. A checkpoint is written at exactly one boundary, so a crash always resumes at a step boundary. An interrupt is an exception that abandons the current superstep, leaving the previous checkpoint and the scheduled frontier intact. Streaming emits one event per superstep, or one per node update, or one per token, because those are the natural granularities the loop already has.

## Build it without a framework

Here is the runtime. Read `stream` slowly; it is the entire mechanism.

```{code-cell} python
import copy
import operator


class Interrupt(Exception):
    """Raised by interrupt(); the runtime catches it and checkpoints the pending step."""

    def __init__(self, value):
        self.value = value


_RESUME = {}                      # filled by the runtime for the superstep being re-run


def interrupt(value):
    if "value" in _RESUME:        # second time through: the human has answered
        return _RESUME.pop("value")
    raise Interrupt(value)        # first time through: stop the world


class MiniGraph:
    """A Pregel-style runtime: channels + reducers, supersteps, one checkpoint per step."""

    def __init__(self, reducers, entry):
        self.reducers, self.entry = reducers, entry        # channel -> reducer (None = last value)
        self.nodes, self.threads = {}, {}                  # thread_id -> list of checkpoints

    def add_node(self, name, fn, edges=()):
        """edges: successor name(s), or a router fn(state) -> name(s). () means END."""
        self.nodes[name] = (fn, edges)

    def _apply(self, values, writes):
        for key, val in writes:                            # reducers merge writes from one superstep
            red = self.reducers.get(key)
            values[key] = red(values[key], val) if red and key in values else val

    def stream(self, input, thread, resume=None):
        cps = self.threads.setdefault(thread, [])
        if not cps:                                        # step -1: the input checkpoint
            values = {}
            self._apply(values, input.items())
            cps.append({"step": -1, "values": copy.deepcopy(values), "next": [self.entry]})
        values = copy.deepcopy(cps[-1]["values"])
        frontier, step = list(cps[-1]["next"]), cps[-1]["step"] + 1
        if resume is not None:
            _RESUME["value"] = resume
        while frontier:                                    # one iteration = one superstep
            snapshot, writes, nxt = copy.deepcopy(values), [], []
            for name in frontier:                          # every task reads the SAME snapshot
                fn, edges = self.nodes[name]
                try:
                    update = fn(snapshot) or {}
                except Interrupt as exc:                   # step abandoned, frontier kept
                    cps.append({"step": step, "values": copy.deepcopy(values), "next": frontier,
                                "interrupt": {"node": name, "value": exc.value}})
                    yield {"__interrupt__": exc.value}
                    return
                writes += list(update.items())
                after = copy.deepcopy(snapshot)            # the router sees this node's own writes
                self._apply(after, update.items())
                targets = edges(after) if callable(edges) else edges
                nxt += [targets] if isinstance(targets, str) else list(targets)
            self._apply(values, writes)                    # bulk-synchronous: apply after all tasks
            frontier = list(dict.fromkeys(nxt))
            cps.append({"step": step, "values": copy.deepcopy(values), "next": frontier})
            yield {"step": step, "ran": list(cps[-2]["next"]), "next": frontier}
            step += 1

    def invoke(self, input, thread, resume=None):
        out = {}
        for out in self.stream(input, thread, resume):
            pass
        values = self.threads[thread][-1]["values"]
        return {**values, "__interrupt__": out["__interrupt__"]} if "__interrupt__" in out else values


print("MiniGraph ready:", [m for m in dir(MiniGraph) if not m.startswith("_")])
assert callable(interrupt)
```

Map the vocabulary before running it:

| Pregel / LangGraph term | In `MiniGraph` |
|---|---|
| channel | a key in `values` |
| reducer (`Annotated[list, operator.add]`) | `self.reducers[key]` |
| last-value channel | a key with no reducer: the write replaces |
| task | one `(name, fn)` in the current `frontier` |
| superstep | one iteration of the `while frontier` loop |
| edge / conditional edge | `edges` as a name, a list, or a router callable |
| checkpoint | one dict appended to `cps` |
| thread | the key in `self.threads` |
| `interrupt()` | raise `Interrupt`; on resume return `_RESUME["value"]` |
| `stream_mode="updates"` | the `yield` per superstep |

Now the five questions, answered by running the skeleton graph.

```{code-cell} python
g = MiniGraph(reducers={"log": operator.add}, entry="a")
g.add_node("a", lambda s: {"log": ["a"], "n": 1}, edges=["b", "c"])
g.add_node("b", lambda s: {"log": [f"b saw n={s.get('n')}"]}, edges="gate")
g.add_node("c", lambda s: {"log": [f"c saw n={s.get('n')}, log_len={len(s['log'])}"]}, edges="gate")
g.add_node("gate", lambda s: {"log": [f"gate:{interrupt({'question': 'approve?'})}"]}, edges="d")
g.add_node("d", lambda s: {"log": ["d"]})

for event in g.stream({"log": [], "n": 0}, thread="t1"):
    print(event)
cps = g.threads["t1"]
print("checkpoints:", [(c["step"], c["next"]) for c in cps])
assert cps[-1]["next"] == ["gate"] and "interrupt" in cps[-1]
assert "gate:" not in " ".join(cps[-1]["values"]["log"]), "nothing from the abandoned step leaked"

# Later, possibly in another process: resume the same thread with the human's answer.
out = g.invoke(None, thread="t1", resume="yes")
print(out["log"])
print("checkpoints:", [(c["step"], c["next"]) for c in cps])
assert out["log"] == ["a", "b saw n=1", "c saw n=1, log_len=1", "gate:yes", "d"]
assert [c["next"] for c in cps] == [["a"], ["b", "c"], ["gate"], ["gate"], ["d"], []]
```

Read the output against the questions:

1. **Parallelism.** `b` and `c` ran in the same superstep (`ran: ['b', 'c']`). `c` saw `log_len=1`: it saw `a`'s write but not `b`'s, because both read the snapshot taken before the step. Whether the runtime runs them in threads or sequentially is an implementation detail; the semantics are the same.
2. **Two writes, one channel.** Both appends survived because `log` has the `operator.add` reducer. Change the reducer to `None` and the last write wins silently; LangGraph is stricter and raises `InvalidUpdateError`, as the next section shows.
3. **What is on disk during the pause.** Checkpoint step 2: the values as of the end of step 1, `next=['gate']`, and the interrupt payload. Nothing from the abandoned step. Resume starts from `next`.
4. **How interrupt "returns".** It does not resume the function mid-way. The `gate` node runs again from the top, and this time `interrupt()` finds the resume value and returns it. That is why the course keeps saying: no side effects before `interrupt()`.
5. **Streaming units.** One event per superstep. LangGraph's `updates` mode is this, keyed by node; `values` mode emits the full state after each superstep; `messages` emits tokens as they are produced inside a node.

### The agent loop is a two-node cycle

The ReAct loop from lesson 00 is nothing special to the runtime: two nodes, one router, and a frontier that alternates. The step count in the checkpoints is the recursion limit's unit.

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


from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def stripe_get_invoice(invoice_id: str) -> dict:
    """Fetch one Stripe invoice by id."""
    return {"invoice_id": invoice_id, "seats_billed": 48, "amount_cents": 48000}


model = scripted([("stripe_get_invoice", {"invoice_id": "in_1042"})],
                 "Invoice in_1042 billed 48 seats for $480.").bind_tools([stripe_get_invoice])

agent_graph = MiniGraph(reducers={"messages": operator.add}, entry="agent")
agent_graph.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]},
                     edges=lambda s: "tools" if s["messages"][-1].tool_calls else ())
agent_graph.add_node("tools", lambda s: {"messages": [
    ToolMessage(content=str(stripe_get_invoice.invoke(c["args"])), tool_call_id=c["id"])
    for c in s["messages"][-1].tool_calls]}, edges="agent")

events = list(agent_graph.stream({"messages": [HumanMessage("What did Acme pay?")]}, thread="acme"))
for e in events:
    print(e)
final = agent_graph.threads["acme"][-1]["values"]["messages"][-1]
print(final.content)
assert [e["ran"] for e in events] == [["agent"], ["tools"], ["agent"]]
assert final.content.startswith("Invoice in_1042")
assert len(agent_graph.threads["acme"]) == 4     # input + one checkpoint per superstep
```

Three supersteps, four checkpoints. A twenty-step agent run produces twenty-one checkpoints, each a full copy of the state in this toy. Real checkpointers store the channel values once per step too, which is why lesson 34's context trimming shows up as checkpoint size as much as token count.

## How LangGraph does it

Now the same skeleton graph in LangGraph. The claim to test: same supersteps, same checkpoint count, same `next` sequence, same interrupt semantics. Where LangGraph differs, it differs by being stricter or more complete.

```{code-cell} python
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt as lg_interrupt


class S(TypedDict):
    log: Annotated[list, operator.add]     # a reducer channel, as in MiniGraph
    n: int                                 # a last-value channel


def gate(state):
    answer = lg_interrupt({"question": "approve?"})
    return {"log": [f"gate:{answer}"]}


b = StateGraph(S)
b.add_node("a", lambda s: {"log": ["a"], "n": 1})
b.add_node("b", lambda s: {"log": [f"b saw n={s['n']}"]})
b.add_node("c", lambda s: {"log": [f"c saw n={s['n']}, log_len={len(s['log'])}"]})
b.add_node("gate", gate)
b.add_node("d", lambda s: {"log": ["d"]})
for src, dst in [(START, "a"), ("a", "b"), ("a", "c"), ("b", "gate"), ("c", "gate"), ("gate", "d"), ("d", END)]:
    b.add_edge(src, dst)
lg = b.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

first = lg.invoke({"log": [], "n": 0}, cfg)
paused = lg.get_state(cfg)
print("paused, next =", paused.next, "payload =", paused.tasks[0].interrupts[0].value)
final = lg.invoke(Command(resume="yes"), cfg)
print(final["log"])

history = list(lg.get_state_history(cfg))            # newest first
lg_next = [h.next for h in reversed(history)]
print("LangGraph next per checkpoint:", lg_next)
assert final["log"] == out["log"], "same result as MiniGraph"
assert len(history) == len(cps) == 6, "same number of checkpoints"
# LangGraph stores `next` = tasks about to run; MiniGraph stores the same list after the step.
assert [list(n) for n in lg_next] == [["__start__"], ["a"], ["b", "c"], ["gate"], ["d"], []]

# Where LangGraph is stricter: two writes to a last-value channel in one superstep is an error.
strict = StateGraph(S)
strict.add_node("b", lambda s: {"n": 1})
strict.add_node("c", lambda s: {"n": 2})
strict.add_edge(START, "b"); strict.add_edge(START, "c")
strict.add_edge("b", END); strict.add_edge("c", END)
try:
    strict.compile().invoke({"log": [], "n": 0})
    raise AssertionError("expected InvalidUpdateError")
except InvalidUpdateError as exc:
    print("InvalidUpdateError:", str(exc).split("\n")[0])
```

Line them up:

| MiniGraph | LangGraph | Difference |
|---|---|---|
| `values` dict | channels (`LastValue`, `BinaryOperatorAggregate`, `Topic`, ...) | LangGraph has typed channels; the `Annotated` reducer picks one |
| `frontier` | tasks computed from channel versions | LangGraph triggers a node when a channel it subscribes to changed since it last ran; edges compile to hidden `branch:` channels |
| `while frontier` | the Pregel loop (`langgraph.pregel`) | same superstep semantics; LangGraph runs tasks in a thread pool |
| `_apply` after the step | `apply_writes` | same bulk-synchronous rule |
| `cps.append` | `checkpointer.put` per superstep | LangGraph also stores `channel_versions` and `versions_seen` so it can compute the next frontier without re-running anything |
| `Interrupt` exception | `GraphInterrupt` | LangGraph saves the *successful* sibling tasks' writes as pending writes so only the interrupted node re-runs |
| `_RESUME` global | resume value in the checkpoint's pending writes | same "re-run the node, return the value on the second pass" behavior |
| last write wins on a plain key | `InvalidUpdateError` | LangGraph refuses ambiguity you did not declare |
| step counter | `recursion_limit` (default 25 supersteps) | same unit |

The `Command(goto=...)` you used in the capstone is a write to a `branch:` channel by the node itself. `Send` (lesson 03) is a task created with its own private input instead of the shared snapshot. Subgraphs are nodes whose function is another Pregel loop with its own namespace in the checkpointer. Once you have the superstep in your head, every LangGraph primitive is a variant of "who is in the frontier and what did they write."

### Streaming modes are the loop's natural events

`stream_mode` picks which of the loop's events you receive. `updates` is one event per node write, grouped by superstep. `values` is the full state after each superstep. `messages` is a token-level hook inside model calls. `custom` is whatever a node writes through `get_stream_writer`. `debug` is the checkpoint itself. You can request several at once and get `(mode, chunk)` pairs.

```{code-cell} python
from langgraph.config import get_stream_writer


class ChatState(TypedDict):
    messages: Annotated[list, operator.add]


token_model = scripted("Acme was billed for 48 seats against 40 contracted.")


def answer(state):
    get_stream_writer()({"progress": "asking the model"})       # custom event, no state change
    return {"messages": [token_model.invoke(state["messages"])]}


sg = StateGraph(ChatState)
sg.add_node("answer", answer)
sg.add_edge(START, "answer"); sg.add_edge("answer", END)
streaming = sg.compile()

seen = {"custom": [], "messages": [], "updates": []}
for mode, chunk in streaming.stream({"messages": [HumanMessage("Why is Acme upset?")]},
                                    stream_mode=["custom", "messages", "updates"]):
    if mode == "messages":
        token, metadata = chunk
        seen["messages"].append((token.content, metadata["langgraph_node"]))
    else:
        seen[mode].append(chunk)

print("custom  :", seen["custom"])
print("messages:", "".join(t for t, _ in seen["messages"])[:40], "... in", len(seen["messages"]), "chunks")
print("updates :", list(seen["updates"][0]))
assert seen["custom"] == [{"progress": "asking the model"}]
assert len(seen["messages"]) > 1 and all(node == "answer" for _, node in seen["messages"])
assert "answer" in seen["updates"][0]
```

Notice the layering. `updates` arrived once, after the superstep. `messages` arrived many times, during it. A UI shows tokens as they come (messages), then a "step done" marker (updates), and a progress bar fed by custom events. E. AI Agents on the CLI uses exactly this split for stdout and stderr.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))


def real_answer(state):
    return {"messages": [real_model.invoke(state["messages"])]}


lg2 = StateGraph(ChatState)
lg2.add_node("answer", real_answer)
lg2.add_edge(START, "answer"); lg2.add_edge("answer", END)
for token, metadata in lg2.compile().stream(
        {"messages": [HumanMessage("In one sentence, what is a Pregel superstep?")]},
        stream_mode="messages"):
    print(token.content, end="", flush=True)
print()
```

## What goes wrong in production

1. **Side effects before `interrupt()`.** Symptom: the approval node posts to Slack twice, once before the pause and once after resume. Cause: the node re-runs from the top on resume; the runtime cannot resume a Python frame. Fix: the node before the interrupt is pure; the side effect goes in the next node, as in the capstone. If a side effect must precede the pause, make it idempotent.

2. **Two nodes write the same last-value key in one superstep.** Symptom: `InvalidUpdateError: Can receive only one value per step`. Cause: fan-out nodes both returned `{"result": ...}`. Fix: declare a reducer (`operator.add` for lists, a merge function for dicts), or have each branch write its own key. Do not "fix" it by serializing the branches; you lose the parallelism you fanned out for.

3. **Checkpoint size grows with every step.** Symptom: an hour-long run's checkpoints total gigabytes; `get_state_history` is slow. Cause: every superstep stores the channel values, including a 30 KB tool result in `messages`. Fix: trim or summarize old messages in the agent node, put large results in the Store or object storage and keep references in state, and set retention on the checkpointer (34. Context Engineering, 36. Durable Orchestration).

4. **Recursion limit hit in a healthy loop.** Symptom: `GraphRecursionError` after 25 supersteps on a legitimately long task. Cause: the limit counts supersteps, and a ReAct loop uses two per tool round. Fix: set `recursion_limit` in the config deliberately, and pair it with your own step, time, and cost budgets in state.

5. **Router reads stale state.** Symptom: a conditional edge picks the wrong branch although the node "just set" the flag. Cause: in a fan-out, the router sees the snapshot plus its own node's writes, not the sibling's. Fix: route on values your own node wrote, or add a join node that runs in the next superstep after all writes are merged.

6. **Non-deterministic task order assumed.** Symptom: a log's order flips between runs. Cause: tasks in one superstep may run in a thread pool; the reducer merges in completion order. Fix: never depend on within-superstep order; if order matters, sort in the reducer or add the step index to each item.

7. **Resuming the wrong thread.** Symptom: `Command(resume=...)` on a thread that is not paused starts a fresh run or fails. Cause: resume is only meaningful when the checkpoint has a pending interrupt. Fix: check `get_state(config).next` and the pending interrupts before resuming; treat the interrupt id as part of the approval record.

## What changed by 2026

- **Durable execution is the runtime's job, not the app's.** LangGraph's current docs describe durable execution as a core feature: every superstep checkpointed, resume from any checkpoint, and `interrupt` as the human-in-the-loop primitive. The mechanism is the one you built: a checkpoint boundary per superstep. What changed is that teams expect it by default.
- **Harnesses layer on top of runtimes.** Deep Agents is a LangGraph graph with planning, file tools, and subagents baked in. Anthropic's managed-agents architecture describes the same separation as session (an append-only durable event log), harness (a stateless loop), and sandbox (interchangeable hands). The checkpoint list in `MiniGraph.threads` is a tiny version of that session log.
- **Fresh-context resets over compaction.** Anthropic's harness-design guidance reports that resetting to a fresh context with a handoff artifact often beats compacting a long transcript. In runtime terms, that is a new thread whose input is derived from the old thread's last checkpoint. The superstep model makes that trivial; the hard part is deciding what goes in the handoff.
- **Streaming became a product requirement.** Multi-mode streaming (tokens, node updates, custom progress) is how agent UIs and CLIs show work in progress. The three granularities are the loop's own events; frameworks differ in how they name them.
- **Other runtimes made the same choices.** The OpenAI Agents SDK's sessions and the A2A protocol's task states (working, input-required, completed) encode the same pause-and-resume shape at different layers. Once you see the frontier and the checkpoint, the vocabulary differences stop mattering.

## Interview questions

**Q1. What does a graph runtime like LangGraph actually do when you call `invoke`?**

Strong: It runs a Pregel-style loop. Each iteration, called a superstep, takes a snapshot of the state channels, runs every task scheduled for that step against the same snapshot, collects their writes, merges them through each channel's reducer, writes a checkpoint, and computes the next set of tasks from which channels changed. Edges are compiled into trigger channels, so "node B follows A" means "B is scheduled when A writes B's trigger." The loop ends when no tasks are scheduled or the recursion limit is hit.

Weak: "It calls the nodes in the order of the edges."

**Q2. Two nodes run in parallel and both update the same state key. What happens?**

Strong: It depends on the channel type, which the state annotation chooses. With a reducer like `operator.add`, both writes are merged in the same superstep and nothing is lost, though the order within the step is not guaranteed. With a plain last-value key, LangGraph raises `InvalidUpdateError` because two writes in one step is an ambiguity you did not declare. Neither branch sees the other's write during the step; both read the snapshot from the step before.

Weak: "The second one overwrites the first."

**Q3. How does `interrupt()` work if Python cannot suspend a function?**

Strong: It does not suspend the function. `interrupt` raises an exception that the runtime catches; the runtime keeps the checkpoint from the previous superstep, records the interrupt payload and the pending task, and returns to the caller. On resume with the same thread id, the runtime re-runs the pending node from the top, and this time the `interrupt` call finds the resume value in the checkpoint's pending writes and returns it. Consequently anything before `interrupt` runs twice, which is why side effects go after it or must be idempotent.

Weak: "It pauses the node and waits for input."

**Q4. Where exactly is the checkpoint boundary, and why does that matter for crash recovery?**

Strong: After each superstep, once all writes for that step are applied. So a crash mid-step loses only that step's work and the run resumes by re-executing the scheduled tasks against the last checkpoint. That is a clean boundary for reads, and a hazard for writes: a tool that moved money in the crashed step will run again on resume. So side-effecting tools need idempotency keys, and the runtime's durability guarantee is "at least once per superstep," not "exactly once."

Weak: "It saves after every node so nothing is lost."

**Q5. What is the difference between `stream_mode="updates"`, `"values"`, and `"messages"`?**

Strong: They are three granularities of the same loop. `updates` yields each node's returned delta, once per superstep; `values` yields the full merged state after each superstep; `messages` yields model tokens as they are produced inside a node, tagged with the node name. `custom` lets a node push its own progress events and `debug` yields the checkpoint records. A UI typically streams tokens for the current step and uses updates as step boundaries.

Weak: "Updates is faster."

**Q6. Why does LangGraph count supersteps for the recursion limit rather than model calls?**

Strong: Because the runtime has no notion of a model call; it schedules tasks. A superstep is the unit it can count and checkpoint, so it is the unit it bounds. In a ReAct loop each tool round costs two supersteps, agent and tools, so the default of 25 gives about twelve rounds. Model-call, token, and cost budgets are application concerns and belong in state with reducers.

Weak: "It's just a safety limit."

**Q7. If you were designing a runtime for long-running agents from scratch, what would you keep from Pregel and what would you add?**

Strong: Keep the superstep as the unit of scheduling, checkpointing and streaming; keep reducers as the only way multiple writers touch one channel; keep the interrupt-as-exception model because it makes pause and resume survive process boundaries. Add pending-write persistence so successful siblings of an interrupted task are not re-run, checkpoint retention and offloading of large values, per-task retry policies, and a durable event log separate from state snapshots for audit. Most of those are what LangGraph added on top of the core loop.

Weak: "I'd use a message queue."

## Build it yourself

Extend `MiniGraph`:

1. Add **pending writes**: when one task in a superstep raises `Interrupt`, keep the successful sibling tasks' writes in the checkpoint and on resume run only the interrupted task, merging the saved writes with its new ones.
2. Add a **`recursion_limit`** parameter to `stream` that raises `RecursionError` with the last checkpoint intact when exceeded.
3. Add a **`RetryPolicy`**-like option per node: retry the task up to `n` times on a given exception type before failing the superstep.
4. Add **`Send`**: allow a router to return `("node", private_input)` pairs that schedule tasks whose `fn` receives the private input instead of the snapshot.

Acceptance checks:

- In the `a -> (b, c) -> gate -> d` graph, make `c` the interrupting node and `b` a counter: after resume, `b` has run exactly once.
- A two-node cycle with a router that never returns `()` stops with `RecursionError` at exactly `recursion_limit` checkpoints past the input checkpoint.
- A node that raises `TimeoutError` twice then succeeds completes under a retry policy of 3 and fails under a policy of 2, with the checkpoint list unchanged after the failure.
- A `Send` fan-out over `[1, 2, 3]` produces three tasks in one superstep whose writes merge through `operator.add` into a list of length 3.

## Key takeaways

- A graph runtime is a loop over supersteps: snapshot, run scheduled tasks, merge writes through reducers, checkpoint, schedule the next frontier.
- Parallel nodes read the same snapshot and never see each other's writes until the next step; reducers are the only way two writers share a key.
- The checkpoint boundary is the superstep, so durability is "at least once per step" and side effects need idempotency.
- `interrupt` is an exception, not a suspension: the node re-runs on resume and the second `interrupt` call returns the value.
- Edges, conditional edges, `Command`, and `Send` are all ways of deciding who is in the next frontier.
- Streaming modes are the loop's natural events at three granularities: token, node update, superstep state.
- Seventy lines reproduce the semantics; what LangGraph adds is strictness, pending writes, typed channels, and production checkpointers.
