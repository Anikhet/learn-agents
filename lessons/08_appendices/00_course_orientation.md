# 00. What an agent actually is: the loop, from scratch and in LangGraph

*Track A (agent engineering). Book: Introduction and Chapter 5 (Tool Use), Chapter 17 (ReAct). Prerequisite: none. Start here.*

You want three things from this course: to understand agent architecture, to be able to build it, and to explain your decisions in an AI/Agent Engineer interview. Those are three different tests. Recognizing the word "checkpoint" passes none of them. Explaining what a checkpoint saves, implementing recovery, and predicting what happens to an external write after a crash passes all three.

This lesson builds the one machine that everything else in the course elaborates: the agent loop. First with nothing but Python, so nothing is hidden. Then the exact same machine in LangGraph, so you know what the framework is doing for you.

## The problem

A normal LLM application is a function call:

```text
User
  ↓
LLM
  ↓
Answer
```

```python
response = model.invoke("What is 27 * 43?")
print(response.content)
```

The model gets one input and produces one output. That is enough for summarization, translation, or classification. It is not enough for the request that this course keeps returning to:

> "Why is Acme complaining about their invoice?"

To answer that, something has to search Slack, look up the account in the CRM, check the Stripe invoice, notice the discrepancy, and propose a fix. No single model call can do this because the information is not in the prompt and not in the weights. It is in systems the model must go and ask.

So the shape changes from a function call to a loop:

```text
User goal
    ↓
   LLM
    ↓
Need a tool?
 ┌──┴──┐
Yes    No
 │      │
Tool   Final answer
 │
Result
 │
 └──────→ LLM (again)
```

That loop is the heart of an agent. The book describes the same progression: an LLM alone has no tools, memory, or environment. Once it can use external capabilities over several steps, you have the basis of an agent.

It helps to see agents as a ladder of increasingly capable systems rather than one thing:

```text
LLM
  ↓
LLM + context (system prompt, documents)
  ↓
LLM + retrieval
  ↓
LLM + tools (one shot)
  ↓
LLM + tools + loop            ← this lesson
  ↓
LLM + tools + state + planning + verification
  ↓
long-running agent (checkpoints, memory, approvals)
  ↓
multiple collaborating agents
```

Almost everything in the book is a way of making the "LLM + tools + loop" rung better.

### Workflow versus agent

Suppose we write:

```python
text = summarize(document)
entities = extract_entities(text)
report = create_report(entities)
```

Three model calls, but the route is fixed. The programmer chose it. That is a **workflow**.

An **agent** is different: the model participates in choosing the next action.

```text
Workflow:  developer controls the execution path
Agent:     the model dynamically controls at least part of the execution path
```

This distinction is not academic. Anthropic's guidance on building agents draws exactly this line and recommends the simplest architecture that solves the problem. Many "agents" in production should be workflows, and a strong interview answer says so.

### The central separation

Give the model a calculator. The user asks:

> What is 231 * 482?

How does the model invoke Python? It does not. The model generates a *request*:

```json
{
  "name": "calculator",
  "arguments": {"a": 231, "b": 482, "operation": "multiply"}
}
```

Your program sees that request and actually runs the function. The result goes back to the model:

```text
model:        "I need calculator(231, 482, multiply)"
application:  calculator(...) -> 111342
model:        "The answer is 111342."
```

This separation is the single most important fact in agent engineering:

```text
The model chooses.
Your software executes.
```

Everything about safety, permissions, auditing, retries, and budgets lives on the "your software" side of that line. The model never gets to skip it.

Why not let the model do the arithmetic itself? Because a language model is a probabilistic next-token predictor. For anything that needs determinism, use a deterministic system:

| Need | Use |
|---|---|
| Language understanding, planning, synthesis | LLM |
| Arithmetic | calculator |
| Facts about your data | database |
| Current information | search |
| Money movement | payment API |
| Running code | sandbox |

A mature agent combines probabilistic reasoning with deterministic tools. The loop is the mechanism that combines them.

## Build it without a framework

We will use LangChain's message and tool types because they are convenient and because LangGraph uses them, but the loop itself is plain Python. Nothing is hidden.

Every offline cell in this course uses a scripted stand-in for the model. The script replays fixed turns, which makes the control flow (not the model's judgment) the object of study, and lets every notebook run without network access or keys. A `live` cell at the end shows the same code against a real model.

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

### A tool is a function plus a schema

```{code-cell} python
from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# The model never sees Python. It sees this schema, derived from the
# signature and the docstring. Tool design is schema design.
print(multiply.name)
print(multiply.description)
print(multiply.args)
assert multiply.args == {"a": {"title": "A", "type": "integer"},
                         "b": {"title": "B", "type": "integer"}}
```

The docstring is not decoration. The model decides *whether* and *how* to call a tool from its name, description, argument names, and argument types. This is why "what makes a good tool schema?" is a real interview question, and why 05. Tool Use spends a whole lesson on it.

### One round trip, by hand

Start with the messages list. Ask the model. It returns a tool call rather than text. Execute the tool. Append a `ToolMessage` carrying the result and the call id. Ask again.

```{code-cell} python
from langchain_core.messages import HumanMessage, ToolMessage

tools = [multiply]
tools_by_name = {t.name: t for t in tools}

# Turn 1: the model requests a tool. Turn 2: it answers.
model = scripted([("multiply", {"a": 231, "b": 482})],
                 "231 x 482 = 111,342.").bind_tools(tools)

messages = [HumanMessage("What is 231 multiplied by 482?")]

ai = model.invoke(messages)
messages.append(ai)
assert ai.tool_calls and ai.tool_calls[0]["name"] == "multiply"

call = ai.tool_calls[0]
result = tools_by_name[call["name"]].invoke(call["args"])   # YOUR code runs it
messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = model.invoke(messages)
messages.append(final)

for m in messages:
    print(f"{type(m).__name__:13s} {m.content or m.tool_calls}")
assert result == 111342
assert "111,342" in final.content
```

Look at the four messages. Human, AI (with a tool call), Tool (with the observation), AI (with the answer). That transcript *is* the agent's memory of what happened. Every agent framework you will meet stores some version of this list.

The `tool_call_id` matters. When the model issues several calls in one turn, the id is how each observation is matched to the request that produced it. Lose it and the provider API rejects the message.

### Turn it into a loop

The version above assumed exactly one tool call. A real agent does not know in advance how many actions it needs. So we loop until the model produces a turn with no tool calls.

```{code-cell} python
def run_agent(model, tools, user_text, max_steps=10):
    """The whole agent, in plain Python."""
    tools_by_name = {t.name: t for t in tools}
    messages = [HumanMessage(user_text)]

    for step in range(max_steps):                        # termination: step budget
        ai = model.invoke(messages)                      # reason
        messages.append(ai)
        if not ai.tool_calls:                            # termination: model is done
            return messages

        for call in ai.tool_calls:                       # act (possibly several)
            try:
                observation = tools_by_name[call["name"]].invoke(call["args"])
            except Exception as exc:                     # errors become observations
                observation = f"error: {type(exc).__name__}: {exc}"
            messages.append(ToolMessage(content=str(observation),
                                        tool_call_id=call["id"]))
    raise RuntimeError(f"Agent exceeded {max_steps} steps")


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


model = scripted(
    [("multiply", {"a": 231, "b": 482})],
    [("add", {"a": 111342, "b": 58})],
    "231 x 482 = 111,342 and adding 58 gives 111,400.",
).bind_tools([multiply, add])

history = run_agent(model, [multiply, add], "Compute 231*482 then add 58.")
print(f"{len(history)} messages, {sum(1 for m in history if isinstance(m, ToolMessage))} tool calls")
assert history[-1].content.endswith("111,400.")
```

Read the loop line by line, because every line is a concept:

```text
for step in range(max_steps):    Continue until the model has an answer, but not forever.
ai = model.invoke(messages)      Ask the reasoning engine what should happen next.
if not ai.tool_calls: return     The model has decided it is done. Success termination.
for call in ai.tool_calls:       The model wants the environment to act. Possibly several times.
tool.invoke(call["args"])        OUR program performs the action.
except Exception -> observation  A failed tool is information, not a crash.
ToolMessage(...)                 Tell the model what happened, matched to its request.
raise RuntimeError               Failure termination: the budget ran out.
```

This is the pattern the literature calls **ReAct**: Reason, Act, Observe, repeat. The book introduces it in the reasoning chapter as reasoning interleaved with external actions. You have now implemented it.

### The two bugs the naive loop always has

**Bug 1: it may never stop.** Without `max_steps`, a model that keeps calling `search` runs forever and bills you forever. Production loops need several bounds, not one:

```text
success condition   the model returns no tool calls
failure conditions  max steps, max wall-clock time, max tokens, max dollars
```

Later lessons add time and cost budgets (16. Resource-Aware Optimization) and explicit goal checks (11. Goal Setting and Monitoring).

**Bug 2: tool failures used to crash it.** Our loop already fixes this by converting exceptions into observations. That lets the model recover ("search failed, try the CRM instead"). But it is only the first step. Not every error should be fed back: a timeout might deserve a retry with backoff, a permission error should not be retried at all, and an invalid-argument error should be described precisely so the model can correct itself. 12. Exception Handling and Recovery develops the full taxonomy.

Notice how fast "an agent" turned into ordinary backend engineering: budgets, retries, error classification, idempotency. That is the correct instinct and this course leans into it.

## How LangGraph does it

Our `run_agent` function is a perfectly good agent. So why would anyone use a framework?

The honest answer: for this agent you should not. LangGraph earns its place when the loop grows branches, parallel steps, persistence, human approval, and recovery. But we rebuild the same small agent in LangGraph now so that when those features arrive you already know what the primitives mean.

LangGraph needs one idea we have been using implicitly: **state**. Our loop's state was the `messages` list. In LangGraph you declare the state as a typed dict, and each **node** is a function that reads the state and returns an update. **Edges** say which node runs next. A **conditional edge** is a function that looks at the state and picks the next node. That is the whole model.

```text
STATE  = {"messages": [...]}                     what the agent knows
NODE   = state -> partial state update            one step of work
EDGE   = A -> B                                   fixed transition
COND   = state -> name of next node               routing decision
```

```{code-cell} python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

tools = [multiply, add]
model = scripted(
    [("multiply", {"a": 231, "b": 482})],
    [("add", {"a": 111342, "b": 58})],
    "231 x 482 = 111,342 and adding 58 gives 111,400.",
).bind_tools(tools)


def call_model(state: MessagesState):
    """Node 1: the reasoning step. Same line as in run_agent."""
    return {"messages": [model.invoke(state["messages"])]}


def should_continue(state: MessagesState):
    """Conditional edge: same `if not ai.tool_calls` test as in run_agent."""
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))       # Node 2: the inner for-loop over tool calls
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")               # observations go back to the model
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Compute 231*482 then add 58.")]})
for m in result["messages"]:
    print(f"{type(m).__name__:13s} {m.content or m.tool_calls}")
assert result["messages"][-1].content.endswith("111,400.")
```

Map it back to the from-scratch version:

| In `run_agent` | In LangGraph |
|---|---|
| `messages` list | `MessagesState` (a dict with a `messages` key whose updates *append*) |
| `ai = model.invoke(messages)` | the `agent` node |
| `if not ai.tool_calls` | the `should_continue` conditional edge |
| inner `for call in ai.tool_calls` with try/except and `ToolMessage` | `ToolNode`, which also runs the calls in parallel and turns exceptions into error `ToolMessage`s |
| `for step in range(max_steps)` | the graph's recursion limit (default 25 steps), configurable per run |

`MessagesState` deserves one more sentence. A plain dict update would *replace* the messages list. `MessagesState` declares the key with a reducer that *appends*, which is what a transcript needs. When you define your own state in later lessons you will choose reducers deliberately: append for logs, replace for a plan, sum for a cost counter.

```text
START
  │
  ▼
agent ──(no tool calls)──► END
  │
  │ (tool calls)
  ▼
tools
  │
  └────────────────────────► agent
```

This tiny graph already contains state, tool calling, routing, looping, ReAct, and termination.

### The one-liner, and why we did not start with it

LangChain ships this exact graph as a factory:

```{code-cell} python live
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
agent = create_agent(real_model, tools=[multiply, add])
out = agent.invoke({"messages": [HumanMessage("Compute 231*482 then add 58.")]})
print(out["messages"][-1].content)

# And the from-scratch loop against the real model, unchanged:
history = run_agent(real_model.bind_tools([multiply, add]), [multiply, add],
                    "Compute 231*482 then add 58.")
print(history[-1].content)
```

`create_agent` (and Deep Agents above it) are what you will use day to day. If you had started there, `agent = create_agent(...)` would have worked and you would have learned nothing about tool call ids, reducers, termination, or where the error handling lives. Now you can read its source and recognize every piece.

### Why use LangGraph at all, then?

The while loop is valid, and for a single-tool agent it is preferable. LangGraph becomes valuable when the picture looks like this:

```text
             ┌─ web researcher ─┐
             │                  │
planner ─────┼─ database agent ─┼─ synthesizer ─ reviewer ─ pass / retry
             │                  │
             └─ code executor ──┘
```

Now you need state shared across branches, fan-out and fan-in, checkpoints so a crash at step 30 does not restart at step 1, a pause for human approval that can resume days later, retries per node, streaming of intermediate results, and subgraphs. LangGraph's current docs position it as exactly that: the low-level runtime for long-running, stateful agents with durable execution, human-in-the-loop, and persistence. Each of those becomes a lesson.

### Four terms to keep separate

```text
Model            GPT / Claude / Gemini. Provides the intelligence.
Agent            model + tools + loop + state + decision making.
Agent runtime    LangGraph. Orchestrates execution: state, transitions, checkpoints.
Agent harness    Deep Agents, Claude Code, OpenAI Agents SDK. Opinionated bundle of
                 loop + context management + tools + planning + permissions.
```

A loose but useful analogy: model as CPU, tools as peripherals, state as memory, runtime as the operating system's scheduler, agent as the running process, harness as the whole OS distribution. Never say "LangGraph is the AI." It is not.

## What goes wrong in production

1. **Unbounded loops.** Symptom: a run costs $40 and never finishes. Cause: the only exit is "model returns no tool calls," and the model keeps calling tools. Fix: multiple bounds (steps, time, tokens, cost) enforced by your code, not requested in the prompt. LangGraph's recursion limit is the first of these.

2. **Tool exceptions kill the run.** Symptom: a transient 503 from Slack ends a 20-minute task. Cause: exceptions propagate out of the loop. Fix: convert to observations (as above), then classify. Retry transient errors with exponential backoff and jitter. Never retry permission errors. Return validation errors verbatim so the model can fix its arguments.

3. **Non-idempotent tools plus retries.** Symptom: a customer is refunded twice. Cause: the refund succeeded but the response was lost, so the loop retried. Fix: idempotency keys on every side-effecting tool, and approval gates on consequential actions. This is where agents meet distributed-systems engineering, and 36. Durable Agent Orchestration is about exactly this.

4. **The transcript grows without limit.** Symptom: quality degrades after 40 steps and the context window overflows. Cause: every observation stays in `messages` forever, including 30 KB search results the model already used. Fix: context engineering. Trim or summarize old tool output, keep decisions, and offload large results to files or a store (34. Context Engineering).

5. **Prompt-level rules are treated as enforcement.** Symptom: "never delete production data" in the system prompt, and the agent deletes production data. Cause: a prompt is a suggestion; a tool that exists can be called. Fix: capability boundaries in code. Do not register `delete_table` in production. Validate arguments. Require approval (13. Human in the Loop, 18. Guardrails).

6. **No trace.** Symptom: "it gave a wrong answer" with no way to see which tool returned what. Cause: nothing recorded the message sequence. Fix: the transcript is already a trace; persist it. LangSmith or any tracer records each model call, tool call, and its latency (32. Observability and Tracing).

## What changed by 2026

The loop has not changed. What surrounds it has.

- **From prompt engineering to context engineering.** The question moved from "how do I word the instruction" to "what should the model see at this exact step: which tools, which memory, which prior results, in what form." Anthropic's context-engineering guidance names compaction, structured note-taking, and subagents as the core long-horizon techniques.
- **From loop to harness.** Production systems wrap the loop in a harness that owns context management, planning, tool execution, permissions, sandboxing, checkpoints, and subagents. Deep Agents (built on LangGraph), Claude Code, and the OpenAI Agents SDK are harnesses. Understanding the loop is what lets you evaluate a harness instead of trusting it.
- **From "ReAct prompts" to reasoning models.** Modern reasoning models do their own step-by-step thinking. Explicit "think step by step" scaffolding is no longer the default; clear goals, constraints, and a definition of done are. And a model's written reasoning is not a faithful audit log: verify outcomes with tests, tools, and graders rather than by reading prose.
- **From single-agent scripts to durable execution.** A 45-minute task that crashes at minute 40 must resume, not restart. Checkpointing every step is now a baseline expectation, not an advanced feature.
- **From ad-hoc integrations to protocols.** MCP standardizes how agents discover and call external capabilities; A2A standardizes agent-to-agent collaboration. The loop is unchanged, but where tools come from is now a protocol question.
- **From "looks good" to eval-driven development.** Build, trace, find failures, turn them into eval cases, fix, run regression evals, ship, monitor. Treat the agent as software.

## Interview questions

**Q1. What is an agent loop?**

A strong answer: An agent loop repeatedly gives the model the current task and state, lets the model either produce a final response or request an action through a tool, executes that action in application code, feeds the observation back to the model, and continues until a termination condition is met. Production loops need bounds on steps, time, tokens, and cost, tool argument validation, error classification and retries, tracing, and capability controls. The key property is that the model chooses and the software executes.

Weak: "An agent is an LLM that uses tools."

**Q2. What is the difference between a workflow and an agent, and when would you pick each?**

Strong: In a workflow the developer fixes the control flow; the model fills in steps. In an agent the model chooses at least part of the control flow at runtime. Workflows are cheaper, more predictable, easier to test and debug, so they are the default when the path is known. Agents earn their cost when the path depends on what is discovered along the way, such as open-ended research or debugging. Most production systems are hybrids: deterministic retrieval and validation around a small agentic core.

Weak: "Agents are more advanced than workflows."

**Q3. How do you guarantee an agent loop terminates?**

Strong: You cannot rely on the model to stop, so termination is enforced in code with several independent bounds: a step budget, a wall-clock timeout, a token budget, and a cost budget. On exhaustion you fail explicitly with the partial transcript preserved for debugging. Beyond hard bounds, goal monitoring checks whether progress is being made and stops stalled loops, and reflection steps are capped. In LangGraph the recursion limit is the built-in step bound; the others are yours to add.

Weak: "I tell it in the prompt to stop when done."

**Q4. What happens when a tool times out halfway through an agent loop?**

Strong: First, the exception must not escape the loop; it becomes an observation or triggers a retry. Whether to retry depends on the tool: a read can be retried with backoff and jitter; a side-effecting call must only be retried if it is idempotent, because a timeout does not mean the operation did not happen. That is why side-effecting tools get idempotency keys. If retries are exhausted, the model gets a precise error message so it can choose a fallback, and for long tasks the state is checkpointed so the run can resume later rather than restart.

Weak: "Catch the exception and retry."

**Q5. Why does the model not execute tools itself, and why does that matter?**

Strong: The model only emits a structured request: a tool name and arguments. Application code decides whether to execute it. That boundary is where authorization, argument validation, rate limiting, approval gates, idempotency, and audit logging live. It means safety does not depend on the prompt, and it means the same model can be given different capabilities in different environments simply by registering different tools.

Weak: "The model calls the API."

**Q6. Why use LangGraph instead of a while loop?**

Strong: For a single tool loop, a while loop is better: less to learn, nothing hidden. A graph runtime pays off when the loop needs branches, parallel fan-out with fan-in, per-step checkpoints for crash recovery, pause and resume for human approval, per-node retry policies, streaming, and composition via subgraphs. Those are hard to bolt onto a while loop cleanly. The decision is about the shape of the control flow and the durability requirements, not about whether it is "an agent."

Weak: "LangGraph is the standard framework."

**Q7. What does a good tool schema look like?**

Strong: Narrow purpose, a name that says what it does, a description that says when to use it and when not to, typed arguments with constraints, non-overlapping responsibilities across the tool set, and results that are concise and structured so they do not flood the context. `get_invoice(invoice_id)` is a good tool. `execute_anything(command)` is not, because it forces the model to do the API's job and gives you nothing to validate. Tools should be tested with agent evals, not just unit tests.

Weak: "Give it a docstring."

## Build it yourself

Extend `run_agent` into a bounded, observable loop:

1. Add a `max_seconds` wall-clock budget alongside `max_steps`. Both must be enforced.
2. Add a `budget_tokens` counter that sums `len(str(m.content))` (a stand-in for tokens) over the transcript and stops when exceeded.
3. Return a small trace object with, for each step, the tool names called and the length of each observation.
4. Add a `permission_error` tool that always raises `PermissionError`, and make the loop feed that error back exactly once without retrying.

Acceptance checks:

- A scripted model that emits tool calls forever ends with `RuntimeError` and the trace shows exactly `max_steps` entries.
- Setting `max_seconds=0` stops before the first model call.
- The final message is unchanged for the happy path (`111,400.`).
- The `PermissionError` appears once in a `ToolMessage` and the model is invoked again after it.

## Key takeaways

- An agent is a loop: reason, act through a tool, observe, repeat until a termination condition is met.
- The model chooses; your software executes. All enforcement lives on the software side of that line.
- Workflows fix control flow in code; agents let the model choose part of it. Prefer the simplest that works.
- The transcript of Human, AI, Tool, AI messages is the agent's state, its memory, and its trace.
- Termination must be enforced by code with several bounds. Errors must become observations, then be classified.
- LangGraph adds state, nodes, edges, and conditional edges. For a small loop it is optional; for branching, durable, human-gated workflows it is the point.
- What changed by 2026 is not the loop but the harness around it: context engineering, durable execution, protocols, and evals.
