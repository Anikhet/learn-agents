# 01. Prompt Chaining

*Track A (agent engineering). Book: Chapter 1 (Prompt Chaining). Prerequisite: 00. What an agent actually is.*

Prompt chaining is the first pattern in the book and the least glamorous. It is also the one you will ship most often, because most "agent" features in production are chains: a fixed sequence of model calls where each call does one job and hands a small, checked artifact to the next. This lesson builds a chain with nothing hidden, shows what breaks when the artifacts between steps are untyped prose, rebuilds it as a LangGraph node-to-node-to-node graph, and then asks the harder question: when is one big prompt actually better?

## The problem

The support agent receives a Slack message:

> "Acme says their September invoice is wrong. Can someone look at it and draft a reply? Also loop in their account owner."

The naive approach is one prompt:

```text
System: You are a support agent. Read the message, figure out the customer,
        determine what is wrong with the invoice, draft a reply, and identify
        the account owner to CC.
User:   <slack message>
```

One call. It often works on the demo message. Then it meets reality:

```text
Slack message
     ↓
one giant prompt ──► "Hi Acme team, we've corrected your invoice..."
                          ▲
                          │ which invoice? corrected how? who is CC'd?
                          │ nothing in the output is checkable
```

Three things go wrong, and they compound.

1. **You cannot validate the middle.** The model decided the customer was "Acme", picked an invoice, decided what was wrong, and wrote a reply, all inside one generation. If the customer identification was wrong, the reply is confidently wrong about everything downstream, and you have no hook to catch it.
2. **The prompt does too many jobs.** Instructions for extraction, diagnosis, tone, and CC policy all compete for attention. Add a fourth requirement ("always mention the SLA") and the tone quality drops. This is the attention-dilution problem the book describes: one prompt, many constraints, worse at each.
3. **You cannot test or improve one part.** When the reply is too formal you edit the mega-prompt, and the extraction accuracy moves too. Nothing is isolated.

A chain fixes all three by making the intermediate results explicit:

```text
Slack message
     ↓
[1] extract      → {"customer": "Acme", "period": "2026-09", "issue": "amount"}
     ↓ validate: customer non-empty, period is YYYY-MM
[2] diagnose     → {"invoice_id": "in_9F2", "finding": "duplicate seat charge", "delta_usd": 480}
     ↓ validate: delta is a number, finding non-empty
[3] draft reply  → "Hi Acme team, we found a duplicate seat charge of $480 on in_9F2..."
     ↓ validate: mentions invoice_id, under 120 words
Reply + CC list
```

Every arrow is now a place where your software can look at the data and say no. That is the whole idea. The book calls this decomposition and the "divide and conquer" of prompts; the engineering framing is: **a chain is a pipeline with typed stages, and the model is the worker at each stage.**

Notice that this is a **workflow**, not an agent. The developer fixed the path. The model fills in each step. Anthropic's "Building effective agents" guidance lists prompt chaining first among the workflow patterns for exactly that reason: it is the simplest thing that works when the steps are known up front. Say that in an interview and you have already separated yourself from the candidates who reach for an autonomous loop.

## Build it without a framework

We use the scripted model from STYLE.md so the control flow, not the model's judgment, is the object of study. Each scripted turn stands in for what a real model would return; a `live` cell at the end runs the same chain against a real model.

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

### Typed artifacts between steps

The intermediate outputs are the design. If step 1 emits free text ("Looks like Acme's September invoice has an amount problem"), step 2 has to re-parse it and step 2's prompt becomes a second extraction job. If step 1 emits a schema, step 2 receives exactly the fields it needs and nothing else.

We declare each artifact as a Pydantic model. Pydantic gives us the validation gate for free: if the model's JSON does not fit, we know at the boundary, not three steps later.

```{code-cell} python
import json
import re
from pydantic import BaseModel, Field, ValidationError, field_validator


class Extraction(BaseModel):
    customer: str = Field(min_length=1)
    period: str
    issue: str = Field(min_length=1)

    @field_validator("period")
    @classmethod
    def period_is_year_month(cls, value):
        if not re.fullmatch(r"\d{4}-\d{2}", value):
            raise ValueError("period must be YYYY-MM")
        return value


class Diagnosis(BaseModel):
    invoice_id: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    delta_usd: float


class Reply(BaseModel):
    body: str
    cc: list[str]


def parse_as(schema, text):
    """The validation gate: JSON out of the model, a typed object or an error in."""
    try:
        return schema.model_validate(json.loads(text)), None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


good, err = parse_as(Extraction, '{"customer": "Acme", "period": "2026-09", "issue": "amount"}')
bad, err2 = parse_as(Extraction, '{"customer": "Acme", "period": "September", "issue": "amount"}')
print("valid:", good)
print("invalid:", err2.splitlines()[0])
assert good is not None and err is None
assert bad is None and "YYYY-MM" in err2
```

### The chain, with a gate at every arrow

A step is: build a narrow prompt from the previous artifact, call the model, validate, and if validation fails, call once more with the error attached. The retry is bounded. A real model that fails twice on a schema is telling you the prompt or the schema is wrong, and the right response is to stop and surface that, not loop.

```{code-cell} python
from langchain_core.messages import HumanMessage, SystemMessage

STEP_PROMPTS = {
    "extract": "Extract customer, billing period (YYYY-MM) and the issue type from the message. Return JSON.",
    "diagnose": "Given the extraction and the invoice lines, identify the invoice id, the finding and the dollar delta. Return JSON.",
    "draft": "Write a reply under 120 words that names the invoice id and the finding. Return JSON with body and cc.",
}


def run_step(model, name, schema, payload, trace, max_attempts=2):
    """One stage: narrow prompt in, validated artifact out, bounded retry on bad output."""
    messages = [SystemMessage(STEP_PROMPTS[name]), HumanMessage(json.dumps(payload))]
    for attempt in range(1, max_attempts + 1):
        raw = model.invoke(messages).content
        artifact, error = parse_as(schema, raw)
        trace.append({"step": name, "attempt": attempt, "ok": error is None})
        if artifact is not None:
            return artifact
        # Feed the validation error back. The model sees exactly what was wrong.
        messages += [AIMessage(raw), HumanMessage(f"Invalid output. {error}. Return corrected JSON only.")]
    raise RuntimeError(f"step {name!r} failed validation {max_attempts} times")


def lookup_invoice_lines(customer, period):
    """Deterministic tool standing in for Stripe. Chains mix model steps and plain code steps."""
    return [{"id": "in_9F2", "desc": "Seats x 12", "usd": 2880}, {"id": "in_9F2", "desc": "Seats x 2", "usd": 480}]


def support_chain(model, slack_text):
    trace = []
    extraction = run_step(model, "extract", Extraction, {"message": slack_text}, trace)
    lines = lookup_invoice_lines(extraction.customer, extraction.period)   # code step, no model
    diagnosis = run_step(model, "diagnose", Diagnosis,
                         {"extraction": extraction.model_dump(), "lines": lines}, trace)
    reply = run_step(model, "draft", Reply,
                     {"diagnosis": diagnosis.model_dump(), "customer": extraction.customer}, trace)
    # A final code-side check that the reply is grounded in the diagnosis.
    assert diagnosis.invoice_id in reply.body, "reply must cite the invoice id"
    return reply, trace


# Script the model to fail step 1 once (period in the wrong format), then recover.
model = scripted(
    '{"customer": "Acme", "period": "September 2026", "issue": "amount"}',
    '{"customer": "Acme", "period": "2026-09", "issue": "amount"}',
    '{"invoice_id": "in_9F2", "finding": "duplicate seat charge", "delta_usd": 480}',
    '{"body": "Hi Acme team, we found a duplicate seat charge of $480 on in_9F2 and will credit it.", "cc": ["owner@ourco.com"]}',
)
reply, trace = support_chain(model, "Acme says their September invoice is wrong. Loop in their account owner.")
for t in trace:
    print(t)
print(reply.body)
assert [t["ok"] for t in trace] == [False, True, True, True]
assert reply.cc == ["owner@ourco.com"]
```

Read the trace. Step 1 failed validation on the first attempt because the period came back as prose, the error went back to the model, and the second attempt passed. Steps 2 and 3 never saw the bad output. That containment is the point of the gate: **an error is caught where it is cheap, at the boundary that produced it, before downstream steps amplify it.**

Two more things to notice in that code, because interviewers ask about both:

- **Not every step is a model call.** `lookup_invoice_lines` is plain code. A chain interleaves model steps with deterministic steps freely. Anything that can be code should be code.
- **Each prompt sees only what it needs.** The draft step receives the diagnosis and the customer name, not the original Slack message and not the raw invoice lines. That is deliberate context control. Passing everything forward recreates the mega-prompt one stage at a time.

## How LangGraph does it

The from-scratch version is a function that calls three functions. Why would a graph help?

For a fixed three-step chain, it barely does. LangGraph earns its place when the chain grows a retry loop that needs its own state, when you want every step checkpointed so a crash at step 3 resumes at step 3, or when the chain becomes one component inside a larger graph. We build it now so that the primitives are familiar before those needs arrive.

The mapping is direct:

```text
from scratch                       LangGraph
------------                       ---------
local variables passed forward     a typed State dict shared by every node
run_step(...) for each stage       one node per stage
parse_as(...) validation gate      a node that writes `error`, plus a conditional edge
for attempt in range(...)          a loop edge back to the stage, bounded by a counter in state
support_chain() calling in order   START -> extract -> validate -> diagnose -> draft -> END
```

```{code-cell} python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class ChainState(TypedDict, total=False):
    slack_text: str
    extraction: dict          # replaced, not appended: a chain has one current artifact per stage
    diagnosis: dict
    reply: dict
    error: str
    extract_attempts: int


model = scripted(
    '{"customer": "Acme", "period": "September 2026", "issue": "amount"}',
    '{"customer": "Acme", "period": "2026-09", "issue": "amount"}',
    '{"invoice_id": "in_9F2", "finding": "duplicate seat charge", "delta_usd": 480}',
    '{"body": "Hi Acme team, we found a duplicate seat charge of $480 on in_9F2 and will credit it.", "cc": ["owner@ourco.com"]}',
)


def extract(state: ChainState):
    prompt = [SystemMessage(STEP_PROMPTS["extract"]), HumanMessage(state["slack_text"])]
    if state.get("error"):
        prompt.append(HumanMessage(f"Previous attempt was invalid. {state['error']}"))
    raw = model.invoke(prompt).content
    artifact, error = parse_as(Extraction, raw)
    return {"extraction": artifact.model_dump() if artifact else None,
            "error": error or "",
            "extract_attempts": state.get("extract_attempts", 0) + 1}


def gate_after_extract(state: ChainState):
    """Conditional edge = the `if artifact is not None` check from run_step."""
    if not state["error"]:
        return "diagnose"
    if state["extract_attempts"] >= 2:
        return END                      # bounded: surface the failure, do not loop forever
    return "extract"


def diagnose(state: ChainState):
    lines = lookup_invoice_lines(state["extraction"]["customer"], state["extraction"]["period"])
    raw = model.invoke([SystemMessage(STEP_PROMPTS["diagnose"]),
                        HumanMessage(json.dumps({"extraction": state["extraction"], "lines": lines}))]).content
    artifact, error = parse_as(Diagnosis, raw)
    return {"diagnosis": artifact.model_dump(), "error": error or ""}


def draft(state: ChainState):
    raw = model.invoke([SystemMessage(STEP_PROMPTS["draft"]),
                        HumanMessage(json.dumps({"diagnosis": state["diagnosis"],
                                                 "customer": state["extraction"]["customer"]}))]).content
    artifact, error = parse_as(Reply, raw)
    return {"reply": artifact.model_dump(), "error": error or ""}


builder = StateGraph(ChainState)
builder.add_node("extract", extract)
builder.add_node("diagnose", diagnose)
builder.add_node("draft", draft)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", gate_after_extract, ["extract", "diagnose", END])
builder.add_edge("diagnose", "draft")
builder.add_edge("draft", END)
graph = builder.compile()

result = graph.invoke({"slack_text": "Acme says their September invoice is wrong."})
print("extract attempts:", result["extract_attempts"])
print(result["reply"]["body"])
assert result["extract_attempts"] == 2
assert result["diagnosis"]["invoice_id"] in result["reply"]["body"]
```

```text
START
  │
  ▼
extract ──(invalid, attempts < 2)──┐
  │  ▲                              │
  │  └──────────────────────────────┘
  │ (valid)                 (invalid, attempts == 2) ──► END with error set
  ▼
diagnose
  │
  ▼
draft
  │
  ▼
END
```

What the primitives are doing underneath:

- **`ChainState`** is the set of local variables from `support_chain`, made explicit. Because each key has no reducer, an update *replaces* the value. That is right for a chain: there is one current extraction, not a history of them. Compare `MessagesState` in lesson 00, where the reducer appends.
- **Each node** is `run_step` without its retry loop. The loop moved into the graph.
- **The conditional edge** is the `if artifact is not None` test. The bound (`extract_attempts >= 2`) is the `for attempt in range(2)`.
- **Edges `diagnose -> draft -> END`** are the sequential calls in `support_chain`.

Once the chain lives in a graph you get things the function did not have: attach `InMemorySaver` (or a database checkpointer) and every node boundary is a checkpoint, so a crash in `draft` resumes at `draft` with the diagnosis intact (36. Durable Agent Orchestration). Add `RetryPolicy` on the `diagnose` node and transient Stripe failures retry without touching your code. Stream `graph.stream(...)` and the UI shows each stage as it completes.

### With a real model, the plumbing simplifies

With a real provider you would not parse JSON by hand. `model.with_structured_output(Extraction)` asks the provider to constrain the output to the schema, which the serving stack implements as constrained decoding (see 27. Structured Output). That guarantees the *syntax*. Your validators still guarantee the *semantics*: a well-formed `period` can still be the wrong month.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))

extraction = real.with_structured_output(Extraction).invoke(
    [SystemMessage(STEP_PROMPTS["extract"]),
     HumanMessage("Acme says their September 2026 invoice is wrong. Loop in their account owner.")])
lines = lookup_invoice_lines(extraction.customer, extraction.period)
diagnosis = real.with_structured_output(Diagnosis).invoke(
    [SystemMessage(STEP_PROMPTS["diagnose"]),
     HumanMessage(json.dumps({"extraction": extraction.model_dump(), "lines": lines}))])
reply = real.with_structured_output(Reply).invoke(
    [SystemMessage(STEP_PROMPTS["draft"]),
     HumanMessage(json.dumps({"diagnosis": diagnosis.model_dump(), "customer": extraction.customer}))])
print(extraction, diagnosis, reply.body, sep="\n")
```

### Where chaining beats one prompt, and where it does not

This is the judgment call interviewers probe. The honest answer has two sides.

**Chain when:**

| Situation | Why a chain wins |
|---|---|
| Steps need different tools or data | Step 2 needs invoice lines that only exist after step 1 identifies the customer |
| A step's output must be checked before it is acted on | You can validate a schema; you cannot validate the middle of a paragraph |
| Steps have different quality requirements | Extraction wants a cheap fast model; drafting wants a strong one |
| You need to test and tune parts independently | Fix the extraction prompt without moving the tone of the reply |
| Failures must be localized for debugging | The trace says "step 2, attempt 1, invalid" instead of "the answer was wrong" |

**Do not chain when:**

| Situation | Why one prompt wins |
|---|---|
| The steps are not separable | "Summarize this in the customer's tone" is one act of judgment, not two |
| Latency budget is tight | Each stage is a full round trip; three stages triple time-to-first-token |
| The intermediate artifact loses information the last step needs | Compressing a nuanced thread into five fields, then drafting from the fields, produces a generic reply |
| The model is capable and the task is small | A 2026 frontier model does "extract, diagnose, draft" in one call reliably for simple cases |

The last row deserves a sentence. Chains were partly a workaround for weaker models. As models improved, some chains collapsed back into single calls, and Anthropic's harness guidance explicitly recommends removing scaffolding as models improve. The reason to keep a chain in 2026 is not "the model cannot do it in one shot." It is "I need to check, log, route, or approve the middle." If none of those apply, one call is simpler and faster, and simpler is the default.

## What goes wrong in production

1. **Error amplification through unvalidated stages.** Symptom: a reply confidently references the wrong customer. Cause: step 1 mis-extracted, step 2 and 3 trusted it. Fix: a validation gate after each stage, with checks that go beyond schema where possible (does the customer exist in the CRM?). Fail the chain early with a clear error rather than finish it wrong.

2. **Context re-inflation.** Symptom: the last step's prompt is larger than the original message. Cause: each stage appended its full input and output and passed everything forward "to be safe." Fix: each stage receives only the fields it needs. If a later stage genuinely needs the original text, pass a reference and fetch it, not a copy of everything.

3. **Latency stacking.** Symptom: p95 went from 4 s to 14 s after "improving" the pipeline. Cause: three sequential round trips plus a retry. Fix: run independent stages in parallel (03. Parallelization), use a small fast model for cheap stages, stream the final stage, and ask whether each stage earns its round trip.

4. **Retry loops on schema failures.** Symptom: a stage retries five times and burns budget on every request. Cause: an unbounded or high retry count hiding a bad prompt or an impossible schema. Fix: bound retries at 1 or 2, log the raw output, and treat repeated failures as a prompt bug to fix, not a runtime condition to absorb.

5. **Partial completion with side effects.** Symptom: the CRM note was written but the reply was never sent, and a re-run writes the note again. Cause: a side-effecting step in the middle of a chain with no checkpointing and no idempotency. Fix: checkpoint after each node so re-runs resume rather than restart; give side-effecting steps idempotency keys; push side effects to the end of the chain where possible.

6. **Silent schema drift.** Symptom: step 3 starts producing empty CC lists after a model upgrade. Cause: the new model formats the artifact differently and the loose parser accepted it. Fix: strict Pydantic models, asserts on invariants, and a regression eval that runs the whole chain on saved inputs before any model change ships (22. Evaluation Reliability Deep Dive).

## What changed by 2026

- **Structured output became the default seam.** Provider-side structured output and strict tool schemas mean the "parse the model's JSON" step is mostly gone. The validation gate moved from syntax (now guaranteed by constrained decoding) to semantics (still yours).
- **Chains got absorbed into harnesses as workflows, not agents.** Anthropic's guidance and the LangChain and LangGraph docs both frame prompt chaining as a workflow pattern: fixed code path, model at each step. The advice to prefer the simplest architecture that works has hardened as models improved, and chains are often the simplest.
- **Some chains collapsed.** Frontier models handle multi-part tasks in a single call well enough that the case for chaining is now about *control* (validation, logging, approval, per-stage model choice), not capability. Anthropic's harness-design guidance says to remove harness components as models improve; a chain stage you cannot justify by a check it enables is a candidate.
- **Durable execution made long chains safe.** Checkpointing at node boundaries is a baseline in LangGraph 1.x; a chain that fails at stage 6 resumes at stage 6. This changed what people are willing to chain: multi-hour document pipelines that were previously fragile.
- **Per-stage model routing is normal.** Cheap models for extraction and classification, strong models for synthesis, with the choice made per node. This is cost engineering that the single-prompt design cannot express (16. Resource-Aware Optimization).
- **Eval-driven iteration replaced prompt fiddling.** Each stage gets its own eval set, so you can see that a prompt change improved extraction F1 and left draft quality unchanged. The chain's structure is what makes stage-level evals possible.

## Interview questions

**Q1. What is prompt chaining and when would you use it over a single prompt?**

Strong: Prompt chaining decomposes a task into a fixed sequence of model calls where each call has a narrow job and hands a structured artifact to the next. I use it when the intermediate result needs to be checked, logged, or acted on before continuing, when different stages need different tools, data, or models, or when I need to test and tune stages independently. I do not chain when the steps are not really separable, when latency matters more than inspectability, or when a capable model does the whole thing reliably in one call and nothing in the middle needs a check. It is a workflow, not an agent: the developer fixes the path.

Weak: "Break the task into smaller prompts so the model does better."

**Q2. How do you stop errors propagating through a chain?**

Strong: Validate at every boundary. Each stage emits a typed artifact, a Pydantic model or a strict schema, and the next stage does not run until it validates. Schema validation catches malformed output; semantic checks catch well-formed but wrong output, like a customer id that does not exist in the CRM. On failure I feed the specific error back to the model once or twice, bounded, and if it still fails I stop the chain and surface it, because repeated schema failures mean the prompt or schema is wrong. The trace records which stage and attempt failed so debugging is local.

Weak: "Add error handling."

**Q3. What should the intermediate artifacts in a chain look like?**

Strong: Small, typed, and containing only what the next stage needs. A schema rather than prose, because prose forces the next stage to re-parse and turns every stage into an extraction task. I pass references rather than copies for large data; if the draft stage needs the original thread, it gets an id and fetches it. Each stage's context should be smaller than the mega-prompt it replaces, otherwise the chain has recreated the original problem one step at a time.

Weak: "Pass the output of one prompt into the next."

**Q4. How does a chain map onto LangGraph, and what does the graph give you that a function does not?**

Strong: Each stage is a node, the shared typed state holds the artifacts, sequential edges connect the stages, and a conditional edge after a validation step implements the bounded retry. For a three-step chain a plain function is fine. The graph pays off when you want checkpointing at node boundaries so a crash resumes at the failed stage, per-node retry policies for transient tool failures, streaming of stage results to a UI, and composition, so the chain becomes a subgraph inside a larger agent. The state's keys use replace semantics, unlike message lists, because a chain has one current artifact per stage.

Weak: "LangGraph lets you draw the chain as a graph."

**Q5. A product manager wants to add a fourth requirement to the reply, and quality of the other three dropped. What happened and what do you do?**

Strong: That is instruction dilution: one prompt carrying too many competing constraints degrades at each of them. If the requirements are separable I split the stage, for example a draft stage and a compliance-check stage that verifies the SLA mention and the CC policy, each with a narrow prompt and its own eval. If they are not separable I keep one stage but move the checkable parts into code, like asserting the invoice id appears in the body. Either way I add the new requirement as an eval case first so I can see whether the change improved it without regressing the others.

Weak: "Make the prompt more detailed."

**Q6. When would you collapse a chain back into a single call?**

Strong: When none of the intermediate boundaries is doing work: no validation, no branching, no approval, no per-stage model choice, no logging anyone reads. Then the chain is paying two extra round trips for nothing. Capable models handle combined extract-diagnose-draft tasks in one call for simple cases, and Anthropic's harness guidance says to remove scaffolding as models improve. I would run the eval set against the single-call version and keep the chain only if a boundary check is catching real failures.

Weak: "Never, chains are always more reliable."

**Q7. How do you handle a chain where step 2 has side effects and step 3 fails?**

Strong: Treat it like any partially-failed distributed transaction. First, checkpoint after every stage so a retry resumes at step 3 rather than re-running step 2. Second, make side-effecting steps idempotent with a key derived from the input, so even an accidental re-run is safe. Third, where possible reorder the chain so side effects are last, after everything that can fail has succeeded. Fourth, log the partial state so a human can reconcile if all else fails. These are the same rules as any pipeline with external writes.

Weak: "Wrap it in a try/except and retry the whole chain."

## Build it yourself

Extend the LangGraph chain:

1. Add a `verify` node after `draft` that checks the reply is under 120 words and cites `diagnosis["invoice_id"]`, writing `error` if not, and a conditional edge that routes back to `draft` at most once.
2. Add a `cost` key to the state that sums `len(prompt) + len(output)` characters per node (a stand-in for tokens), and print the per-stage breakdown at the end.
3. Add `InMemorySaver` as a checkpointer and run with a `thread_id`. Script the model so `draft` returns invalid JSON on the first attempt, then inspect `graph.get_state(config)` to show that `diagnosis` survived the failure.
4. Replace `gate_after_extract`'s hard-coded bound with a `max_attempts` value read from `config["configurable"]`.

Acceptance checks:

- A reply of 150 words causes exactly one extra `draft` invocation and the final reply passes.
- `result["cost"]` equals the sum of the per-stage entries and the `draft` stage is the largest.
- After the scripted `draft` failure, `graph.get_state(config).values["diagnosis"]["invoice_id"] == "in_9F2"`.
- Setting `max_attempts=1` and a permanently-invalid extraction ends at `END` with `error` set and `diagnosis` absent.

## Key takeaways

- A chain is a pipeline with typed stages; the model is the worker at each stage and the developer fixes the path. It is a workflow, not an agent.
- The intermediate artifacts are the design: small, typed, containing only what the next stage needs.
- Validate at every boundary so errors are caught where they are cheap, before downstream stages amplify them; bound the retries.
- Mix model steps with plain code steps freely; anything that can be code should be code.
- In LangGraph, stages are nodes, artifacts are replace-semantics state keys, validation is a conditional edge, and the graph adds checkpointing, per-node retries, and streaming.
- Chain for control (validation, logging, approval, per-stage models), not for capability; collapse stages that no check justifies.
- Side-effecting stages need idempotency and checkpoints, exactly as in any distributed pipeline.
