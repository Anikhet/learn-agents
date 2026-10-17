# 24. Structured Output: schemas, validation, and model-agnostic design

*Track A (agent engineering) with Track C production concerns. Course addition; extends the book's Chapter 5 (Tool Use) and Appendix A (Advanced Prompting). Prerequisites: 05. Tool Use and Function Calling, 12. Exception Handling and Recovery. Pairs with 44. Decoding, reasoning models, and structured output, which covers the constrained-decoding mechanics this lesson builds on.*

## The problem

The support agent has finished investigating. It found that Acme was charged twice for invoice `in_1042`. Now it must hand the result to software: create a CRM ticket, and propose a Stripe refund that a human will approve.

Software does not read prose. The CRM needs a `category` from a fixed list, an `urgency` from 1 to 5, and a boolean `needs_human`. The refund proposal needs an `invoice_id` that exists and an `amount_cents` that does not exceed what was charged.

The naive approach is to ask nicely and parse:

```python
reply = model.invoke("Classify this ticket. Respond in JSON with category, urgency, needs_human.")
ticket = json.loads(reply.content)
crm.create_ticket(**ticket)
```

Every line of that breaks in production, in a different way:

```text
model output                                   what happens
------------------------------------------------------------------------------
(markdown fence) {"category": "billing"...}    json.loads fails on the fence
Sure! Here is the JSON: {...}                    json.loads fails on the prose
{"category": "refund", "urgency": 9}            parses; CRM rejects bad enum, range
{"category": "billing", "urgency": 4            truncated at max_tokens; no closing brace
{"category": "billing", "urgency": 4,           parses, validates, and is WRONG:
 "needs_human": false}                            a double charge must go to a human
I can't help with that.                          a refusal, not a parse error
```

These are five different failures with five different fixes. Four of them are about *shape* and one is about *truth*. The core lesson is that you need a separate layer for each:

```text
                    ┌─────────────────────────────────────────────┐
  model  ──tokens──►│ 1. SYNTAX    is it parseable JSON?          │  constrained decoding
                    │ 2. SCHEMA    right fields, types, enums?    │  strict mode / pydantic
                    │ 3. SEMANTICS consistent with the world?     │  your validator + source data
                    │ 4. POLICY    allowed to act on it?          │  guardrails, approval
                    └─────────────────────────────────────────────┘
                                         │
                                         ▼
                              CRM / Stripe / SQL write
```

Provider features can guarantee layers 1 and 2. Nothing a provider ships can guarantee layer 3. That is the sentence interviewers are listening for.

There is a second problem hiding here. Your code talks to OpenAI today. Next quarter, procurement wants a second provider for failover, and the finance team wants a cheaper model for classification. If `response.choices[0].message.parsed` is sprinkled through your business logic, you cannot swap anything. So this lesson is also about keeping the provider on one side of an adapter and your domain objects on the other.

### Three ways to get structure out of a model

```text
Prompt-and-parse     "Reply in JSON" + json.loads + validate + retry
                     works everywhere; costs a round trip per failure

Tool/function call   declare a tool whose arguments ARE the schema; force the call
                     works on any tool-calling model; the "tool" is never executed

Native / strict      provider compiles the schema into a grammar and masks tokens
                     guarantees syntax + schema; only on supporting providers/models
```

In 44. Decoding you built the third one: a state machine that sets every illegal token's logit to negative infinity. Here we treat it as a black box and ask the engineering questions: when to use which, what still goes wrong, and how to write code that does not care which one is underneath.

## Build it without a framework

The scripted model replays fixed turns, so the control flow of validation and repair is what we study. A live cell later runs the same schema against a real model.

### The schema is the contract

A pydantic model is three things at once: the JSON Schema you send to the provider, the validator you run on the result, and the typed object your business code receives. One definition, three uses. That is why it is the standard choice in Python.

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


import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Triage(BaseModel):
    """Triage decision for one support ticket."""
    model_config = ConfigDict(extra="forbid")  # emits additionalProperties: false

    # Evidence first: the model generates left to right, so the fields it writes
    # early condition the fields it writes later.
    evidence: str = Field(description="One sentence citing what was observed.")
    category: Literal["billing", "technical", "account", "general"]
    urgency: int = Field(ge=1, le=5, description="1 = can wait a week, 5 = outage or money lost")
    needs_human: bool = Field(description="True for refunds, legal, security, or angry VIPs.")


def strict_mode_problems(schema: dict) -> list[str]:
    """Checks the two rules strict schema modes commonly impose on objects."""
    problems = []
    props = schema.get("properties", {})
    missing = sorted(set(props) - set(schema.get("required", [])))
    if missing:
        problems.append(f"optional fields not allowed, make them required/nullable: {missing}")
    if schema.get("additionalProperties") is not False:
        problems.append("additionalProperties must be false")
    return problems


class LooseTriage(BaseModel):
    category: str
    urgency: int = 3            # a default makes the field optional in the schema
    notes: str | None = None


schema = Triage.model_json_schema()
print(json.dumps(schema["properties"]["category"]))
print("Triage problems:", strict_mode_problems(schema))
print("LooseTriage problems:", strict_mode_problems(LooseTriage.model_json_schema()))
assert strict_mode_problems(schema) == []
assert len(strict_mode_problems(LooseTriage.model_json_schema())) == 2
```

Three design decisions are packed into that schema.

**Enums, not free strings.** `category: str` invites `"Billing"`, `"billing issue"`, and `"refund"`. A `Literal` becomes a JSON Schema `enum`, which a strict provider enforces at decode time and pydantic enforces after.

**Descriptions are prompt.** The provider sends field descriptions to the model. "1 = can wait a week, 5 = outage or money lost" is a rubric. Without it, urgency is a mood.

**Field order matters.** Putting `evidence` before `category` gives the model a place to ground itself before committing to a label. This is not chain-of-thought for auditing (a model's stated reasoning is not a faithful log of why it chose; see 17. Reasoning Techniques). It is ordering for quality, and you should measure whether it helps on your eval set.

The `strict_mode_problems` check encodes the two object rules that OpenAI's strict mode documents and that other providers' strict modes resemble: every property listed in `required`, and no extra properties. Optional fields become required-but-nullable (`str | None` with no default). Current provider docs also list JSON Schema keywords that each strict mode does not support, and those lists differ by provider. Run a check like this in CI so a schema change fails a test rather than a production request.

### Parse, validate, repair: the loop that works everywhere

When the provider has no strict mode, or the model does not support it, or you are calling a self-hosted model without a grammar backend, you fall back to the oldest pattern: parse, validate, and on failure feed the *exact* error back and ask again.

The important engineering is classification. Not every failure deserves the same response:

```text
failure            detect                          response
--------------------------------------------------------------------------
fenced / prose     json.loads fails, but {...}     extract, no retry needed
                   found inside
bad JSON           no parseable object             retry with parse error
schema violation   pydantic ValidationError        retry with field-level errors
truncated          finish_reason == "length"       do NOT retry as-is; raise max_tokens
                                                   or shrink the schema
refusal            refusal text / refusal field    do NOT retry; route to a human
```

```{code-cell} python
import re
from langchain_core.messages import HumanMessage
from pydantic import ValidationError


class StructuredOutputFailed(Exception):
    def __init__(self, kind, detail, attempts):
        super().__init__(f"{kind} after {attempts} attempt(s): {detail}")
        self.kind, self.attempts = kind, attempts


def extract_json(text: str):
    """Tolerate fences and chatter: take the outermost {...} if there is one."""
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else json.loads(text)


def structured_call(model, schema, prompt, max_attempts=3, log=None):
    log = log if log is not None else []
    messages = [HumanMessage(
        f"{prompt}\n\nReply with only a JSON object matching this schema:\n"
        f"{json.dumps(schema.model_json_schema())}")]
    for attempt in range(1, max_attempts + 1):
        ai = model.invoke(messages)
        messages.append(ai)
        if ai.response_metadata.get("finish_reason") == "length":
            raise StructuredOutputFailed("truncated", "raise max_tokens", attempt)
        if ai.content.lower().startswith(("i can't", "i cannot")):
            raise StructuredOutputFailed("refusal", ai.content, attempt)
        try:
            obj = schema.model_validate(extract_json(ai.content))
            log.append(("ok", attempt))
            return obj, log
        except json.JSONDecodeError as exc:
            error = f"Your reply was not valid JSON ({exc.msg}). Reply with only the JSON object."
            log.append(("parse_error", attempt))
        except ValidationError as exc:
            # Field-level errors are the most useful repair signal a model can get.
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            error = f"Your JSON failed validation: {details}. Fix only these fields."
            log.append(("schema_error", attempt))
        messages.append(HumanMessage(error))
    raise StructuredOutputFailed("exhausted", error, max_attempts)


model = scripted(
    'Sure! Here you go: {"evidence": "Two charges for in_1042", "category": "refund", "urgency": 9, "needs_human": true}',
    '```json\n{"evidence": "Two charges for in_1042", "category": "billing", "urgency": 5, "needs_human": true}\n```',
)
ticket, log = structured_call(model, Triage, "Acme says they were charged twice for invoice in_1042.")
print(ticket)
print(log)
assert ticket.category == "billing" and ticket.urgency == 5
assert log == [("schema_error", 1), ("ok", 2)]

refusing = scripted("I can't help with that request.")
try:
    structured_call(refusing, Triage, "...")
except StructuredOutputFailed as exc:
    print("stopped:", exc)
    assert exc.kind == "refusal" and exc.attempts == 1
```

Read the first attempt. The JSON was wrapped in chatter, which `extract_json` quietly tolerated. The schema check then failed on two fields, and the repair message named both: `category: Input should be 'billing', ...` and `urgency: Input should be less than or equal to 5`. The second attempt came back in a markdown fence, which was also tolerated. Two attempts, one round trip of repair.

Cost is the hidden variable. Each retry resends the whole conversation. A 3-attempt cap on a 4,000-token prompt can triple the bill for the unlucky requests. Log the attempt count per call; a rising average is an early signal that a prompt or model change broke something.

### Semantic validation needs the world

The schema says `amount_cents` is a positive integer. It cannot say "no larger than what Stripe actually charged." That needs data the model did not produce. Pydantic lets you pass that data in as validation *context*, so the check lives next to the schema instead of scattered through business code.

```{code-cell} python
from pydantic import ValidationInfo, field_validator, model_validator


class RefundProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_id: str = Field(pattern=r"^in_\d+$")
    amount_cents: int = Field(gt=0)
    reason: Literal["duplicate", "fraudulent", "requested_by_customer"]

    @field_validator("invoice_id")
    @classmethod
    def invoice_exists(cls, v, info: ValidationInfo):
        if info.context is None:
            return v  # schema-only validation, no world knowledge available
        invoices = info.context.get("invoices", {})
        if v not in invoices:
            raise ValueError(f"unknown invoice {v}; known: {sorted(invoices)}")
        return v

    @model_validator(mode="after")
    def not_more_than_charged(self, info: ValidationInfo):
        charged = (info.context or {}).get("invoices", {}).get(self.invoice_id)
        if charged is not None and self.amount_cents > charged:
            raise ValueError(f"amount {self.amount_cents} exceeds charged {charged}")
        return self


# Source of truth, fetched by YOUR code from Stripe, not by the model.
stripe_view = {"invoices": {"in_1042": 12_000, "in_1043": 4_500}}

candidates = [
    {"invoice_id": "in_1042", "amount_cents": 12_000, "reason": "duplicate"},
    {"invoice_id": "in_1042", "amount_cents": 120_000, "reason": "duplicate"},   # 10x slip
    {"invoice_id": "in_9999", "amount_cents": 100, "reason": "duplicate"},       # invented id
]
verdicts = []
for c in candidates:
    try:
        RefundProposal.model_validate(c, context=stripe_view)
        verdicts.append("valid")
    except ValidationError as exc:
        verdicts.append(exc.errors()[0]["msg"])
for c, v in zip(candidates, verdicts):
    print(f"{c['invoice_id']} {c['amount_cents']:>7} -> {v}")

assert verdicts[0] == "valid"
assert "exceeds charged" in verdicts[1] and "unknown invoice" in verdicts[2]
# Without the Stripe context (schema-only, like a strict provider) all three pass.
assert all(RefundProposal.model_validate(c) for c in candidates)
```

All three candidates are syntactically perfect. A strict-mode provider would emit any of them. Two would move the wrong money. That is layer 3 in the diagram, and it is always your code's job. The error messages are also written for the model: if this validator sits inside the repair loop, "known: ['in_1042', 'in_1043']" gives the model what it needs to fix the id instead of guessing again.

## How LangGraph does it

LangChain exposes structured output in two places. Both are the machinery you just wrote, with the provider details hidden.

**`model.with_structured_output(Schema)`** returns a runnable that yields a pydantic object. Underneath, the default path for a tool-calling model is exactly the "tool call" option from the table above: it binds the schema as a tool, forces the model to call it, and parses the call's arguments with the schema. On providers with native strict output, the `method` argument selects that path instead (current docs describe `"json_schema"`, `"function_calling"`, and `"json_mode"`; support varies by provider).

**`create_agent(..., response_format=Schema)`** is the agent-level version. The agent may call real tools for several turns, and its *final* answer must be the schema. It picks a strategy:

```text
ProviderStrategy   use the provider's native structured output (constrained decoding)
ToolStrategy       add a synthetic tool named after the schema; the "call" is the answer
AutoStrategy       (default when you pass a bare schema) choose based on the model
```

With `ToolStrategy`, a validation failure becomes an error `ToolMessage` and the model gets another turn. That is our repair loop, with the error delivered as a tool result instead of a human message.

```{code-cell} python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage

# 1. Model level: schema bound as a forced tool, arguments parsed into Triage.
m = scripted([("Triage", {"evidence": "Two charges on in_1042", "category": "billing",
                          "urgency": 5, "needs_human": True})])
triage = m.with_structured_output(Triage).invoke("Acme was charged twice.")
print("with_structured_output ->", triage)
assert isinstance(triage, Triage) and triage.needs_human

# 2. Agent level: first attempt violates the schema, the error goes back as a
#    ToolMessage, the second attempt is accepted as structured_response.
agent_model = scripted(
    [("Triage", {"evidence": "Two charges", "category": "refund", "urgency": 9, "needs_human": True})],
    [("Triage", {"evidence": "Two charges", "category": "billing", "urgency": 5, "needs_human": True})],
)
agent = create_agent(agent_model, tools=[], response_format=ToolStrategy(Triage))
out = agent.invoke({"messages": [HumanMessage("Acme was charged twice for in_1042.")]})

errors = [msg.content for msg in out["messages"]
          if isinstance(msg, ToolMessage) and msg.content.startswith("Error")]
print("repair feedback:", errors[0][:110], "...")
print("structured_response ->", out["structured_response"])
assert len(errors) == 1 and "category" in errors[0]
assert out["structured_response"].category == "billing"
```

Map it back:

| From scratch | LangChain |
|---|---|
| `Triage.model_json_schema()` sent in the prompt | schema bound as a tool or passed as the provider's response format |
| `extract_json` + `model_validate` | output parser inside `with_structured_output`, or the strategy's parse step |
| `HumanMessage("Your JSON failed validation: ...")` | error `ToolMessage` under `ToolStrategy`'s error handling |
| `max_attempts` | the agent's model-call limits and recursion limit |
| `RefundProposal` context validators | still yours; call them in a node after the agent |

### A model-agnostic extraction node

Now the provider question. The trick is to put every provider-specific concern behind one function signature, `(prompt) -> Triage`, and write fallback logic against that signature. In a graph this becomes a node that tries providers in order, with the attempt history in state for tracing.

```{code-cell} python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


def make_adapter(name, model, max_attempts):
    """Everything provider-specific lives here: prompt format, parse path, retries."""
    def run(prompt):
        return structured_call(model, Triage, prompt, max_attempts=max_attempts)[0]
    run.provider = name
    return run


class ExtractState(TypedDict):
    prompt: str
    providers_tried: Annotated[list[str], operator.add]  # append-only audit trail
    triage: Triage | None
    error: str | None


def build(adapters):
    def extract(state: ExtractState):
        adapter = adapters[len(state["providers_tried"])]
        try:
            return {"triage": adapter(state["prompt"]), "error": None,
                    "providers_tried": [adapter.provider]}
        except StructuredOutputFailed as exc:
            return {"error": str(exc), "providers_tried": [adapter.provider]}

    def route(state: ExtractState):
        if state["triage"] is not None:
            return END
        # Refusals and truncation are not provider-quality problems; do not fail over.
        if state["error"].startswith(("refusal", "truncated")):
            return END
        return "extract" if len(state["providers_tried"]) < len(adapters) else END

    g = StateGraph(ExtractState)
    g.add_node("extract", extract)
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", route, ["extract", END])
    return g.compile()


primary = make_adapter("provider_a", scripted("{broken", "still {not json"), max_attempts=2)
backup = make_adapter("provider_b", scripted(
    '{"evidence": "Two charges on in_1042", "category": "billing", "urgency": 5, "needs_human": true}'),
    max_attempts=2)

result = build([primary, backup]).invoke(
    {"prompt": "Acme was charged twice.", "providers_tried": [], "triage": None, "error": None})
print(result["providers_tried"], "->", result["triage"])
assert result["providers_tried"] == ["provider_a", "provider_b"]
assert result["triage"].urgency == 5 and result["error"] is None
```

The business code only ever sees `Triage`. It never sees which provider answered, which parse path was used, or how many retries it took; that goes to the trace via `providers_tried`. Swapping providers is now a configuration change, and the contract test suite (the same inputs, the same expected `Triage` objects) runs against every adapter.

In production you would also reach for `ModelFallbackMiddleware` in `create_agent`, which retries a failed model call on the next model in a list. It handles provider *outages*. The graph above handles provider *quality*: a model that is up but cannot produce your schema.

### The live version

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

llm = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
structured = llm.with_structured_output(Triage)
print(structured.invoke("Acme says they were charged twice for invoice in_1042 and are furious."))

# Pydantic AI teaches the same contract with an explicit output validator + retry.
from pydantic_ai import Agent, ModelRetry

pai = Agent(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"), output_type=Triage,
            instructions="Triage the support ticket. Refunds always need a human.")


@pai.output_validator
def refunds_need_humans(output: Triage) -> Triage:
    if output.category == "billing" and not output.needs_human:
        raise ModelRetry("Billing disputes involving money must set needs_human=true.")
    return output


print(pai.run_sync("I was charged twice and want my money back.").output)
```

Pydantic AI's `ModelRetry` is the clearest expression of the repair loop in any framework: a business rule raises, the framework sends the message back, the model tries again. Its output modes (`ToolOutput`, `NativeOutput`, `PromptedOutput`) are the same three options from the table at the top of this lesson, named explicitly.

## What goes wrong in production

1. **Valid JSON, wrong value.** Symptom: the CRM fills with well-formed tickets marked `urgency: 1` for outages. Cause: strict mode guarantees syntax and schema, not truth. Fix: semantic validators with source data, field-level evals (per-field accuracy on a labelled set), and approval gates on anything that moves money.

2. **Required fields force invention.** Symptom: `customer_email` is always filled, sometimes with a plausible address that does not exist. Cause: a required string gives the model no honest way to say "not found." Fix: make it nullable, add an explicit `"unknown"` enum member, and test on inputs where the field is absent. An extraction eval without negative cases rewards hallucination.

3. **Schema rejected or silently weakened by a provider.** Symptom: the call works on provider A and 400s on provider B, or a `maxLength` constraint is simply not enforced. Cause: each strict mode supports a different JSON Schema subset. Fix: keep schemas simple (flat objects, enums, nullable fields), run a strict-compatibility check in CI, and always re-validate with pydantic after the provider returns.

4. **Truncation masquerading as a parse error.** Symptom: intermittent `JSONDecodeError` on long extractions, and retries do not help. Cause: the output hit `max_tokens`. Retrying the same request truncates at the same place. Fix: check the finish reason first, raise the limit or split the schema, and never count truncation as a model-quality failure.

5. **Retry storms.** Symptom: cost doubles after a model upgrade with no change in traffic. Cause: the new model fails validation more often, and every failure is a full-context retry. Fix: cap attempts, log attempts per call as a metric, alert on its mean, and gate model changes on a structured-output eval (see 22. Evaluation Reliability Deep Dive).

6. **Schema evolution breaks stored data and in-flight runs.** Symptom: yesterday's checkpoints fail to deserialize after a field rename. Cause: the schema is also a storage format. Fix: version it (`schema_version` field or versioned class names), make changes additive, and migrate old records explicitly. This is the same discipline as any API contract.

7. **Provider objects leak into business logic.** Symptom: a provider SDK upgrade requires edits in twelve files. Cause: code reads provider-specific response fields directly. Fix: the adapter boundary above. Business code receives domain objects, and a contract test suite runs against each adapter.

8. **Giant schemas.** Symptom: slower, costlier calls and worse accuracy on the fields that matter. Cause: a 60-field schema is a 60-field prompt; every field competes for attention. Fix: extract in stages (triage first, then the details only the chosen category needs), which is prompt chaining applied to schemas (01. Prompt Chaining).

## What changed by 2026

- **Native structured output became the default path.** Current OpenAI, Anthropic, and Google docs all describe schema-constrained output. The engineering question moved from "how do I get valid JSON" to "what is my semantic validator and what is my fallback when a model or provider cannot honor the schema." The supported JSON Schema subset still differs between providers, so portable schemas stay simple.
- **Structured output moved into the agent loop.** In LangChain 1.x, `create_agent` takes `response_format` and returns `structured_response`, with explicit `ToolStrategy` and `ProviderStrategy` choices. The final answer of an agent is now routinely a typed object instead of a string that a second call must parse.
- **Frameworks converged on the same three modes.** Pydantic AI names them (`ToolOutput`, `NativeOutput`, `PromptedOutput`) and adds output validators with `ModelRetry`. The OpenAI Agents SDK takes an `output_type` on an agent. Different spellings of one idea.
- **Reasoning models changed where thinking goes.** A reasoning model thinks before emitting the structured answer, so a `reasoning` field in your schema is often redundant. A short `evidence` field that cites observations is still useful because you can check it. Neither is a faithful account of why the model decided; verify the outcome against source data.
- **Model routing made model-agnosticism practical, not theoretical.** Teams now routinely run a cheap model for classification and a frontier model for synthesis (30. Fine-tuning, Routing, and Prompt Optimization). That only works if every model sits behind the same schema contract.

## Interview questions

**Q1. What is the difference between structured output and function calling? (reported: Mistral AI, Apple)**

Strong: Function calling lets the model decide whether to act and which tool to call; it can also reply in prose, and that option is a feature. Structured output constrains the final response to a schema every time. Mechanically they overlap: many libraries implement structured output by declaring the schema as a tool and forcing the model to call it, and strict modes for both use the same constrained-decoding machinery. So the choice is about semantics: use tools when the model is choosing an action, use a response format when downstream code always needs an object of a known shape, like the ticket record at the end of a support run.

Weak: "Function calling is for APIs and structured output is for JSON."

**Q2. If the provider guarantees schema-valid output, why do you still validate?**

Strong: Strict mode guarantees syntax and schema: parseable JSON, required fields, types, enum members. It says nothing about whether the values are true. A refund amount can be a valid integer and ten times too large; an invoice id can match the pattern and not exist. So I keep a semantic validator that checks values against source data, like the Stripe charge amount, and I re-run pydantic validation anyway because provider schema subsets differ and some constraints are not enforced. Semantics and policy are always my code's job.

Weak: "Strict mode makes the output correct."

**Q3. Walk me through a validation-and-retry loop. What do you retry and what do you not?**

Strong: Parse leniently first, stripping fences and chatter, because those are not worth a round trip. Then validate. On a schema error I send back the field-level messages, not a generic "try again," and I cap attempts at two or three. I do not retry truncation, because the same request truncates the same way; I raise the token limit or split the schema. I do not retry refusals; they go to a human or a different path. And I log attempts per call as a metric, because a rise in average attempts is the earliest signal that a model or prompt change regressed.

Weak: "Wrap it in a try/except and retry until it works."

**Q4. How would you design an LLM layer that is not locked into one provider?**

Strong: Business code depends on domain types, like a `Triage` pydantic model, and on one function signature, prompt in and domain object out. Each provider gets an adapter that owns request formatting, the structured-output mode it supports, parsing, and retries. A contract test suite runs the same inputs through every adapter and compares the domain objects. Fallback logic is written against the signature: outages go to the next provider through a fallback middleware or gateway, and quality failures, like a model that cannot follow the schema, are tracked separately. The schema itself stays in the portable subset so it works on all of them.

Weak: "Use LangChain so you can switch models."

**Q5. How do you design a schema so the model fills it correctly?**

Strong: Enums instead of free strings for anything code will branch on. Descriptions that are rubrics, not labels, since the description is prompt. Nullable fields or an explicit "unknown" value so the model is never forced to invent. A short evidence field early in the object, because generation is left to right and later fields condition on earlier ones. Keep it small; if it needs sixty fields, extract in stages. And evaluate per field on a labelled set that includes cases where the answer is absent.

Weak: "Make the schema as detailed as possible so it captures everything."

**Q6. How do you evaluate an extraction or classification step?**

Strong: Field-level metrics on a labelled set: exact match for enums and ids, tolerance for numbers, and a separate rate for "correctly said unknown" versus "invented a value." Plus operational metrics: schema-failure rate, mean attempts per call, latency, and cost. I track these per model and per schema version, so a model swap or a schema edit is gated by the same regression suite. A single "accuracy" number hides the failure that matters, which is usually confident invention on the absent-field cases.

Weak: "I check that it returns valid JSON."

**Q7. Where does a structured output step fit in an agent built with create_agent?**

Strong: As the agent's `response_format`. The agent runs its tool loop as usual, and its final turn must produce the schema. With a tool strategy the schema is a synthetic tool and validation errors come back as tool messages, so the model repairs inside the same loop; with a provider strategy the provider's constrained decoding does the work. Either way the result lands in `structured_response`. Semantic checks against live data still belong in a node or middleware after that, before any side effect.

Weak: "Parse the last message with json.loads."

## Build it yourself

Extend `structured_call` and the extraction graph into a production-shaped component:

1. Add a `semantic_context` argument to `structured_call` and pass it to `model_validate(..., context=...)`, so `RefundProposal` validation errors become repair messages.
2. Return a telemetry record alongside the object: attempts, the failure kinds seen, and the provider name.
3. Detect truncation by scripting an `AIMessage` with `response_metadata={"finish_reason": "length"}`, and make the graph end without failover.
4. Add a third adapter and a per-request `max_total_attempts` budget across all providers.

Acceptance checks:

- A scripted model that proposes `amount_cents=120000` for `in_1042` and then `12000` returns a valid `RefundProposal` on attempt 2, and the repair message contains `exceeds charged`.
- A truncated first reply ends the graph with `providers_tried == ["provider_a"]` and an error starting with `truncated`.
- With `max_total_attempts=3` and three adapters that each always fail, the model is invoked exactly three times in total.
- The telemetry record for the happy path is `{"attempts": 1, "failures": [], "provider": "provider_a"}`.

## Key takeaways

- Structured output has four layers: syntax, schema, semantics, and policy; providers can guarantee only the first two.
- One pydantic model serves as the provider schema, the validator, and the typed object your code receives.
- Enums, rubric-style descriptions, nullable fields, and evidence-first ordering are schema design, and schema design is prompt design.
- Retry schema errors with field-level feedback and a cap; never retry truncation or refusals as-is.
- `with_structured_output` and `create_agent(response_format=...)` are the from-scratch loop with the provider path chosen for you.
- Keep providers behind an adapter that returns domain objects, and run one contract test suite against every adapter.
- Use function calling when the model chooses an action; use a response format when code always needs a known shape.
