# Lesson authoring guide

Every file under `lessons/` is the single source of truth for one notebook. The
generator turns Markdown into markdown cells and every
```` ```{code-cell} python ```` fence into a code cell. Read this guide and the
exemplar `08_appendices/00_course_orientation.md` before writing or editing a lesson.

## Who the reader is

A strong backend/distributed-systems engineer who wants to (1) understand agent
architecture deeply, (2) build it, and (3) pass AI/Agent Engineer interviews.
They have read *Agentic Design Patterns*. They are not a beginner programmer and
they hate hand-waving. Teach the machinery, then the framework, then production
reality, then the interview answer.

## The fixed structure

Use these sections, in this order, with these exact H2 titles. Each lesson is
roughly 2,000 to 3,500 words of prose plus 3 to 6 code cells.

1. `# NN. Title` then one italic line: which track (A agent engineering /
   B LLM fundamentals / C retrieval + production AI / D infra + interviews),
   which book chapter it maps to, and the prerequisite lesson(s).
2. `## The problem` — start from a concrete scenario (the running example is an
   enterprise support/ops agent with Slack, Notion, Stripe, CRM, SQL, web).
   Show the naive approach and exactly where it breaks. Include at least one
   ASCII diagram in a ```text fence.
3. `## Build it without a framework` — plain Python first. The reader must see
   the mechanism with nothing hidden. Offline code, scripted model, assertions.
4. `## How LangGraph does it` — the same behavior with LangGraph primitives
   (StateGraph, nodes, edges, conditional edges, Send, Command, checkpointer,
   interrupt, Store, subgraphs) or LangChain primitives (tools, messages,
   structured output, `create_agent`). Explain *what the primitive is doing
   underneath*, referencing the from-scratch version. Where another framework
   teaches the idea better (OpenAI Agents SDK, Google ADK, Pydantic AI, Deep
   Agents, MCP), show a short snippet in a `live` cell and say why.
5. `## What goes wrong in production` — numbered failure modes, each with the
   symptom, the cause, and the fix. Draw on distributed-systems knowledge
   (idempotency, retries, timeouts, partial failure, backpressure, audit).
6. `## What changed by 2026` — how the practice moved: harness/context
   engineering, durable execution, skills, sandboxes, MCP/A2A, tool retrieval,
   reasoning models, eval-driven development. Only state things that are true
   of the installed stack (LangChain 1.4, LangGraph 1.2, Deep Agents 0.7,
   OpenAI Agents SDK 0.18, Pydantic AI 2.9, Google ADK 2.4) or clearly hedge
   with "current docs describe". Never invent version numbers or dates.
7. `## Interview questions` — 5 to 8 questions in the style of the interview
   repository (Anthropic/OpenAI/Microsoft/Cognition/etc. reported questions).
   Each question gets a **strong answer** of 3 to 8 sentences, written as the
   reader should say it aloud. Then one line: what a weak answer sounds like.
8. `## Build it yourself` — one exercise that extends the code in this lesson,
   with 3 to 5 acceptance checks phrased as asserts or observable behaviors.
9. `## Key takeaways` — 5 to 7 bullets, each one sentence.

## Voice

Short paragraphs. Short sentences. One idea each. Explain the *why* before the
*how*. Prefer a diagram over a paragraph when describing flow. Prefer a table
when comparing three or more things. Name the interview question the concept
answers. Say "the model chooses, your software executes" style truths plainly.
No filler ("in this section we will"), no marketing language, no emojis.

Use this pattern liberally:

```text
model:        "I need calculator"
application:  calculator(...) -> 111342
model:        "The answer is 111342."
```

## Code rules (the verifier executes every offline cell)

- Default fence is offline: ```` ```{code-cell} python ````. It must run in a
  fresh kernel with no network, no API keys, in under 60 seconds, deterministic.
  Allowed imports: stdlib, `langgraph`, `langchain_core`, `langchain`,
  `pydantic`, `numpy`. Nothing else in offline cells.
- Model calls in offline cells use the scripted model below. Copy it verbatim
  into the first code cell of any lesson that needs a model. Explain in prose
  that the script stands in for a real model so the *control flow* is the
  object of study, and that a live cell shows the real thing.
- Live model cells use ```` ```{code-cell} python live ````. The generator wraps
  them so they run only when `RUN_LIVE=1`. Inside, create the model with
  `from langchain.chat_models import init_chat_model` and
  `init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))`. Optional
  packages (openai-agents, pydantic-ai, google-adk, deepagents) are imported
  only inside live cells.
- Remote cells (```` ```{code-cell} python remote ````) are only for LangSmith
  upload/eval and run only with `RUN_REMOTE=1`.
- Every offline cell ends with `print(...)` of something meaningful and at least
  one `assert` that pins the behavior taught.
- Each cell is self-contained enough to read on its own but may rely on earlier
  cells in the same lesson. Notebooks run top to bottom.
- Comments explain decisions, not syntax. No `# import json` comments.

### Scripted model helper (copy verbatim)

```python
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
```

## Cross-references

Refer to other lessons by number and title, e.g. "see 13. Human in the Loop".
Refer to the book as "the book" with the chapter name. Refer to the interview
repository as "the interview repository". Do not link to URLs you have not
verified; prefer naming the doc page ("the LangGraph persistence guide").

## Variant for Track B/C/D lessons (LLM internals, retrieval internals, serving, system design)

These lessons keep the same skeleton, but section 4 is titled `## How it shows up in the stack`
instead of `## How LangGraph does it`: implement the mechanism in numpy/stdlib (attention,
KV cache sizing, BM25, RRF, HNSW-style greedy search, batching simulation), then show where the
concept surfaces for an agent engineer (context budgets, prompt caching, tool result sizes,
retrieval nodes, cost/latency). Section 6 stays `## What changed by 2026`. Formulas go in
```text fences with a worked numeric example. Cite the primary source by name (paper title,
year) rather than URL, drawing on `docs/research/*.md`.
